"""The probe pipeline.

Sample, embed, establish ground truth, fit candidates, measure, decide. The order
matters and the sharing matters: normalisation, the sample and the ground truth
are computed **once** and reused by every candidate. Recomputing them per
candidate would triple the cost of the single most expensive step.

What ``probe`` does *not* do is write anything to the user's index. That is a
contract, not an intention: ``probe`` and ``fit`` are read-only and must work
against a read-only filesystem, and a contract test wraps the store in a guard
that fails the run if a write is attempted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np  # noqa: TC002 - runtime annotation on a slotted dataclass

from rebasis.core import ScoreCalibrator, csls_bias, fit_candidates, l2_normalize, select_best
from rebasis.observability import Events, Spans, get_logger, instrument, span
from rebasis.probe.decision import DecisionResult, decide
from rebasis.probe.groundtruth import (
    GroundTruth,  # noqa: TC001 - runtime annotation on a slotted dataclass
)
from rebasis.probe.metrics import (
    bootstrap_ci,
    bootstrap_ratio_ci,
    mrr_at_k,
    ndcg_at_k,
    overlap_at_k,
    recall_at_k,
    recall_per_query,
    score_shift_ks,
    spearman_topk,
    tail_recall,
    top_k_search,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rebasis.core.selection import AdapterCandidate
    from rebasis.types import FloatArray

__all__ = ["CandidateMetrics", "ProbeResult", "evaluate_candidate", "run_probe"]

log = get_logger(__name__)

#: Fraction of held-out queries reserved for fitting the score calibrator.
#: Fitting and measuring on the same scores would flatter the calibration.
CALIBRATION_FRACTION = 1 / 3

#: Below this there is nothing to calibrate against, and a calibrator fitted on
#: one or two points would be noise dressed up as a correction.
_MIN_CALIBRATION_POINTS = 2


@dataclass(slots=True)
class CandidateMetrics:
    """Everything measured about one candidate."""

    name: str
    arr: float
    arr_sparse: float
    arr_ci: tuple[float, float]
    mrr: float
    overlap: float
    spearman: float
    score_shift_raw: float
    #: Rank-sensitive quality. Recall counts what was found, nDCG counts where —
    #: and an adapter can improve one while degrading the other.
    ndcg: float = 0.0
    score_shift_calibrated: float | None = None
    #: Recall in the sparsest clusters. ``None`` when the sample was not
    #: clustered, or held too few clusters for a tail to mean anything.
    tail_arr: float | None = None
    used_csls: bool = False
    n_params: int = 0
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "adapter_type": self.name,
            "arr_r10": round(self.arr, 4),
            "arr_sparse": round(self.arr_sparse, 4),
            "arr_ci": [round(v, 4) for v in self.arr_ci],
            "arr_mrr": round(self.mrr, 4),
            "naive_overlap": round(self.overlap, 4),
            "spearman_topk": round(self.spearman, 4),
            "ndcg": round(self.ndcg, 4),
            "score_shift": round(self.score_shift_raw, 4),
            "score_shift_calibrated": (
                round(self.score_shift_calibrated, 4)
                if self.score_shift_calibrated is not None
                else None
            ),
            "tail_arr": None if self.tail_arr is None else round(self.tail_arr, 4),
            "used_csls": self.used_csls,
            "n_params": self.n_params,
        }


@dataclass(slots=True)
class ProbeResult:
    """The full outcome of a probe run."""

    decision: DecisionResult
    best: CandidateMetrics
    candidates: list[CandidateMetrics]
    baselines: dict[str, float]
    ground_truth_tier: str
    n_documents: int
    n_queries: int
    n_fit_pairs: int
    k: int
    seed: int
    duration_seconds: float
    calibrator: ScoreCalibrator | None = None
    adapter: Any = None
    #: What this run cost. Written to the report and the audit record so
    #: "it got slower" becomes a fact rather than a feeling.
    resources: dict[str, Any] = field(default_factory=dict)
    #: What a full reindex would cost, extrapolated from this run's own
    #: measured embedding rate.
    reindex_cost: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form, for the report and the audit record."""
        return {
            **self.decision.to_dict(),
            "best_adapter": self.best.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
            "baselines": {k: round(v, 4) for k, v in self.baselines.items()},
            "tier": self.ground_truth_tier,
            "n_documents": self.n_documents,
            "n_queries": self.n_queries,
            "n_fit_pairs": self.n_fit_pairs,
            "k": self.k,
            "seed": self.seed,
            "duration_seconds": round(self.duration_seconds, 2),
            "resources": self.resources,
            "cost_full_reindex": self.reindex_cost,
        }


def evaluate_candidate(  # noqa: PLR0913 - one argument per measurement input
    candidate: AdapterCandidate,
    *,
    query_vectors_new: FloatArray,
    old_doc_vectors: FloatArray,
    ground_truth: GroundTruth,
    k: int,
    csls_sample: FloatArray | None = None,
    query_clusters: np.ndarray | None = None,
) -> CandidateMetrics:
    """Measure one candidate, with and without CSLS.

    CSLS is evaluated as a **variant** rather than applied unconditionally. M0
    measured it adding +0.103 on weak adapters and costing −0.045 on strong ones,
    so applying it always would degrade exactly the adapters that are working.
    """
    mapped = l2_normalize(candidate.adapter.apply(query_vectors_new), copy=False)

    indices, scores = top_k_search(mapped, old_doc_vectors, k=k, self_mask=ground_truth.self_mask)
    plain_arr = _arr(indices, ground_truth, k)

    used_csls = False
    best_indices, best_scores, best_arr = indices, scores, plain_arr
    if csls_sample is not None:
        bias = csls_bias(old_doc_vectors, csls_sample)
        csls_indices, csls_scores = top_k_search(
            mapped, old_doc_vectors, k=k, self_mask=ground_truth.self_mask, doc_bias=bias
        )
        csls_arr = _arr(csls_indices, ground_truth, k)
        if csls_arr > plain_arr:
            used_csls = True
            best_indices, best_scores, best_arr = csls_indices, csls_scores, csls_arr

    per_query = recall_per_query(best_indices, ground_truth.relevant_sparse, k)
    return CandidateMetrics(
        name=_display_name(candidate, used_csls=used_csls),
        arr=best_arr,
        arr_sparse=recall_at_k(best_indices, ground_truth.relevant_sparse, k),
        arr_ci=_arr_interval(per_query, ground_truth),
        mrr=mrr_at_k(best_indices, ground_truth.relevant, k),
        ndcg=ndcg_at_k(best_indices, ground_truth.relevant_sparse, k),
        overlap=overlap_at_k(best_indices, ground_truth.oracle_indices, k),
        spearman=spearman_topk(ground_truth.oracle_indices, best_indices, k),
        score_shift_raw=score_shift_ks(best_scores, ground_truth.oracle_scores),
        tail_arr=_tail(per_query, query_clusters, ground_truth),
        used_csls=used_csls,
        n_params=candidate.adapter.n_params(),
        extras={"arr_without_csls": plain_arr},
    )


def _ndcg_advantage(best: CandidateMetrics, baselines: dict[str, float]) -> float | None:
    """Bridged nDCG over the current model's — the break-even, rank-sensitively.

    Returns ``None`` when the old model's nDCG was not computed, which is the
    case at tiers that have no old-model query vectors to score.
    """
    old = baselines.get("old_model_only_ndcg")
    if old is None or old <= 0 or best.ndcg <= 0:
        return None
    return best.ndcg / old


def _display_name(candidate: AdapterCandidate, *, used_csls: bool) -> str:
    """The name shown in the report, carrying both suffixes.

    ``AdapterCandidate.name`` cannot be used directly: it reads ``use_csls``,
    which is only set after this candidate has been measured.
    """
    base = f"{candidate.adapter.type_name}+csls" if used_csls else candidate.adapter.type_name
    return base if candidate.fit_kind == "document" else f"{base}@query"


def _tail(
    per_query: FloatArray, query_clusters: np.ndarray | None, ground_truth: GroundTruth
) -> float | None:
    """Tail recall on the same scale as ARR.

    Normalised by the oracle for the same reason ARR is: the number has to be
    comparable with the overall figure it is warned against, or the gap between
    them means nothing.
    """
    if query_clusters is None:
        return None
    raw = tail_recall(per_query, query_clusters)
    if raw is None:
        return None
    oracle = ground_truth.oracle_recall or 1.0
    return raw / oracle if oracle > 0 else None


def _arr_interval(per_query: FloatArray, ground_truth: GroundTruth) -> tuple[float, float]:
    """The interval for ARR, on the same quantity as ARR itself.

    ARR is ``mean(candidate recall) / mean(oracle recall)``. Bootstrapping only
    the numerator gives the interval of the numerator, which at T1 — where the
    oracle is not perfect — is a visibly different number: the first real-corpus
    run reported ARR 0.908 against an interval of 0.712-0.808, the point
    estimate sitting outside its own range.

    At T0 the oracle is perfect by construction, so the two agree exactly. That
    is why the defect survived every synthetic test.
    """
    oracle = ground_truth.oracle_recall_per_query
    if oracle is None or oracle.size != per_query.size:
        return bootstrap_ci(per_query)
    return bootstrap_ratio_ci(per_query, oracle)


def _arr(indices: np.ndarray, ground_truth: GroundTruth, k: int) -> float:
    """ARR against the sparsity-matched ground truth, normalised by the oracle.

    Sparsity-matched because M0 showed the strict variant disagreeing with real
    queries on the decision 58% of the time. Divided by the oracle because at T1
    a full reindex is not perfect either, which makes ARR a ratio rather than a
    bare recall.
    """
    recall = recall_at_k(indices, ground_truth.relevant_sparse, k)
    oracle = ground_truth.oracle_recall or 1.0
    return recall / oracle if oracle > 0 else float("nan")


def run_probe(  # noqa: PLR0913 - one argument per pipeline stage
    *,
    old_doc_vectors: FloatArray,
    new_doc_vectors: FloatArray,
    fit_indices: np.ndarray,
    ground_truth: GroundTruth,
    old_query_vectors: FloatArray,
    new_query_vectors: FloatArray,
    k: int = 10,
    seed: int = 0,
    methods: Sequence[str] | None = None,
    with_csls: bool = True,
    upgrade_gain: float | None = None,
    new_doc_vectors_as_queries: FloatArray | None = None,
    query_clusters: np.ndarray | None = None,
    resources: dict[str, Any] | None = None,
    reindex_cost: dict[str, Any] | None = None,
) -> ProbeResult:
    """Run the full probe pipeline and return a decision.

    Args:
        old_doc_vectors: The index's own vectors, ℓ2-normalised.
        new_doc_vectors: The same documents under the new model.
        fit_indices: Which documents may be used for fitting — disjoint from the
            query set, or every ARR number is meaningless.
        ground_truth: T0 or T1 judgements.
        old_query_vectors: Queries under the old model, for the "do nothing" baseline.
        new_query_vectors: Queries under the new model.
        k: Cut-off for every metric.
        seed: Recorded so the decision can be replayed.
        methods: Candidates to try; defaults to the full `auto` list.
        with_csls: Evaluate the CSLS variant alongside each candidate.
        upgrade_gain: Oracle recall over old-model recall. Only measurable at T1,
            and decisive when it is available.
        new_doc_vectors_as_queries: The same documents encoded the way the new
            model encodes a *query*. Supplied only for asymmetric models, where
            fitting on document pairs and applying to query vectors is a
            train/serve mismatch: it turns on a second full candidate list fitted
            on query-encoded pairs, measured against the first. M0 found the mean
            difference to be -0.003 with the sign varying by model pair, which is
            why this is measured rather than assumed either way.
        query_clusters: Cluster of each query, when the sample was stratified.
            Enables ``tail_arr`` — the heterogeneous-drift signal.
        resources: What the run cost so far, for the summary.
        reindex_cost: Estimated cost of a full reindex.
    """
    started = time.perf_counter()
    log.info(
        Events.PROBE_RUN_STARTED,
        count=int(old_doc_vectors.shape[0]),
        dim=int(old_doc_vectors.shape[1]),
    )

    kwargs: dict[str, Any] = {"methods": methods} if methods else {}
    with span(Spans.ADAPTER_FIT, {"count": int(fit_indices.size)}):
        fit_started = time.perf_counter()
        candidates = fit_candidates(
            new_doc_vectors[fit_indices], old_doc_vectors[fit_indices], normalize=False, **kwargs
        )
        if new_doc_vectors_as_queries is not None:
            # An asymmetric model encodes a query differently from a document.
            # Fitting only on document pairs and applying to query vectors is a
            # train/serve mismatch — so the query-encoded pairs get their own
            # candidates and the two strategies compete.
            candidates.extend(
                fit_candidates(
                    new_doc_vectors_as_queries[fit_indices],
                    old_doc_vectors[fit_indices],
                    normalize=False,
                    fit_kind="query",
                    **kwargs,
                )
            )
        instrument("rebasis.adapter.fit.duration").record(
            (time.perf_counter() - fit_started) * 1000,
            {"adapter_type": "auto" if not methods else ",".join(methods)},
        )
    if not candidates:
        msg = "no adapter candidate could be fitted"
        raise RuntimeError(msg)

    # The CSLS sample is mapped source-distribution vectors, available at fit
    # time with no query log — which is what makes CSLS usable in the T0 tier.
    csls_sample = None
    if with_csls:
        sample_size = min(4000, fit_indices.size)
        csls_sample = l2_normalize(
            candidates[0].adapter.apply(new_doc_vectors[fit_indices[:sample_size]]), copy=False
        )

    with span(Spans.ADAPTER_EVALUATE, {"count": len(candidates)}):
        measured = [
            evaluate_candidate(
                c,
                query_vectors_new=new_query_vectors,
                old_doc_vectors=old_doc_vectors,
                ground_truth=ground_truth,
                k=k,
                csls_sample=csls_sample,
                query_clusters=query_clusters,
            )
            for c in candidates
        ]
    for candidate, metrics in zip(candidates, measured, strict=True):
        candidate.score = metrics.arr
        candidate.use_csls = metrics.used_csls

    winner = select_best(candidates)
    best_metrics = next(m for c, m in zip(candidates, measured, strict=True) if c is winner)

    baselines = _baselines(
        old_doc_vectors=old_doc_vectors,
        old_query_vectors=old_query_vectors,
        new_query_vectors=new_query_vectors,
        ground_truth=ground_truth,
        k=k,
    )

    calibrator, calibrated_shift = _calibrate(
        winner, new_query_vectors, old_doc_vectors, ground_truth, k
    )
    best_metrics.score_shift_calibrated = calibrated_shift

    with span(Spans.DECISION) as decision_span:
        decision = decide(
            best_metrics.arr,
            confidence_interval=best_metrics.arr_ci,
            upgrade_gain=upgrade_gain,
            score_shift=calibrated_shift,
            tail_arr=best_metrics.tail_arr,
            # The alternative the user actually has. Measured on the same scale
            # as ARR, so the two are directly comparable.
            old_model_arr=baselines.get("old_model_only"),
            # The same comparison on a rank-sensitive metric. Recall counts what
            # was found and nDCG counts where, and they can disagree — measured
            # once, on a run where recall improved while nDCG fell.
            ndcg_advantage=_ndcg_advantage(best_metrics, baselines),
            tier=ground_truth.tier,
            # A synthesised task the retriever solves every time cannot tell two
            # models apart, so the upgrade estimate built on it is not one.
            estimate_uninformative=bool(ground_truth.metadata.get("task_too_easy")),
        )
        _annotate(decision_span, decision, best_metrics)

    log.info(
        Events.PROBE_DECISION_MADE,
        decision=decision.decision,
        arr_r10=round(decision.arr_at_k, 4),
        borderline=decision.borderline,
        count=int(ground_truth.query_indices.size),
        seed=seed,
    )

    duration = time.perf_counter() - started
    log.info(
        Events.PROBE_RUN_COMPLETED,
        duration_ms=round(duration * 1000, 1),
    )

    return ProbeResult(
        decision=decision,
        best=best_metrics,
        candidates=measured,
        baselines=baselines,
        ground_truth_tier=ground_truth.tier,
        n_documents=int(old_doc_vectors.shape[0]),
        n_queries=int(ground_truth.query_indices.size),
        n_fit_pairs=int(fit_indices.size),
        k=k,
        seed=seed,
        duration_seconds=duration,
        calibrator=calibrator,
        adapter=winner.adapter,
        resources=resources or {},
        reindex_cost=reindex_cost or {},
    )


def _annotate(active: Any, decision: DecisionResult, best: CandidateMetrics) -> None:
    """Put the decision on the span and the metrics gauge.

    Only the decision and the headline ratio — no ids, no text, no vectors,
    which redaction refuses anyway. Span attributes are always indexed, so what
    goes on one is a deliberate choice rather than everything that was to hand.
    """
    from rebasis.observability.semconv import (
        REBASIS_ADAPTER_TYPE,
        REBASIS_PROBE_ARR_R10,
        REBASIS_PROBE_DECISION,
    )

    active.set_attribute(REBASIS_PROBE_DECISION, decision.decision)
    active.set_attribute(REBASIS_PROBE_ARR_R10, round(decision.arr_at_k, 4))
    active.set_attribute(REBASIS_ADAPTER_TYPE, best.name)

    gauge = instrument("rebasis.probe.arr")
    gauge.set(best.arr, {"metric": "r10", "adapter_type": best.name})
    gauge.set(best.mrr, {"metric": "mrr", "adapter_type": best.name})


def _baselines(
    *,
    old_doc_vectors: FloatArray,
    old_query_vectors: FloatArray,
    new_query_vectors: FloatArray,
    ground_truth: GroundTruth,
    k: int,
) -> dict[str, float]:
    """The two comparisons that make ARR interpretable.

    An absolute ARR means little on its own. What a user needs is: how much worse
    is doing nothing, and how bad would it be to skip the adapter entirely.
    """
    oracle = ground_truth.oracle_recall or 1.0
    baselines: dict[str, float] = {}

    old_indices, _ = top_k_search(
        old_query_vectors, old_doc_vectors, k=k, self_mask=ground_truth.self_mask
    )
    baselines["old_model_only"] = recall_at_k(old_indices, ground_truth.relevant_sparse, k) / oracle
    baselines["old_model_only_ndcg"] = ndcg_at_k(old_indices, ground_truth.relevant_sparse, k)

    # Only meaningful when the dimensions match; otherwise the new vector cannot
    # physically enter the old index at all.
    if new_query_vectors.shape[1] == old_doc_vectors.shape[1]:
        naive_indices, _ = top_k_search(
            new_query_vectors, old_doc_vectors, k=k, self_mask=ground_truth.self_mask
        )
        baselines["unadapted"] = (
            recall_at_k(naive_indices, ground_truth.relevant_sparse, k) / oracle
        )

    return baselines


def _calibrate(
    winner: AdapterCandidate,
    new_query_vectors: FloatArray,
    old_doc_vectors: FloatArray,
    ground_truth: GroundTruth,
    k: int,
) -> tuple[ScoreCalibrator | None, float | None]:
    """Fit the score calibrator on one subset and measure it on another.

    Fitting and measuring on the same scores would report a calibration quality
    the user will never see in practice.
    """
    n = new_query_vectors.shape[0]
    n_calibration = int(n * CALIBRATION_FRACTION)
    if n_calibration < _MIN_CALIBRATION_POINTS or n - n_calibration < _MIN_CALIBRATION_POINTS:
        return None, None

    mapped = l2_normalize(winner.adapter.apply(new_query_vectors), copy=False)
    _, scores = top_k_search(mapped, old_doc_vectors, k=k, self_mask=ground_truth.self_mask)

    calibrator = ScoreCalibrator.fit(
        scores[:n_calibration], ground_truth.oracle_scores[:n_calibration]
    )
    calibrated = calibrator.transform(scores[n_calibration:])
    shift = score_shift_ks(calibrated, ground_truth.oracle_scores[n_calibration:])

    log.info(Events.FIT_CALIBRATOR_FITTED, count=n_calibration, score_shift=round(shift, 4))
    return calibrator, shift
