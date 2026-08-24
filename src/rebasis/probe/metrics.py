"""Retrieval metrics.

Recall@k, MRR and nDCG are implemented here on numpy. ``ranx`` would give them
for free but pulls in Numba, which lags new Python releases, and the core path
has to stay dependency-free. ``ranx`` remains an ``[eval]`` extra for
significance testing and fusion, and an equivalence test keeps the two
implementations of nDCG in agreement.

nDCG was originally left to ``ranx``. It moved here because it turned out to
change a decision rather than only decorate a report: on BEIR/scifact an adapter
improved recall@10 while nDCG@10 fell, and a recall-only rule called that an
improvement. A metric the decision depends on cannot sit behind an optional
extra.

``score_shift`` deserves its own note. It is rarely discussed and it matters more
than its obscurity suggests: many RAG pipelines filter on a fixed threshold such
as ``similarity > 0.7``. An adapter can preserve ranking perfectly and still move
the absolute scores enough to empty that filter silently. M0 measured a median KS
distance of **0.924** without calibration, with **100%** of configurations above
the decision rule's warning threshold of 0.1 — which is why the warning has to be
evaluated *after* calibration to mean anything.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.stats

from rebasis.compute.search import top_k_search
from rebasis.types import FloatArray, as_float32

if TYPE_CHECKING:
    from collections.abc import Sequence

#: A correlation needs at least two distinct ranks to be defined.
_MIN_DISTINCT_RANKS = 2

#: Below this, a "sparsest decile" is one cluster, and one cluster's mean is
#: not the tail of a distribution.
_MIN_CLUSTERS_FOR_TAIL = 4

#: Fewest queries a tail may hold and still be reported.
#:
#: ``tail_arr`` is compared against a gap of 0.1, so its own standard error has
#: to be well under that. For a mean of values in [0, 1] the standard deviation
#: is at most 0.5, giving ``0.5/sqrt(n) < 0.05`` — about 100 — and in practice
#: recall's spread is nearer 0.4, which puts the requirement at 64.
#: Thirty is the point where the error falls below the gap limit itself rather
#: than half of it, and it is set there because suppressing a real signal is
#: also a cost.
#:
#: Measured on real corpora: at T0 the tail held 42-59 of 1,000 held-out
#: queries and is reported. At T1 it held **six**, and produced a tail_arr of
#: exactly 0.000 that read as catastrophic heterogeneous drift and was six
#: queries missing.
MIN_TAIL_QUERIES = 30

#: Version of the metric definitions, recorded with every decision.
#:
#: A decision is only reproducible if the metrics behind it are. Changing how
#: ARR, recall or ``score_shift`` are computed changes what a recorded number
#: means, so `audit replay` needs to know that a record predates the change
#: rather than silently comparing two different quantities. Bump this whenever a
#: metric's definition changes; adding a new metric does not count.
METRIC_VERSION = 1

__all__ = [
    "METRIC_VERSION",
    "MIN_TAIL_QUERIES",
    "bootstrap_ci",
    "bootstrap_ratio_ci",
    "estimate_reindex_cost",
    "mrr_at_k",
    "ndcg_at_k",
    "overlap_at_k",
    "recall_at_k",
    "recall_per_query",
    "score_shift_ks",
    "spearman_topk",
    "tail_recall",
    "top_k_search",
]


def recall_per_query(retrieved: np.ndarray, relevant: Sequence[set[int]], k: int) -> FloatArray:
    """Per-query recall — the raw input to the bootstrap interval.

    Queries with no relevant documents are skipped rather than scored zero:
    counting them as failures would understate every system equally, which
    changes the absolute numbers without changing any comparison.
    """
    values = [
        len(rel & set(row.tolist())) / len(rel)
        for row, rel in zip(retrieved[:, :k], relevant, strict=True)
        if rel
    ]
    return as_float32(values)


def recall_at_k(retrieved: np.ndarray, relevant: Sequence[set[int]], k: int) -> float:
    """Mean Recall@k."""
    per_query = recall_per_query(retrieved, relevant, k)
    return float(per_query.mean()) if per_query.size else float("nan")


def mrr_at_k(retrieved: np.ndarray, relevant: Sequence[set[int]], k: int) -> float:
    """Mean Reciprocal Rank@k — the basis of the reported ``ARR_MRR``."""
    values = []
    for row, rel in zip(retrieved[:, :k], relevant, strict=True):
        if not rel:
            continue
        reciprocal = 0.0
        for rank, doc in enumerate(row.tolist(), start=1):
            if doc in rel:
                reciprocal = 1.0 / rank
                break
        values.append(reciprocal)
    return float(np.mean(values)) if values else float("nan")


def overlap_at_k(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Top-k intersection ratio between two rankings. Symmetric, in [0, 1]."""
    pairs = zip(a[:, :k], b[:, :k], strict=True)
    return float(np.mean([len(set(x.tolist()) & set(y.tolist())) / k for x, y in pairs]))


def spearman_topk(ground_truth: np.ndarray, predicted: np.ndarray, k: int) -> float:
    """Spearman correlation between the ground-truth order and its order in the prediction.

    Items missing from the prediction are given a penalty rank of ``k``.
    Correlating only over the items that were found flatters bad adapters, which
    defeats the purpose of the metric.
    """
    values = []
    for gt_row, pred_row in zip(ground_truth[:, :k], predicted[:, :k], strict=True):
        position = {doc: i for i, doc in enumerate(pred_row.tolist())}
        ranks = [position.get(doc, k) for doc in gt_row.tolist()]
        if len(set(ranks)) < _MIN_DISTINCT_RANKS:
            continue
        rho = scipy.stats.spearmanr(list(range(len(gt_row))), ranks).statistic
        if not np.isnan(rho):
            values.append(float(rho))
    return float(np.mean(values)) if values else float("nan")


def score_shift_ks(adapted: FloatArray, oracle: FloatArray) -> float:
    """KS distance between two score distributions — the ``score_shift`` metric.

    Measure this **after** calibration. Before it, M0 found 100% of
    configurations above the 0.1 threshold, which makes the warning constant and
    therefore uninformative.
    """
    return float(scipy.stats.ks_2samp(np.ravel(adapted), np.ravel(oracle)).statistic)


def bootstrap_ci(
    per_query: FloatArray, *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Bootstrap confidence interval for a mean (percentile method).

    Needed because ``probe`` outputs a DECISION taken against thresholds. M0
    measured a median half-width of ±0.024 for ARR on ~1,000 held-out queries,
    against decision bands 0.10 wide — so a result within a few hundredths of a
    threshold is decided by sampling noise, and the report has to say so.
    """
    if per_query.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, per_query.size, size=(n_boot, per_query.size))
    means = per_query[draws].mean(axis=1)
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def bootstrap_ratio_ci(
    numerator: FloatArray,
    denominator: FloatArray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap interval for a **ratio of means**, paired at query level.

    ARR is ``mean(adapter recall) / mean(oracle recall)``, so its interval has to
    be the interval of that ratio. Bootstrapping only the numerator produces an
    interval for a different quantity — and at T1, where the oracle is not
    perfect, the point estimate can land outside its own interval.

    Paired, because the two series are measured on the *same* queries: a query
    that is hard for the adapter is usually hard for the oracle too, and
    resampling them independently would inflate the width by pretending that
    correlation away. Each draw picks query indices once and scores both series
    on the same ones.

    Args:
        numerator: Per-query recall of the candidate.
        denominator: Per-query recall of the oracle, for the same queries in the
            same order.
        n_boot: Resamples.
        alpha: Two-sided level; 0.05 gives a 95% interval.
        seed: Recorded so the interval can be reproduced.
    """
    if numerator.size == 0 or numerator.size != denominator.size:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, numerator.size, size=(n_boot, numerator.size))
    top = numerator[draws].mean(axis=1)
    bottom = denominator[draws].mean(axis=1)

    # A resample in which the oracle retrieves nothing carries no information
    # about the ratio; dropping it is honest, scoring it as infinity is not.
    usable = bottom > 0
    if not usable.any():
        return (float("nan"), float("nan"))

    ratios = top[usable] / bottom[usable]
    low, high = np.percentile(ratios, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def tail_recall(
    per_query: FloatArray,
    cluster_labels: np.ndarray,
    *,
    decile: float = 0.1,
    min_queries: int = MIN_TAIL_QUERIES,
) -> float | None:
    """Recall in the sparsest clusters — the heterogeneous-drift signal.

    The case this exists for: one global adapter reaching 0.85 overall while two
    domain-specific ones reached 0.94, because the corpus had two regions and the
    adapter split the difference. An overall figure cannot show that; the gap
    between it and the tail can.

    "Sparsest" means fewest members. Small clusters are where heterogeneous
    drift hides: a region with fifty documents contributes almost nothing to the
    mean and can be losing most of its quality unnoticed.

    Args:
        per_query: Per-query recall, aligned with ``cluster_labels``.
        cluster_labels: Cluster of each query.
        decile: Fraction of clusters counted as the tail, smallest first.
        min_queries: Fewest queries the tail may hold and still be reported.

    Returns:
        Mean recall over queries in the tail clusters, or ``None`` when the tail
        holds too few clusters or too few queries to mean anything.
    """
    if per_query.size == 0 or cluster_labels.size != per_query.size:
        return None

    labels, counts = np.unique(cluster_labels, return_counts=True)
    if labels.size < _MIN_CLUSTERS_FOR_TAIL:
        return None

    n_tail = max(1, round(labels.size * decile))
    sparsest = labels[np.argsort(counts)[:n_tail]]
    in_tail = np.isin(cluster_labels, sparsest)
    # A mean over a handful of queries is noise with a decimal point, and this
    # one feeds a warning about heterogeneous drift. Reporting nothing is the
    # honest answer; a real query log concentrates on documents people actually
    # ask about, which are in dense clusters, so the tail can end up nearly
    # empty even on a large corpus.
    if int(in_tail.sum()) < min_queries:
        return None
    return float(per_query[in_tail].mean())


def estimate_reindex_cost(
    n_documents: int,
    *,
    seconds_per_document: float,
    watts: float | None = None,
) -> dict[str, float | None]:
    """What a full reindex would cost, from a rate this run measured.

    The rate comes from the sample this probe just embedded, so the estimate is
    calibrated on the user's own hardware and their own documents rather than on
    a table of averages. That matters: the same corpus takes minutes on a GPU
    and hours on a laptop, and a number that ignores which one you have is not
    worth printing.

    Energy is returned only when a power figure was actually measured; an
    estimate derived from a nominal TDP would read as a measurement.

    Args:
        n_documents: Size of the whole collection, not of the sample.
        seconds_per_document: Measured embedding rate.
        watts: Measured average power draw, if the hardware reported one.
    """
    seconds = max(0.0, n_documents * seconds_per_document)
    return {
        "n_documents": float(n_documents),
        "seconds": round(seconds, 1),
        "seconds_per_document": seconds_per_document,
        "energy_wh": None if watts is None else round(watts * seconds / 3600.0, 3),
    }


def ndcg_at_k(
    retrieved: np.ndarray,
    relevant: Sequence[set[int]],
    k: int,
    *,
    grades: Sequence[dict[int, float]] | None = None,
) -> float:
    """Normalised discounted cumulative gain — the rank-sensitive metric.

    Recall counts what was found; nDCG counts *where*. The difference is not
    academic. Measured on BEIR/scifact, MiniLM to bge-base: the adapter improved
    recall@10 from 0.7833 to 0.7867 while nDCG@10 fell from 0.6451 to 0.6294. It
    retrieved the same documents and ordered them worse, and a recall-only
    decision called that an improvement.

    Which of the two matters depends on what consumes the results. A RAG
    pipeline that hands ten chunks to a model re-ranks them implicitly, so
    recall is close to the real objective. A list shown to a person does not,
    and there nDCG is.

    Args:
        retrieved: Retrieved document indices, best first.
        relevant: Relevant document ids per query.
        k: Cut-off.
        grades: Per-query graded relevance. Binary when omitted, which is what
            the sparsity-matched ground truth provides.

    Returns:
        Mean nDCG@k over queries that have at least one relevant document.
        Queries with none are skipped rather than scored zero, matching
        :func:`recall_at_k`.
    """
    if retrieved.size == 0:
        return 0.0

    # Precomputed once: the discount depends only on rank.
    discount = 1.0 / np.log2(np.arange(2, k + 2))

    scores: list[float] = []
    for row, rel in zip(retrieved[:, :k], relevant, strict=True):
        if not rel:
            continue
        grade = (grades[len(scores)] if grades is not None else None) or {}
        gains = np.array(
            [float(grade.get(int(doc), 1.0)) if int(doc) in rel else 0.0 for doc in row],
            dtype=np.float64,
        )
        dcg = float((gains * discount[: gains.size]).sum())

        # The best achievable ordering for this query: every relevant document
        # first, highest grade first.
        ideal = np.sort(np.array([float(grade.get(doc, 1.0)) for doc in rel], dtype=np.float64))[
            ::-1
        ][:k]
        idcg = float((ideal * discount[: ideal.size]).sum())
        scores.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(scores)) if scores else 0.0
