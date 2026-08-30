"""Comparing several candidate models on one corpus.

``probe`` answers "is this model better on my corpus" without rebuilding the
index. This module asks it of N candidates at once, which is a different
question in one respect that decides the whole design: what a user wants back is
an **ordering**, not N independent verdicts.

That matters because of what
[section 9](../bridge-band.md#9-what-the-counting-is-worth) found. Read as a
threshold, ``probe``'s estimate is weak — the count that said otherwise was an
identity. Read as a **ranker** it carries real information: over 57 runs it
orders them by the margin they returned at Spearman rho = 0.60, p ~ 1e-6. A
multi-candidate comparison is exactly a ranking problem, which is the one shape
this instrument is measured to be good at.

**One sample, one split, one reference.** Every candidate is scored on the same
drawn sample, the same fit/held-out split and the same queries; only the
embedding pass and everything downstream of it is per candidate. Redrawing per
candidate would introduce the shift `docs/access-weighting.md` measured — a
4,000-document mini-index already sits +0.048 above the whole-corpus quantity —
and consistency across candidates matters more here than absolute accuracy,
because the answer is a comparison.

**The index's model is the reference, not a candidate.** It is already in the
index and its vectors are read rather than recomputed. Passing it as a candidate
would be asking how well the incumbent bridges to itself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rebasis.observability import Events, Spans, get_logger, span
from rebasis.probe.decision import BORDERLINE_BAND
from rebasis.probe.session import draw_corpus_sample, probe_store

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from rebasis.audit import AuditWriter
    from rebasis.probe.runner import ProbeResult
    from rebasis.probe.session import CorpusSample, QueryLog
    from rebasis.store.base import VectorStore
    from rebasis.types import Embedder

__all__ = [
    "CandidateComparison",
    "ComparisonResult",
    "compare_models",
    "estimate_candidate_cost",
]

log = get_logger(__name__)

#: Documents embedded to measure a candidate's rate before committing to it.
#:
#: Small enough to be free and large enough not to be measuring warm-up: a
#: transformer's first forward pass allocates buffers the next ones reuse, so a
#: rate taken from one document would over-state the cost several-fold.
COST_PROBE_DOCUMENTS = 64

#: Sample size the first round of ``--tiered`` runs at.
#:
#: A round that cannot separate anything is a round that cost time and decided
#: nothing, and where this should sit is a measurement rather than a guess —
#: `docs/model-selection.md` M2 is that measurement. Until it lands this is the
#: figure the flag defaults to and the report says which round it came from.
TIERED_FIRST_ROUND = 2_000

#: Backends that send document text off the machine.
#:
#: Named rather than inferred: `pyproject.toml` already records that
#: ``openai_compat`` is the only backend that can, and a comparison run is the
#: worst place to find that out afterwards — it embeds the same sample once per
#: candidate, so a single careless candidate sends the corpus somewhere N times
#: over.
_REMOTE_BACKENDS = frozenset({"openai", "openai_compat", "ollama"})


@dataclass(slots=True)
class CandidateComparison:
    """One candidate, measured on the shared sample."""

    model: str
    result: ProbeResult
    duration_seconds: float
    #: Which round of a tiered run produced this. ``1`` on an untiered run.
    round_number: int = 1
    #: Set when a candidate was dropped after the first round rather than
    #: measured at the full sample. Its numbers are real and were taken on a
    #: smaller sample, which is a different precision and says so.
    eliminated: bool = False

    @property
    def upgrade_gain(self) -> float | None:
        """How much better this candidate retrieves than the index's own model.

        The model-selection quantity, and the one the table is ordered by. It
        compares the candidate against the **incumbent** rather than against the
        other candidates, so a row is meaningful on its own as well as in the
        ordering.
        """
        return self.result.decision.upgrade_gain

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form — the whole probe result, under the model's name."""
        decision = self.result.decision
        return {
            "model": self.model,
            "upgrade_gain": (
                None if decision.upgrade_gain is None else round(decision.upgrade_gain, 4)
            ),
            "arr": round(decision.arr_at_k, 4),
            "bridge_advantage": (
                None if decision.bridge_advantage is None else round(decision.bridge_advantage, 4)
            ),
            "cascade_arr": (
                None if decision.cascade_arr is None else round(decision.cascade_arr, 4)
            ),
            "cascade_advantage": (
                None if decision.cascade_advantage is None else round(decision.cascade_advantage, 4)
            ),
            "decision": decision.decision,
            "arrangement": decision.arrangement,
            "provisional": decision.provisional,
            "reindex_cost": self.result.reindex_cost,
            "round": self.round_number,
            "eliminated": self.eliminated,
            "duration_seconds": round(self.duration_seconds, 2),
        }


#: What the ordering is worth, printed with every comparison.
#:
#: Not a footnote, and not a general statement about ``probe``. It quotes the
#: measurement that scored **this command**, and that measurement lost: over 16
#: corpora with three candidates each, picking whatever tops the published MTEB
#: table got the best candidate right 14 times and this ordering got it right 9.
#: A user reading the table has to be holding that number, because the
#: alternative is free and they already have it.
#:
#: What survives is weaker and real: the ordering correlates with the truth, and
#: it improves with the sample. `docs/model-selection.md` is the whole result.
RANKING_CAVEAT = (
    "This ordering did not beat the published leaderboard. Measured over 16 "
    "corpora against human judgements, picking whatever scores highest on MTEB "
    "named the genuinely best candidate 14 times out of 16; this ordering named "
    "it 9 (docs/model-selection.md). What it carries is a correlation with the "
    "true ordering, mean Spearman rho = +0.47, improving with --sample. Read it "
    "as evidence about your corpus to weigh against the table, not as a "
    "replacement for it, and treat a small gap between two rows as unresolved."
)


@dataclass(slots=True)
class ComparisonResult:
    """Every candidate against one reference, on one sample."""

    candidates: list[CandidateComparison]
    #: The model already in the index, and where that was established.
    reference: dict[str, Any]
    #: The draw every candidate shares. One entry, because there is one draw.
    sample: dict[str, Any]
    ranking_caveat: str = RANKING_CAVEAT
    #: Candidates whose text would leave the machine, and which backend does it.
    remote_candidates: list[str] = field(default_factory=list)

    def ranked(self) -> list[CandidateComparison]:
        """Candidates best first, by how much better they are than the incumbent.

        Ordered on ``upgrade_gain`` rather than on a break-even: the break-even
        answers "should I bridge to this one", which is a question about the
        adapter, and the question here is which model to adopt. A candidate with
        no estimate sorts last — it is not a low score, it is the absence of one,
        and putting it in the middle of the order would read as a measurement.
        """
        return sorted(
            self.candidates,
            key=lambda c: (c.upgrade_gain is not None, c.upgrade_gain or 0.0),
            reverse=True,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form, for scripts and for the audit record."""
        return {
            "candidates": [c.to_dict() for c in self.ranked()],
            "reference": dict(self.reference),
            "sample": dict(self.sample),
            "ranking_caveat": self.ranking_caveat,
            "remote_candidates": list(self.remote_candidates),
        }


def estimate_candidate_cost(
    embedder: Embedder, texts: Sequence[str], *, total: int
) -> dict[str, float]:
    """Time a short embedding pass and extrapolate it to the whole sample.

    The same rule ``probe``'s reindex estimate follows: the rate is measured on
    this machine with this model, because the same corpus takes minutes on a GPU
    and hours on a laptop and a number that ignores which one you have is not
    worth printing.

    It is measured per candidate rather than once, because that is the thing
    that varies — a 300M-parameter model and an 8M static one differ by two
    orders of magnitude, and the whole reason to print an estimate is that one
    of the candidates might be the expensive one.
    """
    probe_texts = list(texts[:COST_PROBE_DOCUMENTS])
    started = time.perf_counter()
    embedder.encode(probe_texts, kind="document", progress=False)
    elapsed = time.perf_counter() - started
    per_document = elapsed / max(1, len(probe_texts))
    return {
        "seconds_per_document": per_document,
        "seconds": round(per_document * total, 1),
        "measured_on": float(len(probe_texts)),
    }


def compare_models(  # noqa: PLR0913 - one argument per pipeline input
    store: VectorStore,
    candidates: Mapping[str, Embedder],
    *,
    old_embedder: Embedder | None = None,
    query_log: QueryLog | None = None,
    size: int = 10_000,
    heldout: int = 1_000,
    strategy: str = "stratified",
    k: int = 10,
    seed: int = 0,
    tiered: bool = False,
    first_round: int = TIERED_FIRST_ROUND,
    synth_queries: str | None = None,
    cache_dir: Path | str | None = None,
    audit: AuditWriter | None = None,
    store_uri: str = "",
    old_model: str = "",
    device: str = "cpu",
    access_counts: Mapping[str, float] | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> ComparisonResult:
    """Score every candidate against the index's own model, on one sample.

    Args:
        store: The index. Read only, as everywhere in ``probe``.
        candidates: Model id to embedder, in the order they were named.
        old_embedder: The index's model, needed to measure the upgrade.
        query_log: A real query log. Without one every row is provisional, and
            the *ordering* is the thing this command exists for — so the caveat
            is louder here than in a single-candidate run.
        size: Documents in the shared sample.
        heldout: Documents held out as query proxies.
        strategy: How the sample is drawn.
        k: Cut-off for every metric.
        seed: Recorded so the comparison can be replayed.
        tiered: Run every candidate at ``first_round`` first and take only what
            the small round could not separate through to the full sample.
        first_round: Sample size for that first round.
        synth_queries: Estimate the upgrade from the documents, with no log.
        cache_dir: Where embeddings are kept between runs. Keyed on the encoding
            profile, so candidates cannot contaminate one another and a
            candidate somebody evaluated once ages out on its own under ``gc``.
        audit: Where the decisions are recorded.
        store_uri: Recorded with them.
        old_model: The reference model's id.
        device: Where to run.
        access_counts: Weights which sampled records become query proxies.
        on_stage: Progress callback.
    """
    stage = on_stage if on_stage is not None else (lambda _text: None)
    with span(Spans.PROBE, {"seed": seed, "k": k, "count": len(candidates)}):
        stage("Sampling the index — once, for every candidate")
        corpus = draw_corpus_sample(
            store,
            size=size,
            heldout=heldout,
            strategy=strategy,
            seed=seed,
            access_counts=access_counts,
        )

        shared: dict[str, Any] = {
            "old_embedder": old_embedder,
            "query_log": query_log,
            "k": k,
            "seed": seed,
            "synth_queries": synth_queries,
            "cache_dir": cache_dir,
            "audit": audit,
            "store_uri": store_uri,
            "old_model": old_model,
            "device": device,
        }

        selected = dict(candidates)
        eliminated: list[CandidateComparison] = []
        if tiered and len(selected) > 1:
            stage(f"Round one: every candidate on {min(first_round, len(corpus)):,} documents")
            # A second draw rather than a subset of the first. A subset would
            # keep "one sample" literally true and would not be what the user
            # gets from `--sample 2000`: the strategy stratifies over whatever
            # it is given, so a slice of a 10,000-document stratified draw is
            # not a 2,000-document stratified draw. The round that decides is
            # still the full one, on the shared sample.
            small = draw_corpus_sample(
                store,
                size=min(first_round, size),
                heldout=max(1, heldout * min(first_round, size) // max(1, size)),
                strategy=strategy,
                seed=seed,
                access_counts=access_counts,
            )
            first = [
                _measure(store, model, embedder, small, shared, round_number=1)
                for model, embedder in selected.items()
            ]
            survivors = _survivors(first)
            eliminated = [c for c in first if c.model not in survivors]
            for candidate in eliminated:
                candidate.eliminated = True
            selected = {m: e for m, e in selected.items() if m in survivors}
            log.info(
                Events.PROBE_RUN_STARTED,
                count=len(selected),
                seed=seed,
            )

        measured = []
        for index, (model, embedder) in enumerate(selected.items(), start=1):
            stage(f"Candidate {index} of {len(selected)}: {model}")
            measured.append(
                _measure(
                    store,
                    model,
                    embedder,
                    corpus,
                    shared,
                    round_number=2 if tiered and len(candidates) > 1 else 1,
                )
            )

    return ComparisonResult(
        candidates=[*measured, *eliminated],
        reference={
            "model": old_model or (old_embedder.profile.model_id if old_embedder else ""),
            # The index is where the reference's vectors came from, and saying
            # so is not decoration: it is the reason the reference costs nothing
            # to evaluate and cannot be one of the candidates.
            "source": "index",
        },
        sample={
            "size": len(corpus),
            "seed": corpus.seed,
            "strategy": corpus.strategy,
            "n_total": corpus.n_total,
            "n_queries": int(corpus.query_positions.size),
            "shared": True,
            "tiered": bool(tiered and len(candidates) > 1),
            "first_round": first_round if tiered else None,
        },
        remote_candidates=sorted(remote_candidates(candidates)),
    )


def remote_candidates(candidates: Mapping[str, Embedder]) -> list[str]:
    """Which candidates would send document text off this machine.

    Read off the embedder's own module rather than off the model id, because a
    model id says nothing about where it runs: the same name can be a local
    checkpoint or a hosted endpoint, and it is the backend that decides.
    """
    remote = []
    for model, embedder in candidates.items():
        module = type(embedder).__module__.rsplit(".", 1)[-1]
        if module in _REMOTE_BACKENDS:
            remote.append(model)
    return remote


def _measure(  # noqa: PLR0913 - the store, the candidate, the sample and the shared inputs
    store: VectorStore,
    model: str,
    embedder: Embedder,
    corpus: CorpusSample,
    shared: dict[str, Any],
    *,
    round_number: int,
) -> CandidateComparison:
    """One candidate against the shared sample."""
    started = time.perf_counter()
    result, _ = probe_store(store, embedder, sample=corpus, **shared)
    return CandidateComparison(
        model=model,
        result=result,
        duration_seconds=time.perf_counter() - started,
        round_number=round_number,
    )


def _survivors(first_round: list[CandidateComparison]) -> set[str]:
    """Which candidates the small round could not separate from the leader.

    The band the decision rule already reports its own borderline cases at. A
    round that eliminated everything but the leader would be claiming the small
    sample resolved an ordering the full sample exists to resolve; a round that
    eliminated nothing would have cost time and decided nothing. The band is
    where those two meet, and it is the same +-0.025 for the same reason —
    below it, two numbers from one sample are not two numbers.

    Every candidate survives when none has an estimate: with nothing measured
    there is nothing to eliminate on.
    """
    scored = [c for c in first_round if c.upgrade_gain is not None]
    if not scored:
        return {c.model for c in first_round}
    best = max(c.upgrade_gain or 0.0 for c in scored)
    survivors = {c.model for c in scored if (c.upgrade_gain or 0.0) >= best - BORDERLINE_BAND}
    # A candidate the round could not score is not a candidate the round
    # rejected. It goes through, and the full sample gets to answer.
    survivors.update(c.model for c in first_round if c.upgrade_gain is None)
    return survivors
