"""Ground-truth tiers.

Three tiers, in descending order of trustworthiness:

* **T1 — a real query log.** Preferred whenever it exists. It is also the only
  tier that can answer "is the new model actually better here", because under T0
  the new model *is* the ground truth and is therefore perfect by construction.
* **T0 — documents as query proxies.** The default when no query log exists.
* **T2 — synthesised queries.** Not implemented in M1.

**M0 changed how T0 is built.** T0's ground truth was originally the new model's
exhaustive top-k neighbours — reproduce the whole neighbour set. Measured against
real queries across 4 corpora, that gave a mean absolute error of **0.261** and
agreed on the decision only **41.7%** of the time.

Re-measuring T0 with a **sparsity-matched** ground truth — count only the nearest
neighbour, mirroring the ~1.1 relevant documents per query that real qrels have —
cut the error to **0.095** and lifted agreement to 53.6%. The proxy was never the
problem; the strictness of the target was (section 3 of ``docs/m0-findings.md``).

So T0 reports both: the strict overlap, which says how faithfully the neighbour
set is reproduced, and the sparsity-matched recall, which is what the decision
rule uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from rebasis.observability import Events, get_logger
from rebasis.probe.metrics import top_k_search

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rebasis.types import FloatArray

__all__ = [
    "GroundTruth",
    "build_tier0",
    "build_tier1",
    "build_tier1_unjudged",
    "build_tier2",
]

log = get_logger(__name__)

#: How many of the ground-truth neighbours count as relevant in the
#: sparsity-matched variant. One, because real qrels average ~1.1 relevant
#: documents per query and that is what the decision thresholds were compared
#: against.
SPARSE_RELEVANT = 1

#: Above this, a synthesised task is trivial: every model finds the document a
#: query was cut from, so T2 cannot tell the two models apart.
#:
#: Measured: ``lead`` and ``title`` land at 0.98-0.99 on every corpus tried and
#: are correctly caught; ``keywords`` lands at 0.79-0.97 and is not. The guard
#: does what it says — but catching a trivial task is not the same as having a
#: usable estimate, which is what :data:`SYNTH_GAIN_ERROR` is for.
TRIVIAL_ORACLE_RECALL = 0.95

#: Absolute error to expect from a synthesised upgrade estimate, against the
#: figure a real query log gives.
#:
#: Measured over 60 runs — five corpora, three strategies, four held-out sizes:
#: mean absolute error 0.14 for ``lead``, 0.17 for ``title``, **0.22** for
#: ``keywords``. The value here is the worst of the three, because a guard set
#: to the best one would pass estimates the other strategies cannot support.
#:
#: The decision threshold is 1.0, so an estimate within this distance of it
#: cannot settle the question — a true gain of 1.05 can be estimated anywhere
#: from 0.83 to 1.27. Query count is not what limits this: raising the
#: synthesised set from 100 to 1,000 barely moved the error.
SYNTH_GAIN_ERROR = 0.22


@dataclass(slots=True)
class GroundTruth:
    """Relevance judgements plus the oracle results they were derived from."""

    tier: str
    query_indices: np.ndarray
    relevant: list[set[int]]
    relevant_sparse: list[set[int]]
    oracle_indices: np.ndarray
    oracle_scores: FloatArray
    self_mask: np.ndarray | None = None
    oracle_recall: float | None = None
    #: The oracle's recall on each query, in the same order as ``relevant_sparse``
    #: with unjudged queries dropped. ARR is a ratio of means, so its confidence
    #: interval has to resample both series together.
    oracle_recall_per_query: FloatArray | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def build_tier0(
    new_doc_vectors: FloatArray,
    new_query_vectors: FloatArray,
    query_indices: np.ndarray,
    *,
    k: int = 10,
) -> GroundTruth:
    """T0: held-out documents act as queries; the new model defines relevance.

    The ground truth is literally "what a full reindex would return": documents
    encoded with the new model, query encoded with the new model, exhaustive kNN.
    The adapter's job is to reproduce that inside the *old* index.

    Query documents sit inside the index, so each is masked out of its own
    results — without that, every query trivially retrieves itself and ARR is
    inflated for every candidate equally.
    """
    self_mask = query_indices.astype(np.int64)
    oracle_indices, oracle_scores = top_k_search(
        new_query_vectors, new_doc_vectors, k=k, self_mask=self_mask
    )

    relevant = [set(row.tolist()) for row in oracle_indices]
    relevant_sparse = [set(row[:SPARSE_RELEVANT].tolist()) for row in oracle_indices]

    from rebasis.probe.metrics import recall_per_query

    log.info(
        Events.PROBE_GROUNDTRUTH_COMPUTED,
        count=int(query_indices.size),
        tier="t0",
    )
    return GroundTruth(
        tier="t0",
        query_indices=query_indices,
        relevant=relevant,
        relevant_sparse=relevant_sparse,
        oracle_indices=oracle_indices,
        oracle_scores=oracle_scores,
        self_mask=self_mask,
        # Perfect by construction: the ground truth IS the new model's output.
        oracle_recall=1.0,
        oracle_recall_per_query=recall_per_query(oracle_indices, relevant_sparse, k),
        metadata={"note": "documents used as query proxies, not real queries"},
    )


def build_tier1(
    new_doc_vectors: FloatArray,
    new_query_vectors: FloatArray,
    qrels: list[set[int]],
    *,
    k: int = 10,
) -> GroundTruth:
    """T1: real queries with human relevance judgements.

    The oracle is no longer perfect — even a full reindex does not achieve
    complete recall — which is what makes ARR a genuine ratio here rather than a
    bare recall figure.
    """
    from rebasis.probe.metrics import recall_at_k, recall_per_query

    oracle_indices, oracle_scores = top_k_search(new_query_vectors, new_doc_vectors, k=k)
    oracle_recall = recall_at_k(oracle_indices, qrels, k)
    oracle_per_query = recall_per_query(oracle_indices, qrels, k)

    log.info(
        Events.PROBE_GROUNDTRUTH_COMPUTED,
        count=len(qrels),
        tier="t1",
    )
    return GroundTruth(
        tier="t1",
        query_indices=np.arange(len(qrels)),
        relevant=qrels,
        relevant_sparse=qrels,
        oracle_indices=oracle_indices,
        oracle_scores=oracle_scores,
        self_mask=None,
        oracle_recall=oracle_recall,
        oracle_recall_per_query=oracle_per_query,
        metadata={"note": "real query log with relevance judgements"},
    )


def build_tier1_unjudged(
    new_doc_vectors: FloatArray,
    new_query_vectors: FloatArray,
    *,
    k: int = 10,
) -> GroundTruth:
    """Real queries, no relevance judgements — the common case.

    Most people have a search log. Almost nobody has a human-labelled one, and
    rebasis used to reject that log with a message about qrel ids, which is not
    what happened and not what to do about it.

    What such a log can still answer is the retention half of the question:
    the full reindex's top-k is the reference, and ARR becomes "how much of what
    a rebuild would have returned does the adapter return", measured on the
    queries people actually type rather than on documents standing in for them.

    What it cannot answer is whether the new model is *better*. Deciding that
    needs someone to have said which document was right; the oracle cannot grade
    itself. So ``upgrade_gain`` is not computed from this tier and the run is
    reported as provisional — the same treatment T0 gets, for the same reason.
    """
    from rebasis.probe.metrics import recall_per_query

    oracle_indices, oracle_scores = top_k_search(new_query_vectors, new_doc_vectors, k=k)
    qrels = [{int(position) for position in row} for row in oracle_indices]

    log.info(
        Events.PROBE_GROUNDTRUTH_COMPUTED,
        count=len(qrels),
        tier="t1u",
    )
    return GroundTruth(
        tier="t1u",
        query_indices=np.arange(len(qrels)),
        relevant=qrels,
        relevant_sparse=qrels,
        oracle_indices=oracle_indices,
        oracle_scores=oracle_scores,
        self_mask=None,
        # 1.0 by construction: the oracle defined the answer. ARR is then a
        # ratio to a perfect denominator, which is what makes it read as pure
        # retention rather than as a share of an imperfect ceiling.
        oracle_recall=1.0,
        oracle_recall_per_query=recall_per_query(oracle_indices, qrels, k),
        metadata={
            "note": "real queries without relevance judgements; retention only",
        },
    )


def build_tier2(
    new_doc_vectors: FloatArray,
    new_query_vectors: FloatArray,
    source_positions: Sequence[int],
    *,
    k: int = 10,
    strategy: str = "lead",
) -> GroundTruth:
    """T2: queries synthesised from the documents.

    The judgement is structural, not model-derived: query *i* was extracted from
    document ``source_positions[i]``, so that document is its answer whatever
    either embedding model thinks. That is the whole point — it makes the
    oracle's recall a real number, and therefore makes ``upgrade_gain``
    computable without a query log.

    **The source document is not masked out**, unlike T0's proxies. At T0 the
    query document would trivially retrieve itself and inflate every candidate
    equally; here that document *is* the correct answer, so masking it would
    make recall structurally zero and the tier would measure nothing at all.

    The cost of not masking is that an easy strategy can make the task trivial:
    a lead sentence is a literal substring of its document, and every model
    finds it. When that happens both models score near 1.0, ``upgrade_gain``
    comes back near 1.0, and the estimate carries no information — which is why
    :attr:`GroundTruth.oracle_recall` is reported and why ``keywords`` exists as
    a strategy that does not hand the answer to the retriever.
    """
    from rebasis.probe.metrics import recall_at_k, recall_per_query

    sources = np.asarray(source_positions, dtype=np.int64)
    oracle_indices, oracle_scores = top_k_search(new_query_vectors, new_doc_vectors, k=k)
    qrels = [{int(position)} for position in sources]
    oracle_recall = recall_at_k(oracle_indices, qrels, k)

    log.info(
        Events.PROBE_GROUNDTRUTH_COMPUTED,
        count=len(qrels),
        tier="t2",
    )
    return GroundTruth(
        tier="t2",
        query_indices=sources,
        relevant=qrels,
        relevant_sparse=qrels,
        oracle_indices=oracle_indices,
        oracle_scores=oracle_scores,
        self_mask=None,
        oracle_recall=oracle_recall,
        oracle_recall_per_query=recall_per_query(oracle_indices, qrels, k),
        metadata={
            "note": "queries synthesised from documents; not real user queries",
            "strategy": strategy,
            # Near 1.0 means the synthesis handed the answer to the retriever
            # and the upgrade estimate carries no information.
            "task_too_easy": oracle_recall > TRIVIAL_ORACLE_RECALL,
        },
    )
