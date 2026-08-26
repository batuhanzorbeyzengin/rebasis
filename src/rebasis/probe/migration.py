"""Fitting the map a migration needs, and scoring it on the question a migration asks.

An adapter has a direction, and the two are mirror images.

    query_to_old   f_new(q)  ->  the index's space     what `Bridge` sends
    old_to_new     f_old(d)  ->  the new model's space  what `migrate` writes

`rebasis fit` produces the first. This module produces the second, and it is not
the same fit with its arguments swapped — the *evaluation* differs, and that is
the part that matters.

**The two questions are not the same question.** A query map is judged on whether
a mapped query, sent at an untouched index, retrieves what a full reindex would.
A document map is judged on whether an index rewritten with it answers a **raw
new-model query** the way a full reindex would. The second is what a user has
after `migrate` finishes: no bridge, no adapter on the hot path, the new model
querying an index that is supposed to be in its own space. Scoring a forward map
with the query-side metric would measure a configuration nobody runs.

So the score here is exactly the thing being promised:

    migrated = A(f_old(d))  for every sampled document
    hits     = search(f_new(q), migrated)
    score    = recall(hits, ground truth) / oracle

with the ground truth being what the new model's own index returns. A real
reindex scores 1.0 by construction, so the number reads directly as *the
fraction of a reindex this migration delivers* — which is the number the
decision to migrate should be made on, and the one nothing in the tool could
produce before.

**What this does not do.** It does not make migrating a good idea. `ADR 10`
bounds retention by what the source space carries, and that bound applies to
documents exactly as it applies to queries; a forward map is not exempt from it
because it points the other way. The score is here so the answer is measured
rather than assumed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rebasis.compute import top_k_search
from rebasis.core import fit_candidates, l2_normalize, select_best
from rebasis.observability import Events, get_logger
from rebasis.probe.metrics import recall_at_k

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from rebasis.core.base import BaseAdapter
    from rebasis.core.selection import AdapterCandidate
    from rebasis.probe.groundtruth import GroundTruth
    from rebasis.types import FloatArray

__all__ = ["MigrationFit", "fit_migration_adapter"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class MigrationFit:
    """The forward map, and what a migration with it would be worth."""

    #: The chosen adapter. Maps the index's vectors into the new model's space.
    adapter: BaseAdapter
    #: Which family won.
    method: str
    #: Fraction of a full reindex the migrated index delivers, measured. A real
    #: reindex is 1.0 by construction, so this reads directly as the ratio.
    retention: float
    #: Every candidate's score, so the report can show what was tried.
    scores: dict[str, float]
    #: Seconds spent fitting all candidates.
    fit_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form, for the `.rbs` evaluation block and the audit record."""
        return {
            "direction": "old_to_new",
            "adapter_type": self.method,
            "migration_retention": round(self.retention, 4),
            "candidates": {name: round(score, 4) for name, score in sorted(self.scores.items())},
            "fit_seconds": round(self.fit_seconds, 2),
        }


def _retention(
    adapter: BaseAdapter,
    *,
    old_doc_vectors: FloatArray,
    new_query_vectors: FloatArray,
    ground_truth: GroundTruth,
    k: int,
) -> float:
    """Score one forward map by rewriting the index with it and querying.

    The whole sample is mapped, not only the held-out part: a migration rewrites
    every record, and a score computed over a subset would describe an index
    that was never built. The fit pairs are a subset of what is mapped, which is
    the same arrangement `migrate` produces — the adapter is fitted on some
    documents and applied to all of them.
    """
    migrated = l2_normalize(adapter.apply(old_doc_vectors), copy=False)
    indices, _ = top_k_search(new_query_vectors, migrated, k=k, self_mask=ground_truth.self_mask)
    oracle = ground_truth.oracle_recall or 1.0
    return float(recall_at_k(indices, ground_truth.relevant_sparse, k) / oracle)


def fit_migration_adapter(  # noqa: PLR0913 - one argument per input the fit needs
    *,
    old_doc_vectors: FloatArray,
    new_doc_vectors: FloatArray,
    new_query_vectors: FloatArray,
    ground_truth: GroundTruth,
    fit_indices: np.ndarray,
    k: int = 10,
    methods: Sequence[str] | None = None,
) -> MigrationFit:
    """Fit `old -> new` and score it on what a completed migration would deliver.

    Args:
        old_doc_vectors: The sampled documents as the index holds them.
        new_doc_vectors: The same documents under the candidate model, in the
            same row order. That correspondence is the fit's entire content.
        new_query_vectors: Queries under the candidate model — sent *raw*,
            because after a migration there is no adapter on the query path.
        ground_truth: What a full reindex returns. The denominator.
        fit_indices: Rows the adapter may be fitted on. Held-out rows are still
            *mapped* — every record moves in a migration — but never fitted on.
        k: Cut-off for the retention measurement.
        methods: Restrict the candidate list; ``None`` tries them all.

    Raises:
        RuntimeError: When no candidate could be fitted at all.
    """
    started = time.perf_counter()
    kwargs: dict[str, Any] = {"methods": list(methods)} if methods else {}
    # Source and target are the reverse of the query-side fit, and nothing else
    # about the call changes: the same families, the same preprocessing, the
    # same tie-break on cost. `normalize=False` because both sides arrive
    # normalised, as they do on the query side.
    candidates: list[AdapterCandidate] = fit_candidates(
        old_doc_vectors[fit_indices], new_doc_vectors[fit_indices], normalize=False, **kwargs
    )
    if not candidates:
        msg = "no adapter candidate could be fitted for the migration direction"
        raise RuntimeError(msg)

    for candidate in candidates:
        candidate.score = _retention(
            candidate.adapter,
            old_doc_vectors=old_doc_vectors,
            new_query_vectors=new_query_vectors,
            ground_truth=ground_truth,
            k=k,
        )

    winner = select_best(candidates)
    elapsed = time.perf_counter() - started
    scores = {c.method: c.score for c in candidates if c.score is not None}

    log.info(
        Events.FIT_ADAPTER_FITTED,
        adapter_type=winner.method,
        count=int(fit_indices.size),
        dim=int(old_doc_vectors.shape[1]),
        duration_ms=round(elapsed * 1000, 2),
    )
    return MigrationFit(
        adapter=winner.adapter,
        method=winner.method,
        retention=float(winner.score or 0.0),
        scores=scores,
        fit_seconds=elapsed,
    )
