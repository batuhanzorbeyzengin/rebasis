"""Hybrid search across a partially migrated index.

During a gradual migration the corpus is split: some records still carry
old-model vectors, some have been rewritten with the new model. A query has to
reach both, and the two result sets carry **scores from different spaces** that
cannot be compared directly.

    q_new    = f_new(q)
    q_old    = g_query(q_new)
    hits_old = old_index.search(q_old, k)
    hits_new = new_index.search(q_new, k)
    result   = calibrated_merge(hits_old, hits_new, k)

Two merge strategies, and the fallback matters more than it looks:

* **Calibrated merge** (preferred). The isotonic calibrator maps old-space scores
  onto the new-space distribution, after which the two are directly comparable.
  Being monotone it cannot reorder either side.
* **Reciprocal rank fusion** (fallback). When no calibrator exists, scores are
  discarded and only ranks are used. That is strictly less information, but it
  is *correct* — whereas comparing raw scores across the two spaces is not.
  M0 measured a median KS distance of 0.924 between them, so raw comparison
  would let one side dominate for reasons unrelated to relevance.

This is what keeps queries correct on a partially migrated index, with quality
moving in one direction only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rebasis.core.calibration import ScoreCalibrator
    from rebasis.types import Hit

__all__ = ["RRF_K", "calibrated_merge", "reciprocal_rank_fusion"]

#: RRF's smoothing constant. 60 is the value from the original paper and the
#: de-facto default; the metric is not sensitive to it.
RRF_K = 60


def reciprocal_rank_fusion(*result_sets: Sequence[Hit], k: int, rrf_k: int = RRF_K) -> list[Hit]:
    """Merge result sets by rank alone.

    Used when no calibrator is available. Ranks are all that can be compared
    honestly across two embedding spaces, so this deliberately throws the scores
    away rather than pretending they are commensurable.
    """
    from rebasis.types import Hit as _Hit

    scores: dict[str, float] = {}
    for hits in result_sets:
        for hit in hits:
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (rrf_k + hit.rank + 1)

    # Ties broken on the id, not on insertion order. Every result set starts at
    # rank 0, so each one's best hit scores exactly `1/(rrf_k + 1)` and a
    # first-place tie is the *common* case rather than an edge one — and a
    # stable sort over a dict filled set by set would hand every one of them to
    # whichever set was passed first. On a half-migrated index that is a
    # standing bias toward one embedding space, which is precisely what fusing
    # by rank is supposed to avoid.
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
    return [_Hit(id=doc_id, score=score, rank=rank) for rank, (doc_id, score) in enumerate(ordered)]


def calibrated_merge(
    old_hits: Sequence[Hit],
    new_hits: Sequence[Hit],
    *,
    k: int,
    calibrator: ScoreCalibrator | None = None,
) -> list[Hit]:
    """Merge old-index and new-index results into one ranking.

    With a calibrator the old-space scores are mapped onto the new-space
    distribution and the two are merged by score. Without one this falls back to
    :func:`reciprocal_rank_fusion`, because comparing raw scores across spaces
    would let the space with the wider distribution win regardless of relevance.

    A document appearing in both sets keeps its **better** score rather than
    being counted twice: it is one document, and during migration overlap is
    expected rather than exceptional.
    """
    if calibrator is None:
        return reciprocal_rank_fusion(old_hits, new_hits, k=k)

    import numpy as np

    from rebasis.types import Hit as _Hit

    merged: dict[str, float] = {}
    if old_hits:
        raw = np.array([h.score for h in old_hits], dtype=np.float32)
        for hit, score in zip(old_hits, calibrator.transform(raw), strict=True):
            merged[hit.id] = max(merged.get(hit.id, float("-inf")), float(score))
    for hit in new_hits:
        merged[hit.id] = max(merged.get(hit.id, float("-inf")), hit.score)

    # Same tie-break as the fusion path, for the same reason. Calibrated scores
    # collide far less often than RRF's, but "less often" is not "never" and a
    # tie resolved by which side was passed first is a bias either way.
    ordered = sorted(merged.items(), key=lambda item: (-item[1], item[0]))[:k]
    return [_Hit(id=doc_id, score=score, rank=rank) for rank, (doc_id, score) in enumerate(ordered)]
