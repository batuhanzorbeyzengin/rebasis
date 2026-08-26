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

* **Calibrated merge** (preferred where its assumption holds). The isotonic
  calibrator maps old-space scores onto the new-space distribution, after which
  the two are comparable. The calibrator itself is monotone and cannot reorder
  anything — but the *merge* around it can, because a step function produces ties
  and a tie has to be broken by something. See `calibrated_merge` for what breaks
  them and what it cost before it did.

  **Its assumption is worth stating, because the code cannot check it.** The
  calibrator was fitted to map *bridged old-space* scores onto the distribution
  the **new model** produces. That holds while the migrated half really is the
  new model's own vectors. Where a migration instead writes an adapter's image of
  the old vectors, those records are not in that distribution, they score
  systematically low against a raw new-model query, and the calibrated side
  starves — measured, the migrated half took as little as 0.3% of the result at
  90% migrated. `calibrated_merge` branches on whether a calibrator *exists*,
  which is not the same question, and it has no way to ask the right one.
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

    # Each id keeps its best score and, alongside it, the rank it held on the
    # side that produced that score. The rank is what breaks a tie *within* one
    # side; see the sort below for why it has to.
    merged: dict[str, tuple[float, int]] = {}

    def offer(doc_id: str, score: float, rank: int) -> None:
        current = merged.get(doc_id)
        if current is None or score > current[0]:
            merged[doc_id] = (score, rank)

    if old_hits:
        raw = np.array([h.score for h in old_hits], dtype=np.float32)
        for hit, score in zip(old_hits, calibrator.transform(raw), strict=True):
            offer(hit.id, float(score), hit.rank)
    for hit in new_hits:
        offer(hit.id, hit.score, hit.rank)

    # Three keys, and the middle one was missing.
    #
    # The calibrator is isotonic regression: pool-adjacent-violators produces a
    # step function with far fewer levels than it has inputs, and `clip` flattens
    # both tails. Measured, ten bridged scores land on **five to seven** distinct
    # calibrated values — so ties are the common case here, not the rare one the
    # original comment assumed. Sorting on `(-score, id)` alone then handed every
    # one of them to whichever document id sorts first, which is arbitrary.
    #
    # What that cost is visible at the endpoints. At 0% and 100% migrated the
    # index holds one space and there is a single right answer — what the store
    # returned. Measured over four corpora, this merge reproduced it on **4% to
    # 16%** of queries; reciprocal rank fusion, which never looks at a score,
    # reproduced it on 100%. A merge that cannot reduce to the single-space
    # answer is wrong at the endpoints rather than merely worse.
    #
    # `rank` fixes it without giving up what the id was there for. Two hits from
    # the same side hold different ranks, so their original order survives a
    # shared calibrated level. Two hits from *different* sides can hold the same
    # rank, and there the id still decides — which is the neutrality the original
    # comment was defending: a tie resolved by which side was passed first is a
    # standing bias toward one embedding space, and that argument was right about
    # cross-side ties and silent about within-side ones.
    ordered = sorted(merged.items(), key=lambda item: (-item[1][0], item[1][1], item[0]))[:k]
    return [
        _Hit(id=doc_id, score=score, rank=rank) for rank, (doc_id, (score, _)) in enumerate(ordered)
    ]
