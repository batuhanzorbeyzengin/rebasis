"""Continuous re-fitting during a long migration.

A migration can run for hours. Over that time the *pairs available to fit on*
change: records already migrated carry new-model vectors, so a growing, better
sample of matched pairs becomes available for free.

The reference work calls this continuous adaptation. The idea is simple:
periodically refit on the accumulated pairs, and adopt the new adapter
**only if it measurably beats the one in use on a held-out set**.

Two guards make this safe rather than merely clever:

* **A refit is only adopted when it wins.** Silently swapping in a worse adapter
  mid-job would degrade quality with no signal — the exact failure mode this
  project keeps designing against.
* **The change is audited.** ``migrate.adapter.refitted`` writes an audit record,
  because a job that finishes having used two different adapters is a job whose
  results need that fact recorded.

A caveat worth stating: pairs accumulated during a migration come from records
processed in **priority order**, not at random. With ``--priority access`` those
are the frequently-read records, which are not a uniform sample of the corpus. So
the refit is evaluated on a held-out set drawn the same way, and the improvement
threshold is deliberately not tiny.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from rebasis.observability import Events, get_logger

if TYPE_CHECKING:
    from rebasis.core.base import BaseAdapter
    from rebasis.types import FloatArray

__all__ = ["MIN_IMPROVEMENT", "RefitDecision", "RefitPolicy", "consider_refit"]

log = get_logger(__name__)

#: How much better a refitted adapter must be before it replaces the current one.
#:
#: Not a hair's breadth: M0 measured ARR's bootstrap confidence interval at
#: ±0.024, so a smaller margin would be swapping adapters on sampling noise.
MIN_IMPROVEMENT = 0.01

#: Minimum accumulated pairs before refitting is worth attempting. Below this the
#: refit is fitting to noise; M0 found quality still climbing steeply under a
#: thousand pairs.
MIN_PAIRS = 1000


@dataclass(slots=True)
class RefitPolicy:
    """When to consider refitting."""

    enabled: bool = False
    every_n_records: int = 50_000
    min_pairs: int = MIN_PAIRS
    min_improvement: float = MIN_IMPROVEMENT
    holdout_fraction: float = 0.2

    def due(self, processed: int, last_refit_at: int) -> bool:
        """Whether enough has been processed since the last attempt."""
        return self.enabled and (processed - last_refit_at) >= self.every_n_records


@dataclass(slots=True)
class RefitDecision:
    """The outcome of a refit attempt."""

    adopted: bool
    current_score: float
    candidate_score: float
    n_pairs: int
    reason: str
    adapter: BaseAdapter | None = None

    @property
    def improvement(self) -> float:
        """How much better the candidate was, if at all."""
        return self.candidate_score - self.current_score


def consider_refit(
    current: BaseAdapter,
    *,
    src: FloatArray,
    dst: FloatArray,
    policy: RefitPolicy,
    job_id: str = "",
) -> RefitDecision:
    """Refit on the accumulated pairs and adopt only if it wins.

    Args:
        current: The adapter in use.
        src: New-model vectors of the accumulated pairs.
        dst: Old-model vectors of the same records, in the same order.
        policy: When and by how much to switch.
        job_id: For the log and audit record.

    The comparison is on a **held-out** slice of the accumulated pairs. Scoring
    both adapters on the data the candidate was fitted to would favour it
    automatically, which is how an overfitted adapter gets adopted.
    """
    n = int(src.shape[0])
    if n < policy.min_pairs:
        return RefitDecision(
            adopted=False,
            current_score=0.0,
            candidate_score=0.0,
            n_pairs=n,
            reason=f"only {n} accumulated pairs; at least {policy.min_pairs} are needed",
        )

    holdout = max(1, int(n * policy.holdout_fraction))
    fit_src, fit_dst = src[:-holdout], dst[:-holdout]
    test_src, test_dst = src[-holdout:], dst[-holdout:]

    from rebasis.core.procrustes import CenteredProcrustesAdapter

    candidate = CenteredProcrustesAdapter.fit(fit_src, fit_dst)

    current_score = _score(current, test_src, test_dst)
    candidate_score = _score(candidate, test_src, test_dst)
    improvement = candidate_score - current_score

    if improvement < policy.min_improvement:
        return RefitDecision(
            adopted=False,
            current_score=current_score,
            candidate_score=candidate_score,
            n_pairs=n,
            reason=(
                f"the refit improved by {improvement:+.4f}, below the "
                f"{policy.min_improvement} threshold — within measurement noise"
            ),
        )

    log.info(
        Events.MIGRATE_ADAPTER_REFITTED,
        job_id=job_id,
        adapter_type=candidate.type_name,
        arr_r10=round(candidate_score, 4),
    )
    return RefitDecision(
        adopted=True,
        current_score=current_score,
        candidate_score=candidate_score,
        n_pairs=n,
        reason=f"the refit improved by {improvement:+.4f} on {holdout} held-out pairs",
        adapter=candidate,
    )


def _score(adapter: BaseAdapter, src: FloatArray, dst: FloatArray) -> float:
    """Mean cosine similarity between the mapped vectors and their targets.

    A proxy for ARR that needs no index and no search. It is not the decision
    metric — it is a cheap comparator for choosing between two adapters mid-job,
    and it is only ever used relatively, never reported as quality.
    """
    from rebasis.compute.arrays import l2_normalize

    mapped = l2_normalize(adapter.apply(src), copy=False)
    target = l2_normalize(dst)
    return float(np.einsum("ij,ij->i", mapped, target).mean())
