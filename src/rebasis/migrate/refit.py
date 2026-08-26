"""Continuous re-fitting during a long migration.

A migration can run for hours, and over that time the index it is rewriting may
stop looking like the index the adapter was fitted on. Periodically refit on
pairs drawn from what is *left*, and adopt the new adapter **only if it
measurably beats the one in use on a held-out set**.

Two guards make this safe rather than merely clever:

* **A refit is only adopted when it wins.** Silently swapping in a worse adapter
  mid-job would degrade quality with no signal — the exact failure mode this
  project keeps designing against.
* **The change is audited.** ``migrate.adapter.refitted`` writes an audit record,
  because a job that finishes having used two different adapters is a job whose
  results need that fact recorded.

## What this is for, measured

An earlier version of this docstring said pairs become available "for free"
during a migration, because records already migrated carry new-model vectors.
**They do not.** A migrated record carries ``A(old)`` — the adapter's own image
of the old vector — so fitting on those pairs fits ``A`` to reproduce ``A``.
Every real pair costs a document re-embedded, which is why the engine needs an
embedder to do this at all.

Once it is being paid for, `spikes/continuous_refit.py` says what it buys, over
216 cells on real corpora:

* On a corpus that has not changed, a refit is a **pair-count** effect and
  nothing more. Against `rebasis fit`'s default 4,000-pair budget, even 12,000
  pairs moves retention a median +0.0075 and clears the 0.01 threshold below in
  17% of cells. Where the pairs came from does not matter: ``remaining`` against
  ``migrated`` at equal K is -0.002, which is noise.
* On a corpus that **grew into a domain the adapter never saw**, refitting on
  1,000 pairs drawn from what is left is worth a median **+0.16** and wins in
  12 of 12 cells. Held at equal pair count, drawing from the remainder rather
  than from the migrated half is worth **+0.20**, at every budget tested.

So the sample source is the whole design, and it is the opposite of what the
"for free" premise implied: fit on the records **not yet migrated**, because
those are the ones the refitted adapter is about to be applied to. Carrying the
original pairs alongside them makes it *worse* in the case that matters (+0.209
against +0.191 at 8,000 pairs), because they pull the map back toward a domain
that is no longer what is being written.

The guard is what keeps both readings true at once. On an unchanged corpus a
1,000-pair refit loses to a 4,000-pair adapter and is declined; on a drifted one
it wins by an order of magnitude more than the threshold and is adopted.

**This is not the "continuous adaptation" of arXiv:2509.23471 §5.6**, and the
difference is worth stating because the names collide. There, a fixed adapter
degrades from 0.95 to about 0.83 over 24 hours because it maps *queries into the
old space* while the index fills with items "now purely in the f_new space" —
refitting chases a target that is moving underneath it. rebasis serves that same
index with two-space search (:mod:`rebasis.serve.mixed`) instead, which is a
structural answer rather than a moving one. What is left of the scenario once
that is removed is the corpus changing in kind, which is what the numbers above
measure.
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
    """When to consider refitting, and how much to spend finding out."""

    enabled: bool = False
    every_n_records: int = 50_000
    #: Records sampled from the queue and re-embedded per attempt. Distinct from
    #: ``min_pairs``, which is the floor below which the attempt is abandoned: a
    #: store where some records carry no text returns fewer usable pairs than
    #: were asked for, and the two numbers answer different questions.
    sample_size: int = MIN_PAIRS
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
        src: Source-space vectors of the accumulated pairs — for the direction
            `migrate` uses, the old-model vectors the index still holds.
        dst: Target-space vectors of the same records, in the same order.
        policy: When and by how much to switch.
        job_id: For the log and audit record.

    Direction-neutral by construction: it fits ``src -> dst`` and scores the
    result against ``dst``, so it serves whichever direction the caller is
    migrating in. `migrate` passes ``old -> new``; nothing here assumes it.

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
