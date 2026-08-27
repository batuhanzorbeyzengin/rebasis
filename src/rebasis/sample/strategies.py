"""Sampling strategies.

Sample size is the single knob that trades measurement cost against confidence,
and M0 measured what it buys:

* The adapter's quality curve **flattens at about 4,000 matched pairs**. Going
  from 4,000 to 24,000 gained +0.001 — nothing. The reference work put saturation
  at 16,000, so the tool needs far fewer new-model embeddings than expected
  (section 10 of ``docs/m0-findings.md``).
* ARR measured on ~1,000 held-out queries carries a bootstrap 95% CI half-width
  of **±0.024**. That is the floor on how precisely any decision threshold can be
  placed, whatever the sample strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from rebasis.errors import InsufficientSamples, LeakageDetected
from rebasis.observability import Events, get_logger

if TYPE_CHECKING:
    from rebasis.types import FloatArray

__all__ = [
    "DEFAULT_FIT_PAIRS",
    "SampleResult",
    "draw_sample",
    "random_sample",
    "stratified_sample",
    "suggested_cluster_count",
]

log = get_logger(__name__)

#: Default matched-pair count for fitting.
#:
#: 4,000 rather than the reference work's implied 5,000–16,000: M0 measured the
#: quality curve flattening at 4,000, with 20,000 further pairs adding +0.001.
DEFAULT_FIT_PAIRS = 4000

#: Minimum sample that still yields a defensible measurement. Below this the
#: confidence interval is wider than the decision bands, so the answer would be
#: a coin flip dressed up as a recommendation.
MIN_SAMPLE = 200

_KMEANS_FIT_CEILING = 50_000


@dataclass(slots=True)
class SampleResult:
    """A drawn sample, plus everything needed to reproduce it."""

    indices: np.ndarray
    strategy: str
    seed: int
    cluster_labels: np.ndarray | None = None
    cluster_sizes: dict[int, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.indices.size)


def suggested_cluster_count(n: int) -> int:
    """Cluster count for stratified sampling: ``clip(√n / 2, 8, 64)``."""
    return int(np.clip(np.sqrt(n) / 2, 8, 64))


def random_sample(n_total: int, size: int, *, seed: int = 0) -> SampleResult:
    """Uniform random sample without replacement.

    The honest baseline. It is what stratification is measured against, and the
    right choice when the corpus is small enough that clustering would produce
    clusters of one.
    """
    _check_size(n_total, size)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=min(size, n_total), replace=False)
    indices.sort()
    return SampleResult(indices=indices, strategy="random", seed=seed)


def stratified_sample(
    vectors: FloatArray,
    size: int,
    *,
    seed: int = 0,
    n_clusters: int | None = None,
    weights: FloatArray | None = None,
) -> SampleResult:
    """Stratified k-means sample — the default.

    MiniBatchKMeans over at most 50,000 vectors, then a proportional draw from
    each cluster with a floor. The floor is what makes this worth doing: a
    cluster holding 0.5% of the corpus still contributes, and that is where
    heterogeneous drift shows up first.

    Args:
        vectors: Existing vectors to cluster.
        size: Sample size.
        seed: Seed — recorded in the audit trail so a decision can be replayed.
        n_clusters: Override for the cluster count.
        weights: Optional per-record weights, typically access frequency, so a
            corpus with a hot subset is sampled where it is actually read.
    """
    from sklearn.cluster import MiniBatchKMeans

    n_total = vectors.shape[0]
    _check_size(n_total, size)
    size = min(size, n_total)
    rng = np.random.default_rng(seed)
    k = n_clusters or suggested_cluster_count(n_total)
    k = max(1, min(k, size, n_total))

    # Clustering the whole corpus would violate the O(batch × d) memory
    # invariant on a large index; the centroids from 50,000 vectors are
    # indistinguishable from the centroids of five million for this purpose.
    fit_indices = (
        rng.choice(n_total, size=_KMEANS_FIT_CEILING, replace=False)
        if n_total > _KMEANS_FIT_CEILING
        else np.arange(n_total)
    )
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=1024)
    kmeans.fit(vectors[fit_indices])
    labels = kmeans.predict(vectors)

    per_cluster = _allocate(labels, size, k)
    chosen: list[np.ndarray] = []
    for cluster, quota in per_cluster.items():
        members = np.flatnonzero(labels == cluster)
        if members.size == 0 or quota == 0:
            continue
        probability = None
        if weights is not None:
            member_weights = np.asarray(weights, dtype=np.float64)[members]
            total = member_weights.sum()
            if total > 0:
                probability = member_weights / total
        chosen.append(
            rng.choice(members, size=min(quota, members.size), replace=False, p=probability)
        )

    indices = np.sort(np.concatenate(chosen)) if chosen else np.array([], dtype=np.int64)
    sizes = {int(c): int((labels == c).sum()) for c in range(k)}

    log.info(
        Events.PROBE_SAMPLE_DRAWN,
        count=int(indices.size),
        strategy="stratified",
        seed=seed,
    )
    return SampleResult(
        indices=indices,
        strategy="stratified",
        seed=seed,
        cluster_labels=labels,
        cluster_sizes=sizes,
    )


def _allocate(labels: np.ndarray, size: int, k: int) -> dict[int, int]:  # noqa: C901 - three allocation passes, each documented below
    """Split a sample budget across clusters proportionally, with a floor.

    Three passes, and each exists for a reason:

    1. **Floor.** Every non-empty cluster gets a minimum. Without it, proportional
       allocation rounds the smallest clusters to zero — and those are precisely
       the ones worth watching, since heterogeneous drift shows up there first
       and ``tail_arr`` is what reports it.
    2. **Largest remainder.** The rest is split in proportion to cluster size.
       Shares are computed against a *fixed* budget: proportioning against a
       shrinking remainder starves later clusters and silently under-fills the
       sample.
    3. **Top-up.** Small clusters cap out at their own size, leaving the budget
       short. A final pass spends what is left wherever there is headroom, so the
       user actually receives the sample size they asked for. Quietly returning
       fewer would widen the confidence interval with no indication why.
    """
    counts = np.array([(labels == c).sum() for c in range(k)], dtype=np.int64)
    live = [c for c in range(k) if counts[c] > 0]
    if not live:
        return {}

    size = min(size, int(counts.sum()))
    floor = max(1, size // (len(live) * 4))
    allocation = {c: min(int(counts[c]), floor) for c in live}

    # Pass 2 — largest remainder against a fixed budget
    budget = size - sum(allocation.values())
    if budget > 0:
        total = int(counts[live].sum())
        exact = {c: budget * int(counts[c]) / total for c in live}
        for cluster in live:
            headroom = int(counts[cluster]) - allocation[cluster]
            allocation[cluster] += max(0, min(int(exact[cluster]), headroom))

        # Fractional parts, largest first
        leftover = size - sum(allocation.values())
        for cluster in sorted(live, key=lambda c: -(exact[c] % 1)):
            if leftover <= 0:
                break
            if int(counts[cluster]) - allocation[cluster] > 0:
                allocation[cluster] += 1
                leftover -= 1

    # Pass 3 — spend whatever the headroom caps left behind
    leftover = size - sum(allocation.values())
    while leftover > 0:
        spendable = [c for c in live if int(counts[c]) - allocation[c] > 0]
        if not spendable:
            break
        for cluster in sorted(spendable, key=lambda c: -(int(counts[c]) - allocation[c])):
            if leftover <= 0:
                break
            allocation[cluster] += 1
            leftover -= 1

    return allocation


def draw_sample(  # noqa: PLR0913 - strategy dispatch needs every input it forwards
    vectors: FloatArray | None,
    n_total: int,
    size: int,
    *,
    strategy: str = "stratified",
    seed: int = 0,
    weights: FloatArray | None = None,
) -> SampleResult:
    """Draw a sample using the named strategy, falling back when it cannot apply."""
    if strategy == "random" or vectors is None:
        return random_sample(n_total, size, seed=seed)
    return stratified_sample(vectors, size, seed=seed, weights=weights)


def split_disjoint(
    sample: SampleResult,
    query_size: int,
    *,
    seed: int = 0,
    weights: FloatArray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a sample into a query set and a fit set that share nothing.

    Disjointness is checked at runtime, not only in tests. If it ever leaks, every
    ARR number the tool has produced becomes meaningless — so this fails hard
    rather than warning.

    ``weights`` draws the **query proxies** with probability proportional to
    them, leaving the sample itself uniform. That is where an access log
    belongs, and it is not where it looks like it belongs: the sample does two
    jobs at once — it is the mini-index every measurement runs against, and it
    is the pool the queries come out of — so weighting the *sample* fills the
    mini-index with frequently-read documents and changes the distractors, which
    is a property of the index rather than of the questions asked of it.

    Measured over 36 cells, weighting the split leaves the estimate about half
    as far from the whole-corpus quantity as weighting the sample does — +0.025
    against +0.051 — and the bootstrap interval stays within 6% of its correct
    width. `docs/access-weighting.md` separates that from the effect a sample
    has anyway: a mini-index is an easier place to retrieve in than the corpus
    it came from, so **any** design sits above the whole-corpus quantity, and
    today's uniform default sits +0.048 above it.

    Args:
        sample: The drawn sample, whose ``indices`` are split.
        query_size: How many records become query proxies.
        seed: Recorded in the audit trail so a decision can be replayed.
        weights: Optional per-record weights aligned to ``sample.indices``,
            typically access frequency.

    Raises:
        LeakageDetected: If the two sets intersect.
        InsufficientSamples: If there is nothing left to fit on.
    """
    rng = np.random.default_rng(seed)
    query_size = min(query_size, max(1, sample.indices.size // 2))
    if weights is None:
        shuffled = rng.permutation(sample.indices)
        query_ids, fit_ids = shuffled[:query_size], shuffled[query_size:]
    else:
        probability = np.asarray(weights, dtype=np.float64)
        mass = float(probability.sum())
        if probability.size != sample.indices.size or mass <= 0:
            # A weight vector that does not line up with the sample, or that is
            # all zeros, describes nothing. Falling back to uniform is right
            # rather than raising: the caller asked for a probe, the weights
            # were an optimisation on top of one, and `probe_store` reports
            # which of the two it actually ran.
            shuffled = rng.permutation(sample.indices)
            query_ids, fit_ids = shuffled[:query_size], shuffled[query_size:]
        else:
            picked = rng.choice(
                sample.indices.size, size=query_size, replace=False, p=probability / mass
            )
            mask = np.zeros(sample.indices.size, dtype=bool)
            mask[picked] = True
            query_ids = sample.indices[mask]
            fit_ids = sample.indices[~mask]

    overlap = np.intersect1d(query_ids, fit_ids)
    if overlap.size:
        raise LeakageDetected(
            f"The query and fit sets overlap on {overlap.size} records.",
            hint="This is a bug in rebasis; every ARR number would be invalid.",
            context={"count": int(overlap.size)},
        )
    if fit_ids.size < MIN_SAMPLE:
        # Distinguish "you asked for too small a sample" from "the corpus itself
        # is too small". Only the first has a flag that fixes it, and telling a
        # user to raise --sample on a 200-document corpus sends them in circles.
        total = int(sample.indices.size)
        hint = (
            f"Increase --sample, or lower --heldout: {total} records were sampled "
            f"and {query_ids.size} went to queries."
            if total > MIN_SAMPLE + query_ids.size
            else (
                f"The collection only has {total} usable records. rebasis needs "
                f"at least {MIN_SAMPLE} to fit on plus a held-out query set; "
                "below that the confidence interval is wider than the decision "
                "bands and the recommendation would be noise."
            )
        )
        raise InsufficientSamples(
            f"Only {fit_ids.size} records remain to fit on after holding out "
            f"{query_ids.size} as queries.",
            hint=hint,
            context={"count": int(fit_ids.size)},
        )
    return np.sort(query_ids), np.sort(fit_ids)


def _check_size(n_total: int, size: int) -> None:
    if n_total < MIN_SAMPLE:
        raise InsufficientSamples(
            f"The collection holds {n_total} records; at least {MIN_SAMPLE} are needed "
            f"for a defensible measurement.",
            hint=(
                "Below this the confidence interval is wider than the decision "
                "bands, so the recommendation would be noise."
            ),
            context={"count": n_total},
        )
    if size < 1:
        raise InsufficientSamples(
            f"A sample size of {size} was requested.",
            hint="Pass a positive --sample.",
            context={"count": size},
        )
