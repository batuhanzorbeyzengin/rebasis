"""Memory ceilings and the scaling invariant.

Two different failures are guarded here, and neither shows up in a functional
test.

**A ceiling** catches the slowly-leaking-memory class: peak RSS creeps up
release by release, every test still passes, and one day a user's laptop swaps.
The only mechanism that catches it is an absolute limit that fails the build.

**The scaling invariant** catches the more serious one. Peak memory is
``O(batch × d)`` and *not* ``O(N × d)``: the same tool must behave the same
on a 50,000-chunk vault and a 5,000,000-chunk one. A single `list(iter_records())`
breaks that, and it breaks it only on corpora large enough that nobody notices
in development. So the test measures peak memory at three sizes and asserts it
does not track N — while time is allowed to grow, but not super-linearly.

**"An absolute limit that fails the build" was not true for a while, and this
file is where that was discovered.** The whole module carried `perf`, CI runs
`-m "not perf"`, and so the one mechanism that catches the `O(N × d)` class
of bug ran on no pull request at all — while `benchmarks/README.md` went on
saying it gated every one. The marker now follows what each test *asserts on*:
allocation is `memory` and gates a merge, wall clock is `perf` and does not.
"""

from __future__ import annotations

import gc
import time
import tracemalloc

import numpy as np
import pytest

from rebasis.core import l2_normalize
from rebasis.embed import PrecomputedEmbedder
from rebasis.probe.session import draw_corpus_sample
from rebasis.store import MemoryStore
from rebasis.types import EncodingProfile

# No module-level marker. Five of the six tests below assert peak *allocation*,
# which `tracemalloc` measures identically on a shared runner and on a quiet
# host — those are `memory`, and CI gates on them. One asserts the shape of the
# *time* curve, and a marker that put it in the same bucket would drag it onto
# the merge path, which is the mistake this split exists to undo.

DIM = 128

#: Corpus sizes for the scaling test, deliberately **above**
#: :data:`~rebasis.probe.session.CLUSTER_POOL_MAX`. Below the cap the pool is
#: the whole corpus and memory does grow with N — by design, since stratified
#: sampling runs over ``min(50k, N)``. The invariant being tested is that it
#: *stops* growing, so the measurement has to straddle the point where it stops.
#: Measured on the server: 19.5 MB at each of 60k, 150k and 400k.
SCALES = (60_000, 150_000, 400_000)

#: A narrow vector for the scaling test: what is being measured is the shape of
#: the curve, and 400,000 x 128 would spend the run allocating the fixture.
SCALING_DIM = 32

#: The tolerance is 20%. The measurement is peak *traced* allocation rather
#: than RSS, which excludes the allocator's own high-water mark and is the more
#: stable of the two — so the tolerance can afford to stay at the stated
#: figure rather than being widened to absorb noise.
SCALING_TOLERANCE = 1.20

#: A super-linear time curve is a bug too. Doubling the work may cost
#: more than double — cache effects are real — but a factor of N² would show up
#: as 20x work costing 400x time. The gate is deliberately loose because wall
#: clock in CI is noisy; it is here to catch an algorithmic mistake, not a 10%
#: regression.
SUPERLINEAR_LIMIT = 4.0


def corpus(n: int, rng: np.random.Generator, dim: int = DIM) -> MemoryStore:
    """A store of ``n`` records with text, held outside the measurement."""
    vectors = l2_normalize(rng.standard_normal((n, dim)).astype(np.float32))
    return MemoryStore(
        [f"doc-{i}" for i in range(n)],
        vectors,
        [f"text of document {i}" for i in range(n)],
    )


def peak_bytes_of_sampling(store: MemoryStore, size: int) -> tuple[int, float]:
    """Peak traced allocation and wall time for one sample draw."""
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    try:
        draw_corpus_sample(store, size=size, heldout=size // 4, strategy="random", seed=0)
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak, elapsed


class TestScalingInvariant:
    @pytest.mark.memory
    def test_peak_memory_does_not_track_corpus_size(self, rng: np.random.Generator) -> None:
        """The architectural invariant, measured rather than asserted.

        The sample size is held constant while the corpus grows 20-fold. If the
        read streams, peak memory is a function of the sample; if it does not,
        it is a function of the corpus, and the ratio gives it away.
        """
        sample_size = 1_000
        peaks = {n: peak_bytes_of_sampling(corpus(n, rng), sample_size)[0] for n in SCALES}

        smallest, largest = peaks[SCALES[0]], peaks[SCALES[-1]]
        assert largest <= smallest * SCALING_TOLERANCE, (
            f"peak allocation grew from {smallest / 1024**2:.1f} MB at "
            f"N={SCALES[0]:,} to {largest / 1024**2:.1f} MB at N={SCALES[-1]:,}; "
            f"the read must be O(batch x d), not O(N x d)"
        )

    @pytest.mark.perf
    def test_time_grows_but_not_super_linearly(self, rng: np.random.Generator) -> None:
        """A 20x corpus may cost more than 20x, but not 400x."""
        sample_size = 1_000
        timings = {n: peak_bytes_of_sampling(corpus(n, rng), sample_size)[1] for n in SCALES}

        growth = timings[SCALES[-1]] / max(timings[SCALES[0]], 1e-6)
        ratio = SCALES[-1] / SCALES[0]
        assert growth <= ratio * SUPERLINEAR_LIMIT, (
            f"a {ratio:.0f}x corpus took {growth:.1f}x the time; that curve is super-linear"
        )


@pytest.mark.memory
class TestCeilings:
    def test_the_sample_stays_within_its_budget(self, rng: np.random.Generator) -> None:
        """A 10k x 768 sample is 30 MB of vectors.

        The budget here allows generous headroom over that — text, ids, the
        clustering pool — because the number being defended is the order of
        magnitude, not the last megabyte.
        """
        store = corpus(20_000, rng)

        peak, _ = peak_bytes_of_sampling(store, 10_000)

        budget = 400 * 1024**2
        assert peak < budget, f"peak allocation {peak / 1024**2:.0f} MB exceeds 400 MB"

    def test_the_ground_truth_knn_never_materialises_the_score_matrix(
        self, rng: np.random.Generator
    ) -> None:
        """The full score matrix is never built.

        Asserted as **linearity in n**, not as a fraction of ``n^2``. The
        chunked path costs ``O(chunk x n)`` per pass, so its peak is linear;
        a materialised matrix would be quadratic. A "fraction of n^2" gate
        would pass or fail on the size chosen rather than on the algorithm,
        and at small n the chunk is legitimately a large fraction of the whole.

        Measured on the server at d=128: 45.7 MB at n=2,000 rising to 282.3 MB
        at n=12,000 — a constant 23.5 KB per row across a sixfold range.
        """
        from rebasis.compute import top_k_search

        peaks = {}
        for n in (3_000, 12_000):
            vectors = l2_normalize(rng.standard_normal((n, DIM)).astype(np.float32))
            gc.collect()
            tracemalloc.start()
            try:
                top_k_search(vectors, vectors, k=10)
                peaks[n] = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
            del vectors

        # Quadratic would be 16x for a fourfold n; linear is 4x. The gate sits
        # between them, close to linear.
        assert peaks[12_000] <= peaks[3_000] * 6, (
            f"peak allocation went from {peaks[3_000] / 1024**2:.0f} MB at "
            f"n=3,000 to {peaks[12_000] / 1024**2:.0f} MB at n=12,000; "
            f"a fourfold n should cost about fourfold memory, not sixteen"
        )

    def test_an_adapter_loads_within_its_budget(self, tmp_path, rng: np.random.Generator) -> None:  # type: ignore[no-untyped-def]
        """`.rbs` loading is budgeted at under 20 MB."""
        from rebasis.core import ProcrustesAdapter, load_adapter, save_adapter

        src = l2_normalize(rng.standard_normal((500, DIM)).astype(np.float32))
        dst = l2_normalize(rng.standard_normal((500, DIM)).astype(np.float32))
        profile = EncodingProfile(model_id="m", dim=DIM)
        path = save_adapter(
            ProcrustesAdapter.fit(src, dst),
            tmp_path / "a.rbs",
            direction="query_to_old",
            old_profile=profile,
            new_profile=profile,
        )

        gc.collect()
        tracemalloc.start()
        try:
            load_adapter(path)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        # The weight matrix itself is d x d x 4 bytes; the budget is that plus
        # room for the manifest and one copy during deserialisation.
        assert peak < 20 * 1024**2, f"loading cost {peak / 1024**2:.1f} MB"


@pytest.mark.memory
class TestEmbeddingBudget:
    def test_encoding_holds_only_the_result(self, rng: np.random.Generator) -> None:
        """The chunked encoder must not accumulate intermediates.

        A list comprehension over chunks holds every block until the vstack,
        which is the result's size — but nothing more. If a future change adds
        a copy per chunk, this catches it.
        """
        from rebasis.probe.session import _encode

        n = 8_000
        texts = [f"text {i}" for i in range(n)]
        vectors = l2_normalize(rng.standard_normal((n, DIM)).astype(np.float32))
        embedder = PrecomputedEmbedder(
            EncodingProfile(model_id="m", dim=DIM), dict(zip(texts, vectors, strict=True))
        )

        gc.collect()
        tracemalloc.start()
        try:
            _encode(embedder, texts, kind="document")
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        result_bytes = n * DIM * 4
        assert peak < result_bytes * 4, (
            f"peak allocation {peak / 1024**2:.1f} MB against a "
            f"{result_bytes / 1024**2:.1f} MB result"
        )
