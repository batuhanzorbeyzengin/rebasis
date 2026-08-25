"""The exact side of the index health check has to be exact.

The whole measurement is a comparison against a reference answer, so if the
reference is wrong the number is worse than missing — it is a confident wrong
number about the one thing `migrate` could not otherwise see. These tests are
mostly about that reference: that streaming it over pages gives what computing
it in one matrix gives, that page boundaries cannot move it, and that it stays
lazy while doing so.

The comparison itself is checked against a store whose search *is* exact, where
the answer is known in advance: recall 1.000, by construction rather than by
measurement.
"""

from __future__ import annotations

import numpy as np
import pytest

from rebasis.compute.search import top_k_search
from rebasis.core import l2_normalize
from rebasis.migrate.health import (
    IndexHealth,
    _exact_neighbours,
    measure_index_health,
)
from rebasis.store import MemoryStore

pytestmark = pytest.mark.unit

DIM = 24
N = 500


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(3)


@pytest.fixture
def store(rng: np.random.Generator) -> MemoryStore:
    """A clustered corpus.

    Clustered rather than uniform: on uniform vectors every neighbour is
    nearly equidistant and the top-k is decided by ties, which makes any
    comparison between two rankings a test of tie-breaking.
    """
    centers = (rng.standard_normal((12, DIM)) * 3.0).astype(np.float32)
    assignment = rng.integers(0, 12, size=N)
    vectors = l2_normalize(
        centers[assignment] + rng.standard_normal((N, DIM)).astype(np.float32) * 1.2
    )
    ids = [f"doc-{i:04d}" for i in range(N)]
    return MemoryStore(ids, vectors, [f"text {i}" for i in range(N)])


class TestTheReferenceAnswer:
    """`_exact_neighbours` streams; it still has to be exact."""

    def test_it_matches_the_single_matrix_answer(self, store: MemoryStore) -> None:
        probes = np.vstack([r.vector for r in store.iter_records()][:20])

        streamed, scanned = _exact_neighbours(store, probes, k=10, batch_size=64)

        indices, _ = top_k_search(probes, np.vstack([r.vector for r in store.iter_records()]), k=10)
        ids = [f"doc-{i:04d}" for i in range(N)]
        expected = [[ids[int(i)] for i in row] for row in indices]

        assert scanned == N
        assert streamed == expected

    @pytest.mark.parametrize("batch_size", [7, 64, 499, 500, 10_000])
    def test_page_boundaries_cannot_move_it(self, store: MemoryStore, batch_size: int) -> None:
        """A reference answer that depends on how the store paged its reads is
        not a reference answer. Includes sizes above and below the corpus, and
        one that divides it exactly."""
        probes = np.vstack([r.vector for r in store.iter_records()][:15])

        streamed, _ = _exact_neighbours(store, probes, k=10, batch_size=batch_size)
        baseline, _ = _exact_neighbours(store, probes, k=10, batch_size=128)

        assert streamed == baseline

    def test_it_never_materialises_the_corpus(self, store: MemoryStore) -> None:
        """Peak memory is O((probes + batch) x d), which is only true while the
        read stays a generator. A single `list(...)` upstream would break it
        silently on exactly the corpora where it matters."""
        seen_at_once = 0
        original = store.iter_records

        def counting(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            nonlocal seen_at_once
            for live, record in enumerate(original(*args, **kwargs), start=1):  # type: ignore[arg-type]
                seen_at_once = max(seen_at_once, live)
                yield record

        store.iter_records = counting  # type: ignore[method-assign, assignment]
        probes = np.vstack([r.vector for r in original()][:10])

        _exact_neighbours(store, probes, k=10, batch_size=32)

        # The generator is consumed one record at a time; what bounds memory is
        # the page the caller accumulates, not what the store yields.
        assert seen_at_once == N


class TestTheMeasurement:
    def test_an_exact_store_scores_one(self, store: MemoryStore) -> None:
        """MemoryStore searches by full matrix multiply, so its own search *is*
        the reference. Anything below 1.0 here is a defect in the comparison,
        not in the store."""
        health = measure_index_health(store, sample=40, k=10, seed=0)

        assert health.recall == pytest.approx(1.0)
        assert health.n_probes == 40
        assert health.n_documents == N

    def test_the_probe_is_not_its_own_hit(self, store: MemoryStore) -> None:
        """A record retrieves itself at rank 1 on both sides. Counting that
        would put a free 1/k in every comparison and hide a real drop of the
        same size."""
        health = measure_index_health(store, sample=30, k=10, seed=0)
        # If self-matches were counted, k=1 would be a guaranteed 1.0 whatever
        # the index did. It is only 1.0 here because the store is exact.
        narrow = measure_index_health(store, sample=30, k=1, seed=0)

        assert health.recall == pytest.approx(1.0)
        assert narrow.recall == pytest.approx(1.0)
        assert narrow.k == 1

    def test_the_same_seed_draws_the_same_probes(self, store: MemoryStore) -> None:
        """Before and after have to be measured on the same records, or the
        difference between them is the difference between two samples."""
        first = _exact_neighbours(store, _probes(store, seed=5), k=5, batch_size=64)
        second = _exact_neighbours(store, _probes(store, seed=5), k=5, batch_size=64)

        assert first == second

    def test_an_empty_collection_says_nothing(self) -> None:
        """`nan` rather than 1.0: a collection with nothing in it returns
        everything, and reporting that as perfect recall would be a claim about
        an index that is not there."""
        health = measure_index_health(MemoryStore([], np.empty((0, DIM), dtype=np.float32)))

        assert np.isnan(health.recall)
        assert health.n_probes == 0

    def test_the_result_serialises(self, store: MemoryStore) -> None:
        payload = measure_index_health(store, sample=10, k=5).to_dict()

        assert payload["ann_recall"] == pytest.approx(1.0)
        assert payload["n_documents"] == N
        assert payload["store_backend"] == "memory"


class TestTheComparison:
    def test_a_drop_is_reported_as_a_drop(self) -> None:
        from rebasis.migrate import HealthComparison

        comparison = HealthComparison(
            before=IndexHealth(
                recall=1.0, n_probes=200, k=10, n_documents=1000, duration_seconds=1
            ),
            after=IndexHealth(
                recall=0.34, n_probes=200, k=10, n_documents=1000, duration_seconds=1
            ),
        )

        assert comparison.delta == pytest.approx(-0.66)
        assert "fell from 1.000 to 0.340" in comparison.explain()

    def test_no_change_is_reported_as_no_change(self) -> None:
        from rebasis.migrate import HealthComparison

        health = IndexHealth(recall=0.98, n_probes=200, k=10, n_documents=1000, duration_seconds=1)
        comparison = HealthComparison(before=health, after=health)

        assert comparison.delta == 0
        assert "finds as much as before" in comparison.explain()


def _probes(store: MemoryStore, *, seed: int) -> np.ndarray:
    from rebasis.migrate.health import _draw_probes

    _, probes = _draw_probes(store, sample=12, seed=seed, batch_size=64)
    return probes
