"""The index health check, against every store that really has an index.

The unit tests prove the exact side is exact, on a store whose search is a
matrix multiply. What they cannot prove is that the *other* side works: every
backend's `search` has its own id conventions, its own scoring convention and
its own idea of what happens when you ask for more neighbours than it has. A
check that quietly returned nothing on one of them would report perfect recall
about an index it never queried.

So the property asserted here is not a number. At three hundred records every
one of these backends is effectively exact, and asserting 1.000 would be
asserting a property of the corpus size. What is asserted is that the check
*ran*: it drew probes, it scanned, it queried, and it came back with a real
figure over a real number of documents.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from rebasis.core import l2_normalize
from rebasis.migrate import measure_index_health
from rebasis.store import open_store

pytestmark = [pytest.mark.integration]

DIM = 32
N = 300

#: Every backend with an index of its own. The in-memory store is covered by
#: the unit tests; these are the ones with a client library between rebasis and
#: the answer.
BACKENDS = ("chroma", "faiss", "lancedb", "qdrant", "sqlite-vec")


def closing(store: object) -> None:
    """Release a handle if this backend holds one."""
    close = getattr(store, "close", None)
    if callable(close):
        with contextlib.suppress(Exception):
            close()


@pytest.fixture(params=BACKENDS, ids=lambda n: n)
def live_store(request, tmp_path, rng, make_store):  # type: ignore[no-untyped-def]
    """A clustered corpus in each backend.

    Clustered rather than uniform: the check compares two rankings, and on
    uniform vectors in 32 dimensions the top ten are near-ties whose order is
    decided by float noise. That would make this a test of tie-breaking.
    """
    centers = (rng.standard_normal((10, DIM)) * 3.0).astype(np.float32)
    assignment = rng.integers(0, 10, size=N)
    vectors = l2_normalize(
        centers[assignment] + rng.standard_normal((N, DIM)).astype(np.float32) * 1.2
    )
    ids = [f"doc-{i:04d}" for i in range(N)]
    texts = [f"text of document {i}" for i in range(N)]
    uri = make_store(request.param, tmp_path, ids, vectors, texts)
    return {"uri": uri, "backend": request.param}


def _vectors(store: object) -> dict[str, np.ndarray]:
    return {r.id: r.vector for r in store.iter_records(with_text=False)}  # type: ignore[attr-defined]


class TestItRunsEverywhere:
    def test_the_check_reaches_the_backend_and_comes_back(self, live_store) -> None:  # type: ignore[no-untyped-def]
        """Both sides had to work: the scan produced a reference and the
        store's own `search` produced something to compare it with."""
        store = open_store(live_store["uri"])
        try:
            health = measure_index_health(store, sample=40, k=10, seed=0)
        finally:
            closing(store)

        assert health.n_documents == N
        assert health.n_probes == 40
        assert 0.0 <= health.recall <= 1.0
        assert health.backend == live_store["backend"]

    def test_at_this_size_every_backend_is_effectively_exact(self, live_store) -> None:  # type: ignore[no-untyped-def]
        """A floor, not an equality.

        Three hundred records is below where any of these backends becomes
        meaningfully approximate, so a low number here means the comparison is
        broken — a mismatched id convention, a distance read as a similarity —
        rather than that the index is. The floor is deliberately loose: the
        point is to catch a broken comparison, not to pin a backend's recall.
        """
        store = open_store(live_store["uri"])
        try:
            health = measure_index_health(store, sample=40, k=10, seed=0)
        finally:
            closing(store)

        assert health.recall > 0.9, f"{live_store['backend']} returned {health.recall:.3f}"

    def test_it_reads_the_store_without_writing_to_it(self, live_store) -> None:  # type: ignore[no-untyped-def]
        """A diagnostic that modified the index would be a strange diagnostic.

        The control is a plain `search`, not a second read. On Qdrant's local
        mode the first cosine query normalises the stored vectors in place, so a
        collection read after any search differs from the same collection read
        before one — by about 3e-08, the same order as the shift the migration
        guide records for Chroma at `hnsw:space=cosine`. That is the backend's
        behaviour and not the check's, so the question worth asking is whether
        the check moves anything a bare query would not.
        """
        store = open_store(live_store["uri"])
        try:
            before = _vectors(store)
            store.search(before["doc-0000"], k=5)
            control = _vectors(store)
            measure_index_health(store, sample=20, k=5, seed=0)
            after = _vectors(store)
        finally:
            closing(store)

        assert before.keys() == after.keys()
        for record_id, vector in control.items():
            np.testing.assert_array_equal(after[record_id], vector)
