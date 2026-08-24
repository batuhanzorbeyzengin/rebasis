"""Store contract.

Every registered backend runs the same suite. That is what makes "adding a store
is three steps" true: write the file, register the entry point, make these pass
— and the reviewer's job becomes reading the file rather than re-deriving what a
store must do.

The two tests that matter most are the ones a backend is most likely to get
wrong quietly: **laziness**, because a materialising ``iter_records`` breaks the
memory invariant only on corpora large enough that nobody notices in
development, and **truthful capabilities**, because a store that claims more than
it can do fails halfway through a migration instead of at second zero.
"""

from __future__ import annotations

import itertools
import types
from typing import Any

import numpy as np
import pytest

from rebasis.core import l2_normalize
from rebasis.errors import RebasisError
from rebasis.store import MemoryStore, open_store

DIM = 32
N = 300

#: Written out rather than imported from the conftest that builds them:
#: `@pytest.fixture(params=...)` is evaluated at collection, and a conftest is
#: not importable by name — the suite has several, and the module name is not
#: unique. The builders themselves live in `tests/conftest.py`.
LIVE_BACKENDS = ("chroma", "faiss", "lancedb", "qdrant", "sqlite-vec")


def _memory_store(rng: np.random.Generator) -> MemoryStore:
    vectors = l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))
    return MemoryStore(
        [f"doc-{i}" for i in range(N)],
        vectors,
        [f"text of document {i}" for i in range(N)],
    )


#: Every backend runnable without an external service. The suite ran against
#: the in-memory store alone for far longer than it should have: the two things
#: it checks hardest -- laziness and truthful capabilities -- are precisely the
#: things a real client library gets wrong and a dict cannot.
STORE_BACKENDS = ("memory", *LIVE_BACKENDS)


@pytest.fixture(params=STORE_BACKENDS, ids=lambda n: n)
def store(  # type: ignore[no-untyped-def]
    request: pytest.FixtureRequest,
    rng: np.random.Generator,
    tmp_path,
    make_store,
    release_store,
):
    """One instance per registered backend, closed on the way out."""
    if request.param == "memory":
        yield _memory_store(rng)
        return

    vectors = l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))
    uri = make_store(
        request.param,
        tmp_path,
        [f"doc-{i}" for i in range(N)],
        vectors,
        [f"text of document {i}" for i in range(N)],
    )
    opened = open_store(uri)
    try:
        yield opened
    finally:
        release_store(opened)


@pytest.mark.contract
def test_count_and_dimension_agree_with_the_data(store: Any) -> None:
    """The basics every backend must report correctly."""
    assert store.count() == N
    assert store.dimension() == DIM


@pytest.mark.contract
def test_iter_records_is_lazy(store: Any) -> None:
    """Streaming is the default, and it is not optional.

    A backend that returns a list passes every other test here and then breaks
    the memory invariant on a corpus large enough that nobody notices until a
    user reports it.
    """
    result = store.iter_records()
    assert isinstance(result, types.GeneratorType), (
        f"{type(store).__name__}.iter_records must be lazy, not materialise the collection"
    )


@pytest.mark.contract
def test_iter_records_is_complete_and_unique(store: Any) -> None:
    """Every record exactly once — no gaps, no duplicates."""
    ids = [record.id for record in store.iter_records(with_vectors=False, with_text=False)]
    assert len(ids) == N
    assert len(set(ids)) == N


@pytest.mark.contract
def test_iter_records_honours_its_projection_flags(store: Any) -> None:
    """Asking for no vectors must actually skip them.

    On a large corpus this is the difference between reading ids and reading
    gigabytes.
    """
    record = next(store.iter_records(with_vectors=False, with_text=False))
    assert record.vector is None
    assert record.text is None

    record = next(store.iter_records(with_vectors=True))
    assert record.vector is not None
    assert record.vector.shape == (DIM,)


@pytest.mark.contract
def test_search_respects_k(store: Any) -> None:
    """Exactly k results, ranked, in descending score order."""
    query = next(store.iter_records()).vector
    hits = store.search(query, k=5)
    assert len(hits) == 5
    assert [h.rank for h in hits] == [0, 1, 2, 3, 4]
    assert all(a.score >= b.score for a, b in itertools.pairwise(hits))


@pytest.mark.contract
def test_search_finds_an_exact_match_first(store: Any) -> None:
    """Searching with a record's own vector must return that record."""
    record = next(store.iter_records())
    assert store.search(record.vector, k=3)[0].id == record.id


@pytest.mark.contract
def test_dimension_mismatch_raises_a_rebasis_error(store: Any) -> None:
    """No third-party exception crosses the module boundary.

    The user must get RB-Exxxx with a next step, not whatever the client library
    happened to raise.
    """
    with pytest.raises(RebasisError):
        store.search(np.zeros(DIM * 3, dtype=np.float32), k=5)


@pytest.mark.contract
def test_capabilities_are_truthful(store: Any) -> None:
    """A declared capability must actually work.

    Silent partial support is worse than no support, because it fails in the
    middle of a job rather than before it starts.
    """
    capabilities = store.capabilities
    assert capabilities.name

    if capabilities.can_read_vectors:
        assert next(store.iter_records(with_vectors=True)).vector is not None
    if capabilities.can_read_text:
        assert next(store.iter_records(with_text=True)).text is not None
    if capabilities.can_upsert_vectors:
        ids = ["doc-0", "doc-1"]
        replacement = l2_normalize(np.ones((2, DIM), dtype=np.float32))
        store.upsert_vectors(ids, replacement)
        written = {r.id: r.vector for r in store.iter_records(ids)}
        assert np.allclose(written["doc-0"], replacement[0])


@pytest.mark.contract
def test_upsert_does_not_change_the_record_count(store: Any) -> None:
    """The only write path upserts. It never creates and never deletes."""
    if not store.capabilities.can_upsert_vectors:
        pytest.skip("backend does not declare upsert")
    before = store.count()
    store.upsert_vectors(["doc-5"], l2_normalize(np.ones((1, DIM), dtype=np.float32)))
    assert store.count() == before


@pytest.mark.contract
def test_unknown_ids_raise_rather_than_being_skipped(store: Any) -> None:
    """Silently ignoring a missing id would corrupt a migration invisibly."""
    with pytest.raises(RebasisError):
        list(store.iter_records(["doc-0", "no-such-document"]))
