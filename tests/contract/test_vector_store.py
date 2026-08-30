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
from rebasis.errors import CapabilityMissing, RebasisError
from rebasis.store import MemoryStore, open_store

DIM = 32
N = 300

#: Written out rather than imported from the conftest that builds them:
#: `@pytest.fixture(params=...)` is evaluated at collection, and a conftest is
#: not importable by name — the suite has several, and the module name is not
#: unique. The builders themselves live in `tests/conftest.py`.
LIVE_BACKENDS = ("chroma", "faiss", "lancedb", "pgvector", "qdrant", "sqlite-vec")


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
@pytest.mark.parametrize("backend", LIVE_BACKENDS)
def test_a_store_that_will_not_open_raises_a_rebasis_error(backend: str) -> None:
    """The other half of the boundary rule, and four backends failed it.

    `test_dimension_mismatch_raises_a_rebasis_error` covers a store that opened.
    Nothing covered the store that does not open at all, which is the more common
    thing to get wrong on a first run: a path typed with a missing directory, a
    database owned by another user, a volume that is not mounted yet.

    Measured before this test existed, against `/proc/1/nope` — a path that
    exists as a parent and refuses everything under it. Only FAISS converted.
    Chroma raised `chromadb.errors.InternalError`, LanceDB and Qdrant raised
    `FileNotFoundError`, sqlite-vec raised `sqlite3.OperationalError`. Each
    reached the caller with no `RB-Exxxx` code and no next step — and
    `rebasis doctor --store` on such a path printed the leaked exception right
    next to its own note that a leaked client-library exception is a bug.

    `/proc/1/nope` rather than a `tmp_path` child, because a temporary directory
    is writable and several of these backends will happily create a database
    there rather than failing. What is wanted is a path that cannot be opened,
    not one that is merely empty.
    """
    if backend == "faiss":
        uri = "faiss:///proc/1/nope.faiss"
    elif backend == "sqlite-vec":
        uri = "sqlite-vec:///proc/1/nope.db#vec_documents"
    elif backend == "pgvector":
        # A path means nothing to a client that connects over a socket, so the
        # unreachable thing here is the server. Port 1 is reserved and nothing
        # listens on it, which makes the refusal immediate and the same on every
        # machine — unlike a hostname that does not resolve, where the failure
        # arrives at whatever speed the resolver decides.
        uri = "pgvector://nobody@127.0.0.1:1/nope#public.documents"
    else:
        uri = f"{backend}:///proc/1/nope#documents"

    # Spelled out rather than derived from the backend name. The first version
    # of this test derived it, asked for `chroma` and `qdrant`, and skipped both
    # while they were installed — the exact failure `ci.yml` greps for, and the
    # reason it does: a skipped test reports the same green summary as a passing
    # one. The import names are `chromadb` and `qdrant_client`.
    module = {
        "chroma": "chromadb",
        "faiss": "faiss",
        "lancedb": "lancedb",
        "pgvector": "pg8000",
        "qdrant": "qdrant_client",
        "sqlite-vec": "sqlite_vec",
    }[backend]
    pytest.importorskip(module, reason=f"the {backend} extra is not installed")

    with pytest.raises(RebasisError) as raised:
        open_store(uri)

    assert raised.value.code.startswith("RB-E"), "the error must carry a code"
    assert raised.value.hint, "and a next step"


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
def test_rebuild_index_matches_what_is_declared(store: Any) -> None:
    """Either the backend can rebuild its index or it says so.

    This is the capability that decides whether a migration's damage to the
    search structure is recoverable, so both answers have to be honest: a
    backend that claims it must actually do it, and one that cannot must refuse
    at the moment it is asked rather than appear to succeed and change nothing.
    """
    if store.capabilities.can_rebuild_index:
        store.rebuild_index()
        # Still usable afterwards; the point of the documented mechanism is that
        # the collection keeps serving while the new structure is built.
        assert store.count() >= 0
        return

    with pytest.raises(CapabilityMissing):
        store.rebuild_index()


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
