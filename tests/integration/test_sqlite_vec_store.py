"""sqlite-vec backend against a real database.

Skipped when the extension cannot be loaded, which is a real condition rather
than a hypothetical one: a Python built against a system SQLite compiled
without extension support cannot load vec0 at all, and the backend says so
rather than failing obscurely later.
"""

from __future__ import annotations

import numpy as np
import pytest

from rebasis.core import l2_normalize
from rebasis.errors import CollectionNotFound, EmbeddingDimensionMismatch

pytestmark = [pytest.mark.integration]

sqlite_vec = pytest.importorskip("sqlite_vec", reason="the sqlite-vec extension is not installed")

DIM = 24
N = 120


@pytest.fixture
def database(tmp_path, rng):  # type: ignore[no-untyped-def]
    """A vec0 table plus the metadata table beside it — sqlite-vec's own layout."""
    import sqlite3

    path = tmp_path / "index.db"
    connection = sqlite3.connect(path)
    try:
        connection.enable_load_extension(True)  # noqa: FBT003 - sqlite3's own signature
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)  # noqa: FBT003 - sqlite3's own signature
    except (AttributeError, sqlite3.Error):
        pytest.skip("this Python's sqlite3 cannot load extensions")

    vectors = l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))
    connection.execute("CREATE TABLE documents (id TEXT NOT NULL, text TEXT)")
    connection.execute(f"CREATE VIRTUAL TABLE vec_documents USING vec0(embedding float[{DIM}])")
    for i, vector in enumerate(vectors):
        connection.execute(
            "INSERT INTO documents(rowid, id, text) VALUES (?, ?, ?)",
            (i + 1, f"doc-{i}", f"text of document {i}"),
        )
        connection.execute(
            "INSERT INTO vec_documents(rowid, embedding) VALUES (?, ?)",
            (i + 1, vector.astype("<f4").tobytes()),
        )
    connection.commit()
    connection.close()

    return {"uri": f"sqlite-vec://{path}#vec_documents", "vectors": vectors, "path": path}


@pytest.fixture
def store(database):  # type: ignore[no-untyped-def]
    from rebasis.store import open_store

    return open_store(database["uri"])


class TestReads:
    def test_it_finds_the_metadata_table_by_convention(self, store) -> None:  # type: ignore[no-untyped-def]
        """`vec_documents` beside `documents` is what sqlite-vec's examples produce."""
        assert store.capabilities.can_read_text
        assert store.count() == N
        assert store.dimension() == DIM

    def test_records_carry_the_users_own_id_not_the_rowid(self, store) -> None:  # type: ignore[no-untyped-def]
        records = list(store.iter_records())

        assert {r.id for r in records} == {f"doc-{i}" for i in range(N)}

    def test_vectors_round_trip_exactly(self, store, database) -> None:  # type: ignore[no-untyped-def]
        """float32 in, the same float32 out — the premise rollback rests on."""
        by_id = {r.id: r.vector for r in store.iter_records()}

        for i in range(N):
            np.testing.assert_array_equal(by_id[f"doc-{i}"], database["vectors"][i])

    def test_iter_records_is_lazy(self, store) -> None:  # type: ignore[no-untyped-def]
        """A materialising read breaks the memory invariant on exactly the
        corpora where it matters."""
        import itertools
        import types

        stream = store.iter_records()

        assert isinstance(stream, types.GeneratorType)
        assert len(list(itertools.islice(stream, 3))) == 3

    def test_selected_ids_come_back(self, store) -> None:  # type: ignore[no-untyped-def]
        wanted = ["doc-5", "doc-40", "doc-99"]

        records = list(store.iter_records(wanted))

        assert sorted(r.id for r in records) == sorted(wanted)

    def test_unknown_ids_raise_rather_than_being_skipped(self, store) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(CollectionNotFound):
            list(store.iter_records(["doc-1", "not-a-real-id"]))

    def test_search_finds_an_exact_match_first(self, store, database) -> None:  # type: ignore[no-untyped-def]
        hits = store.search(database["vectors"][7], k=3)

        assert hits[0].id == "doc-7"

    def test_a_missing_collection_names_what_is_there(self, database) -> None:  # type: ignore[no-untyped-def]
        from rebasis.store import open_store

        with pytest.raises(CollectionNotFound, match="not_a_table") as caught:
            open_store(f"sqlite-vec://{database['path']}#not_a_table")

        # The hint is where the recovery lives: a name that is merely wrong is
        # unhelpful without the list of names that are right.
        assert "vec_documents" in (caught.value.hint or "")


class TestWrites:
    def test_upsert_replaces_the_vector_and_keeps_the_count(self, store) -> None:  # type: ignore[no-untyped-def]
        replacement = l2_normalize(np.ones((1, DIM), dtype=np.float32))

        store.upsert_vectors(["doc-3"], replacement)

        assert store.count() == N
        written = next(r for r in store.iter_records(["doc-3"]))
        np.testing.assert_allclose(written.vector, replacement[0], atol=1e-6)

    def test_upsert_leaves_the_text_alone(self, store) -> None:  # type: ignore[no-untyped-def]
        """rebasis writes vectors and nothing else."""
        store.upsert_vectors(["doc-3"], l2_normalize(np.ones((1, DIM), dtype=np.float32)))

        assert next(r for r in store.iter_records(["doc-3"])).text == "text of document 3"

    def test_a_dimension_mismatch_is_refused(self, store) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(EmbeddingDimensionMismatch):
            store.upsert_vectors(["doc-3"], np.ones((1, DIM + 8), dtype=np.float32))

    def test_the_write_survives_reopening(self, store, database) -> None:  # type: ignore[no-untyped-def]
        """A write that is not committed is a write that did not happen."""
        from rebasis.store import open_store

        replacement = l2_normalize(np.ones((1, DIM), dtype=np.float32))
        store.upsert_vectors(["doc-3"], replacement)

        reopened = open_store(database["uri"])

        written = next(r for r in reopened.iter_records(["doc-3"]))
        np.testing.assert_allclose(written.vector, replacement[0], atol=1e-6)
