"""Qdrant backend against a real embedded database.

Local mode, so no server and no container: ``QdrantClient(path=...)`` is a real
Qdrant, and the same class serves a cluster.

The tests that matter here are the two things this backend does that the others
do not: it resolves the user's own id out of the point payload, and it writes
vectors with ``update_vectors`` so the payload survives.
"""

from __future__ import annotations

import numpy as np
import pytest

from rebasis.core import l2_normalize
from rebasis.errors import CollectionNotFound, EmbeddingDimensionMismatch

pytestmark = [pytest.mark.integration]

qdrant_client = pytest.importorskip("qdrant_client", reason="qdrant-client is not installed")

DIM = 24
N = 120


@pytest.fixture
def database(tmp_path, rng):  # type: ignore[no-untyped-def]
    """A collection whose points carry `id` and `text` in the payload."""
    from qdrant_client import QdrantClient, models

    path = tmp_path / "qdrant"
    vectors = l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))

    client = QdrantClient(path=str(path))
    client.create_collection(
        collection_name="documents",
        vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
    )
    client.upsert(
        collection_name="documents",
        points=[
            models.PointStruct(
                id=i + 1,
                vector=vectors[i].tolist(),
                payload={"id": f"doc-{i}", "text": f"text of document {i}"},
            )
            for i in range(N)
        ],
    )
    client.close()

    return {"uri": f"qdrant://{path}#documents", "vectors": vectors, "path": path}


@pytest.fixture
def store(database):  # type: ignore[no-untyped-def]
    from rebasis.store import open_store

    return open_store(database["uri"])


class TestReads:
    def test_it_reports_the_collections_own_dimension(self, store) -> None:  # type: ignore[no-untyped-def]
        """Read from the config, so an empty collection still answers."""
        assert store.dimension() == DIM
        assert store.count() == N

    def test_records_carry_the_payload_id_not_the_point_id(self, store) -> None:  # type: ignore[no-untyped-def]
        """Points are numbered 1..N; the user's ids are `doc-0`..`doc-N`.

        Reporting the point id would make every id rebasis prints, records in
        the audit trail and writes back a different id from the user's own.
        """
        records = list(store.iter_records())

        assert {r.id for r in records} == {f"doc-{i}" for i in range(N)}

    def test_text_comes_from_the_payload(self, store) -> None:  # type: ignore[no-untyped-def]
        assert store.capabilities.can_read_text
        by_id = {r.id: r.text for r in store.iter_records()}
        assert by_id["doc-4"] == "text of document 4"

    def test_iter_records_is_lazy(self, store) -> None:  # type: ignore[no-untyped-def]
        """Scroll is a cursor, and the read must stay one."""
        import itertools
        import types

        stream = store.iter_records()

        assert isinstance(stream, types.GeneratorType)
        assert len(list(itertools.islice(stream, 3))) == 3

    def test_vectors_round_trip(self, store, database) -> None:  # type: ignore[no-untyped-def]
        by_id = {r.id: r.vector for r in store.iter_records()}

        for i in (0, 17, N - 1):
            np.testing.assert_allclose(by_id[f"doc-{i}"], database["vectors"][i], atol=1e-6)

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

        with pytest.raises(CollectionNotFound, match="not_a_collection") as caught:
            open_store(f"qdrant://{database['path']}#not_a_collection")

        assert "documents" in (caught.value.hint or "")


class TestWrites:
    def test_upsert_replaces_the_vector(self, store) -> None:  # type: ignore[no-untyped-def]
        replacement = l2_normalize(np.ones((1, DIM), dtype=np.float32))

        store.upsert_vectors(["doc-3"], replacement)

        written = next(r for r in store.iter_records(["doc-3"]))
        np.testing.assert_allclose(written.vector, replacement[0], atol=1e-6)

    def test_upsert_leaves_the_payload_intact(self, store) -> None:  # type: ignore[no-untyped-def]
        """`update_vectors`, not `upsert` — anything not resent in a full upsert
        would be silently dropped, and the payload is the user's."""
        store.upsert_vectors(["doc-3"], l2_normalize(np.ones((1, DIM), dtype=np.float32)))

        written = next(r for r in store.iter_records(["doc-3"]))
        assert written.id == "doc-3"
        assert written.text == "text of document 3"

    def test_upsert_does_not_change_the_point_count(self, store) -> None:  # type: ignore[no-untyped-def]
        store.upsert_vectors(["doc-3"], l2_normalize(np.ones((1, DIM), dtype=np.float32)))

        assert store.count() == N

    def test_a_dimension_mismatch_is_refused(self, store) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(EmbeddingDimensionMismatch):
            store.upsert_vectors(["doc-3"], np.ones((1, DIM + 8), dtype=np.float32))
