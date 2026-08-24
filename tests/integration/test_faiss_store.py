"""FAISS backend against a real index.

FAISS is an index, not a database: it stores vectors and returns row numbers.
Ids and text are the caller's problem, which is why this backend expects a
sidecar beside the index file — and why most of what can go wrong here is about
that pairing rather than about search.

Two behaviours get their own tests because getting them wrong is silent. A
compressed index returns an approximation of what was stored, so measuring an
adapter against it would produce a number that looks like an answer; the backend
refuses at second zero. And a sidecar whose length disagrees with the index is
ignored rather than trusted, because pairing the wrong text with a vector is
worse than having no text.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from rebasis.core import l2_normalize
from rebasis.errors import (
    CollectionNotFound,
    EmbeddingDimensionMismatch,
    StoreUnsupported,
)

pytestmark = [pytest.mark.integration]

faiss = pytest.importorskip("faiss", reason="the faiss extra is not installed")

DIM = 24
N = 120

# Deliberately not `0..N-1`. FAISS addresses an id-mapped index by label, and a
# label that happens to equal its row number hides every place the backend
# confuses the two. Real indexes are built from real document ids, so these
# fixtures are too.
LABELS = np.arange(1000, 1000 + N * 3, 3, dtype=np.int64)


@pytest.fixture
def vectors(rng: np.random.Generator) -> np.ndarray:
    return l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))


@pytest.fixture
def index_path(tmp_path, vectors):  # type: ignore[no-untyped-def]
    """A flat index wrapped in an id map, plus the sidecar the backend expects."""
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(DIM))
    index.add_with_ids(vectors, LABELS)
    path = tmp_path / "vectors.faiss"
    faiss.write_index(index, str(path))

    sidecar = tmp_path / "vectors.faiss.meta.json"
    sidecar.write_text(
        json.dumps(
            {
                "ids": [f"doc-{i}" for i in range(N)],
                "texts": [f"text of document {i}" for i in range(N)],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def store(index_path):  # type: ignore[no-untyped-def]
    from rebasis.store import open_store

    return open_store(f"faiss://{index_path}")


class TestReads:
    def test_it_reports_count_and_dimension(self, store) -> None:  # type: ignore[no-untyped-def]
        assert store.count() == N
        assert store.dimension() == DIM

    def test_the_sidecar_supplies_ids_and_text(self, store) -> None:  # type: ignore[no-untyped-def]
        records = list(store.iter_records())

        assert {r.id for r in records} == {f"doc-{i}" for i in range(N)}
        assert store.capabilities.can_read_text
        assert next(r for r in records if r.id == "doc-4").text == "text of document 4"

    def test_vectors_reconstruct_exactly(self, store, vectors) -> None:  # type: ignore[no-untyped-def]
        """A flat index stores what was given. Anything less makes the whole
        measurement approximate."""
        by_id = {r.id: r.vector for r in store.iter_records()}

        for i in (0, 17, N - 1):
            np.testing.assert_allclose(by_id[f"doc-{i}"], vectors[i], atol=1e-6)

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

        assert sorted(r.id for r in store.iter_records(wanted)) == sorted(wanted)

    def test_unknown_ids_raise_rather_than_being_skipped(self, store) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(CollectionNotFound):
            list(store.iter_records(["doc-1", "not-a-real-id"]))

    def test_search_finds_an_exact_match_first(self, store, vectors) -> None:  # type: ignore[no-untyped-def]
        assert store.search(vectors[7], k=3)[0].id == "doc-7"

    def test_a_missing_index_says_what_the_uri_should_be(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from rebasis.store import open_store

        with pytest.raises(CollectionNotFound, match="No FAISS index"):
            open_store(f"faiss://{tmp_path / 'nope.faiss'}")


class TestTheSidecar:
    def test_without_one_the_faiss_labels_become_ids(self, tmp_path, vectors) -> None:  # type: ignore[no-untyped-def]
        """Still usable for `probe` in vector-only mode; the capability
        declaration says text is unavailable rather than pretending.

        The label is the id, not the row number: it is the only name the index
        itself knows a vector by, and it is usually the caller's document id.
        """
        from rebasis.store import open_store

        index = faiss.IndexIDMap2(faiss.IndexFlatIP(DIM))
        index.add_with_ids(vectors, LABELS)
        path = tmp_path / "bare.faiss"
        faiss.write_index(index, str(path))

        store = open_store(f"faiss://{path}")

        assert store.count() == N
        assert not store.capabilities.can_read_text
        assert [record.id for record in store.iter_records()][:3] == ["1000", "1003", "1006"]

    def test_a_bare_index_still_names_rows_by_number(self, tmp_path, vectors) -> None:  # type: ignore[no-untyped-def]
        """Without an id map a row *is* its address, so row numbers are the
        honest ids — the same code path, with labels that happen to be rows."""
        from rebasis.store import open_store

        index = faiss.IndexFlatIP(DIM)
        index.add(vectors)
        path = tmp_path / "plain.faiss"
        faiss.write_index(index, str(path))

        store = open_store(f"faiss://{path}")

        assert [record.id for record in store.iter_records()][:3] == ["0", "1", "2"]

    def test_a_mismatched_sidecar_is_ignored(self, tmp_path, vectors) -> None:  # type: ignore[no-untyped-def]
        """Pairing the wrong text with a vector produces plausible numbers that
        mean nothing — worse than having no text at all."""
        from rebasis.store import open_store

        index = faiss.IndexIDMap2(faiss.IndexFlatIP(DIM))
        index.add_with_ids(vectors, np.arange(N, dtype=np.int64))
        path = tmp_path / "wrong.faiss"
        faiss.write_index(index, str(path))
        path.with_suffix(".faiss.meta.json").write_text(
            json.dumps({"ids": ["a", "b"], "texts": ["x", "y"]}), encoding="utf-8"
        )

        store = open_store(f"faiss://{path}")

        assert not store.capabilities.can_read_text
        assert next(iter(store.iter_records())).id == "0"


class TestRefusals:
    def test_a_compressed_index_is_refused(self, tmp_path, rng) -> None:  # type: ignore[no-untyped-def]
        """IVFPQ cannot return what was put in. Measuring an adapter against an
        approximation would produce a number that looks like an answer."""
        from rebasis.store import open_store

        training = l2_normalize(rng.standard_normal((2000, DIM)).astype(np.float32))
        quantizer = faiss.IndexFlatL2(DIM)
        index = faiss.IndexIVFPQ(quantizer, DIM, 8, 4, 8)
        index.train(training)
        index.add(training)
        path = tmp_path / "compressed.faiss"
        faiss.write_index(index, str(path))

        with pytest.raises(StoreUnsupported, match="reconstruct"):
            open_store(f"faiss://{path}")

    def test_an_index_without_an_id_map_refuses_to_write(self, tmp_path, vectors) -> None:  # type: ignore[no-untyped-def]
        """Without an id map a row's position is its identity, and removing one
        renumbers every row after it."""
        from rebasis.store import open_store

        index = faiss.IndexFlatIP(DIM)
        index.add(vectors)
        path = tmp_path / "plain.faiss"
        faiss.write_index(index, str(path))
        store = open_store(f"faiss://{path}")

        with pytest.raises(StoreUnsupported, match="cannot replace vectors") as caught:
            store.upsert_vectors(["0"], l2_normalize(np.ones((1, DIM), dtype=np.float32)))

        # The recovery lives in the hint: the message says what went wrong, the
        # hint says to wrap the index in an id map.
        assert "IndexIDMap2" in (caught.value.hint or "")


class TestLabelsAreNotRowNumbers:
    """FAISS speaks in labels; the sidecar is indexed by row.

    Every call the backend makes -- `search`, `reconstruct_batch`, `remove_ids`,
    `add_with_ids` -- takes labels on an id-mapped index. These tests exist
    because using a row number in any of those places is not an error FAISS
    reports: it either raises far from the cause or returns the wrong document.
    """

    def test_reading_a_record_returns_its_own_vector(self, store, vectors) -> None:  # type: ignore[no-untyped-def]
        record = next(iter(store.iter_records(["doc-7"])))

        assert np.allclose(record.vector, vectors[7], atol=1e-6)

    def test_search_maps_labels_back_to_ids(self, store, vectors) -> None:  # type: ignore[no-untyped-def]
        """A row number read as a label would raise or name another document."""
        hits = store.search(vectors[42], k=3)

        assert hits[0].id == "doc-42"

    def test_upsert_writes_to_the_row_it_was_asked_for(self, store, vectors) -> None:  # type: ignore[no-untyped-def]
        """The failing case this guards: labels 1000.. and rows 0.., so writing
        by row either raises on removal or lands on a different document."""
        replacement = l2_normalize(np.full((1, DIM), 0.5, dtype=np.float32))
        store.upsert_vectors(["doc-42"], replacement)

        assert np.allclose(
            next(iter(store.iter_records(["doc-42"]))).vector, replacement[0], atol=1e-6
        )
        assert np.allclose(
            next(iter(store.iter_records(["doc-41"]))).vector, vectors[41], atol=1e-6
        )

    def test_an_index_holding_a_label_twice_is_refused(self, tmp_path, vectors) -> None:  # type: ignore[no-untyped-def]
        """`add_with_ids` appends instead of replacing, so an index topped up
        without removing first has a label naming two vectors. Neither reading
        nor replacing has an answer then, so the backend refuses to open it."""
        from rebasis.errors import StoreUnsupported
        from rebasis.store import open_store

        index = faiss.IndexIDMap2(faiss.IndexFlatIP(DIM))
        index.add_with_ids(vectors, LABELS)
        index.add_with_ids(vectors[:1], LABELS[:1])
        path = tmp_path / "duplicated.faiss"
        faiss.write_index(index, str(path))

        with pytest.raises(StoreUnsupported) as caught:
            open_store(f"faiss://{path}")

        assert "duplicate" in str(caught.value).lower()


class TestWrites:
    def test_upsert_replaces_the_vector(self, store) -> None:  # type: ignore[no-untyped-def]
        replacement = l2_normalize(np.ones((1, DIM), dtype=np.float32))

        store.upsert_vectors(["doc-3"], replacement)

        written = next(r for r in store.iter_records(["doc-3"]))
        np.testing.assert_allclose(written.vector, replacement[0], atol=1e-6)

    def test_upsert_does_not_change_the_count(self, store) -> None:  # type: ignore[no-untyped-def]
        store.upsert_vectors(["doc-3"], l2_normalize(np.ones((1, DIM), dtype=np.float32)))

        assert store.count() == N

    def test_a_dimension_mismatch_is_refused(self, store) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(EmbeddingDimensionMismatch):
            store.upsert_vectors(["doc-3"], np.ones((1, DIM + 8), dtype=np.float32))

    def test_the_write_survives_reopening(self, store, index_path) -> None:  # type: ignore[no-untyped-def]
        """A write that is not persisted is a write that did not happen."""
        from rebasis.store import open_store

        replacement = l2_normalize(np.ones((1, DIM), dtype=np.float32))
        store.upsert_vectors(["doc-3"], replacement)

        reopened = open_store(f"faiss://{index_path}")

        written = next(r for r in reopened.iter_records(["doc-3"]))
        np.testing.assert_allclose(written.vector, replacement[0], atol=1e-6)


class TestWritesReorderTheIndex:
    """FAISS appends the replacement, so the written row ends up last.

    Everything the backend hands back is keyed by row, so a write that does not
    put the sidecar into the new order leaves every id after the replacement
    naming the wrong vector -- and nothing raises.
    """

    def test_every_id_still_names_its_own_vector(self, store, vectors) -> None:  # type: ignore[no-untyped-def]
        replacement = l2_normalize(np.full((1, DIM), 0.25, dtype=np.float32))
        store.upsert_vectors(["doc-3"], replacement)

        by_id = {record.id: record.vector for record in store.iter_records()}

        assert np.allclose(by_id["doc-3"], replacement[0], atol=1e-6)
        for i in (0, 2, 4, 60, N - 1):
            assert np.allclose(by_id[f"doc-{i}"], vectors[i], atol=1e-6), f"doc-{i} moved"

    def test_text_follows_its_vector(self, store) -> None:  # type: ignore[no-untyped-def]
        store.upsert_vectors(["doc-3"], l2_normalize(np.ones((1, DIM), dtype=np.float32)))

        by_id = {record.id: record.text for record in store.iter_records()}

        assert by_id["doc-3"] == "text of document 3"
        assert by_id["doc-119"] == "text of document 119"

    def test_the_new_order_survives_reopening(self, store, index_path, vectors) -> None:  # type: ignore[no-untyped-def]
        """The sidecar on disk has to be reordered too, not just the copy in
        memory -- otherwise the damage only appears in the next process."""
        from rebasis.store import open_store

        store.upsert_vectors(["doc-3"], l2_normalize(np.zeros((1, DIM), dtype=np.float32) + 0.5))
        reopened = open_store(f"faiss://{index_path}")

        by_id = {record.id: record for record in reopened.iter_records()}

        assert by_id["doc-50"].text == "text of document 50"
        assert np.allclose(by_id["doc-50"].vector, vectors[50], atol=1e-6)

    def test_two_writes_in_a_row_stay_consistent(self, store, vectors) -> None:  # type: ignore[no-untyped-def]
        """Each write reorders again; the second one must start from the order
        the first one left behind, not the order the index was built in."""
        store.upsert_vectors(["doc-3"], l2_normalize(np.ones((1, DIM), dtype=np.float32)))
        second = l2_normalize(np.full((1, DIM), -0.3, dtype=np.float32))
        store.upsert_vectors(["doc-77"], second)

        by_id = {record.id: record.vector for record in store.iter_records()}

        assert np.allclose(by_id["doc-77"], second[0], atol=1e-6)
        assert np.allclose(by_id["doc-4"], vectors[4], atol=1e-6)

    def test_search_still_names_the_right_document(self, store, vectors) -> None:  # type: ignore[no-untyped-def]
        store.upsert_vectors(["doc-3"], l2_normalize(np.ones((1, DIM), dtype=np.float32)))

        assert store.search(vectors[90], k=1)[0].id == "doc-90"


class TestCapabilitiesAreTruthful:
    """A backend declares what it can genuinely do.

    Both of FAISS' limitations are structural rather than incidental — no
    sidecar means no text, no id map means no writes — and both have to reach
    `migrate` as a refusal at second zero rather than a failure halfway through.
    """

    def test_an_index_without_an_id_map_declares_it_cannot_write(self, tmp_path, vectors) -> None:  # type: ignore[no-untyped-def]
        from rebasis.store import open_store

        index = faiss.IndexFlatIP(DIM)
        index.add(vectors)
        path = tmp_path / "plain.faiss"
        faiss.write_index(index, str(path))

        assert not open_store(f"faiss://{path}").capabilities.can_upsert_vectors

    def test_an_id_mapped_index_declares_it_can(self, store) -> None:  # type: ignore[no-untyped-def]
        assert store.capabilities.can_upsert_vectors

    def test_migrate_refuses_up_front_on_an_index_it_cannot_write(self, tmp_path, vectors) -> None:  # type: ignore[no-untyped-def]
        """The check that matters: the user is told before anything starts."""
        from rebasis.errors import CapabilityMissing
        from rebasis.store import open_store
        from rebasis.store.base import require_capability

        index = faiss.IndexFlatIP(DIM)
        index.add(vectors)
        path = tmp_path / "plain2.faiss"
        faiss.write_index(index, str(path))

        with pytest.raises(CapabilityMissing, match="can_upsert_vectors"):
            require_capability(
                open_store(f"faiss://{path}"), "can_upsert_vectors", operation="migrate"
            )
