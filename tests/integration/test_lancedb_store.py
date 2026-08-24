"""LanceDB backend against the real client (K12).

Skipped without the extra. K12 is the concrete failure this addresses: changing
the embedding model in a LanceDB-backed agent memory currently requires a
destructive reset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from rebasis.core import l2_normalize
from rebasis.errors import EmbeddingDimensionMismatch
from rebasis.store import parse_store_uri

if TYPE_CHECKING:
    from pathlib import Path

lancedb = pytest.importorskip("lancedb", reason="lancedb extra not installed")

DIM = 48
N = 600  # above the MIN_SAMPLE floor once a query set is held out


@pytest.fixture
def lance_store(tmp_path: Path, rng: np.random.Generator) -> Any:
    """A populated LanceDB table."""
    from rebasis.store.backends.lancedb import LanceDBStore

    path = tmp_path / "lance"
    connection = lancedb.connect(str(path))
    vectors = l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))
    connection.create_table(
        "documents",
        data=[
            {"id": f"doc-{i}", "vector": vectors[i].tolist(), "text": f"document {i}"}
            for i in range(N)
        ],
    )
    return LanceDBStore.from_uri(parse_store_uri(f"lancedb:///{path}#documents"))


@pytest.mark.integration
def test_lancedb_reports_count_and_dimension(lance_store: Any) -> None:
    """The two facts every probe starts from."""
    assert lance_store.count() == N
    assert lance_store.dimension() == DIM


@pytest.mark.integration
def test_lancedb_detects_its_column_names(lance_store: Any) -> None:
    """Column discovery by convention.

    Guessing `vector` and failing on a table that named it `embedding` would be
    an unnecessary way to lose a user.
    """
    assert lance_store._vector_column == "vector"
    assert lance_store._id_column == "id"


@pytest.mark.integration
def test_lancedb_streams_in_arrow_batches(lance_store: Any) -> None:
    """Arrow batches page naturally, so laziness comes for free."""
    import types

    stream = lance_store.iter_records(batch_size=32)
    assert isinstance(stream, types.GeneratorType)

    records = list(stream)
    assert len(records) == N
    assert records[0].vector is not None
    assert records[0].vector.shape == (DIM,)


@pytest.mark.integration
def test_lancedb_search_finds_an_exact_match(lance_store: Any) -> None:
    """Searching with a record's own vector returns that record first."""
    record = next(lance_store.iter_records())
    hits = lance_store.search(record.vector, k=5)
    assert len(hits) == 5
    assert hits[0].id == record.id


@pytest.mark.integration
def test_lancedb_upsert_replaces_without_adding(lance_store: Any) -> None:
    """Upsert only, never create, never delete."""
    before = lance_store.count()
    replacement = l2_normalize(np.ones((1, DIM), dtype=np.float32))
    lance_store.upsert_vectors(["doc-0"], replacement)
    assert lance_store.count() == before


@pytest.mark.integration
def test_lancedb_dimension_mismatch_is_translated(lance_store: Any) -> None:
    """No lancedb exception crosses the boundary."""
    with pytest.raises(EmbeddingDimensionMismatch):
        lance_store.search(np.zeros(DIM * 3, dtype=np.float32), k=5)


@pytest.mark.integration
def test_lancedb_declares_capabilities_truthfully(lance_store: Any) -> None:
    """Unlike Chroma, LanceDB is not dimension-locked."""
    capabilities = lance_store.capabilities
    assert capabilities.name == "lancedb"
    assert capabilities.can_read_vectors
    assert capabilities.can_upsert_vectors
    assert not capabilities.dimension_locked
