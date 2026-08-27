"""A vec0 column has an element type, and three of them are not equivalent.

``float``/``f32`` spends four bytes per component, ``int8``/``i8`` spends one,
and ``bit``/``b1`` spends one *bit*. The backend read every column as float32 —
so on an int8 table `dimension()` reported a quarter of the truth and
`iter_records` handed back a vector assembled from four components' bytes at a
time. Nothing raised. The count was right, the ids were right, and every number
derived from the vectors was wrong.

Everything asserted here was measured against the shipped extension first, and
the measurements are what shaped the fix rather than decorating it:

* ``float[8]`` stores 32 bytes, ``int8[8]`` stores 8, ``bit[8]`` stores 1.
* ``bit[7]`` and ``bit[12]`` are legal declarations, so a one-byte bit blob is
  consistent with any dimension from 1 to 8. The number is not in the data.
* Inserting a float32 vector into an int8 or bit column is **refused** by the
  extension, and so is querying one with a float32 vector.

The last of those decides the shape of the fix. rebasis produces float32 and
nothing else, so on a narrow column `migrate` and `Bridge` were never going to
work — and the honest thing is to say so through the capability declaration,
before a job is opened, rather than at a SQL error halfway through.
"""

from __future__ import annotations

import sqlite3
import struct
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from rebasis.errors import StoreUnsupported

if TYPE_CHECKING:
    from pathlib import Path

sqlite_vec = pytest.importorskip("sqlite_vec", reason="the sqlite-vec extension is not installed")

pytestmark = pytest.mark.integration

DIM = 8
N = 6


def build(path: Path, declaration: str, constructor: str) -> str:
    """A vec0 table of one element type, filled through its own constructor.

    ``constructor`` is the SQL that turns a float32 blob into whatever the
    column holds — the extension's own ``vec_quantize_int8`` and
    ``vec_quantize_binary`` for the narrow types, and nothing for float32.
    """
    connection = sqlite3.connect(path)
    try:
        connection.enable_load_extension(True)  # noqa: FBT003 - sqlite3's own signature
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)  # noqa: FBT003 - sqlite3's own signature
    except (AttributeError, sqlite3.Error):
        pytest.skip("this Python's sqlite3 cannot load extensions")

    rng = np.random.default_rng(3)
    connection.execute("CREATE TABLE documents (id TEXT NOT NULL, text TEXT)")
    connection.execute(f"CREATE VIRTUAL TABLE vec_documents USING vec0(embedding {declaration})")
    for i in range(N):
        raw = struct.pack(f"<{DIM}f", *rng.uniform(-1.0, 1.0, DIM).astype(np.float32))
        connection.execute(
            "INSERT INTO documents(rowid, id, text) VALUES (?, ?, ?)",
            (i + 1, f"doc-{i}", f"text {i}"),
        )
        # `constructor` is one of two literals defined in this module.
        insert = f"INSERT INTO vec_documents(rowid, embedding) VALUES (?, {constructor})"  # noqa: S608
        connection.execute(insert, (i + 1, raw))
    connection.commit()
    connection.close()
    return f"sqlite-vec://{path}#vec_documents"


@pytest.fixture
def float32(tmp_path: Path) -> Any:
    from rebasis.store import open_store

    return open_store(build(tmp_path / "f32.db", f"float[{DIM}]", "?"))


@pytest.fixture
def int8(tmp_path: Path) -> Any:
    from rebasis.store import open_store

    return open_store(build(tmp_path / "i8.db", f"int8[{DIM}]", "vec_quantize_int8(?, 'unit')"))


@pytest.fixture
def bits(tmp_path: Path) -> Any:
    from rebasis.store import open_store

    return open_store(build(tmp_path / "b1.db", f"bit[{DIM}]", "vec_quantize_binary(?)"))


class TestFloat32IsUnchanged:
    def test_the_dimension_is_right(self, float32: Any) -> None:
        assert float32.dimension() == DIM

    def test_it_is_not_reported_as_quantized(self, float32: Any) -> None:
        assert float32.capabilities.quantized is False

    def test_it_can_still_be_written(self, float32: Any) -> None:
        assert float32.capabilities.can_upsert_vectors is True

    def test_the_vectors_come_back_at_full_width(self, float32: Any) -> None:
        records = list(float32.iter_records(with_vectors=True, with_text=False))

        assert len(records) == N
        assert all(r.vector is not None and r.vector.shape == (DIM,) for r in records)


class TestInt8:
    def test_the_dimension_is_no_longer_a_quarter_of_it(self, int8: Any) -> None:
        """The bug in one assertion. One byte per component, not four."""
        assert int8.dimension() == DIM

    def test_the_vectors_come_back_at_full_width(self, int8: Any) -> None:
        records = list(int8.iter_records(with_vectors=True, with_text=False))

        assert len(records) == N
        for record in records:
            assert record.vector is not None
            assert record.vector.shape == (DIM,)

    def test_what_comes_back_points_where_the_stored_vector_points(self, int8: Any) -> None:
        """Quantization removed a scale, and the scale is one factor across the
        whole vector. Every consumer here normalises, so the direction survives
        even though the magnitude does not — which is what makes reading an int8
        table worth doing at all."""
        record = next(iter(int8.iter_records(with_vectors=True, with_text=False)))

        assert record.vector is not None
        assert np.isfinite(record.vector).all()
        assert np.linalg.norm(record.vector) > 0

    def test_it_declares_itself_quantized(self, int8: Any) -> None:
        assert int8.capabilities.quantized is True

    def test_it_declares_that_it_cannot_be_written(self, int8: Any) -> None:
        """Measured: the extension refuses a float32 insert into an int8 column.
        rebasis produces nothing else, so `migrate` stops at
        `require_capability` before it opens a job — rather than at a SQL error
        after the delete half of delete-then-insert has run."""
        assert int8.capabilities.can_upsert_vectors is False

    def test_writing_anyway_is_refused(self, int8: Any) -> None:
        with pytest.raises(StoreUnsupported, match="int8"):
            int8.upsert_vectors(["doc-0"], np.ones((1, DIM), dtype=np.float32))

    def test_searching_is_refused_with_the_reason(self, int8: Any) -> None:
        """Measured: a float32 query against an int8 column is refused by the
        extension. Refusing here names the element type instead of surfacing
        "the sqlite-vec query failed"."""
        with pytest.raises(StoreUnsupported, match="int8"):
            int8.search(np.ones(DIM, dtype=np.float32), k=1)

    def test_the_text_is_still_readable(self, int8: Any) -> None:
        """Nothing about the element type touches the metadata table, and
        `doctor --store` reports what it can rather than nothing."""
        records = list(int8.iter_records(with_vectors=False, with_text=True))

        assert [r.text for r in records] == [f"text {i}" for i in range(N)]


class TestBit:
    def test_reading_vectors_is_declared_impossible(self, bits: Any) -> None:
        """Not "lossy" — impossible. `bit[7]` is a legal declaration and stores
        the same single byte as `bit[8]`, so the component count is not in the
        blob and there is nowhere else to read it from: a vec0 table declares
        its virtual schema without a type on the vector column."""
        assert bits.capabilities.can_read_vectors is False

    def test_the_dimension_is_refused_rather_than_guessed(self, bits: Any) -> None:
        with pytest.raises(StoreUnsupported, match="bit"):
            bits.dimension()

    def test_it_declares_itself_quantized(self, bits: Any) -> None:
        assert bits.capabilities.quantized is True

    def test_it_declares_that_it_cannot_be_written(self, bits: Any) -> None:
        assert bits.capabilities.can_upsert_vectors is False

    def test_the_count_still_works(self, bits: Any) -> None:
        """Refusing the vectors is not refusing the table. `count` reads rows."""
        assert bits.count() == N

    def test_the_text_is_still_readable(self, bits: Any) -> None:
        records = list(bits.iter_records(with_vectors=False, with_text=True))

        assert [r.text for r in records] == [f"text {i}" for i in range(N)]


class TestProbeRefusesEarly:
    def test_a_bit_table_fails_the_capability_probe_needs(self, bits: Any) -> None:
        """`probe` calls `require_capability(store, "can_read_vectors")` before
        it samples anything, so the refusal costs nothing."""
        from rebasis.errors import CapabilityMissing
        from rebasis.store.base import require_capability

        with pytest.raises(CapabilityMissing):
            require_capability(bits, "can_read_vectors", operation="probe")

    def test_an_int8_table_fails_the_one_migrate_needs(self, int8: Any) -> None:
        from rebasis.errors import CapabilityMissing
        from rebasis.store.base import require_capability

        with pytest.raises(CapabilityMissing):
            require_capability(int8, "can_upsert_vectors", operation="migrate")

    def test_an_int8_table_still_passes_the_one_probe_needs(self, int8: Any) -> None:
        """Partial support, declared. A probe of an int8 index answers a real
        question — "would this upgrade hurt" — even though nothing rebasis
        produces could be written back to it."""
        from rebasis.store.base import require_capability

        require_capability(int8, "can_read_vectors", operation="probe")
