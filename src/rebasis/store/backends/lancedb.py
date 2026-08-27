"""LanceDB backend.

Priority two, because LanceDB is where agent memories live, and the concrete
failure it addresses is that changing the embedding model there requires a
destructive reset, throwing away everything accumulated so far.

LanceDB is columnar and Arrow-backed, which suits the streaming contract well:
``to_batches()`` pages naturally, so the ``O(batch × d)`` invariant holds
without any special handling.

Vectors are read from whichever column the table actually uses. Guessing
``"vector"`` and failing on a table that named it ``embedding`` would be an
unnecessary way to lose a user.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rebasis.errors import (
    CollectionNotFound,
    EmbeddingDimensionMismatch,
    MissingDependency,
    StoreError,
    StoreWriteFailed,
)
from rebasis.types import FloatArray, Hit, Record, StoreCapabilities, as_float32

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from rebasis.store.uri import StoreURI

__all__ = ["LanceDBStore"]

#: What to check when a store will not open at all. Shared wording because the
#: distinction it draws is the same on every backend: a database that cannot be
#: opened (RB-E3000) is a different problem from one that opened and does not
#: hold the collection you named (RB-E3003), and a user who is told the wrong one
#: looks in the wrong place.
HINT = (
    "Check the path exists and is readable. A store that cannot be opened is a "
    "different problem from a collection that is not in it, which reports RB-E3003."
)

DEFAULT_BATCH = 1000

#: Column names tried in order when the caller does not say. Covers what the
#: common LanceDB tutorials and LangChain's integration produce.
_VECTOR_COLUMNS = ("vector", "embedding", "embeddings", "vec")
_ID_COLUMNS = ("id", "_id", "doc_id", "pk")
_TEXT_COLUMNS = ("text", "content", "document", "page_content")

#: Width of the dtype rebasis reads and writes in. A column narrower than this
#: cannot hold what it is given; a wider one loses nothing.
_FLOAT32_BITS = 32


class LanceDBStore:
    """A LanceDB table."""

    def __init__(
        self,
        table: Any,
        *,
        vector_column: str,
        id_column: str,
        text_column: str | None = None,
    ) -> None:
        self._table = table
        self._vector_column = vector_column
        self._id_column = id_column
        self._text_column = text_column
        self._dimension: int | None = None
        # Two attributes rather than one, because ``None`` is a real answer
        # here — "the schema did not say" — and not a "not looked yet".
        self._quantized: bool | None = None
        self._quantization_checked = False

    @classmethod
    def from_uri(cls, uri: StoreURI, **kwargs: Any) -> LanceDBStore:
        """Open the table a URI points at.

        ``lancedb:///path/to/db#table_name``
        """
        try:
            import lancedb
        except ImportError as exc:
            raise MissingDependency(
                "The LanceDB backend needs lancedb.",
                hint='Install it with `pip install "rebasis[lancedb]"`.',
                context={"store_backend": "lancedb"},
                cause=exc,
            ) from exc

        if not uri.collection:
            raise CollectionNotFound(
                "The LanceDB URI does not name a table.",
                hint="Append it after a #, for example `lancedb:///path/to/db#documents`.",
                context={"store_backend": "lancedb"},
            )

        # Opening the database and finding the table inside it are different
        # failures, and only the second was converted. `lancedb.connect` raises
        # `FileNotFoundError` on a path that is not there — not a rebasis error,
        # and carrying no code for a user to look up.
        try:
            connection = lancedb.connect(uri.path or ".", **kwargs)
        except Exception as exc:
            raise StoreError(
                f"LanceDB could not open {uri.path or '.'!r}.",
                hint=HINT,
                context={"store_backend": "lancedb"},
                cause=exc,
            ) from exc
        try:
            table = connection.open_table(uri.collection)
        except Exception as exc:
            available = _table_names(connection)
            raise CollectionNotFound(
                f"LanceDB has no table named {uri.collection!r}.",
                hint=f"Available: {', '.join(available) if available else '(none)'}.",
                context={"store_backend": "lancedb"},
                cause=exc,
            ) from exc

        columns = list(table.schema.names)
        return cls(
            table,
            vector_column=_pick(columns, _VECTOR_COLUMNS, "vector", uri),
            id_column=_pick(columns, _ID_COLUMNS, "id", uri),
            text_column=next((c for c in _TEXT_COLUMNS if c in columns), None),
        )

    @property
    def capabilities(self) -> StoreCapabilities:
        """LanceDB supports everything rebasis needs."""
        return StoreCapabilities(
            can_read_vectors=True,
            can_read_text=self._text_column is not None,
            can_upsert_vectors=True,
            can_filter=True,
            dimension_locked=False,
            supports_in_place_update=True,
            quantized=self._column_is_narrower_than_float32(),
            name="lancedb",
        )

    def _column_is_narrower_than_float32(self) -> bool | None:
        """Whether the vector *column* stores something narrower than float32.

        The column, deliberately, and not the index — which is where LanceDB's
        quantization actually lives and where it is easiest to look in the wrong
        place. ``IVF_PQ`` and its relatives build a compressed copy in
        index-internal columns (``__pq_code``, ``__sq_code``); the user's vector
        column is untouched, which is why ``bypass_vector_index`` is documented
        as giving "ground truth results … to calculate your recall"
        (`lancedb/query.py`, 0.37.1). This backend reads with a plain
        ``search().select(...).to_arrow()`` and no ``nearest_to``, so it reads
        that column and never the codes. An IVF_PQ index therefore costs a
        LanceDB migration nothing in fidelity, and reporting it here would fire
        a warning about a round trip that is exact.

        What does change the round trip is the column's own Arrow type. LanceDB
        infers ``fixed_size_list<uint8>`` for an all-integer list column in
        ``[0, 255]`` (``lancedb/table.py``, 0.37.1) — binary and ``uint8``
        embeddings are a real and supported thing to store — and float32 written
        into that comes back rounded.

        Read as a bit width rather than by name: ``value_type`` is a pyarrow
        type, pyarrow is LanceDB's dependency and not rebasis', and a width is
        the property being asked about anyway. ``None`` when the schema does not
        answer — an unknown, not a finding.
        """
        if not self._quantization_checked:
            self._quantized = _narrower_than_float32(self._table, self._vector_column)
            self._quantization_checked = True
        return self._quantized

    def count(self) -> int:
        """Number of rows in the table."""
        try:
            return int(self._table.count_rows())
        except Exception as exc:
            raise StoreError(
                "Could not read the row count from LanceDB.",
                hint="Check that the database path is readable.",
                context={"store_backend": "lancedb"},
                cause=exc,
            ) from exc

    def dimension(self) -> int:
        """Vector dimensionality, read from the first row."""
        if self._dimension is None:
            first = self._table.head(1).to_pylist()
            if not first:
                raise StoreError(
                    "The table is empty, so its dimensionality cannot be determined.",
                    hint="rebasis needs an existing index to measure against.",
                    context={"store_backend": "lancedb"},
                )
            self._dimension = len(first[0][self._vector_column])
        return self._dimension

    def iter_records(
        self,
        ids: Sequence[str] | None = None,
        *,
        with_vectors: bool = True,
        with_text: bool = True,
        batch_size: int = DEFAULT_BATCH,
    ) -> Iterator[Record]:
        """Stream rows in Arrow batches — lazy by construction."""
        columns = [self._id_column]
        if with_vectors:
            columns.append(self._vector_column)
        if with_text and self._text_column:
            columns.append(self._text_column)

        query = self._table.search().select(columns).limit(None)
        if ids is not None:
            wanted = {str(i) for i in ids}
            escaped = ", ".join(f"'{i}'" for i in wanted)
            query = query.where(f"{self._id_column} IN ({escaped})")

        seen: set[str] = set()
        for batch in query.to_arrow().to_batches(max_chunksize=batch_size):
            for row in batch.to_pylist():
                record_id = str(row[self._id_column])
                seen.add(record_id)
                yield Record(
                    id=record_id,
                    vector=(
                        as_float32(row[self._vector_column])
                        if with_vectors and row.get(self._vector_column) is not None
                        else None
                    ),
                    text=(row.get(self._text_column) if with_text and self._text_column else None),
                )

        if ids is not None:
            missing = [str(i) for i in ids if str(i) not in seen]
            if missing:
                raise CollectionNotFound(
                    f"{len(missing)} requested ids are not in this table.",
                    hint=f"First missing id: {missing[0]!r}.",
                    context={"store_backend": "lancedb", "count": len(missing)},
                )

    def search(self, vector: FloatArray, k: int, where: dict[str, Any] | None = None) -> list[Hit]:
        """Nearest neighbours."""
        query = as_float32(vector).reshape(-1)
        if query.shape[0] != self.dimension():
            raise EmbeddingDimensionMismatch(
                f"The table is {self.dimension()}-dimensional but the query is "
                f"{query.shape[0]}-dimensional.",
                hint="The adapter's output dimension does not match the index.",
                context={"store_backend": "lancedb", "dim": self.dimension()},
            )
        try:
            search = self._table.search(query.tolist()).limit(k)
            if where:
                search = search.where(" AND ".join(f"{k_} = '{v}'" for k_, v in where.items()))
            rows = search.to_list()
        except Exception as exc:
            raise StoreError(
                "The LanceDB query failed.",
                hint="Check the filter expression, if you passed one.",
                context={"store_backend": "lancedb"},
                cause=exc,
            ) from exc

        return [
            # LanceDB returns `_distance`; rebasis speaks similarity everywhere.
            Hit(
                id=str(row[self._id_column]),
                score=float(1.0 - row.get("_distance", 0.0)),
                rank=rank,
            )
            for rank, row in enumerate(rows)
        ]

    def upsert_vectors(self, ids: Sequence[str], vectors: FloatArray) -> None:
        """Replace vectors in place. The only write path rebasis has."""
        matrix = as_float32(vectors)
        if matrix.shape[1] != self.dimension():
            raise EmbeddingDimensionMismatch(
                f"The table is {self.dimension()}-dimensional but the vectors are "
                f"{matrix.shape[1]}-dimensional.",
                hint="Refit the adapter against this table's dimension.",
                context={"store_backend": "lancedb", "dim": self.dimension()},
            )
        try:
            for record_id, vector in zip(ids, matrix, strict=True):
                self._table.update(
                    where=f"{self._id_column} = '{record_id}'",
                    values={self._vector_column: vector.tolist()},
                )
        except Exception as exc:
            raise StoreWriteFailed(
                f"LanceDB rejected an update of {len(ids)} records.",
                hint="Transient failures are retried automatically; this one was not.",
                context={"store_backend": "lancedb", "count": len(ids)},
                cause=exc,
            ) from exc

    def rebuild_index(self) -> None:
        """Not declared, because rebasis cannot tell what it would be rebuilding.

        LanceDB OSS builds vector indexes only when asked, and searches
        unindexed rows by brute force. `create_index` would therefore either
        rebuild an index the table has or *create* one it never had — and
        creating an index on somebody's table is a change to their data, not a
        repair of ours.
        """
        from rebasis.store.base import require_capability

        require_capability(self, "can_rebuild_index", operation="rebuilding the index")


def _pick(columns: list[str], candidates: tuple[str, ...], kind: str, uri: StoreURI) -> str:
    """Find a column by convention, or say clearly which names were tried."""
    override = (uri.options or {}).get(f"{kind}_column")
    if override:
        if override not in columns:
            raise StoreError(
                f"The table has no column named {override!r}.",
                hint=f"Columns present: {', '.join(columns)}.",
                context={"store_backend": "lancedb"},
            )
        return override
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise StoreError(
        f"Could not identify the {kind} column in this table.",
        hint=(
            f"Tried {', '.join(candidates)}; the table has {', '.join(columns)}. "
            f"Name it explicitly with ?{kind}_column=<name> in the URI."
        ),
        context={"store_backend": "lancedb"},
    )


def _narrower_than_float32(table: Any, column: str) -> bool | None:
    """Whether ``column``'s element type holds fewer than 32 bits per component.

    Everything here is duck-typed off pyarrow objects rather than imported from
    pyarrow, which is LanceDB's dependency and not rebasis'. A vector column is
    a ``fixed_size_list``, so the element type is ``.value_type`` and its width
    is ``.bit_width``; anything that does not answer both leaves the question
    open rather than being resolved to a guess.

    Checked against lancedb 0.13.0 and 0.37.1, whose ``Table.schema`` is a
    pyarrow ``Schema`` in both.
    """
    try:
        value_type = table.schema.field(column).type.value_type
        bits = int(value_type.bit_width)
    except Exception:  # noqa: BLE001 - a schema that does not answer is an unknown, not a failure
        return None
    return bits < _FLOAT32_BITS


def _table_names(connection: Any) -> list[str]:
    try:
        return list(connection.table_names())
    except Exception:  # noqa: BLE001 - only used to enrich an error message
        return []
