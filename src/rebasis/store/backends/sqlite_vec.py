"""sqlite-vec backend.

Priority three, and the one that fits the project's premise best: zero
dependencies beyond a loadable extension, one file on disk, no server. That is
what "local-first" means in practice, and it is exactly the shape of index that
gets abandoned rather than reindexed when a better model comes out.

Two things about sqlite-vec shape this file.

**The vectors live in a virtual table, the metadata does not.** A ``vec0`` table
holds an integer ``rowid`` and the embedding; the text and the user's own id
live in an ordinary table beside it. So the backend needs to know both, and it
joins them on ``rowid``.

**A ``vec0`` table cannot be updated through a plain UPDATE** in every version.
Deleting and re-inserting the same rowid is the portable form, and it is done
inside a transaction so an interrupted migration cannot leave a row with no
vector at all.
"""

from __future__ import annotations

import contextlib
import sqlite3
from typing import TYPE_CHECKING, Any, Self

import numpy as np

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

__all__ = ["SqliteVecStore", "serialize_f32"]

DEFAULT_BATCH = 1000

#: Tried in order when the URI does not name the metadata table's columns.
_ID_COLUMNS = ("id", "doc_id", "key", "uuid")
_TEXT_COLUMNS = ("text", "content", "document", "chunk", "body")


def serialize_f32(vector: FloatArray) -> bytes:
    """Pack a vector the way sqlite-vec expects: raw little-endian float32.

    sqlite-vec's own Python examples use ``struct.pack``; this is the same
    bytes via numpy, which avoids a per-element loop on the migration hot path.
    """
    return np.ascontiguousarray(as_float32(vector).reshape(-1)).astype("<f4").tobytes()


class SqliteVecStore:
    """A sqlite-vec virtual table plus the metadata table beside it."""

    def __init__(  # noqa: PLR0913 - the table layout is not guessable from fewer
        self,
        connection: sqlite3.Connection,
        *,
        vector_table: str,
        metadata_table: str | None = None,
        id_column: str | None = None,
        text_column: str | None = None,
        vector_column: str = "embedding",
    ) -> None:
        self._connection = connection
        self._vector_table = vector_table
        self._metadata_table = metadata_table
        self._id_column = id_column
        self._text_column = text_column
        self._vector_column = vector_column
        self._dimension: int | None = None

    @classmethod
    def from_uri(cls, uri: StoreURI, **kwargs: Any) -> SqliteVecStore:
        """Open the table a URI points at.

        ``sqlite-vec:///path/to/index.db#vec_documents``

        Options in the URI name the layout when it is not conventional::

            ?metadata_table=documents&id_column=doc_id&text_column=body
        """
        if not uri.collection:
            raise CollectionNotFound(
                "The sqlite-vec URI does not name a table.",
                hint="Append it after a #, e.g. `sqlite-vec:///index.db#vec_documents`.",
                context={"store_backend": "sqlite-vec"},
            )

        connection = _connect(uri.path or ":memory:", **kwargs)
        options = uri.options or {}
        tables = _table_names(connection)
        if uri.collection not in tables:
            raise CollectionNotFound(
                f"{uri.path} has no table named {uri.collection!r}.",
                hint=f"Tables present: {', '.join(tables) if tables else '(none)'}.",
                context={"store_backend": "sqlite-vec"},
            )

        metadata_table = options.get("metadata_table") or _guess_metadata_table(
            connection, uri.collection, tables
        )
        id_column, text_column = _guess_columns(connection, metadata_table, options)
        return cls(
            connection,
            vector_table=uri.collection,
            metadata_table=metadata_table,
            id_column=id_column,
            text_column=text_column,
            vector_column=options.get("vector_column", "embedding"),
        )

    @property
    def capabilities(self) -> StoreCapabilities:
        """Everything except server-side filtering on arbitrary metadata."""
        return StoreCapabilities(
            can_read_vectors=True,
            can_read_text=self._text_column is not None,
            can_upsert_vectors=True,
            can_filter=False,
            # A vec0 table fixes its dimension at creation, which is the
            # constraint rebasis exists to work around.
            dimension_locked=True,
            supports_in_place_update=True,
            name="sqlite-vec",
        )

    def count(self) -> int:
        """Number of rows in the vector table."""
        sql = f"SELECT count(*) FROM {_quote(self._vector_table)}"  # noqa: S608 - identifier quoted by _quote
        return int(self._one(sql)[0])

    def dimension(self) -> int:
        """Vector dimensionality, read from the first row."""
        if self._dimension is None:
            column, table = _quote(self._vector_column), _quote(self._vector_table)
            sql = f"SELECT {column} FROM {table} LIMIT 1"  # noqa: S608 - quoted by _quote
            row = self._connection.execute(sql).fetchone()
            if row is None:
                raise StoreError(
                    "The table is empty, so its dimensionality cannot be determined.",
                    hint="rebasis needs an existing index to measure against.",
                    context={"store_backend": "sqlite-vec"},
                )
            self._dimension = len(row[0]) // 4
        return self._dimension

    def iter_records(
        self,
        ids: Sequence[str] | None = None,
        *,
        with_vectors: bool = True,
        with_text: bool = True,
        batch_size: int = DEFAULT_BATCH,
    ) -> Iterator[Record]:
        """Stream rows a page at a time — lazy by construction."""
        wanted = [str(i) for i in ids] if ids is not None else None
        seen: set[str] = set()

        for row in self._select(
            wanted, with_vectors=with_vectors, with_text=with_text, page=batch_size
        ):
            record_id = str(row["record_id"])
            seen.add(record_id)
            yield Record(
                id=record_id,
                vector=(
                    _deserialize(row["vector"])
                    if with_vectors and row["vector"] is not None
                    else None
                ),
                text=(row["text"] if with_text and "text" in row.keys() else None),  # noqa: SIM118 - sqlite3.Row has no __contains__
            )

        if wanted is not None:
            missing = [i for i in wanted if i not in seen]
            if missing:
                raise CollectionNotFound(
                    f"{len(missing)} requested ids are not in this table.",
                    hint=f"First missing id: {missing[0]!r}.",
                    context={"store_backend": "sqlite-vec", "count": len(missing)},
                )

    def search(self, vector: FloatArray, k: int, where: dict[str, Any] | None = None) -> list[Hit]:
        """Nearest neighbours, via sqlite-vec's KNN syntax."""
        del where  # capabilities.can_filter is False, and says so rather than pretending
        query = as_float32(vector).reshape(-1)
        if query.shape[0] != self.dimension():
            raise EmbeddingDimensionMismatch(
                f"The table is {self.dimension()}-dimensional but the query is "
                f"{query.shape[0]}-dimensional.",
                hint="The adapter's output dimension does not match the index.",
                context={"store_backend": "sqlite-vec", "dim": self.dimension()},
            )

        sql = (
            f"SELECT v.rowid AS rowid, v.distance AS distance "  # noqa: S608 - identifiers are quoted, values bound
            f"FROM {_quote(self._vector_table)} v "
            f"WHERE v.{_quote(self._vector_column)} MATCH ? AND k = ? "
            f"ORDER BY distance"
        )
        try:
            rows = self._connection.execute(sql, (serialize_f32(query), k)).fetchall()
        except sqlite3.Error as exc:
            raise StoreError(
                "The sqlite-vec query failed.",
                hint="Check that the extension is loaded and the table is a vec0 table.",
                context={"store_backend": "sqlite-vec"},
                cause=exc,
            ) from exc

        by_rowid = self._ids_for_rowids([int(row["rowid"]) for row in rows])
        return [
            # sqlite-vec returns a distance; rebasis speaks similarity everywhere.
            Hit(
                id=by_rowid.get(int(row["rowid"]), str(row["rowid"])),
                score=float(1.0 - row["distance"]),
                rank=rank,
            )
            for rank, row in enumerate(rows)
        ]

    def upsert_vectors(self, ids: Sequence[str], vectors: FloatArray) -> None:
        """Replace vectors in place. The only write path rebasis has.

        Delete-then-insert on the same rowid rather than UPDATE: vec0 tables do
        not accept an UPDATE of the embedding column in every released version,
        and the pair inside one transaction is equivalent and portable.
        """
        matrix = as_float32(vectors)
        if matrix.shape[1] != self.dimension():
            raise EmbeddingDimensionMismatch(
                f"The table is {self.dimension()}-dimensional but the vectors are "
                f"{matrix.shape[1]}-dimensional.",
                hint="Refit the adapter against this table's dimension.",
                context={"store_backend": "sqlite-vec", "dim": self.dimension()},
            )

        rowids = self._rowids_for_ids([str(i) for i in ids])
        table, column = _quote(self._vector_table), _quote(self._vector_column)
        try:
            with self._connection:
                for record_id, vector in zip(ids, matrix, strict=True):
                    rowid = rowids[str(record_id)]
                    self._connection.execute(
                        f"DELETE FROM {table} WHERE rowid = ?",  # noqa: S608 - identifier quoted, value bound
                        (rowid,),
                    )
                    self._connection.execute(
                        f"INSERT INTO {table}(rowid, {column}) VALUES (?, ?)",  # noqa: S608 - identifiers quoted, values bound
                        (rowid, serialize_f32(vector)),
                    )
        except (sqlite3.Error, KeyError) as exc:
            raise StoreWriteFailed(
                f"sqlite-vec rejected an update of {len(ids)} records.",
                hint="Check that the database is writable and not held by another process.",
                context={"store_backend": "sqlite-vec", "count": len(ids)},
                cause=exc,
            ) from exc

    def close(self) -> None:
        """Close the database connection. Safe to call more than once."""
        with contextlib.suppress(Exception):
            self._connection.close()

    def __enter__(self) -> Self:
        """Support ``with open_store(...) as store:``."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close on the way out."""
        self.close()

    # ── internals ─────────────────────────────────────────────────────

    def _select(
        self,
        ids: list[str] | None,
        *,
        with_vectors: bool,
        with_text: bool,
        page: int,
    ) -> Iterator[sqlite3.Row]:
        """Page through the join, yielding rows without materialising the table."""
        columns = ["v.rowid AS rowid"]
        columns.append(
            f"v.{_quote(self._vector_column)} AS vector" if with_vectors else "NULL AS vector"
        )
        if self._metadata_table and self._id_column:
            columns.append(f"m.{_quote(self._id_column)} AS record_id")
            if with_text and self._text_column:
                columns.append(f"m.{_quote(self._text_column)} AS text")
            source = (
                f"{_quote(self._vector_table)} v "
                f"JOIN {_quote(self._metadata_table)} m ON m.rowid = v.rowid"
            )
        else:
            # No metadata table: the rowid is the only identity there is.
            columns.append("v.rowid AS record_id")
            source = f"{_quote(self._vector_table)} v"

        clause, parameters = "", []
        if ids is not None:
            target = "m." + _quote(self._id_column) if self._id_column else "v.rowid"
            clause = f" WHERE {target} IN ({', '.join('?' * len(ids))})"
            parameters = list(ids)

        base = f"SELECT {', '.join(columns)} FROM {source}{clause}"  # noqa: S608 - identifiers quoted, values bound
        offset = 0
        while True:
            rows = self._connection.execute(
                f"{base} LIMIT ? OFFSET ?", (*parameters, page, offset)
            ).fetchall()
            if not rows:
                return
            yield from rows
            offset += len(rows)

    def _rowids_for_ids(self, ids: list[str]) -> dict[str, int]:
        if not (self._metadata_table and self._id_column):
            return {i: int(i) for i in ids}
        sql = (
            f"SELECT rowid, {_quote(self._id_column)} AS record_id "  # noqa: S608 - identifier quoted, values bound
            f"FROM {_quote(self._metadata_table)} "
            f"WHERE {_quote(self._id_column)} IN ({', '.join('?' * len(ids))})"
        )
        rows = self._connection.execute(sql, ids).fetchall()
        return {str(row["record_id"]): int(row["rowid"]) for row in rows}

    def _ids_for_rowids(self, rowids: list[int]) -> dict[int, str]:
        if not rowids or not (self._metadata_table and self._id_column):
            return {}
        sql = (
            f"SELECT rowid, {_quote(self._id_column)} AS record_id "  # noqa: S608 - identifier quoted, values bound
            f"FROM {_quote(self._metadata_table)} "
            f"WHERE rowid IN ({', '.join('?' * len(rowids))})"
        )
        rows = self._connection.execute(sql, rowids).fetchall()
        return {int(row["rowid"]): str(row["record_id"]) for row in rows}

    def _one(self, sql: str) -> sqlite3.Row:
        try:
            return self._connection.execute(sql).fetchone()  # type: ignore[no-any-return]
        except sqlite3.Error as exc:
            raise StoreError(
                "The sqlite-vec database could not be read.",
                hint="Check that the file exists and the extension is available.",
                context={"store_backend": "sqlite-vec"},
                cause=exc,
            ) from exc


def _connect(path: str, **kwargs: Any) -> sqlite3.Connection:
    """Open the database and load the vec0 extension."""
    try:
        import sqlite_vec
    except ImportError as exc:
        raise MissingDependency(
            "The sqlite-vec backend needs sqlite-vec.",
            hint='Install it with `pip install "rebasis[sqlite-vec]"`.',
            context={"store_backend": "sqlite-vec"},
            cause=exc,
        ) from exc

    connection: sqlite3.Connection = sqlite3.connect(path, **kwargs)
    connection.row_factory = sqlite3.Row
    try:
        connection.enable_load_extension(True)  # noqa: FBT003 - sqlite3's own signature
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)  # noqa: FBT003 - sqlite3's own signature
    except (AttributeError, sqlite3.Error) as exc:
        raise MissingDependency(
            "This Python's sqlite3 cannot load extensions, so sqlite-vec is unavailable.",
            hint=(
                "Python built against a system SQLite without extension support "
                "cannot load vec0. A pyenv or conda Python usually can; "
                "`rebasis doctor` reports which you have."
            ),
            context={"store_backend": "sqlite-vec"},
            cause=exc,
        ) from exc
    return connection


def _table_names(connection: sqlite3.Connection) -> list[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name"
    ).fetchall()
    return [str(row[0]) for row in rows]


def _guess_metadata_table(
    connection: sqlite3.Connection, vector_table: str, tables: list[str]
) -> str | None:
    """Find the ordinary table that carries ids and text.

    Conventionally the vector table is the metadata table's name with a ``vec_``
    prefix or a ``_vec`` suffix, which is what sqlite-vec's own examples produce.
    """
    stripped = vector_table.removeprefix("vec_").removesuffix("_vec").removesuffix("_vectors")
    for candidate in (stripped, f"{stripped}s", stripped.rstrip("s")):
        if candidate and candidate != vector_table and candidate in tables:
            return candidate
    for candidate in tables:
        if candidate == vector_table:
            continue
        columns = _column_names(connection, candidate)
        if any(c in columns for c in _ID_COLUMNS):
            return candidate
    return None


def _guess_columns(
    connection: sqlite3.Connection, table: str | None, options: dict[str, str]
) -> tuple[str | None, str | None]:
    if table is None:
        return None, None
    columns = _column_names(connection, table)
    id_column = options.get("id_column") or next((c for c in _ID_COLUMNS if c in columns), None)
    text_column = options.get("text_column") or next(
        (c for c in _TEXT_COLUMNS if c in columns), None
    )
    return id_column, text_column


def _column_names(connection: sqlite3.Connection, table: str) -> list[str]:
    try:
        rows = connection.execute(f"PRAGMA table_info({_quote(table)})").fetchall()
    except sqlite3.Error:
        return []
    return [str(row[1]) for row in rows]


def _quote(identifier: str) -> str:
    """Quote an SQL identifier.

    Table and column names come from a URI, so they are user input that lands
    in SQL that cannot be parameterised. Doubling embedded quotes and wrapping
    is the standard escape, and it is applied at every interpolation site.
    """
    return '"' + identifier.replace('"', '""') + '"'


def _deserialize(blob: bytes) -> FloatArray:
    """Unpack sqlite-vec's raw little-endian float32."""
    return np.frombuffer(blob, dtype="<f4").astype(np.float32, copy=True)
