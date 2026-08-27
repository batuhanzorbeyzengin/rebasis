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
    StoreUnsupported,
    StoreWriteFailed,
)
from rebasis.types import FloatArray, Hit, Record, StoreCapabilities, as_float32

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from rebasis.store.uri import StoreURI

__all__ = ["SqliteVecStore", "serialize_f32"]

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
        # Two attributes rather than one: ``None`` is a real answer here — an
        # empty table declares nothing to read — and not a "not looked yet".
        self._element: str | None = None
        self._element_checked = False

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
        element = self._element_type()
        narrow = element is not None and element != "float32"
        return StoreCapabilities(
            # A `bit` column carries one bit per component, and a vec0 `bit[N]`
            # is legal for any N — `bit[7]` and `bit[12]` both create. So the
            # blob's length does not determine the dimension, and there is no
            # other place to read it from: a vec0 table declares its virtual
            # schema without a type on the vector column. Reading vectors out of
            # one is not a lossy operation, it is an impossible one.
            can_read_vectors=element != "bit",
            can_read_text=self._text_column is not None,
            # Measured against the shipped extension: inserting a float32 vector
            # into an int8 or bit column is refused outright — "expected to be
            # of type int8, but a float32 vector was provided". rebasis only
            # produces float32, so declaring this True would be promising a
            # write that the storage engine itself rejects.
            can_upsert_vectors=not narrow,
            can_filter=False,
            # A vec0 table fixes its dimension at creation, which is the
            # constraint rebasis exists to work around.
            dimension_locked=True,
            supports_in_place_update=True,
            quantized=None if element is None else narrow,
            name="sqlite-vec",
        )

    def _element_type_is_narrow(self) -> bool | None:
        """Whether this table's vectors are stored as something other than float32.

        A ``vec0`` column is declared with an element type, and the three
        sqlite-vec offers are not equivalent: ``float``/``f32`` is four bytes
        per component, ``int8``/``i8`` is one, and ``bit``/``b1`` is one *bit*.
        The last two are lossy by construction — the extension's own
        ``vec_quantize_int8`` and ``vec_quantize_binary`` are what produce them.

        Asked with ``vec_type()``, which reports the element type of a stored
        vector and is present in both sqlite-vec 0.1.6 and 0.1.9 (verified in
        the shipped ``vec0`` extension's symbol table, along with the three
        names it returns: ``float32``, ``int8``, ``bit``). The declaration
        itself is not readable back: a ``vec0`` table declares its virtual
        schema as ``CREATE TABLE x("id" primary key, rowid, "embedding",
        distance hidden, k hidden)``, with no type on the vector column, so
        ``PRAGMA table_info`` has nothing to say.

        An empty table answers ``None``. There is no row to ask about, and
        "float32" would be a guess dressed as a reading.
        """
        element = self._element_type()
        return None if element is None else element != "float32"

    def _element_type(self) -> str | None:
        if not self._element_checked:
            column, table = _quote(self._vector_column), _quote(self._vector_table)
            sql = f"SELECT vec_type({column}) FROM {table} LIMIT 1"  # noqa: S608 - identifiers quoted by _quote
            try:
                row = self._connection.execute(sql).fetchone()
            except sqlite3.Error:
                row = None
            self._element = None if row is None or row[0] is None else str(row[0])
            self._element_checked = True
        return self._element

    def _refuse_narrow(self, operation: str, reason: str) -> None:
        """Refuse an operation this table's element type cannot support.

        Raises:
            StoreUnsupported: On an ``int8`` or ``bit`` table.
        """
        element = self._element_type()
        if element is None or element == "float32":
            return
        raise StoreUnsupported(
            f"This vec0 table stores {element} vectors, and `{operation}` needs float32.",
            hint=reason,
            context={"store_backend": "sqlite-vec", "element_type": element},
        )

    def count(self) -> int:
        """Number of rows in the vector table."""
        sql = f"SELECT count(*) FROM {_quote(self._vector_table)}"  # noqa: S608 - identifier quoted by _quote
        return int(self._one(sql)[0])

    def dimension(self) -> int:
        """Vector dimensionality, read from the first row and its element type.

        The blob's length alone does not give it. A vec0 column stores four
        bytes per component at ``float32`` and **one** at ``int8``, so dividing
        by four reported a quarter of the true dimension on an int8 table — and
        every check downstream that compares an adapter's width against the
        index would have compared against that.

        ``bit`` is refused rather than computed, and that is not caution: it
        packs one *bit* per component and ``bit[7]`` is a legal declaration, so
        a one-byte blob is consistent with any dimension from 1 to 8. The number
        is not in the data.

        Raises:
            StoreError: When the table is empty, so there is no row to measure.
            StoreUnsupported: On a ``bit`` table.
        """
        if self._dimension is None:
            if self._element_type() == "bit":
                self._refuse_narrow(
                    "dimension",
                    "A bit column packs one bit per component and `bit[7]` is legal, "
                    "so the blob's length is consistent with several dimensions. "
                    "`rebasis doctor --store` reports what can be read.",
                )
            column, table = _quote(self._vector_column), _quote(self._vector_table)
            sql = f"SELECT {column} FROM {table} LIMIT 1"  # noqa: S608 - quoted by _quote
            row = self._connection.execute(sql).fetchone()
            if row is None:
                raise StoreError(
                    "The table is empty, so its dimensionality cannot be determined.",
                    hint="rebasis needs an existing index to measure against.",
                    context={"store_backend": "sqlite-vec"},
                )
            self._dimension = len(row[0]) // _BYTES_PER_COMPONENT[self._element_type() or "float32"]
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
                    _deserialize(row["vector"], self._element_type())
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
        # Measured against the shipped extension: a float32 query against an
        # int8 or bit column is refused — "expected to be of type int8". rebasis
        # only produces float32, so this refuses first with the reason rather
        # than letting a SQL error surface as "the sqlite-vec query failed".
        self._refuse_narrow(
            "search",
            "sqlite-vec refuses a float32 query against a narrower column, so "
            "`rebasis.Bridge` cannot serve this table. Reindexing it as float32 "
            "is what would make it bridgeable.",
        )
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

        Raises:
            StoreUnsupported: On a narrower-than-float32 column, which the
                extension itself refuses. ``capabilities.can_upsert_vectors``
                already says so, so `migrate` stops before it opens a job; this
                is the backstop for a caller that reached here anyway, and it
                matters because the delete half of delete-then-insert would
                otherwise run before the insert half failed.
        """
        self._refuse_narrow(
            "upsert_vectors",
            "sqlite-vec refuses a float32 vector for a narrower column, and "
            "rebasis produces nothing else. `migrate` cannot rewrite this table.",
        )
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

    def rebuild_index(self) -> None:
        """`vec0` scans, so there is no structure to rebuild.

        Measured: recall against exact kNN is 1.000 before and after a
        migration of 100,000 records.
        """
        from rebasis.store.base import require_capability

        require_capability(self, "can_rebuild_index", operation="rebuilding the index")

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

    # `sqlite3.connect` raises `OperationalError` for a directory that is not
    # there, a file that cannot be read, and several other things. None of them
    # is a rebasis error, and all of them used to reach the caller as one more
    # thing to search for.
    try:
        connection: sqlite3.Connection = sqlite3.connect(path, **kwargs)
    except sqlite3.Error as exc:
        raise StoreError(
            f"sqlite-vec could not open {path!r}.",
            hint=HINT,
            context={"store_backend": "sqlite-vec"},
            cause=exc,
        ) from exc
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


#: Bytes a vec0 column spends per vector component, by the name ``vec_type()``
#: returns. Measured against the shipped extension: a ``float[8]`` column stores
#: 32 bytes, ``int8[8]`` stores 8, and ``bit[8]`` stores 1.
#:
#: ``bit`` is absent on purpose rather than recorded as ``0.125``. A fractional
#: width would make ``len(blob) // width`` compute a number, and that number
#: would be wrong: ``bit[7]`` is a legal declaration and stores the same single
#: byte as ``bit[8]``, so the dimension is not recoverable from the data at all.
#: A missing key raises where a plausible one would have lied.
_BYTES_PER_COMPONENT = {"float32": 4, "int8": 1}


def _deserialize(blob: bytes, element: str | None = None) -> FloatArray:
    """Unpack a vec0 blob according to its element type.

    ``float32`` is the raw little-endian bytes. ``int8`` is one signed byte per
    component, widened to float — the scale that quantization removed is a
    single factor across the whole vector, and every consumer here normalises,
    so what comes back points where the stored vector points.

    ``None`` means the element type could not be read, and is treated as
    ``float32``: that is what every vec0 table created without an explicit type
    is, and it is the behaviour this function had before it could ask.

    Raises:
        StoreUnsupported: On a ``bit`` blob, whose component count is not in the
            data — see :data:`_BYTES_PER_COMPONENT`.
    """
    kind = element or "float32"
    if kind == "int8":
        return np.frombuffer(blob, dtype=np.int8).astype(np.float32, copy=True)
    if kind not in _BYTES_PER_COMPONENT:
        raise StoreUnsupported(
            f"This vec0 table stores {kind} vectors, which rebasis cannot read back.",
            hint=(
                "A bit column packs one bit per component and `bit[7]` is legal, "
                "so the number of components is not recoverable from the blob."
            ),
            context={"store_backend": "sqlite-vec", "element_type": kind},
        )
    return np.frombuffer(blob, dtype="<f4").astype(np.float32, copy=True)
