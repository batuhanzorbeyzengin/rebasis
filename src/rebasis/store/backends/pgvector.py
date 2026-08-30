"""pgvector backend.

The store most teams actually run, and the only one where `migrate`'s weakest
guarantee can be handed to the database instead of hand-built above it.

Three things shape this file, in the order they matter.

**A table has no fixed shape.** A Chroma collection does; a Postgres table is
whatever its owner made. Which column holds the vector, which the id, which the
text — that is the user's schema, and it is named in the URI rather than
guessed::

    pgvector://user:pass@host:5432/dbname#public.documents
    pgvector://…/db#public.documents?vector=embedding&id=chunk_id&text=content

Conventional names are *tried*, and when none is found the error lists the
table's columns with their types and says which option supplies the missing one.
Nothing is inferred from a column merely being the only one of its type: that is
how a backend ends up rewriting the wrong column.

**The element type is read, never assumed.** pgvector has four vector types and
they are not equivalent: ``vector`` is float32 and round-trips exactly,
``halfvec`` is float16 and a read-back is a rounding, ``bit`` is a code and
``sparsevec`` is a different shape entirely. The sqlite-vec backend was written
before this was understood there, and an ``int8`` column silently made a quarter
of the reported dimension correct and every number derived from it wrong. So the
type comes from ``pg_type`` through ``format_type``, and
:attr:`~rebasis.types.StoreCapabilities.quantized` follows it.

**A batch is one transaction.** `migrate`'s durability chain is a shadow copy, a
read-back, a fresh-connection check and `rollback`, all correct and all
durability rebuilt at the application layer. Here ``BEGIN … COMMIT`` takes over
the part it can: a batch lands whole or not at all, and the half-written batch
disappears as a state. The shadow copy stays, because a transaction rolls back
one batch and `rollback <job-id>` rolls back a finished job three days later.
Different scopes, and the pre-flight plan says which layer holds which.
"""

from __future__ import annotations

import contextlib
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

__all__ = ["PgvectorStore"]

DEFAULT_BATCH = 1000

#: Column names tried, in order, when the URI does not name one.
#:
#: A list rather than "the only column of that type": a table with two text
#: columns would then be read differently depending on which one somebody added
#: first, and a table with two vector columns would have `migrate` rewrite
#: whichever one `information_schema` happened to return first.
_VECTOR_COLUMNS = ("embedding", "vector", "embeddings", "vec")
_ID_COLUMNS = ("id", "doc_id", "chunk_id", "key", "uuid")
_TEXT_COLUMNS = ("text", "content", "document", "chunk", "body")

#: Every pgvector type, and whether storing a float32 vector in it is lossy.
#:
#: Answered from the column's declared type rather than from a round trip,
#: because a round trip on a *particular* vector can come back exact by luck —
#: 0.5 survives float16 — and the declaration is what governs every other
#: vector. `tests/integration/test_pgvector_types.py` measures the round trip
#: as well, which is how these entries are checked rather than asserted.
_LOSSY: dict[str, bool] = {
    "vector": False,
    "halfvec": True,
    "bit": True,
    "sparsevec": True,
}

#: Types this backend can read a dense float32 vector out of.
#:
#: ``bit`` is out because what comes back is a code and not a reconstruction of
#: what was written — the same reason sqlite-vec refuses its ``bit`` element
#: type. ``sparsevec`` is out because its text form is an index/value map rather
#: than a dense array; nothing here converts one, and a backend that half
#: converted it would be the silent partial support this project refuses.
_READABLE = frozenset({"vector", "halfvec"})

#: Types this backend will write into.
#:
#: rebasis produces float32 and nothing else. Writing it into ``halfvec`` is a
#: rounding the *database* performs and declares, which is a different thing
#: from rebasis quietly narrowing a vector — so it is allowed, and
#: ``quantized=True`` is what tells `migrate` its read-back cannot be exact.
_WRITABLE = frozenset({"vector", "halfvec"})

HINT = (
    "Check the host, the database and the credentials. A server that cannot be "
    "reached (RB-E3000) is a different problem from a table that is not there, "
    "which reports RB-E3003."
)


class PgvectorStore:
    """One table in one PostgreSQL database, with a pgvector column."""

    def __init__(  # noqa: PLR0913 - a table's layout is not guessable from fewer
        self,
        connection: Any,
        *,
        schema: str,
        table: str,
        vector_column: str,
        id_column: str,
        text_column: str | None = None,
        element_type: str = "vector",
    ) -> None:
        self._connection = connection
        self._schema = schema
        self._table = table
        self._vector_column = vector_column
        self._id_column = id_column
        self._text_column = text_column
        self._element = element_type
        self._dimension: int | None = None

    @classmethod
    def from_uri(cls, uri: StoreURI, **kwargs: Any) -> PgvectorStore:
        """Open the table a URI points at.

        ``pgvector://user:pass@host:5432/dbname#schema.table``

        The fragment names the table, optionally schema-qualified; without a
        schema, ``public``. Query options name the columns when the conventional
        ones are not there::

            ?vector=embedding&id=chunk_id&text=content
        """
        if not uri.collection:
            raise CollectionNotFound(
                "The pgvector URI does not name a table.",
                hint=("Append it after a #, e.g. `pgvector://user@host/db#public.documents`."),
                context={"store_backend": "pgvector"},
            )

        schema, _, table = uri.collection.rpartition(".")
        schema = schema or "public"
        connection = _connect(uri, **kwargs)
        columns = _columns(connection, schema, table)
        if not columns:
            _close(connection)
            raise CollectionNotFound(
                f"{schema}.{table} is not a table in this database, or holds no columns.",
                hint=(
                    "Name it as `#schema.table`. `rebasis doctor --store <uri>` "
                    "reports what the connection can see."
                ),
                context={"store_backend": "pgvector"},
            )

        options = uri.options or {}
        try:
            vector_column = _pick(
                options.get("vector"), _VECTOR_COLUMNS, columns, kind="vector", what="vector"
            )
            id_column = _pick(options.get("id"), _ID_COLUMNS, columns, kind="id", what="id")
            text_column = _optional(options.get("text"), _TEXT_COLUMNS, columns)
        except StoreError:
            _close(connection)
            raise

        element = columns[vector_column]
        if element not in _LOSSY:
            _close(connection)
            raise StoreUnsupported(
                f"Column {vector_column!r} is {element!r}, which is not a pgvector type.",
                hint=(
                    "Name the vector column with `?vector=<name>`. This table's "
                    f"columns: {_describe(columns)}."
                ),
                context={"store_backend": "pgvector", "element_type": element},
            )

        return cls(
            connection,
            schema=schema,
            table=table,
            vector_column=vector_column,
            id_column=id_column,
            text_column=text_column,
            element_type=element,
        )

    @property
    def capabilities(self) -> StoreCapabilities:
        """What this table supports, read off its own column type.

        Two of these are firsts for this project and both are earned rather than
        declared. ``can_filter`` is a SQL ``WHERE``, the richest filter of the
        backends here. ``can_rebuild_index`` is ``REINDEX INDEX CONCURRENTLY``,
        which makes pgvector the second backend that can repair the recall an
        in-place vector rewrite costs a graph index (`docs/index-health.md`).
        """
        return StoreCapabilities(
            can_read_vectors=self._element in _READABLE,
            can_read_text=self._text_column is not None,
            can_upsert_vectors=self._element in _WRITABLE,
            can_filter=True,
            # `vector(n)` fixes n in the column's own type. Changing it is DDL,
            # which is the thing rebasis does not do: `migrate` changes vectors,
            # not schemas.
            dimension_locked=True,
            supports_in_place_update=True,
            can_rebuild_index=True,
            quantized=_LOSSY[self._element],
            name="pgvector",
        )

    def count(self) -> int:
        """Number of rows in the table."""
        return int(self._one(f"SELECT count(*) FROM {self._qualified}")[0])  # noqa: S608 - identifiers quoted by _qualified

    def dimension(self) -> int:
        """Vector dimensionality, read from the column's declared type.

        From the type rather than from a row, which is the difference between a
        fact and a sample: ``vector(768)`` says 768 whether the table holds a
        million rows or none. A column declared without a dimension — pgvector
        allows bare ``vector`` — has none to read, and that is an error rather
        than a guess.

        Raises:
            StoreError: When the column carries no dimension.
        """
        if self._dimension is None:
            row = self._one(
                "SELECT a.atttypmod FROM pg_attribute a "
                "JOIN pg_class c ON c.oid = a.attrelid "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = %s AND c.relname = %s AND a.attname = %s",
                (self._schema, self._table, self._vector_column),
            )
            # pgvector stores the dimension in atttypmod directly, and -1 is
            # PostgreSQL's "no modifier" for every type.
            if row is None or int(row[0]) < 0:
                raise StoreError(
                    f"Column {self._vector_column!r} is declared without a dimension.",
                    hint=(
                        "pgvector allows a bare `vector` column, and rebasis "
                        "needs the width to check an adapter against it. "
                        "`ALTER TABLE … ALTER COLUMN … TYPE vector(n)` fixes it."
                    ),
                    context={"store_backend": "pgvector"},
                )
            self._dimension = int(row[0])
        return self._dimension

    def iter_records(
        self,
        ids: Sequence[str] | None = None,
        *,
        with_vectors: bool = True,
        with_text: bool = True,
        batch_size: int = DEFAULT_BATCH,
    ) -> Iterator[Record]:
        """Stream rows through a server-side cursor — lazy by construction.

        A named cursor rather than ``fetchall``: psycopg's default client-side
        cursor reads the whole result set into the client before the first row
        is yielded, which is exactly the ``O(N × d)`` peak the store contract
        forbids. The name makes it a portal on the server, fetched
        ``batch_size`` rows at a time.
        """
        wanted = [str(i) for i in ids] if ids is not None else None
        select = [_quote(self._id_column)]
        if with_vectors:
            select.append(f"{_quote(self._vector_column)}::text")
        if with_text and self._text_column is not None:
            select.append(_quote(self._text_column))

        sql = f"SELECT {', '.join(select)} FROM {self._qualified}"  # noqa: S608 - identifiers quoted
        parameters: tuple[Any, ...] = ()
        if wanted is not None:
            sql += f" WHERE {_quote(self._id_column)}::text = ANY(%s)"
            parameters = (wanted,)

        seen: set[str] = set()
        # A transaction because a server-side cursor is a portal and a portal
        # needs one — "DECLARE CURSOR can only be used in transaction blocks".
        # It opens here and closes with the stream rather than living for the
        # connection's lifetime, which is the whole difference between a read
        # that holds a snapshot while it reads and one that holds it forever.
        with (
            self._connection.transaction(),
            self._connection.cursor(name="rebasis_iter") as cursor,
        ):
            cursor.itersize = batch_size
            cursor.execute(sql, parameters)
            for row in cursor:
                record_id = str(row[0])
                seen.add(record_id)
                offset = 1
                vector = None
                if with_vectors:
                    vector = _parse_vector(row[offset])
                    offset += 1
                text = row[offset] if with_text and self._text_column is not None else None
                yield Record(id=record_id, vector=vector, text=text)

        if wanted is not None:
            missing = [i for i in wanted if i not in seen]
            if missing:
                raise CollectionNotFound(
                    f"{len(missing)} requested ids are not in this table.",
                    hint=f"First missing id: {missing[0]!r}.",
                    context={"store_backend": "pgvector", "count": len(missing)},
                )

    def search(self, vector: FloatArray, k: int, where: dict[str, Any] | None = None) -> list[Hit]:
        """Nearest neighbours by cosine distance.

        ``<=>`` is pgvector's cosine distance, so the similarity rebasis speaks
        everywhere is ``1 - distance``. Cosine rather than the inner product
        because rebasis ℓ2-normalises every vector it produces and cosine is
        then the same ordering with a bounded range — which is what makes the
        returned score comparable with the other five backends'.

        ``where`` is an equality filter over columns, rendered as SQL. That is
        this backend's own capability rather than a shared one: it is the only
        store here that can filter on a column it was not told about in advance.
        """
        # Refused with the reason rather than left to surface as "the query
        # failed": `<=>` is not defined for `bit` or `sparsevec` — pgvector
        # gives those Hamming and Jaccard operators instead — so a float32
        # query against one is not a worse search, it is not a search.
        if self._element not in _READABLE:
            raise StoreUnsupported(
                f"This column is {self._element!r}, which cosine distance is not defined for.",
                hint=(
                    "pgvector scores `bit` by Hamming or Jaccard distance and "
                    "`sparsevec` in its own space; rebasis produces a dense "
                    "float32 query and speaks cosine everywhere. A `vector` or "
                    "`halfvec` column is what `Bridge` can serve."
                ),
                context={"store_backend": "pgvector", "element_type": self._element},
            )
        query = as_float32(vector).reshape(-1)
        if query.shape[0] != self.dimension():
            raise EmbeddingDimensionMismatch(
                f"The table is {self.dimension()}-dimensional but the query is "
                f"{query.shape[0]}-dimensional.",
                hint="The adapter's output dimension does not match the index.",
                context={"store_backend": "pgvector", "dim": self.dimension()},
            )

        clause, parameters = _where(where)
        sql = (
            f"SELECT {_quote(self._id_column)}, "  # noqa: S608 - identifiers quoted, values bound
            f"{_quote(self._vector_column)} <=> %s::{self._element} AS distance "
            f"FROM {self._qualified}{clause} ORDER BY distance LIMIT %s"
        )
        rows = self._all(sql, (_literal(query), *parameters, k))
        return [
            Hit(id=str(row[0]), score=float(1.0 - row[1]), rank=rank)
            for rank, row in enumerate(rows)
        ]

    def upsert_vectors(self, ids: Sequence[str], vectors: FloatArray) -> None:
        """Replace the vectors of existing rows. One batch, one transaction.

        This is the whole reason the backend is worth having. Every other store
        here leaves a partially written batch as a state `migrate` has to detect
        and undo; here the batch either commits or it does not exist. The shadow
        copy is still written, because it answers a different question — a
        `rollback` days after the job finished.

        Raises:
            StoreUnsupported: On a column type rebasis cannot write.
        """
        if self._element not in _WRITABLE:
            raise StoreUnsupported(
                f"This column is {self._element!r}, and rebasis writes float32 vectors.",
                hint=(
                    "A `bit` or `sparsevec` column holds a code rather than the "
                    "vector that produced it, so there is nothing for `migrate` "
                    "to write back into."
                ),
                context={"store_backend": "pgvector", "element_type": self._element},
            )

        matrix = as_float32(vectors)
        if matrix.ndim != 2 or matrix.shape[1] != self.dimension():  # noqa: PLR2004 - a matrix is two-dimensional
            raise EmbeddingDimensionMismatch(
                f"The table is {self.dimension()}-dimensional but the vectors are "
                f"{matrix.shape[-1]}-dimensional.",
                hint="Refit the adapter against this table's dimension.",
                context={"store_backend": "pgvector", "dim": self.dimension()},
            )

        payload = [
            (_literal(row), str(record_id)) for record_id, row in zip(ids, matrix, strict=True)
        ]
        sql = (
            f"UPDATE {self._qualified} SET {_quote(self._vector_column)} = "  # noqa: S608 - identifiers quoted, values bound
            f"%s::{self._element} WHERE {_quote(self._id_column)}::text = %s"
        )
        try:
            with self._connection.transaction(), self._connection.cursor() as cursor:
                cursor.executemany(sql, payload)
        except Exception as exc:
            raise StoreWriteFailed(
                f"PostgreSQL rejected an update of {len(payload)} records.",
                hint=(
                    "The batch was rolled back whole — this backend writes one "
                    "transaction per batch, so no partial write is left behind. "
                    "Check that the role has UPDATE on the table."
                ),
                context={"store_backend": "pgvector", "count": len(payload)},
                cause=exc,
            ) from exc

    def rebuild_index(self) -> None:
        """Rebuild every index on the vector column, concurrently.

        Concurrently because this is a live database: a plain ``REINDEX`` takes
        an ACCESS EXCLUSIVE lock and stops every read of the table for its
        duration, which is not a thing a tool should do to somebody's production
        index without saying so. The concurrent form builds the replacement
        beside the original and swaps, at the cost of more disk and a longer
        run.

        It cannot run inside a transaction block, which is one of the reasons
        the connection is opened in autocommit. Each index is rebuilt on its
        own, so a failure part-way leaves the ones already rebuilt rebuilt —
        the correct outcome, because each is independently valid.

        A table with no index on the vector column has nothing to rebuild and
        this is a no-op. That is not the same as the operation being
        unsupported, and the capability stays ``True``: an exact scan cannot
        lose recall, so there is nothing for the repair to repair.
        """
        found = self._all(_INDEXES_ON_COLUMN, (self._schema, self._table, self._vector_column))
        names = [str(row[0]) for row in found]
        if not names:
            return
        try:
            with self._connection.cursor() as cursor:
                for name in names:
                    target = f"{_quote(self._schema)}.{_quote(name)}"
                    cursor.execute(f"REINDEX INDEX CONCURRENTLY {target}")
        except Exception as exc:
            raise StoreError(
                f"Rebuilding the index on {self._schema}.{self._table} failed.",
                hint=(
                    "REINDEX CONCURRENTLY leaves an invalid index behind when it "
                    "fails; `\\d` on the table names it and `DROP INDEX` removes "
                    "it. The vectors themselves are untouched either way."
                ),
                context={"store_backend": "pgvector"},
                cause=exc,
            ) from exc

    def close(self) -> None:
        """Close the connection. Safe to call more than once."""
        _close(self._connection)

    def __enter__(self) -> Self:
        """Support ``with open_store(...) as store:``."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close on the way out."""
        self.close()

    # ── internals ─────────────────────────────────────────────────────

    @property
    def _qualified(self) -> str:
        return f"{_quote(self._schema)}.{_quote(self._table)}"

    def _one(self, sql: str, parameters: tuple[Any, ...] = ()) -> Any:
        with self._connection.cursor() as cursor:
            cursor.execute(sql, parameters)
            return cursor.fetchone()

    def _all(self, sql: str, parameters: tuple[Any, ...] = ()) -> list[Any]:
        with self._connection.cursor() as cursor:
            cursor.execute(sql, parameters)
            return list(cursor.fetchall())


#: Indexes defined on one column of one table, by name.
_INDEXES_ON_COLUMN = """
SELECT i.relname
FROM pg_index x
JOIN pg_class i ON i.oid = x.indexrelid
JOIN pg_class t ON t.oid = x.indrelid
JOIN pg_namespace n ON n.oid = t.relnamespace
JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(x.indkey)
WHERE n.nspname = %s AND t.relname = %s AND a.attname = %s
ORDER BY i.relname
"""


def _connect(uri: StoreURI, **kwargs: Any) -> Any:
    """Open a psycopg connection from the URI's own parts.

    Rebuilt from the parsed pieces rather than handed the original string:
    `parse_store_uri` has already separated the credentials from everything
    else, and passing the raw URI on would put the password back into whatever
    the client library decides to log.
    """
    try:
        import psycopg
    except ImportError as exc:
        raise MissingDependency(
            "The pgvector backend needs psycopg, which is not installed.",
            hint='Install it with `pip install "rebasis[pgvector]"`.',
            context={"store_backend": "pgvector"},
            cause=exc,
        ) from exc

    database = (uri.path or "").lstrip("/")
    if not database:
        raise CollectionNotFound(
            "The pgvector URI does not name a database.",
            hint="It goes in the path: `pgvector://user@host:5432/dbname#public.documents`.",
            context={"store_backend": "pgvector"},
        )
    settings: dict[str, Any] = {
        "dbname": database,
        "host": uri.host,
        "port": uri.port,
        "user": uri.username,
        "password": uri.password,
        # **Autocommit, and it is not a shortcut.** psycopg's default opens a
        # transaction on the first statement and holds it until somebody
        # commits, and rebasis reads far more than it writes: a `probe` against
        # a live database would sit `idle in transaction` for the length of the
        # run, pinning the vacuum horizon of somebody's production table for a
        # read that needed no transaction at all. Measured here first as a
        # deadlock — the integration suite's next test could not `DROP TABLE`
        # past the lock the previous store's dangling read still held.
        #
        # The two places that genuinely need a transaction open one explicitly:
        # `upsert_vectors`, where the batch is the unit of atomicity, and
        # `iter_records`, where a server-side cursor cannot be declared without
        # one. `REINDEX CONCURRENTLY` needs the opposite — it refuses to run
        # inside a transaction block — and gets it for free here.
        "autocommit": True,
    }
    settings.update(kwargs)
    try:
        return psycopg.connect(**{k: v for k, v in settings.items() if v is not None})
    except Exception as exc:
        raise StoreError(
            f"Could not connect to the PostgreSQL database {database!r}.",
            hint=HINT,
            context={"store_backend": "pgvector"},
            cause=exc,
        ) from exc


def _close(connection: Any) -> None:
    with contextlib.suppress(Exception):
        connection.close()


def _columns(connection: Any, schema: str, table: str) -> dict[str, str]:
    """Every column of a table, mapped to its type name.

    ``format_type`` rather than ``information_schema.columns.data_type``: the
    latter reports ``USER-DEFINED`` for every extension type, which makes
    ``vector`` and ``halfvec`` indistinguishable — and telling those two apart
    is the whole point of asking. The dimension is stripped, so ``vector(768)``
    comes back as ``vector``.
    """
    sql = (
        "SELECT a.attname, format_type(a.atttypid, NULL) "
        "FROM pg_attribute a "
        "JOIN pg_class c ON c.oid = a.attrelid "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = %s AND c.relname = %s AND a.attnum > 0 AND NOT a.attisdropped "
        "ORDER BY a.attnum"
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, (schema, table))
            return {str(name): str(kind) for name, kind in cursor.fetchall()}
    except Exception as exc:
        raise StoreError(
            f"Could not read the columns of {schema}.{table}.",
            hint=HINT,
            context={"store_backend": "pgvector"},
            cause=exc,
        ) from exc


def _pick(
    named: str | None,
    conventional: Sequence[str],
    columns: dict[str, str],
    *,
    kind: str,
    what: str,
) -> str:
    """The column to use, from the URI or from the conventional names.

    Raises:
        CollectionNotFound: When neither is there. The message lists the table's
            own columns and their types, because the user is looking at a schema
            rebasis cannot see and the fastest way to unblock them is to show
            what it *did* see — the same shape `doctor --store` uses.
    """
    if named:
        if named in columns:
            return named
        raise CollectionNotFound(
            f"This table has no column named {named!r}.",
            hint=f"Its columns: {_describe(columns)}.",
            context={"store_backend": "pgvector"},
        )
    for candidate in conventional:
        if candidate in columns:
            return candidate
    raise CollectionNotFound(
        f"Could not find the {what} column, and rebasis will not guess one.",
        hint=(
            f"Name it with `?{kind}=<column>`. Tried {', '.join(conventional)}. "
            f"This table's columns: {_describe(columns)}."
        ),
        context={"store_backend": "pgvector"},
    )


def _optional(
    named: str | None, conventional: Sequence[str], columns: dict[str, str]
) -> str | None:
    """The text column, or ``None``.

    Absent is a legitimate answer here and not a failure: a table with no text
    column is a table `probe` and `migrate` still work against, and
    ``can_read_text=False`` is what says so. A *named* column that is not there
    is still an error — that one is a typo, not a shape.
    """
    if named:
        if named in columns:
            return named
        raise CollectionNotFound(
            f"This table has no column named {named!r}.",
            hint=f"Its columns: {_describe(columns)}.",
            context={"store_backend": "pgvector"},
        )
    return next((candidate for candidate in conventional if candidate in columns), None)


def _describe(columns: dict[str, str]) -> str:
    """``name (type)`` for every column, for an error a user can act on."""
    return ", ".join(f"{name} ({kind})" for name, kind in columns.items())


def _quote(identifier: str) -> str:
    """Quote a SQL identifier, doubling any embedded quote.

    Every identifier in this file reaches SQL through here. Column and table
    names come from a URI the user wrote, and a URI is a string from outside —
    which makes this the boundary where it stops being one.
    """
    return '"' + identifier.replace('"', '""') + '"'


def _literal(vector: FloatArray) -> str:
    """Pgvector's text input form: ``[1,2,3]``.

    Text rather than a binary adapter so the backend works with plain
    ``psycopg`` and does not require the ``pgvector`` Python package to be
    registered on the connection. The cast at the call site (``%s::vector``)
    is what turns it into the column's own type, which is also what makes the
    ``halfvec`` narrowing the *database's* rounding rather than one performed
    here.
    """
    return "[" + ",".join(repr(float(x)) for x in vector.tolist()) + "]"


def _parse_vector(text: Any) -> FloatArray | None:
    """Read pgvector's text output form back into an array."""
    if text is None:
        return None
    return np.fromstring(str(text).strip("[]"), sep=",", dtype=np.float32)


def _where(where: dict[str, Any] | None) -> tuple[str, tuple[Any, ...]]:
    """An equality filter over columns, as a SQL clause and its parameters.

    Identifiers are quoted and values are bound, never interpolated. The keys
    come from a caller's metadata filter, which is the one place in this backend
    where a name arrives at query time rather than at open time.
    """
    if not where:
        return "", ()
    clauses = " AND ".join(f"{_quote(key)} = %s" for key in where)
    return f" WHERE {clauses}", tuple(where.values())
