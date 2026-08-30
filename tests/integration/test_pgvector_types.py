"""What each pgvector column type does to a vector, measured rather than argued.

pgvector has four vector types and rebasis' behaviour has to differ per type.
The failure this exists to prevent already happened once on another backend: a
``sqlite-vec`` ``int8`` column was read as if it held float32, and every number
derived from the dimension it reported was a quarter of the truth. Nothing
raised. The rule since is that the element type is read from the schema and the
consequences are measured, not reasoned about.

Four claims, one test each:

| type | what it is | `quantized` | writable |
|---|---|---|---|
| `vector` | float32 | `False` | yes |
| `halfvec` | float16 | `True` | yes |
| `bit` | a code | `True` | no |
| `sparsevec` | a different shape | `True` | no |

The first two are round-tripped against ``VERIFY_ATOL``, which is the tolerance
`migrate`'s read-back check uses — so what these measure is not "is float16
lossy" in the abstract but whether a migration into such a column would stop on
its own first batch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from rebasis.core import l2_normalize
from rebasis.errors import StoreUnsupported
from rebasis.migrate.engine import VERIFY_ATOL
from rebasis.store import open_store

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.integration]

DIM = 32
N = 128


@pytest.fixture
def vectors(rng: np.random.Generator) -> Any:
    return l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))


@pytest.fixture
def typed_table(  # type: ignore[no-untyped-def]
    tmp_path: Path, vectors: Any, pgvector_dsn: str, pgvector_schema: str
):
    """Build one table per pgvector type and hand back its URI.

    The vectors are written through pgvector's own text form and cast to the
    column's type by the server, so whatever narrowing happens is the
    **database's** — which is the thing being measured. Casting in Python first
    would measure numpy.
    """
    import psycopg

    dsn, schema = pgvector_dsn, pgvector_schema
    built: list[str] = []

    def build(kind: str) -> str:
        table = f"t_{tmp_path.name}_{kind}".lower().replace("-", "_")[:63]
        literals = ["[" + ",".join(repr(float(x)) for x in row.tolist()) + "]" for row in vectors]
        if kind == "bit":
            # `bit` holds a bit string rather than a vector literal, so the
            # binary quantization pgvector's own docs use is what fills it:
            # one bit per component, set where the component is positive.
            literals = ["".join("1" if x > 0 else "0" for x in row.tolist()) for row in vectors]
        if kind == "sparsevec":
            literals = [
                "{"
                + ",".join(f"{i + 1}:{float(x)!r}" for i, x in enumerate(row.tolist()))
                + "}"
                + f"/{DIM}"
                for row in vectors
            ]
        with psycopg.connect(dsn, autocommit=True) as connection, connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            cursor.execute(f'DROP TABLE IF EXISTS {schema}."{table}"')
            cursor.execute(
                f'CREATE TABLE {schema}."{table}" '
                f"(id text PRIMARY KEY, text text, embedding {kind}({DIM}))"
            )
            cursor.executemany(
                f'INSERT INTO {schema}."{table}" (id, text, embedding) '  # noqa: S608 - identifier from pytest's own tmp_path
                # The width is part of the cast: `'1010'::bit` is bit(1) and
                # truncates, which fails against a bit(32) column with a length
                # mismatch rather than storing three quarters of a code.
                f"VALUES (%s, %s, %s::{kind}({DIM}))",
                [(f"doc-{i}", f"text {i}", literal) for i, literal in enumerate(literals)],
            )
        _, _, rest = dsn.partition("://")
        built.append(table)
        return f"pgvector://{rest}#{schema}.{table}"

    yield build

    with psycopg.connect(dsn, autocommit=True) as connection:
        for table in built:
            connection.execute(f'DROP TABLE IF EXISTS {schema}."{table}"')


def _read_back(uri: str) -> tuple[Any, dict[str, Any]]:
    store = open_store(uri)
    return store, {r.id: r.vector for r in store.iter_records()}


def _worst(read_back: dict[str, Any], expected: Any) -> float:
    """The largest single-component deviation over the whole collection."""
    return max(
        float(np.abs(read_back[f"doc-{i}"] - expected[i]).max()) for i in range(expected.shape[0])
    )


class TestTheTypeIsReadFromTheSchema:
    """`information_schema` reports every extension type as USER-DEFINED.

    Which is why the backend asks `format_type` instead: `vector` and `halfvec`
    are indistinguishable through the standard view, and telling them apart is
    the entire point of asking.
    """

    @pytest.mark.parametrize(
        ("kind", "quantized"),
        [("vector", False), ("halfvec", True), ("bit", True), ("sparsevec", True)],
    )
    def test_quantized_follows_the_column_type(
        self, typed_table: Any, kind: str, quantized: bool
    ) -> None:
        store = open_store(typed_table(kind))
        try:
            assert store.capabilities.quantized is quantized
        finally:
            store.close()


class TestTheRoundTrip:
    """The declaration above, checked against what actually comes back."""

    def test_a_vector_column_returns_exactly_what_it_was_given(
        self, typed_table: Any, vectors: Any
    ) -> None:
        """float32 in, float32 out. This is the promise `rollback` rests on."""
        store, read_back = _read_back(typed_table("vector"))
        try:
            assert store.capabilities.quantized is False
            assert _worst(read_back, vectors) == 0.0
        finally:
            store.close()

    def test_a_halfvec_column_rounds_and_by_more_than_migrate_tolerates(
        self, typed_table: Any, vectors: Any
    ) -> None:
        """The measurement that decides whether `migrate` can use such a column.

        float16 carries about three decimal digits, and `migrate` re-reads a
        sample of every batch and compares it to what it sent at 1e-4. If the
        rounding is coarser than that, a migration into a `halfvec` column stops
        on its own first batch — and the pre-flight plan has to say so rather
        than letting it surface as a failed write.

        The sqlite-vec `int8` finding is the precedent, and this is the same
        check run before the same mistake: *probably* fine is not an answer.
        """
        store, read_back = _read_back(typed_table("halfvec"))
        try:
            worst = _worst(read_back, vectors)
            assert store.capabilities.quantized is True
            assert worst > 0.0, "a float16 column that lost nothing would not be float16"
            assert worst > VERIFY_ATOL, (
                f"halfvec rounded by {worst:.2e}, inside migrate's {VERIFY_ATOL} "
                f"read-back tolerance — the guide's claim needs re-measuring"
            )
        finally:
            store.close()


class TestWhatRebasisRefuses:
    """Partial support declares what it cannot do, at the moment it is asked."""

    @pytest.mark.parametrize("kind", ["bit", "sparsevec"])
    def test_a_code_column_is_not_readable_and_not_writable(
        self, typed_table: Any, kind: str, vectors: Any
    ) -> None:
        """Neither type hands back a reconstruction of the vector written.

        A `bit` column holds one bit per component and a `sparsevec` holds an
        index/value map; converting either into a dense float32 array would be
        rebasis inventing a vector rather than reading one. So both are declared
        unreadable and unwritable, and `upsert_vectors` refuses with the reason
        even though `capabilities` already stopped `migrate` before it started.
        """
        store = open_store(typed_table(kind))
        try:
            assert store.capabilities.can_read_vectors is False
            assert store.capabilities.can_upsert_vectors is False
            with pytest.raises(StoreUnsupported) as raised:
                store.upsert_vectors(["doc-0"], vectors[:1])
            assert raised.value.code.startswith("RB-E")
            assert raised.value.hint
        finally:
            store.close()
