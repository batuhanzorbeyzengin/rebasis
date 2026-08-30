"""Resolving a table's columns, and the errors when it cannot be done.

A Chroma collection has a shape; a Postgres table is whatever its owner made.
Everything a pgvector user gets wrong happens here rather than in the queries —
a column named `emb` instead of `embedding`, a text column in a joined table, a
`halfvec` where a `vector` was assumed — so this is where the error messages
have to be good.

These need no server: they are the pure half of the backend, and running them
everywhere is the point. The half that needs a real PostgreSQL is
`tests/integration/test_pgvector_types.py` and the contract suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from rebasis.errors import CollectionNotFound
from rebasis.store.backends.pgvector import (
    _LOSSY,
    _describe,
    _literal,
    _optional,
    _parse_vector,
    _pick,
    _quote,
    _where,
)

pytestmark = pytest.mark.unit

COLUMNS = {"chunk_id": "text", "body": "text", "emb": "vector", "created": "timestamp"}


class TestFindingTheColumns:
    def test_a_named_column_is_used(self) -> None:
        assert _pick("emb", ("embedding",), COLUMNS, kind="vector", what="vector") == "emb"

    def test_a_conventional_name_is_found(self) -> None:
        columns = {"id": "text", "embedding": "vector"}

        assert _pick(None, ("embedding", "vec"), columns, kind="vector", what="vector") == (
            "embedding"
        )

    def test_nothing_is_inferred_from_being_the_only_column_of_its_type(self) -> None:
        """`emb` is the only vector column in the table and is still not chosen.

        A table with two vector columns would otherwise have `migrate` rewrite
        whichever one the catalogue happened to return first, and the failure
        would be silent and total.
        """
        with pytest.raises(CollectionNotFound) as raised:
            _pick(None, ("embedding", "vec"), COLUMNS, kind="vector", what="vector")

        assert raised.value.code.startswith("RB-E")

    def test_the_error_lists_the_table_it_could_not_read(self) -> None:
        """The user is looking at a schema rebasis cannot see, so the fastest
        way to unblock them is to show what it *did* see."""
        with pytest.raises(CollectionNotFound) as raised:
            _pick(None, ("embedding",), COLUMNS, kind="vector", what="vector")

        message = f"{raised.value} {raised.value.hint}"
        for name, kind in COLUMNS.items():
            assert name in message
            assert kind in message
        assert "?vector=" in message

    def test_a_named_column_that_is_not_there_is_a_typo_and_says_so(self) -> None:
        with pytest.raises(CollectionNotFound) as raised:
            _pick("embeding", ("embedding",), COLUMNS, kind="vector", what="vector")

        assert "embeding" in str(raised.value)

    def test_a_missing_text_column_is_a_shape_rather_than_an_error(self) -> None:
        """A table with no text is a table `probe` and the bridge still work
        against; `can_read_text: false` is what says so."""
        assert _optional(None, ("text", "content"), {"id": "text"}) is None

    def test_a_named_text_column_that_is_not_there_is_still_an_error(self) -> None:
        """That one is a typo, not a shape."""
        with pytest.raises(CollectionNotFound):
            _optional("contnet", ("text",), COLUMNS)

    def test_describe_names_every_column_with_its_type(self) -> None:
        described = _describe({"id": "text", "emb": "halfvec"})

        assert described == "id (text), emb (halfvec)"


class TestTheTypeTable:
    def test_only_vector_round_trips_exactly(self) -> None:
        """The other three are lossy, and `quantized` follows this table.

        `vector` declaring `False` is the promise `rollback` rests on, which is
        why it is the one entry worth asserting on its own.
        """
        assert _LOSSY["vector"] is False
        assert all(_LOSSY[kind] for kind in ("halfvec", "bit", "sparsevec"))


class TestSQLSafety:
    def test_an_identifier_with_a_quote_in_it_is_doubled(self) -> None:
        """Column names come from a URI, which is a string from outside. This is
        the boundary where it stops being one."""
        assert _quote('we"ird') == '"we""ird"'

    def test_a_filter_binds_its_values_and_quotes_its_keys(self) -> None:
        clause, parameters = _where({"tenant": "acme", "lang": "en"})

        assert clause == ' WHERE "tenant" = %s AND "lang" = %s'
        assert parameters == ("acme", "en")

    def test_no_filter_is_no_clause(self) -> None:
        assert _where(None) == ("", ())
        assert _where({}) == ("", ())


class TestTheTextForm:
    def test_a_vector_round_trips_through_pgvector_s_own_syntax(self) -> None:
        vector = np.array([1.5, -0.25, 0.0], dtype=np.float32)

        assert _literal(vector) == "[1.5,-0.25,0.0]"
        assert np.array_equal(_parse_vector(_literal(vector)), vector)

    def test_a_null_column_reads_back_as_absent(self) -> None:
        """A row whose vector is NULL has no vector, which is not a zero one."""
        assert _parse_vector(None) is None
