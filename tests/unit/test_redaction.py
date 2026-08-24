"""Redaction contract.

Two distinct claims are tested:

1. Sensitive content never reaches the output.
2. The redaction counter stays at **zero** on production code paths. Redaction
   firing is a bug signal, not a safety net doing its job — the correct
   behaviour is never to log the field at all.

The second is why this file uses property-based testing: a fixed list of
"sensitive field names" is a denylist by another name, and denylists miss the
field somebody adds next month.
"""

from __future__ import annotations

import io
import json
import logging
import sys
from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rebasis.observability import Events, configure_logging, get_logger, redaction_counter
from rebasis.observability.env import LogSettings, Runtime
from rebasis.observability.redaction import ALLOWED_FIELDS, make_redactor

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Categories forbidden outright. Vectors are on the list because text can
#: be reconstructed from embeddings (K23) — a vector in a log is as sensitive as
#: plaintext.
FORBIDDEN_SAMPLES = {
    "chunk_text": "the user's private note about their salary",
    "document": "confidential internal memo",
    "text": "raw document body",
    "query": "how do I reset my production database",
    "query_text": "another private query",
    "vector": [0.1, 0.2, 0.3],
    "embedding": [0.9, 0.8],
    "path": "/Users/someone/vault/secrets.md",
    "file_path": "/home/user/private/notes.md",
    "api_key": "sk-live-abcdef",
    "store_uri": "postgres://user:password@host/db",
}


def _capture(settings: LogSettings) -> tuple[io.StringIO, object]:
    buffer = io.StringIO()
    real_stderr = sys.stderr
    sys.stderr = buffer
    configure_logging(settings=settings, force=True)
    return buffer, real_stderr


@pytest.fixture
def json_logger() -> Iterator[tuple[object, io.StringIO]]:
    """A logger writing JSON into a buffer."""
    settings = LogSettings(
        runtime=Runtime.CI,
        level=logging.DEBUG,
        fmt="json",
        colour=False,
        utc_timestamps=True,
        caller_info=False,
    )
    buffer, real_stderr = _capture(settings)
    try:
        yield get_logger("rebasis.test"), buffer
    finally:
        sys.stderr = real_stderr  # type: ignore[assignment]


@pytest.mark.unit
@pytest.mark.parametrize(("field", "value"), sorted(FORBIDDEN_SAMPLES.items()))
def test_forbidden_content_never_reaches_output(
    json_logger: tuple[object, io.StringIO], field: str, value: object
) -> None:
    """No forbidden category may appear in the rendered log."""
    log, buffer = json_logger
    log.info(Events.EMBED_BATCH_COMPLETED, count=1, **{field: value})  # type: ignore[attr-defined]
    output = buffer.getvalue()

    assert f"<redacted:{field}>" in output
    needle = str(value)
    assert needle not in output, f"{field} leaked into the log"


@pytest.mark.unit
def test_allowed_fields_pass_through(json_logger: tuple[object, io.StringIO]) -> None:
    """Allowlisted fields must survive — redaction must not eat real signal."""
    log, buffer = json_logger
    log.info(  # type: ignore[attr-defined]
        Events.PROBE_DECISION_MADE,
        decision="bridge_sufficient",
        arr_r10=0.96,
        count=10_000,
        seed=0,
        borderline=False,
    )
    record = json.loads(buffer.getvalue().strip())
    assert record["decision"] == "bridge_sufficient"
    assert record["arr_r10"] == pytest.approx(0.96)
    assert record["count"] == 10_000
    assert redaction_counter.count == 0


@pytest.mark.unit
def test_counter_is_zero_on_a_clean_path(json_logger: tuple[object, io.StringIO]) -> None:
    """A legitimate call site must never trip the counter.

    A non-zero counter in production means somebody logged a field they should
    not have — that is the signal this counter exists to give.
    """
    log, _ = json_logger
    log.info(Events.STORE_OPENED, store_backend="memory", count=100, dim=384)  # type: ignore[attr-defined]
    log.info(Events.EMBED_PROFILE_RESOLVED, embed_backend="fastembed", dim=384, symmetric=True)  # type: ignore[attr-defined]
    assert redaction_counter.count == 0


@pytest.mark.property
@given(
    field=st.text(
        alphabet=st.characters(min_codepoint=97, max_codepoint=122),
        min_size=1,
        max_size=20,
    ),
    value=st.text(min_size=1, max_size=50),
)
def test_arbitrary_unknown_field_is_redacted(field: str, value: str) -> None:
    """Any field outside the allowlist is redacted, whatever it is called.

    This is the allowlist property: unknown means rejected. A denylist would
    only cover the names somebody thought of.
    """
    if field in ALLOWED_FIELDS:
        return
    redact = make_redactor()
    result = redact(None, "info", {"event": "probe.run.started", field: value})
    # Equality with the marker is the complete check: the value is gone by
    # construction. Asserting `value not in result[field]` in addition looks
    # stronger but is wrong — the marker embeds the FIELD NAME, so a value that
    # happens to equal its own field name fails a check that nothing leaked.
    # (hypothesis found exactly that: field="a", value="a".)
    assert result[field] == f"<redacted:{field}>"


@pytest.mark.unit
def test_unsafe_mode_disables_redaction() -> None:
    """`--unsafe-log-content` is a real escape hatch.

    It exists for debugging, prints a warning on every run and is written to the
    audit trail — disabling redaction cannot be hidden, but it must actually work.
    """
    passthrough = make_redactor(enabled=False)
    payload = {"event": "probe.run.started", "chunk_text": "secret"}
    assert passthrough(None, "info", dict(payload))["chunk_text"] == "secret"


@pytest.mark.unit
def test_counter_records_which_fields_tripped_it() -> None:
    """The counter names the offending fields — otherwise it is not actionable."""
    redaction_counter.reset()
    redact = make_redactor()
    redact(None, "info", {"event": "probe.run.started", "leak_a": 1, "leak_b": 2})
    assert redaction_counter.count == 2
    assert redaction_counter.fields == {"leak_a", "leak_b"}
