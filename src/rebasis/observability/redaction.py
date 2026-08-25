"""Allowlist redaction.

In a tool that reads personal corpora, content leaking into logs is the most
serious risk. The choice of approach is decisive:

**Denylist (wrong):** "mask these fields" — a newly added field is forgotten and
leaks silently.

**Allowlist (right):** "only these fields may be logged" — a new field is
rejected by default.

The processor replaces every non-allowed field with ``<redacted:name>`` and
increments a counter. **That counter must stay at zero in production code
paths**: redaction firing is a bug signal, not a safety net doing its job. The
correct behaviour is never to log the field in the first place.

Never logged, categorically:

- document / chunk text
- raw or derived vector values
- query text
- filesystem paths (they contain usernames)
- model API keys
- the credential portion of a store URI

Vectors are on that list because text can be reconstructed from embeddings —
that is what vec2text does. A vector in a log file is as sensitive as plaintext.
This is a requirement, not a preference.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator, MutableMapping

__all__ = [
    "ALLOWED_FIELDS",
    "RedactionCounter",
    "make_redactor",
    "redaction_counter",
]

#: Fields permitted to appear in a log record. Anything else is redacted.
#:
#: Adding a field here is a deliberate act: it asserts that the field can never
#: carry corpus content. Ids, counts, dimensions, durations and enum-like
#: strings qualify; anything free-form does not.
ALLOWED_FIELDS: frozenset[str] = frozenset(
    {
        # structlog / rendering internals
        "event",
        "level",
        "levelname",
        "timestamp",
        "logger",
        "exc_info",
        "stack_info",
        "pathname",
        "lineno",
        "func_name",
        "module",
        # correlation identities
        "run_id",
        "trace_id",
        "span_id",
        "audit_seq",
        "job_id",
        # invocation
        "command",
        "environment",
        "log_level",
        "log_format",
        "rebasis_version",
        # backends and configuration — enum-like, never free-form
        "store_backend",
        "embed_backend",
        "adapter_type",
        "direction",
        "backend",
        "device",
        "strategy",
        "tier",
        # A public model identifier, never corpus content — and an audit record
        # must carry it among a decision's reproducible inputs, so redacting it
        # in the log while recording it in the audit trail helped nobody.
        "model_id",
        # Why some inputs were skipped. A fixed vocabulary, not free text: see
        # DROPPED_* in rebasis.observability.events.
        "dropped",
        "state",
        "action",
        "symmetric",
        # identifiers and counters
        "record_id",
        "batch_index",
        "attempt",
        "count",
        "field_count",
        "seed",
        "dim",
        # measurements
        "duration_ms",
        "peak_rss_bytes",
        "blas_threads",
        # decision metrics
        "arr_r10",
        "arr_mrr",
        "naive_overlap",
        "score_shift",
        "tail_arr",
        "spearman_topk",
        "decision",
        "borderline",
        "upgrade_gain",
        # A property of the index structure, not of the corpus: how much of the
        # exact answer the store's own search returns.
        "ann_recall",
        # Aggregate geometry of the two spaces: a root-mean-square over pairwise
        # similarities and the bound it implies. Carries no document.
        "geometry_delta",
        "alignment_bound",
        # errors
        "error_code",
        "transient",
    }
)

_REDACTED_TEMPLATE = "<redacted:{}>"


class RedactionCounter:
    """Thread-safe counter for redaction events.

    Exposed so tests can assert it stays at zero across production code paths.
    Property-based tests feed random field names through the logger and require
    that no legitimate call site ever trips it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count = 0
        self._fields: set[str] = set()

    def record(self, field: str) -> None:
        """Note that ``field`` was redacted."""
        with self._lock:
            self._count += 1
            self._fields.add(field)

    @property
    def count(self) -> int:
        """How many fields have been redacted since the last reset."""
        with self._lock:
            return self._count

    @property
    def fields(self) -> frozenset[str]:
        """Which field names were redacted — the actionable part for a fix."""
        with self._lock:
            return frozenset(self._fields)

    def reset(self) -> None:
        """Clear the counter (test fixtures use this between cases)."""
        with self._lock:
            self._count = 0
            self._fields.clear()


#: Process-wide counter. Read by tests and by the ``doctor`` command.
redaction_counter = RedactionCounter()


def make_redactor(*, allowed: frozenset[str] = ALLOWED_FIELDS, enabled: bool = True) -> Any:
    """Build the structlog redaction processor.

    Args:
        allowed: Field names permitted through. Defaults to :data:`ALLOWED_FIELDS`.
        enabled: When ``False`` the processor is a pass-through. This backs
            ``--unsafe-log-content``, which additionally prints a red warning on
            every run and writes ``config.unsafe_logging_enabled`` to the audit
            trail. Disabling redaction cannot be hidden.

    The processor sits **immediately before the renderer**, not at the end of the
    chain, so that no later processor can add a field after redaction has run.
    """

    def redact(
        _logger: Any, _method_name: str, event_dict: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        if not enabled:
            return event_dict
        for key in _offending_keys(event_dict, allowed):
            event_dict[key] = _REDACTED_TEMPLATE.format(key)
            redaction_counter.record(key)
        return event_dict

    return redact


def _offending_keys(event_dict: MutableMapping[str, Any], allowed: frozenset[str]) -> Iterator[str]:
    """Keys not in the allowlist.

    Materialised into a list by the caller before mutation: mutating a mapping
    while iterating it raises at runtime, and this code path runs on every log
    record.
    """
    return iter([k for k in event_dict if k not in allowed])
