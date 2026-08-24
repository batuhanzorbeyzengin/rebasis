"""Observability: the single entry point for logging, telemetry and events.

Modules import ``get_logger`` from here and nothing else. The layer contract
allows ``observability`` to depend on ``errors`` and ``types`` and nothing more,
which is what lets ``core`` depend on it without dragging in weight — with OTel
absent the tracer is a no-op.
"""

from __future__ import annotations

from rebasis.observability.env import (
    LogSettings,
    Runtime,
    detect_runtime,
    resolve_log_settings,
)
from rebasis.observability.events import (
    DROPPED_NO_TEXT,
    DROPPED_UNJUDGED,
    EVENT_CATALOG,
    Events,
    EventSpec,
)
from rebasis.observability.logging import (
    configure_logging,
    current_settings,
    get_logger,
    is_configured,
    reset_logging,
)
from rebasis.observability.otel import get_meter, get_tracer, otel_enabled
from rebasis.observability.redaction import ALLOWED_FIELDS, redaction_counter
from rebasis.observability.resources import (
    ResourceSummary,
    cpu_seconds,
    measure_resources,
    peak_rss_bytes,
)
from rebasis.observability.retry import retry_transient
from rebasis.observability.telemetry import (
    METRICS,
    SPAN_TREE,
    Spans,
    instrument,
    record_error,
    setup_telemetry,
    should_span_batch,
    shutdown_telemetry,
    span,
    telemetry_status,
)

__all__ = [
    "ALLOWED_FIELDS",
    "DROPPED_NO_TEXT",
    "DROPPED_UNJUDGED",
    "EVENT_CATALOG",
    "METRICS",
    "SPAN_TREE",
    "EventSpec",
    "Events",
    "LogSettings",
    "ResourceSummary",
    "Runtime",
    "Spans",
    "configure_logging",
    "cpu_seconds",
    "current_settings",
    "detect_runtime",
    "get_logger",
    "get_meter",
    "get_tracer",
    "instrument",
    "is_configured",
    "measure_resources",
    "otel_enabled",
    "peak_rss_bytes",
    "record_error",
    "redaction_counter",
    "reset_logging",
    "resolve_log_settings",
    "retry_transient",
    "setup_telemetry",
    "should_span_batch",
    "shutdown_telemetry",
    "span",
    "telemetry_status",
]
