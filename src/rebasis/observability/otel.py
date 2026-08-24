"""OpenTelemetry integration, off by default.

**rebasis collects no telemetry and sends data nowhere.** OTel is an option for
users to send data to *their own* infrastructure — not to rebasis. That
distinction is stated plainly in the README, because "OpenTelemetry support"
reads easily as "they are watching me".

When the OTel packages are not installed this module returns no-op
implementations. Call sites write ``with tracer.start_as_current_span(...)``
unconditionally; without OTel that context manager simply does nothing. The core
path never depends on OTel.

``REBASIS_OTEL_ENABLED`` is a separate gate on top of the standard OTel
variables, because a user's machine may already have
``OTEL_EXPORTER_OTLP_ENDPOINT`` set for a different application. rebasis must not
start shipping data there on its own.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

__all__ = [
    "get_meter",
    "get_tracer",
    "otel_available",
    "otel_enabled",
    "span_context_fields",
]


def otel_available() -> bool:
    """Whether the OTel API package is importable."""
    try:
        import opentelemetry.trace  # noqa: F401
    except ImportError:
        return False
    return True


def otel_enabled() -> bool:
    """Whether the user explicitly turned telemetry on *and* it can run."""
    flag = os.environ.get("REBASIS_OTEL_ENABLED", "").strip().lower()
    return flag in ("1", "true", "yes", "on") and otel_available()


class _NoOpSpan:
    """A span that records nothing and costs nothing."""

    __slots__ = ()

    def set_attribute(self, key: str, value: Any) -> None:
        return

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        return

    def add_event(self, name: str, attributes: Mapping[str, Any] | None = None) -> None:
        return

    def record_exception(self, exception: BaseException) -> None:
        return

    def is_recording(self) -> bool:
        return False


class _NoOpTracer:
    """Tracer stand-in used when OTel is absent or disabled."""

    __slots__ = ()

    @contextmanager
    def start_as_current_span(
        self, name: str, attributes: Mapping[str, Any] | None = None
    ) -> Iterator[_NoOpSpan]:
        """Context manager that yields an inert span."""
        del name, attributes
        yield _NoOpSpan()


class _NoOpInstrument:
    """Counter / histogram / gauge stand-in."""

    __slots__ = ()

    def add(self, amount: float, attributes: Mapping[str, Any] | None = None) -> None:
        return

    def record(self, amount: float, attributes: Mapping[str, Any] | None = None) -> None:
        return

    def set(self, amount: float, attributes: Mapping[str, Any] | None = None) -> None:
        return


class _NoOpMeter:
    """Meter stand-in that hands out inert instruments."""

    __slots__ = ()

    def create_counter(self, name: str, **kwargs: Any) -> _NoOpInstrument:
        del name, kwargs
        return _NoOpInstrument()

    def create_histogram(self, name: str, **kwargs: Any) -> _NoOpInstrument:
        del name, kwargs
        return _NoOpInstrument()

    def create_gauge(self, name: str, **kwargs: Any) -> _NoOpInstrument:
        del name, kwargs
        return _NoOpInstrument()


_NOOP_TRACER = _NoOpTracer()
_NOOP_METER = _NoOpMeter()


def get_tracer(name: str = "rebasis") -> Any:
    """Return the real tracer when telemetry is on, otherwise a no-op."""
    if not otel_enabled():
        return _NOOP_TRACER
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except Exception:  # pragma: no cover - defensive; telemetry never breaks a run
        return _NOOP_TRACER


def get_meter(name: str = "rebasis") -> Any:
    """Return the real meter when telemetry is on, otherwise a no-op."""
    if not otel_enabled():
        return _NOOP_METER
    try:
        from opentelemetry import metrics

        return metrics.get_meter(name)
    except Exception:  # pragma: no cover - defensive
        return _NOOP_METER


def span_context_fields() -> dict[str, str | None]:
    """Current ``trace_id`` / ``span_id`` for log correlation.

    Returns ``None`` for both when no span is active. This is what lets a reader
    go from a JSON log line to a trace, from the trace to an audit record, and
    from the audit record to ``replay``.
    """
    if not otel_enabled():
        return {"trace_id": None, "span_id": None}
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if not ctx.is_valid:
            return {"trace_id": None, "span_id": None}
        return {
            "trace_id": format(ctx.trace_id, "032x"),
            "span_id": format(ctx.span_id, "016x"),
        }
    except Exception:  # pragma: no cover - defensive
        return {"trace_id": None, "span_id": None}
