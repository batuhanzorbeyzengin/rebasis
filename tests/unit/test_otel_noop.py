"""OpenTelemetry must be genuinely optional.

A CI job installs without the `otel` extra and runs the whole unit suite.
Without it, an optional dependency becomes mandatory sooner or later — someone
writes `span.set_attribute(...)` against the real API and nobody notices until a
user without the extra reports a crash.
"""

from __future__ import annotations

import pytest

from rebasis.observability.otel import (
    get_meter,
    get_tracer,
    otel_available,
    otel_enabled,
    span_context_fields,
)


@pytest.mark.unit
def test_telemetry_is_off_by_default() -> None:
    """rebasis collects no telemetry and sends data nowhere."""
    assert not otel_enabled()


@pytest.mark.unit
def test_enabling_needs_the_explicit_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """`REBASIS_OTEL_ENABLED` is a separate gate on top of the standard variables.

    A user's machine may already have `OTEL_EXPORTER_OTLP_ENDPOINT` set for
    another application; rebasis must not start shipping data there on its own.
    """
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    assert not otel_enabled(), "an endpoint alone must not enable telemetry"

    monkeypatch.setenv("REBASIS_OTEL_ENABLED", "1")
    assert otel_enabled() == otel_available()


@pytest.mark.unit
def test_span_usage_works_without_otel() -> None:
    """Call sites write spans unconditionally; without OTel they do nothing."""
    tracer = get_tracer()
    with tracer.start_as_current_span("rebasis.probe") as span:
        span.set_attribute("rebasis.probe.arr_r10", 0.96)
        span.set_attributes({"rebasis.probe.decision": "bridge_sufficient"})
        span.add_event("sample.drawn", {"count": 10_000})
        assert span.is_recording() is False
        with tracer.start_as_current_span("rebasis.embed.batch") as inner:
            inner.set_attribute("gen_ai.request.model", "bge-m3")


@pytest.mark.unit
def test_metric_instruments_work_without_otel() -> None:
    """Counters, histograms and gauges are all inert but callable."""
    meter = get_meter()
    meter.create_counter("rebasis.embed.texts").add(128, {"backend": "fastembed"})
    meter.create_histogram("rebasis.embed.duration").record(42.0)
    meter.create_gauge("rebasis.migrate.progress").set(0.5)


@pytest.mark.unit
def test_span_context_is_empty_without_a_span() -> None:
    """Correlation fields are `None` rather than absent or raising."""
    assert span_context_fields() == {"trace_id": None, "span_id": None}
