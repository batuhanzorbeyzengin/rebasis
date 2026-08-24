"""Span and metric names, and the SDK setup behind them.

**rebasis collects no telemetry and sends data nowhere.** What this module does
is let a user send data to *their own* collector. Nothing here runs unless
``REBASIS_OTEL_ENABLED`` is set, and even then the endpoint is the user's.

The names live here rather than at the call sites for the same reason the
attribute names live in :mod:`rebasis.observability.semconv`: a span tree is an
interface, and an interface that is spelled out at fifteen call sites drifts.
:data:`SPAN_TREE` is the documented shape, and a test asserts every name a call
site uses appears in it.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Final

from rebasis.observability.otel import get_meter, get_tracer, otel_available, otel_enabled

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "METRICS",
    "SPAN_TREE",
    "Spans",
    "instrument",
    "record_error",
    "setup_telemetry",
    "shutdown_telemetry",
]


class Spans:
    """Every span name rebasis may open."""

    PROBE: Final = "rebasis.probe"
    SAMPLE_DRAW: Final = "rebasis.sample.draw"
    EMBED_CORPUS: Final = "rebasis.embed.corpus"
    EMBED_BATCH: Final = "rebasis.embed.batch"
    GROUNDTRUTH_KNN: Final = "rebasis.groundtruth.knn"
    ADAPTER_FIT: Final = "rebasis.adapter.fit"
    ADAPTER_EVALUATE: Final = "rebasis.adapter.evaluate"
    DECISION: Final = "rebasis.decision"

    MIGRATE: Final = "rebasis.migrate"
    MIGRATE_BATCH: Final = "rebasis.migrate.batch"
    STORE_UPSERT: Final = "rebasis.store.upsert"
    ADAPTER_REFIT: Final = "rebasis.adapter.refit"


#: Parent → children, the documented tree. Kept as data so it can be checked
#: rather than trusted, and printed by `rebasis doctor --telemetry`.
SPAN_TREE: Final[dict[str, tuple[str, ...]]] = {
    Spans.PROBE: (
        Spans.SAMPLE_DRAW,
        Spans.EMBED_CORPUS,
        Spans.GROUNDTRUTH_KNN,
        Spans.ADAPTER_FIT,
        Spans.ADAPTER_EVALUATE,
        Spans.DECISION,
    ),
    Spans.EMBED_CORPUS: (Spans.EMBED_BATCH,),
    Spans.MIGRATE: (Spans.MIGRATE_BATCH, Spans.ADAPTER_REFIT),
    Spans.MIGRATE_BATCH: (Spans.EMBED_BATCH, Spans.STORE_UPSERT),
}

#: name → (kind, unit, description). The metric catalogue, as data.
METRICS: Final[dict[str, tuple[str, str, str]]] = {
    "rebasis.embed.duration": ("histogram", "ms", "Time spent embedding a batch"),
    "rebasis.embed.texts": ("counter", "1", "Texts embedded"),
    "rebasis.adapter.fit.duration": ("histogram", "ms", "Time spent fitting one adapter"),
    "rebasis.adapter.apply.duration": ("histogram", "us", "Time spent applying an adapter"),
    "rebasis.probe.arr": ("gauge", "1", "Adapted Recall Retention"),
    "rebasis.migrate.items": ("counter", "1", "Records that reached a terminal state"),
    "rebasis.migrate.progress": ("gauge", "1", "Fraction of a job completed"),
    "rebasis.store.search.duration": ("histogram", "ms", "Time spent in a store query"),
    "rebasis.errors": ("counter", "1", "Errors raised, by stable code"),
}

#: Batch spans are sampled: the first few, then every hundredth. A 500,000
#: record migration emits 2,000 batch spans otherwise, which buries the run's
#: shape in its own detail — the same reasoning as the hot-loop rule for logs.
BATCH_SPAN_HEAD = 3
BATCH_SPAN_EVERY = 100

_providers: list[Any] = []
_instruments: dict[str, Any] = {}


def should_span_batch(index: int) -> bool:
    """Whether batch number ``index`` (0-based) gets a span."""
    return index < BATCH_SPAN_HEAD or index % BATCH_SPAN_EVERY == 0


def setup_telemetry(*, service_name: str = "rebasis") -> bool:
    """Configure the OTel SDK, if the user asked for it.

    Returns whether telemetry is now running. Called once from the CLI; a
    library user who wants tracing configures their own provider and rebasis
    picks it up through the global one.

    Three ways this deliberately does nothing:

    - ``REBASIS_OTEL_ENABLED`` is unset. The standard OTel variables are not
      enough on their own: a machine may already have
      ``OTEL_EXPORTER_OTLP_ENDPOINT`` pointing somewhere for another
      application, and rebasis must not start shipping there uninvited.
    - The SDK is not installed. The ``[otel]`` extra is optional.
    - A provider is already configured. The host application owns it.
    """
    if not otel_enabled():
        return False

    from rebasis.__about__ import __version__
    from rebasis.observability.semconv import SERVICE_NAME, SERVICE_VERSION

    try:
        from opentelemetry import metrics, trace
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        # The API is installed but not the SDK: spans still work if the host
        # application configured a provider, and cost nothing if it did not.
        return False

    resource = Resource.create({SERVICE_NAME: service_name, SERVICE_VERSION: __version__})

    tracer_provider = TracerProvider(resource=resource)
    span_exporter = _span_exporter()
    if span_exporter is not None:
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)
    _providers.append(tracer_provider)

    metric_exporter = _metric_exporter()
    if metric_exporter is not None:
        meter_provider = MeterProvider(
            resource=resource, metric_readers=[PeriodicExportingMetricReader(metric_exporter)]
        )
        metrics.set_meter_provider(meter_provider)
        _providers.append(meter_provider)

    _instruments.clear()
    return True


def shutdown_telemetry() -> None:
    """Flush and stop the providers this module created.

    Without this a short CLI run exits before the batch processor's first
    export interval, and the user sees nothing in their collector and concludes
    the integration is broken.
    """
    import contextlib

    while _providers:
        provider = _providers.pop()
        # Telemetry must never break a run, least of all on the way out.
        with contextlib.suppress(Exception):
            provider.shutdown()
    _instruments.clear()


def instrument(name: str) -> Any:
    """Get (and cache) the instrument registered under ``name``.

    Raises:
        KeyError: When the name is not in :data:`METRICS`. Metric names are an
            interface; inventing one at a call site is how a dashboard silently
            stops matching.
    """
    if name in _instruments:
        return _instruments[name]

    kind, unit, description = METRICS[name]
    meter = get_meter()
    factory = {
        "counter": meter.create_counter,
        "histogram": meter.create_histogram,
        "gauge": meter.create_gauge,
    }[kind]
    created = factory(name, unit=unit, description=description)
    _instruments[name] = created
    return created


def record_error(code: str, *, transient: bool = False) -> None:
    """Count an error by its stable code.

    The second payoff of stable codes: documentation and metric labels share
    one vocabulary, so a spike on a dashboard is greppable in the docs.
    """
    instrument("rebasis.errors").add(1, {"error_code": code, "transient": transient})


def telemetry_status() -> dict[str, Any]:
    """What `rebasis doctor` reports about telemetry."""
    return {
        "available": otel_available(),
        "enabled": otel_enabled(),
        "endpoint": os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", ""),
        "active": bool(_providers),
    }


def _span_exporter() -> Any:
    """OTLP when an endpoint is configured, console when explicitly asked.

    Console export is not a fallback: printing spans to a terminal that was
    expecting a report is worse than exporting nothing.
    """
    if os.environ.get("REBASIS_OTEL_CONSOLE", "").strip().lower() in ("1", "true", "yes", "on"):
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        return ConsoleSpanExporter()
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return None
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError:
        return None
    return OTLPSpanExporter()


def _metric_exporter() -> Any:
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return None
    try:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    except ImportError:
        return None
    return OTLPMetricExporter()


def span(name: str, attributes: Mapping[str, Any] | None = None) -> Any:
    """Open a span. Inert when telemetry is off, so call sites stay unconditional."""
    return get_tracer().start_as_current_span(name, attributes=dict(attributes or {}))
