"""Telemetry: off by default, and inert when off.

The property that matters most is the negative one. rebasis collects nothing and
sends nothing; the ``[otel]`` extra exists so a user can send data to *their
own* collector. A test suite that only checked "spans work when enabled" would
not catch the failure that actually matters — data leaving a machine because a
variable was set for some other application.
"""

from __future__ import annotations

import pytest

from rebasis.observability import (
    METRICS,
    SPAN_TREE,
    Spans,
    instrument,
    otel_enabled,
    record_error,
    setup_telemetry,
    should_span_batch,
    span,
    telemetry_status,
)

pytestmark = pytest.mark.unit


class TestOffByDefault:
    def test_telemetry_is_off_with_no_configuration(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.delenv("REBASIS_OTEL_ENABLED", raising=False)

        assert not otel_enabled()

    def test_a_foreign_otlp_endpoint_does_not_turn_it_on(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """A machine may already export to a collector for another application.
        rebasis must not start shipping there uninvited."""
        monkeypatch.delenv("REBASIS_OTEL_ENABLED", raising=False)
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector.example:4318")

        assert not otel_enabled()
        assert not setup_telemetry()

    def test_setup_does_nothing_when_the_flag_is_unset(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.delenv("REBASIS_OTEL_ENABLED", raising=False)

        assert setup_telemetry() is False
        assert telemetry_status()["active"] is False


class TestInertWhenOff:
    def test_spans_are_usable_and_do_nothing(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """Call sites are unconditional, so the no-op must behave like a span."""
        monkeypatch.delenv("REBASIS_OTEL_ENABLED", raising=False)

        with span(Spans.PROBE, {"seed": 0}) as active:
            active.set_attribute("rebasis.probe.decision", "bridge_sufficient")
            assert active.is_recording() is False

    def test_instruments_accept_values_and_discard_them(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.delenv("REBASIS_OTEL_ENABLED", raising=False)

        instrument("rebasis.embed.texts").add(10, {"model": "m"})
        instrument("rebasis.embed.duration").record(1.5, {"model": "m"})
        instrument("rebasis.probe.arr").set(0.93, {"metric": "r10"})
        record_error("RB-E3004", transient=True)


class TestNames:
    def test_every_metric_declares_a_kind_a_unit_and_a_description(self) -> None:
        for name, (kind, unit, description) in METRICS.items():
            assert kind in {"counter", "histogram", "gauge"}, name
            assert unit
            assert description

    def test_an_unregistered_metric_name_is_refused(self) -> None:
        """Metric names are an interface. Inventing one at a call site is how a
        dashboard silently stops matching."""
        with pytest.raises(KeyError):
            instrument("rebasis.made.this.up")

    def test_the_span_tree_only_references_declared_names(self) -> None:
        declared = {v for k, v in vars(Spans).items() if not k.startswith("_")}

        for parent, children in SPAN_TREE.items():
            assert parent in declared
            assert set(children) <= declared

    def test_the_names_used_at_call_sites_are_in_the_tree(self) -> None:
        """The tree is documented; the code must not quietly grow a branch."""
        in_tree = set(SPAN_TREE) | {c for children in SPAN_TREE.values() for c in children}
        declared = {v for k, v in vars(Spans).items() if not k.startswith("_")}

        assert declared == in_tree


class TestBatchSampling:
    def test_the_first_few_batches_get_spans(self) -> None:
        assert should_span_batch(0)
        assert should_span_batch(2)

    def test_most_batches_do_not(self) -> None:
        """A trace where the batches outnumber everything else hides the shape
        of the run."""
        assert not should_span_batch(7)
        assert not should_span_batch(99)

    def test_every_hundredth_does(self) -> None:
        assert should_span_batch(100)
        assert should_span_batch(500)
