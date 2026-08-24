# OpenTelemetry

## What this is and is not

**rebasis collects no telemetry and sends nothing anywhere.** This directory is
about the opposite direction: letting *you* send rebasis traces and metrics to
*your own* collector.

It is off unless you turn it on, and setting `REBASIS_OTEL_ENABLED` is the only
thing that turns it on. Having `OTEL_EXPORTER_OTLP_ENDPOINT` set for some other
application on the same machine is deliberately not enough — rebasis will not
start shipping to an endpoint it was not pointed at.

## Turning it on

```bash
pip install "rebasis[otel]"

export REBASIS_OTEL_ENABLED=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318

rebasis probe --store chroma:///path/to/db#docs --old ... --new ...
```

`rebasis doctor` tells you which of the three off-states you are in, because
"off" on its own sends people looking in the wrong place:

```
telemetry   off (install `pip install "rebasis[otel]"`)
telemetry   off (REBASIS_OTEL_ENABLED=1 enables it)
telemetry   on (no OTEL_EXPORTER_OTLP_ENDPOINT — nothing is exported)
telemetry   on → http://localhost:4318
```

To see spans without a collector at all:

```bash
REBASIS_OTEL_ENABLED=1 REBASIS_OTEL_CONSOLE=1 rebasis probe ...
```

## A collector to point at

[`collector.yaml`](collector.yaml) is a minimal OpenTelemetry Collector config
that accepts OTLP over HTTP and prints what it receives.

```bash
docker run --rm -p 4318:4318 \
  -v "$PWD/collector.yaml:/etc/otelcol/config.yaml" \
  otel/opentelemetry-collector:latest
```

## The span tree

```
rebasis.probe                                (root)
├── rebasis.sample.draw
├── rebasis.embed.corpus                     gen_ai.operation.name=embeddings
│   └── rebasis.embed.batch    ×N            (sampled: first 3, then every 100th)
├── rebasis.groundtruth.knn
├── rebasis.adapter.fit
├── rebasis.adapter.evaluate
└── rebasis.decision                         rebasis.probe.decision=...

rebasis.migrate                              (root, long-running)
├── rebasis.migrate.batch      ×N            (sampled the same way)
│   ├── rebasis.embed.batch
│   └── rebasis.store.upsert
└── rebasis.adapter.refit      ×M
```

Batch spans are sampled on purpose. A 500,000-record migration emits 2,000 batch
spans otherwise, and a trace where the batches outnumber everything else hides
the shape of the run.

## Metrics

| Metric | Type | Unit | Labels |
|---|---|---|---|
| `rebasis.embed.duration` | histogram | ms | model, kind |
| `rebasis.embed.texts` | counter | 1 | model, kind |
| `rebasis.adapter.fit.duration` | histogram | ms | adapter_type |
| `rebasis.adapter.apply.duration` | histogram | µs | adapter_type |
| `rebasis.probe.arr` | gauge | ratio | metric, adapter_type |
| `rebasis.migrate.items` | counter | 1 | state |
| `rebasis.migrate.progress` | gauge | ratio | job_id |
| `rebasis.store.search.duration` | histogram | ms | backend |
| `rebasis.errors` | counter | 1 | error_code, transient |

`rebasis.errors` is labelled by the same stable codes the documentation uses, so
a spike on a dashboard is greppable in
[the error reference](../../docs/reference/errors.md).

## What never appears in a span

No document text, no queries, no vectors, no ids. Span attributes are always
indexed and size-limited, and the GenAI convention says content belongs in span
events rather than attributes for exactly that reason. rebasis records no
content at all, so the question does not arise — but the convention is followed
so the right answer is already in place if anyone ever proposes adding it.

## Correlation

When telemetry is on, log lines carry `trace_id` and `span_id`. That gets you
from a JSON log line to a trace, from the trace to an audit record, and from the
audit record to `rebasis audit replay`.
