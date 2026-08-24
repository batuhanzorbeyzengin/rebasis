# rebasis

**Change the embedding model of your local RAG without deleting the index.**

A better embedding model comes out. Your index was built with the old one. The
standard advice is to re-embed everything — which, on a personal vault or an
agent's accumulated memory, means hours of compute and a window where retrieval
is broken. Often enough it means not upgrading at all.

rebasis fits a small learned adapter that maps the new model's query vectors
into the space your index already uses. The index is not touched. If the adapter
recovers enough of the quality, you can stop there. If it does not, rebasis says
so, and can migrate the index gradually in the background while retrieval keeps
working.

```bash
pip install rebasis
rebasis probe --store chroma:///path/to/db#documents \
              --old sentence-transformers/all-MiniLM-L6-v2 \
              --new BAAI/bge-base-en-v1.5 \
              --report report.html
```

## What `probe` tells you

It does not return a similarity score. It returns a **decision** and the
evidence behind it:

| Decision | What it means |
|---|---|
| `bridge_sufficient` | The adapter recovers essentially everything. Use the new model today; leave the index alone. |
| `bridge_and_migrate` | Bridge now, migrate in the background. Retrieval keeps working throughout. |
| `caution` | Some of the corpus is losing more than the average. Worth looking at before committing. |
| `full_reindex` | An adapter will not close this gap. |
| `no_upgrade_needed` | The new model is not measurably better **on your corpus**. The cheapest upgrade is the one you skip. |

What decides between them is not ARR alone but the **break-even** —
`ARR × upgrade_gain`, the adapter's quality times how much better the new model
actually is on your corpus. Across 62 measured pairs its sign matched the
outcome 61 times, where the ARR bands alone matched 10 of 15 — and 33 of those
62 are corpora the rule was frozen before it ever saw.

It needs a real query log. Without one rebasis reports how well an adapter
bridges and declines to recommend, because that is all it can honestly do.

See [when bridging is worth it](bridge-band.md) for the measurement.

## What it does not do

- **It does not send anything anywhere.** No telemetry, no phoning home. The
  OpenTelemetry extra exists so *you* can send data to *your own* collector; it
  is off unless you turn it on.
- **It does not log your documents.** Log redaction is an allowlist, not a
  denylist: a field has to be explicitly permitted to appear. Reports carry
  chunk ids and metrics, never text.
- **It does not take ownership of your data.** `probe` and `fit` are read-only
  and work against a read-only filesystem. `migrate` is the only command that
  writes, it only upserts, and it never deletes.

## Where to go next

- [Getting started](getting-started.md) — the five-minute version
- [What drift is](concepts/drift.md) — why this works at all
- [The decision rule](concepts/decision-rule.md) — how the bands were chosen
- [Migration and rollback](guides/migration.md) — the command that writes
