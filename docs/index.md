# rebasis

**Measure whether an embedding upgrade is worth it, bridge it without reindexing when it is, and migrate safely when you are ready.**

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
actually is on your corpus.

**The count this page used to quote here has been withdrawn.** "61 of 62" was
an identity: read off one run's own scores, `ARR × upgrade_gain` is
`(bridged / reindex) × (reindex / status quo)`, which is the same inequality as
the outcome it was being scored against, so it could not have disagreed.
What survives that check is a rank correlation rather than an accuracy — over
the 57 runs the repository still holds, the estimate orders them by the margin
they actually returned at **Spearman ρ = +0.60, p ≈ 1e-6**
([section 9](bridge-band.md#9-what-the-counting-is-worth)). The quantity still
decides and the bands are still where they were; what changed is how much
confidence a reader should take from a proportion.

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
  writes, it only upserts, and it never deletes — not even a metadata field of
  its own. When it needs to know which records it has already moved, it reads
  its own manifest rather than tagging yours.

## Three more questions it answers

`probe` asks whether to change the model. Three commands ask adjacent questions
about the same index, from the same sample, without rebuilding anything.

| | question |
|---|---|
| [`rebasis compare`](reference/cli.md#rebasis-compare) | **which** model, out of several — an ordering rather than a verdict. Read [what the ordering is worth](model-selection.md) first: on sixteen corpora it did not beat the published leaderboard. |
| [`probe --truncate`](truncation-band.md) | what a **cheaper representation** of this index would cost. No model change and no adapter — the same vectors held more narrowly. int8 turns out to be free; binary is not, until the full-precision vectors reorder its candidates. |
| [`rebasis expose`](exposure.md) | how **alignable** this index is to a space somebody else already has. Returns a number and no translation. |

## Where to go next

- [Getting started](getting-started.md) — the five-minute version
- [What drift is](concepts/drift.md) — why this works at all
- [The decision rule](concepts/decision-rule.md) — how the bands were chosen
- [Migration and rollback](guides/migration.md) — the command that writes
- [When bridging is worth it](bridge-band.md) — 62 runs, and the one it got wrong
- [The bridge as a recall stage](cascade-band.md) — 57 runs on the assumption
  underneath that band, including nine on the hard-negative tasks where the
  squeeze turns out to be much weaker
- [Which model, on your corpus](model-selection.md) — 16 corpora, and the
  measurement where the published leaderboard beat the tool
- [What a cheaper index costs](truncation-band.md) — 48 grids, and why int8 is
  free while a deep truncation is not
- [How alignable an index is](exposure.md) — what `rebasis expose` measures, and
  the four things it does not say
- [What a migration does to the index](index-health.md) — the check the read-back
  cannot do
- [Merging two embedding spaces](mixed-space-fusion.md) — serving an index that
  is halfway between two models
- [Against a published result](vs-drift-adapter.md) — the same three adapters,
  measured against the paper that reports 95–99%
- [What a completed migration is worth](migration-band.md) — 51 runs, and why
  it is not distinguishable from bridging
- [Refitting during a migration](continuous-refit.md) — 216 cells, and the one
  case where it earns its cost
- [Related work](related-work.md) — and the door that is closed
