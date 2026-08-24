# Getting started

## Install

```bash
pip install rebasis
```

The core install is deliberately small: numpy, scipy, and nothing that needs a
GPU. Add what you actually use:

```bash
pip install "rebasis[chroma]"                  # your store
pip install "rebasis[sentence-transformers]"   # your embedder
pip install "rebasis[all]"                     # everything
```

Check what rebasis can see:

```bash
rebasis doctor
```

It lists the store backends and embedders it found, the devices available, and
whether telemetry is on. Run it first when anything is confusing — it is the
command written to work in the most broken environment.

## Measure before you change anything

```bash
rebasis probe \
  --store chroma:///path/to/db#documents \
  --old sentence-transformers/all-MiniLM-L6-v2 \
  --new BAAI/bge-base-en-v1.5 \
  --sample 10000 \
  --report report.html
```

This reads your index and never writes to it. It samples the corpus, re-embeds
the sample with the candidate model, fits several adapters, measures each one
against what a full reindex would have returned, and prints a decision with a
confidence interval.

The `--report` suffix picks the format: `.html` gives a self-contained page,
anything else gives Markdown.

### The number to read is not the only number

`probe` prints ARR — the fraction of a full reindex's results the adapter
recovers — together with its 95% confidence interval. **When the interval spans
a decision boundary, the measurement has not settled the question.** rebasis
says so rather than rounding to a recommendation. Raising `--sample` narrows it.

### If you have a query log, use it

```bash
rebasis probe ... --queries queries.jsonl
```

One JSON object per line:

```json
{"query": "how do I rotate the keys", "relevant": ["chunk-8842"]}
```

Without it, held-out documents stand in for queries. That is a reasonable
default and a real limitation — it assumes your queries resemble your documents.

More importantly, without queries `probe` cannot measure whether the new model is
actually **better** on your corpus, and that is half the decision. A run with no
upgrade estimate is reported as **provisional**: it says how well the adapter
bridges and declines to recommend acting on it.

If you have no query log at all, `--synth-queries keywords` builds queries from
the documents and produces a rough estimate. It carries about 0.22 of error
against what a real log gives, so it settles clear-cut cases and says so when it
cannot settle a close one.

```bash
rebasis probe ... --synth-queries keywords
```

## Use the new model today

```bash
rebasis fit \
  --store chroma:///path/to/db#documents \
  --old sentence-transformers/all-MiniLM-L6-v2 \
  --new BAAI/bge-base-en-v1.5 \
  --out adapter.rbs
```

Then in your retrieval code:

```python
from rebasis import Bridge

bridge = Bridge.load("adapter.rbs")
vector = new_model.encode(query)
results = store.search(bridge.to_index_space(vector), k=10)
```

`to_index_space` is on the hot path of every query, so it does nothing but the
matrix multiply — no logging, no validation, no dictionary copies. Validation
happens once, in `Bridge.load`.

## Migrate when you are ready

```bash
rebasis migrate --adapter adapter.rbs --store chroma:///path/to/db#documents
```

Shows what it will do, asks, then rewrites the index in batches. Every batch is
copied to a shadow store before it is overwritten, so
`rebasis rollback <job-id>` restores the original vectors. Interrupt
it and `--resume <job-id>` picks up where it stopped: the queue *is* the
checkpoint.

See [Migration and rollback](guides/migration.md) for what happens when things
go wrong.
