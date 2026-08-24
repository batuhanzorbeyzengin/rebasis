# An Obsidian vault in Chroma

## The situation

Four years of notes — 40,000 chunks — indexed with `all-MiniLM-L6-v2` because
that is what the tutorial used. `bge-base-en-v1.5` is better on every benchmark
you can find. Re-embedding the vault is about two hours on a laptop, during
which search does not work, and you are not certain it is worth it.

That uncertainty is the actual problem. The compute is affordable; committing to
it blind is not.

## What to do

```bash
pip install "rebasis[chroma,sentence-transformers]"

rebasis probe \
  --store chroma:///Users/me/vault/.chroma#notes \
  --old sentence-transformers/all-MiniLM-L6-v2 \
  --new BAAI/bge-base-en-v1.5 \
  --sample 8000 \
  --report vault-report.html
```

Ten minutes, most of it embedding the sample. Nothing is written to the vault.

Open `vault-report.html`. It leads with the decision and the reason, then the
number and its confidence interval, then what the alternatives look like.

## Reading the answer

**`bridge_sufficient`** — fit an adapter and use the new model today. The index
stays as it is. This is the outcome that saves the two hours.

**`bridge_and_migrate`** — bridge now, and let rebasis rewrite the index in the
background while search keeps working. You get the new model immediately and
full quality eventually.

**`no_upgrade_needed`** — the benchmark improvement does not show up on *your*
notes. This happens more than people expect, and it is the cheapest possible
outcome. To get this answer you need a real query log (see below); without one
the tool cannot tell you.

**`full_reindex`** — the gap is real and an adapter will not close it. Now you
know the two hours are worth spending.

## Use your actual searches

If you can export what you have searched for, do. It changes the question from
"do these two models agree" to "which one finds what I was looking for".

```jsonl
{"query": "that thing about postgres connection pooling", "relevant": ["notes/db/pooling.md#2"]}
{"query": "rate limit design", "relevant": ["notes/arch/ratelimit.md#0"]}
```

```bash
rebasis probe ... --queries my-searches.jsonl
```

Forty or fifty queries is enough to be worth having. It is also the only way to
get `no_upgrade_needed`.

## Then

```bash
rebasis fit \
  --store chroma:///Users/me/vault/.chroma#notes \
  --old sentence-transformers/all-MiniLM-L6-v2 \
  --new BAAI/bge-base-en-v1.5 \
  --out vault-adapter.rbs
```

And in whatever queries the vault:

```python
from rebasis import Bridge

bridge = Bridge.load("vault-adapter.rbs")
results = collection.query(
    query_embeddings=[bridge.to_index_space(new_model.encode(text)).tolist()],
    n_results=10,
)
```

## One thing to watch

Chroma locks a collection's dimension. `all-MiniLM-L6-v2` is 384-dimensional and
`bge-base-en-v1.5` is 768 — so migrating the index in place is not possible
here, and rebasis will say so before starting rather than halfway through.

Bridging works regardless: the adapter maps the 768-dimensional query down into
the 384-dimensional space the index uses. That asymmetry is the reason this tool
exists.

See [`run.py`](run.py) for the same flow in Python.
