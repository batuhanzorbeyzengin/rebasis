# A codebase indexed in LanceDB

## The situation

A repository chunked by function and indexed with a general-purpose embedding
model. A code-specialised model exists and should, in principle, be better at
"where do we validate the auth token" than a model trained mostly on prose.

Unlike the vault case, the question here is not "is re-embedding worth two
hours". It is whether the specialised model is better **on this repository** —
and code retrieval is where general benchmarks are least likely to transfer,
because your codebase's vocabulary is your own.

This is the case where a query log matters most.

## Probe with real queries

```bash
pip install "rebasis[lancedb,sentence-transformers]"

rebasis probe \
  --store lancedb:///data/code-index#chunks \
  --old sentence-transformers/all-mpnet-base-v2 \
  --new BAAI/bge-base-en-v1.5 \
  --queries dev-queries.jsonl \
  --sample 12000 \
  --report code-report.md
```

`dev-queries.jsonl` is questions people actually asked, with the chunk that
answered each one:

```jsonl
{"query": "where do we validate the auth token", "relevant": ["src/auth/verify.py#12"]}
{"query": "retry logic for the payment webhook", "relevant": ["src/webhooks/payment.py#4"]}
{"query": "how is the rate limiter configured", "relevant": ["src/middleware/ratelimit.py#0"]}
```

Forty of these are worth more than four thousand document proxies, because they
carry what your team means by a question.

## The outcome worth waiting for

With a real query log, `probe` can compute **upgrade gain**: how much better the
new model retrieves your corpus than the current one does. If that is not
meaningfully above 1.0, the report says `no_upgrade_needed` — the specialised
model is not actually better here, and the whole question is moot.

That answer is only reachable at this tier. Without queries, the ground truth
*is* the new model's output, so it scores perfectly against itself and the tool
cannot tell you.

## Watch the tail

Code corpora are heterogeneous in a way note vaults usually are not. Tests,
generated protobuf, vendored dependencies and hand-written business logic sit in
very different regions of the space, and one global adapter can score 0.93
overall while doing much worse on one of them.

rebasis stratifies its sample by cluster and reports the gap between the overall
figure and the sparsest clusters. When that gap is large the decision comes back
as `caution` with a note, and the fix is usually per-cluster adapters rather
than a reindex.

## Migrating

LanceDB does not lock the dimension, so unlike the Chroma case a full in-place
migration is available:

```bash
rebasis fit --store lancedb:///data/code-index#chunks \
            --old sentence-transformers/all-mpnet-base-v2 \
            --new BAAI/bge-base-en-v1.5 \
            --out code-adapter.rbs

rebasis migrate --adapter code-adapter.rbs \
                --store lancedb:///data/code-index#chunks \
                --priority access --access-log retrieval-log.jsonl \
                --limit 20000
```

`--priority access` migrates the chunks that actually get retrieved first, so
quality improves where anyone will notice it. `--limit` migrates a chunk of the
work at a time; the queue is the checkpoint, so the next run resumes exactly
where this one stopped.

See [`run.py`](run.py) for the same flow in Python, including the tail check.
