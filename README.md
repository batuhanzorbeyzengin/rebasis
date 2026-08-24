# rebasis

**Change the embedding model of your local RAG without deleting the index.**

[**Docs**](https://batuhanzorbeyzengin.github.io/rebasis/) · [**Backends**](#supported-backends) · [**The evidence**](docs/bridge-band.md) · [**Limits**](#limits--stated-plainly) · [**Roadmap**](ROADMAP.md)

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![Version](https://img.shields.io/badge/version-0.1-orange)](#status-01)

A better embedding model comes out. Your vault, codebase or agent memory is
indexed with the old one, the two coordinate systems have nothing to do with
each other, and every tool you have says the same thing: reindex from scratch.

rebasis fits a small adapter that maps the **new** model's query vectors into
the space your index already uses. The index is never touched.

```python
from rebasis import Bridge

bridge = Bridge.load("adapters/minilm-to-bge.rbs")

q = new_model.encode(["how do I deploy?"])  # meaningless in the old index
q = bridge.to_index_space(q)  # now it means something

results = collection.query(query_embeddings=q, n_results=10)
```

That is the whole integration: one line, in front of the query you already run.

It also tells you whether to bother — which, measured honestly, is **not most of
the time**. That is the other half of the tool, and the more useful half.

---

## Status: 0.1

The first released version is `0.1`, and the version number is the honest part
of this section. Two different promises are being withheld, and they are worth
separating:

**The API will change.** Read `0.1 → 0.2` the way you would read a major bump.
The Python API, the `.rbs` adapter format and the decision thresholds may all
move before 1.0. Breaking changes go in the changelog with the reason.

**`migrate` writes to your index.** It is the one command that does. It is
covered by tests against every supported backend on every commit, it copies each
batch before it overwrites it, it verifies the write on a fresh connection when
it finishes, and `rollback` restores the originals from a bit-identical shadow
copy — exactly, unless your store rewrites what it is given, and then to within
its own float32 round trip. But nobody has
yet run it against an index they could not rebuild. Until somebody has: take a
backup that rebasis is not part of, and try `--limit` on a slice first.
Everything else — `probe`, `fit`, `eval` — only reads.

---

## Supported backends

`probe` needs to read vectors and text. `migrate` also needs to write. Support is
not uniform, and the differences are mostly in what each store *is* rather than
in how much work went into the adapter.

| Store | Read | Text | `migrate` | Filters | Notes |
|---|---|---|---|---|---|
| **Chroma** | ✅ | ✅ | ✅ | ✅ | Tested from 0.5.5 through 1.x, across the Rust rewrite. |
| **LanceDB** | ✅ | ✅ | ✅ | ✅ | Text comes from a column you name in the URI. |
| **sqlite-vec** | ✅ | ✅ | ✅ | — | One file, no server, no dependencies. |
| **Qdrant** | ✅ | ✅ | ✅ | ✅ | Local and server. Local mode locks its folder; rebasis releases it. |
| **FAISS** | ✅ | ◐ | ◐ | — | An index, not a database — see below. |
| in-memory | ✅ | ✅ | ✅ | — | For tests, and for vectors you already hold. |

Every row above runs the same store contract suite on every commit, plus a
migrate-and-rollback test that checks the vectors really changed, the record
count did not, the text survived and the originals came back.

**FAISS is the one to read twice.** It stores vectors and returns row numbers;
ids, text and metadata stay yours to keep. rebasis expects a `vectors.meta.json`
sidecar next to the index, and can only write to an index wrapped in
`IndexIDMap2`. Both limits are reported through the backend's capabilities, so
`migrate` refuses at second zero rather than halfway through.

### Not there yet

Listed because they are planned, and marked because they are not finished. None
of these is ready to point at an index you care about.

| | Status | What is missing |
|---|---|---|
| **LangChain** vector stores | code written, **no tests** | The adapter wraps the store object you already have and duck-types its methods, which is the code most likely to break quietly on a dependency bump — and nothing yet catches that. [Guide](docs/guides/langchain.md). |
| **LlamaIndex** vector stores | code written, **no tests**, read-only | Same adapter, and it declares no write capability, so `migrate` refuses it by design rather than by accident. |
| **usearch** | not started | The second "bare index plus side metadata" backend, alongside FAISS. |

What comes after these, and what is deliberately not planned:
[ROADMAP.md](ROADMAP.md).

A bridge adapter is worth having because it reaches dozens of stores for the
cost of one file. The catch is that those interfaces do not offer
`iter_records(with_vectors=True)` and `upsert_vectors` uniformly, so a bridged
store reports honestly restricted capabilities: `probe` and the bridge work,
`migrate` may not. Partial support beats none; silent partial support does not.

**Embedding backends:** `sentence-transformers`, `fastembed` (ONNX, no torch),
`ollama`, any OpenAI-compatible API, `llama-cpp`, or vectors you computed
yourself.

## How it works

A small learned transform — orthogonal Procrustes by default, chosen from six
candidates by measurement — maps the new model's query vector into the old
model's space.

Fitting it is **independent of corpus size**: it depends only on how many
matched pairs you can produce. For a 500k-chunk vault, embedding a few thousand
chunks with the new model is enough. Fitting takes seconds, not hours.

```bash
# 1. Diagnose. What do I actually lose if I switch?
rebasis probe --store chroma:///path/to/db#notes \
  --old sentence-transformers/all-MiniLM-L6-v2 --new BAAI/bge-base-en-v1.5 \
  --queries queries.jsonl --sample 10000 --report report.html

# 2. Fit the adapter.
rebasis fit --store chroma:///path/to/db#notes \
  --old sentence-transformers/all-MiniLM-L6-v2 --new BAAI/bge-base-en-v1.5 \
  --method auto --pairs 4000 --out adapters/minilm-to-bge.rbs

# 3. Optional: rewrite the index in the background, a batch at a time.
#    --dry-run prints the plan and stops, which is the first thing to run.
rebasis migrate --adapter adapters/minilm-to-bge.rbs \
  --store chroma:///path/to/db#notes --priority access --limit 5000

# Undo it, from the shadow copy.
rebasis rollback <job-id>
```

`probe` and `fit` print the command to run next, so you do not have to come back
here for step two. `migrate` shows an `X of Y` bar while it runs, and resumes
from where it stopped with `rebasis migrate --resume <job-id>` and nothing else.

`rebasis doctor` lists the backends, embedders and devices it can see. Run it
first when anything is confusing.

### In a script

Every command that prompts takes `--yes`, and refuses to guess without it rather
than hanging or crashing. The commands that report take `--json`:

```bash
# Gate a deploy on the decision, not on a human reading a table.
decision=$(rebasis probe --store "$STORE" --old "$OLD" --new "$NEW" \
  --queries queries.jsonl --json | jq -r .decision)

rebasis migrate --adapter adapter.rbs --store "$STORE" --yes
rebasis status --json | jq -r '.[] | select(.state=="paused") | .job_id'
```

Progress and diagnostics go to stderr, so `--json` on stdout stays parseable
while you still see the run move. `rebasis doctor --json` is the thing to attach
to a bug report.

## What the measurement says

Measured on **11 corpora, 398,010 documents and 10,346 questions real people
typed**, with human relevance judgements, scored with
[ranx](https://github.com/AmenRa/ranx) rather than with rebasis' own metric code.

- **Bridging retrieves 6.4× what swapping the model naively gives.** A naive
  swap keeps 12.5% of what a full reindex would give you. Bridging keeps 71%.
- **Bridging is worth doing about one time in five** — 12 of the 62 measured
  pairs. The other four-fifths, the honest answer is "reindex" or "you did not
  need this upgrade", and rebasis says so instead of selling you the adapter.
- **The rule that decides has been right 61 times out of 62**, including 32 of
  33 on corpora it was frozen before it ever saw.

That second bullet is the point. A tool that always recommends itself is not a
measurement.

### The decision rule

`probe` returns a decision, not a number. What decides is the **break-even**:

```
bridge_advantage = ARR × upgrade_gain
```

— how much of a full reindex the adapter recovers, times how much better the new
model is *on your corpus*. Above 1.0 bridging beats leaving things alone.

Neither factor settles it alone, and in practice they pull against each other: a
big upgrade means the old model was weak, and a weak source space carries less
for the adapter to map. Measured correlation **−0.958**, reproduced at **−0.940**
on the held-out corpora. That is why the band is narrow, and why measuring first
is the whole idea.

**The break-even needs a real query log.** Without `--queries`, rebasis can tell
you how well an adapter bridges but not whether bridging is worth it — so it says
that, rather than guessing. `--synth-queries keywords` estimates it from your
documents when you have no log, and marks the result provisional.

Full workings, including the runs where it was wrong:
[when bridging is worth it](docs/bridge-band.md).

## Limits — stated plainly

Using the tool without knowing what it cannot do costs more than not using it.

- **It does not fix hard drift.** Between very different architectures an adapter
  is not enough. The tool says so rather than hiding it.
- **It needs paired data.** If you can no longer run the old model — weights
  lost, API shut down — no adapter can be fitted, and a full reindex is the only
  option left.
- **It does not fix chunking changes.** Different chunk boundaries are corpus
  drift, not embedding drift. Different problem.
- **Fixed similarity thresholds break.** If you filter on `similarity > 0.7`,
  retune it after bridging. An adapter preserves ranking, not the absolute score
  scale. (Calibration closes most of the gap; `docs/m0-findings.md` has the numbers.)
- **The measurement has its own uncertainty.** ARR is computed from a sample, and
  results near a threshold are reported as borderline rather than rounded to a
  side.

## Installation

```bash
pip install rebasis                            # core — no torch, CPU only
pip install "rebasis[chroma]"                  # plus a store backend
pip install "rebasis[sentence-transformers]"   # plus an embedding backend
pip install "rebasis[all]"                     # every backend that ships a wheel
```

The core install has no torch and no heavyweight dependencies. GPU is optional
and accelerates exactly one thing — generating embeddings, measured at 25–40× —
so install torch yourself if you want it:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Shell completion, for the store URIs and model ids nobody types correctly twice:

```bash
rebasis --install-completion
```

## Privacy

**rebasis sends no data anywhere.** No phone-home, no usage counter.

Logs never print document text or vectors: text can be reconstructed from an
embedding, which makes a vector in a log file as sensitive as plaintext. Audit
records store parameters, never content. Optional OpenTelemetry support
(`rebasis[otel]`) exports to **your own** collector and is off by default.

## Documentation

rebasis' complete documentation is at
**[batuhanzorbeyzengin.github.io/rebasis](https://batuhanzorbeyzengin.github.io/rebasis/)**
— getting started, the concepts, a guide per store, and the full CLI and API
reference.

- [`examples/`](examples/) — an Obsidian vault in Chroma, a codebase in LanceDB,
  an OpenTelemetry setup
- [`docs/bridge-band.md`](docs/bridge-band.md) — every number above, with the
  runs that produced it
- [`docs/adr/`](docs/adr/) — the decisions that would otherwise be re-argued,
  and the measurement behind each
- [`docs/m0-findings.md`](docs/m0-findings.md) — 84 configurations measured
  against the design's own assumptions, and the 21 places they disagreed

## Contributing

Adding a store or an embedder is three steps: write the file, add the entry
point, make the contract suite pass — it runs against every backend, so a new
one inherits the whole thing. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0
