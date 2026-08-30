<p align="center">
  <!-- Absolute, not relative: this README is also the PyPI long description,
       and PyPI cannot resolve a repository-relative path. -->
  <img src="https://raw.githubusercontent.com/batuhanzorbeyzengin/rebasis/main/docs/assets/logo-banner.png"
       alt="rebasis" width="440">
</p>

<h1 align="center">rebasis</h1>

<p align="center">
  <strong>Measure whether an embedding upgrade is worth it, bridge it without reindexing when it is, and migrate safely when you are ready.</strong>
</p>

<p align="center">
  <a href="https://batuhanzorbeyzengin.github.io/rebasis/"><strong>Docs</strong></a> ·
  <a href="#supported-backends"><strong>Backends</strong></a> ·
  <a href="docs/bridge-band.md"><strong>The evidence</strong></a> ·
  <a href="#limits--stated-plainly"><strong>Limits</strong></a> ·
  <a href="ROADMAP.md"><strong>Roadmap</strong></a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python"></a>
  <a href="#status-01"><img src="https://img.shields.io/badge/version-0.1-orange" alt="Version"></a>
  <a href="https://scorecard.dev/viewer/?uri=github.com/batuhanzorbeyzengin/rebasis"><img src="https://api.scorecard.dev/projects/github.com/batuhanzorbeyzengin/rebasis/badge" alt="OpenSSF Scorecard"></a>
</p>

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

It also tells you whether to bother, which is the more useful half:

> Measured across **62 model migrations** on real corpora with human relevance
> judgements: a naive model swap retains **12.5%** of what a full reindex gives
> you; bridging retains **71%**.
>
> rebasis recommended bridging in **12 of those 62**. The other fifty, the
> honest answer was "reindex" or "you did not need this upgrade", and it said so
> instead of selling you the adapter.

The second number is the one that matters. A tool that always recommends itself
is not a measurement. Read it for what it is: how often the tool declined to sell
you the adapter, which is not the same as how often it was *right*. The accuracy
version of that count turned out to be an algebraic identity and
[has been withdrawn](#the-decision-rule).

Read the count itself with one more caveat. **Four of those twelve are runs whose
numbers no longer reproduce**: the four held-out crossings
[section 8](docs/bridge-band.md#the-one-miss-is-where-the-rule-says-it-is-uncertain)
lists were written from an artifact the repository no longer holds, and on the
rows that are here none of the four crosses the break-even at all. So twelve is
the published figure and eight is what the surviving rows support.

---

## What rebasis adds

Aligning two embedding spaces is not new, and rebasis does not claim it is: the
transform is orthogonal Procrustes, published in 1966. What the tool adds is
everything around it, because the alignment is the easy part.

- **It decides whether the upgrade is worth doing at all.** Retention alone
  cannot tell you: an adapter that recovers 94% of a reindex still loses to a 3%
  upgrade. `probe` weighs both and returns a decision — which is sometimes to
  reindex instead, or to stay where you are.
- **It measures against your queries, on your corpus.** Not a benchmark average.
  A real query log is what makes the answer yours; without one the tool says the
  result is provisional rather than guessing.
- **It can rewrite the index too, and it says what that is worth.** Shadow copy
  before every batch, read-back after, a fresh-connection check when the queue
  empties, `rollback` to the originals, five backends on every commit. The
  direction matters and nothing but the manifest records it: `migrate` needs a
  map *out* of the index's space, `Bridge` needs the map *into* it, and handing
  over the wrong one passes every guard while destroying the index. `fit` writes
  either; `migrate` refuses the wrong one. Measured, a completed migration is
  worth about what bridging is worth — which is usually not much. See
  [Status](#status-01).
- **It knows a half-migrated index is a broken one.** A migration stopped
  part-way leaves two embedding spaces in one collection, and no ordinary query
  is correct against both. rebasis detects that, says so unprompted, and can
  serve such an index correctly while the job finishes.

## Where this sits

Two things worth knowing before you evaluate it, both of which cut against the
project as easily as for it.

**No vector database offers this, and each of them says so in its own
documentation.** Where a vendor documents changing the embedding model of an
existing collection, the documented path is to re-embed the corpus. Qdrant's
[migration tutorial](https://qdrant.tech/documentation/tutorials-operations/embedding-model-migration/)
gives two routes — blue-green and named vectors — and both "re-embed the points
using the new embedding model" from text kept in the payload. Weaviate's
[vectorizer migration](https://docs.weaviate.io/weaviate/tutorials/vectorizer-migration)
copies to a new collection where "vectors will be auto-generated". What they
*do* offer, and offer well, is a **zero-downtime cutover** once the re-embedding
is done: a collection alias flipped instantly, with instant rollback. That is a
real answer to a different question. rebasis' question is whether you have to
pay for the re-embedding at all.

**The technique is not this project's, and the strongest published result for it
did not reproduce here.**
[Drift-Adapter](https://aclanthology.org/2025.emnlp-main.805.pdf) (EMNLP 2025)
evaluates the same three adapters on the same problem and reports recovering
**95–99%** of a full re-embedding. Run on its own corpora, its own model pair and
its own fit budget, this project's harness measures **0.24 to 0.50** — and an
adapter-independent ceiling puts the published band above what the old space can
hold. Either the reproduction differs from the paper in a way neither document
has identified, or the published figures do not describe what an end-to-end
retrieval measurement returns. [The workings](docs/vs-drift-adapter.md).

What that means for you: **take this project's own band, not the published one.**
Bridging recovers a mean of about **0.72** of a full reindex, and is worth doing
in roughly one run in five of those measured. If a number sounds better than
that, check which measurement produced it.

**The adapter is weak as a product and stronger as an instrument** — and the
second half of that sentence has now been tested rather than asserted.
Google's [Embedding-Converter](https://aclanthology.org/2025.acl-long.1122.pdf)
(ACL 2025) reaches the same framing from the opposite end: a cheap transform
used to *predict* which model is better, on 11 of 13 datasets. `rebasis compare`
is that claim made explicitly, and it was scored against the null anybody
actually uses — pick whatever tops the published MTEB table. Over 16 corpora
with three candidates each, **the null wins: 14 of 16 against 9 of 16.** The
command ships reporting a ranking and a caveat rather than a winner, and
[which model, on your corpus](docs/model-selection.md) has both numbers, what
the ordering does carry, and how it moves with sample size.

## Status: 0.1

The first released version is `0.1`, and the version number is the honest part
of this section. Two different promises are being withheld, and they are worth
separating:

**The API will change.** Read `0.1 → 0.2` the way you would read a major bump.
The Python API, the `.rbs` adapter format and the decision thresholds may all
move before 1.0. Breaking changes go in the changelog with the reason.

**`migrate` writes to your index, and every run before this release wrote an
index no query could answer.** It was applying the adapter the wrong way round:
`fit` produced a map from the new model's space into the index — what a *query*
needs — and `migrate` rewrites the *documents*, which needs the reverse. Every
guard it had passed. The write landed, the count held, the text survived, the
read-back compared what was written against what came back, and the index health
check measured the store's search against exact kNN over the vectors it now
held. None of those asks whether the vectors still mean anything. Measured on
data where both spaces are known exactly and the bridge itself scores 1.000: the
index a completed migration left behind answered **recall@1 0.000** to a raw
new-model query, a bridged query and an old-model query alike.

Two things changed. `fit --direction old_to_new` produces the map `migrate`
actually needs, fitted and scored on the question a migration asks — what a
*raw* new-model query retrieves from a rewritten index. And `migrate` reads the
direction out of the adapter's manifest and refuses the other one before it
opens the store. `rollback` is untouched and still restores from the shadow
copy, which is the path out for anyone who ran a migration before this.

**What is still withheld is the evidence, not the code.** Nobody has yet run
`migrate` against an index they could not rebuild. Until somebody has: take a
backup rebasis is not part of, and try `--limit` on a slice first —
[knowing what that leaves behind](https://batuhanzorbeyzengin.github.io/rebasis/guides/migration/#stopping-short-leaves-two-spaces-in-one-index).

Everything else — `probe`, `fit`, `eval` — only reads, and always did.

---

## How it works

A small learned transform — orthogonal Procrustes by default, chosen from six
candidates by measurement — maps the new model's query vector into the old
model's space.

Fitting is **independent of corpus size**: it depends only on how many matched
pairs you can produce, and a few thousand is enough for a 500k-chunk vault. The
transform itself fits in seconds; **generating the paired embeddings is the
expensive part**, and how long that takes is a property of your model or API
rather than of rebasis.

```bash
# 0. Optional: which model, before deciding *whether*. Every candidate is
#    scored on one sample, one split and one query set — the index's own model
#    is the reference rather than a row. Read what the ordering is worth first:
#    on 16 corpora it did not beat the published MTEB table on top-1.
rebasis compare --store chroma:///path/to/db#notes \
  --old sentence-transformers/all-MiniLM-L6-v2 \
  --candidates BAAI/bge-small-en-v1.5,BAAI/bge-base-en-v1.5 \
  --queries queries.jsonl

# 1. Diagnose. What do I actually lose if I switch?
rebasis probe --store chroma:///path/to/db#notes \
  --old sentence-transformers/all-MiniLM-L6-v2 --new BAAI/bge-base-en-v1.5 \
  --queries queries.jsonl --sample 10000 --report report.html

# 2. Fit the adapter.
rebasis fit --store chroma:///path/to/db#notes \
  --old sentence-transformers/all-MiniLM-L6-v2 --new BAAI/bge-base-en-v1.5 \
  --method auto --pairs 4000 --out adapters/minilm-to-bge.rbs

# 3. Serve through the bridge. The index is never touched.

# 4. Optional: rewrite the index so the new model can query it directly.
#    This needs the *other* direction — a map out of the index rather than into
#    it — and `fit` produces one on request. `--dry-run` prints the plan first.
rebasis fit --store chroma:///path/to/db#notes \
  --old sentence-transformers/all-MiniLM-L6-v2 --new BAAI/bge-base-en-v1.5 \
  --direction old_to_new --out adapters/forward.rbs

rebasis migrate --adapter adapters/forward.rbs \
  --store chroma:///path/to/db#notes --priority access --limit 5000

# 5. A different question about the same index: what would a *cheaper*
#    representation of it cost? No candidate model and no adapter — the
#    reference is the index's own full-width, float32 state.
rebasis probe --store chroma:///path/to/db#notes \
  --truncate 768,512,256,128 --quantize float32,int8,binary --floor 0.95

# Stop it at a batch boundary; pick it up where it stopped.
rebasis pause <job-id>
rebasis resume <job-id>

# Undo it, from the shadow copy.
rebasis rollback <job-id>
```

**Measured, migrating is worth about as much as bridging and usually worth
less than neither.** A completed migration delivers a mean **0.727** of a full
reindex across 51 runs on real corpora with human judgements — against 0.719 for
bridging, tracking it at Spearman 0.993. It beat leaving the index alone in
**5 of those 51**. What it buys is the adapter leaving the query path; what it
costs is rewriting every vector. It does not buy retrieval quality
([the numbers](docs/migration-band.md)).

`probe` and `fit` print the command to run next.
`rebasis doctor` lists the backends, embedders and devices it can see, and with
`--store <uri>` it asks the same questions of a live index: does it open, what
does it declare it can do, is its SQLite file intact, and — the one that
explains a quality collapse nothing else explains — is it holding two embedding
spaces at once. Read-only in every path. Run it first when anything is
confusing.

### In a script

Every command that prompts takes `--yes` and refuses to guess without it. The
commands that report take `--json`:

```bash
# Gate a deploy on the decision, not on a human reading a table.
decision=$(rebasis probe --store "$STORE" --old "$OLD" --new "$NEW" \
  --queries queries.jsonl --json | jq -r .decision)

# Refuse to serve from an index that is halfway between two models.
rebasis status --json | jq -e 'all(.mixed_space == null)' >/dev/null
```

Progress goes to stderr, so `--json` on stdout stays parseable while you watch
the run move. `rebasis doctor --json` is the thing to attach to a bug report.

## What the measurement says

Measured on **11 corpora, 398,010 documents and 10,346 questions real people
typed**, with human relevance judgements, scored with
[ranx](https://github.com/AmenRa/ranx) rather than with rebasis' own metric code.

### The decision rule

`probe` returns a decision, not a number. What decides is the **break-even**:

```
bridge_advantage = ARR × upgrade_gain
```

— how much of a full reindex the adapter recovers, times how much better the new
model is *on your corpus*. Above 1.0 bridging beats leaving things alone.

**The count that used to follow that sentence was an identity, and it has been
withdrawn.** Read off one run's own
scores, `ARR × upgrade_gain` is `(bridged/reindex) × (reindex/status quo)` —
which is `bridged/status quo`, the same inequality as the outcome it was being
scored against. It agreed 57 times out of 57 on the runs still on disk because it
could not do otherwise. What survives a proper test is narrower and positive: the
estimate ranks runs by the margin they actually returned at **Spearman ρ = 0.60,
p ≈ 1e-6**, so it carries real information about how much bridging will cost or
buy — but it does not support being read as a threshold at the accuracy that
count implied. [The workings](docs/bridge-band.md#9-what-the-counting-is-worth).

Neither factor settles it alone, and they pull against each other: a big upgrade
means the old model was weak, and a weak source space carries less for the
adapter to map. Measured correlation **−0.958**, reproduced at **−0.940** on the
held-out corpora. That is why the band is narrow, and why measuring first is the
whole idea.

**The break-even needs a real query log.** Without `--queries`, rebasis can tell
you how well an adapter bridges but not whether bridging is worth it — so it
says that, rather than guessing. `--synth-queries keywords` estimates it from
your documents when you have no log, and marks the result provisional.

Full workings, including the runs where it was wrong:
[when bridging is worth it](docs/bridge-band.md).

### Three findings worth knowing about

**The band widens if the bridge only has to recall, and `probe` now recommends
that arrangement.** Every number above assumes the bridge produces the final
ranking. If it instead produces a candidate set that the new model reorders in
its own space, it is bounded by recall@N rather than nDCG@10 — a weaker
requirement. Measured over **48 runs on sixteen corpora**: single-stage bridging
beat doing nothing in **1**, a two-stage arrangement in **36** — and in 36 of
the 37 runs where a reindex was genuinely an upgrade.

What kept that out of the recommendation was a price, not a doubt: the
arrangement re-embeds N documents per query, and how many of those are already
cached is a property of your traffic rather than of your corpus. Given
`--queries` it is a property of a file you handed over, and `probe` counts it —
the overlap between your query log's own candidate sets, reported as the lower
bound it is. So a run whose single stage loses now says so **and** names the
arrangement that wins, with the documents-per-query it will cost.
`rebasis.serve.Cascade` serves it. [The measurement](docs/cascade-band.md).

**A default migration costs no measurable retrieval quality.** Rewriting vectors
does not rewrite the graph an index built around them, so it can cost recall
even when every vector is correct. Measured on 100,000 records: with the
orthogonal transform `auto` picks, the index's own recall against exact kNN
moved within measurement noise on every backend. Non-orthogonal transforms
degrade it for real — up to 12 points — so `migrate` measures before and after
and names any drop. [The measurement](docs/index-health.md).

**A half-migrated index answers a third of its queries wrongly, and now says
so.** Measured, an ordinary bridged query against a half-migrated collection
dropped from a hit rate above 0.90 to below 0.65 — silently. `migrate` and
`status` now report it unprompted, `status --json` carries it as `mixed_space`,
and `rebasis.serve.MixedSpaceSearch` restores it to above 0.90 at every stage of
the migration.

### What migrating costs, in one paragraph

Record loss: **none observed** across the migrate-and-rollback cycle that runs
on five backends on every commit — shadow copy before each batch, read-back
after, a fresh-connection check when the queue empties, and `rollback` to
bit-identical originals. That is a tested property, not a guarantee against a
storage bug or a failing disk, which is why the advice above is still to take a
backup rebasis is not part of.

## Supported backends

`probe` needs to read vectors and text. `migrate` also needs to write. Support is
not uniform, and the differences are mostly in what each store *is* rather than
in how much work went into the adapter.

| Store | Read | Text | `migrate` | Filters | Notes |
|---|---|---|---|---|---|
| **pgvector** | ✅ | ◐ | ✅ | ✅ | Your own Postgres. One batch is one **transaction**, so a half-written batch is not a state. Name the columns in the URI; the column *type* decides the rest — `halfvec` rounds past `migrate`'s tolerance, `bit` and `sparsevec` are refused. |
| **Chroma** | ✅ | ✅ | ✅ | ✅ | Tested from 0.5.5 through 1.x, across the Rust rewrite. |
| **LanceDB** | ✅ | ✅ | ✅ | ✅ | Text comes from a column you name in the URI. |
| **sqlite-vec** | ✅ | ✅ | ✅ | — | One file, no server, no dependencies. |
| **Qdrant** | ✅ | ✅ | ✅ | ✅ | Local and server. Local mode locks its folder for the whole command — run a server if you need concurrent readers. |
| **FAISS** | ✅ | ◐ | ◐ | — | An index, not a database — see below. |
| in-memory | ✅ | ✅ | ✅ | — | For tests, and for vectors you already hold. |

Every row above runs the same store contract suite on every commit, plus a
migrate-and-rollback test that checks the vectors really changed, the record
count did not, the text survived and the originals came back. pgvector runs
against a real PostgreSQL stood up as a CI service, not a fake — a suite that
skips a backend reports the same green summary as one that ran it.

**pgvector is the one that changes what the safety story rests on.** Everywhere
else `migrate`'s batch integrity is four mechanisms rebasis built: a shadow copy,
a read-back, a fresh-connection check and `rollback`. On Postgres the batch is a
transaction — it lands whole or it does not exist. The shadow copy stays, because
a transaction rolls back one batch and `rollback <job-id>` rolls back a finished
job three days later. [The guide](docs/guides/pgvector.md) says which layer holds
which.

**FAISS is the one to read twice.** It is an index, not a database: it stores
vectors and returns row numbers, so ids and text stay yours to keep in a
`vectors.meta.json` sidecar, and writing needs an `IndexIDMap2`. Both limits are
declared through the backend's capabilities, so `migrate` refuses at second zero
rather than halfway through.

### Not there yet

**LangChain** and **LlamaIndex** vector stores are bridged and now tested — a
contract suite of duck-typed fakes, one per capability the bridge has to infer,
plus a layer that drives the frameworks' own in-memory stores where they are
installed. Writing it found both bridges declaring capabilities they could not
deliver, and the LangChain bridge no longer passes a similarity score through at
all: LangChain guarantees nothing about whether a higher number means closer,
and several stores flip on a constructor argument. **usearch** has not been
started. A bridge
adapter reaches dozens of stores for the cost of one file, but those interfaces
do not expose `iter_records(with_vectors=True)` and `upsert_vectors` uniformly,
so a bridged store declares honestly restricted capabilities: `probe` and the
bridge work, `migrate` may not. Partial support beats none; *silent* partial
support does not. [ROADMAP.md](ROADMAP.md) has the rest, including what is
deliberately not planned.

**Embedding backends:** `sentence-transformers`, `fastembed` (ONNX, no torch),
`ollama`, any OpenAI-compatible API, `llama-cpp`, or vectors you computed
yourself.

## Limits — stated plainly

Using the tool without knowing what it cannot do costs more than not using it.

- **It does not fix hard drift.** Between very different architectures an adapter
  is not enough. The tool says so rather than hiding it.
- **It needs the index to hand back vectors *and* text.** The pairs an adapter
  is fitted on are the index's own vectors on one side and the same documents
  re-embedded with the candidate model on the other, so **the old model is never
  run** — `fit` does not load it. What losing the old model costs is the
  *decision*: without it a real query log cannot be encoded the way the current
  system encodes it, so there is no `upgrade_gain`, and the run is reported as
  provisional. The case no adapter survives is an index that kept vectors and
  discarded the text they came from.
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
- [`docs/cascade-band.md`](docs/cascade-band.md) — 57 runs on the assumption
  underneath that band, including nine on the hard-negative tasks where the
  squeeze turns out to be much weaker
- [`docs/index-health.md`](docs/index-health.md) — what a migration does to the
  index, on two graph backends and five adapters
- [`docs/mixed-space-fusion.md`](docs/mixed-space-fusion.md) — which of the two
  merges to use while an index holds two spaces, and the case where the answer
  reverses
- [`docs/vs-drift-adapter.md`](docs/vs-drift-adapter.md) — the same three
  adapters against a published result: what reproduced, what did not, and what
  would change the conclusion
- [`docs/related-work.md`](docs/related-work.md) — where this sits in the
  literature, and which neighbouring approaches a user of this tool
  structurally cannot take
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
