# Roadmap

What is here, what is not, and what would have to be true before 1.0.

This file is a statement of intent, not a schedule. There are no dates because
there is one maintainer and no promises worth making about calendar time. What
each item does carry is the reason it is not done yet, which is the part that
actually helps you decide whether to depend on this.

Two rules the project holds itself to, and this document with it:

- **Nothing is claimed that has not been measured.** Every number below comes
  from a run whose output is in `docs/`. Where something is expected rather than
  measured, it says so.
- **Partial support beats none; silent partial support does not.** A backend or
  a feature that half works declares what it cannot do, at the moment you ask
  rather than halfway through.

---

## Where 0.1 stands

**Working and measured.** `probe` answers "what does switching cost me on *my*
corpus" against a real index, `fit` produces an adapter, `eval` scores one, and
`migrate` rewrites the index with `rollback` behind it. The decision rule has
been right 61 times out of 62 on real corpora with human relevance judgements —
32 of those 33 on corpora it was frozen before it ever saw
([the evidence](docs/bridge-band.md)).

**Backends.** Chroma, LanceDB, sqlite-vec, Qdrant, FAISS and in-memory all run
the same store contract suite and a migrate-and-rollback test on every commit.

**Not proved at scale.** Everything above is tested on hundreds of records, not
millions. Nobody has yet pointed `migrate` at an index they could not rebuild.
That is the single largest gap between 0.1 and something you should trust
unsupervised, and no amount of unit testing closes it — it needs somebody's real
vault and a backup.

---

## 0.2 — finish what is half-built

Each of these exists in the codebase as working, tested code that nothing calls.
They were built ahead of the thing that would use them; connecting them is the
work.

| | State | What is missing |
|---|---|---|
| **Continuous refit during migration** | `migrate/refit.py` is complete, with unit tests and a win-only adoption guard | The engine never calls it. It needs to re-embed already-migrated records to build the accumulated pairs, which means the engine needs an embedder and a `--refit` flag. |
| **LangChain / LlamaIndex bridges** | adapters written, exported, documented | **No tests at all.** They duck-type a foreign object, which is exactly the code that breaks quietly on a dependency bump. |
| **Sample and embedding cache** | `.rebasis/cache/` is defined and garbage-collected | Nothing ever writes to it, so every `probe` re-embeds from scratch. The most visible everyday cost in the tool. |
| **`pause` / `resume` commands** | `migrate --resume <job-id>` works | There is no way to pause a running job from outside it. |
| **`doctor` against a live index** | `doctor` reports devices, backends and thread counts | It takes no store URI, so the checks that need one do not exist: encoding-profile mismatch, chunking drift, the "you do not need an adapter, truncate instead" advice for Matryoshka models, and a SQLite integrity check. |

## 0.3 — the parts that need a measurement first

These are not blocked on code. They are blocked on not yet knowing the right
answer, and guessing would be worse than waiting.

- **`old → new` direction (virtual backfill).** The adapter format already
  carries a direction field; only `query_to_old` is ever produced. Mapping old
  vectors forward instead is a different trade-off, and which one wins has not
  been measured.
- **`--shadow-precision fp16`.** Halves the shadow copy's disk cost and gives up
  the bit-identical rollback guarantee. A half-guarantee may be more dangerous
  than no guarantee; the plumbing is in place and the option is deliberately not
  exposed until that is settled.
- **Local threshold calibration.** The GPU/CPU decisions in `compute/` come from
  measurements on two machines. Whether they hold on yours is unknown, and
  `doctor` should be able to find out rather than assume.
- **Access-log-weighted sampling for `probe`.** The sampler supports weights;
  nothing passes them. Weighting the sample by what people actually read would
  make ARR describe the queries that matter, but it also makes the sample
  non-uniform in a way the confidence interval does not currently model.

## Beyond 0.3 — where the headroom actually is

The measurement that shapes everything after 0.2 is this: **retention is not
improvable by fitting harder.** Six times the fit data buys 0.005–0.025 ARR, and
centred Procrustes beat a residual MLP 15 times out of 15. Gain and retention
anti-correlate at −0.958, reproduced at −0.940 on held-out corpora — a bigger
upgrade means the old model was weaker, and a weaker source space carries less
for any adapter to map ([the squeeze, and the held-out runs](docs/bridge-band.md)).

So the interesting work is not a better fit of the same shape. It is a different
shape:

- **Per-cluster adapters.** One global map is leaving quality on the table where
  drift is heterogeneous. `probe` already reports `tail_arr` to detect that; it
  cannot yet do anything about it.
- **Chained adapters.** v1 → v2 → v3 without a full refit at each step. Error
  accumulation across a chain has not been measured, and refitting against the
  original is probably more accurate — which is worth knowing rather than
  assuming.
- **Matryoshka shortcut.** For models trained with nested representations, the
  right answer may be "truncate and renormalise", with no adapter at all.
- **`vec2vec`-style unpaired alignment.** The one direction that would remove the
  hardest limit in the tool: today, if you cannot run the old model any more, no
  adapter can be fitted at all.

## Before 1.0

1.0 means the API and the `.rbs` format stop moving. That needs:

- **`migrate` proved at production scale**, by someone other than the author, on
  an index that matters.
- **The `.rbs` format settled.** Schema 1 is the only version and there are no
  migrations yet, deliberately — the first migration is the worst possible time
  to design the migration machinery.
- **Full coverage on the modules where a bug costs data.** `audit/chain.py` is
  there (100%). `storage/shadow.py` is at 93.5% against a target of 95, and
  `storage/atomic.py` at 92.5% against 100. Overall coverage is 81.8% against a
  floor of 75, enforced in CI, where `tools/check_coverage_floors.py` also
  checks the per-module targets against the finished report.
- **More than one operating system in CI.** Linux and macOS run today; the
  storage layer has documented Windows-specific behaviour that nothing exercises.
- **A decision rule that has not moved for a release.** It changed twice on
  evidence and then held across 33 held-out runs. One more release without a
  change and the thresholds can be called settled.

## Explicitly not planned

Saying no is part of a roadmap.

- **A server, a daemon, or a hosted service.** rebasis is a command you run on
  your own machine against your own index. There is no phone-home and no usage
  counter, and there will not be.
- **Becoming a vector database.** It reads and writes yours.
- **Approximate ground truth.** The kNN that produces the ground truth is exact.
  An approximate index would make the measurement cheaper and the number
  meaningless.
- **Recommending itself by default.** Bridging is worth doing in about one run in
  five of those measured. A tool that always says yes is not a measurement, and
  the day this one starts doing that is the day to stop trusting it.

## Contributing to any of this

The 0.2 list is the best place to start: each item is a connection between two
pieces that already exist and already have tests. Adding a store or an embedder
is three steps and the contract suite covers the rest — see
[CONTRIBUTING.md](CONTRIBUTING.md).

If you run `migrate` against something real, that is the most useful thing
anybody can contribute right now. Say what happened either way.
