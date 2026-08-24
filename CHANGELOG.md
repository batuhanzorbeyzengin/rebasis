# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/) and this
project uses [Semantic Versioning](https://semver.org/).

Entries below the marker are assembled by `towncrier` from news fragments in
`changelog.d/`, one per user-visible change. This file is not edited by hand.

**Behaviour changes lead every release.** When a decision threshold or a metric
definition moves, the fragment carries the reasoning and the measurement, and it
appears above the feature list — because that is the part a reader upgrading
needs first.

<!-- towncrier release notes start -->

## Before the first release

The work below predates the news-fragment convention, so it was written by hand
rather than assembled. Everything after it is generated.

### Added

**Measurement**
- M0 spike harness and findings (`docs/m0-findings.md`): 4 corpora × 3 model
  pairs × 7 adapters = 84 configurations, closing five open questions and
  proposing 21 corrections to the technical design.

**Core package**
- `types` — shared types, the `FloatArray` alias, and `EncodingProfile` with the
  fingerprint that makes a wrong adapter structurally unloadable.
- `errors` — the full hierarchy with stable `RB-Exxxx` codes, retry eligibility
  and documented exit codes.
- `observability` — 41-event catalogue, allowlist redaction, the environment and
  level matrix, the structlog processor chain, and an OpenTelemetry layer that
  is a no-op when the extra is absent.
- `compute` — device abstraction with defensive detection; torch is imported
  lazily so `import rebasis` never pulls it in.
- `storage` — atomic writes with the three-step fsync discipline, backup
  rotation, space pre-checks and integrity hashing.
- `core` — Procrustes (plain and centred), ridge affine, low-rank affine,
  diagonal scaling, residual MLP, CSLS, isotonic calibration, `auto` selection,
  and the `.rbs` format with schema versioning.
- `sample` — stratified k-means sampling with a per-cluster floor, and the
  disjoint fit/query split that leakage checking depends on.
- `store` — the vector store protocol, URI parsing, entry-point registry, and
  an in-memory reference backend.
- `embed` — encoding profile registry with 14 known models, and the
  precomputed and sentence-transformers backends.
- `probe` — metrics, both ground-truth tiers, the decision rule, and the runner
  that ties them together.
- `cli` — `probe`, `doctor` and `version`, with centralised error rendering.

**Stores and serving (M2)**
- `serve` — `Bridge`, the serving-time API; hybrid search with calibrated merge
  and RRF fallback; `wrap_retriever` for one-line adoption. This layer never
  imports torch, enforced by a runtime test.
- `store` — Chroma and LanceDB backends, plus LangChain and LlamaIndex bridges.
  The bridges declare their capabilities honestly: `probe` and the bridge phase
  work through them, `migrate` refuses up front rather than failing halfway.
- `compute` — `TorchBackend` for cuda/mps with OOM recovery that halves the
  batch rather than killing the job, the three documented MPS traps handled at
  the boundary, and TF32 control on measurement paths.
- Device parity contract tests across every available device, asserting that
  numerical drift never changes the recommendation.

**Migration, audit and durability (M3)**
- `manifest` — SQLite state with WAL plus `synchronous=FULL`. The weaker
  settings are not enough here: the file holds audit records, and manifest
  writes are per-batch rather than per-record, so the fsync cost is unmeasurable
  against a batch's embedding time.
- `audit` — record schema, hash chain, writer and reader. Tamper-evident,
  not tamper-proof, and the distinction is documented rather than glossed.
- `storage` — file locking, append-only shadow copies with segment hashing, the
  pre-migration space and time budget, and dry-run-by-default garbage
  collection.
- `migrate` — job engine with a durable state machine, checkpointing and resume,
  priority ordering, power awareness, the memory watchdog with adaptive
  batching, sampled read-back verification and bit-identical rollback.
- `audit replay`: re-run a recorded decision and compare. Across devices an
  equivalence band is used and the report says so, because identical seeds do
  not guarantee identical results between CPU and GPU.
- Continuous re-fitting: as a migration accumulates matched pairs, the adapter
  is periodically refitted and adopted **only when it measurably beats** the one
  in use on a held-out set. Off by default.
- CLI — `migrate`, `status`, `rollback`, `gc` and the `audit` subcommands.

**Polish, telemetry and the performance gate (M4)**
- `probe`, `fit`, `eval`, `migrate` and `rollback` are wired to live stores.
  `probe/session.py` is the layer that was missing: it draws a sample from a
  real index, reads its vectors, and re-embeds its text — reservoir-sampling the
  clustering pool so peak memory stays a function of the pool rather than of the
  corpus.
- `audit replay` re-runs a recorded decision against the live store and compares
  it, using the cross-device equivalence band where the devices differ. It exits
  3 on a difference, so a script can branch on it: a difference means either a
  regression or a changed corpus, and both are worth noticing.
- `report` — Markdown and single-file HTML reports that lead with the decision
  and carry the caveats beside the number. The HTML makes no external request: a
  report about a private corpus should not tell anyone it exists.
- `store` — sqlite-vec and Qdrant backends. Both run against real databases in
  CI without a server: sqlite-vec through its extension, Qdrant in local mode.
- `observability` — the `[otel]` extra now configures a real SDK when
  `REBASIS_OTEL_ENABLED` is set, exports to the user's own endpoint, and emits
  the documented span tree and metric set. Batch spans are sampled the way batch
  logs are. Off by default, and a foreign `OTEL_EXPORTER_OTLP_ENDPOINT` does not
  turn it on.
- `memory://` gained a file-backed form (`memory:///corpus.npz`) that writes
  through atomically, making the reference backend usable from the CLI and the
  end-to-end tests.
- Generated `docs/reference/profiles.md`, alongside the events and error
  catalogues.
- The mkdocs-material site, and three worked examples: an Obsidian vault in
  Chroma, a codebase in LanceDB, and an OpenTelemetry collector setup.
- Performance test layers 2, 3 and 4: memory ceilings, the scaling invariant,
  and macro benchmarks against the performance budgets. CI gains a CodSpeed
  job, a memory-ceiling job that blocks, and a strict docs build.

**Tooling**
- Generated reference pages: `docs/reference/events.md`,
  `docs/reference/errors.md` and `docs/reference/profiles.md`
  (`just docs-gen`).
- Enforced: ruff (`ALL` with justified ignores), mypy `--strict`, import-linter
  layer contracts, pytest marker layers.
- CI: lint, a Python 3.12/3.13 matrix, and three jobs that stop optional
  dependencies becoming mandatory — `no-torch`, `lowest-direct` and a secret
  scan.

**Golden corpora and the defects they found**
- `tests/golden/` and `tools/make_golden.py`: four model pairs over
  BEIR/scifact with 295 human-judged queries, stored as vectors so the tests run
  with no model download and no network. The first time the package produced
  decisions about real embeddings — recorded in `docs/golden-findings.md`.
- Two models added to the profile table, with dimensions measured rather than
  assumed: `all-MiniLM-L12-v2` (384) and `minishlab/potion-base-8M` (256).

### Fixed

- **ARR's confidence interval was an interval for a different quantity.** ARR is
  `mean(candidate recall) / mean(oracle recall)`; the interval bootstrapped only
  the numerator. At T0 the oracle is perfect by construction so the two
  coincided, which is why every synthetic test passed — but the first real-corpus
  run reported `ARR 0.908 (95% CI 0.712-0.808)`, the estimate outside its own
  range. Replaced with a paired ratio bootstrap (`bootstrap_ratio_ci`).
- **`model_id` was redacted from logs.** A public model identifier is not corpus
  content, and a decision's reproducible inputs need it. The free-form `reason`
  field it travelled with was replaced by a closed `dropped` vocabulary, which
  is what the allowlist can honestly admit.

### Behaviour changes

Each is backed by a measurement recorded in `docs/m0-findings.md`.

- **Mean centering is now part of the Procrustes path by default.**
  Measured worth +0.166 ARR at T0 and **+0.260 at T1** across 12 corpus/model
  combinations, with a best case of +0.75. It hurt in 3 of 24 measurements,
  always by ≤0.018 — inside measurement noise. Centred Procrustes now matches the
  residual MLP in quality at half the memory and a third of the latency.
- **CSLS is a variant `auto` selects, not an always-on correction.** It was
  expected to "raise ARR for free". Measured: **+0.103** on weak
  adapters (ARR<0.5) and **−0.045** on strong ones (ARR≥0.8), with a Spearman
  correlation of −0.704 between its gain and adapter quality. Applying it
  unconditionally would degrade precisely the adapters that are working.
- **The decision rule gains a fifth outcome, `no_upgrade_needed`.** All
  four original bands assume the upgrade is going ahead. Across four corpora,
  staying on the old model retained a mean ARR of **0.983** and sometimes beat
  the new model outright. Whether the new model is better on *this* corpus is now
  asked first.
- **The borderline band widens from ±0.005 to ±0.025.** The measured
  bootstrap 95% CI half-width for ARR is ±0.024 on ~1,000 held-out queries and
  ±0.042 with real queries. A ±0.005 band asserted a precision the measurement
  cannot deliver.
- **`score_shift` is evaluated after calibration.** Before
  calibration, **100%** of measured configurations exceeded the 0.1 warning
  threshold — a warning that always fires carries no information. Isotonic
  calibration brings the median from 0.924 to 0.094 while preserving ranking in
  100% of cases.
- **The default low-rank affine rank is proportional to dimension**, not a fixed
  64. At d=384 a fixed 64 retains 17% of the dimensions and collapsed
  quality to 0.458 — below even plain Procrustes.
- **The default fit-pair count is 4,000**, not the 16,000 originally implied.
  The quality curve flattens at 4,000; the following 20,000 pairs added +0.001.
- **The layer contract is enforced over every module, not five of them.**
  The import-linter contract had carried a note since M1 saying modules would be
  added as they were written. Extending it to the full stack immediately found
  two real inversions that had been in place for two milestones:
  `compute.numpy_backend` reaching up into `probe.metrics` for `top_k_search`
  (now in `compute/search.py`, where the memory invariant it embodies belongs),
  and `storage.gc` reaching up into `manifest.paths` for the state-directory
  layout (now in `storage/layout.py`, re-exported upward). A partial contract is
  not a weak contract; it is an absent one for everything it omits.
- **The nightly GPU workflow was selecting nothing.** It ran
  `pytest -m "gpu or slow"` and no test carried either marker, so it had been
  passing by doing nothing. The device-parity suite now marks its accelerator
  parametrisations `gpu` — verified: 4 tests where there were 0.
- **`fit` no longer requires `--dim` for an unregistered old model.** The index
  is authoritative about its own dimension, and for an adapter fitted against
  vectors that already exist, the dimension is all that is needed.
- **The decision rule now compares bridging against keeping the current model.**
  It compared the adapter to the oracle ("how much of a reindex does
  this recover?") and asked separately whether the new model was better — but
  never whether *bridging* beat *doing nothing*. Measured on BEIR/scifact,
  MiniLM to bge-small: bridging recovers 0.903 of a reindex while keeping MiniLM
  gives 0.944, and the tool recommended migrating. It now answers `full_reindex`
  when the new model is worth having and `no_upgrade_needed` when it is not, and
  warns when the difference sits inside the noise band.
- **The T0 caveat now names what T0 cannot see.** Removing an asymmetric model's
  prefixes changes T0's answer by 0.005 — because T0's ground truth is the new
  model's own output, so a consistently wrong prefix moves the reference along
  with the measurement. At T1 the same error moves ARR by 0.05 and flips the
  recommendation. The prefix trap is real, but it is a T1 instrument.
- **Minimum Python is 3.12, not 3.11.** The dependency policy follows SPEC 0,
  under which a release is dropped three years after publication; 3.11 (October
  2022) passed that mark in October 2025. Declaring 3.11 would have been an
  untested support promise.
