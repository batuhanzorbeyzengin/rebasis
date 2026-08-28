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
been measured against a proper null for the first time, and the headline count it
used to carry turned out to be an algebraic identity rather than a prediction.
What survives is that its estimate ranks runs by the margin they returned at
Spearman ρ = 0.60, p ≈ 1e-6 — real information about the size of the effect,
not the accuracy the count implied
([what the counting is worth](docs/bridge-band.md#9-what-the-counting-is-worth)).

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
| **Continuous refit during migration** | ~~nothing calls it~~ **done, and the premise was wrong** | `migrate --refit` samples records **not yet migrated**, re-embeds them and refits on those pairs alone. The entry said to use already-migrated records: those carry the adapter's own image, so fitting on them fits `A` to reproduce `A`. Measured over 216 cells, the sample source is the whole effect — on a corpus that grew into a new domain, 1,000 pairs from the remainder beat 8,000 from the migrated half by **+0.20 nDCG**, while on an unchanged corpus nothing clears the adoption guard. [The numbers](docs/continuous-refit.md). |
| **LangChain / LlamaIndex bridges** | ~~adapters written, no tests~~ **done** | A contract suite of duck-typed fakes, one per capability the bridge has to infer, plus a layer driving the frameworks' own in-memory stores where installed. Writing it found both bridges declaring capabilities they could not deliver. |
| **Sample and embedding cache** | ~~defined, nothing writes to it~~ **done** | One SQLite file per encoding profile under `.rebasis/cache/embeddings/`, keyed on the profile fingerprint so a stale vector cannot be returned. `probe`, `fit` and `eval` use it; `audit replay` deliberately does not. |
| **`pause` / `resume` commands** | ~~no way to pause from outside~~ **done** | `rebasis pause <job-id>` records a request the engine reads at the top of every batch, so a job stops at a boundary rather than mid-batch; `rebasis resume <job-id>` is the verb that pairs with it. A request is a column of its own, not a `JobState`: only the engine says where a job *is*, and a second process writing `state` would claim a stop that had not happened. |
| **Serving a two-stage arrangement** | `rebasis.serve.Cascade`, measured over 48 runs ([the evidence](docs/cascade-band.md)), with a cache and per-stage timings | The decision rule does not recommend it. What is missing is not code but a measurement that cannot be taken on a corpus: how the cache behaves under a real query distribution, and what the path costs end to end on somebody's hardware. `Cascade.stats` is the instrument; somebody's traffic is the missing input. |
| **`doctor` against a live index** | ~~no store URI~~ **partly done** | `doctor --store <uri>` is in, read-only in every path: URI, open, text, SQLite integrity, manifest integrity, the recorded encoding profile, and whether the collection holds two embedding spaces. Two of the four checks originally listed were dropped rather than guessed. **Chunking drift** needs a baseline nothing records. The **Matryoshka truncate advice** needs a measurement nobody has taken — it is below, under *Matryoshka shortcut*, and it is advice only once that exists. |

## 0.3 — the parts that need a measurement first

These are not blocked on code. They are blocked on not yet knowing the right
answer, and guessing would be worse than waiting.

- ~~**`old → new` direction (virtual backfill).**~~ **Built, and measured.**
  `rebasis fit --direction old_to_new` produces the map `migrate` rewrites an
  index with, and `Bridge` refuses it while `migrate` refuses the other one —
  each is useless in the other's place and both guards now say so.

    Measured over 51 cells on real corpora with human judgements
    ([the band](docs/migration-band.md)): a completed migration delivers a mean
    **0.727** of a full reindex, and is **not distinguishable from bridging** —
    0.727 against 0.719, tracking at Spearman 0.993, with a paired median
    difference of +0.004. Which one wins had not been measured; the answer is
    neither, and that is ADR 10 arriving from a direction it was never tested
    from. Migrating buys the adapter leaving the hot path and costs a rewrite of
    every vector; it does not buy retrieval quality.

    It also inherits the band's other result: `migrated` beat doing nothing in
    **5 of 51**, and the five all had an upgrade gain of 1.2 or more.
- ~~**`--shadow-precision fp16`.**~~ **Measured, and it ships.** The worry was
  that a half guarantee is more dangerous than no guarantee, so the plumbing sat
  unexposed until somebody measured which this is. Over 68 corpus/model runs: no
  vector leaves the format, the top-10 **set** survives on 99.78% of queries at
  worst, and nDCG@10 moves by at most **0.0017** — inside ARR's own confidence
  interval and below the threshold `RefitPolicy` calls noise. What moves is the
  *order* within the top ten, on about 2% of queries.

    `float32` stays the default, because the disk it costs is temporary. What
    makes the option safe is that nothing claims bit-identity when it is on: the
    pre-flight plan says so, the shadow manifest records the precision, and
    `rollback` prints it off that file rather than off anyone's memory.
    [The numbers](docs/shadow-precision.md).
- ~~**Local threshold calibration.**~~ **Done.** `rebasis doctor --calibrate`
  times this machine and records the result in `.rebasis/calibration.json` — the
  one path in `doctor` that writes, and it writes only there. It measures what it
  can reach without downloading anything (kNN, and the residual MLP's fit where
  torch is installed) and records **only those**: `embed` needs a model, so it is
  omitted rather than guessed, which is the same rule energy and the reindex
  estimate follow.

    Connecting it found `worth_accelerating` promising a per-key fallback its
    code did not perform — it swapped the whole table for the local one, so a
    partial calibration would have read an absent `embed` as *not worth
    accelerating* and moved the dominant cost of a probe back onto the CPU, on a
    machine that had just been measured and found fast.

    It is a diagnostic and is documented as one. Nothing in the runtime
    dispatches per operation: a session runs under one ambient device and
    `top_k_search` consults no size threshold. On the project's own A10G host it
    reports kNN at 31.4x against the recorded 22x, and the MLP fit at 7.5x
    against 5.9x.
- ~~**Access-log-weighted sampling for `probe`.**~~ **Done, and the entry named
  the wrong place for the weights.** `rebasis probe --access-log` is in.

    A `probe` sample does two jobs: it is the mini-index every measurement runs
    against, and it is the pool the query proxies are split out of. Handing
    weights to the sampler — what this entry described — fills the mini-index
    with frequently-read documents and changes the **distractors**, which is a
    property of the index rather than of the questions asked of it. The weights
    go on the **split** instead, leaving the mini-index a fair miniature.
    Measured, that leaves the estimate about half as far from the whole-corpus
    quantity as weighting the sample does (+0.025 against +0.051).

    **The interval survives it**, which is what this entry was blocked on. Over
    36 cells and 12,960 replicate probes, the bootstrap half-width divided by the
    estimator's actual spread is 1.92 for the plain design against a correct
    1.96, and 1.84 under weighted queries — about **6% narrow**, in the direction
    the entry worried about and small against decision bands 0.10 wide. Median
    coverage is unchanged at 0.94; the tail moves from 2 cells under 0.90 to 6.

    Weighting shifts ARR by a median +0.015 at a 100x access ratio and up to
    +0.073, so it is a different quantity and the report says which one it is:
    `--json` carries `access_weighted`, and both report formats say so in prose.
    [The numbers](docs/access-weighting.md).

## Beyond 0.3 — where the headroom actually is

The measurement that shapes everything after 0.2 is this: **retention is not
improvable by fitting harder.** Six times the fit data buys 0.005–0.025 ARR, and
centred Procrustes beat a residual MLP 15 times out of 15. Gain and retention
anti-correlate at −0.958, reproduced at −0.940 on held-out corpora — a bigger
upgrade means the old model was weaker, and a weaker source space carries less
for any adapter to map ([the squeeze, and the held-out runs](docs/bridge-band.md)).

So the interesting work is not a better fit of the same shape. It is a different
shape:

- **Per-cluster adapters.** The premise was that one global map leaves quality on
  the table where drift is heterogeneous, and that `probe`'s `tail_arr` detects
  that without being able to act on it. **Measured, half of that is right and the
  half naming the instrument is wrong.**

    Per-cluster maps do beat one global map, on 12 of 18 corpus/model cells —
    k=8, one seed, each cluster fitted on its own full budget. But what decides
    it is not heterogeneity. Over those 18 cells:

    | predictor of the per-cluster gain | Spearman ρ | p |
    |---|---|---|
    | **corpus size** | **+0.900** | <1e-5 |
    | `tail_arr` gap | +0.046 | 0.87 |

    The relationship with size is monotone and the crossover is visible: all six
    cells on the two smallest collections lose (nfcorpus at 3,633 documents,
    scifact at 5,183), arguana at 8,674 is marginal, and every cell above that
    wins — up to +0.102 on TREC-COVID at 171,331. The tail gap has no
    relationship with the gain at all, and it never crossed its own warning
    threshold on any of the 22 corpus/model combinations surveyed for it —
    including collections built to be heterogeneous by reading two unrelated
    domains as one index. Whatever k local maps are fitting that one global map
    cannot, `tail_arr` is not the instrument that finds it.

    That the instrument fails is this project's own measurement; that it might
    have been the wrong thing to look for is someone else's. Lee et al. report
    that *local* neighbourhood geometry is shared across models much as global
    geometry is, and build a single global linear map on that basis rather than
    region-specific ones ([arXiv:2503.21073](https://arxiv.org/abs/2503.21073)).
    They study token embeddings in language models, not document embeddings in
    retrieval models, so it is a pointer and not a result — but it points where
    the 18 cells above already point, and from the other end.

    Three costs, and the first is the one that decides whether this ships:

    - **It costs k times the fit data.** Every gain above is the arm where each
      cluster got its own full budget. Split *one* budget across k clusters and
      it loses in **17 of 18** — so the honest framing is not "a better shape"
      but "more fitting, and this is where to spend it".
    - **Routing is not the bottleneck.** Nearest-centroid in the new space,
      through the bridge in the old, and an oracle assignment differ by about
      0.01.
    - **k was not varied.** Everything here is k=8. Where the gain peaks, and
      whether it is really about pairs-per-cluster rather than corpus size — the
      two are confounded in this design — is the measurement that would settle
      the mechanism, and it has not been taken.

    **The item stays open**, and what it needs has changed: not a way to detect
    heterogeneous drift, but a fit budget worth spending k ways and a rule for
    when the corpus is large enough to spend it.

- **Chained adapters.** v1 → v2 → v3 without a full refit at each step. Error
  accumulation across a chain has not been measured, and refitting against the
  original is probably more accurate — which is worth knowing rather than
  assuming.
- ~~**Matryoshka shortcut.**~~ **Measured, and the answer is no.** The idea was
  that for models with nested representations the right answer might be
  "truncate and renormalise", with no adapter at all. Two things came out of
  measuring it.

    First, *truncate and renormalise is what `IdentityAdapter` already does* —
    every consumer of an adapter's output normalises after `apply()`, on the
    scoring path, the serving path and the migration write-back alike, so the
    "new" candidate was bit-identical to the existing one. It is pinned as a
    property test rather than left as a claim.

    Second, `auto` was run with `identity` in the candidate list over 18 cells —
    three corpora, six model pairs. It won **none**, and on the fifteen cells
    where the dimensions actually differ it retained **0.001–0.004** against
    0.798–0.984 for the best fitted adapter. Where the dimensions match, and
    truncation is therefore a no-op, the same candidate retains 0.201–0.498 — so
    truncating costs nearly everything the pass-through kept.

    The reason is the direction of the cut, and it is not a matter of degree.
    Published work finds embeddings robust to truncation *within one model's own
    space* ([arXiv:2605.16608](https://arxiv.org/abs/2605.16608)); a bridge cuts
    a **new** model's vector down to enter an **old and different** model's
    index, where the surviving coordinates are compared against a basis they
    have nothing to do with. It is not a weaker bridge. It is not a bridge.

    `identity` is deliberately **not** added to `CANDIDATE_METHODS`. `auto`
    breaks ties on cost, a zero-parameter candidate wins every tie it reaches,
    and on a rung where nothing works `auto` would report `identity` as the
    selected adapter — which reads as "truncation was right" rather than
    "nothing worked". What is worth fixing instead is the *baseline*: `probe`'s
    "do nothing" number is dimension-gated and therefore missing on exactly the
    cross-dimensional runs where a user most needs it.
- **Unpaired alignment.** This is the one direction that would remove the
  hardest limit in the tool.

    **The limit was stated wrongly here, and in `README.md`, until now.** Both
    said it was being unable to run the old model. `fit` does not load the old
    model — it reads the index's own vectors and re-embeds the same documents
    with the candidate, so the pairs come from the store rather than from two
    live models. The limit unpaired alignment would actually remove is an index
    that **kept vectors and discarded the text**: there is nothing left to
    re-embed, so no correspondence can be built at all. That is a narrower case
    than the one written here before, and it is the real one.

    **The first step is no longer Wasserstein Procrustes**
    ([Grave, Joulin, Berthet, AISTATS 2019](https://arxiv.org/abs/1805.11222)),
    though it stays on the list. It optimises the correspondence and the
    orthogonal map together, which fits this codebase almost exactly. What moved
    it is published evidence rather than a measurement taken here, and the
    distinction matters: **`vec2vec`**
    ([Jha et al., NeurIPS 2025](https://arxiv.org/abs/2505.12540)) reports that
    its optimal-transport baselines performed comparably to a naive baseline on
    same-backbone pairs and near random on cross-backbone ones — and its own
    appendix concedes the comparison was run between embeddings of the *same*
    texts, which favours OT. Those baselines were Hungarian, EMD, Sinkhorn and
    Gromov-Wasserstein; **none of them was Grave et al.'s joint algorithm**. So
    the evidence is against the family, not against this specific method, and
    that is why it is reordered rather than dropped.

    **The first step is `mini-vec2vec`**
    ([Dar, arXiv:2510.02348](https://arxiv.org/abs/2510.02348)), and it has now
    been measured on this project's own ladder rather than argued for from the
    paper. It is the same orthogonal-solve-plus-assignment shape this project
    already wanted, with the assignment moved off the individual points and onto
    k-means centroids; its preprocessing is ADR 1 exactly, and it needs `scipy`
    and `scikit-learn` and no torch.

    **It works, on the rungs where it works at all.** 36 cells — four corpora,
    three rungs, three seeds — comparing a map fitted with *no correspondence
    whatsoever* against the paired ceiling `rebasis fit` reaches on the same
    data. Median recovery **0.81** of that ceiling; excluding the one
    cross-family rung, **0.84** with a floor of 0.61 and a ceiling of 0.94. The
    split is structural and asserted rather than assumed: the two halves share no
    document.

    **Where it fails, it fails completely, and it fails at stage one.**
    `potion-base-8M → all-MiniLM-L6` — a 256→384 jump across model families —
    recovers 0.00 to 0.66 depending on corpus, against 0.77–0.93 for every
    same-family rung. The spike reports a **centroid-agreement** diagnostic for
    exactly this: how often the quadratic assignment matches a centroid to the
    one an oracle map would have chosen. It ranks the outcome at Spearman
    **+0.833** (p = 3e-10), where the method's own confidence signal — the QAP
    objective, which is what the paper reports — manages **+0.519**. Confidence
    in the matching and correctness of the matching are different quantities, and
    the gap between them is worth 0.31 in rank correlation.

    The relationship is strong and not a clean threshold: below 0.20 agreement
    the mean recovery is 0.14 but one cell reaches 0.68, and above it the mean is
    0.81 but one cell sits at 0.04. It orders runs; it does not classify them.

    **What this does not establish.** One adapter family, four English corpora,
    one cluster count, and a fit that took a median of two minutes on a contended
    host. Nothing here has been run against the case the direction exists for —
    an index holding vectors whose text is gone — because that case cannot be
    constructed from a corpus that still has its text. What the numbers say is
    that the correspondence is recoverable without being given; whether that
    survives the setting a user would actually be in is untested.

    **`vec2vec` itself stays behind both.** It is adversarial plus
    cycle-consistency training, a great deal of machinery for a tool that runs
    on a laptop. If a cheaper route clears the same limit, that machinery is not
    needed; if it does not, the failure is the argument for paying for it.

## Before 1.0

1.0 means the API and the `.rbs` format stop moving. That needs:

- **`migrate` proved at production scale**, by someone other than the author, on
  an index that matters.
- **The `.rbs` format settled.** Schema 1 is the only version and there are no
  migrations yet, deliberately — the first migration is the worst possible time
  to design the migration machinery.
- **Full coverage on the modules where a bug costs data.** `audit/chain.py` is
  there (100%), as are `errors.py`, `types.py` and `observability/redaction.py`.
  `storage/shadow.py` is at 93.5% against a target of 95 and `storage/atomic.py`
  at 93.3% against 100. Neither number was enforced anywhere until now — this
  entry named both targets while `tools/check_coverage_floors.py` held
  the two modules only through the `src/rebasis/storage/` package floor of 80, a
  bar they clear together at 90.6%. They now carry a floor of their own at 90:
  a ratchet against sliding back, not the goal, which is still the 95 and the
  100 above. Overall coverage is 84.9% against a floor of 75, enforced by
  `fail_under` in CI.
- **More than one operating system in CI.** ~~Linux is the only one today.~~
  **Partly closed.** The macOS leg was added to the full suite and then removed:
  `faiss-cpu` and `torch` each link their own OpenMP runtime there, and a process
  holding both aborts before either library does any work
  ([faiss-wheels#40](https://github.com/kyamagu/faiss-wheels/issues/40),
  [pytorch#149201](https://github.com/pytorch/pytorch/issues/149201)). What was
  lost with it was coverage of the storage layer, which is exactly where the
  platforms differ — directory fsync, `os.replace`, a system sqlite3 that cannot
  load extensions — and exactly where a bug costs data. The narrower leg this
  entry asked for now exists: `core install` installs no extras, so neither
  library is present and the conflict cannot arise, and it runs
  `unit or property or contract` on `ubuntu-latest` and `macos-latest` both.
  That is the storage layer covered: `atomic`, `shadow` and the manifest schema
  are all `unit`, so all three now run on the second platform. **What it does
  not cover is a real backend on macOS.** A core-only install has none, so the
  five live stores `importorskip` out of the contract suite there and only
  `memory` runs, and `integration` and `e2e` are not selected at all. Windows
  has documented behaviour in the storage layer that nothing exercises at all.
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
