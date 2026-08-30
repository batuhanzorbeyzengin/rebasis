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

## [0.1.0] - 2026-08-27

### Behaviour changes

- **`rebasis migrate` refuses the adapter `rebasis fit` produces, and every migration run before this one produced an index no query could answer.**

  An adapter has a direction. `fit` writes `query_to_old`: a map from the new model's space *into* the index's, which is what lets `Bridge` send a new-model query at an index nobody has touched. `migrate` does the opposite job — it rewrites the **indexed document vectors** — and for that it needs `old_to_new`, a map *out* of the index's space. It never checked, and `README.md` and the migration guide both showed `fit` writing an adapter and `migrate` taking it.

  Applying the query map to document vectors passes every guard the tool had, and that is why this survived. The write lands. The record count holds. The text survives. The read-back verifies — it compares what was written against what comes back, not against anything meaningful. `migrate`'s own index-health check measures the store's search against exact kNN *over the vectors it now holds*, which its module docstring states is a property of the index structure rather than of the vectors' meaning. The end-to-end suite ran `fit` into `migrate` in eight tests and asserted that the vectors changed, the count did not, the text survived and `rollback` restored the originals. All of that was true. None of it asks whether the index can still find anything.

  Measured, on 4,000 documents where both spaces are known exactly and the bridge scores recall@1 **1.000** against the untouched index: the index a completed migration leaves behind answers **0.000** to a raw new-model query, **0.000** to a bridged query and **0.000** to an old-model query. There is no query that is correct against it. For an orthogonal adapter the arithmetic says why — `A(q)·A(d) = q·d`, so a bridged query against a fully migrated index reduces to the naive swap, which retains 0.125 of a reindex.

  `migrate` now refuses a `query_to_old` adapter before it opens the store, naming the direction it got and the one it needs. `old_to_new` is a roadmap item and deliberately unbuilt — which direction is the better trade-off has not been measured — so there is currently no adapter `migrate` can be run with, and it says so instead of writing. **`rollback` is untouched**: it restores from the shadow copy and never applies an adapter, so anyone who has already migrated can still put their index back.

  Found while measuring something else: two independent agents reading `serve/hybrid.py`'s fusion behaviour and `cli/migrate.py`'s adapter handling arrived at the same discrepancy, and it was confirmed by direct measurement before anything was changed. The end-to-end tests now build a genuine forward adapter for the migration machinery — the queue, the shadow copy, the read-back, resume and rollback, all of which are indifferent to which way the map points — and one new test asserts that the adapter `fit` writes is refused, with the index untouched.

  **Checked against the published design, after the fact and on a challenge.** Drift-Adapter — the work this tool's approach is closest to — migrates a corpus by *re-encoding* it: "the corpus is gradually re-encoded with the new model in the background … complicating querying unless a strategy like Drift-Adapter is used to harmonize query and database embeddings" (section 2.1), and its own gradual-migration experiment refreshes items "with `f_new` embeddings" (section 5.6). The adapter's job there is the query, in the settled state and the mixed one alike; it is never applied to a stored document vector. Industry accounts of the same problem — Qdrant's migration tutorial, the blue-green and dual-column patterns — describe re-embedding for the same reason. So applying a query-side map to stored vectors is not a variant of the published approach that was implemented in the wrong direction; it is an arrangement the published approach does not contain.

  That leaves a second question this change does not answer. Even with `old_to_new` built, an adapter migration is bounded by what the adapter can carry — [ADR 10](https://batuhanzorbeyzengin.github.io/rebasis/adr/0010-retention-is-bounded-by-the-source/) applies to documents exactly as it applies to queries. Measured with a correctly directed adapter on two corpora, a completed migration reached 84% and 88% of a real reindex, and on one of them landed below the status quo it started from. Making `migrate` runnable and making it worth running are different pieces of work.
- A run that cannot measure whether the new model is better is now marked provisional: it reports how well an adapter bridges and declines to recommend acting on it. Without a query log the previous rule said `bridge_and_migrate` in six of six cases where bridging actually lost ground.
- A synthesised upgrade estimate is now treated as provisional when it lands within its own error of the break-even. Measured over 60 runs across five corpora, three strategies and four sample sizes, a synthesised `upgrade_gain` carries a mean absolute error of 0.14 to 0.22 against a real query log's figure — and the threshold it is compared to is 1.0. It settles clear-cut cases and says so when it cannot settle a close one.
- T0 now encodes its query proxies the way a query is encoded rather than the way a document is. For symmetric models nothing changes; for asymmetric ones the previous figure was optimistically biased, because it measured document-retrieves-document rather than what happens at serve time. T0 still cannot evaluate the query encoding itself — see ADR 8.
- The decision is now driven by the break-even (`ARR x upgrade_gain`) rather than by the ARR bands, which now only choose between the two bridging answers. Measured over 15 runs on corpora searched with real user queries, the break-even matched the independent outcome 14 times and the bands 10 — and every disagreement was a genuine win the bands rejected, including one that measured +16.0%. Agreement rises from 10/15 to 13/15, and the four real wins go from 0/4 recommended to 4/4.
- When the measured break-even lands inside its own noise band, `probe` now marks the recommendation provisional instead of printing a confident headline over a warning that contradicts it. On BEIR/scifact with a real query log, `bridge_advantage` came out at 1.017x — inside the borderline band — and the run printed "bridge and migrate" above "they cannot settle whether to use it". This is the same situation an unsettled synthesised estimate already took the provisional route for; a measured one now takes it too. The ARR band and every number are unchanged.
- `probe` now compares bridging against keeping the current model, not only against a full reindex. Measured on BEIR/scifact, MiniLM to bge-small: bridging recovered 0.903 of a full reindex while keeping the current model gave 0.944 — so the previous rule recommended migrating to something measurably worse.

### Removed

- CI is six jobs and one test run, down from ten jobs and two. The suite was running twice — `tests` and `coverage` differed only in their marker expression — for fourteen minutes of runner time and two chances to go red. Also gone: the perf layer, which asserts timing that a shared runner cannot measure (it failed twice by 1% and 2.6%, and [ADR 11](https://batuhanzorbeyzengin.github.io/rebasis/adr/0011-the-hot-path-budget-is-per-dimension/) already says those numbers belong on a fixed runner); the macOS leg, which could not finish because `faiss-cpu` and `torch` each link their own OpenMP runtime there and a process holding both aborts ([faiss-wheels#40](https://github.com/kyamagu/faiss-wheels/issues/40), [pytorch#149201](https://github.com/pytorch/pytorch/issues/149201)); and the second Python version, which across this repository's runs caught nothing the floor did not. Every job now carries a `timeout-minutes`. The perf layer and the macOS suite both still run on the maintainer's machines, and `docs/development/testing.md` records what would justify bringing any of these back.
- Two CI jobs are gone. The CodSpeed benchmark job measured nothing — `pytest --codspeed` collects only what uses the `benchmark` fixture and no test did — and running it needs a token this project does not have. It was described here as leaving the memory ceilings and the scaling test as the guarantees that gate a PR; they were not gating anything, because they wore `perf` too. That is fixed separately, by splitting the marker. A second job grepped the tree for a section symbol and for a document that is already covered by `.gitignore` and a pre-commit hook, which is one mechanism too many for the same thing.

### Added

- **`rebasis migrate` can be run again, with an adapter that points the right way — and the measurement says it is worth about as much as bridging, which is usually not much.**

  `rebasis fit --direction old_to_new` produces the map a migration needs: out of the index's space and into the new model's, rather than the reverse the query path uses. It is the same `fit_candidates` call with source and target exchanged, but it is **not** the same fit with its arguments swapped, because the evaluation differs and that is the part that matters. A query map is judged on what a bridged query retrieves from an untouched index; a document map is judged on what a **raw** new-model query retrieves from a rewritten one. `rebasis.probe.migration` scores the second, which is the configuration a user is left in once `migrate` finishes and there is no adapter on the hot path at all.

  **Both directions are now guarded, in both directions.** `migrate` refuses a `query_to_old` adapter before it opens the store; `Bridge.load` refuses an `old_to_new` one. Each is useless in the other's place and neither check existed a release ago — which is how a query map came to be written over indexes until it was measured at recall@1 0.000 for every query type there is.

  **What it is worth, over 51 runs on seventeen corpora with human relevance judgements** ([the band](https://batuhanzorbeyzengin.github.io/rebasis/migration-band/)):

  | | |
  |---|---|
  | a completed migration delivers | **0.727** of a full reindex |
  | bridging, on the same runs | **0.719** |
  | the two track each other at | Spearman **0.993** (p ≈ 1e-46) |
  | paired median difference | **+0.004** in favour of migrating |
  | migrating beat leaving the index alone in | **5 of 51** |

  So the two are the same number. That is [ADR 10](https://batuhanzorbeyzengin.github.io/rebasis/adr/0010-retention-is-bounded-by-the-source/) reaching the document side for the first time: the same source space under the same family of map carries the same amount whichever end it is applied to. The ADR was measured entirely on the query side and could not have said so.

  What migrating buys is the adapter leaving the query path — nothing on the hot path, no `.rbs` shipped with a service, the new model querying its own space. What it costs is rewriting every vector, the shadow copy behind it, and a window in which the index holds two spaces. The guide now says to choose on those grounds, because retrieval quality is not one of them.

  The end-to-end suite migrates through the real command now rather than through an adapter built by hand in the test file, which was a workaround for the window in which nothing could produce a forward map.
- A contract suite for the LangChain and LlamaIndex bridges, `tests/contract/test_bridge_stores.py`, which were the only backends in the project with no tests at all. Its fakes are built from the upstream source of `langchain-core` 1.6.0, `langchain-chroma` 1.1.0 and `llama-index-core` 0.14.24 rather than from a reading of the bridges — a fake shaped like the adapter passes by construction and keeps passing after the real interface has moved — and a second layer drives both frameworks' own in-memory reference stores where the extras are installed. Most of what the bridges declared turned out to be untrue. `can_read_text` was read off `similarity_search`, which is abstract on LangChain's base class and therefore present on every store ever written; `can_upsert_vectors` off `add_embeddings`, a method that appends and that `upsert_vectors` refused regardless; `can_read_vectors` off a `_client` the bridge never reads through, or off a LlamaIndex `client` property that is abstract and so never absent. Each now answers for the handle that performs the read, and each that cannot is `False`. Six defects went with them: neither bridge had `rebuild_index`, so the refusal arrived as an `AttributeError`; `iter_records` fetched the whole collection in one call and deferred its capability check into the caller's loop, and now pages and refuses at the call; unknown ids were dropped silently; a hit with no id was given its position in the result list, which matches no record; a `_collection` property that raises — Chroma's does — escaped `capabilities`; and a missing `llama-index-core` was reported as a missing store capability rather than a missing dependency. Behaviour change worth reading twice: the LangChain bridge no longer passes a search score through. LangChain guarantees nothing about the direction of that number, and Chroma returns a raw distance from a method whose name says relevance, so every hit is scored `0.0` and only the rank is reported unless the adapter is built with `score_kind="similarity"` or `score_kind="distance"`. A silently inverted score is worse than a visibly absent one.
- A partially migrated index now says so. `--limit`, `--priority access` and every pause leave the collection holding both models' vectors, and until the job finishes no query is correct against all of it — a bridged query mis-scores the records that moved, an unbridged one mis-scores the rest. Nothing raised, nothing was missing, and the ranking was quietly wrong. `migrate` now warns before you confirm whenever `--limit` will stop the run short, again at the end of a run that did, and `rebasis status` reports it unprompted until the job finishes or is rolled back — including as `mixed_space` in `--json`, so a script can refuse to serve from a mixed index rather than measure the wrong thing.
- CI runs on macOS again, on the one job that can carry it.

  The macOS leg was removed from the full suite because `faiss-cpu` and `torch` each link their own OpenMP runtime there, and a process holding both aborts before either library does any work. What went with it was coverage of the **storage layer** — directory fsync, `os.replace`, a system sqlite3 that cannot load extensions — which is exactly where the two platforms differ and exactly where a bug costs data rather than accuracy.

  A core-only install has neither library, so the conflict cannot arise. The `no torch, no otel` job was already installing nothing optional for its own unrelated reason, so a second operating system costs it one matrix entry and no new risk. It now runs the unit, property and contract layers on Ubuntu and on macOS.

  This does not restore the full macOS suite: the backends, which is where `faiss-cpu` enters, still run on Linux only.
- CI runs the suite on macOS as well as Linux, on the floor Python. The storage layer is where the two differ — directory fsync, `os.replace`, and a system sqlite3 that cannot load extensions — and the users this tool is aimed at are mostly on macOS. The `3.14` trove classifier is gone: nothing tested it.
- Embedding backends for FastEmbed, Ollama, llama.cpp and any OpenAI-compatible endpoint; a FAISS store backend. FastEmbed needs no torch; the OpenAI-compatible one is the only backend that can send document text off the machine, and says so.
- Every arXiv identifier in the committed documentation now has to say which paper it is. `docs/citations.toml` records the title beside each one; `tools/check_citations.py` checks on every pull request that the manifest and the documentation name the same set of papers, and weekly that arXiv still returns those titles. This is not tidiness: `docs/cascade-band.md` cited arXiv:2605.24297 for the claim that an off-the-shelf reranker can degrade a strong first stage, and that identifier is a benchmark of 22 embedding models on patents. The paper meant was arXiv:2411.11767, *Drowning in Documents*. Nothing can tell whether a paper supports a claim — but writing the title down next to the identifier turns a private belief into something arXiv can contradict, and it would have contradicted that one immediately.
- Every probe run records what it cost — wall and CPU time, peak RSS, BLAS threads, device, and energy where the hardware reports it — plus an estimate of a full reindex extrapolated from the run's own measured embedding rate. Energy is measured or absent, never estimated.
- Four things a security-conscious team asks for before depending on a package, none of which this repository could previously answer without a conversation.

  **An SBOM.** Every release now attaches a CycloneDX 1.5 bill of materials to its GitHub release, covering the tree `uv.lock` resolves with all extras installed — a vulnerability in the chroma or torch half is one a user who installed that half has, so an SBOM of the core alone would describe a tree almost nobody runs. It goes on the release rather than to PyPI, which accepts distributions and nothing else. The rehearsal builds it too, so a broken export is found before it can stop a release.

  **A vulnerability gate on the merge path.** `actions/dependency-review-action` runs on every pull request and fails on a high-severity advisory or a copyleft licence in what the change introduces. That is the gap Dependabot leaves: Dependabot reacts once an advisory exists for something already merged, and works from the manifest rather than the resolved tree.

  **A periodic audit of what is actually resolved.** A weekly `pip-audit` over `uv.lock` with every extra, asking the question the manifest cannot: given the environment this lock produces, is anything in it known-vulnerable today. Weekly and never on a pull request, for the same reason the citation check is — a new advisory published upstream is not a reason to block a merge that has nothing to do with it.

  **A score from somebody else.** An OpenSSF Scorecard run, weekly and on every push to `main`, with the badge in the README. Two of its checks this project cannot pass and `SECURITY.md` says so rather than working around them: *Code-Review* counts pull requests reviewed by someone other than their author, and *Branch-Protection* reads a setting that would block the release workflow's own push until a rule is written to allow it. Neither improves by configuring something.

  `SECURITY.md` also now states what a reporter should expect — acknowledgement in three working days, an assessment in ten, no promised fix date — that an accepted report can carry a CVE and reaches OSV through GitHub's advisory database, and how to verify a downloaded release against its PEP 740 attestation.
- Migrate-and-rollback integration tests across every writable backend — chroma, lancedb, qdrant, sqlite-vec and faiss — checking that the vectors actually change, the record count does not, text survives and `rollback` restores the originals. Previously this was covered on the in-memory store and by hand on Chroma.
- Release automation (`python-semantic-release` + towncrier), Dependabot, a feature-request template, eight Architecture Decision Records, and development guides for the GPU host and for releasing.
- Shell completion is enabled: `rebasis --install-completion`. It was switched off, and the arguments this CLI takes are the kind nobody types correctly twice — store URIs like `chroma:///long/path#collection` and model ids like `sentence-transformers/all-MiniLM-L6-v2`.
- Tests for `rebasis gc`, the one command whose purpose is to delete: that a bare run removes nothing, that `--apply` removes only what it listed, and that a shadow copy — the thing that makes a migration reversible — needs `--i-understand` before it goes.
- The project has a logo, and the README leads with it. The artwork is kept at `docs/assets/logo.png`; the README uses a trimmed banner crop of it, referenced by absolute URL because the same file is the PyPI long description and PyPI cannot resolve a repository-relative path.
- The report now says when recall and ranking disagree about whether bridging helps. An adapter can return the same documents in a worse order; the decision runs on recall and could not see it. Which metric matters depends on whether a model or a person consumes the results, so the disagreement is surfaced rather than resolved.
- The store contract suite runs against every embeddable backend — chroma, lancedb, qdrant, sqlite-vec and faiss — rather than the in-memory store alone. The two properties it checks hardest, lazy iteration and truthful capabilities, are precisely the ones a real client library gets wrong and a dictionary cannot.
- `--old-dim`, `--new-dim`, `--query-prefix` and `--document-prefix` on `probe`, `fit` and `eval`, so a model rebasis has never seen can still be measured. The error that told users to pass these flags predated the flags themselves.
- `--synth-queries title|lead|keywords` estimates the upgrade from the documents when no query log exists (the T2 tier). `keywords` is the strategy worth using — `lead` and `title` hand the retriever the answer, and rebasis detects that and marks the run provisional rather than reporting a meaningless 1.00x.
- `Release` takes a `mode` instead of a `dry_run` boolean: `dry-run`, `rehearse`, or `release`.

  `dry-run` prints the version the commits imply and the changelog that would be written, and stops. `release` is what it was. `rehearse` is new, and it is the one worth having before a first release: it assembles the changelog, commits it on the runner's throwaway checkout, tags locally, builds at that tag and uploads to TestPyPI — everything the real run does except the two irreversible parts, since nothing is pushed and nothing reaches PyPI. The job is not granted `contents: write`, which makes "nothing is pushed" a property rather than a promise.

  What a rehearsal proves is what a dry run cannot: that towncrier assembles, that `hatch-vcs` reads the tag, and that Trusted Publishing works end to end for this repository. It needs a pending publisher registered on TestPyPI against the `testpypi` environment.

  It commits rather than building a dirty tree because `hatch-vcs` appends a local version segment to a build made from uncommitted changes, and PyPI rejects local version identifiers — so a rehearsal from a dirty tree fails at the upload for a reason unrelated to the release. Both publishing jobs now check the built filenames for that segment and name it, rather than letting it surface as a rejected upload.

  The pre-release suite also drops the `perf` layer, matching `ci.yml`. It was running the wall-clock assertions CI had already removed as unmeasurable on a shared runner, including the one that had gone red twice on noise — a release blocked by a 2.6% timing wobble is a release blocked by nothing. Those numbers are gated on the host before a release instead.
- `StoreCapabilities.can_rebuild_index`, and `migrate --rebuild-index`. Measured on a 100,000-record Chroma collection: an unconstrained affine adapter cost 5.2 points of the index's own recall against exact kNN, and rebuilding the collection recovered all of it — the vectors were correct throughout and the graph was describing where they used to be. A low-rank adapter cost 11.7 points and rebuilding recovered *none*, because that loss is in the vectors rather than the structure. So the capability is declared per backend and the rebuild is offered rather than performed: Qdrant can do it (changing `ef_construct` triggers a full background rebuild and keeps serving from the old index meanwhile, which its documentation states); Chroma exposes no supported way; sqlite-vec and FAISS scan exhaustively and have nothing to rebuild. Full measurement in `docs/index-health.md`.
- `StoreCapabilities.quantized` — whether a store keeps the vectors it is given or a code they were made from — and `migrate` states the difference before it writes anything. `rollback` is sold on a bit-identical shadow copy, and the shadow is bit-identical to *what the store returned*; a quantized store returns a value decoded from its codes, so there "the original" means the state the migration replaced and not the vectors the embedding model produced. Nothing detected that case and nothing said so, which is exactly the silent partial support this project refuses everywhere else. `migrate` does not refuse it either — a quantized index is a deliberate choice and its owner has as much right to migrate one as anybody — it says what changes, most fully under `--dry-run`.

  Three values, not two, and the third is the point. `can_rebuild_index` next door defaults to `False` safely, because declining to offer a repair costs nobody anything; `quantized=False` is a *guarantee* that what you write is what you read back, so a backend that answered it without looking would have made a promise it could not keep. The default is `None` — not determinable — which is what a third-party store behind the LangChain or LlamaIndex bridge honestly is, and it is surfaced as the absence of an answer rather than as a warning: a caveat printed on every run is one that stops being read.

  Every backend was checked against the installed library rather than assumed, and two of the answers are narrower than they look. FAISS answers from `sa_code_size()` against `4 × d`, reached through the `IndexIDMap2` wrapper — closing a real hole, because the existing refusal catches only an index whose `reconstruct` *raises*, while an `IndexPQ` or an `IndexScalarQuantizer` reconstructs happily and returns a decode. sqlite-vec answers from `vec_type()`, which reports `float32`, `int8` or `bit`. LanceDB answers from the Arrow element type of the vector column and **not** from `IVF_PQ`: the compressed copy lives in the index's own columns and the vector column is untouched. Qdrant answers from `VectorParams.datatype` and **not** from `quantization_config`: Qdrant keeps the quantized codes alongside the originals, which is what makes its own rescoring possible, so a scalar-quantized Qdrant collection round-trips exactly and warning about it would be a false alarm. Chroma is a hard `False` — one storage encoding for a dense vector and it is float32, checked at both ends of the supported range.

  Measured rather than argued: `tests/integration/test_quantized_roundtrip.py` puts known vectors through an 8-bit scalar-quantized FAISS index and finds the deviation larger than `migrate`'s own read-back tolerance (`VERIFY_ATOL`, `1e-4` at this release) in both directions — so on a codec that coarse the run stops at its first batch, with the shadow copy already on disk and nothing lost. How much a given codec costs a given corpus is a separate measurement and is not claimed here. `migrate` has no `--json`, so the finding travels with the job in the audit trail, as `store_quantized` on the `migrate.job.started` record, where `null` and `false` are different answers.
- `auto` now fits and measures both adapter strategies for asymmetric models — one shared adapter, and one fitted on query-encoded pairs — instead of choosing between them on principle. The winner carries an `@query` suffix when the query-specific fit won.
- `close()` and context-manager support on the Qdrant and sqlite-vec backends. Qdrant's local mode takes an exclusive lock on its storage folder, so a handle that is never released makes the next read fail rather than wait.
- `gc` lists the shadow copy of a job that has already been rolled back. `rolled_back` is terminal, so that copy can never be used again — and it is the largest thing rebasis leaves on disk, the size of the vectors it replaced. It is listed unprompted and removed without `--i-understand`, because there is no longer anything to protect. Shadows of jobs that can still be rolled back are unchanged: named explicitly, confirmed explicitly.
- `migrate --dry-run` (`-n`) prints the plan and stops. `gc` accepts the same flag for the dry run it already does by default, because `-n` is what people type, and `gc --apply --dry-run` is now refused rather than silently resolved one way.
- `migrate` now measures whether the index can still *find* what it wrote to it. The existing checks answer neighbouring but different questions — the per-batch read-back proves the store took the write, the fresh-connection check proves it kept it, and neither proves the record is still retrievable. A graph index chooses a record's edges from the geometry of its neighbours at insert time, and rewriting the vector does not rewrite the graph. Before and after the run, `migrate` samples records, uses their own stored vectors as queries, and compares the store's search against exact kNN streamed over the collection (`O((sample + batch) × d)`, no matrix). Any drop is named. `--no-health-check` turns it off; it costs two scans.
- `migrate` reopens the store on a fresh connection when the queue empties and re-checks a 64-record sample drawn from across the whole run. The per-batch read-back goes through the handle that wrote, which is exactly the handle a caching client answers from memory; this catches a store that accepted every write and did not keep them (`RB-E6005`).
- `migrate` shows progress while it runs. It printed the plan, then nothing at all until it finished — on the 487k-record index the README uses as its example, that is a long silence from the one command that writes to your data. It now shows an `X of Y` bar with elapsed and remaining time, because the queue knows the total before the first batch. `probe` and `eval` name the stage they are on (sampling, embedding, ground truth, scoring) instead of holding one static spinner for the whole run: a spinner that never changes cannot distinguish slow from wedged, and embedding dominates the others by an order of magnitude. Progress goes to stderr, so `--json` still pipes cleanly and redirecting stdout still lets you watch.
- `probe` and `fit` print the command to run next. rebasis is a three-step tool whose second step was only written down in the README, so a user who reached a decision had to go back and find the invocation. `probe` now prints the `fit` command for the store and models it just measured; `fit` prints both ways to use what it wrote — the `Bridge.load` line and the `migrate` command. A decision of `no_upgrade_needed` or `full_reindex` says plainly that there is nothing to fit.
- `probe` now reports a **ceiling**: the best score any query-side map could reach in the index it is looking at. It answers the question `arr` cannot. `arr` says how much of a full reindex the fitted adapter recovered; it says nothing about whether more was available, and a low `arr` has two readings with opposite responses — *this adapter is weak*, which is a reason to keep looking, and *the old space does not hold these neighbourhoods at all*, which is [ADR 10](https://batuhanzorbeyzengin.github.io/rebasis/adr/0010-retention-is-bounded-by-the-source/) and a reason to reindex instead. Until now the report could not separate them.

  The construction is an oracle and is labelled as one wherever it appears: for each query, its relevant documents' vectors **as the old index holds them**, averaged and normalised. Among unit vectors that is exactly the one maximising summed similarity to the target set, so it estimates the best a single query point can do in that space. Nothing a user can run produces it — it is built from the answer — and it is there to bound the rows above it rather than to be quoted on its own.

  **Why this number and not the cheaper one.** `core/geometry.py` already computes a pre-fit bound from Maystre et al.'s Procrustes result, and its docstring is careful that the bound runs one way: geometry preserved implies alignment possible, and the converse does not hold. That caution has now been measured. Over 144 runs carrying both quantities, the pre-fit bound ranks runs by their eventual retention at Spearman **−0.30** (p ≈ 0.03) — detectable, and nowhere near usable. The ceiling manages **0.90** (p ≈ 1e-17). The bound stays what it says it is; the ceiling is what predicts.

  It costs one extra search over the sample `probe` has already embedded, and it is **omitted rather than reported as 1.0** where every query has a single relevant document — there the centroid is that document, it retrieves itself, and the number would carry no information. That is the ordinary case at T0, where `SPARSE_RELEVANT` is 1, so most runs will not show it.

  **It deliberately does not enter the decision rule.** Predicting well over 144 runs is not the same as a threshold that has been validated, and that second measurement has not been taken. It is a reported number, and the report says what it is for.
- `probe` reports a geometry-preservation δ and the alignment bound it implies, from Corollary 1 of [Maystre et al., *When Embedding Models Meet*](https://arxiv.org/abs/2510.13406): if two models' pairwise similarities agree to within δ, the best orthogonal alignment satisfies `E[‖x̄ᵢ − yᵢ‖²] ≤ √(2D)·δ`. It costs one Gram-matrix difference and no fit, so it is known before the candidate search starts. It is a bound and is described as one — it says an alignment of at least that quality exists, never that retrieval will realise it. ADR 10 records why that does not overturn its rejection of *predicting* retention, and ADR 1 gains the same paper's counter-example showing that centring on its own is not principled.
- `probe` reports what the same adapter would retain if it produced a **candidate set** for the new model to rerank rather than the final ranking. That arrangement is bounded by recall at candidate depth instead of by nDCG@10, which is a weaker requirement and measurably so: across 48 runs, retention 0.717 against 0.893, and bridging beat keeping the current model in 1 run against 36 ([the measurement](https://batuhanzorbeyzengin.github.io/rebasis/cascade-band/)). It is reported and **not acted on**: `rebasis.serve.Cascade` serves that arrangement, but what it costs turns on how often a candidate is already cached — a property of a query log rather than of the corpus a probe reads — and the decision rule runs on the number it can price from the corpus alone. Widening the search to reach candidate depth leaves ARR, its interval, nDCG, MRR and the decision bit for bit where they were, which is asserted rather than assumed.
- `probe` reuses embeddings it has already computed. `.rebasis/cache/` has been declared, named by `REBASIS_CACHE_DIR` and collected by `rebasis gc` on a 30-day retention since the state directory existed, and nothing ever wrote to it — so every run embedded its whole sample from scratch: the same ten thousand documents, with the same model, every time the sample size moved, a query log arrived, or a second candidate model was tried. `rebasis.storage.EmbeddingCache` is what writes to it, one SQLite file per model profile under `.rebasis/cache/embeddings/`. How much time that saves has not been measured; what it removes is a repeat of work the previous run already did.

  **A partial hit is the case that matters**, because `--sample` is a flag people move: only the texts the cache does not hold reach the model, and the array that comes back has one row per text the caller passed, in the caller's order. One file per profile rather than one for everything, because `gc`'s unit of retention is a file — a candidate model somebody evaluated once and abandoned can age out on its own instead of being held alive by the model they still use.

  **The cache cannot change an answer.** Every key carries the encoding profile's fingerprint, the `kind`, whether the vectors are ℓ2-normalised and the dtype, so a vector produced under one description of a model is unreachable under another — a stale vector would not raise, it would be *measured*, and the recommendation that came out would be a plausible answer to a question nobody asked. It cannot take a run down either: an unwritable directory, a corrupt file, a database from a newer release and a truncated row are all misses, counted rather than raised.

  **The reindex estimate is measured over the documents a run actually embedded**, and omitted rather than extrapolated from zero when the cache answered all of them. Nothing is claimed that has not been measured, and "a full reindex takes no time" is not something a warm run measured.

  On by default in `rebasis probe`, which has already created `.rebasis/` for the audit trail; off in `probe_store` unless a directory is passed, because a library that starts writing into someone's project directory unasked has taken a liberty. `REBASIS_EMBED_CACHE=0` turns it off everywhere, for when the question is "is this number real, or did it come from a cache?".

  `rebasis.storage.EmbeddingCache` also satisfies `serve.cascade`'s `VectorCache` protocol, so a serving process whose working set has outgrown `DiskVectorCache`'s directory of one-file-per-vector can hand `Cascade` one of these instead. The two remain separate classes: they key different things, expire on different clocks, and merging them would have made one of them worse.
- `probe`, `eval`, `status`, `gc` and `doctor` take `--json`, and `status` also takes `--plain`. A decision a script wants to branch on was only available as a Rich table, which draws box characters and truncates job ids with an ellipsis — so `rebasis status` could show you a job id that `rebasis rollback` would then reject. The JSON is the same structure the report and the audit record are built from. `doctor --json` is meant for bug reports, where a screenshot of a table is a poor attachment.
- `rebasis --version` prints the version. There was a `version` subcommand, but `--version` is what people type first, and answering `No such option: --version` reads as a broken tool before anything has run.
- `rebasis adapter upgrade`, `inspect` and `profiles`. `upgrade` was referenced from a user-facing error before it existed; it writes a new directory and never modifies the original.
- `rebasis doctor --calibrate` times this machine and writes `.rebasis/calibration.json`. It is the one path in `doctor` that writes, it writes only into the state directory, and it only runs when asked for by name.

  `rebasis.compute.thresholds` records speedups measured on one GPU against one CPU and says so — a faster host narrows every one of them. `Calibration` and `load_calibration` have been in the module since it was written, with a docstring conceding that nothing produced one. This produces one: kNN through the same `top_k_search` a probe calls, and the residual MLP's fit where torch is installed, each timed on the CPU and on the accelerator with a warm-up and a median over repeats.

  **It records only what it measured.** `embed` dominates a probe and needs a model, and a diagnostic that downloaded 400 MB to answer a question nobody asked would be a bad citizen — so it is omitted rather than guessed, which is the rule energy and the reindex estimate already follow. The shape of each measurement (sizes, dimensionality, repeats) goes into the file beside the ratio, because a ratio without its configuration is not reproducible and the file outlives the terminal.

  **Connecting it found a documented-but-absent behaviour.** `worth_accelerating`'s docstring promised to fall back to the reference table where a local calibration has no entry; the code swapped the whole table for the local one. A partial calibration — which is the only kind this produces — would therefore have read an absent `embed` as *not worth accelerating*, turning off the dominant cost of a probe on a machine that had just been measured and found fast. The fallback is now per key.

  It is a diagnostic, and the documentation says so rather than implying more: nothing in the runtime dispatches per operation, so a calibration changes what `doctor` reports about this machine and not where work runs. On the project's own A10G host it reports kNN at 31.4x against the recorded 22x and the MLP fit at 7.5x against 5.9x — the reference numbers hold there, which is a finding about that host and not about anyone else's.
- `rebasis doctor --store <uri>` points the diagnostic at a live index. It reports whether the URI parses and the index opens — and when it does not, the backend's own error, code and next step rather than a stack trace; which backend and what it declares it can do; the record count and the dimensionality; whether document text comes back. It runs SQLite's own `PRAGMA integrity_check` against the file under the index wherever there is one to reach, and against rebasis' own manifest, which holds the audit trail. Both paths appear in `--json`, which is what the README asks people to attach to a bug report.

  **The check that earns its place is the half-migrated index.** `migrate --limit`, `--priority access` and every pause leave the collection holding two embedding spaces at once, and no query is correct against both — the count is right, the text is right, the ranking is wrong. `status` has reported that since it existed and `doctor` could not, because `doctor` had no store; a user whose retrieval has quietly got worse runs `doctor` first. It costs one indexed aggregate per unfinished job in the local manifest: no store is opened, no vector is read, nothing goes over a network.

  **Read-only, including the local half.** Nothing is opened for writing, and the manifest is opened only when this release would not migrate its schema — `ManifestDB` upgrades on connect and takes a backup on the way, which is right for `status` and wrong for a command whose whole promise is that it changes nothing. A manifest from an older release is reported rather than upgraded. The schema is read out of the SQLite file header to decide that, so even the decision is a read.

  **Two of the four checks the roadmap named are not here, and the reason is the same rule.** *Chunking drift* needs a baseline: nothing records what the chunking was when the index was built, and a length distribution measured today cannot tell drift from a corpus that simply looks like that. The *"truncate instead of an adapter" advice for Matryoshka models* needs to know that truncation is adequate for this model on this corpus, which is a measurement nobody has taken — it is on the roadmap as an open question, and shipping it as advice would be guessing. The *encoding-profile check* is here in the form the records actually support: a `query_to_old` adapter maps into the index's own space, so an adapter recorded against this collection whose output dimension the index does not have was fitted against a different one. Where rebasis has recorded nothing for a collection, the check says so, which is a third answer and not a pass — a comparison that was never made is not a comparison that succeeded.
- `rebasis migrate --refit` refits the adapter part-way through a long migration, on records that have **not** been migrated yet, adopting the result only when it beats the adapter in use by 0.01 on a held-out slice. Off by default.

  `migrate/refit.py` has had the machinery and the adoption guard since it was written and nothing has ever called it. Connecting it found the premise it was built on to be wrong: its docstring said pairs become available "for free" during a migration because migrated records carry new-model vectors. They carry `A(old)` — the adapter's own image — so fitting on them fits `A` to reproduce `A`. Every real pair costs a document re-embedded, which is why `--refit` opens the new model recorded in the adapter's manifest.

  **216 cells say the sample source is the whole effect** ([the measurement](https://batuhanzorbeyzengin.github.io/rebasis/continuous-refit/)). On a corpus that has not changed, a refit is a pair-count effect: three times the fit budget moves retention a median +0.0075 and clears the adoption threshold in one cell in six, and where the pairs came from is worth −0.002, the wrong sign. On an index that **grew into a domain the adapter never saw**, 1,000 pairs drawn from what is left are worth a median **+0.16** and win in 12 of 12 cells — and beat 8,000 pairs drawn from the migrated half by **+0.20**, at every budget tested. Keeping the original fit pairs alongside the new ones makes it worse, because they pull the map back toward a domain that is no longer being written.

  The guard is what lets both readings be true at once: on an unchanged corpus the refit loses and is declined, on a drifted one it wins by sixteen times the threshold.

  An adopted adapter is written to `.rebasis/adapters/<job>-refit-<n>.rbs` and the job row is pointed at it, so a `rebasis resume` continues with what the refit gained rather than reloading the file the job started with. Every attempt is audited, adopted or not: "the refit was considered and declined" is the answer to "why did this job not improve".

  Also corrected: `refit.py`'s caveat that priority order biases the accumulated pairs. Measured, `--priority none` and an access-ordered migration give +0.0073 and +0.0080 — the caveat describes something real about the sample and nothing about the outcome.
- `rebasis migrate --shadow-precision float16` halves the shadow copy's disk cost. `float32` remains the default.

  `ShadowStore` has taken a `precision` argument since it was written and nothing ever passed `float16`, because halving the shadow gives up the bit-identical rollback guarantee and **a half guarantee may be more dangerous than no guarantee** — a user who believes `rollback` is exact and gets something else has been told a smaller truth than one who was told nothing. The option waited on a measurement of what the smaller truth is.

  **68 runs — seventeen corpora, four models, 256 to 768 dimensions** ([the numbers](https://batuhanzorbeyzengin.github.io/rebasis/shadow-precision/)). No vector overflowed the format. The top-10 **set** survives on 99.78% of queries at worst and 99.95% in the median run; nDCG@10 against human judgements moves by a median of 0.00000 and at most **0.0017**, which is inside ARR's own ±0.024 confidence interval and below the 0.01 that `RefitPolicy` declines to adopt on. What does move is the *order* within the top ten, on about 2% of queries in the median run — a float16 step is enough to swap two documents whose scores were already within 3e-04, and no index promises that adjacent pairs are further apart than that.

  Nothing claims bit-identity when it is on. The disk-space plan before the confirmation says the rollback becomes approximate, the shadow's own manifest records the precision, `rollback` prints it off that file — the one record that cannot disagree with itself — and `migrate.job.started` carries it into the audit trail. Nothing in the index says which precision a job used, so those two are where it survives.

  `--shadow-precision` rejects anything but the two values, before the adapter is read. `ShadowStore` treats an unrecognised precision as `float32`, which is right for a library and wrong for a flag: a typo would have silently given the safe behaviour while the user believed they had asked for the other.
- `rebasis migrate` catches SIGTERM and stops at the next batch boundary instead of being killed where it stands.

  A long migration does not run on a laptop with somebody watching it. It runs as a Kubernetes `Job`, an Airflow task or an Argo step, and all three end a process the same way — SIGTERM, a grace period, then SIGKILL. There was no handler for either signal, so the default applied: immediate termination, typically mid-batch, leaving the store holding records that were never read back and compared. The checkpoint made that survivable; it did not make it clean.

  The signal is now the same request `rebasis pause` makes, arriving from a different direction. The run stops at a boundary, records the job as paused, names the signal that asked, and `rebasis resume <job-id>` picks it up. SIGINT is caught too, so Ctrl-C asks rather than aborts.

  **The second signal is not caught.** The handler restores the default before it returns, so a supervisor escalating still stops the process at once — a graceful stop that cannot be interrupted is a hang with better manners.

  **The grace period has to outlast a batch.** A batch that takes longer than `terminationGracePeriodSeconds` is still killed part-way and no handler changes that; raise the period or lower `--batch`. The new [production guide](https://batuhanzorbeyzengin.github.io/rebasis/guides/operations/) has the numbers, along with exit codes, secrets, offline installation and what the state lock does and does not coordinate.

  The CLI installs the handler, not `MigrationEngine`. A library caller keeps their own signal handling: taking SIGTERM away from the application embedding you is not a library's to take.
- `rebasis pause <job-id>` stops a running migration after its current batch, and `rebasis resume <job-id>` starts it again.

  Killing the process was always safe — the queue is the checkpoint and a shadow copy is written before the vector it copies is overwritten — but it lands mid-batch, and the read-back that verifies a write is a per-batch guarantee that half a batch does not get. `pause` returns immediately and the job stops at the next boundary.

  It takes no lock, because the migration it is interrupting holds the state lock for its whole run and a command that waited for it would wait for the thing it is trying to stop. What makes that safe is that it writes one column nothing else writes: `jobs.pause_requested`, new in manifest schema 3, is a **request**. Only the engine ever says what state a job is in — writing `PAUSED` from a second process would claim a stop that had not happened and race the engine over the same column.

  `status` shows an outstanding request as `running (pausing)` and carries it in `--json` as a separate `pause_requested` field, so a script branching on `state == "running"` keeps working. A request never outlives the run it was made for: it is cleared when a run ends and again when one starts, so a process killed between `rebasis pause` and the engine reading it cannot leave a flag that silently pauses the next run.

  `resume` forwards to `migrate --resume`, which is unchanged. Only the flags that describe *this run* are accepted; `--priority` and `--access-log` are not, because they order the queue, the queue was ordered when the job was created, and re-ordering half a migration would be a different job.
- `rebasis probe --access-log` weights which sampled records become query proxies, so ARR describes retention on the questions people actually send rather than on a uniform draw over the corpus.

  The sampler has taken weights since it was written and nothing passed them. Connecting them found that the roadmap entry naming this named the **wrong place** for them: a `probe` sample does two jobs at once — it is the mini-index every measurement runs against, and it is the pool the query proxies are split out of. Handing weights to the sampler fills the mini-index with frequently-read documents, which changes the *distractors*, a property of the index rather than of the questions asked of it. The weights go on the split.

  Measured over 36 cells and 12,960 replicate probes, that placement leaves the estimate about half as far from the whole-corpus quantity as weighting the sample does (+0.025 against +0.051) ([the numbers](https://batuhanzorbeyzengin.github.io/rebasis/access-weighting/)).

  **The confidence interval survives it**, which is what the entry was blocked on. Dividing the bootstrap's median half-width by the estimator's actual spread across replicates gives 1.92 for the plain design against a correctly calibrated 1.96, and 1.84 under weighted queries — about 6% narrow, in the direction the entry worried about and small against decision bands 0.10 wide. Median coverage is unchanged at 0.94; what moves is the tail, from 2 cells under 0.90 to 6.

  That check needed one correction on the way: the ratio is read against **1.96**, not 1, because a correctly calibrated 95% interval around a roughly normal estimator is exactly that many standard deviations wide. Read against 1, a correct interval looks twice too wide.

  Weighting shifts ARR by a median +0.015 at a 100x access ratio and by up to +0.073, so it estimates a different quantity — and the run says which. `probe --json` carries `access_weighted`, both report formats say so in prose, and a log that names nothing in the sample reports `false` rather than claiming a weighting that did not happen.

  Also: a third measurement fell out of the same grid and belongs to the default rather than to the flag. A 4,000-document mini-index is an easier place to retrieve in than the corpus it came from, so today's uniform `probe` already sits **+0.048** above the whole-corpus quantity — a larger gap than anything weighting does.
- `rebasis.serve.MixedSpaceSearch` — search an index that a migration left holding two embedding spaces. Until now the only honest advice for that window was "do not search it": a bridged query mis-scores the records that already moved and a raw one mis-scores the rest. This sends both and keeps only the half each is right about, merging them through the isotonic calibrator already in the `.rbs` — the `calibrated_merge` code that had been in `serve/hybrid.py` since the design with nothing to call it. Which records moved comes from the **manifest**, not from a `rebasis_space` field written into your payloads: the store contract is one write path that only ever replaces vectors, and the migration queue already knows. The cost is over-fetching, reported per query as `search.over_fetch` and bounded so a 2%-done migration produces a short result rather than a slow one.
- `rebasis.serve.cascade.Cascade` serves the two-stage arrangement `probe` has been reporting: the bridge produces a candidate set, and the new model reranks it in its own space. That is bounded by recall@N rather than by nDCG@10, which is the weaker requirement — measured across 48 runs on sixteen corpora, single-stage bridging beat keeping the current model in 1 and this arrangement beat it in 36, landing within two percent of a full reindex in all sixteen runs of the upper rung ([the measurement](https://batuhanzorbeyzengin.github.io/rebasis/cascade-band/)). Nothing about the adapter improved and ADR 10 is untouched; a different quantity does the bounding.

  **A cache is part of the design, not an option.** Re-embedding the candidate set is on the hot path — about 0.2 s per 100 documents at an A10G's measured rate for bge-base, seconds on a laptop CPU — and it is the cache that makes this a lazy migration rather than a permanent tax: the documents people actually retrieve get embedded once and stay embedded. There is an in-memory LRU by default and a `DiskVectorCache` that survives a restart, under the `.rebasis/cache/` that `gc` already collects on a 30-day retention and that `REBASIS_CACHE_DIR` already names. Every key carries the new model's `EncodingProfile` fingerprint, because a stale vector from the previous model would not raise — it would rank. A cache write that fails is counted, never raised: the search has already succeeded by then.

  `Cascade.stats` reports what the arrangement costs — cache hit rate, documents embedded, and the split between mapping the query, searching the index and reranking, with the embedder's own share called out. `docs/cascade-band.md` named the cache's behaviour under a real query distribution as the thing nobody had measured; that is a property of a query log rather than of a corpus, so the only place it can be taken is a running system, and this is the instrument for taking it. `probe` still reports the arrangement without recommending it.

  Records the store holds no text for cannot be re-embedded. They keep the rank the bridge gave them and the reranked documents flow around them, rather than being dropped — `probe` may drop a sampled record with no text because a sample is allowed to come back smaller, and a result set is not. `Cascade.stats.kept_bridged` counts them, and a store that can return no text at all is refused at construction rather than on the first query.
- `tail_arr` is computed. The `caution` band's rationale told users to look at it, the README said it was in the report, and nothing produced it. It is recall in the sparsest clusters — the signal that drift is concentrated in part of the corpus rather than spread evenly.
- `tools/bridge_band.py` measures a fifth configuration: the bridge used as a **recall stage**, with the new model reranking its candidate set in its own space. This is the arrangement `docs/bridge-band.md`'s ceiling does not cover — that band is measured at nDCG@10 with the final ranking produced in the old space, and a two-stage arrangement produces it in the new one, so its only loss is a relevant document that failed to reach the top N. `--cascade 100,200` measures it; `tools/bridge_band_report.py --view cascade` reads it back against both alternatives a user actually has.
- `tools/bridge_band.py` reproduces the four-configuration measurement behind `docs/bridge-band.md` — status quo, naive swap, bridged, full reindex — over any ir_datasets corpus and model pair, scored with `ranx` through the shipped `fit` and `Bridge` path. The 62 runs that decision rule rests on were previously not reproducible from the repository at all. `--k` takes a list of cut-offs rather than the single 10 those runs used, and `tools/bridge_band_report.py` reads the rows back as the band table, a recall-at-N table, or the aggregate claims. It also reads MMTEB's own layout from Hugging Face (`--corpus mmteb`), so the three tasks arXiv:2510.13406 reports its cross-model grid on — hard-negative HotpotQA and FEVER, and TREC-COVID — can be measured against the same ladder as everything else.
- nDCG@10 is computed in the core path and reported alongside recall. It moved out of the optional `[eval]` extra because it changes decisions: on BEIR/scifact an adapter improved recall@10 while nDCG@10 fell, and a recall-only rule read that as an improvement.

### Fixed

- **The sqlite-vec backend read every `vec0` column as float32, and on an `int8` table every number derived from a vector was wrong.**

  A `vec0` column declares an element type and the three are not equivalent: `float`/`f32` spends four bytes per component, `int8`/`i8` spends one, `bit`/`b1` spends one *bit*. `dimension()` divided the blob's length by four unconditionally, so an int8 table reported a quarter of its true dimensionality — and `iter_records` handed back vectors assembled from four components' bytes at a time. Nothing raised. The count was right, the ids were right, the text was right.

  Measured against the shipped extension first, and the measurements shaped the fix rather than decorating it. `float[8]` stores 32 bytes, `int8[8]` stores 8, `bit[8]` stores 1. `bit[7]` and `bit[12]` are legal declarations, so a one-byte bit blob is consistent with any dimension from 1 to 8 — the number is **not in the data**. And inserting a float32 vector into a narrow column is refused by the extension, as is querying one with a float32 vector.

  That last measurement decides the shape. rebasis produces float32 and nothing else, so `migrate` and `rebasis.Bridge` were never going to work on a narrow table — and the honest way to say so is the capability declaration, before a job is opened. `can_upsert_vectors` is now `False` for `int8` and `bit`; `can_read_vectors` is `False` for `bit`. `search` and `upsert_vectors` refuse with the element type named, rather than letting a SQL error surface as "the sqlite-vec query failed". `probe` on an int8 table keeps working and is now correct: one byte per component decodes to a direction, and the scale quantization removed is a single factor that normalisation takes out.

  Also fixed, and the same shape one layer up: `open_store` read the dimensionality **only to put it in a log line**, so a backend that could not answer failed to open. That already broke on an empty `vec0` table, where `dimension()` correctly refuses to guess. A log line does not decide whether a store opens — `rebasis doctor --store` exists to report on an index somebody is already confused by.
- A rejected batch is retried and then split, instead of failing whole.

  `StoreWriteFailed` declares itself transient — a store that refused a write because a node was rebalancing usually takes it a moment later — and `retry_transient` was written for exactly that and called from nowhere. It is now wired onto the migration's write: three attempts, exponential backoff with jitter, every attempt after the first logged.

  When retrying does not help, the batch is halved and each half written separately, recursively. Before this a rejected batch was marked `FAILED` whole, so one oversized payload or one id the store would not take cost its two hundred and fifty-five neighbours a place in the failed list and a second pass on the next `resume`. Nothing was lost — the queue is the checkpoint — but the operator had 256 records to look at instead of one.

  Two bounds keep the cost honest, and both were set by measurement rather than taste. **Splitting stops after four levels**, because a store that is simply unreachable fails every half and splitting all the way down costs 511 writes to learn what the first one already said. **The halves are not retried**: with the retry on every node, isolating one bad record from a batch of sixteen took 23 seconds, almost all of it backing off from a refusal the batch's own three attempts had already settled. Removing it took the same test file from 216 seconds to 22.

  `docs/guides/migration.md` said the job stops when a batch fails. It does not, and did not — the loop continues to the next batch and a run finishes with a count of failures rather than at the first one. That section now describes the retry, the split, and both bounds.

  A run's `processed` count is now what landed rather than what was offered. Those were the same number while a batch was all-or-nothing; they are not any more, and reporting the batch size would have had a run claim it processed records the queue holds as `FAILED`.
- A report written with `--report` is created with the permissions an ordinary write would give it. Routing it through the atomic writer had it inherit 0600 from the temporary file underneath, which is the right default for rebasis' own state directory and wrong for a file the user named and may want to serve or send.
- ARR's confidence interval is now an interval for ARR. It bootstrapped only the numerator, so at T1 — where the oracle is imperfect — the point estimate could fall outside its own range (`ARR 0.908`, `CI 0.712-0.808`). Replaced with a paired ratio bootstrap.
- CI lints `tools/`. The pre-commit hook always did and CI did not, which is the same divergence `ci.yml` already documents one directory over — and a fork's pull request never runs the hook.

  It is not a quiet directory: `tools/check_citations.py` runs as a step in that very job, and `tools/bridge_band.py` is what every published measurement comes out of. The release workflow's paths are updated to match, because they diverging once is how the gap in `examples/` survived.
- Chroma's dimension-lock error named `--direction query_to_old`, a flag that does not exist. It now says what to do instead of naming a way to do it.
- Each CI job gives `setup-uv` its own `cache-suffix`. Every job shared one cache key, so whichever finished second lost the race to reserve it and logged `Failed to save: Unable to reserve cache with key ...`. The test job's suffix carries its matrix leg too, since its three legs had the same problem among themselves.
- Every GitHub Action in every workflow is pinned by commit SHA instead of by tag, and every workflow declares a read-only default token.

  A tag is a mutable pointer. `actions/checkout@v7` is whatever the `v7` tag names *today*, and whoever can move that tag can run their code in this repository's CI — with whatever permissions the workflow granted. A commit SHA cannot be moved. The tag stays beside each pin as a comment so a reader can still see what version it is, and Dependabot updates both together.

  Forty of the forty-nine findings OpenSSF Scorecard reported were this, across seven workflow files. Two more were the token: GitHub's default `GITHUB_TOKEN` is write-scoped unless a workflow says otherwise, so `ci.yml` and `gpu-tests.yml` (which is not in a clone — it targets the project's own GPU host and is gitignored) were granting every job write access to this repository for no reason. Both now declare `contents: read` at the top, and the jobs that need more say so themselves — the release job's `contents: write`, the Scorecard and CodeQL jobs' `security-events: write`.

  `codeql.yml` is new, and answers the SAST finding with something worth having rather than with a checkbox. `ruff` already runs every rule, which includes flake8-bandit's security lints — but those are pattern checks that see a dangerous call where it is written. CodeQL follows values through the program, so it sees the untrusted input three functions away from the call that consumes it, which a linter structurally cannot. Python needs no build step; it runs on every pull request and weekly, because a query-pack update can surface a finding on code that has not changed.

  The findings this repository cannot close are named in `SECURITY.md` rather than worked around: Code-Review counts pull requests reviewed by somebody other than their author and there is one maintainer, Branch-Protection is a setting that would block the release workflow's own push until a rule allows it, Fuzzing has none, and CII-Best-Practices wants a badge nobody has applied for.
- Every deliberate non-zero exit was reported as an internal crash. `typer.Exit` subclasses `RuntimeError`, so the CLI's top-level `except Exception` caught it, printed "This is a bug in rebasis" with a pre-filled issue link, and replaced the intended exit code with the unexpected-error one. Answering "no" to the `migrate` confirmation was the most visible case: declining to write to your index told you the tool had crashed.
- Every error panel pointed at `docs/reference/errors.md#rb-exxxx`, a repository path that a `pip install rebasis` user does not have and an anchor the docs site does not define. It now names the published page.
- FAISS: a write reorders the index — `remove_ids` closes the gap and `add_with_ids` appends — so the metadata sidecar, which is a plain list in row order, was left naming the wrong vectors after the first `migrate`. The sidecar is now rewritten in the new order with the index. An index holding the same label twice is refused on open rather than resolved arbitrarily.
- FAISS: an `IndexIDMap2` is addressed by label, not by row number, and the backend was passing row numbers to `reconstruct_batch`, `remove_ids` and `add_with_ids`, and indexing the sidecar with the labels `search` returns. On any index whose ids are not exactly `0..n-1` that read the wrong vector or raised. The two are now kept apart throughout, and the fixtures use labels that are deliberately not row numbers.
- Four of the five store backends leaked their client library's exception when the store could not be opened.

  `errors.py` states the rule in its own module docstring: *"Third-party exceptions never cross a module boundary. Each backend catches its own library's exception, converts it to a `RebasisError` subclass, and keeps the original as `__cause__`. Contract tests enforce this."* They did not enforce this half of it.

  The half that was covered is a store that opened and then refused something — a dimension mismatch, a missing collection. The half that was not is the store that does not open at all, which is the more common thing to get wrong on a first run: a path typed with a missing directory, a database owned by another user, a volume that is not mounted yet.

  Measured against a path that exists as a parent and refuses everything under it. Only FAISS converted. Chroma raised `chromadb.errors.InternalError`, LanceDB and Qdrant raised `FileNotFoundError`, sqlite-vec raised `sqlite3.OperationalError` — each reaching the caller with no `RB-Exxxx` code, no hint and nothing to look up.

  All four now raise `StoreError` (`RB-E3000`) naming the path, with the original kept as `__cause__`. The code is deliberately not `RB-E3003`: a database that cannot be opened is a different problem from one that opened and does not hold the collection you named, and a user told the wrong one looks in the wrong place.

  Found by running `rebasis doctor --store` against a bad path and reading what it printed. The tool diagnosed itself — beside the leaked exception it printed its own note that *"a backend is meant to convert its client library's exceptions into a rebasis error, so this one is a bug"*. It was.

  A contract test now covers it on every backend. Its own first version derived the import name from the backend name, asked for `chroma` and `qdrant`, and skipped both while they were installed — which is the failure `ci.yml` greps for, and the reason it does.
- GitHub Actions are pinned to versions that run on Node 24: `checkout@v7`, `upload-artifact@v7`, `gitleaks-action@v3`, `upload-pages-artifact@v5`, `deploy-pages@v5`, and `setup-uv@v10.0.1`. The last is pinned to a full version on purpose — setup-uv stopped publishing floating major tags at v8, so `@v10` does not resolve.
- Install hints reached the terminal with their extra removed. `rich` reads square brackets as a style tag, so `pip install "rebasis[chroma]"` — printed by `MissingDependency`, which is the first error many users will ever see — rendered as `pip install "rebasis"`, telling them to install what they already had. Everything taken from an exception is now escaped before it is placed inside markup, so a message, hint or context value containing a bracket survives: a path, a document id or a URI is data, not markup.
- Removed the Apache-2.0 trove classifier. PEP 639 makes it mutually exclusive with the `license = "Apache-2.0"` expression the project already declares, and PyPI rejects the combination on upload.
- Running a command that needs confirmation without `--yes`, where stdin is not a terminal, is a usage error naming the flag that fixes it. It used to reach the unexpected-error boundary: `typer.confirm` raises `Abort`, which is not `typer.Exit`, so a normal invocation in a script or a CI job was reported as a bug in rebasis, complete with a pre-filled issue link and exit code 1. `migrate` and `rollback` also take `--no-input`, which refuses to prompt even on a terminal.
- The CI test job installed no extras, so every store backend test opened with `pytest.importorskip` and reported success having run none of them — about seventy tests, including the whole migrate-and-rollback suite. The job now syncs all extras, runs the end-to-end suite as well, and fails if any test is skipped for a missing dependency. The two jobs that build a deliberately restricted environment now invoke `pytest` directly, because `uv run` re-syncs from the lockfile and was replacing the environment each of them existed to test.
- The OpenTelemetry attribute catalogue now describes what rebasis emits rather than what it might.

  Three names were defined and never used, which for a module whose whole purpose is "these are the names we send" is a catalogue that misdescribes itself. `db.system.name` — one of the few names in this area that is *stable* rather than development-status — now goes on the store-upsert span, carrying the backend's own declared name. There is no vector-database semantic convention to conform to (the upstream issue asking for one is open and unassigned), so rebasis puts the backend in the standard field and invents no `db.vector.*` namespace of its own. `gen_ai.embeddings.dimension.count` goes on the embedding span, because the dimension is the one number that explains a duration: the same corpus through a 384-wide model and a 1024-wide one is the same `count` and a very different wait. `rebasis.migrate.state` goes on the migration span at the end, with the pause reason beside it, so a trace can be filtered to the runs that stopped.

  `semconv_opt_in()` is gone. It parsed `OTEL_SEMCONV_STABILITY_OPT_IN` and nothing branched on the result — the variable exists so a library can emit an old attribute name and a new one during a migration period, and this project has never emitted an old one. A reader is entitled to assume a function that parses a standard environment variable does something with it.

  `gen_ai.provider.name` is documented as deliberately absent rather than emitted as a guess. The honest value is the embedding backend, and the `Embedder` protocol does not carry the entry-point name it was opened under; adding one is a change to make for a real reason rather than for conformance to a namespace that is still development-status.

  Separately, `REBASIS_ENV` is in the table `rebasis doctor` prints. It was read by the logging setup and absent from the list whose docstring calls itself "every variable rebasis reads" — so the one variable an orchestrated deployment most wants was the one that had to be found by grepping.
- The `lowest direct dependencies` CI job passes again. click 8.5.0 turned `click.utils.get_binary_stream` into a deprecated alias, and every typer up to 0.25 imports it by that name — so pairing the declared typer floor with a current click raised a `DeprecationWarning` during collection, which this project's `error::DeprecationWarning` policy turns into ten collection errors.

  Measured across releases: 0.16 through 0.25 warn, 0.26.0 is the first that does not. The floor is left where it is and the warning is filtered, for the reason the chromadb line beside it gives — raising it would withdraw support for ten typer releases over a notice in someone else's code, and typer 0.16 runs correctly. No other job sees it: the lock resolves typer 0.27.1.
- The memory ceilings gate a pull request again, which is what three separate documents already claimed they did.

  `benchmarks/README.md` said `test_memory_ceiling.py` runs on "every PR" with an "absolute — exceeding blocks" gate, and closed with "Only the absolute-memory ceilings are gates." None of it was true. Every test in that file, and every test in `test_hot_path.py`, carried `perf`; CI runs `-m "not network and not perf"`. So the one mechanism that catches the `O(N × d)` class of bug — a `list(iter_records())` that fails only on corpora nobody has in development — ran on no pull request at all. `test_macro_budgets.py`'s docstring pointed at an instruction-count benchmark as the real gate; there was none, the CodSpeed job that would have been it collected nothing and was removed.

  The marker now follows what a test **asserts on** rather than which directory it lives in. `perf` means it asserts wall clock, which a shared runner cannot measure — this suite has two red Xs at 1% and 2.6% to prove it — so it stays on the measurement host. The new `memory` marker means it asserts peak allocation, which `tracemalloc` measures identically on a loaded runner and a quiet one, so CI gates on it. Nine tests moved: seven ceilings and the `O(batch × d)` scaling invariant, plus the two hot-path checks that count bytes per call rather than microseconds. `test_time_grows_but_not_super_linearly` stayed `perf`, being the one test in that file that asserts on seconds.

  A test must not carry both markers, and `benchmarks/README.md` no longer tells contributors to mark every new benchmark `perf` — that instruction is what would quietly re-open the hole. `just gate` runs the layer CI gates on; `just bench` runs the wall-clock layer.
- The release tooling is in `uv.lock` instead of being resolved fresh on every run.

  `release.yml` called `uv run --with python-semantic-release`, which bypasses the lock entirely and takes whatever is newest at that moment. GitPython 3.1.60 removed `Actor.name_email_regex`; python-semantic-release 10.6.1 reads it to parse `commit_author`. So the release workflow broke on a repository where nothing had changed, with `type object 'Actor' has no attribute 'name_email_regex'` and no version printed.

  **A release path that can break from somebody else's upload is not a release path.** The tooling is now a `release` dependency group, pinned by the lock like everything else, with `gitpython<3.1.60` and the reason recorded next to it. Dependabot proposes the upgrade as a pull request rather than applying it silently at the worst moment.

  `ci.yml` gains the other half: a ten-second step on every pull request that asks the tooling to work out a version and fails if it cannot. The breakage was invisible until somebody wanted a release; now it is visible on the change that introduces it.

  Found by `mode: dry-run`, which is what that mode is for — it printed the error and stopped, having touched nothing.

  `ci.yml`'s own first attempt at that check then failed for a second, unrelated reason, and the fix is worth recording because it is not a workaround. On a pull request `actions/checkout` leaves a **detached HEAD** at the merge commit, and python-semantic-release refuses to match a release group there — "Detached HEAD state cannot match any release groups". The step now points a local `main` at that commit before asking: the merge commit is exactly what `main` would become, which is the state the check is about.

  It uses `version --print` rather than `--version`, and that was measured rather than assumed. Against the GitPython that broke the release path, `--version` exits 0 and would have caught nothing; `version --print` exits 1.
- The release workflow assembles the changelog. It did not, and a release cut before this would have shipped an unchanged `CHANGELOG.md` with every news fragment still sitting in `changelog.d/`.

  `semantic-release` was called with `--no-changelog` and a comment explaining that towncrier owns the file — which is right, and the reason two writers on one file is the wrong design. Nothing then called towncrier. Neither `release.yml` nor `docs/development/release.md` had the step; the `justfile` had only `--draft`, which previews without writing.

  The assembly is now the release commit, so the tag lands on a tree that contains the changelog rather than one commit behind it. `verify` also refuses to release at all when `changelog.d/` is empty — an empty directory means either nothing user-visible changed, or somebody forgot the fragment, and the second is worth catching.

  The version and the tag are read from `semantic-release version --print` and `--print-tag`, both of which exit before touching anything, and the tag is then placed by `git`. The commit messages still choose the version and `tag_format` still decides the tag's shape; what changed is that the tagging step no longer depends on how the release tool behaves when there is no version file to rewrite — this project has none, since `hatch-vcs` derives the version from the tag.
- Three declared dependency lower bounds were wrong, and the CI job meant to catch that was not testing them. `qdrant-client>=1.9` did not have `query_points`, which the backend calls for every search; `faiss-cpu>=1.8` was built against numpy 1 and cannot import alongside the numpy 2 this package allows; `chromadb>=0.5` has no wheel for the declared minimum Python. Each floor is now the oldest version its integration suite actually passes on, checked by running it there.
- Three wrong dependency floors, each of which installed a package that does not work. `scipy>=1.11` resolved to 1.11.1, which has no cp312 wheel against a `requires-python = ">=3.12"` package, so the install ended in a meson source build; the floor is 1.11.2. `typer>=0.12` was wrong three times over: 0.12 cannot convert a `Path | None` option annotation, everything through 0.15.2 calls click's `Parameter.make_metavar()` without the `ctx` argument click now requires, and 0.15.4 caps `click<8.2`, where `CliRunner` still merges stderr into stdout — so the separation `--json` depends on could not even be asserted. The floor is 0.16. Nothing is wrong with chromadb 0.5.5 itself: it reads `model_fields` off a pydantic instance, which pydantic 2.11 deprecated, and only the lowest-direct job pairs the oldest chromadb with the newest pydantic. That warning is now ignored the way torch's and transformers' already were, rather than withdrawing the 0.5.5 support the README documents over a deprecation in someone else's code.

  The job that exists to catch all three had been dying in its install step before it could report any of them.
- Two guards on the serving paths that a half-migrated or mis-configured index reaches, and the tests for the one branch that had none.

  **`MixedSpaceSearch` requires one width, and now says so at construction.** It sends the *unmapped* new-model query at the index beside the bridged one, so the new model's width has to be the index's. That is not an arbitrary restriction: a partial migration that changed the width would leave two vector widths in one collection, which every `dimension_locked` store rejects outright and which no single query could search anyway. The arrangement is impossible before it is unsupported. Checked when the searcher is built rather than on the first query, because a caller who installs it at start-up should find out then.

  **`Cascade` refuses an embedder that does not honour its own profile.** The rerank cache is keyed on the encoding profile's fingerprint, which covers `dim`, so one namespace is supposed to hold one width — and `_rerank` stacks whatever the cache returns into a single matrix. A hand-set `--new-dim` that does not match the model, or a model id that started resolving to different weights, surfaced as `all the input array dimensions ... must match exactly` from inside numpy. The check sits where vectors enter the cache and names both numbers.

  **And the calibrated merge is under test.** `MixedSpaceSearch` merges through `calibrated_merge`, which has two branches; every test of the class built a `Bridge` with no calibrator, so the branch that runs whenever somebody loads a real `.rbs` was the untested one. That is where the tie-break defect lived — at the endpoints of a migration the merge reproduced the store's own ranking on 4% to 16% of queries against rank fusion's 100%. The new tests fail on the reverted fix, which is the only evidence that a regression test is one.
- `--report` wrote through `write_text`, which truncates before it writes: a full disk or a Ctrl-C at the wrong moment destroyed the previous report, and a probe run is minutes of embedding to reproduce. It goes through the atomic writer now, like every other file rebasis owns.
- `Bridge.to_index_space` no longer rewrites the caller's query vector when the adapter is `identity`.

  `to_index_space` normalises its result in place, deliberately: it is one allocation off a path budgeted at 15 µs. That is safe only while every adapter returns a new array, which every adapter that multiplies does for free. `IdentityAdapter` has nothing to multiply by — it handed back the input unchanged whenever the widths already matched, so the normalisation landed on the caller's own array.

  Measured: `bridge.to_index_space(q)` left `q` normalised. A caller reusing `q` for a second index, for a rerank, or for a log line was working with a different vector from the one they encoded, and nothing raised. Two paths reach it — `fit --method identity` and loading a `.rbs` that records that type — and `auto` never selects that adapter, which is how it survived.

  `BaseAdapter.apply` now states the contract the other implementations were already keeping, and a property test asserts it across identity, Procrustes, centred Procrustes and linear: `apply` must not return the caller's array or a view of it, and a full serving call must leave the input unchanged.

  Found while writing down whether `Bridge` is safe to share between threads. It is — it is immutable after `load`, holds no per-call state, and needs no lock — and that is now in its docstring, along with why there is no `async` variant.
- `migrate`, `rollback` and `gc --apply` take the exclusive state lock the module has always claimed they take. It was implemented, documented and never called: two concurrent runs against one state directory were not prevented. `gc`'s dry run stays outside the lock — "what would you delete?" is a read, and making it wait behind a running migration would be a worse answer than showing it.
- `probe --json` and `eval --json` emit JSON and nothing else, even when the optional energy backend is installed. Zeus probes for an AMD SMI library at startup and prints the dlopen failure straight to stdout, so on a runner with the CPU torch build `rebasis probe --json | jq` received `/opt/rocm/lib/libamd_smi.so: cannot open shared object file` and then the JSON. Its loggers were already silenced; that line never went through logging. Stdout is now taken away from it for the whole window, import included, and a test installs a stand-in that narrates the same way so the guarantee is checked on machines that do not have zeus.
- `probe` no longer tells a run that was given `--queries` to supply `--queries`. When the confidence interval straddled a decision threshold the remedy was printed unconditionally, so a T1 run — one already using a real query log — was advised to do the thing it had done. It now says only "Increase --sample" once the query log is there.
- `rebasis doctor` reports when faiss and torch are both installed on macOS, because that pair cannot share a process: each wheel links its own OpenMP runtime and the second to initialise aborts with `OMP: Error #15`. Measured directly — faiss alone runs a reconstruct and sixty searches without complaint; the same script with `import torch` in front of it dies before the first call. Neither wheel can be told not to link it, and the workaround the error suggests, `KMP_DUPLICATE_LIB_OK=TRUE`, is documented as liable to silently produce incorrect results ([faiss-wheels#40](https://github.com/kyamagu/faiss-wheels/issues/40), [pytorch#149201](https://github.com/pytorch/pytorch/issues/149201)). So it is reported rather than worked around: better learned from `doctor` than from an abort halfway through a migration. The FAISS tests skip themselves in that combination with the same reason attached, which is why the macOS CI job had been aborting, segfaulting and hanging by turns.
- `rebasis migrate --resume <job-id>` works with just the job id. It required `--adapter` and `--store` to be passed again, which is the documented behaviour reversed: a migration is resumed after an interruption, and the queue being the checkpoint is supposed to mean nothing else has to survive it. `store_uri` was already on the job; `adapter_path` was a column that had existed since schema 1 and was written as the empty string every time, so it is now filled in. A job created before this release has no adapter path recorded and `--resume` will ask for `--adapter` once more. Passing either explicitly still wins, so resuming with a different adapter stays an error you can see rather than one that is silently overridden.
- `rebasis status` showed "available" in the rollback column for a job that had already been rolled back, offering something that was gone. It now reports availability only while a rollback is actually possible.
- `rebasis[all]` now includes the ollama and OpenAI-compatible embedders, which have registered entry points and so were listed by `rebasis doctor` while being impossible to open. torch, llama-cpp, langchain, llamaindex and energy stay out on purpose; the reason for each is in `pyproject.toml`.
- `reciprocal_rank_fusion` and `calibrated_merge` break ties on the document id rather than on which result set was passed first. Every result set starts at rank 0, so each one's best hit scores exactly `1/(rrf_k + 1)` — a first-place tie is the common case, not an edge one — and a stable sort over a dict filled set by set handed every one of them to the first set. On a half-migrated index that was a standing bias toward one embedding space, which is precisely what fusing by rank exists to avoid. Found by the integration suite for `MixedSpaceSearch`, where the no-calibrator path is the default.
- `rollback` printed a design-document section reference in its preview. That document is not in the repository, so the reference pointed at something the reader cannot open — the thing the output-hygiene tests exist to prevent, in one of the few lines they did not cover. They cover every printed line now, not only `--help` and `doctor`.
- `serve.calibrated_merge` no longer reorders results within one embedding space, and at the endpoints of a migration it now returns what the store returned.

  The calibrated merge sorted on `(-score, id)`. That tie-break was added deliberately and its reasoning was right as far as it went: two hits from *different* sides at the same score must not be separated by which side was passed first, because on a half-migrated index that is a standing bias toward one embedding space. What it did not cover is a tie *within* one side, and those turn out to be the common case rather than the rare one. The isotonic calibrator is a step function — pool-adjacent-violators produces far fewer levels than it has inputs, and clipping flattens both tails — so ten distinct bridged scores land on **five to seven** distinct calibrated values, and every collision was then resolved alphabetically.

  What that cost is clearest where the answer is not in doubt. At 0% and 100% migrated the collection holds one space, so there is a single correct ranking: the one the store returned. Measured across four corpora, the calibrated merge reproduced it on **4% to 16%** of queries. Reciprocal rank fusion, which never looks at a score, reproduced it on **100%** — its scores are a strictly decreasing function of rank, so it cannot tie. A merge that cannot reduce to the single-space answer is wrong at the endpoints, not merely worse there.

  The sort now carries each hit's rank on the side that produced its score: `(-score, rank, id)`. Two hits from the same side hold different ranks, so a shared calibrated level keeps the order it arrived in; two hits from different sides can share a rank, and there the id still decides, which is the neutrality the original tie-break existed for. Three tests pin it, including one that constructs a genuine cross-side tie by feeding the new side exactly the value the calibrator produces for the old.

  The nDCG cost of the old behaviour was small — between −0.002 and +0.002 at 0% migrated on all four corpora, because documents sharing a calibrated level had similar scores to begin with. It is fixed because a merge should be able to reduce to the single-space case, not because the measurement was alarming. `docs/concepts/adapters.md` said the calibrator preserves ranking "exactly"; that is true of the transform and was not true of the sort around it, and it now says which.
- `sqlite_vec` was registered as a second store backend rather than as a spelling of `sqlite-vec`, so `rebasis doctor` reported seven backends where there are six and named one that appears nowhere in the docs. The underscore form still resolves; it is an alias now.
- `tail_arr` is suppressed when the sparsest clusters hold too few queries to mean anything. Measured on real corpora: at T0 the tail held 42-59 of 1,000 held-out proxies, but at T1 it held **six** — a real query log concentrates on documents people actually ask about, and those sit in dense clusters. Six queries all missing produced a `tail_arr` of exactly 0.000 that read as catastrophic heterogeneous drift.
- `tests/performance/test_macro_budgets.py` is marked `perf` as well as `slow`, which takes its wall-clock gates off the pull-request path.

  The file's own docstring said "wall clock never blocks a PR", and it did. CI runs `-m "not network and not perf"`, so `slow` alone was not enough to exclude it, and six budget assertions — 20s, 90s, 180s, 360s, 30s, and 50ms for an adapter load — were gating merges on a shared runner.

  `ci.yml` already carries the reasoning, written when the same thing happened to the `perf` layer: *"the perf layer asserts timing, and a shared runner cannot measure timing: this job failed twice on `test_batching_amortises_the_per_call_cost`, by 1% and by 2.6%, which is noise wearing a red X."* That argument is about what a test measures rather than which marker it happens to carry, and it applies here exactly — it was simply not reached, because the exclusion is keyed on `perf` and this file was `slow`.

  Observed rather than anticipated: on a host running two other jobs, the residual-MLP fit blew its 180-second budget; measured alone on the same machine minutes later it took **17 seconds**, a tenfold swing with nothing in the code changed. That is the second timing assertion to fail this way in one day — `test_the_centred_adapter_folds_its_offset` was the first, and it has since been split so that the property it protects is checked algebraically instead.

  Nothing is lost. `slow` still keeps the file out of the default developer run; `perf` keeps it off the merge path; `scripts/remote.sh test` runs `-m "not network"` and so still executes all six on the project's own host, which is the machine whose hardware their numbers describe.

### Performance

- The CI test job runs the suite once instead of twice. Collection imports torch and every store client, and that cost is paid per pytest process, so splitting the run into "unit, property and contract" then "integration and e2e" paid it twice for an isolation the markers already provided.
- `l2_normalize` takes a scalar route for a single vector: 8.5 µs to 3.8 µs at d=768. `np.linalg.norm` re-derives its axis and dtype handling on every call, which at one query is most of the cost. The centred Procrustes adapter folds `μ_dst − μ_src·R` into a bias at construction, removing one full-length array operation per query. Together, `Bridge.to_index_space` at d=768 goes from 32.7 µs to 24.5 µs; at d=256 it now fits inside the 15 µs budget.

### Documentation

- **Per-cluster adapters were measured, and the roadmap entry proposing them named the wrong instrument.**

  The premise was that one global map leaves quality on the table where drift is heterogeneous, and that `probe`'s `tail_arr` — retention in the sparsest decile of clusters — detects that without being able to act on it. Half of that holds. Fitting k local maps instead of one does beat the global map, on **12 of 18** corpus/model cells at k=8, by up to +0.102. What decides whether it wins is not heterogeneity.

  Over those 18 cells the gain correlates with **corpus size at Spearman +0.900** (p < 1e-5) and with the `tail_arr` gap at **+0.046** (p = 0.87). The size relationship is monotone with a visible crossover: every cell on the two smallest collections loses (nfcorpus at 3,633 documents, scifact at 5,183), arguana at 8,674 is marginal, and every larger cell wins. The tail gap, meanwhile, never crossed its own warning threshold on any of the 22 corpus/model combinations surveyed for it — including collections assembled to be heterogeneous by reading two unrelated domains as one index (mathematica with gaming, COVID papers with English-language Q&A). Fusing two domains produces heterogeneous *content*; it does not appear to produce heterogeneous *drift*.

  Three costs came out with it. Per-cluster wins only where each cluster gets its **own full budget**: split one budget k ways and it loses in **17 of 18**, so this is "more fitting, and here is where to spend it" rather than "a better shape". Routing is not the bottleneck — nearest-centroid in the new space, through the bridge in the old, and an oracle assignment differ by about 0.01. And k was fixed at 8 throughout, which leaves corpus size and pairs-per-cluster confounded; separating them is the measurement that would settle the mechanism, and it has not been taken.

  Nothing ships from this. `probe` still reports `tail_arr` and still cannot act on it, and the roadmap item stays open with what it needs restated: not a way to detect heterogeneous drift, but a fit budget worth spending k ways and a rule for when the corpus is large enough to spend it. Recorded because a premise that was measured and found to name the wrong quantity is worth more written down than quietly dropped.
- **Unpaired alignment was measured, and it recovers most of what a paired fit does — on the rungs where its first stage works at all.**

  The roadmap has carried "fit an adapter with no correspondence between the two spaces" as a direction for some time, most recently pointing at [mini-vec2vec](https://arxiv.org/abs/2510.02348). `spikes/unpaired_align.py` implements it and runs it against the paired ceiling `rebasis fit` reaches on the same data: 36 cells over four corpora, three ladder rungs and three seeds, with the two halves sharing no document and the split asserted rather than assumed.

  Median recovery is **0.81** of the paired ceiling. Excluding the one cross-family rung it is **0.84**, floor 0.61, ceiling 0.94 — from a map given no pairs at all.

  **The failures are total and they are all in stage one.** `potion-base-8M → all-MiniLM-L6`, a 256→384 jump across model families, recovers between 0.00 and 0.66 depending on corpus, against 0.77–0.93 for every same-family rung. That is worth being able to *see* rather than infer, so the spike reports a **centroid-agreement** diagnostic: how often the quadratic assignment pairs a centroid with the one an oracle map — fitted on the paired data the unpaired fit is forbidden to touch — would have chosen. It is a reference rather than a truth, because two disjoint halves have no exact centroid correspondence, and it is labelled as one wherever it appears.

  It is also the best predictor available. Ranking the 36 cells by eventual recovery:

  | signal | Spearman ρ | p |
  |---|---|---|
  | centroid agreement | **+0.833** | 3e-10 |
  | ICP final objective | +0.628 | 4e-05 |
  | QAP objective — *what the method itself reports* | +0.519 | 1e-03 |
  | orthogonality error of the fitted map | −0.231 | 0.17 |
  | pre-fit geometry bound | +0.223 | 0.19 |

  The QAP objective says how *confident* the matching is; the agreement says whether it is *right*. A high, stable QAP score sitting next to near-chance agreement is a specific and diagnosable failure, and without the second number it reads as "the method does not work here" — which is a different and much less useful conclusion. The relationship is strong and not a clean threshold: below 0.20 agreement the mean recovery is 0.14 with one cell at 0.68, and above it the mean is 0.81 with one cell at 0.04. It orders runs rather than classifying them.

  Nothing ships. This is a spike, the roadmap item stays open, and the honest limit is stated there: the case the direction exists for — an index holding vectors whose text is gone — cannot be constructed from a corpus that still has its text, so what has been shown is that the correspondence is recoverable without being given, not that it survives the setting a user would actually be in.
- A guide each for Qdrant, sqlite-vec and FAISS. The README's backend table promoted five stores and `docs/guides/` covered two.

  Each answers the question that store actually raises rather than repeating the same page three times. **Qdrant**: where your ids and text live, because a point carries a UUID and your document id is in the payload; that local mode locks its folder for the whole command, and what the engine does about that at the end of a job; and that it is the only one of the five that can rebuild its own search structure. **sqlite-vec**: that the vectors and the metadata are two tables and how rebasis finds the second one, and that the declared element type decides what is possible — `int8` is lossy and says so, `bit` cannot be read at all, because a `vec0` `bit[N]` is legal for any N so the blob's length does not determine the width. **FAISS**: the sidecar, what each of its two keys buys, that writing needs an `IndexIDMap2` and is refused at second zero without one, and that the reconstruction check catches the indexes that raise but not the ones that decode an approximation and return it.

  Every capability claim in the three was verified by opening a real store of that kind and reading what it declares, rather than from the source alone.
- The ArguAna figures in `docs/bridge-band.md` are corrected. ArguAna is evaluated with self-removal — a query is itself an argument that also appears in the corpus — and the evaluation harness omitted it. With the standard convention the harness matches Anserini's reproduction to within 0.003. Every conclusion in the document is unchanged; the break-even still predicts 29 of 29.
- The clustered-corpus and drift builders duplicated across four test files are now `make_corpus` and `make_drift` fixtures in the root conftest. The role of `tests/fixtures/` and `tests/helpers/` is filled by conftest in this suite: pytest resolves fixtures by name with no import path to get wrong, where an importable helper module collides once the whole suite is collected.
- The evidence set gains the three retrieval tasks [Maystre et al.](https://arxiv.org/abs/2510.13406) report their cross-model grid on — MMTEB's hard-negative HotpotQA and FEVER, and TREC-COVID — measured on the same ladder as everything else. They reproduce the paper's qualitative finding: a bridged query loses 62% of the status quo on HotpotQA and nothing at all on TREC-COVID. More usefully, they put a number on a limit `bridge-band.md` had only named. The gain/retention anti-correlation, −0.933 on the technical corpora, measures **−0.454** here — so the squeeze is a property of that corpus family more than it is a law, and the break-even does not model whatever separates them. The rule itself held 9 of 9 on thresholds it was never fitted to, which is the strongest evidence yet that it describes a real relationship rather than one corpus.
- The retrieval-quality harness is validated against published reproductions: it matches BEIR nDCG@10 for all-MiniLM-L6-v2 and bge-small-en-v1.5 on SciFact, FiQA and NFCorpus to three decimal places, and Anserini's own bge-base reproduction on CQADupStack-programmers to within 0.0004.
- The roadmap's unpaired-alignment item is rewritten, and the correction matters more than the reordering. **The limit it named was the wrong one.** It said — as `README.md` did — that an adapter cannot be fitted if the old model can no longer be run. `fit` never loads the old model: it reads the index's own vectors on one side and re-embeds the same documents with the candidate on the other, so the pairs come from the store. Losing the old model costs the *decision*, not the adapter — without it a real query log cannot be encoded the way the current system encodes it, so there is no `upgrade_gain` and the run is reported as provisional. The case no adapter survives is an index that kept vectors and discarded the text they came from, and that is the limit unpaired alignment would remove.

  **The first step is now [mini-vec2vec](https://arxiv.org/abs/2510.02348)** rather than [Wasserstein Procrustes](https://arxiv.org/abs/1805.11222), which stays on the list behind it. This is published evidence, not a measurement taken here, and the distinction is kept in the roadmap: [vec2vec](https://arxiv.org/abs/2505.12540) reports its optimal-transport baselines performing comparably to a naive baseline on same-backbone pairs and near random on cross-backbone ones, with its own appendix conceding the comparison favoured OT — but those baselines were Hungarian, EMD, Sinkhorn and Gromov-Wasserstein, and none of them was Grave et al.'s joint algorithm. The evidence is against the family rather than against that method. mini-vec2vec is the same orthogonal-solve-plus-assignment shape, with the assignment moved onto k-means centroids instead of individual points; its preprocessing is ADR 1 exactly, and it needs only `scipy` and `scikit-learn`, both already core dependencies. Whether it clears the limit on this project's ladders is unmeasured, and the roadmap says so.
- Three documents an evaluator looks for and could not find.

  **[Stability and support](https://batuhanzorbeyzengin.github.io/rebasis/stability/)** says what may change and what may not: what 0.x means here, which three things carry the compatibility promise (the Python API, the CLI's names and exit codes, and the stable error codes), the deprecation window that starts at 1.0, and — being specific, because this is the part that matters — a table of what is gated on every pull request, what runs nightly on the project's own host, and what is only claimed. Windows has never run. Nobody has pointed `migrate` at an index they could not rebuild.

  It also states the Python support policy rather than leaving it in a `pyproject.toml` comment: rebasis follows [SPEC 0](https://scientific-python.org/specs/spec-0000/), so a Python release is dropped three years after it appears. **3.12 reaches that mark in Q4 2026**, and the floor rises to 3.13 then rather than drifting. The dependency floors go further back than SPEC 0 requires, deliberately — each is the oldest version the suite actually passes on, and the `lowest direct dependencies` job is what keeps that true.

  **SUPPORT.md** says where to ask, what to attach, and what to expect: a first response usually within a week, no promised fix date, and no commercial support, no SLA and no LTS branch — stated because finding out later is worse.

  **MAINTAINERS.md** names the single maintainer and then spends the rest of the file on what follows from that. A bus factor of one is the largest risk in depending on this, and the two things that reduce it are Apache-2.0 and the fact that the reasoning is written down — eleven ADRs with the measurement behind each, so somebody picking it up does not have to re-derive the decisions. A governance document is explicitly not among them: a file describing how multiple maintainers decide things, written by the only maintainer, describes nothing.
- Three documents said things that had stopped being true.

  `README.md`'s decision-rule paragraph ended mid-sentence — "Above 1.0 bridging beats leaving things alone. It has" — and then jumped straight into the withdrawal of the count that sentence used to carry. The break is closed, and the headline "12 of those 62" now says what it is: how often the tool declined to recommend the adapter, which is not the same as how often it was right. The accuracy reading of that count is the identity that was withdrawn.

  `ROADMAP.md` said "Linux and macOS run today" under what 1.0 needs. Only Linux runs. The macOS leg was added and then removed over the `faiss-cpu`/`torch` OpenMP conflict, which is the wrong thing to have lost — the storage layer is where the platforms actually differ and where a bug costs data. The entry now says so, and names the narrower leg that would clear the conflict.

  `pyproject.toml` pointed at `tests/unit/test_coverage_floors.py` for the per-module coverage targets; that check is `tools/check_coverage_floors.py`, and it is a script rather than a test for a reason worth keeping written down. A second comment described an instruction-count benchmark as what the pull-request gate is built on, left behind when the CodSpeed dependency was removed.
- `README.md`'s quickstart shows how to **stop** a migration as well as how to undo one. It showed `rollback` and not `pause` / `resume`, which reads as though a running job can only be abandoned or finished.

  `docs/index.md`'s reading list gains the two measurement pages written since it was last updated — what a completed migration is worth, and what refitting during one costs.
- `ROADMAP.md`'s **Matryoshka shortcut** — "the right answer may be 'truncate and renormalise', with no adapter at all" — reads as a seventh `auto` candidate waiting to be written, and it is not one. `IdentityAdapter.apply` is `pad_or_truncate`, so it already truncates whenever the new model is wider than the old index, and **every** consumer of an adapter's output renormalises immediately afterwards: the held-out scoring path, the CSLS sample and the calibrator fit in `probe`, `Bridge.to_index_space` and the three serving paths that go through it, and `migrate`'s write-back. Truncate-and-renormalise is `identity` under another name — the same arithmetic in the same order, now pinned by a property test that asserts bit equality rather than closeness. The one place an adapter's output is read unnormalised is `ScaledAdapter.fit`, which uses it to fit the DSM diagonal and never serves or scores it; and even there a ranking could not move, because `top_k_search` is an inner product and a positive scalar on the query cannot reorder one.

  What is missing is the measurement, and three separate things kept it missing rather than one. `identity` is not in `CANDIDATE_METHODS`, so `auto` never fits it. `probe`'s "do nothing" baseline is computed only when the two dimensions agree, so on exactly the cross-dimensional rungs where truncation is the question there is no no-adapter number at all. And M0's `identity` figure of **0.2741** was measured on three model pairs that are all 384-to-384, where `pad_or_truncate` is a no-op — it measured no adaptation and never truncated anything. `tools/bridge_band.py` does not fill the gap either, and says so: its `naive_swap_padded` configuration zero-pads both spaces out to the wider one precisely because padding changes no inner product.

  `spikes/truncate_candidate.py` is the harness that can take it: `auto`'s own held-out comparison with `identity` named in the candidate list, over that harness's corpora and ladders, reporting every candidate's score, each rung's truncation ratio, and what `select_best`'s parameter-count tie-break would do with a candidate that has no parameters. Nothing in the shipped code changed and the roadmap entry stays open until it has run.
- `ROADMAP.md`'s **chained adapters** entry — "v1 → v2 → v3 without a full refit at each step, error accumulation across a chain has not been measured" — is measured and closed. Nothing ships.

  Over 204 cells (seventeen corpora, six spans of a four-model ladder, both directions, two adapter families), a chain costs a median 1% of the direct fit's retention at two links and **9% at three**, losing in 17 of 17 cells on the document side. A direct fit is never unavailable: the old vectors are in the index and the new ones come from one embedding pass over a sample, so a chain only buys back a pass `rebasis fit` runs anyway. [The numbers](https://batuhanzorbeyzengin.github.io/rebasis/chained-adapters/).

  **One span came out ahead, and chasing it was worth the run.** potion → MiniLM → bge-small scored +0.0114 over its direct fit, and a chain beating the thing it approximates wants an explanation before it gets a recommendation. `procrustes_centered` subtracts a mean before it rotates, so a two-link chain performs *two* centrings — a strictly richer function than the single centred rotation a direct fit produces. Re-running the whole grid with plain `procrustes`, where a chain of orthogonal matrices is one rotation exactly like the direct fit, the chain wins **0 of 34** at two links and the same span is −0.0119. The gain was the centring, not the chain.

  That run also serves as a check on the harness: under plain `procrustes` the query and document costs come out identical to four decimals in every cell, which is what `A(q)·A(d) = q·d` requires of an orthogonal map.
- `SECURITY.md` says what the open advisories in the dependency tree are, and why neither is reachable through rebasis.

  Five, in two packages, and none of them has a fixed version upstream — so neither is closed by an upgrade, and a reviewer who finds them needs the assessment rather than a promise to bump something.

  **chromadb carries four**, all of them properties of the Chroma *server*: pre-authentication code injection through its collections endpoint, an authenticated variant of the same, missing authorisation validation across tenants, and an RBAC provider that never checks which tenant a permission applies to. rebasis opens `chromadb.PersistentClient(path=...)` and nothing else — there is no `HttpClient` in the backend and the Chroma URI carries no host — so it cannot reach a Chroma server at all. That is a statement about rebasis and not about your deployment: if you run a Chroma server, those advisories apply to it.

  **diskcache carries one**, unsafe pickle deserialization, and arrives through `llama-cpp-python` — an optional extra deliberately outside `rebasis[all]` because it compiles from source. The attack needs write access to the cache directory, which is already local compromise.

  Both are re-checked weekly by the `Audit` workflow, over the tree `uv.lock` actually resolves with every extra installed.
- `SECURITY.md` states where rebasis sits with respect to the regulations a compliance reviewer will ask about, because the answers are short and working them out otherwise costs somebody a day.

  It falls outside the **EU Cyber Resilience Act**: the European Commission's own guidance says the CRA does not apply to developers contributing source code to free and open-source software not under their responsibility, and that providing FOSS its maintainers do not monetise is not a commercial activity. The lighter "open-source software steward" regime under Article 24 applies to a legal person — a foundation or a company — not to an individual. rebasis is neither a manufacturer nor a steward as things stand, and the note says what would change that.

  **SOC 2, ISO/IEC 27001, ISO/IEC 42001 and the EU AI Act certify organisations and deployed systems, not libraries.** Claiming any of them for a tool a user runs on their own machine against their own index would be a category error, and the note says so plainly rather than leaving the absence to be read as an oversight. What rebasis offers a compliance programme instead is evidence: a hash-chained, replayable audit trail that records who ran what with which parameters and carries no document content or vectors by construction. The certification belongs to the organisation deploying it.
- `docs/bridge-band.md` and ADR 10 record why adapter retention is not an engineering target: it is bounded by how much structure the old embedding space holds. Six times the fit data buys one to two points, and the most constrained candidate (`procrustes_centered`) wins 15 times out of 15 against the residual MLP. Retention correlates with the old model's own quality at +0.901, which is why gain and retention pull against each other at −0.958.
- `docs/bridge-band.md` gains a section testing its own counts, and `tools/band_stats.py` is what computes them. The headline "the break-even predicted the outcome" turns out to be an identity wherever both sides are read off one run's own scores: `ARR x upgrade_gain` is `(bridged / reindex) x (reindex / status quo)`, which is `bridged / status quo` — the same inequality as the outcome, so it agrees 57 times out of 57 because it cannot do otherwise, and `tools/bridge_band_report.py --view summary` reports a perfect score on every file for the same reason. Against the null that matters — bridging beat doing nothing in 3 of 57 runs, so a rule that always answered "do not bridge" scores 54 of 57 — even that count clears the baseline at only **p = 0.046**, against p ≈ 7e-18 for the coin flip the number invites. The break-even as `probe` actually reports it agrees in **37 of 57** (95% Clopper-Pearson 0.51–0.77), which is *below* the baseline. What does survive is the quantity rather than the threshold: the estimate ranks the runs by the margin they returned at Spearman **+0.60**, p ≈ 1e-6. Using the per-query sidecars the harness now writes, a paired Fisher randomisation test of `bridged` against `status quo` over 48 runs finds 40 differences at raw p < 0.05 and **35 surviving Holm** — every one of them a loss, with no run anywhere in this evidence showing bridging ahead by an amount distinguishable from zero. The direction of every finding in the document is unchanged and the negative ones are strengthened; what is withdrawn is the counting as evidence for the rule. The randomisation test is implemented against a seeded generator rather than called from `ranx`: `ranx.compare` needs `Qrels` and `Run` objects that a per-query sidecar cannot rebuild, and `ranx.statistical_tests.fisher_randomization_test` is `@njit(parallel=True)` over one global seed, returning 0.948/0.945/0.948/0.942 for the same call and seed. The two agree to Monte Carlo error.
- `docs/cascade-band.md`: the bridge used as a recall stage rather than as the final ranking. Measured over **48 runs on sixteen corpora** — single-stage bridging beat keeping the current model in **1**, a two-stage arrangement in **36**, and of the 37 runs where a full reindex actually beat doing nothing it delivered that upgrade in **36**. It does not contradict ADR 10: the same harness reproduces every figure in `bridge-band.md` — retention 0.717, the gain/retention anti-correlation −0.933, the naive swap at 0.151, the break-even right 48 times out of 48. What changes is which quantity bounds the arrangement: recall@200 retention is 0.893 for the same adapters, because reaching the top 200 is a weaker requirement than ranking in the top 10. The cost is real and written down: N documents embedded per query. `rebasis.serve.Cascade` serves the arrangement with the cache that makes it affordable, and `Cascade.stats` is the instrument for the one measurement a corpus cannot supply — what it costs under a real query distribution.
- `docs/mixed-space-fusion.md`: which of `calibrated_merge`'s two merges is right on a half-migrated index, measured against human judgements at seven points along a real `migrate` job. **They are not interchangeable, and they fail in opposite directions.** Reciprocal rank fusion returns exactly half its results from each embedding space at every stage of a migration — during one no document is in both result sets, so the fusion degenerates into a strict interleave and at 10% migrated it gives half the top 10 to a tenth of the corpus. Over 20 mid-migration cells on four corpora, with migrated records holding the new model's own vectors, the calibrated merge beat RRF in 17, and RRF came in *below* bridging and ignoring the mixture in 4 — all at 10% or 25% migrated. But under the migration `rebasis migrate` actually performs, where a migrated record holds an adapter's image of its old vector rather than a new-model vector, the result reverses: RRF wins 6 of 10, because the calibrator was fitted against the new model's score distribution and starves a migrated half that is not in it, down to 0.3% of the result. `calibrated_merge` branches on whether a calibrator exists, and that says nothing about which of the two cases it is in. Three smaller findings: the calibrated merge does not reproduce the single-space ranking at 0% migrated (the isotonic calibrator collapses ten scores onto five to seven levels and the id tie-break reorders them, changing the top 10 on 84–96% of queries) where RRF is exact; `MAX_OVER_FETCH` binds at both extremes, costing 9.2x retrieval at 10% and 90% migrated, but never returned a short result and — re-run at a ceiling of 32 — cost no measurable quality either; and a *completed* adapter migration reached 84% and 88% of a real reindex on the two corpora it was measured on, which on NFCorpus is below the status quo it started from. Nothing in the shipped code changed. This is the measurement that was missing, and one model pair on four corpora is where it stops.
- `docs/related-work.md`: where rebasis sits in the literature, and — the part
  that earns the page — which neighbouring approaches its users structurally
  cannot take. A decade of backward- and forward-compatible representation
  learning (BCT [arXiv:2003.11942](https://arxiv.org/abs/2003.11942), FCT
  [arXiv:2112.02805](https://arxiv.org/abs/2112.02805), BiCT
  [arXiv:2204.13919](https://arxiv.org/abs/2204.13919), MixBCT
  [arXiv:2308.06948](https://arxiv.org/abs/2308.06948)) solves the same
  problem — a better model, an index you cannot afford to rebuild — more
  completely than any adapter can, and every method in it needs something a user
  calling a vendor's API does not have: the new model's training run, the old
  model's classifier, labelled data, or a per-item feature that had to be stored
  before the old index existed. That is a hard boundary, not a quality comparison,
  and the Embedding-Converter paper draws it independently in its own related work.

  Three findings came out of reading the papers rather than summarising them.
  **Seo et al.** ([arXiv:2301.03767](https://arxiv.org/abs/2301.03767), WACV 2025)
  is the closest analogue to `rebasis.serve.MixedSpaceSearch`, and the split is
  sharper than expected: their untrained distance rank merge beats BCT and BiCT on
  all four datasets while needing nothing, their reverse query transform *is*
  rebasis' adapter, and only their metric-compatible contrastive loss is out of
  reach — because it needs class labels over the corpus, not because it retrains a
  model. rebasis approximates that calibration post-hoc with an isotonic map and
  falls back to RRF; how much of the difference that recovers is unmeasured, and
  is the page's one open question. **Google's Embedding-Converter** (ACL 2025)
  frames its transform as a way to evaluate a candidate model cheaply — it
  predicts which of two models is better on 11 of 13 datasets in domain and 12 of
  12 out of it, which is `probe`'s claim reached independently — and it runs in
  the *forward* direction the roadmap has not measured. And **mini-vec2vec**
  ([arXiv:2510.02348](https://arxiv.org/abs/2510.02348)) reports, confirmed
  against vec2vec's own text, that optimal-transport alignment failed to beat a
  naive baseline on sentence embeddings even in an oracle setup — which bears on
  the roadmap's plan to try Wasserstein Procrustes first, and suggests
  mini-vec2vec's centroid-level assignment as the cheaper step instead.
- `docs/vs-drift-adapter.md`: rebasis' harness running the protocol of [Drift-Adapter](https://aclanthology.org/2025.emnlp-main.805.pdf) (EMNLP 2025), which evaluates the same three adapters on the same problem and reports 95–99% recall recovery against this project's measured 0.714–0.722. The document was commissioned on the premise that the published protocol is the more generous one and both numbers are right about different questions; **the measurement inverts that premise and the inversion is the finding.** On the paper's own corpora, model pair and fit budget the reproduction measures **0.24–0.50**, and across rebasis' own 48 runs a mean of 0.408 — with the same harness reproducing every published rebasis figure in the same pass (retention 0.717, anti-correlation −0.933, naive swap 0.151, break-even 48/48). Three readings of the protocol were run and all three eliminated: the strict top-10 kNN, the nearest-neighbour-only ground truth rebasis itself uses (`SPARSE_RELEVANT`), and a duplicated corpus at the factor each dataset's training split implies — the last confirmed in its mechanism, since duplication lifts the *ceiling* to 1.000 on Emotion, and refuted in its consequence, since the adapter does not follow. `bridge_band.py` grew `--protocol`, a `ceiling_old_space` configuration that bounds every row above it with a query built from the answer, and per-query score sidecars; the ceiling is where the sharpest claim comes from, because **zero of 48 ceilings reach 0.95** and the published band therefore sits above what an oracle achieves. Three protocols over 48 runs also itemise what separates the tiers: the query-proxy assumption is worth **+0.009** and the ground-truth definition **+0.337**, independently reproducing M0's conclusion at scale — and the two protocols disagree about what to do in **37 of 48 runs**, because the paper has no status-quo configuration and so no `bridge_advantage` can be computed from its numbers at all.
- `migrate` is marked experimental in `--help`, in the pre-flight preview and in the docs, with a per-backend table of what is actually tested. It works and every guarantee is covered by a test against a real store, but nobody has yet run it against an index they could not rebuild.
- `rollback` no longer claims to restore the index "bit for bit". The shadow copy is bit-identical — that part was always true and is rebasis' to guarantee — but writing it back goes through the store's own upsert, and a store that normalises on write returns the vectors one float32 ulp away. Measured with rebasis nowhere in it: Chroma at `hnsw:space=cosine` shifts a plain write-and-read by ~3e-08, while the same collection at `l2` round-trips exactly. Cosine similarity against the original is 1.000000; over three migrate-and-rollback rounds on 5,183 documents, one query in 300 saw a result on the k boundary change sides. Anyone who checked the old claim on a cosine collection would have found it false.


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
