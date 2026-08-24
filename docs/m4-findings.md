# M4 measurements — memory, budgets and the layer contract

What M4 measured on the project's own host, and what those measurements
changed. Companion to [`m0-findings.md`](m0-findings.md), which covers the
pre-implementation spike; this one covers the polish milestone, where the
questions were about the shape of the running system rather than about whether
the idea works.

Every number here was produced on **AWS `g5.xlarge`** (4 vCPU, NVIDIA A10G,
Ubuntu 24.04, Python 3.12.3, numpy 2.x) by tests in `tests/performance/`. Each
one is reproducible: the test that produced it is named beside it.

---

## 1. The `O(batch × d)` invariant, measured

Peak memory is a function of the batch, not of the corpus. That is the single
most consequential architectural claim in the project — it is what makes the
same tool usable on a 50,000-chunk vault and a 5,000,000-chunk one — and until
M4 it was asserted rather than measured.

`draw_corpus_sample`, sample size held at 1,000, d=32:

| Corpus N | Peak traced allocation |
|---|---|
| 20,000 | 7.8 MB |
| 60,000 | **19.5 MB** |
| 150,000 | **19.5 MB** |
| 400,000 | **19.5 MB** |

The plateau is exact. It appears at `CLUSTER_POOL_MAX = 50_000`, which is the
sampling rule: stratification clusters `min(50k, N)` vectors, so above 50,000
the pool stops growing and so does everything downstream of it.

### 1.1 The first version of this test was wrong

The test originally measured N ∈ {2k, 10k, 40k} and failed: 2.1 MB → 30.3 MB.
That looked like a violated invariant and was not one. Every one of those sizes
sits **below** the 50,000 cap, where the pool legitimately is the corpus. The
test was measuring the ramp and calling it the plateau.

This is worth recording because the corrected test is a better test for a reason
that generalises: **a scaling test has to straddle the point where the scaling
is supposed to stop.** Measuring only inside the growth region proves nothing
either way.

*Test:* `tests/performance/test_memory_ceiling.py::TestScalingInvariant`

---

## 2. Chunked top-k is linear, and the obvious test for that is wrong too

The other half of that invariant: the full score matrix is never materialised.
At 10,000 × 10,000 it would be 400 MB; 1,024-row chunks cost 40 MB.

`top_k_search`, d=128:

| n | Peak allocation | Full matrix would be | Peak ÷ n |
|---|---|---|---|
| 2,000 | 45.7 MB | 15.3 MB | 23.4 KB |
| 4,000 | 94.2 MB | 61.0 MB | 24.1 KB |
| 6,000 | 141.2 MB | 137.3 MB | 24.1 KB |
| 12,000 | 282.3 MB | 549.3 MB | 24.1 KB |

Peak per row is constant to three significant figures across a sixfold range.
That is the invariant: **linear in n**, where a materialised matrix would be
quadratic.

The natural-looking assertion — "peak must be a small fraction of n²" — is not a
test of the algorithm. At n=2,000 the chunked path uses *three times* the full
matrix size, because a 1,024-row chunk of a 2,000-row problem is half the
problem, and the constant factor (roughly five live `chunk × n` float32 buffers:
the matmul output, its negation, and the `argpartition` intermediates) dominates.
The same code at n=12,000 uses half the full matrix. A fraction-of-n² gate would
therefore pass or fail on the size chosen rather than on the code, which is the
definition of a test that does not test what it claims to.

*Test:* `tests/performance/test_memory_ceiling.py::TestCeilings::test_the_ground_truth_knn_never_materialises_the_score_matrix`

---

## 3. The macro budgets, at the sizes they name

M0 measured these at smaller sizes and found them 30–600× too loose. M4 measured
them at exactly the inputs the table specifies (20,000 pairs, d=768):

| Operation | Budget | Measured | Headroom |
|---|---|---|---|
| Adapter fit — OP | 20 s / 500 MB | **0.32 s / 61 MB** | 62× / 8× |
| Adapter fit — LA | 90 s / 600 MB | **0.45 s / 297 MB** | 200× / 2× |
| Adapter fit — MLP | 180 s / 800 MB | **6.56 s / 298 MB** | 27× / 2.7× |
| `auto` (all + evaluation) | 360 s / 800 MB | **8.54 s / 308 MB** | 42× / 2.6× |
| Ground truth kNN 10k×10k | 30 s / 300 MB | **1.07 s / 235 MB** | 28× / **1.3×** |
| `.rbs` load (MLP + DSM) | 50 ms / 20 MB | **<10 ms / 8 MB** | >5× / 2.5× |

Two things follow.

**The time budgets are not gates.** A target met by 27–200× cannot detect a
regression; a change would have to make the code two orders of magnitude slower
before any of these fired. They are useful as documentation of intent and as a
crash barrier, and the actual PR gate is the instruction-count benchmark. This
confirms M0's proposal rather than adding to it.

**The memory budgets are real, and one is nearly binding.** The kNN row is at
78% of its budget. That is the only figure in the table doing work: a change
that raised the chunk size, or added one more live copy inside the loop, would
cross it. The others have between 2× and 8× headroom — tight enough to notice a
doubling, loose enough not to fire on noise.

*Test:* `tests/performance/test_macro_budgets.py`, gated at 120% of budget.

---

## 4. Two layer-contract violations the partial contract could not see

The import-linter contract carried a note from M1: *"Modules not yet written are
added as M1 progresses."* By M4 every module existed but the contract still
listed five of them. Extending it to the full layer stack immediately found two
real inversions:

**`compute.numpy_backend` → `probe.metrics`.** `NumpyBackend.matmul_topk`
delegated to `top_k_search`, which lived in `probe/metrics.py` — so the bottom
of the stack imported from near the top. The delegation itself was right (two
copies of that loop would be two places for the memory invariant to break
independently); the *location* was wrong. `top_k_search` moved to
`compute/search.py`, where the invariant it embodies belongs, and
`probe.metrics` re-exports it.

**`storage.gc` → `manifest.paths`.** The garbage collector needed the state
directory's layout constants, which lived in `manifest`, and the layer contract
puts `storage` below `manifest`. The constants moved down to `storage/layout.py` and
`manifest.paths` re-exports them, so the state directory still reads as one idea
from above.

Neither was a bug in the sense of producing a wrong answer. Both were the kind
of drift that a contract exists to prevent, and both had been sitting there for
two milestones because the contract was scoped to the modules that existed when
it was written.

**Generalisation:** a partial contract is not a weak contract, it is an absent
one for everything it omits. The note that says "more will be added later" is
where the enforcement stops.

---

## 5. `probe`, `fit`, `eval` and `migrate` were not connected to anything

The measurement pipeline, the adapter mathematics, the `.rbs` format, the
migration engine with its shadow copies and checkpointing — all complete and
tested since M2/M3. All four CLI commands printed a message saying so and exited
1.

What was missing was one layer: sample a live store, read its vectors, re-embed
its text. That is `probe/session.py`, and writing it surfaced four things the
functional tests had no way to reach.

### 5.1 Text must be matched by id, never by position

`iter_records(ids=[...])` returns records in whatever order the backend chooses.
Chroma pages them; LanceDB uses `IN (...)`; Qdrant returns them by point id. A
positional `zip` of requested ids against returned records pairs every document
with someone else's text — and nothing raises. The pipeline runs, the adapter
fits, and the ARR is meaningless.

The test for this uses a deliberately hostile store that returns ids in
**reverse** order, because a backend that happens to preserve request order
would let the bug through.

*Test:* `tests/integration/test_probe_session.py::test_text_is_matched_by_id_not_by_position`

### 5.2 Reservoir sampling, not "the first 50,000"

The clustering pool has to come from somewhere. Reading the first 50,000 records
is one line shorter and wrong: insertion order is rarely random, and on a vault
that grew by topic the first 50,000 chunks are a biased slice. Algorithm R costs
one `rng.integers` per record past the cap and needs no reliable `count()` —
which the bridge backends cannot always give.

### 5.3 `fit` should not need `--dim` for an unregistered old model

`fit` resolved the old model's encoding profile through the profile table, which
raises `RB-E2003` for a model it does not know. But the *index* knows its own
dimension, and for an adapter fitted against vectors that already exist, the
dimension is all that is needed — prefixes affect encoding, and nothing is being
encoded with the old model here. `fit` now reads `store.dimension()`.

Found by the end-to-end test, which uses two model ids that are deliberately not
in the profile table.

### 5.4 A store that accepts a write and forgets it

`memory://` gained a file-backed form (`memory:///corpus.npz`) so the end-to-end
tests could exercise the real CLI path. The first version loaded from the file
and upserted in memory — so `migrate` reported success, and the file on disk was
unchanged. Read-back verification passed, because it read back from the same
in-memory object.

That is exactly the silent-data-loss shape read-back verification exists to
catch, arriving through the one door it does not cover: verification that reads
from the same place the write went. The file-backed store now writes through, atomically.

*Test:* `tests/e2e/test_cli_flow.py::test_migrate_then_rollback_restores_the_index`

---

## 6. sqlite-vec and Qdrant

Both turned out to have one structural quirk each that the `VectorStore`
protocol had to absorb.

**sqlite-vec splits identity from vectors.** A `vec0` virtual table holds a
`rowid` and an embedding; the user's own id and text live in an ordinary table
beside it, joined on `rowid`. The backend therefore has to find that table —
by the `vec_`/`_vec` naming convention its own examples produce, then by looking
for a column that resembles an id — and every read is a join.

It also cannot `UPDATE` the embedding column in every released version. Delete
plus insert on the same `rowid`, inside one transaction, is the portable form.

**Qdrant hides the user's id in the payload.** Points carry an integer or UUID
id; the document id is conventionally a payload field. Reporting the point id
would make every id rebasis prints, records in the audit trail and writes back a
*different* id from the user's own — so the backend resolves the payload key
from a sample point and maps back when writing.

Its write path uses `update_vectors`, not `upsert`. `upsert` replaces the whole
point, so anything not resent is dropped — which would make rebasis take
ownership of the payload, which is exactly what rebasis must not do: the user's
data is theirs.

Both run against real databases in CI: sqlite-vec through its extension, Qdrant
in local mode (`QdrantClient(path=...)`), neither needing a server.

---

## 7. The nightly GPU workflow was selecting nothing

The nightly GPU workflow — which drives the project's own host and is not part
of this repository — runs `pytest -m "gpu or slow"`. No test carried either
marker, so the job had been passing by doing nothing.

The device-parity suite parametrises over `available_devices()`, so on
the server it *did* exercise CUDA — but as part of the `contract` layer, in the
default run. The fix marks the accelerator parametrisations `gpu` via
`pytest.param(..., marks=...)`, so they stay in the default suite on CPU and
join the nightly on an accelerator. The macro benchmarks above carry `slow`.

Verified on the host: `pytest -m gpu` now collects and passes 4 tests on the
A10G, where it previously collected 0.

**Generalisation:** a marker-selected CI job that reports success is
indistinguishable from one that ran nothing. Worth asserting the count.

---

## 8. What these measurements do not establish

- **d=32 and d=128, not d=768, for the scaling curves.** The shape of the curve
  is dimension-independent; the constants are not. The plateau at 19.5 MB scales
  with d.
- **One machine.** Every number is from one `g5.xlarge`. The headroom figures in
  section 3 would look different on a laptop, and the budgets were written for a
  laptop-class machine.
- **Traced allocation, not RSS.** `tracemalloc` measures Python-level
  allocations and misses the allocator's high-water mark and anything BLAS
  allocates outside Python. It is the more *stable* measurement, which is why
  the scaling tolerance could stay at the stated 20% instead of being widened to
  absorb noise — but it is not RSS, and the budgets are RSS.
- **Synthetic corpora.** The memory curves used random vectors. Memory does not
  care, but the timing curves would look different on real text where the
  embedding step dominates.

---

## 9. Re-running these

All three run on the host, against the working tree synced there:
`tests/performance/test_memory_ceiling.py` for the memory ceilings and the
scaling invariant; `tests/performance/test_macro_budgets.py -m slow -s` for the
macro budgets with the measurements printed; `-m gpu` for the device-parity
suite.
