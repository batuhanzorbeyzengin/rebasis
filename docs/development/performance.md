# Performance

## The invariant

**Peak memory is `O(batch × d)`, never `O(N × d)`.**

The same tool has to behave the same on a 50,000-chunk vault and a
5,000,000-chunk one. One `list(iter_records())` breaks that, and breaks it only
on corpora large enough that nobody notices in development.

Two consequences run through the code:

**Reads stream.** `iter_records` returns an iterator and no caller wraps it in
`list()`. The contract suite asserts it for every backend.

**The score matrix is never materialised.** At 10,000 × 10,000 it is 400 MB in
float32; 1,024-row chunks cost 40 MB. Top-k is a chunked matmul plus
`argpartition` — `O(n)` selection rather than `O(n log n)` sorting.

Both are measured, not asserted. See [the M4 findings](../m4-findings.md).

## float32 is a contract

`FloatArray` is the single alias. Input that arrives as float64 is converted
once, at the boundary. Promoting to float64 doubles memory and has no measurable
effect on ARR.

## The hot path

`Bridge.to_index_space()` runs on every query of a RAG request.

Inside it: **no logging, no dictionary copies, no validation.** Validation
happens once, in `Bridge.load()`. A regression test counts the objects allocated
per call, because the way this budget gets broken is somebody adding a debug log
line.

### The budget depends on the dimension

The budget was 15 µs for a single query at d=768. That figure is below the floor:
a 768×768 float32 matrix is 2.36 MB, larger than L2, so the multiply is bound by
how fast the weights can be read and costs 15.8 µs on the reference host by
itself — the whole budget, before anything else happens. [ADR 11](../adr/0011-the-hot-path-budget-is-per-dimension.md)
replaces the constant with a measured figure per dimension.

Measured on the reference host, before and after interleaved in one process:

| d | budget | measured | the matvec alone |
|---|---|---|---|
| 256 | 15 µs | 11.9 µs | 5.0 µs |
| 384 | 20 µs | 17.9 µs | 10.6 µs |
| 768 | 30 µs | 24.5 µs | 15.8 µs |
| 1024 | 40 µs | 35.6 µs | 27.4 µs |

Two things were worth optimising and both shipped: `l2_normalize` takes a scalar
route for a single vector (8.5 → 3.8 µs at d=768, since `np.linalg.norm`
re-derives its axis and dtype handling on every call), and the centred
Procrustes adapter folds its two mean vectors into one bias at construction.
Together, −25% to −32% depending on the dimension.

One thing was measured and **not** shipped: storing the weight matrix
column-major, which is worth −40% at d=768 and +20% *against* you at d=384. The
crossover is where OpenBLAS starts threading its GEMM, so the sign of the effect
belongs to one BLAS build rather than to the arithmetic. If you serve at d≥768,
measure it on your own hardware and decide deliberately.

At high dimension the answer to a latency problem is batching: the overhead is
per call, not per vector.

## Thread oversubscription

The most commonly missed trap in scientific Python. numpy's BLAS spawns threads;
so does torch; so does the embedding backend. Left alone they multiply, and
performance collapses.

rebasis has no parallel regions of its own — no worker pool, no `n_jobs` — so
there is no multiplication to prevent yet. What it does is **report**: `rebasis
doctor` names the BLAS in use and its thread count, and says whether you set
that yourself.

The rule — pin BLAS to 1 inside our own parallel regions — is therefore vacuous
today, and deliberately so: `auto` fits its candidates in sequence, and the one
measurement that would argue for parallelising them has not been taken. When a
parallel region does appear, the limit goes in with it, not before.

## GPU

Optional, and it pays in exactly one place: embedding generation, measured at
25–40× on an A10G against a 4-vCPU host. That is 80–90% of a `probe` run's wall
clock, so it is the right 40×.

Everything else is closer than the intuition suggests: adapter fitting is 3–6×,
and a linear fit gains nothing. kNN is the exception in the other direction. It
was expected to be borderline at the default 10,000-document sample; measured, it
is 22–58× on the accelerator at every size tested, transfer included
([ADR 6](../adr/0006-no-gpu-threshold-for-knn.md)).

The core path always works on CPU. There is no GPU requirement.

## What we deliberately do not do

- **Parallelise early.** Hit the budgets single-threaded first. A multi-process
  architecture makes debugging and memory accounting much harder.
- **Write a Cython or Rust extension.** The adapter mathematics is already in
  BLAS. A compiled extension imports a whole installation problem — wheels,
  platforms, a higher contribution barrier — before Python overhead is even a
  measured bottleneck.
- **Add a cache.** Caching before measuring is the fastest way to produce a
  consistency bug. Measure first.

## The numbers

- [M0 — the spike](../m0-findings.md): 84 configurations across 4 corpora and
  3 model pairs, GPU/CPU breakdown, the sample-size curve.
- [M4 — memory and budgets](../m4-findings.md): the scaling plateau, the top-k
  linearity curve, and the macro budgets measured at the sizes they name.
