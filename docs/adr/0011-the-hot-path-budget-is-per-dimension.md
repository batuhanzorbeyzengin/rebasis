# 11. The hot-path budget is per dimension, not a single number

**Status:** Accepted · **Date:** 2026-08 · **Evidence:** `docs/development/performance.md`

## Decision

The "one query in under 15 µs" budget holds at d=256 and is replaced above it by a
measured figure per dimension. The budget that is enforced is no longer a
constant; what is enforced instead is that the path stays close to the cost of
the matrix multiply it cannot avoid.

| d | budget | measured |
|---|---|---|
| 256 | 15 µs | 11.9 µs |
| 384 | 20 µs | 17.9 µs |
| 768 | 30 µs | 24.5 µs |
| 1024 | 40 µs | 35.6 µs |

## Context

The budget was set at 15 µs for a single query at d=768, from a reference
measurement of "MLP apply ~8 µs". That figure was never checked against the arithmetic the
budget has to pay for.

A d=768 adapter multiplies a query by a 768×768 float32 matrix. That matrix is
2.36 MB — larger than L2 on the hosts this runs on — so the operation is bound
by how fast the weights can be read, not by floating-point throughput. Doubling
the dimension quadruples the bytes.

## Evidence

Measured on the reference host (AMD EPYC 7R32, 8 vCPU, scipy-openblas), before
and after interleaved within a single process so both versions see the same
machine. `before` is the code as it stood; `after` is what ships.

| d | before | after | change | the matvec alone |
|---|---|---|---|---|
| 256 | 17.6 µs | **11.9 µs** | −32% | 5.0 µs |
| 384 | 24.2 µs | **17.9 µs** | −26% | 10.6 µs |
| 768 | 32.7 µs | **24.5 µs** | −25% | 15.8 µs |
| 1024 | 43.2 µs | **35.6 µs** | −18% | 27.4 µs |

**At d=768 the matrix multiply alone costs the entire 15 µs budget**, before
anything is normalised or added. At d=1024 it costs nearly twice the budget. No
arrangement of the surrounding code reaches the old figure, because the old
figure is below the floor.

What was left to win was the code around the multiply, and that was worth taking:

- `l2_normalize` on a single vector, 8.5 µs → 3.8 µs. `np.linalg.norm` is a
  Python function that re-derives its axis and dtype handling on every call; at
  d=768 the dot product it ends up doing costs 1.4 µs of the 4.7 µs it charges.
- The centred Procrustes adapter folds `μ_dst − μ_src·R` into a bias at
  construction. `(x − μ_src)R + μ_dst` and `xR + b` are the same map; the second
  is one fewer full-length array operation per query. Measured equal to 7e-08
  across a thousand queries, leaving the ranking identical.

## What was rejected

Storing the weight matrix column-major. `x @ W` with one row of `x` sums down
the columns of `W`, and column-major makes each of those contiguous: at d=768
that is 18.5 µs against 11.2 µs, and at d=1536 85.9 against 43.5 — enough to
bring d=768 close to the old budget.

It was rejected because the sign of the effect is a property of one BLAS build
rather than of the arithmetic. Below d=768 the same change *costs* 10–20% —
9.7 µs against 11.7 at d=384 — and the crossover sits exactly where OpenBLAS
starts threading its GEMM. Shipping it would mean tuning the library to one
host's BLAS and slowing down the most common small-model dimension to do it. It
is recorded here so anyone serving at d≥768 who measures it on their own
hardware can make that trade deliberately.

## Consequences

- The budget stops claiming a precision it never had, and stops being met by
  ignoring it.
- The performance tests continue to assert **relative** cost — a single vector
  against the batch route, centred against plain Procrustes — because those hold
  on hardware the budget was never calibrated for. Absolute numbers belong on a
  fixed runner.
- Batching remains the answer to a latency problem at high dimension: the
  overhead is per call, not per vector.
