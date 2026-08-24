# 6. There is no CPU/GPU crossover for kNN

**Status:** Accepted · **Date:** 2026-08 · **Evidence:** `docs/m0-findings.md`, section 8

## Decision

`compute/thresholds.py` holds no crossover size. It records measured speedups
per operation, and the answer for kNN is "use the accelerator when there is
one".

## Context

kNN was called "borderline", pending a measured threshold. The reasoning
came from FAISS's own guidance: an index of a few thousand vectors fits in CPU
cache and beats a GPU once transfer is counted, while hundreds of thousands do
not. rebasis' default sample is 10,000 — apparently right in the ambiguous zone.

## Evidence

Chunked top-k, A10G against 4 vCPU, transfer included:

| Documents | CPU | CUDA | Speedup |
|---|---|---|---|
| 10,000 × 10,000 | 0.82 s | 0.05 s | 16× |
| 10,000 × 50,000 | 3.55 s | 0.04 s | 89× |
| 10,000 × 200,000 | 13.41 s | 0.17 s | 79× |

There is no crossover in the range that matters. Even at 2,000 documents the
accelerator won by 22×.

The published guidance is not wrong; it describes a different workload. FAISS's
comparison is of *index* structures answering one query at a time. rebasis does a
batched matmul of 10,000 queries against 10,000 documents — a shape that suits a
GPU regardless of how small "small" is here.

## Consequences

- No threshold constant to tune, drift, or get wrong on new hardware.
- The question is settled by measurement rather than left open.
- The module keeps `worth_accelerating()` and per-operation ratios, because the
  ratios do differ — embedding is 25–40×, adapter fitting 3–6×, and a linear fit
  gains nothing.
- Those ratios are between *this* GPU and *that* CPU; a faster host narrows all
  of them. Measuring them on the local machine is a planned affordance, not one
  that exists.

## What generalises

A threshold quoted from a reference implementation describes that
implementation's access pattern. Ours is batched where theirs is not, and the
guidance did not survive the difference.
