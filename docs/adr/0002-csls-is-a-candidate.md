# 2. CSLS is a candidate, not a correction

**Status:** Accepted · **Date:** 2026-08 · **Evidence:** `docs/m0-findings.md`, section 5

## Decision

CSLS is evaluated as a **variant of each adapter** and kept only when it scores
better on the held-out set. It is never applied unconditionally.

## Context

CSLS was taken to be something that "could raise ARR for free". The reasoning
is sound: high-dimensional spaces have hubs — vectors that are near-neighbours of
far too many queries — and CSLS penalises them by subtracting half each
document's mean similarity to a reference sample. It is standard in the
cross-lingual alignment literature it comes from.

"For free" is the part that did not survive measurement.

## Evidence

| Adapter quality | Mean ARR change from CSLS |
|---|---|
| Weak (ARR < 0.5) | **+0.103** |
| Strong (ARR ≥ 0.8) | **−0.045** |

Spearman correlation between CSLS gain and adapter quality: **−0.704**.

The relationship is not noise. CSLS helps precisely when the mapping is poor —
when many queries land in the wrong region and hubness dominates the errors.
When the mapping is good, the hubness penalty removes signal instead of noise.

## Consequences

- Applying CSLS by default would have degraded exactly the adapters that work.
- `auto` measures both variants of each candidate, so the cost is one extra
  evaluation per candidate rather than one extra fit.
- The winner's name carries the suffix (`procrustes_centered+csls`), because a
  report that hid which variant won would hide the finding.

## Alternatives

**Always on, as the design proposed.** Rejected: −0.045 on the adapters a user
would actually ship.

**Always off.** Rejected: +0.103 on weak adapters is worth having, and "weak" is
common with genuinely distant model pairs.

**A quality threshold — on below 0.5, off above.** Rejected as a fitted constant.
Measuring both costs one evaluation and needs no threshold to go stale.
