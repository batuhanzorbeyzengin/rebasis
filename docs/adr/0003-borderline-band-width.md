# 3. The borderline band is ±0.025

**Status:** Accepted · **Date:** 2026-08 · **Evidence:** `docs/m0-findings.md`, section 10

## Decision

A result within **±0.025** of a decision threshold is reported as borderline.
The width first proposed was ±0.005.

## Context

The decision bands are 0.10 wide, and the decision they produce is what a user
acts on. ARR is estimated from a sample, so a result near a boundary can land on
either side depending on which documents were drawn. The borderline band exists
to say so.

±0.005 is a claim about precision. It asserts that the measurement resolves ARR
to half a percentage point.

## Evidence

Bootstrap 95% CI half-width for ARR, measured:

| Tier | Sample | Half-width |
|---|---|---|
| T0 | ~1,000 held-out proxies | **±0.024** |
| T1 | real query logs | **±0.042** |

The proposed band was five to eight times narrower than the uncertainty it was
supposed to describe. A result 0.01 from a threshold would have been reported as
settled while the interval spanned the boundary.

## Consequences

- The band matches the measurement instead of flattering it.
- More results are reported as borderline. That is the point: they *are*
  borderline, and the previous number was hiding it.
- The report also says when the **confidence interval itself** straddles a
  threshold, which is stronger than the band and is what a careful reader wants.
- Narrowing the band later requires narrowing the interval first — more samples,
  or a real query log — which is the honest order.

## Alternatives

**Keep ±0.005 and report the interval separately.** Rejected: two signals that
disagree, and the smaller number is the one people read.

**Widen to ±0.042, the T1 figure.** Rejected as too wide for T0, where the
measurement is genuinely tighter. ±0.025 sits at the T0 uncertainty, and the
interval check covers T1's wider spread.
