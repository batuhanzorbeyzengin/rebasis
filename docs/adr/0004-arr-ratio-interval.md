# 4. ARR's interval is a paired ratio bootstrap

**Status:** Accepted · **Date:** 2026-08 · **Evidence:** `docs/golden-findings.md`, section 2

## Decision

ARR's confidence interval is computed by resampling query indices once per draw
and taking the **ratio of means** on those same queries. It previously
bootstrapped the numerator alone.

## Context

ARR is `mean(candidate recall) / mean(oracle recall)`. The point estimate divided
by the oracle. The interval did not.

At T0 this is invisible, and not by luck: T0's ground truth *is* the new model's
output, so `oracle_recall` is 1.0 by construction and the ratio equals the mean.
Every synthetic test in the suite ran at T0.

The first run against a real corpus with real queries printed:

```
ARR@10   0.908   (95% CI 0.712–0.808)
```

The estimate is outside its own interval, which no correct interval can produce.

## Decision detail

**Ratio, because that is what ARR is.** Dividing the numerator's interval by the
oracle afterwards would be closer but still wrong: it treats the oracle as a
constant when it is estimated from the same sample.

**Paired, because the two series are measured on the same queries.** A query that
is hard for the adapter is usually hard for the oracle. Resampling them
independently discards that correlation and reports a wider interval than the
data supports — measured, materially wider.

## Consequences

- Intervals at T1 are correct, and narrower than an unpaired ratio would give.
- `GroundTruth` carries `oracle_recall_per_query`, which it did not need before.
- Golden tests assert, for every scenario and both tiers, that the estimate lies
  inside its interval. That assertion would have caught this on day one.

## What this says about testing

The degenerate case made the wrong code look right. T0's perfect oracle is a
special case of the general formula, and the suite only ever exercised the
special case. **A metric and its interval are two implementations of one
definition, and nothing checks that they agree** — so something has to, on data
where the two can differ.
