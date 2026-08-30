# Architecture Decision Records

One file per decision that would otherwise be re-argued. Not every decision —
only the ones where a reasonable person would choose differently, and where the
reason is not visible from the code.

Format: what was decided, what it was decided against, and **what evidence
decided it**. A record without evidence is an opinion with a number on it.

| # | Decision | Status |
|---|---|---|
| [0001](0001-mean-centering-by-default.md) | Centre before fitting Procrustes | Accepted |
| [0002](0002-csls-is-a-candidate.md) | CSLS is a candidate, not a correction | Accepted |
| [0003](0003-borderline-band-width.md) | The borderline band is ±0.025 | Accepted |
| [0004](0004-arr-ratio-interval.md) | ARR's interval is a paired ratio bootstrap | Accepted |
| [0005](0005-compare-against-doing-nothing.md) | The decision compares bridging against the current model | Accepted |
| [0006](0006-no-gpu-threshold-for-knn.md) | There is no CPU/GPU crossover for kNN | Accepted |
| [0007](0007-audit-is-tamper-evident.md) | The audit trail is tamper-evident, not tamper-proof | Accepted |
| [0008](0008-t0-cannot-see-the-query-encoding.md) | T0 cannot evaluate the query encoding | Accepted |
| [0009](0009-the-break-even-decides.md) | The break-even decides; the bands describe | Accepted |
| [0010](0010-retention-is-bounded-by-the-source.md) | Retention is bounded by the old model | Accepted |
| [0011](0011-the-hot-path-budget-is-per-dimension.md) | The hot-path budget is per dimension | Accepted |
| [0012](0012-the-cascade-decides-when-the-single-stage-does-not.md) | The cascade decides when the single stage does not | Accepted |

## When to write one

- A measurement contradicted the design, and the code now follows the
  measurement.
- Two reasonable options existed and the losing one keeps being suggested.
- A limit was accepted deliberately, and someone will later read it as an
  oversight.

## When not to

If the code says it clearly, the code is the record. An ADR that restates a
docstring is a second place to keep in sync.
