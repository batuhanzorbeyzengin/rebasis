# 5. The decision compares bridging against the current model

**Status:** Accepted · **Date:** 2026-08 · **Evidence:** `docs/golden-findings.md`, section 3

## Decision

`decide()` takes `old_model_arr` — what the current model retrieves, on ARR's
scale. When bridging falls below it by more than the borderline band, the
recommendation becomes `full_reindex` (if the new model is worth having) or
`no_upgrade_needed` (if it is not). Inside the band it becomes a warning.

## Context

The rule asked two questions:

- **ARR against the oracle** — how much of a full reindex does the adapter
  recover?
- **`upgrade_gain`** — is the new model better than the old one?

Neither is the question a user is actually deciding. They are choosing between
*bridging* and *doing nothing*, and that comparison was never made.

## Evidence

BEIR/scifact, MiniLM-L6 → bge-small, 295 human-judged queries:

| | ARR |
|---|---|
| Bridged with the best adapter | 0.903 |
| Keeping MiniLM, changing nothing | **0.944** |
| `upgrade_gain` | 1.060 |

Both original signals were positive: the adapter recovers 90% of a reindex, and
bge really is 6% better than MiniLM here. The tool recommended
`bridge_and_migrate` — a migration that would have made retrieval measurably
worse than leaving the index alone.

## Consequences

- A metric normalised against an ideal answers "how close to perfect", not
  "better than what I have". Those diverge exactly when the ideal is out of
  reach, which is the situation this tool exists for.
- `full_reindex` gains a distinct rationale for this case: the gap is real and an
  adapter is not the way to it. That is a different sentence from "the drift is
  too large", and a user needs the difference.
- A loss inside the noise band warns rather than overrules — it cannot carry a
  decision, and someone about to spend an afternoon migrating should still hear
  it.

## Alternatives

**Warn only, never change the decision.** Rejected: the headline is what gets
acted on, and a warning under a `bridge_and_migrate` heading is a footnote to a
wrong answer.

**Make ARR relative to the old model instead of the oracle.** Rejected. ARR's
meaning — "the fraction of a reindex you keep" — is what the decision bands were
calibrated against, and redefining it would invalidate every threshold and every
recorded measurement.
