# The decision rule

`probe` returns a decision, not a score. This page is how that decision is
reached and why the thresholds are where they are.

## The metric

**ARR — Adapted Recall Retention** — is the fraction of what a full reindex
would have retrieved that the adapter actually retrieves. It is a *ratio*, not a
bare recall, which matters: with real queries and human judgements, a full
reindex does not achieve perfect recall either, so dividing by the oracle asks
the question you actually care about — how much of the achievable am I keeping?

## The bands

| ARR | Decision | What to do |
|---|---|---|
| ≥ 0.95 | `bridge_sufficient` | Use the new model now. Leave the index alone. |
| 0.85 – 0.95 | `bridge_and_migrate` | Bridge today, migrate in the background. |
| 0.70 – 0.85 | `caution` | Check whether the loss is concentrated in part of the corpus. |
| < 0.70 | `full_reindex` | An adapter will not close this gap. |

And, independent of ARR:

| Condition | Decision |
|---|---|
| The new model is not measurably better on this corpus | `no_upgrade_needed` |

That fifth outcome was added because the measurements demanded it. Across the
M0 configurations, simply keeping the old model averaged 0.983 of what the new
one achieved — which means for many real corpus/model pairs the honest answer is
that the upgrade is not worth doing at all. A tool that only ever answers "how
should I upgrade" cannot say that.

It is only measurable with a real query log. Without one, the ground truth *is*
the new model's output, so the new model scores perfectly against itself by
construction.

## The arrangement, beside the decision

The decision says what to do with the **index**. A second field, `arrangement`,
says what to put in **front** of it, and the two are independent — a run can
honestly be told `full_reindex` and `cascade` at once.

| `arrangement` | What to do |
|---|---|
| `single_stage` | Whatever the decision says. The bridge, if any, produces the final ranking. |
| `cascade` | Leave the index alone. The bridge fetches a candidate set and the new model reranks it in its own space. |

It is a separate field rather than a sixth `Decision` because a new value in that
literal breaks every script branching on it, and because they are not
alternatives to each other.

`cascade` is set when four things hold together: the two-stage break-even clears
the ±0.025 band, the single stage does not already win, the store returns
document text, and the arrangement's **price** could be counted. That last
condition is why the field exists at all. The arrangement re-embeds N documents
per query, and what that costs turns on how many are already cached — a property
of your traffic. Given `--queries`, `probe` counts how much the candidate sets
overlap between your queries and reports it as `candidate_reuse`: a lower bound
on the hit rate, because a running cache accumulates across queries a sample does
not contain. Without a real query log it stays unpriced, and an unpriced
arrangement is reported rather than recommended.

Measured over 48 runs: of the 23 the rule named, 23 won. A rule that always
recommended the arrangement would have been right 36 times out of 48.
[ADR 12](../adr/0012-the-cascade-decides-when-the-single-stage-does-not.md) has
the evidence, including the test the rule loses.

## The borderline band

ARR is estimated from a sample, so it has an interval around it. If that
interval straddles a band boundary, two runs on the same corpus can land on two
different recommendations — and a user changing machines would get different
advice for the same data.

rebasis reports **borderline** when the estimate is within ±0.025 of a
threshold.

The design originally proposed ±0.005. Measurement found the actual sampling
uncertainty to be ±0.024 with document proxies and ±0.042 with real queries, so
±0.005 claimed a precision that does not exist. The band was widened to match
what was measured. When a result is borderline, the report says so and suggests
a larger sample rather than picking a side.

## Ground truth tiers

**T0 — document proxies.** Held-out documents stand in for queries; the ground
truth is what the new model itself retrieves. Always available, needs nothing
from you, and rests on the assumption that your queries resemble your documents.
Unbiased in the M0 measurements, with a wide error bar.

**T1 — real queries.** Your query log plus relevance judgements. Narrower
conclusions, a genuine oracle to divide by, and the only tier where
`upgrade_gain` — and therefore `no_upgrade_needed` — can be computed. Pass
`--queries`.

The report always states which tier produced the number, because the same ARR
means different things at each.

## Warnings that travel with the decision

**Heterogeneous drift.** When the sparsest clusters score far below the corpus
average, one global adapter is leaving quality on the table in part of the
corpus. The report says so; per-cluster adapters are the fix.

**Score shift.** Ranking can be preserved while absolute scores move. If your
pipeline filters on a fixed similarity threshold, that threshold needs retuning.
This is evaluated **after** calibration — before it, the warning fired in 100%
of measured configurations and therefore said nothing.

## Reproducing a decision

Every decision is written to an append-only audit trail with everything needed
to reproduce it: model ids and profile fingerprints, sampling strategy and seed,
sizes, thresholds, metric version, rebasis version, and the environment.

```bash
rebasis audit list
rebasis audit replay <seq>
```

`replay` re-runs the probe with the recorded inputs and compares. A difference
means either a regression or a changed corpus — both of which you want to know
about.

The trail is a hash chain, which makes it **tamper-evident, not tamper-proof**:
anyone who can write the file can rewrite the whole chain. What it detects is
accidental corruption and partial edits, which is what it is for.
