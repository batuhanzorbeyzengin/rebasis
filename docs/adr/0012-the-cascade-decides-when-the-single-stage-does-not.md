# 12. The cascade decides when the single stage does not

**Status:** Accepted · **Date:** 2026-08 · **Evidence:** `docs/cascade-band.md` §6–§7

## Decision

`probe` reports an **arrangement** beside its decision, and sets it to `cascade`
when four conditions hold together:

| condition | why it can veto |
|---|---|
| `cascade_advantage > 1 + BORDERLINE_BAND` | the quantity carries the same measurement error as every other threshold in the rule |
| `bridge_advantage <= 1 + BORDERLINE_BAND` | where bridging alone pays, a rerank stage is cost for nothing |
| the store returns document text | `Cascade` re-embeds its candidates and refuses at construction without it |
| `candidate_reuse` was measurable | an arrangement whose price was assumed is not one this tool recommends |

The candidate depth every cascade figure is measured at moves from **100 to
200**, in `probe` and in `serve.Cascade` together, and `--cascade-n` exposes it.

## Context

[Section 9](../bridge-band.md#9-what-the-counting-is-worth) demolished the
single-stage product thesis: over the 57 runs the repository holds, bridging beat
keeping the current model in 3, and the count that used to say otherwise was an
identity. `docs/cascade-band.md` measured the two-stage arrangement on the same
ladder and found it beating the status quo in **36 of 48** — 36 of the 37 runs
where a reindex is genuinely an upgrade.

So the largest measured win in this repository was one the tool measured, served
(`rebasis.serve.Cascade`), and did not recommend. A user ran `probe`, read
`full_reindex`, and left.

**What blocked it was a price rather than a doubt.** The arrangement re-embeds N
documents per query; how many of those are already cached is a property of a
query distribution, and `probe` reads a corpus. `bridge_advantage` costs a matrix
multiply whatever the traffic, so the rule ran on the number it could price.

That objection does not survive `--queries`. A real query log **is** a sample of
the distribution, and what sets the hit rate is how much the candidate sets
overlap across queries — countable straight off a search the run already ran. No
extra model call, no extra scan. It had simply never been counted.

## Evidence

### M1 — the count is a lower bound on a running cache's hit rate

`tools/cascade_reuse.py`, 48 runs on the same ladder. For each run:
`candidate_reuse` over a **sample** of the judged query log, against the hit rate
a real `Cascade` with a real cache reached replaying the **whole** log in order.

| sample | mean count | mean hit rate | mean gap | count at or below the hit rate |
|---|---|---|---|---|
| 25% of the log | 0.610 | 0.850 | **+0.240** | **48 / 48** |
| 50% of the log | 0.750 | 0.850 | **+0.101** | **48 / 48** |
| 100% (the identity control) | 0.853 | 0.850 | −0.002 | 44 / 48 |

The bound holds on every run at every sample size that is actually a sample, and
the count is not vacuous: across the 48 runs it ranks them by the hit rate they
went on to achieve at Spearman ρ = **+0.955**.

**Where it breaks is a cache capacity, not a rounding error.** Three runs — all
`cqadupstack/tex` — have a candidate working set larger than the 50,000 vectors
`MemoryVectorCache` holds by default. The cache evicts, the same document is
embedded twice, and the count then sits **above** the real hit rate by up to
0.061. So the claim is bounded precisely: `candidate_reuse` is a lower bound on
the hit rate of a cache **large enough to hold the working set**. Of the 45 runs
whose working set fits, 42 embedded exactly `|union|` documents and the other
three differ by one.

The row at fraction 1.00 is the control and not a finding: on the same query set
with a cache that does not evict, the two quantities are the same arithmetic —
misses are the union's size, requests are the sum of the set sizes. It is printed
because agreement there is what proves the rows above it are about something else.

### M2 — the rule against the constant rule

`tools/band_stats.py --view arrangement`, over the 48 runs in
`reports/band/cascade.jsonl`. The arrangement wins on 36 of them, so "always
recommend cascade" is 75% accurate and is the baseline to beat.

| | fires on | of those, won | precision |
|---|---|---|---|
| the shipped rule | 23 / 48 | 23 | **1.0000** |
| the same rule with the single-stage gate at 1.0 | 21 / 48 | 21 | 1.0000 |
| always recommend cascade | 48 / 48 | 36 | 0.7500 |

**On accuracy the rule loses**, 35 of 48 against the constant rule's 36 of 48, and
that number is published rather than omitted. It is also the wrong summary:
accuracy scores silence as a wrong answer, and `arrangement` sits beside a
decision rather than replacing one, so declining to name the arrangement is not
asserting that it loses. What the rule is for is naming runs that will win, and
of the 23 it named, 23 won.

The two tests that survive an outcome this one-sided:

- **Ranking.** Spearman ρ = **+0.939** (p ≈ 6e-23), Kendall τ = +0.794, between
  `cascade_advantage` and the margin the arrangement actually returned.
- **Selection.** The 23 runs the rule named returned a mean margin of **+0.446**
  against **+0.035** for the 25 it did not; permutation test over which runs it
  selected, 10,000 draws, seed 20260825, **p = 0.0001**.

### The identity check

[Section 9](../bridge-band.md#9-what-the-counting-is-worth)'s lesson is that the
same mistake can be made twice, so it was checked rather than asserted.
`bridge_advantage` collapses because both of its factors are read at the same
cut-off on the same metric. `cascade_advantage` is not that: retention at depth
200 on recall, times an upgrade at k=10 on recall, predicting the graded nDCG@10
of a **reranked** list. Measured, `|cascade_advantage − (1 + margin)|` has a
maximum of **0.5874** and a mean of 0.1508 over the 48 runs, and **0 of 48** sit
inside the tolerance two rounded ratios can be compared at.

## What was rejected

**A sixth `Decision` value.** `Decision` is part of the `--json` contract and a
new value silently breaks every script branching on that field
(`docs/stability.md`). It would also be wrong on its own terms: `full_reindex`
says rebuild the index, `cascade` says leave the index alone and add a stage to
the query path. Different axes, and a run can honestly be told both.

**Assuming a cache hit rate.** A default of "say 50%" would decide a
recommendation on a number nobody measured. Where it cannot be counted the
arrangement stays `single_stage` and the report says why.

**`bridge_advantage <= 1` exactly**, which is what this item's plan specified.
The decision rule already tells a run at 1.01 that bridging and doing nothing
cannot be told apart; a gate that then read 1.01 as a win for the single stage
would claim the precision `BORDERLINE_BAND` exists to deny. Both variants are
measured above and the band version is the better of the two — 23 runs named
against 21, at the same precision. It costs one false negative all the same:
`wordpress, potion→MiniLM` sits at 1.041 on the single stage and returned +80.4%
under the arrangement.

**Making `Cascade` the default serving path.** `Bridge` is a 15 µs,
dependency-free, thread-safe object. `Cascade` holds an embedder and a cache.
Merging them costs `Bridge` every guarantee it carries.

**Keeping the depth at 100.** While the number only decorated a report, 100 was
the smaller claim and half the hot-path cost. It now decides, and a threshold
calibrated on the 36-of-48 measured at 200 applied to a figure computed at 100 is
two measurements wearing one name.

## Consequences

- A run whose single stage loses can now be told what wins, with its price. The
  price is not optional: `cascade_embeddings_per_query` is printed wherever the
  arrangement is, because the arrangement's main risk is what it costs.
- The default candidate depth doubles, which doubles the cold-cache cost of the
  first query after a deploy. That is the trade the report now states.
- `candidate_reuse` is computed only at the tiers whose queries are real —
  held-out documents and synthesised questions produce candidate sets whose
  overlap describes the sampling scheme, not anybody's traffic.
- The claim is bounded by what M1 measured: a sample's overlap under-states a
  running cache's hit rate on the corpora and depth measured here. A different
  traffic shape — one whose popular documents are not popular in the sample —
  is not covered, and nothing in `probe` can see it.
