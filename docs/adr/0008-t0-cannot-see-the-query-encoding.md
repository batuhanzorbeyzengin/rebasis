# 8. T0 cannot evaluate the query encoding

**Status:** Accepted · **Date:** 2026-08 · **Evidence:** `docs/golden-findings.md`, section 4,
`tests/unit/test_asymmetric_strategy.py`

## Decision

T0 encodes its query proxies **the way a query is encoded**, not the way a
document is. And the limitation this exposes is documented rather than papered
over: T0 cannot tell you anything about the query encoding — not whether the
prefix is right, and not which adapter strategy an asymmetric model wants — one
shared adapter, or a separate query-specific one.

## Context

T0 uses held-out documents as stand-ins for queries. They were originally
encoded as documents, which for a symmetric model is the same thing and for an
asymmetric one is not: it measures "document retrieves document" rather than
what happens at serve time.

Encoding them as queries is straightforwardly more correct. It does **not** turn
T0 into a prefix detector, and understanding why is the useful part.

## Evidence

Removing an asymmetric model's prefixes, measured on BEIR/scifact with
e5-small-v2:

| Variant | T0 ARR | T1 ARR | T1 decision |
|---|---|---|---|
| Correct prefixes | 0.830 | **0.921** | `no_upgrade_needed` |
| No prefixes | 0.875 | 0.870 | `full_reindex` |
| Prefixes swapped | 0.859 | 0.866 | `full_reindex` |

The **misconfigured** variants score *higher* at T0. That is not noise and not a
bug: removing the prefix makes queries and documents share one encoding, which
makes the T0 task easier. T0's ARR measures how hard the mapping is, and
misconfiguration makes it easier.

The same structure shows up in the two-adapter comparison. T0's ground truth is
defined by inner products in the new model's **document** space, so an adapter
fitted on document-encoded pairs reproduces that geometry exactly — it is optimal
by construction, and no query-specific adapter can beat it at this tier
regardless of quality.

## Consequences

- T0 measures the right quantity now, which it did not before. For asymmetric
  models the previous figure was **optimistically biased**.
- The report's T0 caveat names this blind spot alongside the query-proxy
  assumption. A limitation a user is not told about is a limitation they will
  discover as a surprise.
- The two strategies are only genuinely compared at T1. At T0 both are fitted
  and measured — the cost is one extra fit, which M0 puts in the hundreds of
  milliseconds — but the comparison is not informative there.
- `--queries` earns another entry on the list of things only it can answer:
  `upgrade_gain`, whether bridging beats doing nothing, whether the prefix is
  right, and which adapter strategy to use.

## Alternatives

**Keep encoding proxies as documents.** Rejected: it measures a retrieval the
tool never performs.

**Define T0's ground truth in query space instead.** Rejected. The oracle is
meant to be "what a full reindex would return", and a reindex indexes documents
with the document encoding. Changing that to make one comparison work would make
the headline metric answer a different question.

**Refuse to run T0 for asymmetric models.** Rejected as disproportionate. T0
remains informative about drift magnitude and adapter quality; it is specifically
uninformative about the query encoding, and saying so is enough.
