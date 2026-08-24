# Golden corpus measurements — the first real evidence

Everything before this document was measured either by the M0 spike (a separate
script) or on synthetic vectors. This is the first time the **rebasis package
itself** produced decisions about real embeddings of a real corpus, and it found
three defects that no synthetic test could have found.

**Setup.** BEIR/scifact, 5,000 documents, 295 queries with human relevance
judgements. Four model pairs, chosen to be four *kinds* of model change rather
than four arbitrary checkpoints. Fixtures built by `tools/make_golden.py`;
assertions in `tests/golden/`.

---

## 1. The measurements

| Scenario | Tier | ARR | 95% CI | Old model | Unadapted | Upgrade gain | Decision |
|---|---|---|---|---|---|---|---|
| `same_family` (L6→L12) | T0 | 0.942 | 0.927–0.956 | 0.897 | 0.694 | — | `bridge_and_migrate` |
| | T1 | 0.946 | 0.907–0.985 | 1.007 | 0.752 | **0.993** | `no_upgrade_needed` |
| `different_family` (MiniLM→bge-small) | T0 | 0.911 | 0.893–0.928 | 0.858 | 0.201 | — | `bridge_and_migrate` |
| | T1 | 0.903 | 0.863–0.942 | **0.944** | 0.267 | 1.060 | `full_reindex` |
| `prefix_trap` (MiniLM→e5-small) | T0 | 0.870 | 0.849–0.891 | 0.838 | 0.083 | — | `bridge_and_migrate` |
| | T1 | 0.921 | 0.870–0.965 | 0.982 | 0.165 | 1.018 | `no_upgrade_needed` |
| `hard_drift` (MiniLM→potion-8M) | T0 | 0.762 | 0.735–0.789 | 0.688 | n/a | — | `caution` |
| | T1 | **0.971** | 0.895–1.052 | 1.184 | n/a | **0.845** | `no_upgrade_needed` |

`unadapted` is n/a for `hard_drift` because 384→256 means the new vector cannot
enter the old index at all — the dimension-mismatch limit, arriving as a
measurement.

---

## 2. Defect: the confidence interval was for a different quantity

The first real run reported:

```
ARR@10   0.908   (95% CI 0.712–0.808)
```

The point estimate sits **outside** its own interval. That is not a wide
interval or a noisy one; it is an interval for something else.

ARR is `mean(candidate recall) / mean(oracle recall)`. The point estimate
divided by the oracle; the interval bootstrapped the raw numerator and did not.
At **T0 the oracle is perfect by construction** (`oracle_recall = 1.0`), so the
two coincide exactly — which is why every synthetic test passed. The defect
existed only at T1, the tier the design describes as the more trustworthy one.

**Fix.** `bootstrap_ratio_ci` resamples query indices once per draw and scores
both series on the same ones — a paired ratio bootstrap. Pairing matters
independently: a query that is hard for the adapter is usually hard for the
oracle too, and resampling them separately would inflate the width by pretending
that correlation away.

*Guarded by* `tests/unit/test_arr_interval.py`, and by an assertion on every
golden scenario that the estimate lies inside its interval at both tiers.

**Generalisation.** A metric and its interval are two implementations of one
definition. Nothing checks that they agree, and the degenerate case — here, a
perfect oracle — can make them agree for the wrong reason.

---

## 3. Defect: the decision compared against the wrong alternative

`different_family` at T1: the adapter recovers **0.903** of what a full reindex
would give. Keeping MiniLM entirely gives **0.944**. Bridging is measurably
*worse than changing nothing*, and the tool recommended `bridge_and_migrate`.

The rule asked two questions and neither was this one:

- ARR against the oracle — "how much of a reindex does the adapter recover?" → 0.903, a good number.
- `upgrade_gain` = oracle ÷ old model — "is the new model better?" → 1.060, yes.

Both true, and the conclusion still wrong, because the user's actual choice is
between *bridging* and *doing nothing* — and that comparison was never made.

**Fix.** `decide()` takes `old_model_arr`. When bridging falls below it by more
than the borderline band, the recommendation becomes `full_reindex` (when the
new model is worth having) or `no_upgrade_needed` (when it is not). Inside the
band it becomes a warning: not enough to overrule a decision, too important to
leave unsaid to someone about to spend an afternoon migrating.

*Guarded by* `tests/unit/test_decision_alternatives.py`, which carries the
scifact numbers as a named regression case.

**Generalisation.** A metric normalised against an ideal answers "how close to
perfect", not "better than what I have now". Those diverge exactly when the
ideal is out of reach — which is the situation this tool exists for.

---

## 4. The prefix trap only traps at T1

The `prefix_trap` scenario is defined as "symmetric to asymmetric: collapses if
prefix handling breaks". Measured, with e5-small-v2:

| Variant | T0 ARR | T1 ARR | T1 decision |
|---|---|---|---|
| Correct prefixes (`passage:` / `query:`) | 0.870 | **0.921** | `no_upgrade_needed` |
| No prefixes at all | 0.875 | 0.870 | `full_reindex` |
| Prefixes swapped | 0.859 | 0.866 | `full_reindex` |

**At T0 the misconfigured variants score *higher*.** At T1 the prefix moves ARR
by 0.05 and changes the recommendation outright.

The reason is structural rather than incidental, and the sign is the giveaway.
T0's ground truth is defined by inner products in the new model's **document**
space. Removing the prefix makes queries and documents share one encoding, which
makes the mapping T0 measures *easier* — so ARR goes up. T0's number reflects how
hard the mapping is, and misconfiguration makes it less hard.

Real queries have no such symmetry: they are text that was never indexed, so a
wrong prefix costs recall against documents a human judged relevant.

The same structure decides the two-adapter comparison for asymmetric models.
Because T0's oracle
lives in document space, an adapter fitted on document-encoded pairs reproduces
it exactly and is optimal there by construction — no query-specific adapter can
beat it at this tier, whatever its quality.

**Consequence.** T0 cannot evaluate the query encoding at all: not the prefix,
not the choice of adapter strategy. That is now stated in the report's T0 caveat
and recorded as [ADR 8](adr/0008-t0-cannot-see-the-query-encoding.md). A
subsequent change also made T0 encode its proxies the way a query is encoded,
which is more correct in itself — for asymmetric models the earlier figure was
optimistically biased — but does not close this gap.

---

## 5. ARR is not comparable across scenarios

`hard_drift` produced the **highest** T1 ARR of all four — 0.971, against 0.903
for `different_family`. Read as "quality retained", that says static distilled
embeddings bridge better than a same-dimension transformer. They do not.

`upgrade_gain` for that pair is **0.845**: potion-base-8M retrieves *worse* than
MiniLM on this corpus. The oracle is weak, and ARR divides by the oracle — so a
high ratio against a poor reference means very little.

The tool got the answer right (`no_upgrade_needed`) because the upgrade question
is asked before the bridging question. But the number is misleading in
isolation, and the golden tests assert **decisions** exactly and ARR only within
a band for exactly this reason: a numerical difference is acceptable, a different
decision is not.

---

## 6. Two models were missing from the profile table

`all-MiniLM-L12-v2` and `potion-base-8M` are both well known, and neither was in
the table — so building the fixtures failed with `RB-E2003`, and the error told
the user to pass `--dim` and `--query-prefix`, **flags that do not exist on any
command**.

Both models were added, with dimensions measured rather than assumed (384 and
256). The missing CLI flags are still missing: a user whose model is not in the
table still cannot run `probe`. That is recorded as an open gap, not fixed here.

---

## 7. What these measurements do not establish

- **One corpus.** scifact is scientific abstracts: short, dense, a narrow
  vocabulary. Bands measured here are not bands for an Obsidian vault.
- **Four pairs, all small models.** Every model is 256–384 dimensional. Nothing
  here says how a 1024-dimensional pair behaves.
- **295 queries.** Wide intervals — the T1 half-widths run from ±0.02 to ±0.08 —
  which is why the bands are ±0.04 and the decisions are what get asserted.
- **The decisions are scifact's, not universal.** `no_upgrade_needed` for
  L6→L12 says those models are equivalent *on scientific abstracts*. It is not a
  statement about the models.

---

## 8. Re-running

```bash
# Rebuild the fixtures (needs models and the corpus; ~2 minutes on the host)
uv run --extra sentence-transformers --with ir-datasets --with model2vec \
    python tools/make_golden.py --out tests/golden/data

# The assertions (no models, no network — the fixtures hold the vectors)
uv run pytest tests/golden -m slow -q
```
