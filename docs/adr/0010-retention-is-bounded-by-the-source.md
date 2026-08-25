# 10. Retention is bounded by the old model, not by our adapters

**Status:** Accepted · **Date:** 2026-08 · **Evidence:** `docs/bridge-band.md`, sections 3 and 7

## Decision

Stop treating adapter retention as an engineering target. It is bounded by how
much structure the **old** embedding space contains, and neither more fit data
nor more adapter capacity moves it. The tool's job is to measure it and say what
it implies, not to raise it.

## Context

The break-even is `ARR × upgrade_gain > 1`. Of the two factors, `upgrade_gain`
is a property of the models and not ours to change; retention looked like ours.
Measured at 0.47–0.87 across fifteen real-query runs, it is the binding
constraint: at a routine refresh gain of 1.10 a run needs retention above 0.91,
and **none of the fifteen reached it**.

So the obvious move was to raise it. Two hypotheses, both cheap to test, both
wrong.

## Evidence

**It is not the fit-pair count.** M0 found the quality curve flattening at 4,000
pairs, measured at 384→384. The low-retention runs are 256→768, where a linear
adapter's weight is 196,608 parameters and 4,000 pairs is fifteen samples per
input dimension — apparently badly underdetermined. Raising it to 25,000:

| run | 4k | 10k | 25k | change |
|---|---|---|---|---|
| unix, potion→bge-base | 0.414 | 0.415 | 0.429 | +0.015 |
| fiqa, potion→bge-base | 0.466 | 0.465 | 0.476 | +0.010 |
| programmers, potion→bge-base | 0.475 | 0.478 | 0.480 | +0.005 |
| gaming, potion→bge-base | 0.680 | 0.685 | 0.686 | +0.007 |
| unix, MiniLM→bge-small (control) | 0.818 | 0.841 | 0.843 | +0.025 |

Six times the data buys one to two points. M0's default holds at dimensions it
was never measured at.

**It is not adapter capacity.** Across all fifteen fits, including at 25,000
pairs where the residual MLP has ample data, the winner was
`procrustes_centered` **15 times out of 15**. The most constrained candidate in
the list — an orthogonal rotation — beat the ridge affine, the low-rank affine
and the MLP everywhere. What is not a rotation is not learnable by a more
flexible map either.

**It is the source space.** What retention actually tracks:

| predictor | correlation with retention |
|---|---|
| how good the old model is (its own nDCG@10) | **+0.901** |
| source dimension | +0.875 |
| dimension ratio in/out | +0.739 |
| upgrade gain | **−0.958** |

| source model | runs | mean retention |
|---|---|---|
| potion-base-8M (256d, weak) | 5 | 0.547 |
| all-MiniLM-L6-v2 (384d) | 10 | 0.791 |

## The consequence

This explains the −0.958 anti-correlation, and it is arithmetic rather than
coincidence. Retention is a function of the old model's quality. `upgrade_gain`
is the new model's quality **divided by** the old model's. Both are driven by the
same variable, in opposite directions.

Which means: **bridging cannot be a general answer to "the new model is much
better".** The condition that makes an upgrade worth doing — a weak old model —
is the same condition that makes bridging fail. There is a band where the old
model was good enough to map from and the new one is substantially better, and
it is narrow: four of fifteen real-query runs landed in it.

You cannot recover from 256 dimensions what 768 dimensions encode. No adapter
family fixes that, because it is not a modelling problem.

## Consequences

- No change to `--pairs`; the default stays at 4,000 and is now measured at two
  dimension regimes rather than one.
- No new adapter candidates on retention grounds. `auto`'s list is not what is
  limiting these runs.
- The report can say something it could not before: when retention is low
  because the old model is weak, more compute will not help and a reindex is the
  honest path.
- Per-cluster adapters remain worth trying, but for a **different**
  reason — heterogeneous drift within a corpus, which `tail_arr` detects — not
  as a way to lift the ceiling measured here.

## Alternatives

**Add higher-capacity adapters.** Rejected on the measurement: the most
constrained candidate already wins everywhere.

**Raise the default `--pairs`.** Rejected: measured at six times the data for
one to two points, against a real cost in fit time.

**Predict retention from the model pair and skip the fit.** Tempting — the
correlation is +0.901 — and rejected. A correlation over fifteen runs with two
source models is not a predictor, and a tool that guessed instead of measuring
would be the thing this project exists not to be.

## Independently confirmed, and one thing added

Maystre, Ortega Gonzalez, Park, Dolga, Berariu, Zhao and Ciosek,
[*When Embedding Models Meet: Procrustes Bounds and
Applications*](https://arxiv.org/abs/2510.13406), reached the same place from
theory. Their motivating scenario is this one — the query model is upgraded and
the document embeddings cannot be recomputed — and two of their results bear
directly on this decision.

**Why the most constrained candidate wins.** They compare orthogonal Procrustes
against unconstrained linear alignment. By construction the unconstrained
solution cannot be worse on alignment error, and yet orthogonal wins on
retrieval, *particularly when upgrading to a stronger query model* (their
Figure 5): preserving the stronger source model's geometry keeps information an
unconstrained map discards. That is the mechanism behind `procrustes_centered`
winning 15 out of 15 above, arrived at independently.

**A bound is not the prediction that was rejected.** Their Corollary 1 states
that if two models' pairwise inner products agree to within δ, the best
orthogonal alignment satisfies `E[‖x̄ᵢ − yᵢ‖²] ≤ √(2D)·δ`. That is a
one-directional guarantee, data-independent and independent of `N`, and it costs
one Gram-matrix difference — no fit at all.

rebasis now reports δ and the bound it implies (`rebasis.core.geometry`,
`probe`'s report). It does **not** overturn the rejection above, because it is a
different kind of object: it says an alignment of at least this quality exists,
never that retrieval will realise it. The converse does not hold, and a low
bound beside a low ARR is not a contradiction — it means the alignment was
available and something else lost it. The measurement remains the answer.

Two further notes from the same paper, recorded because they touch decisions
made here: their sample saturation sits near 10,000 pairs against M0's 4,000,
which is consistent with the table above showing the point being model-pair
dependent rather than universal; and their zero-padding of the smaller
embedding under a dimension mismatch is the same convention as
`IdentityAdapter`'s padding and the `hard_drift` 384→256 scenario.
