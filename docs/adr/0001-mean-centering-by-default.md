# 1. Centre before fitting Procrustes

**Status:** Accepted · **Date:** 2026-08 · **Evidence:** `docs/m0-findings.md`, section 4

## Decision

`auto` fits **centred** Procrustes by default: subtract each space's mean before
solving for the rotation, restore it after. Plain Procrustes stays in the
candidate list.

## Context

The preprocessing step was originally ℓ2 normalisation and nothing else. That is
the conventional choice and it has a conventional justification: cosine
similarity is scale-free, so normalise and rotate.

The justification is incomplete. A rotation is a linear map that fixes the
origin, so it cannot express a translation — and two embedding spaces do not
share an origin. Whatever offset separates them, the rotation has to absorb, and
absorbing it costs alignment everywhere else.

## Evidence

Across 4 corpora and 3 model pairs (M0, 24 measurements):

| | Mean ARR gain from centering |
|---|---|
| T0 (document proxies) | **+0.166** |
| T1 (real queries) | **+0.260** |
| Best case | +0.75 |

It hurt in 3 of 24 measurements, always by ≤0.018 — inside the ±0.024 confidence
interval measured for ARR at these sample sizes.

Centred Procrustes then matched the residual MLP's quality at half the memory
and a third of the latency, which changed the default candidate ordering too.

## Consequences

- The largest single quality gain in the project came from one subtraction.
- Plain Procrustes is kept so the report can show what centering bought. Removing
  it would save one fit and lose the ability to answer "is this worth it?"
- The adapter stores both means, so `apply` is a translate–rotate–translate. That
  is two extra vector adds on the query path, measured at well under the 15 µs
  budget.

## Centring alone is not a principle — a footnote

The reasoning above is "a rotation cannot express a translation", and it is
sound as far as it goes. What it does not license is the stronger claim that
centring brings two spaces closer together on its own.

Maystre et al., [*When Embedding Models Meet*](https://arxiv.org/abs/2510.13406),
give a two-dimensional counter-example (their Appendix B.3): two spaces where
centring makes the alignment **worse** — `X̃ᵀỸ = 0` after subtracting the means,
while a suitable orthogonal matrix aligns the originals perfectly. Their
measured conclusion is that alignment *with* centring is best and centring on
its own is not principled (their Figure 7).

That is exactly the shape of the decision recorded here: centring is a step
*inside* a fit for a rotation, never a substitute for one. It also gives the
three regressions above something to be. All three are `e5→bge`, and a
systematic loss confined to one model pair is what a counter-example looks like
when it shows up in practice rather than in two dimensions.

Nothing changes. The default was set on 24 measurements and holds; what changes
is the claim made for it, which is now the measured one rather than a general
argument the literature does not support.

## Alternatives

**Normalisation only, as specified.** Rejected on the measurement.

**Centre as a separate adapter type rather than a default.** Considered; it is
what the candidate list does in effect. Making it the *default* is the part that
matters — a user who never reads the candidate table still gets it.
