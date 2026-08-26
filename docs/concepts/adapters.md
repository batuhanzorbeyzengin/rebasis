# Adapters

An adapter is a small learned function from the new model's space to the space
your index already uses. `auto` fits every one of these and keeps the best on a
held-out set.

## The candidates

### Orthogonal Procrustes

Finds the rotation that best aligns matched pairs. Closed-form via SVD: no
iteration, no learning rate, no hyperparameter to tune, and the same input
always gives the same output.

`d × d` parameters. On the query path it is a single matrix multiply.

### Centred Procrustes

The same, with the mean removed from both sides before fitting and restored
after. A rotation cannot express a translation, and the two spaces have
different means, so without this the rotation spends itself absorbing an offset.

This is the default, on measurement rather than principle: it was the largest
single effect found in the M0 spike, and it did no harm in any configuration
tested.

### Diagonal scaling (DSM)

A per-dimension rescale applied after the rotation. `d` extra parameters,
negligible cost, occasionally a real gain.

### Ridge affine

An unconstrained linear map with L2 regularisation. More expressive than a
rotation — it can stretch and shear — and correspondingly more able to overfit a
small sample.

### Low-rank affine

The affine map, truncated to rank `r`. Fewer parameters and less memory.

Worth knowing what this does and does not buy: it halves the arithmetic but
*doubles the number of numpy calls*, and at query scale the call overhead
dominates. It is a memory optimisation, not a latency one — measured, it is
slightly **slower** than full Procrustes on a cloud vCPU.

### Residual MLP

A linear map plus a small learned non-linear correction. The most expressive
candidate, the most expensive to fit, and the one most able to overfit. It wins
when the relationship genuinely is not linear, and `auto` will only keep it when
it beats the cheaper options on data it was not fitted on.

Needs PyTorch, which is an optional extra. Without torch installed, `auto`
simply does not consider it.

### Identity

Pads or truncates and does nothing else. Not a serious candidate — it is the
baseline the others are measured against, and it makes "how bad is doing
nothing" a number rather than a guess.

## Choosing between them

`auto` scores every candidate on a held-out set that is disjoint from the
fitting set. Disjointness is checked at runtime, not only in tests: if the two
ever overlapped, every quality figure rebasis has produced would be
meaningless, so it fails loudly rather than warning.

Where several candidates land within each other's confidence intervals, the
tie-break is cost: fewer parameters, less memory, faster to apply. A 0.002
advantage that is inside the noise is not an advantage.

## Score calibration

An adapter can preserve ranking perfectly and still move the *absolute*
similarity scores. That matters more than it sounds: plenty of RAG pipelines
filter on a fixed threshold like `similarity > 0.7`, and a shifted score
distribution empties that filter silently.

rebasis fits an isotonic calibrator alongside the adapter. Isotonic regression
is monotone, so applied to one list of scores it cannot reorder them. Measured,
it takes the distribution shift from 0.92 to 0.09 — and the report still warns
you if your pipeline uses a fixed threshold, because the right fix is to retune
it.

**"Cannot reorder" is a property of the calibrator, not of everything downstream
of it, and the difference has bitten once.** Isotonic regression is a *step*
function: pool-adjacent-violators produces far fewer output levels than it has
inputs, so distinct scores collide. Anything that then sorts has to break those
ties on something, and `serve.calibrated_merge` broke them on the document id —
which at the endpoints of a migration, where one space is empty and there is a
single right answer, reproduced the underlying ranking on as few as 4% of
queries. It now carries each hit's original rank into the sort, so a shared
level keeps the order it arrived in. If you calibrate scores yourself, the same
care applies: the transform preserves order, and a sort over its output does not
unless you tell it how to.

## What the file contains

An `.rbs` adapter is a directory:

```
adapter.rbs/
├── manifest.json        # dimensions, model ids, profile fingerprints, hashes
├── weights.safetensors  # the tensors
├── calibration.json     # the isotonic calibrator
└── eval.json            # what it scored, on what, when
```

The manifest carries a fingerprint of both encoding profiles. Loading an adapter
against an index it was not built for is refused rather than allowed to quietly
return worse results. `rebasis eval <adapter.rbs> --verify` recomputes every
tensor hash.
