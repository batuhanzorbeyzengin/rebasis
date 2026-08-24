# What drift is, and why an adapter can fix it

## The problem

An embedding model maps text into a vector space. A *different* model maps the
same text into a *different* space. The two spaces are not comparable: a vector
from the new model, dropped into an index built by the old one, retrieves
essentially nothing useful. Often it cannot even be inserted — many stores lock
a collection's dimension at creation, so a 768-dimensional vector cannot enter a
384-dimensional index at all.

That is why the standard answer is to re-embed everything. It is correct, and on
a large personal corpus it is also expensive enough that people simply do not
upgrade.

## Why an adapter works

The two spaces are different, but they are not *unrelated*. Both were trained on
text, both put semantically similar things near each other, and the relationship
between them turns out to be close to a rotation.

That is the useful fact. If the mapping from the new space to the old one is
approximately linear, it can be learned from a few thousand matched pairs — the
same documents encoded both ways — and applied to a query vector in a few
microseconds. The index never changes; only the query does.

The oldest form of this is the orthogonal Procrustes problem: given matched
point sets, find the rotation that best aligns them. It has a closed-form
solution via SVD, no iteration and no hyperparameters, and it is the first thing
rebasis tries.

## Why some drift is *not* a rotation

A rotation preserves distances and angles. Real model changes do more than
rotate:

**The spaces have different means.** Measured on real corpora, subtracting the
mean before fitting raises recovery substantially — it was the single largest
effect M0 found. A rotation cannot express a translation, so the translation has
to be removed first. rebasis centres by default.

**The spaces have different scales, per dimension.** A diagonal rescaling after
the rotation costs `d` parameters and sometimes helps.

**Some of the relationship is not linear at all.** For those cases rebasis fits
a small residual MLP — a linear map plus a learned correction. It is the most
expensive candidate and it wins less often than its flexibility suggests.

**Hubness.** In high dimensions some vectors are near-neighbours of far too many
queries. The CSLS correction penalises them. Measured, it helps weak adapters by
a lot and *hurts* strong ones — so rebasis treats it as a variant to be measured
rather than a correction to be applied.

`auto` fits all of these against the same shared preprocessing and keeps
whichever scores best on a held-out set. Where the top few are within each
other's confidence intervals, the cheapest wins.

## The limit

An adapter recovers the part of the relationship that is learnable from matched
pairs. When the new model is genuinely better because it *understands the
corpus differently* — not just because it arranges the same understanding
differently — that difference lives in the document vectors, and no query-side
map can retrieve it out of an index that does not contain it.

That is the case `full_reindex` names, and it is why rebasis measures instead of
promising. See [the decision rule](decision-rule.md).
