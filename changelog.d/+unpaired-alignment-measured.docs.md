**Unpaired alignment was measured, and it recovers most of what a paired fit does — on the rungs where its first stage works at all.**

The roadmap has carried "fit an adapter with no correspondence between the two spaces" as a direction for some time, most recently pointing at [mini-vec2vec](https://arxiv.org/abs/2510.02348). `spikes/unpaired_align.py` implements it and runs it against the paired ceiling `rebasis fit` reaches on the same data: 36 cells over four corpora, three ladder rungs and three seeds, with the two halves sharing no document and the split asserted rather than assumed.

Median recovery is **0.81** of the paired ceiling. Excluding the one cross-family rung it is **0.84**, floor 0.61, ceiling 0.94 — from a map given no pairs at all.

**The failures are total and they are all in stage one.** `potion-base-8M → all-MiniLM-L6`, a 256→384 jump across model families, recovers between 0.00 and 0.66 depending on corpus, against 0.77–0.93 for every same-family rung. That is worth being able to *see* rather than infer, so the spike reports a **centroid-agreement** diagnostic: how often the quadratic assignment pairs a centroid with the one an oracle map — fitted on the paired data the unpaired fit is forbidden to touch — would have chosen. It is a reference rather than a truth, because two disjoint halves have no exact centroid correspondence, and it is labelled as one wherever it appears.

It is also the best predictor available. Ranking the 36 cells by eventual recovery:

| signal | Spearman ρ | p |
|---|---|---|
| centroid agreement | **+0.833** | 3e-10 |
| ICP final objective | +0.628 | 4e-05 |
| QAP objective — *what the method itself reports* | +0.519 | 1e-03 |
| orthogonality error of the fitted map | −0.231 | 0.17 |
| pre-fit geometry bound | +0.223 | 0.19 |

The QAP objective says how *confident* the matching is; the agreement says whether it is *right*. A high, stable QAP score sitting next to near-chance agreement is a specific and diagnosable failure, and without the second number it reads as "the method does not work here" — which is a different and much less useful conclusion. The relationship is strong and not a clean threshold: below 0.20 agreement the mean recovery is 0.14 with one cell at 0.68, and above it the mean is 0.81 with one cell at 0.04. It orders runs rather than classifying them.

Nothing ships. This is a spike, the roadmap item stays open, and the honest limit is stated there: the case the direction exists for — an index holding vectors whose text is gone — cannot be constructed from a corpus that still has its text, so what has been shown is that the correspondence is recoverable without being given, not that it survives the setting a user would actually be in.
