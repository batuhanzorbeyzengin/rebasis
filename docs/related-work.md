# Related work, and the door that is closed

The work nearest to rebasis is not the work on vector-database migration tools.
It is two older literatures: **alignment between two fixed embedding spaces**,
which is what rebasis does, and **compatible representation learning**, which
solves the same user-facing problem — a better model, an index you cannot afford
to rebuild — and solves it more completely than any adapter can.

The second is why this page exists. Almost none of it is available to the people
rebasis is for, and the reason is structural rather than a matter of quality:
they have a vendor's API or a downloaded checkpoint, an index some library
built, and — usually — the document text, because the store kept it. **What they
do not have is either training run.** Stating that makes the tool's scope
legible; leaving it out makes the scope look like an oversight.

| Approach | What it needs | Available? |
|---|---|---|
| Orthogonal Procrustes, Drift-Adapter, Maystre et al. | a sample of the same items in both spaces | **yes** — this is `fit` |
| Embedding-Converter | the same, plus a rewrite of every stored vector | yes, at the index's expense |
| Seo et al., rank merge | nothing; two searches per query | **yes** |
| Seo et al., reverse query transform | paired data, both models frozen | **yes** — this *is* rebasis' adapter |
| Seo et al., metric-compatible contrastive learning | class labels over the corpus | no |
| BCT, MixBCT | training the new model; labelled data | **no** |
| BiCT | training the new model to be backward compatible | **no** |
| FCT | a per-item auxiliary feature stored when the old index was built | **no**, and not obtainable after the fact |
| vec2vec | unpaired samples from both spaces; GPU-scale adversarial training | partly |
| mini-vec2vec | unpaired samples from both spaces; a CPU | plausibly |
| Wasserstein Procrustes | an approximate one-to-one match between the two point sets | only if one exists |

---

## 1. The same arrangement: two frozen spaces and a map between them

**Schönemann (1966)** is the solve. *A generalized solution of the orthogonal
Procrustes problem*, Psychometrika 31:1–10
([doi:10.1007/BF02289451](https://doi.org/10.1007/BF02289451)), gives a
closed-form least-squares `T` for `AT = B + E` subject to `TᵀT = I`; its stated
advance over Green's earlier solution is holding for `A` and `B` of less than
full column rank. `rebasis.core.procrustes` calls
`scipy.linalg.orthogonal_procrustes`, which is that algorithm, so sixty years
separate the paper from the tool and one function call.

**Drift-Adapter** — Vejendla, [*Drift-Adapter: A Practical Approach to Near
Zero-Downtime Embedding Model Upgrades in Vector
Databases*](https://arxiv.org/abs/2509.23471) (arXiv:2509.23471, EMNLP 2025
Main) — is the closest published match to what rebasis ships: a transform fitted
on a small sample of paired old/new embeddings, mapping new queries into the
legacy space so the existing ANN index is queried unchanged. Its three
parameterisations — Orthogonal Procrustes, Low-Rank Affine, a compact Residual
MLP — are rebasis' candidate list under different names, and it reports
recovering 95–99% of a full re-embedding's Recall@10 and MRR, against rebasis'
measured retention of 0.71–0.72. The two do measure different quantities — the
paper's ground truth is the new model's own neighbours, not human judgement — but
that is not where the disagreement turns out to live: running the paper's
protocol on its own corpora and model pair, this project's harness measures
0.24–0.50 rather than 0.95–0.99, and an adapter-independent ceiling puts the
published band above what the old space holds.
[The comparison](vs-drift-adapter.md) is where that is worked out, including what
would change the conclusion.

**Maystre et al.**, [*When Embedding Models Meet: Procrustes Bounds and
Applications*](https://arxiv.org/abs/2510.13406) (arXiv:2510.13406), is the
theory this repository already leans on — [ADR 1](adr/0001-mean-centering-by-default.md),
[ADR 10](adr/0010-retention-is-bounded-by-the-source.md),
[the cascade band](cascade-band.md), [index health](index-health.md). Their
motivating retrieval scenario is stated in the same words this project uses:
document embeddings are fixed and cannot be recomputed "because the raw
documents are unavailable", while the query model can be updated. Three results
matter. Corollary 1 bounds the best orthogonal map's alignment error by the
average squared deviation in pairwise dot products — data-independent, one
Gram-matrix difference, and `probe` reports it. Figure 5 finds orthogonal
alignment beating unconstrained linear "especially when upgrading to a stronger
query model", because preserving the stronger source model's geometry retains
information an unconstrained map discards; that is the mechanism behind
`procrustes_centered` winning 15 of 15 in ADR 10. And their alignment saturates
"after roughly 10,000 samples", against rebasis' default of 4,000.

One requirement of theirs rebasis does not carry: their protocol embeds
documents with the source model and queries with the target, so it runs both.
`rebasis fit` runs only the new one — the old vectors come back out of the index.

### Embedding-Converter runs the other way, and is framed as an instrument

Yoon and Arık, [*Embedding-Converter: A Unified Framework for Cross-Model
Embedding Transformation*](https://aclanthology.org/2025.acl-long.1237/) (ACL
2025, pages 25464–25482), is Google's version, and it differs on the two axes
that matter most.

**Direction.** Its converter maps source-model embeddings into the *target*
model's space: the corpus is converted, the queries encoded with the target
model. Its query-conversion variant (Table 4) swaps which side is transformed
but not the direction of the map — a small model's query is converted *up* into
the large model's space, where the corpus already lives. rebasis only ever
produces `query_to_old`: the new model's query mapped *down* into the old index.
The ROADMAP lists the forward direction under 0.3 as "a different trade-off, and
which one wins has not been measured", and this paper is the strongest existing
evidence about that half. Its numbers are not directly comparable — Table 1 puts
the converted corpus between source and target, 0.5067 → 0.5362 → 0.5609
nDCG@10 in-domain for gecko003 → gecko004 — because that arrangement rewrites
every stored vector, which is a migration rather than a bridge. The converter is
a four-layer MLP; [index health](index-health.md) measures what an unconstrained
map of that kind does to an HNSW graph when every vector moves. The paper does
not discuss the ANN index at all.

**Framing.** Figure 1 contrasts "(a) Conventional evaluation framework" with
"(b) Proposed evaluation framework", and §4.3 is explicit: "Traditionally,
determining the better model would require computing embeddings with both.
However, Embedding-Converter offers a compelling alternative … the relative
performance of source and target models is accurately predicted by
Embedding-Converter on 11 of the 13 datasets" — and "perfectly predicted" on the
twelve held-out CQADupStack datasets. The framing is not *only* evaluation; the
paper also claims the migration outright, "efficiently transfer an entire corpus
to a new embedding space with minimal performance loss". But the evaluation
claim is the one it leads with.

That is `rebasis probe`'s claim, arrived at independently and scored the same
way: does the cheap transform predict which model is better without paying for
the expensive answer?

**The number that used to sit beside their 11 of 13 was 61 of 62, and it has
been withdrawn** — it was an identity rather than a measurement
([section 9](bridge-band.md#9-what-the-counting-is-worth)). The two claims are
not directly comparable anyway: theirs is one converter trained across 14 BEIR
datasets and asked to generalise, predicting a *ranking* of two models, where
`probe` fits per corpus and predicts whether bridging beats keeping the current
model on a specific index. What this project can put beside 11 of 13 is the
quantity that survived the identity check — the estimate ranks runs by the
margin they returned at Spearman ρ = +0.60 over 57 runs — and a ranking claim
scored as a ranking is exactly what `rebasis compare` is for. The convergence is
the useful part: the case for treating an adapter as an instrument rather than a
product has now been made twice, from opposite ends.

### ERA points the same way a third time, from the query side

Maekawa et al., [*Align Then Adapt: Label-Efficient Adapter Learning for
Asymmetric Dense Retrieval*](https://arxiv.org/abs/2604.03403)
(arXiv:2604.03403, April 2026) is the mirror image of this project's problem
rather than a competitor to it. Their setting is a **strong query embedder over
a lightweight document index**: align the two spaces using unlabelled corpus
documents, then adapt the aligned query representation with a small number of
labelled query-document pairs. Over 126 retrieval tasks in 6 domains, and — the
sentence that matters here — **without re-indexing the corpus.**

Three things are worth separating.

**The arrangement is the same one.** Leave the documents where they are, move
the query. That is `Bridge`, arrived at from the opposite end: rebasis starts
from an index whose model is out of date, ERA starts from a document embedder
chosen to be cheap. Two motivations, one shape, and neither cites the other.

**The alignment stage is the same operation.** Aligning two spaces from
unlabelled corpus documents is what `fit` does on document pairs. What ERA adds
is the second stage — a supervised adaptation of the query side — and that is a
capability rebasis does not have and has not measured. Their reported gains
(up to 8.2 nDCG@10 in the symmetric setting, over 12 in the asymmetric one) are
gains from *both* stages together, so they are not a number this project's
single-stage retention can be read against.

**Its asymmetric case is the one rebasis is weakest on.** ADR 10 says retention
is bounded by what the old space holds, and the wider the gap between the two
models the less there is to carry. ERA's asymmetric setting is exactly that gap,
and its answer to it is labelled data. That is a direction, not a result anyone
here has reproduced.

---

## 2. Compatible representation learning, and the boundary

There is an older answer to "the model improved and the index is stale", and it
is better than an adapter because it does not approximate anything: **train the
new model so its output is already compatible with the old index.** The
literature is a decade of image retrieval, and the reason none of it is here is
one sentence long — it needs the training run.

The clearest statement of that boundary is not in this repository. It is in the
Embedding-Converter paper's own related work, dismissing the family for the same
reason: BCT and FCT "require modifying the training process of new models or
rely on unavailable 'side information', respectively, which are infeasible with
fixed, pre-trained models."

**BCT** — Shen et al., [*Towards Backward-Compatible Representation
Learning*](https://arxiv.org/abs/2003.11942) (arXiv:2003.11942, CVPR 2020 oral)
— founded it. The new model is trained with an added *influence loss* that
scores its features under the **old model's learned classifier**, biasing the
new space toward one the old gallery vectors can still be compared against.
Three things are missing here, not one: the new model's training loop, the old
model's classifier weights, and a labelled set for the loss to run on. What it
buys is a genuinely backfill-free upgrade — no adapter, no approximation — and
what it costs is the subject of everything after it: compatibility is paid for
out of the new model's own accuracy.

**MixBCT** — Liang et al., [*MixBCT: Towards Self-Adapting Backward-Compatible
Training*](https://arxiv.org/abs/2308.06948) (arXiv:2308.06948) — simplifies
that to a single classification loss over mixed old and new features, still
training the new model, and additionally requiring the *old* model to be
runnable over the new training data. Its central finding travels, and this
repository reached it independently from the other side: previous methods
"overlooked the impact of the old model's quality", and as the old model gets
worse, intra-class variance in its features grows and the methods degrade.
[ADR 10](adr/0010-retention-is-bounded-by-the-source.md) measures the retrieval
version — retention correlates with the old model's own nDCG@10 at +0.901 — on
text, with adapters, at a different task. Two literatures, one shape: **what you
can recover is bounded by what the old space holds.**

**BiCT** — Su et al., [*Privacy-Preserving Model Upgrades with Bidirectional
Compatible Training in Image Retrieval*](https://arxiv.org/abs/2204.13919)
(arXiv:2204.13919) — should be read by anyone using rebasis, because it is aimed
at rebasis' real blind spot. Its scenario is a system that "should only retain
the extracted gallery embeddings, discarding the raw images for privacy
protection". An index holding vectors and no text is exactly the index
`rebasis fit` cannot work from: there is nothing to re-embed. BiCT's answer is to
train the new model to be backward compatible and then push the old gallery
embeddings forward into it — the training run, twice.

**FCT** — Ramanujan et al., [*Forward Compatible Training for Large-Scale
Embedding Retrieval Systems*](https://arxiv.org/abs/2112.02805)
(arXiv:2112.02805, CVPR 2022) — is the sharpest boundary of the four, and the
most interesting, because it does *not* modify the new model's training; its
stated advantage over BCT is exactly that "training of the new model is not
modified, hence, its accuracy is not degraded". What it requires instead is a
decision made **before the old index existed**: stored alongside every gallery
vector, a per-item *side-information* feature learned at old-training time (they
use SimCLR) to carry what the old objective discarded but a future one might
want. Nobody reading this made that decision five years ago, and it cannot be
made retroactively for an index that already exists. Seo et al. measure what FCT
is worth without it: on ImageNet the relative gain falls from 62% to 28%.

### Seo et al. is the one that already ships here, halfway

[*Metric Compatible Training for Online Backfilling in Large-Scale
Retrieval*](https://arxiv.org/abs/2301.03767) (Seo, Uzunbas, Han, Cao and Lim,
arXiv:2301.03767, WACV 2025) is the closest academic analogue to
`rebasis.serve.MixedSpaceSearch`. The comparison is worth doing carefully,
because the obvious reading — "they published what we shipped" — is wrong in
both directions.

Their problem is a *partially* backfilled gallery: some items re-extracted with
the new model, some not, and a system that must keep serving throughout. Three
components, with very different requirements.

**Distance rank merge** is the baseline, and the paper says plainly it is
"model-agnostic, free from extra training". Search the old gallery with an
old-model query and the new gallery with a new-model query; for each query keep
whichever hit has the smaller distance. Their Figure 2 shows mAP and CMC rising
monotonically as backfilling progresses, without negative flips, "even though
the old and new models are not compatible with each other". Their Table 1 puts
that untrained baseline at 36–45% of the new model's gain — **ahead of BCT
(4–27%) and BiCT (16–36%) on every one of the four datasets.** A method needing
nothing beats two that need the training run.

`MixedSpaceSearch` is the same shape with a different merge, and the difference
is the point. Rank merge compares raw distances across the two spaces. rebasis
will not: `serve/hybrid.py` maps old-space scores through an isotonic calibrator
fitted on held-out scores, and falls back to reciprocal rank fusion — discarding
the scores entirely — when there is no calibrator, because M0 measured a median
KS distance of **0.924** between the two score distributions, so raw comparison
would let one side win for reasons unrelated to relevance. Seo et al. hit the
same wall from the other side: "the scales of distance in the embedding spaces
of the two models could be significantly different."

**Reverse query transform** is their fix for encoding each query twice, and it
is, almost exactly, rebasis' adapter. A lightweight network maps a new-model
embedding onto the old model's, both models' parameters frozen, minimising the
distance to the true old embedding on paired examples. They give the same
directional argument Maystre et al. give: it runs new → old rather than FCT's
old → new, and "since the embedding quality of a new model is highly likely to
be better than that of an old one, our reverse transformation module performs
well even without additional side information". This component needs nothing a
rebasis user lacks — it is what `fit` produces.

**Metric-compatible contrastive learning** is where the door closes, and not
where this section's general argument says it should. It retrains neither model.
It trains the same reverse transform with a supervised contrastive loss (their
Eq. 11) whose negatives are drawn from *both* retrieval systems, so distances in
one become directly comparable to distances in the other. What it needs is
**class membership for every training sample**, plus hard-example mining over
those labels. A corpus of documents has no classes. Their fourth component goes
further, adding a learned module on top of the new model and using the
composition as the model of record — so the new index would hold vectors
nobody's API produces.

The honest accounting: **rebasis implements the untrained half of this paper and
cannot implement the trained half.** The distance calibration Seo et al. achieve
with labels, rebasis approximates post-hoc with an isotonic map fitted on
held-out scores, and abandons for RRF when it has none. That is a weaker
instrument for the same purpose under strictly weaker assumptions — and their
ablation says calibration is where their gains come from: their joint loss beats
both single-system variants consistently, and the variant that calibrates each
system independently "still suffers from negative flips". Whether an isotonic
calibrator recovers any of that has not been measured here, on any corpus. It is
the most concrete open question this page turned up.

---

## 3. Unpaired alignment, and a correction to the plan

First, what "unpaired" means for this tool, because the README's version is
imprecise. `rebasis fit` never loads the old model: it calls `probe_store` with
no old embedder, reads the old vectors out of the index and re-embeds the
index's own text with the new model. The requirement is that the store returns
**vectors and text**. What needs the old model is the *decision* — encoding a
real query log with it is how `upgrade_gain` is measured, and without that a run
is `provisional` and declines to say whether bridging is worth doing. So the
genuinely unpaired case is BiCT's case: an index that kept vectors and discarded
the source text.

The ROADMAP puts **Wasserstein Procrustes** first in that direction — Grave,
Joulin and Berthet, [*Unsupervised Alignment of Embeddings with Wasserstein
Procrustes*](https://arxiv.org/abs/1805.11222) (arXiv:1805.11222, AISTATS 2019,
[PMLR 89:1880–1890](https://proceedings.mlr.press/v89/grave19a.html)). It jointly
estimates an orthogonal matrix and a **permutation** matrix, initialised from a
convex relaxation of the quadratic assignment problem and minimised
stochastically, evaluated on unsupervised bilingual lexicon induction over word
embeddings. The reasoning for putting it first is sound as far as it goes: the
orthogonal solve already exists here and already wins, and the only new part is
the assignment step.

**The literature says that step is the problem.** mini-vec2vec — Dar,
[*mini-vec2vec: Scaling Universal Geometry Alignment with Linear
Transformations*](https://arxiv.org/abs/2510.02348) (arXiv:2510.02348) — names
Grave et al. twice. In §3's *Motivation*, before the algorithm is described,
optimal-transport methods "rely heavily on matching points between the sets
(often one-to-one, but this can be alleviated), and they become computationally
intensive … In practice, when Jha et al. (2025) have compared their method to
multiple OT methods, **in an oracle setup, they didn't work better than the most
naive baseline**, indicating that even in the oracle setup, and even as a first
step for learning a mapping, they are insufficient." And in §6: they "require
the existence of a match between points, at least approximately, in order to
work."

Checked against the primary source rather than taken second hand. vec2vec's
oracle-aided OT baseline solves the assignment with Hungarian, Earth Mover's
Distance, Sinkhorn or Gromov-Wasserstein, best solver per experiment, over
embeddings "derived from the same underlying texts, strongly favoring OT
methods". Its result: on same-backbone pairs OT is "comparable to" the naive
cosine baseline, and on cross-backbone pairs "baseline methods perform similarly
to random guessing". The claim holds — **with one qualification that matters for
the plan.** vec2vec did not run Grave et al.'s algorithm. It ran assignment
solvers; Wasserstein Procrustes optimises the permutation and the orthogonal map
*together*. The evidence is against the family and mini-vec2vec draws the
inference, but it is not a direct measurement of what the ROADMAP proposes.

**mini-vec2vec is the better first step, and closer to this codebase than
Wasserstein Procrustes is.** Its pipeline: mean-centre and ℓ2-normalise both
spaces — which is [ADR 1](adr/0001-mean-centering-by-default.md),
independently, and the paper notes the mean vector is "a dominant part of all
vectors" — then k-means each space, match the *centroids* by solving a small QAP
over their similarity matrices, build pseudo-parallel pairs, solve orthogonal
Procrustes, and refine with ICP, running Procrustes again each iteration. It
runs on a Colab CPU runtime with scikit-learn and scipy, claims 60k samples
against vec2vec's two million, and argues the small assignment is *why* it
works: "the small scale does more than speed up the process, but likely also
improves the performance of the algorithm." That is the same orthogonal solve
plus an assignment step the ROADMAP wants, with the assignment moved off the
data points — the move the OT evidence above says is necessary.

**vec2vec** — Jha, Zhang, Shmatikov and Morris, [*Harnessing the Universal
Geometry of Embeddings*](https://arxiv.org/abs/2505.12540) (arXiv:2505.12540,
NeurIPS 2025) — is what mini-vec2vec is measured against, and the source of the
OT evidence above: translation between embedding spaces with no paired data, no
encoders and no predefined matches, via adversarial losses and cycle consistency
through a shared latent space. The ROADMAP's reason for ranking it second is
confirmed by the paper's own accounting: its experiments consumed "almost 176
GPU days for training". For a tool that runs on a laptop that is the argument,
not a detail.

Two pieces of standing context sit under all of this. **Relative
representations** — Moschella et al., [*Relative representations enable
zero-shot latent space communication*](https://arxiv.org/abs/2209.15430)
(arXiv:2209.15430, ICLR 2023) — represents each sample by its similarities to a
fixed set of anchors, invariant to latent isometries and rescalings and needing
no additional training; mini-vec2vec's default path uses it directly, over the
matched centroids as anchors. And the **Platonic Representation Hypothesis** —
Huh, Cheung, Wang and Isola
([arXiv:2405.07987](https://arxiv.org/abs/2405.07987)) — is the conjecture that
any of this works at all: that representations across models and modalities are
converging on a shared statistical model of reality. It is argued rather than
proved, and its own §6 lists the counterexamples — different modalities carry
different information, the mathematical argument holds strictly only for
bijective projections, and the choice of alignment measure is itself contested.

---

## 4. Excluded by a decision, not by ignorance

Some literature is missing from this page because a decision in [the
roadmap](https://github.com/batuhanzorbeyzengin/rebasis/blob/main/ROADMAP.md)
rules it out, and it is worth saying which decision rather than leaving the gap
ambiguous.

- **Backward- and forward-compatible training** — §2 above, in full — is
  excluded by *"rebasis does not retrain models"*. It is the strongest answer to
  the problem and it is not on the roadmap at any horizon, because the premise
  is that the user did not train and cannot retrain either model.
- **ANN index construction, quantisation and graph maintenance** are excluded by
  *"becoming a vector database"*. [Index health](index-health.md) measures what a
  migration does to somebody else's graph and reports it; it does not fix it,
  and "rebuild your index" is not advice the tool gives, because that document
  also shows it is the right remedy for one failure and useless for another.
- **Approximate-ground-truth evaluation** — sampled or ANN-derived kNN, the
  standard cost-saving in retrieval benchmarking — is excluded by *"approximate
  ground truth"*. The kNN behind every ARR here is exact.
- **Embedding inversion** — vec2text and its successors — is not a competing
  method but the reason translated embeddings are treated as sensitive data.
  mini-vec2vec makes the connection explicit: unsupervised alignment extends an
  inversion attack to a proprietary model whose weights the attacker lacks. Not
  surveyed here, but recorded — rebasis provides the same capability.

---

## What this does not establish

**Read in full**, from the paper's own text: Embedding-Converter (ACL 2025,
PDF), Seo et al. (2301.03767), BCT (2003.11942), FCT (2112.02805), mini-vec2vec
(2510.02348), Wasserstein Procrustes (1805.11222). **Read in part** — abstract,
method and the sections cited above: Maystre et al. (2510.13406), Drift-Adapter
(2509.23471), vec2vec (2505.12540), BiCT (2204.13919), MixBCT (2308.06948),
relative representations (2209.15430), Platonic (2405.07987). **Abstract only:**
Schönemann (1966) — Psychometrika is paywalled, so title, journal, pagination,
DOI and the abstract's statement of the contribution are all that could be
verified, and that `scipy.linalg.orthogonal_procrustes` implements it comes from
scipy's own attribution rather than from the 1966 paper.

**Figures were not read.** Every claim above comes from prose or from a table
rendered as text. Where a result lives in a figure — Maystre et al.'s Figures 4
and 5, Seo et al.'s Figures 2, 6 and 8 — what is reported here is the sentence
the authors wrote *about* that figure, which is a weaker thing.

**Venues are asserted only where verified.** EMNLP 2025 Main, CVPR 2020, CVPR
2022, ICLR 2023, ACL 2025, AISTATS 2019 and WACV 2025 were each confirmed
against a proceedings listing or the publisher's page. **BiCT and MixBCT are
cited by arXiv identifier alone**, because their venues could not be confirmed
from a primary source.

**Nothing here is a measurement.** The only numbers quoted are the papers' own,
on their own datasets; the only rebasis numbers are cross-references to
documents that measured them. Three comparisons are structural rather than
measured — that the untrained half of Seo et al. is what `MixedSpaceSearch`
implements, that Embedding-Converter's direction is the one the roadmap has not
measured, and that mini-vec2vec's shape fits this codebase. None has been run
against rebasis' harness, and the calibration gap above is the one most worth
closing.
