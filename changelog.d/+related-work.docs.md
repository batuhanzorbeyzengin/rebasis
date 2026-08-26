`docs/related-work.md`: where rebasis sits in the literature, and — the part
that earns the page — which neighbouring approaches its users structurally
cannot take. A decade of backward- and forward-compatible representation
learning (BCT [arXiv:2003.11942](https://arxiv.org/abs/2003.11942), FCT
[arXiv:2112.02805](https://arxiv.org/abs/2112.02805), BiCT
[arXiv:2204.13919](https://arxiv.org/abs/2204.13919), MixBCT
[arXiv:2308.06948](https://arxiv.org/abs/2308.06948)) solves the same
problem — a better model, an index you cannot afford to rebuild — more
completely than any adapter can, and every method in it needs something a user
calling a vendor's API does not have: the new model's training run, the old
model's classifier, labelled data, or a per-item feature that had to be stored
before the old index existed. That is a hard boundary, not a quality comparison,
and the Embedding-Converter paper draws it independently in its own related work.

Three findings came out of reading the papers rather than summarising them.
**Seo et al.** ([arXiv:2301.03767](https://arxiv.org/abs/2301.03767), WACV 2025)
is the closest analogue to `rebasis.serve.MixedSpaceSearch`, and the split is
sharper than expected: their untrained distance rank merge beats BCT and BiCT on
all four datasets while needing nothing, their reverse query transform *is*
rebasis' adapter, and only their metric-compatible contrastive loss is out of
reach — because it needs class labels over the corpus, not because it retrains a
model. rebasis approximates that calibration post-hoc with an isotonic map and
falls back to RRF; how much of the difference that recovers is unmeasured, and
is the page's one open question. **Google's Embedding-Converter** (ACL 2025)
frames its transform as a way to evaluate a candidate model cheaply — it
predicts which of two models is better on 11 of 13 datasets in domain and 12 of
12 out of it, which is `probe`'s claim reached independently — and it runs in
the *forward* direction the roadmap has not measured. And **mini-vec2vec**
([arXiv:2510.02348](https://arxiv.org/abs/2510.02348)) reports, confirmed
against vec2vec's own text, that optimal-transport alignment failed to beat a
naive baseline on sentence embeddings even in an oracle setup — which bears on
the roadmap's plan to try Wasserstein Procrustes first, and suggests
mini-vec2vec's centroid-level assignment as the cheaper step instead.
