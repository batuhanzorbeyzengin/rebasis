`rebasis probe --access-log` weights which sampled records become query proxies, so ARR describes retention on the questions people actually send rather than on a uniform draw over the corpus.

The sampler has taken weights since it was written and nothing passed them. Connecting them found that the roadmap entry naming this named the **wrong place** for them: a `probe` sample does two jobs at once — it is the mini-index every measurement runs against, and it is the pool the query proxies are split out of. Handing weights to the sampler fills the mini-index with frequently-read documents, which changes the *distractors*, a property of the index rather than of the questions asked of it. The weights go on the split.

Measured over 36 cells and 12,960 replicate probes, that placement leaves the estimate about half as far from the whole-corpus quantity as weighting the sample does (+0.025 against +0.051) ([the numbers](https://batuhanzorbeyzengin.github.io/rebasis/access-weighting/)).

**The confidence interval survives it**, which is what the entry was blocked on. Dividing the bootstrap's median half-width by the estimator's actual spread across replicates gives 1.92 for the plain design against a correctly calibrated 1.96, and 1.84 under weighted queries — about 6% narrow, in the direction the entry worried about and small against decision bands 0.10 wide. Median coverage is unchanged at 0.94; what moves is the tail, from 2 cells under 0.90 to 6.

That check needed one correction on the way: the ratio is read against **1.96**, not 1, because a correctly calibrated 95% interval around a roughly normal estimator is exactly that many standard deviations wide. Read against 1, a correct interval looks twice too wide.

Weighting shifts ARR by a median +0.015 at a 100x access ratio and by up to +0.073, so it estimates a different quantity — and the run says which. `probe --json` carries `access_weighted`, both report formats say so in prose, and a log that names nothing in the sample reports `false` rather than claiming a weighting that did not happen.

Also: a third measurement fell out of the same grid and belongs to the default rather than to the flag. A 4,000-document mini-index is an easier place to retrieve in than the corpus it came from, so today's uniform `probe` already sits **+0.048** above the whole-corpus quantity — a larger gap than anything weighting does.
