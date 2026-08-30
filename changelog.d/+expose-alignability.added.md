`rebasis expose` — how well somebody holding only your vectors could align them to a space they already understand.

[vec2vec](https://arxiv.org/abs/2505.12540) showed that embeddings translate between models with no paired data, preserving enough geometry to infer things about the underlying documents; its framing is that vector databases reveal almost as much as their inputs. [mini-vec2vec](https://arxiv.org/abs/2510.02348) then reduced the cost of that translation from a day of adversarial training to an orthogonal solve and an assignment. So "they took the vectors, not the text" is a smaller mitigation than it reads as, and nothing measured how much smaller on a particular index.

```bash
rebasis expose --store <uri> --json
```

**It returns a number and nothing else.** No aligned vectors, no reconstructed text, no inversion — asserted by a test that walks every field of the result rather than promised in prose. The distinction is not fastidiousness: a number grants nobody a capability, since mini-vec2vec is published and pip-installable, while a command that returned a translation would package that capability and ship it to this tool's user base.

**The three attempts are the finding, not the fine print.** On BEIR scifact three alignments of the same index return 0.958, 0.940 and 0.904 — a spread of 0.054, and any one of them would have told you the same thing. On nfcorpus they return **0.087, 0.624 and 0.034**. Same index, same models, same seed base, and a seven-fold difference in what a single run would have printed. So `SEEDS = 3` does not buy a better number; it buys the only evidence that the number is unstable, and above `SPREAD_LIMIT = 0.10` the command says so in the output. An earlier draft of the documentation read a finding out of a column of single-attempt nfcorpus numbers — that the corpus mattered more than the model pair — and that finding was an artefact of drawing once; it is withdrawn on the page.

**Storage changes what an attacker gets, and int8 does not.** On scifact, `bge-base` indexed at 768/int8 aligns at 0.942 against 0.947 at float32 — a gap of 0.005 against a measured 0.054 noise floor, which is no change. Binary is the one column that moves it: 0.576 at full width, and 0.058 truncated to 256, where the map ranks the right document 103rd of 1,000. The catch is in the same section: what makes binary tolerable for retrieval is rescoring, and rescoring re-ranks with the full-precision vectors, so a deployment that rescores still holds the vectors the binary storage was meant to remove.

Four further limits, each of which is why the command looks the way it does.

**It is an upper bound.** The measurement draws its reference half from your own corpus, so the alignment saw the very distribution it was attacking; a real adversary has only their own text. Reporting the best of the three attempts is the same reading one level up, since an adversary can also run it three times and keep whichever worked.

**It carries no band.** The plan this was built from proposed reporting the *centroid-agreement* diagnostic — which `spikes/unpaired_align.py` measures ranking outcomes at Spearman +0.833 — and banding it. Two things ruled that out. Centroid agreement needs a reference permutation, which needs an orthogonal map fitted on paired data, which nothing the method can see produces. And it does not need to: an index's owner holds both the vectors and the text, so they can measure how well the alignment *actually* worked rather than a predictor of it. Banding an outcome would be choosing a policy threshold with no labelled harm to calibrate against, and this project does not ship those. `cluster_self_consistency`, which the method *can* see, ranks the outcome of its own run at Spearman +0.900 over twelve cells and holds inside each corpus separately (+0.812 and +0.943); it is in `--json` under `diagnostics` for the same reason, as an explanation rather than a threshold.

**The reference model must be local.** A hosted endpoint is refused rather than warned about — every other command in rebasis warns; measuring exposure by creating some is the one failure that would make this command worse than not having it.

**It is not compliance evidence and not a risk score.** What `SECURITY.md` already says about SOC 2, ISO and the AI Act is unchanged: those certify organisations, and a number a library produces certifies nothing.

[How alignable is your index?](https://batuhanzorbeyzengin.github.io/rebasis/exposure/) is written to be read before the number rather than after it, and leads with the four things it does not say.
