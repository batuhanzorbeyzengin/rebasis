`rebasis expose` — how well somebody holding only your vectors could align them to a space they already understand.

[vec2vec](https://arxiv.org/abs/2505.12540) showed that embeddings translate between models with no paired data, preserving enough geometry to infer things about the underlying documents; its framing is that vector databases reveal almost as much as their inputs. [mini-vec2vec](https://arxiv.org/abs/2510.02348) then reduced the cost of that translation from a day of adversarial training to an orthogonal solve and an assignment. So "they took the vectors, not the text" is a smaller mitigation than it reads as, and nothing measured how much smaller on a particular index.

```bash
rebasis expose --store <uri> --json
```

**It returns a number and nothing else.** No aligned vectors, no reconstructed text, no inversion — asserted by a test that walks every field of the result rather than promised in prose. The distinction is not fastidiousness: a number grants nobody a capability, since mini-vec2vec is published and pip-installable, while a command that returned a translation would package that capability and ship it to this tool's user base.

Four limits, each of which is why the command looks the way it does.

**It is an upper bound.** The measurement draws its reference half from your own corpus, so the alignment saw the very distribution it was attacking; a real adversary has only their own text.

**It carries no band.** The plan this was built from proposed reporting the *centroid-agreement* diagnostic — which `spikes/unpaired_align.py` measures ranking outcomes at Spearman +0.833 — and banding it. Two things ruled that out. Centroid agreement needs a reference permutation, which needs an orthogonal map fitted on paired data, which nothing the method can see produces. And it does not need to: an index's owner holds both the vectors and the text, so they can measure how well the alignment *actually* worked rather than a predictor of it. Banding an outcome would be choosing a policy threshold with no labelled harm to calibrate against, and this project does not ship those.

**The reference model must be local.** A hosted endpoint is refused rather than warned about — every other command in rebasis warns; measuring exposure by creating some is the one failure that would make this command worse than not having it.

**It is not compliance evidence and not a risk score.** What `SECURITY.md` already says about SOC 2, ISO and the AI Act is unchanged: those certify organisations, and a number a library produces certifies nothing.

[How alignable is your index?](https://batuhanzorbeyzengin.github.io/rebasis/exposure/) is written to be read before the number rather than after it, and leads with the four things it does not say.
