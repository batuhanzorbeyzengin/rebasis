# How alignable is your index?

`rebasis expose` answers one question and returns one number.

```
$ rebasis expose --store chroma:///path/db#documents

  Alignability   0.184  (1,000 documents, mean rank 27.4)
  Reference      sentence-transformers/all-MiniLM-L6-v2 (local)
  Sample         20,000 of 487,213  seed 0
```

An adversary holding only your vectors, plus a public embedding model over
**their own** documents, can fit a map between the two spaces with no paired
data at all. That is not speculative: it is
[vec2vec](https://arxiv.org/abs/2505.12540)'s finding, and
[mini-vec2vec](https://arxiv.org/abs/2510.02348) reduced the cost of it from a
day of adversarial training to an orthogonal solve and an assignment. This
measures how well it works on your index.

---

## What it does not do

Read this list before the number.

**It returns no translation.** No aligned vectors, no reconstructed text, no
inversion. Not "does not print one" — does not compute one: nothing on the path
produces raw coordinates, and a test asserts that every field of the result is a
scalar or a string. A number grants nobody a capability; mini-vec2vec is
published and pip-installable already. A `translate` command would package that
capability and ship it to this tool's user base. The first is defence, the
second is distribution, and this is the first.

**It does not say which documents could be exposed.** It does not identify a
record, a cluster or a topic. The measurement runs on a hold-out and returns how
often the map was right, not which times.

**A high number does not mean you have leaked.** It means that *if* your vectors
are taken, they can be translated. Whether they can be taken is a question about
your access controls, and this tool knows nothing about those.

**A low number is not an assurance.** It is measured against one family of
methods with one reference model. A better method may appear tomorrow, and a
better reference model exists today for somebody who looks harder than the
default. [ADR 7](adr/0007-audit-is-tamper-evident.md) draws the same line for
the audit trail: what a mechanism buys is worth stating, and so is what it does
not.

---

## Why the number is an upper bound

The measurement is deliberately kinder to the attacker than reality is.

It splits a sample of **your** corpus in two halves that share no document,
reads the index's vectors for one half and embeds the other half's **text** with
the reference model. A real adversary has none of your text. They have their
own, drawn from some other distribution, and every difference between those
distributions makes the centroid matching that the whole method starts from
harder.

So the figure says: *at best, this is what a translation achieves.* It errs
towards alarming you, which is the right direction for a number about exposure
and the wrong direction for a number about cost — the same reasoning
`candidate_reuse` follows in the opposite sense.

There is a second, smaller sense in which it is a bound, and it points the other
way: it is a bound **for one reference model**. `--reference` takes another.

---

## Why there is no low / medium / high

The plan this command was built from proposed banding it. That was the right
instinct for the quantity it proposed reporting, and the wrong one for the
quantity that ships. Two things changed.

**Centroid agreement is not computable by a user.** The earlier design reported
it — the diagnostic `spikes/unpaired_align.py` measures ranking the outcome at
Spearman **+0.833**, against +0.519 for the method's own confidence signal. But
that diagnostic needs a *reference permutation*, and a reference permutation
needs an orthogonal map fitted on **paired** data. The spike's own docstring
says it: *nothing the method can see produces this*. It is available to a
measurement harness that holds both encodings of the same documents. It is not
available to a command.

**And it does not need to be.** Centroid agreement is a *predictor* of how well
the alignment worked. An index's owner holds the vectors and the text, so they
can compute how well it worked — directly, on a hold-out, which is what
`alignability` is. Reporting a predictor of a number you are holding is strictly
worse than reporting the number.

That change removes the case for a band along with the case for the predictor. A
band on a predictor is a **measurement**: you have outcomes, you find the
threshold that separates them, and
[ADR 3](adr/0003-borderline-band-width.md) is this project doing exactly that.
A band on an outcome is a **policy**: somebody deciding that 0.4 is acceptable
and 0.6 is not, with no labelled "this index was exfiltrated" to calibrate
against. That is a judgement about your risk tolerance, and no measurement this
project can take produces it.

So: the number, the pool it is relative to, and this page.

---

## What the number is

The fraction of held-out documents whose **own** index vector the map ranked
first, out of the hold-out pool.

The pool is part of the number, not a footnote to it. Identifying one document
among 1,000 is a weaker result than identifying it among 8,192, which is what
mini-vec2vec's own paper ranks against. `mean_rank` is reported beside it for
the same reason: a map that puts the answer second every time and a map that
puts it five-hundredth score the same `alignability` and are not the same
finding.

### What it looks like elsewhere

`tools/exposure_band.py`, 48 cells: four corpora (SciFact, NFCorpus, ArguAna and
CQADupStack/android), three indexed models, two reference models, two seeds.
Each held-out document is ranked against 1,000 others.

**Excluding the 16 same-model cells** — those are a positive control, not a
result: there is nothing to align, and they come back at a median of 1.000 to
show the measurement can reach it at all.

| | cells | min | median | max |
|---|---|---|---|---|
| **everything** | 32 | 0.001 | **0.083** | 0.998 |
| reference from the same publisher | 8 | 0.003 | **0.275** | 0.998 |
| reference from a different publisher | 24 | 0.001 | **0.083** | 0.993 |

The range spans three orders of magnitude. That is why this page offers you a
range rather than a verdict: 0.18 is unremarkable in this table and 0.18 is also
a fifth of a thousand-document hold-out identified from vectors alone, and which
of those readings matters is yours.

---

## What raises it

Measured, on the grid above:

**The reference model, more than anything else.** A reference from the same
publisher as the index's model raises the median from **0.083 to 0.275** — more
than three-fold. That is the one thing on this list an adversary controls and
you do not, and it means the number `expose` prints with its default reference
is a floor on what a better-chosen one would reach.

**The corpus, enormously.** Same models, same protocol, four collections:

| corpus | documents sampled | median alignability |
|---|---|---|
| ArguAna | 8,674 | **0.745** |
| SciFact | 5,183 | 0.261 |
| CQADupStack/android | 20,000 | 0.044 |
| NFCorpus | 3,633 | 0.007 |

A hundred-fold between the top and the bottom. What separates them is not size —
the largest sample is third from the top — and this measurement does not
establish what it is. ArguAna is argument counter-retrieval, where every
document is a self-contained argument; NFCorpus is nutrition abstracts. That
these differ by two orders of magnitude, and that nothing here says why, is the
strongest argument for measuring your own index rather than reading this table.

**The seed, which is the uncomfortable one.** The same corpus, models and
sample, run at two seeds: the median spread is **0.159** and the maximum is
**0.969**. The method is stochastic in three places — the k-means
initialisation, the assignment's restarts, the ICP sampling — and on some
indexes that decides the answer.

  So `expose` runs **three** alignments by default and reports the **best**,
  which is the reading consistent with everything else on this page: an
  adversary can also run it three times and keep whichever worked. Where the
  attempts disagree by more than 0.10 the run says so and prints all three. A
  single-attempt number on an index in that regime is not a measurement of the
  index.

---

## The reference model must be local

`--reference` takes a model id and a hosted endpoint is **refused**, not warned
about. Every other command in rebasis warns; `pyproject.toml` already records
that `openai_compat` is the only backend that can send document text off the
machine. Here the warning is not enough: measuring exposure by creating some is
the one failure that would make this command worse than not having it.

---

## Why this is in rebasis rather than in a repository of its own

The plan this was built from left that open, to be decided after the
measurement: *if the evidence supports a classifier, split it out; if it
supports only a number, keep it as a diagnostic command inside rebasis.*

It supports only a number, and the decision follows. Three things settle it:

**The shared code is most of it.** The sampling is `probe/session.py`, the
alignment is `core/unpaired.py`, and the Procrustes solve is the one
`core/procrustes.py` already calls. A separate repository would either duplicate
those or depend on rebasis for them, and the second is a package that exists to
add one command.

**A number is not a different liability from a number.** The case for splitting
was a different audience and a different liability surface, and both rest on the
command handing over a capability. It does not: it returns a scalar, and
mini-vec2vec is published and installable by anybody who wants the capability
itself. What is left is a diagnostic about an index rebasis is already connected
to, which is what every other command here is.

**It belongs beside the thing it is about.** Somebody who has just run `probe`
against their index is exactly the person who should be told this, and a second
package they have to hear about first is a package nobody runs.

If the evidence ever supports a classifier — a threshold with outcomes behind it
rather than a policy — that is when the question is worth reopening, and this
paragraph is what it should be reopened against.

## Where this sits

This is a diagnostic, not a security product, and rebasis is not a security
tool. `SECURITY.md` says what the project does and does not claim; the same
register applies here. In particular:

- It is **not compliance evidence**. What `SECURITY.md` says about SOC 2, ISO
  and the AI Act applies unchanged: those certify organisations, not libraries,
  and a number a library produces certifies nothing.
- It is **not a risk score**. There is no scale, no weighting and no aggregation
  behind it — one measurement, reported as itself.
- The audit trail still carries no vectors and no text, by construction. This
  command does not change that and could not: it holds neither by the time it
  has an answer.

## Reproducing

```bash
rebasis expose --store <uri> --json

uv run --extra sentence-transformers --with ir-datasets --with model2vec \
    python tools/exposure_band.py --corpora heldout --corpora beir \
    --cache-dir ~/band-cache --out reports/band/exposure.jsonl
uv run python tools/exposure_band.py --summarise reports/band/exposure.jsonl
```
