# How alignable is your index?

`rebasis expose` answers one question and returns one number.

```
$ rebasis expose --store chroma:///path/db#documents

  Alignability   0.184  (1,000 documents, mean rank 27.4)
  Attempts       0.171, 0.184, 0.166  (the best of them is the figure above;
                 the method is stochastic)
  Reference      sentence-transformers/all-MiniLM-L6-v2 (local)
  Sample         20,000 of 487,213  seed 0
```

*(The shape of the output, on an invented index. What the numbers look like on
real corpora is [further down](#what-it-looks-like-elsewhere) and is measured.)*

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
default. And on some corpora the *same* method scores low or high depending on
which attempt you read — measured, [further down](#one-of-these-columns-is-a-number-and-the-other-is-a-coin-flip) —
so a low figure can mean the attack failed rather than that your index resisted
it. [ADR 7](adr/0007-audit-is-tamper-evident.md) draws the same line for
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

Twelve measurements, on two BEIR corpora, with three models standing in turn as
the index and two as the reference. Every document in both corpora was sampled —
scifact 5,183, nfcorpus 3,633 — and every cell ranks a held-out document against
999 others.

**Each cell below is one alignment, not the best of three that the command
runs.** That distinction turns out to matter more than anything else in the
table, and the next section is about it.

| indexed | reference | scifact | nfcorpus |
|---|---|---|---|
| bge-base | bge-small | 0.950 | 0.084 |
| bge-base | all-MiniLM-L6 | 0.875 | 0.001 |
| bge-small | all-MiniLM-L6 | 0.916 | 0.009 |
| all-MiniLM-L6 | bge-small | 0.927 | 0.734 |
| *bge-small* | *bge-small* | *1.000* | *0.991* |
| *all-MiniLM-L6* | *all-MiniLM-L6* | *1.000* | *0.356* |

The italic rows are controls: the reference model **is** the indexed model, so
the two halves are samples from one distribution and the map the method is
looking for is the identity. They are not results. They are there to show what
the harness scores when the answer is free.

### One of these columns is a number and the other is a coin flip

Run the top-left cell three times instead of once and the attempts are 0.958,
0.940 and 0.904 — a spread of 0.054 around a stable answer. Run the top-right
cell three times and they are **0.087, 0.624 and 0.034**: a spread of 0.590,
which is larger than the entire range of the scifact column.

So the scifact column can be read. The nfcorpus column cannot. Those five
numbers are single draws from a distribution wide enough that "0.084" and
"0.734" may be the same cell twice, and no ordering of models, references or
families survives being read off them. An earlier draft of this page read a
finding out of that column — that the corpus mattered more than the model pair —
and the finding was an artefact of drawing once.

What is left is still the most useful thing here, and it is about *your* index
rather than these:

> A number measured on somebody else's corpus tells you very little about yours,
> and on some corpora a number measured once tells you very little either.

Which is the case for running the command rather than reading a table.

### The control is part of the reading

Look at the last row. On nfcorpus, `all-MiniLM-L6` aligned against **itself**
scores 0.356. There is no distribution shift there and no model gap to bridge;
the map the method is looking for is the identity, and it did not find it.

That is a warning about reading a low number as good news:

> A low alignability means *either* your index resists this attack *or* this
> attack failed on your corpus, and the number alone does not distinguish them.

The way to tell them apart is to run the control yourself — pass `--reference`
the same model your index was built with. If that scores near 1.000, the method
works on your corpus and a low number against a public reference is a finding.
If the control is also low, the method failed and you have learned nothing about
your index, in either direction.

### Why three attempts

`SEEDS = 3`, and the two cells above are why.

Three attempts buy almost nothing on scifact: the spread is 0.054, under the
0.10 that triggers a warning, and any one of them would have told you the same
thing. Three attempts on nfcorpus change the answer from 0.087 to 0.624 — the
same index, the same models, the same seed base, and a seven-fold difference in
what gets printed depending on which attempt you happened to make.

The point of the third attempt is not that the best of three is the true number.
It is that one attempt cannot tell you it is unstable and three can. At a spread
of 0.590 the command prints:

> The 3 attempts disagreed by 0.59 (0.087, 0.624, 0.034). The method is
> stochastic …

which is the honest output for that index — more honest than any single figure,
including the best one. `SPREAD_LIMIT = 0.10` is what decides that the sentence
appears; scifact's 0.054 stays silent and nfcorpus's 0.590 does not.

Three rather than five or ten is a cost decision and is recorded as one: each
attempt is a full alignment, the scifact cell took about two minutes per
attempt, and a diagnostic that takes twenty minutes is a diagnostic nobody runs.
Three is the smallest number that can disagree with itself twice.

*(One caveat for anybody reproducing this: the same seed produced 0.084 in one
run and 0.087 in another. Seeding pins the sample, the split and the
initialisation; it does not pin the floating-point arithmetic underneath, and
three documents out of a thousand moved.)*

---

## What raises it

One thing, measured. Two things that a first draft of this page claimed and the
spread measurement took back.

**Whether the alignment worked at all.** `cluster_self_consistency` — the mean
cosine between the source centroids pushed through the fitted map and where
k-means, started from them, actually settled on the target side — ranks the
outcome of **the same run** at Spearman +0.900 across all twelve cells, and it
is not the corpus split doing that work: within scifact alone it is +0.812
(n = 6, p = 0.050) and within nfcorpus alone +0.943 (n = 6, p = 0.005). The
method's own confidence signal, `qap_score_mean`, manages +0.669 over the
twelve. `orthogonality_error` and `refine_objective_final` rank nothing:
−0.518 and −0.147, neither significant.

Within-run is the whole point of it. On a corpus where three attempts return
0.087, 0.624 and 0.034, no property of the corpus can predict the answer,
because there is no single answer to predict — but a quantity computed *inside*
each attempt still tracks how that attempt turned out. That is what
`cluster_self_consistency` does, and it needs no reference permutation, unlike
the centroid agreement discussed [above](#why-there-is-no-low-medium-high).

`--json` carries it under `diagnostics`; the printed output does not, because a
number that needs this much context is not a line in a summary.

**Not the publisher, and not the stored width — neither claim survived.** An
earlier draft reported that different-publisher pairs aligned better than
same-publisher ones (median 0.804 against 0.517), and that 384-dimensional
indexes aligned better than 768-dimensional ones (0.825 against 0.479). Both
medians were dominated by nfcorpus cells that a re-run moves by up to 0.590.
Restricted to the corpus where the measurement is stable, the four non-control
cells are 0.875, 0.916, 0.927 and 0.950 — and neither family nor width separates
them, with only one same-family cell to separate anything by.

So: nothing in this grid raises or lowers alignability except the corpus and the
run. That is a thinner finding than the one it replaces and it is the one the
measurements support.

Why nfcorpus is the unstable corpus is **unverified**. It is smaller (3,633
against 5,183) and topically narrower, and either would plausibly make twenty
k-means clusters less stable across a split — the measured
`cluster_self_consistency` there is 0.52–0.65 against scifact's 0.66–0.74, with
no overlap — but nothing here isolates a cause. A sweep over corpus size at
fixed topic breadth would settle it.

---

## What storage does

An index stored at a quarter of the width, or a thirty-second of the bits, is
a different set of vectors to align. Whether that is a defence is a question,
and on one corpus it has an answer.

`bge-base` over BEIR scifact, reference `bge-small`, seed 0, pool 1,000. The
first two columns are `expose`; the last two are the same corpus and the same
model in the [truncation grid](truncation-band.md), joined on nothing but corpus
and model — two measurements, not one experiment:

| stored as | alignability | mean rank | retrieval retained | rescored |
|---|---|---|---|---|
| 768 float32 | 0.947 | 1.1 | 1.000 | 1.000 |
| 768 int8 | 0.942 | 1.1 | 1.001 | 1.000 |
| 768 binary | **0.576** | 3.2 | 0.906 | 0.999 |
| 256 float32 | 0.825 | 1.5 | 0.938 | 1.000 |
| 256 int8 | 0.726 | 2.8 | 0.937 | 1.000 |
| 256 binary | **0.058** | 103.6 | 0.695 | 0.964 |

Every cell is one alignment, and three attempts on this corpus and model pair
spread by 0.054. **Read nothing below that.** Two of the gaps here are inside
it, one is barely outside, and three are far outside; each is labelled.

**int8 is free for the attacker.** A quarter of the storage, retrieval within
0.001 of full precision, and alignability moves from 0.947 to 0.942 — a gap of
0.005, an order of magnitude inside the noise floor, which is to say no
measurable change at all. Symmetric per-vector int8 keeps the geometry the
alignment is fitting, which is also why it is nearly free for retrieval. The
same property does both. At 256 dimensions int8 does cost something — 0.825 to
0.726 — but that gap is 0.099 against a 0.054 floor, so read it as "possibly
something" rather than as a number.

**Binary is the only column that clearly costs them.** 0.947 to 0.576 at full
width is a gap of 0.371, seven times the floor, and stacked with a cut to 256
the map is ranking the right document 103rd on average out of 1,000. That is
well short of chance, so something survives, and just as far from
identification.

### The catch, and it is a real one

The `rescored` column is what makes binary tolerable for retrieval — 0.906
becomes 0.999 — and rescoring works by re-ranking the top 200 candidates
**with the full-precision vectors**. `probe/truncation.py` passes them in as
`full_documents`. So a deployment that rescores is a deployment that still holds
the vectors the binary storage was supposed to have removed, and an adversary
who reaches the index reaches those too.

The row that actually buys the reduction is therefore the unrescored one:
0.906 retained at 768 binary, 0.695 at 256 binary. Storing binary *and nothing
else* costs about ten points of nDCG at full width and buys a drop from 0.947 to
0.576. Whether that trade is worth taking is not a question this project can
answer for you, and it is at least a trade rather than a free lunch.

**Read this on one corpus only.** These six cells are scifact, where the
alignment works well for every model pairing. On nfcorpus the same attack fails
at float32 already, and there is nothing for a storage choice to buy. One
corpus, one model pair, one seed — enough to show the shape of the trade-off and
not enough to give you a number for your index. `rebasis expose --store` on the
index you actually have is the number for your index.


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
# your own index
rebasis expose --store <uri> --json

# the twelve-cell table above
uv run --extra sentence-transformers --with ir-datasets --with model2vec \
    python tools/exposure_band.py \
    --corpus beir/scifact/test --corpus beir/nfcorpus/test \
    --seeds 1 --seed 0 --cache-dir ~/band-cache \
    --out reports/band/exposure.jsonl

# the storage axis
uv run --extra sentence-transformers --with ir-datasets --with model2vec \
    python tools/exposure_band.py --corpus beir/scifact/test \
    --indexed BAAI/bge-base-en-v1.5 --reference BAAI/bge-small-en-v1.5 \
    --truncate 768,256 --quantize float32,int8,binary \
    --seeds 1 --seed 0 --cache-dir ~/band-cache \
    --out reports/band/exposure-m3.jsonl

# the two three-attempt cells that decided SEEDS
uv run --extra sentence-transformers --with ir-datasets --with model2vec \
    python tools/exposure_band.py \
    --corpus beir/scifact/test --corpus beir/nfcorpus/test \
    --indexed BAAI/bge-base-en-v1.5 --reference BAAI/bge-small-en-v1.5 \
    --seeds 3 --seed 0 --cache-dir ~/band-cache \
    --out reports/band/exposure-spread.jsonl

uv run python tools/exposure_band.py --summarise reports/band/exposure.jsonl
```
