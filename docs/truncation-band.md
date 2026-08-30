# What a cheaper index would cost

The most common index transformation in the field is not a model change. It is a
cut in dimension and precision, and it raises the question `probe` already
answers: **what do I lose, on my corpus rather than on a benchmark average?**

```bash
rebasis probe --store <uri> --queries queries.jsonl \
  --truncate 1024,512,256,128 --quantize float32,float16,int8,binary --floor 0.95
```

No `--old`, no `--new`, no adapter. The reference is the index's own full-width,
float32 state and every cell is the same vectors held more cheaply — so none of
[ADR 10](adr/0010-retention-is-bounded-by-the-source.md)'s squeeze applies. The
model does not change and the space does not change.

---

## The measurement

Sixteen corpora — the twelve CQADupStack forums, FiQA, SciFact, NFCorpus and
ArguAna — against their own human judgements at nDCG@10, for three models whose
embeddings the band harness already holds. Each cell truncates **both** the
documents and the queries, renormalises, quantizes, searches, and divides by
what the full-width float32 index achieves on the same queries.

A whole grid costs what a single probe costs: the model runs once, and cutting
what it produced is free.

Every figure below is a mean over the sixteen corpora, as a fraction of what the
full-width float32 index achieves. **`bge-base-en-v1.5`, 768 dimensions:**

| dim | float32 | float16 | int8 | binary | binary, rescored |
|---|---|---|---|---|---|
| **768** (full) | 1.000 | 1.000 | 1.001 | 0.873 | **0.999** |
| **512** | 0.980 | 0.980 | 0.979 | 0.797 | **0.995** |
| **256** | 0.921 | 0.921 | 0.921 | 0.590 | **0.956** |
| **128** | 0.820 | 0.820 | 0.819 | 0.353 | 0.824 |
| **64** | 0.632 | 0.632 | 0.631 | 0.152 | 0.593 |

`bge-small-en-v1.5` and `all-MiniLM-L6-v2`, both 384 dimensions:

| dim | bge-small float32 | bge-small int8 | MiniLM float32 | MiniLM int8 |
|---|---|---|---|---|
| **384** (full) | 1.000 | 0.999 | 1.000 | 1.000 |
| **256** | 0.965 | 0.965 | 0.968 | 0.967 |
| **128** | 0.859 | 0.858 | 0.885 | 0.885 |
| **64** | 0.661 | 0.661 | 0.730 | 0.730 |

---

## Three findings

**1. int8 is free, and float16 is free twice over.** At every dimension of every
model the two columns sit within 0.001 of float32 — 1.001, 0.979, 0.921, 0.819,
0.631 against 1.000, 0.980, 0.921, 0.820, 0.632. A four-fold storage reduction
for a difference smaller than the third decimal place. That is the single
largest result here and it is the least surprising one: a unit vector's
components live in a narrow range, and eight bits with a per-vector scale
resolve them well past what a top-ten ordering can distinguish.

**2. Binary alone is poor. Binary plus a rescore is not.** Single-stage binary
retains 0.873 at full width and collapses to 0.590 at 256; with the candidates
reordered by the full-precision vectors it is **0.999 and 0.956**. That is the
cascade's shape on a different axis — a code only has to put the answer
*somewhere* in the top 200, which is a far weaker requirement than ranking it
top ten — and unlike the cascade it costs no embedding at all, because the
full-precision vectors are the ones the index already holds.

  **It is not a storage saving, and that is the catch.** Rescoring means keeping
  both representations. What it saves is the *search*: the candidate scan runs
  over one-bit codes, and only 200 documents are scored at full precision. A
  reader looking for a smaller index should read the int8 column; a reader
  looking for a faster one should read this.

**3. Truncation costs more than precision does, and it is where the corpora
disagree.** Cutting a 768-dimensional model to 512 costs 2 points; to 256, 8; to
128, 18; to 64, 37. Both 384-dimensional models behave the same way relative to
their own width — 0.965 and 0.968 at two thirds, 0.66 and 0.73 at a sixth.

And none of these three is Matryoshka-trained. They tolerate a cut to a third of
their width anyway, which is exactly what
[Takeshita et al.](https://arxiv.org/abs/2605.16608) report — "robust to
truncation without Matryoshka learning, **except in heavy truncation
scenarios**" — and the 64-dimensional row is what "heavy" looks like measured.
A model that *was* trained for it is measured
[below](#the-one-model-that-was-trained-for-this), and the answer is that the
training bought nothing at any shared dimension.

---

## The one model that was trained for this

Every model above was trained without Matryoshka Representation Learning, so
truncating them is an operation performed *on* the model rather than one it was
trained for. `mxbai-embed-large-v1` was — its card states it supports both MRL
and quantization — and it is 1024-dimensional, which makes the comparison
possible at four shared absolute dimensions.

Retention against each model's **own** full-width nDCG@10, so what is being
compared is what truncation costs rather than which model is better. Both
columns are means over the **same sixteen corpora** — the comparison is worth
nothing otherwise, and an earlier draft of this page got it wrong by averaging
`mxbai` over the corpora it had reached and `bge-base` over four more:

| dimensions | mxbai-embed-large (MRL, 1024) | bge-base (no MRL, 768) |
|---|---|---|
| 512 | 0.980 | 0.980 |
| 256 | 0.924 | 0.921 |
| 128 | 0.827 | 0.820 |
| 64 | 0.639 | 0.632 |

**Matryoshka training bought nothing measurable here.** The MRL model is ahead
at three of the four depths and the largest margin is 0.007, against a
corpus-to-corpus spread at the same depths of 0.037, 0.062, 0.146 and 0.276.
Which corpus you run on moves the answer between five and forty times further
than which of these two models you run.

And the sign of that 0.007 does not survive changing the average. Taking medians
instead of means, `bge-base` is ahead at 64 by 0.016 — the two distributions are
skewed differently and the model difference is inside that. A difference whose
direction depends on whether you take the mean or the median is not a
difference.

Read by fraction of width the MRL model does cut further for the same retention
— 0.980 at half width against 0.980 at two thirds — but that is the advantage of
starting at 1024 dimensions, not of the training.

This reproduces [Takeshita et al.](https://arxiv.org/abs/2605.16608) on a
different corpus family. Their title is the finding — *Text Embeddings are
Robust to Truncation Without Matryoshka Learning, Except In Heavy Truncation
Scenarios* — and the 64-dimensional row is the exception they name, where both
models fall to around 0.63 whatever their training.

**And it bought no steadiness either.** The spread across the sixteen corpora is
0.037 against 0.038 at 512, 0.062 against 0.065 at 256, 0.146 against 0.166 at
128, and 0.276 against 0.249 at 64: narrower for the MRL model at three depths
by margins far smaller than the spreads themselves, and wider at the fourth. An
earlier draft claimed MRL was the steadier of the two; that came from the same
unequal corpus sets and is withdrawn.

*(Sixteen corpora: twelve cqadupstack subforums, plus arguana, fiqa, nfcorpus
and scifact. One MRL model, at one width. What a second one, or a corpus family
further from question-answering, would show is not measured here.)*

---

## Where the spread is

If the per-corpus variance is low, the published averages are enough and this
flag is unnecessary. If it is high, "measure on your own corpus" is proved
again. **Both outcomes are here**, and which one you are in depends on how deep
the cut is.

Max minus min across the sixteen corpora, per cell, for `bge-base`:

| dim | float32 | int8 | binary |
|---|---|---|---|
| 768 | 0.000 | 0.009 | 0.130 |
| 512 | **0.038** | 0.043 | 0.118 |
| 256 | **0.065** | 0.067 | 0.237 |
| 128 | **0.166** | 0.161 | 0.252 |
| 64 | **0.249** | 0.248 | 0.118 |

**The spread grows with the depth of the cut, and it grows faster than the
mean falls.** At 512 the sixteen corpora span 3.8 points around a mean of 0.980,
so a published average is a usable guide. At 128 they span **16.6** points
around a mean of 0.820 — a corpus at the top of that range keeps 0.90 and one at
the bottom keeps 0.73, and no average distinguishes them. At 64 the spread is
0.249 on a mean of 0.632.

So the honest answer is conditional, and it cuts both ways:

- **A shallow cut needs no measurement.** If you are truncating a
  768-dimensional model to 512 or storing int8, the published averages are
  right and this flag tells you what you already knew. That is a result against
  the tool and it is the first thing this page should say.
- **A deep cut needs one.** Below half width the corpora separate faster than
  the mean moves, and `docs/golden-findings.md` section 7's warning is the
  reason: these sixteen are technical Q&A and BEIR, and a vault or a codebase is
  neither.
- **The binary column needs one everywhere.** Its spread is 0.118–0.252 at every
  depth, including full width, where every other column is flat.

---

## Against the published numbers

Yousefiramandi and Cooney,
[*Benchmarking Patent Embeddings*](https://arxiv.org/abs/2605.24297)
(arXiv:2605.24297), section 4.11, is the closest published measurement of the
same operation — they truncate and L2-renormalise, which is exactly what
`--truncate` does. Their Table 18, five models on a patent corpus:

| | 512 | 256 | 128 | 64 |
|---|---|---|---|---|
| their mean retention | 95.9% | 91.3% | 83.2% | 71.6% |
| their full width | 2,048 – 4,096 | | | |
| `bge-base` here | **98.0%** | **92.1%** | **82.0%** | **63.2%** |
| its full width | 768 | | | |

Read by **absolute dimension** the two agree closely, and it would be
comfortable to stop there. That reading is wrong, and their own paper says why.

**512 of 4,096 and 512 of 768 are not the same operation.** Theirs is a cut to
an eighth; ours is a cut to two thirds. Compared by the fraction of width that
survives, the published models are far more resilient than these: their 1/8 cut
costs 4 points, and a 1/6 cut here costs 18.

**And that is their finding, extended rather than contradicted.** Section 4.11
reports that "models with higher original dimensions (4096) generally show
better truncation resilience than those with lower dimensions (2048–2560),
suggesting that the information in high-dimensional embeddings is more evenly
distributed across dimensions." Every model here is 384 or 768 — below the
bottom of that range — and lands where the trend predicts.

So the published averages are correct and do not transfer. Not because the
corpora differ, which is the objection this page began with, but because the
**models** differ in the one property that governs the operation. A reader
holding a 768-dimensional index and a table measured on 4,096-dimensional ones
should read the fraction, not the number.

---

## What the precision axis is, and is not

**Simulated.** rebasis produces float32; what a store does with it is the
store's business, and the backends do not all do the same thing —
[`sqlite-vec`'s `int8`](guides/sqlite-vec.md),
[pgvector's `halfvec`](guides/pgvector.md) and Qdrant's `datatype` are three
different narrowings, measured separately in each guide. A cell on this axis
measures what the *arithmetic* costs, which is a lower bound on what a
particular codec costs.

The dimension axis is not simulated. Truncating a vector is the whole operation.

**On one axis the simulation was checked against a real store and matched
exactly.** pgvector's `halfvec` is IEEE-754 binary16 and so is numpy's
`float16`, and a round trip through a real `halfvec(32)` column returns
bit-for-bit what `quantize(v, "float16")` produces —
`tests/integration/test_pgvector_types.py`, with the `vector` column as the
control. So the `float16` column of this grid is not an approximation of what a
`halfvec` index costs; it *is* that number.

**On the `int8` axis it could not be, and the reason is why the label stays.**
No backend here stores int8 the way this grid simulates it. `sqlite-vec`'s
`vec_quantize_int8` takes a caller-supplied range; the grid scales by each
vector's own largest magnitude. Two different quantizers with one name, and no
measurement can make them the same. Read the `int8` column as what per-vector
scalar quantization costs, and your store's guide for what your store's codec
costs.

The three narrowings, precisely:

| | what it does |
|---|---|
| `float16` | a dtype round trip; about three decimal digits survive |
| `int8` | symmetric **per-vector** scalar quantization — scale by the vector's own largest magnitude, round to one of 255 levels, scale back. Per vector because that is what backends that do this do, and because a per-corpus scale would let one outlier document coarsen every other one |
| `binary` | the sign of each component and nothing else, scored as ±1. That orders documents identically to Hamming distance over the packed bits, since the inner product of two sign vectors is `d − 2·hamming` |

---

## Writing it back is not this tool's job

Going from `vector(1024)` to `vector(256)`, or from `vector` to `halfvec`, means
recreating the column. That is DDL, and `migrate` changes vectors rather than
schemas — the line that keeps rebasis from becoming a vector database. Most
stores are `dimension_locked` anyway, so a narrower index is a new collection,
which is a reindex rather than a migration.

This says what the change is worth. Performing it stays yours.

---

## What this does not establish

- **Four models, one of them Matryoshka-trained.** One MRL model is enough to
  show that the training bought nothing here and not enough to say it never
  does: `mxbai-embed-large-v1` is one architecture at one width, and a model
  whose MRL objective was weighted differently might behave differently.
- **English, technical Q&A and BEIR.** `docs/golden-findings.md` section 7's
  warning applies word for word: scifact is scientific abstracts, and a band
  measured here is not a band for an Obsidian vault. That is the whole argument
  for the flag existing.
- **The rescored column assumes the rescore is free.** It is, in the sense that
  no embedding is needed — the full-precision vectors are the ones the index
  already holds. It is not free in storage: keeping them means keeping both
  representations, which is a different arrangement from a cheaper index and the
  grid does not price it.
- **Nothing was written back.** Every number here is a search over arrays, not a
  round trip through a store's own codec. One axis was checked against a real
  one and matched exactly (`float16` against pgvector `halfvec`, above); the
  others were not, and `int8` provably cannot be. Where the two disagree, the
  store's own guide is the authority — `test_quantized_roundtrip.py` and
  `test_pgvector_types.py` are those measurements.

## Reproducing

```bash
uv run --extra sentence-transformers --with ir-datasets --with ranx \
    --with model2vec python tools/truncation_band.py \
    --corpora heldout --corpora beir \
    --model BAAI/bge-base-en-v1.5 --model BAAI/bge-small-en-v1.5 \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --cache-dir ~/band-cache --out reports/band/truncation.jsonl

uv run python tools/truncation_band.py --summarise reports/band/truncation.jsonl
```
