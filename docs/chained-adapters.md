# Chaining adapters, and what it costs

The roadmap has carried this since the first release: *v1 → v2 → v3 without a
full refit at each step. Error accumulation across a chain has not been measured,
and refitting against the original is probably more accurate — which is worth
knowing rather than assuming.* This is the measurement. The guess was right, and
the interesting part is a case where it briefly looked wrong.

**When a chain is even a choice.** Fitting `v3 → v1` directly needs matched
pairs, and the pairs are always there: the old vectors are in the index and the
new ones come from one embedding pass over a sample. A direct fit is never
*unavailable*. Chaining buys that embedding pass back, and only if a `v3 → v2`
adapter already exists that somebody else paid for — so a chain has to be nearly
free of error to be worth taking.

**204 cells.** Seventeen corpora, six spans of a four-model ladder, both
directions, two adapter families. Every arm scored against what a full reindex to
the newest model returns, over the corpus' own real queries — direct and chained
share that ground truth exactly, so the difference between them is the chain and
nothing else.

Two things in the table are there to be checked rather than read. A one-link
"chain" *is* the direct fit, and costs 0.0000 in all 51 cells. The reindex
ceiling is 1.0000 everywhere, by construction.

---

## 1. It costs, and the cost compounds

`procrustes_centered`, the default family:

| links | cells | query cost | document cost | chain wins (query) |
|---|---:|---:|---:|---:|
| 1 | 51 | +0.0000 | +0.0000 | — |
| 2 | 34 | −0.0088 | −0.0100 | 16 / 34 |
| 3 | 17 | −0.0142 | −0.0262 | 2 / 17 |

At two links it is close to a coin flip and the median cost is under a hundredth.
At three it is not: the chain loses in 15 of 17 cells on the query side and in
**all 17** on the document side, at a median 9% of the direct fit's retention.

## 2. The case that looked like a win was a centring artefact

One two-link span came out **ahead** of its direct fit: potion → MiniLM →
bge-small, at +0.0114 on the query side, 1.060 of the direct fit. A chain beating
the thing it approximates wants an explanation before it gets a recommendation.

The candidate: `procrustes_centered` subtracts a mean before it rotates, so a
two-link chain of it performs *two* centrings. That is a strictly richer function
than the single centred rotation a direct fit produces, and the gain would then
belong to the extra centring rather than to chaining.

That is testable. Plain `procrustes` has no centring step, so a chain of it is
one rotation exactly like the direct fit — the composition of two orthogonal
matrices is an orthogonal matrix. Re-running the whole grid with it:

| links | cells | cost | chain wins |
|---|---:|---:|---:|
| 1 | 51 | +0.0000 | — |
| 2 | 34 | −0.0164 | **0 / 34** |
| 3 | 17 | −0.0362 | 0 / 17 |

The chain never wins, and the span that was +0.0114 is −0.0119. **The win was
the centring, not the chain.**

A second thing falls out of that run and is worth stating because it is a check
rather than a finding: under plain `procrustes` the query cost and the document
cost are *identical to four decimals* in every cell. For an orthogonal map
`A(q)·A(d) = q·d`, so mapping the queries and mapping the documents are the same
measurement — and the harness agreeing with the arithmetic is how you know it is
measuring what it says.

## 3. What to do instead

Refit against the original. The pairs are in your index, the cost is one
embedding pass over a sample of a few thousand documents, and
[fit quality saturates near 4,000 pairs](bridge-band.md) — so the pass is small
and the result is better than any chain measured here.

Nothing ships for this. There is no `rebasis adapter chain`, and adding one would
be offering a worse option for a saving that does not exist: the embedding pass a
chain avoids is the same pass `rebasis fit` already runs, and this page is what
it would cost to avoid it.

## What this does not establish

- **One ladder, four models, English.** The spans are contiguous rungs of the
  same ladder every other measurement here uses. A chain across models that
  differ more than these do is unmeasured, and section 1 says the cost grows
  with the number of links rather than with the distance covered — those are
  different variables and this design confounds them.
- **Two families.** `procrustes_centered` and `procrustes`. Whether a chain of
  low-rank affine or residual-MLP links behaves like either is unmeasured; the
  centring result says only that composition can add capacity, not how much any
  particular family adds.
- **One fit budget and one seed.** 4,000 pairs, seed 0, no repeats. The
  per-cell differences are small enough at two links that a second seed could
  move the 16-of-34 count; the three-link result is not close enough for that to
  matter.
- **Nothing about serving cost.** A chain is *k* matmuls on the hot path instead
  of one. [ADR 11](adr/0011-the-hot-path-budget-is-per-dimension.md)'s budget applies
  and is not measured here.

## Reproducing

```bash
PYTHONPATH=src python spikes/chained_adapters.py \
    --corpus heldout --corpus beir --corpus beir/trec-covid \
    --out reports/chain/rows.jsonl --device cuda

# The discriminating run: no centring, so a chain is one rotation like the direct fit
PYTHONPATH=src python spikes/chained_adapters.py \
    --corpus heldout --corpus beir --corpus beir/trec-covid \
    --method procrustes --out reports/chain/plain.jsonl --device cuda
```

Every link comes from the same `fit_candidates` call `rebasis fit` makes and is
applied through the same `BaseAdapter.apply` the engine uses, with a normalise
between links because that is what the serving path does.
