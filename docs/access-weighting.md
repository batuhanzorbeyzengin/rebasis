# Weighting a probe by what people actually read

`probe` draws a sample of your index, holds part of it out as query proxies, and
reports ARR over those. Every record is equally likely to become a query. Most
indexes are not read that way: a small set of documents answers most of the
questions, and retention on *those* is the number that decides whether an upgrade
hurts.

`rebasis probe --access-log` weights the draw. This is what it changes, what it
costs, and the thing the roadmap entry was blocked on — whether the confidence
interval survives a non-uniform draw.

**36 cells.** Six corpora from 5,183 to 57,600 documents, three ladder rungs, two
access ratios, and **120 replicate probes per design per cell** — 12,960 probe
runs. Weights come from the corpora's own judgements: a document some real query
was judged relevant to is a document somebody reads. That is a proxy and it is
measured rather than invented; *how much* more often is not measurable without a
real log, so the hot-to-cold ratio is swept at 10× and 100×.

---

## 1. The entry named one place for the weights, and there are two

A `probe` sample does two jobs at once, and this is the whole of the design
question:

- It is the **mini-index** every measurement runs against.
- It is the **pool** the query proxies are split out of.

Weighting the sample — which is what handing weights to `draw_sample` does —
fills the mini-index with frequently-read documents. That changes the
**distractors**, which is a property of the index rather than of the questions
asked of it. Weighting only the **split** leaves the mini-index a fair miniature
and changes only what is asked.

The second is what "describe the queries that matter" means, and it is where the
weights go.

## 2. Sampling already moves the number more than weighting does

Before anything about weighting: a 4,000-document mini-index is an easier place
to retrieve in than the 23,000-document corpus it was drawn from. Every design
therefore sits **above** the whole-corpus quantity it stands for.

| design | gap to the whole corpus |
|---|---:|
| `uniform` — the status quo | **+0.048** |
| `weighted_queries` — weights on the split | +0.025 |
| `weighted_sample` — weights on the sample | +0.051 |

Read the rest of this page against that row. The sampling gap is larger than
anything weighting does, it is present in today's default, and attributing it to
an access log would be attributing a pre-existing property of `probe` to a flag.

It is also the one measured discriminator between the two placements: weighting
the split leaves the estimate about **half as far** from the whole-corpus
quantity as weighting the sample does.

## 3. Weighting changes the answer, which is the point

Paired within each cell, `weighted_queries` against `uniform`:

| access ratio | median shift | max | cells moving > 0.01 |
|---|---:|---:|---:|
| 10× | +0.0093 | +0.0447 | 8 / 18 |
| 100× | **+0.0152** | **+0.0729** | 12 / 18 |

The sign is consistent: a frequently-read document is one the index already
handles well, so retention measured on the questions people send is **higher**
than retention measured on a uniform draw. At 100× the shift exceeds 0.01 — the
margin `RefitPolicy` treats as the edge of noise — in two thirds of cells, and
reaches 0.073.

So the two are different quantities, and a report has to say which one it is
reporting. It does: `probe --json` carries `access_weighted`, and both report
formats say so in prose.

## 4. The interval survives it

This is what the entry was blocked on. The test needs no estimand: divide the
**median interval half-width** by the **spread of the estimates across
replicates**.

**The target is 1.96, not 1.** A correctly calibrated 95% interval around a
roughly normal estimator is exactly ±1.96 standard deviations wide. Read against
1, a correct interval looks twice too wide — which is how a calibration check
becomes a false alarm.

| design | ci/2 ÷ sd | ÷ 1.96 | coverage | cells under 0.90 |
|---|---:|---:|---:|---:|
| `uniform` | 1.92 | 0.98 | 0.94 | 2 / 36 |
| `weighted_queries` | 1.84 | **0.94** | 0.94 | 6 / 36 |
| `weighted_sample` | 1.93 | 0.98 | 0.95 | 1 / 36 |

The bootstrap resamples a run's queries and never resamples the run, and it turns
out that is close to right: the plain design's interval is within 2% of the
correct width. Under weighted queries it is about **6% narrow** — a real effect,
in the direction the entry worried about, and small. Median coverage is
unchanged; what moves is the tail, from 2 cells under 0.90 to 6.

**Coverage here is slightly optimistic and that is not corrected for.** Each
design's expectation is estimated from the same 120 replicates its coverage is
measured on, so the target carries about a tenth of the spread as its own error.
The ratio column does not have that problem, which is why it is the one to read.

## What this means for using it

Use `--access-log` when you have one and when the question is "will this upgrade
hurt what people actually read". Compare the result against another weighted run,
never against an unweighted one — they estimate different things and the weighted
one is usually the higher.

The interval is 6% optimistic under weighting. Against decision bands 0.10 wide
that is not a reason to withhold the flag; it is a reason for the report to say
the run was weighted, which it does.

## What this does not establish

- **The weights are a proxy.** Judged-relevant stands in for read, and the
  ratio is swept because no real access log was available to measure it from.
  A real log has a heavy tail rather than two levels, and nothing here says how
  a smoother distribution behaves.
- **Six corpora, English, one adapter family.** `procrustes_centered`
  throughout, for the reason [`migration-band.md`](migration-band.md) gives.
  nfcorpus is absent: at 3,633 documents it cannot spare a 4,000-document
  sample, which is itself worth knowing.
- **T0 only.** Query proxies are documents. With a real query log
  (`--queries`) the queries are not drawn from the sample at all, so the
  weighting has nothing to act on — `--access-log` and `--queries` answer the
  same question by different means, and passing both weights nothing.
- **One sample size.** 4,000 documents and 500 queries per replicate. Section 2's
  gap is a function of the ratio between sample and corpus, and only one value of
  it was run.
- **No per-cell significance.** The ratios and coverages above are medians over
  36 cells. A claim about one corpus would need the per-replicate scores, which
  the harness does not write.

## Reproducing

```bash
PYTHONPATH=src python spikes/access_weighted.py \
    --corpus beir --corpus beir/cqadupstack/android \
    --corpus beir/cqadupstack/english --corpus beir/cqadupstack/gaming \
    --corpus beir/fiqa/test \
    --out reports/access/rows.jsonl --device cuda
```

Every replicate goes through the same `fit_candidates` call `rebasis fit` makes
and the same `bootstrap_ci` `probe` reports, so what is measured is the tool
rather than a reimplementation of it.
