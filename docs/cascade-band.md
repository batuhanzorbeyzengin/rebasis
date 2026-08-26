# The bridge as a recall stage

[The measured band](bridge-band.md) says bridging is worth doing in about one run
in five, and [ADR 10](adr/0010-retention-is-bounded-by-the-source.md) says
retention cannot be fitted higher: a single global map into the old space cannot
carry more than the old space holds.

Both hold. This document is about the assumption underneath them, which is that
the **final ranking** is produced in the old space. It does not have to be.

```
1. q_new  = f_new(query)
2. q_old  = bridge.to_index_space(q_new)
3. cand   = index.search(q_old, k=N)        ← the bridge, as a recall stage
4. v_new  = f_new(text of cand)             ← re-embed N documents
5. result = top-10 by cos(q_new, v_new)     ← ranked in the NEW space
```

Step 5 is the new model scoring its own vectors, which is exactly the ranking a
full reindex would produce over those same documents. So the only thing the
bridge can lose is a relevant document that failed to reach the candidate set —
and what bounds the whole arrangement is the bridge's **recall@N**, not its
nDCG@10.

**48 runs. Single-stage bridging beat keeping the current model in 1.
A two-stage arrangement beat it in 36.**

---

## The measurement

Sixteen corpora — the twelve CQADupStack forums, FiQA, SciFact, NFCorpus and
ArguAna — on the same three-rung ladder the held-out runs used (potion-base-8M →
all-MiniLM-L6 → bge-small → bge-base). Five configurations against the same
index, scored with `ranx`, everything reported at **nDCG@10** because that is
what a RAG pipeline consumes:

| | |
|---|---|
| `status quo` | old model query → old index |
| `naive swap` | new model query → old index |
| `bridged` | adapter(new query) → old index, ranked there |
| `cascade@N` | bridged top-N, reranked by the new model in its own space |
| `full reindex` | new model query → new index |

The adapter comes from the same `probe_store` → `save_adapter` path the `rebasis
fit` CLI runs and is applied through the documented `Bridge` API, round-tripped
through a real `.rbs` file. The rerank is measured, not estimated: the candidate
sets are actually reordered by the new model's own similarity. ArguAna is scored
with self-removal, the convention its standard evaluation uses.

**The harness reproduces the existing evidence.** Retention at nDCG@10 measures
0.717 here against the 0.714–0.722 `bridge-band.md` reports; the gain/retention
anti-correlation −0.933 against −0.958 and −0.940; the naive swap retains 0.151
against 0.125–0.145; and the break-even's sign predicted the outcome **48 times
out of 48**. Nothing about the band moved.

---

## 1. The result is structural

| rung | cascade@200 beat keeping the current model |
|---|---|
| potion-base-8M → all-MiniLM-L6 | **16 / 16** |
| all-MiniLM-L6 → bge-small | 4 / 16 |
| bge-small → bge-base | **16 / 16** |
| **total** | **36 / 48** |

The middle rung is not a counter-example. It is the finding `bridge-band.md`
section 8 already recorded: on the CQADupStack corpora **bge-small is not an
upgrade on all-MiniLM-L6 at all**, and `probe` says `no_upgrade_needed` on every
one of them. The four wins on that rung are exactly the four corpora where it
*is* an upgrade — SciFact, NFCorpus, ArguAna and FiQA.

Split by whether a full reindex actually beats doing nothing:

| | upgrade is real (37 runs) | no upgrade (11 runs) |
|---|---|---|
| single-stage `bridged` | **1 / 37** | 0 / 11 |
| `cascade@200` | **36 / 37** | 0 / 11 |

That is the whole finding in one table. **Where the upgrade is real, the
two-stage arrangement delivers it 36 times out of 37; the single-stage one
delivers it once.** Where there is no upgrade, neither delivers anything, which
is correct — there is nothing to deliver, and the tool already says so.

The single miss is `mathematica`, all-MiniLM-L6 → bge-small: a 1.4% upgrade,
which cascade turned into −0.8%. At that size the answer is a coin flip and the
decision rule reports it as borderline.

## 2. It delivers most of a reindex

| rung | cascade@200 as a share of a full reindex | vs. doing nothing |
|---|---|---|
| potion → MiniLM-L6 | 0.77 – 0.99 | +18.7% to +112.1% |
| bge-small → bge-base | **0.98 – 1.00** | +0.4% to +14.8% |

On the upper rung the two-stage arrangement is within two percent of rebuilding
the index in every one of the sixteen runs — 0.507 against 0.507 on `android`,
0.642 against 0.643 on `arguana`. On the bottom rung, where the old space is a
256-dimensional static model and recall@200 retention runs 0.54–0.99, it still
delivers three quarters to nearly all of a reindex.

The largest results, all against the status quo:

| run | single stage | cascade@200 | reindex |
|---|---|---|---|
| mathematica, potion→MiniLM | −2.5% | **+112.1%** | +142% |
| fiqa, potion→MiniLM | −4.8% | **+101.0%** | +122% |
| stats, potion→MiniLM | −12.8% | **+87.6%** | +106% |
| wordpress, potion→MiniLM | −3.6% | **+80.4%** | +98% |
| unix, potion→MiniLM | −14.9% | **+76.4%** | +105% |
| tex, potion→MiniLM | −32.3% | **+76.2%** | +129% |
| arguana, MiniLM→bge-small | −3.8% | **+20.3%** | +20% |

That last row is the shape of the whole thing: a full reindex would give +20%,
single-stage bridging gives −3.8%, and the two-stage arrangement gives +20.3% —
the entire upgrade, from an index nobody rebuilt.

## 3. This does not contradict ADR 10

ADR 10 says retention is bounded by the source space and cannot be fitted higher.
Every number here is consistent with that, and the reproduction figures above say
so quantitatively: retention at nDCG@10 is where it always was, and so is the
squeeze.

Nothing about the adapter improved. What changed is **which quantity the
arrangement is bounded by**:

| cut-off | retention | beat doing nothing |
|---|---|---|
| nDCG@10 | 0.717 | 1 / 48 |
| recall@10 | 0.754 | 3 / 48 |
| recall@100 | 0.865 | 15 / 48 |
| recall@200 | **0.893** | 16 / 48 |

Putting a relevant document somewhere in the top 200 is a weaker requirement than
ranking it in the top 10, and a two-stage arrangement only needs the weaker one.
(The right-hand column measures single-stage bridging at each depth, which is a
different and less favourable question than the 36/48 above — that compares the
*reranked* result at k=10 against the status quo at k=10.)

ADR 10's closing sentence stands unchanged: you cannot recover from 256
dimensions what 768 dimensions encode. You can, it turns out, recover enough of
it to build a candidate set.

---

## 4. What `probe` reports

`probe` measures retention at candidate depth alongside the retention it already
reported, and computes the same break-even from it. On the three golden
scenarios — real embeddings of BEIR/scifact, the fixtures the decision rule is
regression-tested against:

| scenario | decision | ARR | `bridge_advantage` | `cascade_arr` | `cascade_advantage` |
|---|---|---|---|---|---|
| large_upgrade | `bridge_and_migrate` | 0.824 | 1.087 | 0.926 | **1.222** |
| different_family | `full_reindex` | 0.903 | **0.957** | 0.955 | **1.012** |
| hard_drift | `no_upgrade_needed` | 0.971 | 0.820 | 0.985 | 0.832 |

The middle row is the one worth reading twice. Single-stage bridging lands at
0.957 — below the break-even, so the tool says `full_reindex`, and that is the
right advice *for what the tool can do*. At candidate depth the same adapter
sits at 1.012, which is inside the ±0.025 noise band rather than a confident
win, but it is on the other side of the line. A user told to rebuild their index
may have an option that does not require it.

The bottom row is the check that this does not simply flatter every run:
`hard_drift` has an upgrade gain below 1 — the new model is worse on that corpus
— and both figures stay below the break-even, because there is nothing to
deliver and neither arrangement invents any.

**It is reported and not acted on**, and the reason is narrower than it was.
`rebasis.serve.Cascade` serves this arrangement, so the objection is no longer
that the tool cannot do it. It is that the decision rule weighs quality against
cost, and this arrangement's cost turns on how often a candidate is already
cached — a property of a query log, not of the corpus a probe reads.
`bridge_advantage` costs a matrix multiply whatever the traffic. So the rule
runs on the number it can price and reports the other for a reader who can
price it. That widening the search leaves ARR, its interval, nDCG, MRR and the
decision bit-for-bit identical is itself a test.

---

## 5. A different regime: hard negatives

Everything above is technical Q&A and BEIR. The three tasks
[Maystre et al.](https://arxiv.org/abs/2510.13406) report their cross-model grid
on are a different kind of collection, and they are the ones whose published
numbers this can be read against: **HotpotQA** and **FEVER** in MMTEB's
hard-negative form — every test query paired with its correct documents plus the
top 250 negatives a strong retriever surfaced — and **TREC-COVID**. Nine more
runs, same ladder, same code.

| | CQADupStack + BEIR (48) | MMTEB hard negatives (9) |
|---|---|---|
| retention, nDCG@10 | 0.717 | **0.619** |
| retention, recall@200 | 0.893 | **0.812** |
| corr(gain, retention) | **−0.933** | **−0.454** |
| single-stage beat doing nothing | 1 / 48 | 2 / 9 |
| cascade@200 beat doing nothing | 36 / 48 | 6 / 9 |
| break-even predicted the outcome | 48 / 48 | **9 / 9** |

Three things in that table, in order of how much they matter.

**The squeeze is weaker here, and that is the finding.** −0.933 against −0.454
on the same adapters, the same ladder and the same harness. On the technical
corpora retention is almost entirely a function of how good the old model was;
on these it is not. What separates them is the task — multi-hop over paired
paragraphs, claim verification, short topic-dense queries — and
[`bridge-band.md`](bridge-band.md) already listed "academic corpora" as
something its band did not establish. This is what that limit looks like
measured: a variable the break-even does not model, showing up as an
anti-correlation that halves.

**The rule survives the regime it was not calibrated on.** 9 of 9, from a rule
whose thresholds were fitted on StackExchange. That is the strongest evidence in
either document that the break-even is a real relationship rather than a curve
fitted to one corpus family.

**Retention is lower, and hard negatives are why.** A hard negative is a document
a strong retriever already thought was a good answer — the corpus is built out of
exactly the distinctions a bridge finds hardest to preserve. Recall@200 retention
runs 0.67–0.96 here against 0.92–1.00 on the technical corpora, and the
arrangement is worth correspondingly less.

### What the paper says, and what these runs do and do not reproduce

An earlier version of this section said the paper reports alignment collapsing
on two of the three tasks. **That was backwards.** What the paper's prose says,
verbatim, is the opposite count:

> Without alignment, cross-model retrieval fails almost completely. After
> alignment, retrieval becomes feasible across models, and **in two of the three
> tasks, upgrading to a stronger query model can yield substantial performance
> gains.**

Two things follow, and the second is the reason the sentence was worth
re-reading rather than paraphrasing.

**The collapse without alignment is reproduced, and strongly.** That is the
paper's other claim in the same breath, and it is the one these runs are placed
to check: a naive swap retains 0.125 of a reindex across the 62 runs in
[`bridge-band.md`](bridge-band.md). "Fails almost completely" is what 0.125
looks like.

**The gains are a claim about a grid, and this ladder is not that grid.** The
paper varies seven query models against seven document models — 49 pairs per
task — and its claim is that gaining pairs *exist* in two of the three. Three
rungs of one ladder cannot confirm or contradict that, and these three runs sit
on the losing side of it:

| corpus | status quo | bridged | cascade@200 |
|---|---|---|---|
| HotpotQA-HN, potion→MiniLM | 0.390 | **0.147** (−62%) | 0.382 |
| FEVER-HN, MiniLM→bge-small | 0.548 | 0.470 (−14%) | **0.812** (+48%) |
| trec-covid, potion→MiniLM | 0.457 | **0.460** (+0.7%) | 0.481 (+5%) |

On HotpotQA a bridged query loses nearly two thirds of what the user already
had. On TREC-COVID it loses nothing. Those are this harness' numbers and they
stand; what does not stand is reading them as an independent confirmation of the
paper's task-level result, because the paper's prose never names **which** two
tasks gained — that lives in its Figure 4, which was not read. The earlier
version of this section attributed the gaining task to TREC-COVID. Nothing in
the text supports that attribution, and it has been withdrawn.

Worth recording that the two overlap more than the correction implies: MiniLM
and bge-small are both in the paper's seven, so `MiniLM→bge-small` is a cell of
its grid as well as a rung of this ladder. Reading one against the other would
need the per-cell values, which are in the figure rather than the prose.

### The cascade break-even is conservative

`gain × recall@N retention` predicted 7 of these 9. Both misses are in the same
direction and on the same corpus: TREC-COVID, where the formula said 0.883 and
0.868 — below the break-even — and the measured results were **+5.1%** and
**+3.5%**.

That is the safe direction, and it is not luck. The formula assumes every
document the bridge failed to recall would have ranked as well as the ones it
did. It will not have: what a bridge loses first is what it ranked worst, and a
document that was going to place tenth costs less than the arithmetic charges
for it. So the figure reads as a **floor** on what a two-stage arrangement is
worth, not as an estimate of it — which is the right way round for a number a
user might act on.

---

## What this costs, and what has not been measured

**Step 4 is on the hot path.** Embedding N documents per query is not free, and
[ADR 11](adr/0011-the-hot-path-budget-is-per-dimension.md)'s budget — tens of
microseconds for the mapping — does not describe this at all. At an A10G's
measured rate for bge-base, 100 documents is about 0.2 s; on the CPU a laptop
has, it is seconds. **A two-stage arrangement is not usable without a cache**,
and the cache is what makes it a lazy migration rather than a permanent tax: the
documents people actually retrieve get embedded once and stay embedded.

`rebasis.serve.Cascade` is that arrangement, with the cache as part of its
construction rather than an option — in memory by default, on disk under the
`.rebasis/cache/` directory `gc` already collects. Its `stats` splits a query
into bridge, search and rerank, breaks the embedder out of the third, and
reports the hit rate and the documents embedded.

**The measurement it still needs cannot be taken here.** How a cache behaves
depends on a query distribution, and a query distribution is a property of a
running system rather than of a corpus. So the numbers above say what the
arrangement is worth in *quality*, `Cascade.stats` is the instrument for what it
costs on your traffic, and `probe` reports what your adapter would retain that
way **without recommending it** — the decision rule still runs on the
single-stage break-even, because that is the arrangement whose cost is known.

**Other limits:**

- **Nineteen corpora, English, three families** — technical Q&A, BEIR, and
  MMTEB's hard-negative variants. The last of those moved the anti-correlation
  by half, which is a warning about how much a fourth family might move
  something else.
- **One ladder.** Three model pairs, 256 → 384 → 384 → 768. A 2021-era model to a
  2025-era one may sit further right on the gain axis than anything here.
- **The rerank is a bi-encoder**, the model the user is upgrading to — not a
  cross-encoder bolted on. Published work finds that scoring progressively more
  documents with an off-the-shelf reranker helps a strong first stage at first
  and then *degrades* it past a point — Jacob et al., [Drowning in Documents:
  Consequences of Scaling Reranker Inference](https://arxiv.org/abs/2411.11767)
  (arXiv:2411.11767, ReNeuIR 2025 at SIGIR). That failure mode does not apply
  here: the second stage is the new model ranking in its own space, which is
  what a full reindex would have done anyway. It is the reason this was measured
  rather than assumed.
- **N = 100 and 200 only.** The two differ by a point or two at most, which is
  the first evidence that the curve flattens early — but where it flattens has
  not been measured.

## Reproducing

```bash
uv run --extra sentence-transformers --with ir-datasets --with ranx \
    --with model2vec --with datasets python tools/bridge_band.py \
    --corpus heldout --corpus beir --ladder default \
    --k 10,100,200 --cascade 100,200 --out reports/band/cascade.jsonl

# The MMTEB tasks in section 5 come from Hugging Face rather than ir_datasets,
# which is what `--with datasets` is for.
uv run ... python tools/bridge_band.py --corpus mmteb --ladder default \
    --k 10,100,200 --cascade 100,200 --out reports/band/mmteb.jsonl

uv run python tools/bridge_band_report.py reports/band/cascade.jsonl --view cascade
uv run python tools/bridge_band_report.py reports/band/cascade.jsonl --view summary
```
