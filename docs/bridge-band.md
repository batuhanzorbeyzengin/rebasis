# When bridging is worth it — the measured band

The design asks one question: can you change an index's embedding model without
rebuilding the index? The mechanism works. Whether it is *worth using* has a
narrower answer than expected, and this document is the measurement behind it.

**Setup.** 29 runs over eight corpora, with a further 33 runs over eleven
corpora held out in section 8 to check the rule against evidence it was not
shaped by. Scored with
[ranx](https://github.com/AmenRa/ranx) rather than with rebasis' own metric code
— grading a tool with its own scorer tests consistency, not correctness. The
adapter is produced by the shipped `rebasis fit` CLI and applied through the
documented `Bridge` API, so what is measured is the tool a user would run.

The headline numbers come from the corpora searched with **queries real people
typed**: five collections of StackExchange posts and financial questions,
222,680 documents and 5,761 real user questions with human judgements. Those are
the closest public analogue to what the design targets — a personal or technical
knowledge base searched by its owner. A second set of BEIR runs (scifact,
nfcorpus, arguana) is reported alongside; its documents and judgements are real
but its "queries" are constructed claims and article titles, and the two sets
disagree in a way that matters.

Four configurations per run, all against the **same** index:

| | |
|---|---|
| `status quo` | old model query → old index (what you have today) |
| `naive swap` | new model query → old index (just change the model) |
| **`bridged`** | adapter(new query) → old index (what rebasis promises) |
| `full reindex` | new model query → new index (the ceiling) |

---

## 1. The mechanism works

Dropping a new model's vectors into an old index destroys retrieval. On the
real-query corpora, nDCG@10:

| corpus | status quo | naive swap | bridged |
|---|---|---|---|
| programmers | 0.390 | **0.052** | 0.318 |
| unix | 0.413 | **0.038** | 0.317 |
| english | 0.478 | **0.077** | 0.388 |
| gaming | 0.569 | **0.136** | 0.489 |
| fiqa | 0.369 | **0.037** | 0.298 |

A naive swap loses 80–91% of retrieval quality. Bridging recovers **6.2x** what
the naive swap gives. Where the dimension changes — 384 to 768, which a Chroma
collection's dimension lock makes impossible to migrate in place — the naive
swap cannot happen at all, and bridging is the only path short of rebuilding.

That is the premise, confirmed.

---

## 2. The break-even, and why it is one number

Bridging beats doing nothing when

```
ARR x upgrade_gain > 1
```

— the adapter's retention times how much better the new model is on this corpus.
Across **29 corpus/model pairs its sign predicted the outcome 29 times out of
29**, against the independent nDCG measurement: 15 of 15 on the real-query
corpora and 14 of 14 on BEIR. A further 33 runs on eleven corpora the rule had
never seen scored 32 of 33 — [section 8](#8-held-out-33-runs-on-eleven-corpora-the-rule-never-saw).

Two numbers are easy to confuse here and they answer different questions. The
break-even *computed from measured nDCG* is exact by construction and agrees 15
of 15 — which is to say it cannot disagree, and
[section 9](#9-what-the-counting-is-worth) is where that is taken seriously. What
`probe` reports is an **estimate** of it, from recall against a sampled ground
truth rather than from human judgements, and that estimate agrees
14 of 15 — [section 4](#4-the-bands-were-vetoing-real-wins) scores the estimate,
which is the thing a user actually sees.

Neither factor answers the question alone, which is the whole reason `probe`
now reports the product as one figure. A retention of 0.94 loses to a 3% upgrade;
a 46% upgrade wins with a retention of 0.74.

| real-query run | gain | retention | product | measured |
|---|---|---|---|---|
| unix, MiniLM→bge-small | 0.94 | 0.82 | 0.77 | −23.2% |
| english, MiniLM→bge-base | 1.02 | 0.74 | 0.75 | −24.6% |
| fiqa, MiniLM→bge-base | 1.10 | 0.72 | 0.79 | −20.9% |
| gaming, potion→bge-base | 1.52 | 0.71 | 1.07 | **+7.4%** |
| fiqa, potion→bge-base | 2.44 | 0.48 | 1.16 | **+16.0%** |
| unix, potion→bge-base | 2.10 | 0.47 | 0.99 | −0.7% |

That last row is the formula earning its place: the largest gain but one, and it
still lost, because retention fell further than the gain rose.

---

## 3. The squeeze: gain and retention pull against each other

This is the finding that decides how the tool should be positioned.

**Correlation between upgrade gain and adapter retention: −0.958.**

A bigger upgrade needs more retention to clear the break-even, and delivers
less of it. The reason is not mysterious: a large gain means the old model was
weak, and a weak source space carries less recoverable structure — there is less
in it for the adapter to map.

| upgrade gain | bridging wins | mean retention |
|---|---|---|
| < 1.20 | **0/6** | 0.83 |
| 1.20 – 1.35 | 1/2 | 0.71 |
| > 1.35 | 3/6 | 0.66 |

So the band is real but narrow, and it is not "bigger upgrade, better case".
It is: **a large upgrade whose old model was still good enough to map from.**
`potion→bge-base` sits in it; `potion→bge-small` does not, despite a comparable
gain, because mapping 256 dimensions into 384 retained far less than mapping
them into 768.

### What this means for a user

- **A routine model refresh** — a 5–15% benchmark improvement — will not clear
  the bar. Reindex, or stay where you are. Six of six such runs lost ground.
- **A generational jump** *can* clear it, and rebasis tells you which case you
  are in before you spend the compute.
- **A dimension change** is the unambiguous case: bridging held 97% of the
  status quo where no alternative except a full rebuild exists.

---

## 4. The bands were vetoing real wins

The decision rule placed ARR in a band and recommended from that. The break-even
was reported alongside. Where they disagreed, the band won — and it was wrong
every time.

Agreement with the independent nDCG outcome, over the 15 real-query runs:

| criterion | agrees |
|---|---|
| ARR bands alone | 10/15 |
| `bridge_advantage > 1` | **14/15** |

Every disagreement went the same way. A large upgrade bridged imperfectly lands
at a **low** ARR — 0.47 to 0.73 in the four runs where bridging genuinely helped
— which the bands read as `caution` or `full_reindex`. The bands rejected all
four, including a run where bridging measured **+16.0%**.

The rule now lets the break-even decide *whether* to bridge and the bands decide
*which* bridging answer to give — `bridge_sufficient` when little of a reindex is
left on the table, `bridge_and_migrate` when a lot is. With that change:

| | before | after |
|---|---|---|
| T1 agrees with reality | 10/15 | **13/15** |
| genuine wins recommended | 0/4 | **4/4** |

The two remaining misses are a −0.7% run — a tie — and one whose break-even sat
inside the noise band, where the tool says so rather than choosing.

## 5. Without queries, the tool cannot answer at all

On the BEIR runs, where bridging lost ground in six of six, a run **with** a real
query log said so six times out of six. A run **without** one said
`bridge_and_migrate` six times out of six.

That is why a run with no upgrade estimate is now marked **provisional**: it
reports how well an adapter bridges, which is real, and declines to say whether
bridging is worth doing, which it cannot know.

Synthesised queries (`--synth-queries`) close part of the gap, and only with the
right strategy:

| tier | agrees (BEIR, 14 runs) | task difficulty |
|---|---|---|
| T0, no queries | 6/14 | — |
| T2 `lead` | 8/14 | trivial: oracle recall 0.98 |
| T2 `title` | 10/14 | trivial: oracle recall 0.99 |
| T2 `keywords` | **12/14** | real: oracle recall 0.79–0.97 |
| T1, real queries | 9/14 | — |

`lead` and `title` hand the retriever the answer — a lead sentence is a literal
substring of its own document — so both models find it every time and the
estimate separates nothing. rebasis detects that and marks the run provisional
rather than reporting the meaningless 1.00x it produces. `keywords` builds a task
neither model solves for free, and it is the only strategy worth using.

---

## 6. Recall is not enough

One run disagreed between rebasis and ranx, and the disagreement was exact:

| scifact, MiniLM→bge-base | status quo | bridged |
|---|---|---|
| recall@10 | 0.7833 | **0.7867** |
| nDCG@10 | **0.6451** | 0.6294 |

The adapter retrieved the same documents and ordered them worse. rebasis decided
on recall and read that as an improvement.

Whether that matters depends on what consumes the results — a RAG pipeline that
hands ten chunks to a model re-ranks them implicitly, a ranked list shown to a
person does not. nDCG@10 is now computed in the core path and reported alongside
recall for that reason.

---

## 7. The measurement was checked against published numbers

Before concluding anything about the tool, the harness was checked against
independently published reproductions of the same models on the same datasets.
If the absolute numbers were wrong, every conclusion drawn from them would be.

nDCG@10, measured here against published:

| dataset | model | published | measured here |
|---|---|---|---|
| SciFact | all-MiniLM-L6-v2 | 0.645 | **0.645** |
| SciFact | bge-small-en-v1.5 | 0.713 | **0.713** |
| FiQA2018 | all-MiniLM-L6-v2 | 0.369 | **0.369** |
| FiQA2018 | bge-small-en-v1.5 | 0.403 | **0.404** |
| NFCorpus | all-MiniLM-L6-v2 | 0.314 | **0.316** |
| NFCorpus | bge-small-en-v1.5 | 0.349 | **0.344** |
| CQADupStack programmers | bge-base-en-v1.5 | 0.4238 | **0.4242** |

The last row is against
[Anserini's own reproduction](https://github.com/castorini/anserini/blob/master/docs/reproduce/from-document-collection/beir-v1.0.0-cqadupstack-programmers.bge-base-en-v1.5.parquet.flat.onnx.md)
of that exact dataset and model, and lands inside the ±0.001 tolerance that
document states for itself.

Two things follow. The pipeline is not producing numbers of its own invention.
And where it reports something surprising — bge-small scoring at or below
all-MiniLM-L6-v2 on the StackExchange corpora — that is a property of those
corpora rather than a bias in the harness, because the same harness reproduces
the case where bge-small wins (FiQA, 0.404 against 0.369) exactly.

**One number did not check out, and the cause was in this harness.** ArguAna
first measured 0.436 for bge-small against a figure quoted as 0.331. Neither was
right. ArguAna is evaluated with **self-removal** — a query is itself an argument
that also appears in the corpus, and the standard evaluation excludes a query's
own document from its results. Anserini's reproduction uses `-removeQuery` and
reports 0.6375 for bge-base; this harness, with the same convention, measures
**0.6406**.

The three ArguAna runs were re-measured with self-removal:

| pair | status quo | bridged | reindex | product | outcome |
|---|---|---|---|---|---|
| MiniLM→bge-small | 0.506 | 0.485 | 0.607 | 0.96 | −4.0% |
| potion→bge-base | 0.422 | 0.435 | 0.641 | 1.03 | **+3.0%** |
| potion→bge-small | 0.422 | 0.386 | 0.607 | 0.91 | −8.6% |

The break-even predicts all three, as it did on the wrong numbers. Nothing in
this document changed: 29 of 29 across the corrected set, retention mean 0.722,
mean loss against the status quo −8.4%. Worth recording that a measurement error
of this size left every conclusion standing — the formula is a ratio, and a
systematic error in one dataset largely cancels in it.

## 8. Held out: 33 runs on eleven corpora the rule never saw

Everything above is the evidence the decision rule was *shaped by*: the bands
moved twice in response to it. A rule fitted to its own evidence is not a
finding, so it was re-run on eleven corpora and 10,346 real user questions that
took no part in any of those changes — the eight StackExchange collections not
used before, plus the three that were, on a fixed three-rung model ladder
(potion → MiniLM-L6 → bge-small → bge-base).

Nothing was tuned between the two runs. The rule shipped as it stood.

> **Read this table against [section 9](#9-what-the-counting-is-worth) before
> quoting it.** Two things have since been established about it. The `32 / 33`
> is an algebraic identity rather than a prediction — read off a run's own
> scores, `ARR × upgrade_gain` reduces to the same inequality as the outcome it
> is scored against, so it cannot disagree. And the rest of the row does not
> reproduce from the files still on disk: `reports/` is gitignored and the
> original artefacts were overwritten by a later run, so what can be recovered
> is the 33 corpora (uniquely determined by the 10,346 question count) but not
> the numbers. Recomputed on those rows, retention is 0.703 against the 0.714
> below, the range is 0.403–0.984 against 0.423–0.988, and "worth bridging"
> comes out 0 rather than 4. The naive-swap figure and the anti-correlation do
> reproduce. **The table is left as it was published rather than quietly
> restated**, because a number that cannot be re-derived should be visible as
> one.

| | this run | the runs above |
|---|---|---|
| break-even predicted the outcome | **32 / 33** | 29 / 29 |
| retention, mean | 0.714 | 0.722 |
| retention, range | 0.423 – 0.988 | 0.47 – 0.87 |
| naive swap retains | 0.125 | 0.145 |
| corr(gain, retention) | **−0.940** | −0.958 |
| worth bridging | 4 / 33 | 8 / 29 |

Cumulatively the break-even has now called **61 of 62** outcomes —
[a count worth less than it looks](#9-what-the-counting-is-worth), and section 9
is what it is worth. The three findings that matter — the anti-correlation, the
retention band and the catastrophe of a naive swap — reproduced on corpora chosen
after they were written down, and survive the same test.

### The one miss is where the rule says it is uncertain

*Subject to the caveat above: this sub-section is the part of section 8 that
reproduces least. On the recoverable rows none of these four crosses the
break-even at all, and none of the four measured outcomes is distinguishable
from zero under a paired test — Holm-adjusted p of 1.00, 1.00, 1.00 and 0.45.
The reasoning about the borderline band still holds; the four numbers below
should be read as the record of what was published rather than as a measurement
anyone can now re-derive.*

Four of the 33 crossed the break-even. Three of those four won; one lost by
0.8%.

| run | advantage | measured |
|---|---|---|
| mathematica, potion→MiniLM | 1.063 | **+3.5%** |
| fiqa, potion→MiniLM | 1.035 | **+0.7%** |
| android, potion→MiniLM | 1.035 | **+0.0%** |
| gis, potion→MiniLM | 1.036 | −0.8% |

All four sit between 1.035 and 1.063 — inside, or barely outside, the ±0.025
band the rule already reports as borderline. The rule is not wrong there so much
as honest: at that distance from the break-even it is a coin flip, and it says
so. Away from the threshold it was right 29 times out of 29.

### Every win came from replacing a bad model

All four crossings are on the same rung, potion → MiniLM-L6 — the weakest old
model on the ladder. On the two upper rungs, where the existing index was
already decent, bridging did not pay once in 22 runs:

| rung | mean gain | mean retention | mean vs. doing nothing | agreed |
|---|---|---|---|---|
| potion → MiniLM-L6 | 1.83 | 0.49 | 0.93 | 10 / 11 |
| MiniLM-L6 → bge-small | 0.95 | 0.84 | 0.81 | 11 / 11 |
| bge-small → bge-base | 1.06 | 0.81 | 0.87 | 11 / 11 |

The middle row is the clearest statement of the squeeze in section 3: the adapter
retains 84% of the ceiling and still loses 19%, because on these corpora
bge-small is not an upgrade on MiniLM-L6 at all (mean gain 0.95). The tool says
`no_upgrade_needed` on all eleven, which is the right answer and not one the
bands could have reached.

Across all 33 runs the tool recommended acting on 4, sitting still on 10
(`no_upgrade_needed`) and reindexing on 19.

---

## 9. What the counting is worth

Every score above is a count, and a count with no null behind it is not evidence.
`tools/band_stats.py` supplies the nulls. It runs over the **57 distinct runs
`reports/band/` still holds** — the 48 of section 10 (which contain every row of
the held-out set) plus the nine MMTEB runs of section 11. That is not the 62: the
29 runs of sections 1–7 were measured on a different model ladder and their rows
are not in the repository, so nothing below re-tests them.

**The headline count is an identity.** `bridge_advantage` is `ARR x upgrade_gain`,
and read off one run's own scores that is `(bridged / reindex) x (reindex /
status quo)` — which is `bridged / status quo`, the same inequality as the outcome
it is being scored against. [Section 2](#2-the-break-even-and-why-it-is-one-number)
says so of the 15 real-query runs; it is true of every count in this document
computed the same way, including the 48 of 48 in section 10. They agreed 57 times
out of 57 here because they could not do otherwise.
[ADR 4](adr/0004-arr-ratio-interval.md) is the precedent: a degenerate case made
the wrong code look right, and the fix was to check on data where the two sides
can differ.

**Against the null that matters, what survives is thin.** Bridging beat doing
nothing in **3 of those 57** runs, so a rule that ignored every input and always
answered "do not bridge" scores **54 of 57**. That is the baseline a decision rule
has to beat, not a coin. Even taking 57 of 57 at face value it clears the coin at
p ≈ 7e-18 and clears the majority class at **p = 0.046** — and the 0.947 it is
tested against was estimated from the same outcomes, which flatters it further.

**What a user actually sees does worse than the baseline.** The break-even as
`probe` reports it — retention from the fit's own kNN ground truth, gain from
recall rather than from the graded metric — agrees with the nDCG outcome in
**37 of 57**, 95% Clopper–Pearson 0.51–0.77. Clopper–Pearson because near a
proportion of 1 the normal approximation has no coverage left, and because a
document that widened its own borderline band rather than claim precision it
lacked should take the interval that cannot under-cover; Wilson is printed beside
it and says the same thing.

**A proportion is the wrong summary, and the better one is positive.** With 95% of
outcomes on one side, accuracy cannot separate a real rule from a constant. The
rule thresholds a continuous quantity, so score the quantity: across the 57 runs
the estimate ranks them by the margin they actually returned at
**Spearman ρ = +0.60, p ≈ 1e-6**. The estimate carries real information about how
much bridging will cost or buy. What it does not support is being read as a
threshold at the accuracy the counts imply.

**Paired, per query, corrected.** The harness now writes per-query scores beside
each row, so the nDCG differences can be tested rather than eyeballed. A paired
Fisher randomisation test of `bridged` against `status quo` on the same queries,
10,000 permutations, seed 20260825, over the 48 runs: **40 differ from zero at
raw p < 0.05, and 35 survive Holm** across the table. Every one of the 35 is a
*loss*. Not one run in this evidence shows bridging beating the status quo by an
amount distinguishable from zero — the single positive run, scifact on
potion→MiniLM, comes in at +0.16% with p = 0.96. Holm rather than Bonferroni:
same assumption, uniformly more power, and it still costs five of the forty.

**The four near-misses do not reproduce, and would not survive if they did.** On
the rows now in `reports/band/`, the four crossings
[section 8](#the-one-miss-is-where-the-rule-says-it-is-uncertain) lists measure
−2.8%, −3.1%, −5.1% and −9.6% rather than the +3.5% to −0.8% quoted there — that
table was written from an earlier artifact which the files no longer hold, and
until it is re-measured it should be read as unverified. On the rows that are
here, none of the four is distinguishable from zero: raw p of 0.73, 0.41, 0.38
and 0.056, and 1.00, 1.00, 1.00 and 0.45 after Holm. Either way that subsection
stands, and for a stronger reason than it gives: near the break-even the rule is
not merely a coin flip, it is being asked to resolve a difference the measurement
cannot resolve at all.

**What this section does not license.** It does not overturn the direction of any
finding above — the losses are the part that tests strongly, and the squeeze, the
naive-swap catastrophe and the "reindex or stay put" conclusion are all
*strengthened* by it. It says the **counting** was never evidence for the rule,
and that the strongest defensible claim is the rank correlation rather than any
proportion. It says nothing at all about the 29 runs of sections 1–7, whose rows
are not in the repository to test. And a per-corpus p-value tests one corpus
against itself on its own queries: 35 significant losses are 35 statements about
35 measured corpus/model pairs, and none about a pair nobody ran.

---

## 10. The assumption every number here makes

Everything above measures the bridge producing the **final ranking**, inside the
old index. That is what `Bridge` does today and it is the arrangement the
decision rule was calibrated for, so it is the right thing to have measured.

It is not the only arrangement. If the bridge produces a *candidate set* which
the new model reorders in its own space, the only thing that can be lost is a
relevant document that never reached the top N — a weaker requirement than
ranking it in the top 10, and one the same adapters meet far more often.
Measured on the same ladder over 48 runs, on the twelve CQADupStack forums plus
FiQA, SciFact, NFCorpus and ArguAna:

| | beat keeping the current model |
|---|---|
| single stage | **1 / 48** |
| two stage, candidates at 200 | **36 / 48** |
| two stage, of the 37 runs where a reindex actually helps | **36 / 37** |

Nothing in this document changes. That harness reproduces every figure here —
retention at nDCG@10 0.717 against 0.714–0.722, the gain/retention
anti-correlation −0.933 against −0.958 and −0.940, the naive swap at 0.151
against 0.125–0.145, and the break-even's sign right 48 times out of 48.
The squeeze is exactly where it was. What changes is which
quantity bounds the arrangement, and that is a different question from how good
the adapter is. [The cascade measurement](cascade-band.md).

## 11. What this does not establish

- **Small models.** Everything measured is 256–768 dimensional. A 2021-era model
  to a 2025-era one may sit further right on the gain axis than anything here.
- **Academic corpora**, and this one now has a number on it. BEIR is real text
  with real judgements, but its queries are constructed claims and article
  titles rather than what someone types into their own notes. Re-run on MMTEB's
  hard-negative HotpotQA and FEVER plus TREC-COVID, the gain/retention
  anti-correlation measured **−0.454** against the −0.933 the same harness gets
  on these corpora — the squeeze is a property of this corpus family more than
  it is a law. The break-even itself held, 9 of 9, on thresholds it was never
  fitted to ([the runs](cascade-band.md#5-a-different-regime-hard-negatives)).
- **One adapter family.** Retention is a property of the adapters `auto` fits.
  A better adapter moves the whole band, and that is where the headroom is.
- **English only.**

## Reproducing

```bash
uv run --extra sentence-transformers --with ir-datasets --with model2vec \
    python tools/make_golden.py --out tests/golden/data
uv run pytest tests/golden -m slow -q
```

Section 9's numbers come from the rows themselves and cost a second of CPU:

```bash
python tools/band_stats.py reports/band/cascade.jsonl reports/band/mmteb.jsonl
```

The paired tests need the per-query sidecars the harness writes beside each row.
Where a row carries no `per_query` field the tool names it and skips it, rather
than reporting a table with a hole in it.
