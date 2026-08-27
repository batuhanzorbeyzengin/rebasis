# Merging two embedding spaces

A migration that stops short leaves a collection holding two embedding spaces,
and no single query is correct against both.
[`MixedSpaceSearch`](reference/api.md) is what serves it: two searches, each
filtered to the half it is right about, merged by `calibrated_merge`. That merge
has two modes — by calibrated score when the `.rbs` carries an isotonic
`ScoreCalibrator`, by **reciprocal rank fusion** when it does not. The fallback is
defended on solid ground: M0 measured a median KS distance of **0.924** between
the two spaces' score distributions, so comparing raw scores would let one side
win for reasons unrelated to relevance. What has never been measured is what it
costs when it is the thing answering the query.

**The two merges are not interchangeable, and they fail in opposite
directions.** RRF returns *half* its results from each side at every stage of
every migration measured — a fixed interleave carrying no information about how
far the job has got. The calibrated merge tracks the migration but
under-weights the new space. Which failure is cheaper depends on how good the
migrated half actually is, and neither merge can see that.

Measured against human judgements at seven points along a real `migrate` job, one
model pair, `k=10`:

- **Four corpora, twenty mid-migration cells, migrated records holding the new
  model's own vectors.** The calibrated merge beat RRF in **17**. In **4** RRF was
  worse than ignoring the mixture entirely — worse than the silent-failure case
  the mechanism exists to beat — and all four were at 10% or 25% migrated. The
  calibrated merge was worse than ignoring it in none of the twenty.
- **Two corpora, ten cells, migrated records holding what `rebasis migrate`
  writes when it is given a correctly directed adapter** — an adapter's image of
  the old vectors rather than the new model's. **The result reverses: RRF wins 6 of 10.** The calibrated merge
  gives the migrated half between 0.3% and 24% of the result and starves it,
  because the calibrator maps old scores onto the *new model's* distribution and
  those records are not in it.

So "which merge is right" depends on a property of the migration
`calibrated_merge` decides without: it branches on whether a calibrator exists,
and that says nothing about whether the migrated vectors are in the space the
calibrator was fitted against. Two smaller results are unambiguous: **RRF is exact
at the two endpoints and the calibrated merge is not** (section 3), and **the
`MAX_OVER_FETCH` ceiling cost nothing at all** (section 5).

---

## What is measured

nDCG@10 and recall@10 against human judgements, at seven points along a real
`migrate` job, for five configurations against one index:

| | |
|---|---|
| `status quo` | old model query → old index — the floor a user already has |
| `full reindex` | new model query → the fully migrated index — the ceiling |
| `bridged only` | `bridge.to_index_space(q)` → the half-migrated index, as if it were not mixed |
| `mixed, RRF` | `MixedSpaceSearch` with no calibrator |
| `mixed, calibrated` | `MixedSpaceSearch` with the isotonic calibrator from the `.rbs` |

The third is the silent-failure case: what a user gets by running `--limit` and
continuing to query, and the number the mechanism exists to beat.

**The migration is a real one.** `MigrationEngine` fills the queue, writes the
shadow copy, reads back and verifies every batch, and records what moved in the
manifest — and `MixedSpaceSearch` reads what moved **from that manifest**, which is
where it reads it in production; `serve/mixed.py` is explicit that the store is
deliberately not asked. Progress is advanced by `engine.run(limit=...)`, which is
`--limit`. The adapter, the `.rbs` round trip and the `Bridge` come from
`tools/bridge_band.py`'s own `probe_store` → `save_adapter` → `Bridge.load` path,
imported rather than reimplemented, so the adapter under test is the adapter
[the band](bridge-band.md) reports; `auto` selected `procrustes_centered+csls` on
all four corpora, at ARR@10 0.795–0.928. Scoring is `ranx`, for the reason that
harness gives: grading a tool with its own metric code tests consistency, not
correctness. The two merges are handed **byte-identical adapter weights** — the
uncalibrated bridge is the calibrated one saved again without its
`calibration.json` — so the only thing differing between them is the merge.

**One substitution, and it is in the adapter rather than the pipeline.** What a
migrated record holds is the axis this page turns on, so both possibilities were
measured; the queue, the shadow copy, the read-back and the manifest are shipped
code in each.

`--migrated-vectors reembed` (sections 1–5) hands the engine a lookup returning
the new model's own vector per record. The engine's docstring names that path —
"map them with the adapter, *or re-embed with the new model*" — but
`_process_batch_inner` only ever calls `adapter.apply`, so there is nothing in the
package to drive, and a lookup is how re-embedding is expressed to an engine that
only knows how to apply a transform. It is what makes the ends mean something: at
0% the index is exactly the old one, at 100% exactly a full reindex. The lookup
keys on the source vector's bytes, so records sharing one share a new vector too —
exact on three corpora, wrong for **480 of TREC-COVID's 171,331 records** (0.28%).
That is a bound on the error, not a correction of it.

`--migrated-vectors adapter` (section 6) is a real `procrustes_centered` fitted
old-space → new-space and applied by the engine: the transform `rebasis migrate`
runs.

### The two models have to share a dimension, which rules out most of the ladder

A migration rewrites vectors inside one collection; every backend rebasis supports
locks that collection's dimension, and `MixedSpaceSearch` sends a raw new-model
query at that same collection. So a mixed index only exists when `d_old == d_new`.
Of the seven model pairs in `tools/bridge_band.py`'s ladders exactly one
qualifies: **all-MiniLM-L6-v2 → bge-small-en-v1.5**, both 384. That is the pair
measured here, and it is the only one that could be. Four corpora from the band's
own set, seeded at 0:

| corpus | documents | judged queries | status quo | full reindex |
|---|---|---|---|---|
| NFCorpus | 3,633 | 323 | 0.316 | 0.344 |
| SciFact | 5,183 | 300 | 0.645 | 0.713 |
| CQADupStack-android | 22,998 | 699 | 0.538 | **0.476** |
| TREC-COVID | 171,331 | 50 | 0.472 | **0.758** |

Android is the control: bge-small is not an upgrade on the CQADupStack forums,
which is [`cascade-band.md`](cascade-band.md) section 1's finding and `probe`'s own
verdict, and a full reindex there is 11% *worse* than doing nothing. TREC-COVID is
the opposite extreme at 1.6x. The two in between are where that document records
the rung as a real upgrade, at 1.09x and 1.11x.

---

## 1. Reciprocal rank fusion interleaves; it does not merge

The share of the returned top ten that came from the migrated half:

| migrated | RRF | calibrated (range over the four corpora) |
|---|---|---|
| 0% | 0.000 | 0.000 |
| 10% | **0.500** | 0.037 – 0.144 |
| 25% | **0.500** | 0.094 – 0.316 |
| 50% | **0.500** | 0.211 – 0.522 |
| 75% | **0.500** | 0.390 – 0.660 |
| 90% | **0.500** | 0.623 – 0.794 |
| 100% | 1.000 | 1.000 |

The RRF column is not a rounded average. It is **exactly 0.500 in nineteen of the
twenty** intermediate cells, and it has to be: both result sets are renumbered
from rank 0 before merging, so each side's rank-*r* document scores exactly
`1/(61 + r)`; the two sides are disjoint during a migration, so nothing ever
accumulates; and the top ten is therefore ranks 0–4 of each. **The output
composition of an RRF merge carries no information about how far the migration has
got.** At 10% migrated it hands half the top ten to a tenth of the corpus.

The twentieth cell — NFCorpus at 75% migrated — reads 0.5003, which over 323
queries at ten hits each is one hit: a single query on which one side had fewer
than five of its own records to give and the other filled the slot. Every result
was still ten long (section 5).

The calibrated column tracks the migration instead, and sits below the corpus
share at every stage. That residue is the part of the score shift the calibrator
does not remove: [`adapters.md`](concepts/adapters.md) records that isotonic
calibration takes the distribution shift from 0.92 to 0.09 rather than to zero,
and 0.09 is enough to give the bridged side a standing edge.

## 2. What the interleave costs when the migrated half is the new model's space

nDCG@10, every corpus, every stage:

| corpus | merge | 0% | 10% | 25% | 50% | 75% | 90% | 100% |
|---|---|---|---|---|---|---|---|---|
| NFCorpus | bridged only | 0.284 | 0.281 | 0.266 | 0.223 | 0.126 | 0.033 | 0.131 |
| | mixed, RRF | 0.284 | **0.188** | **0.216** | 0.253 | 0.280 | 0.257 | 0.344 |
| | mixed, calibrated | 0.282 | 0.286 | 0.286 | 0.292 | 0.298 | 0.309 | 0.344 |
| SciFact | bridged only | 0.614 | 0.513 | 0.374 | 0.283 | 0.194 | 0.086 | 0.210 |
| | mixed, RRF | 0.614 | **0.471** | 0.559 | 0.615 | 0.643 | 0.667 | 0.713 |
| | mixed, calibrated | 0.615 | 0.617 | 0.626 | 0.640 | 0.663 | 0.691 | 0.713 |
| android | bridged only | 0.468 | 0.417 | 0.338 | 0.220 | 0.147 | 0.081 | 0.165 |
| | mixed, RRF | 0.468 | **0.341** | 0.395 | 0.422 | 0.434 | 0.439 | 0.476 |
| | mixed, calibrated | 0.469 | 0.460 | 0.458 | 0.454 | 0.461 | 0.472 | 0.476 |
| TREC-COVID | bridged only | 0.531 | 0.522 | 0.511 | 0.504 | 0.469 | 0.411 | 0.196 |
| | mixed, RRF | 0.531 | 0.626 | 0.651 | 0.666 | 0.663 | 0.642 | 0.758 |
| | mixed, calibrated | 0.530 | 0.567 | 0.591 | 0.657 | 0.680 | 0.717 | 0.757 |

Bold marks the four cells where RRF scored **below** simply bridging and ignoring
the mixture. All four are at 10% or 25% migrated, which is where the interleave is
most wrong: half the top ten drawn from a corpus half holding a tenth of the
documents. Mean advantage of the calibrated merge over RRF, by stage:

| migrated | 10% | 25% | 50% | 75% | 90% |
|---|---|---|---|---|---|
| mean gap | **+0.076** | +0.035 | +0.022 | +0.020 | +0.046 |
| range | −0.060 to +0.147 | −0.060 to +0.071 | −0.009 to +0.040 | +0.017 to +0.027 | +0.024 to +0.075 |

The gap is largest where the two halves are most unequal in size, which is what
the mechanism predicts. **Every negative in that range row is TREC-COVID** — the
three cells RRF won, at 10%, 25% and 50% — and TREC-COVID is also the one corpus
where a full reindex is worth 1.6x the status quo. Forcing five slots from the new
space is a standing bet that both halves are equally worth reading, and it pays
when the new space is much better. That is a reading of four corpora, not a rule.

**Recall@10 and nDCG@10 disagree about RRF, and the disagreement is the finding
restated.** On SciFact at 10% migrated RRF scored recall@10 **0.739** against
`bridged only`'s 0.634 while its nDCG@10 was *lower*, 0.471 against 0.513, and its
MRR@10 lower still, 0.391 against 0.481. Android at 10% has the same shape: recall
0.537 against 0.520, nDCG 0.341 against 0.417. Guaranteeing five slots to the
new-space side **does** surface relevant migrated documents the bridge would have
missed — and then places them by rank parity rather than by how good they are. RRF
finds them and ranks them badly.

## 3. Neither merge reduces to the single-space answer, and one of them has to

At 0% and 100% the index holds one space. There is a single right answer — what the
store returns for that one query — and a merge that does not reproduce it is wrong
rather than merely worse. The fraction of queries on which each returned exactly
the single-space ranking, ids and order:

| corpus | 0%: RRF | 0%: calibrated | 100%: RRF | 100%: calibrated |
|---|---|---|---|---|
| NFCorpus | 1.000 | **0.152** | 1.000 | 0.929 |
| SciFact | 1.000 | **0.093** | 1.000 | 1.000 |
| android | 1.000 | **0.157** | 1.000 | 1.000 |
| TREC-COVID | 1.000 | **0.040** | 1.000 | 0.900 |

**RRF is exact at both ends, and that is not luck.** Its scores are a function of
rank alone, one side is empty, and `1/(61 + r)` is strictly decreasing — so no two
documents can tie and the input order survives the sort.

**The calibrated merge is not**, and the mechanism is visible in the same run.
`ScoreCalibrator` is isotonic regression: pool-adjacent-violators produces a step
function with far fewer levels than it has inputs, and `out_of_bounds="clip"`
flattens both tails. Of the ten scores the bridge returned per query at 0%
migrated, the calibrator mapped them onto **5.0 to 7.2 distinct values** on
average — the calibrators themselves carry 168 to 195 knots. `calibrated_merge`
then sorts on `(-score, id)`, and every level shared by two documents hands the
choice between them to whichever id sorts first. On 84% to 96% of queries that is
enough to change the top ten.

The tie-break itself is right, and `serve/hybrid.py` argues for it correctly: a tie
resolved by which side was passed first is a standing bias toward one embedding
space. What the argument does not cover is a tie *within* one side, where there is
no space to be neutral between and document id order is simply arbitrary.
[`adapters.md`](concepts/adapters.md) says isotonic regression is monotone "so
ranking is preserved exactly", which is true of `Bridge.calibrate_scores` applied
to one array and **not** true once those scores go through a sort with a
tie-break.

The cost in nDCG is small — between −0.002 and +0.002 at 0% migrated on all four
corpora — because documents sharing a calibrated level had similar scores to begin
with. The 100% column is the same thing without the calibrator involved at all:
the old side is empty, and what collides is raw float32 similarity between
duplicate documents. TREC-COVID holds 1,453 records whose MiniLM vector is
byte-identical to an earlier record's; NFCorpus holds 40; SciFact holds none, and
scores 1.000.

## 4. What the mechanism is worth against a full reindex

Each configuration as a share of the ceiling its own corpus reaches, so the four
can be read on one scale. Range over the four:

| migrated | bridged only | mixed, RRF | mixed, calibrated |
|---|---|---|---|
| 10% | 0.69 – 0.88 | 0.55 – 0.83 | **0.75 – 0.97** |
| 25% | 0.52 – 0.77 | 0.63 – 0.86 | **0.78 – 0.96** |
| 50% | 0.40 – 0.67 | 0.73 – 0.89 | **0.85 – 0.95** |
| 75% | 0.27 – 0.62 | 0.81 – 0.91 | **0.87 – 0.97** |
| 90% | 0.09 – 0.54 | 0.75 – 0.94 | **0.90 – 0.99** |

The `bridged only` row is the silent failure, measured: a bridged query against a
90%-migrated index returns **0.09 to 0.54** of what a reindexed one would,
and nothing raises. Against the other thing a user could have done — not migrating
at all — the calibrated merge ends the window above the status quo on SciFact
(1.07x at 90% migrated) and TREC-COVID (1.52x), and below it on NFCorpus (0.98x)
and android (0.88x). Those two are where a full reindex is worth +8.8% and −11.5%,
so there was little or nothing to deliver: **the migration window is not free on a
corpus with a small upgrade**, which the decision rule says before the migration
starts.

**This is a different quantity from the claim already in the README.** That claim —
a hit rate above 0.90 restored at every stage — comes from
`tests/unit/test_mixed_search.py`: a 400-document synthetic corpus at 32
dimensions, two spaces related by an exact rotation, every document its own best
answer. On that corpus each side's rank-0 hit is the right one, so a strict
interleave always includes it and the property section 1 measures is invisible. A
correct mechanism test, and not a retrieval quality measurement; the two should
not be read as one claim.

## 5. The over-fetch ceiling binds at both extremes and cost nothing measurable

`MixedSpaceSearch` asks each side for `k / share` and caps it at
`MAX_OVER_FETCH = 8` per side — the cap engages whenever a side holds less than an
eighth of the corpus. Measured per query as retrieved over returned:

| migrated | over-fetch | old side depth | new side depth | queries returning < 10 |
|---|---|---|---|---|
| 0% | 1.10 | 11 | *skipped* | 0 |
| 10% | **9.20** | 12 | 80 *(capped from 101)* | 0 |
| 25% | 5.40 – 5.50 | 14 | 41 | 0 |
| 50% | 4.10 – 4.20 | 20 | 21 | 0 |
| 75% | 5.40 – 5.50 | 41 | 14 | 0 |
| 90% | **9.20** | 80 *(capped from 101)* | 12 | 0 |
| 100% | 1.10 | *skipped* | 11 | 0 |

The class's docstring says the result at the ceiling is "short rather than slow".
On these corpora it was neither: **not one query in any configuration, at any
stage, on any corpus, returned fewer than ten results.** The cap was reached at
10% and 90% migrated and the merge still filled the result, because the capped
side searches the whole collection rather than its own half — a depth-80 search of
a corpus that is 10% migrated still surfaces more than the five migrated documents
the merge needs.

**And it cost no quality either.** Re-run on NFCorpus and SciFact with
`MAX_OVER_FETCH` raised to 32, every nDCG@10, recall@10 and MRR@10 in every cell
of both corpora is **identical to four decimal places**. Only the bill changed:
over-fetch at 10% and 90% rose from 9.20 to 11.30, and the three middle stages,
where the cap was never reached, did not move at all. On this evidence the ceiling
is free — it removes 19% of the retrieval at the two extremes and nothing else.
That is two corpora and one `k`, and it is the opposite of what the docstring
warns about, which is worth knowing either way.

## 6. Under an adapter migration rather than a re-embedding, the result reverses

Everything above has the migrated records holding the new model's own vectors.
No migration produces that: `_process_batch_inner` applies an adapter, so what it
writes is `A(old_vector)` — an adapter's image of the stored vector, not the new
model's own. Re-run on NFCorpus and SciFact with a real `procrustes_centered`
fitted old-space → new-space and applied by the engine:

> **A note on what `migrate` does today.** This section was measured with a
> genuine forward (`old_to_new`) adapter, because that is the direction a
> migration needs. It was measured *before* it was established that `rebasis
> fit` produces only the reverse direction and that `migrate` had never checked
> — it now refuses, and there is currently no adapter it can be run with
> ([the migration guide](guides/migration.md) has the reasoning). That does not
> weaken anything below: this is what a *correctly directed* migration writes,
> so it is the case that matters whenever the forward direction exists. It does
> mean the mixed-space state it describes cannot presently be reached through
> the CLI.

| corpus | merge | 0% | 10% | 25% | 50% | 75% | 90% | 100% |
|---|---|---|---|---|---|---|---|---|
| NFCorpus | bridged only | 0.284 | 0.281 | 0.267 | 0.224 | 0.126 | 0.033 | 0.102 |
| | mixed, RRF | 0.284 | 0.186 | 0.208 | 0.237 | **0.249** | **0.217** | 0.290 |
| | mixed, calibrated | 0.282 | 0.282 | 0.274 | 0.249 | 0.183 | 0.156 | 0.291 |
| SciFact | bridged only | 0.614 | 0.513 | 0.374 | 0.285 | 0.196 | 0.091 | 0.158 |
| | mixed, RRF | 0.614 | 0.462 | **0.537** | **0.554** | **0.559** | **0.576** | 0.627 |
| | mixed, calibrated | 0.615 | 0.526 | 0.406 | 0.337 | 0.295 | 0.291 | 0.627 |

RRF wins 6 of the 10 mid-migration cells, by margins up to 0.285. The mechanism is
in the composition: the calibrated merge gives the migrated half **0.003 to
0.239** of the result — against 0.037–0.144 at 10% and 0.623–0.794 at 90% when the
migrated records were the real thing. At 90% migrated on SciFact it draws 88% of
its top ten from the 10% of the corpus that has *not* moved. That is what the
calibrator does when the assumption under it stops holding: it was fitted to map
bridged old-space scores onto the **new model's** distribution, and an
adapter-mapped document is not in that distribution — it scores systematically
lower against a raw new-model query. The calibrated old side then wins nearly
every slot, and RRF's refusal to look at scores at all becomes the safer
behaviour.

**A second finding falls out of the same run, about `migrate` rather than about
fusion.** The ceiling itself moves: a *completed* adapter migration reaches 0.290
on NFCorpus and 0.627 on SciFact, against 0.344 and 0.713 for a real reindex — 84%
and 88% of it. On NFCorpus that is **below the 0.316 the status quo was already
delivering**, so on that corpus, with this pair, a completed `migrate` leaves
retrieval worse than never having migrated. That is consistent with
[ADR 10](adr/0010-retention-is-bounded-by-the-source.md): a single global map
cannot carry more than the source space holds, and applying it to documents is
bounded the same way as applying it to queries. Two corpora is not a band and it
is not what this page set out to measure — but it is measured, and it bears on how
`migrate`'s value should be described.

---

## What this does not establish

- **One model pair.** all-MiniLM-L6-v2 → bge-small-en-v1.5, the only pair in the
  repository's ladders whose two models share a dimension — and a mixed index
  cannot exist without that. Everything here is one point on the gain axis
  `bridge-band.md` runs along, and that document needed 62 runs before it would
  state a band. Four corpora and twenty cells are not a band.
- **One adapter family.** `auto` selected `procrustes_centered+csls` every time. A
  different adapter has a different score distribution and therefore a different
  calibrator, and the calibrator is half of what sections 3 and 6 measure.
- **The controls are two corpora, and one corpus was never run.** Sections 5 and 6
  and the queue-order check ran on NFCorpus and SciFact only; android and
  TREC-COVID were not re-run under a raised ceiling, an adapter migration or a
  shuffled queue. **FiQA was planned and not measured** — the fourth corpus
  `cascade-band.md` records this rung as a real upgrade on, and the obvious next
  row. The command is below.
- **TREC-COVID is not a small exception.** It is the only corpus here where the
  upgrade is large, the only one where RRF wins anything in section 2, and it
  carries 50 judged queries. Whether the rule is "the interleave pays when the new
  space is much better" or "TREC-COVID is odd" needs corpora this run does not
  have.
- **An exact index.** `MemoryStore` scans, so nothing here is confounded by a graph
  index degrading under in-place updates — [`index-health.md`](index-health.md)
  measures that separately. On an approximate index the over-fetch in section 5 is
  a second cost on top of the recall loss that document reports, and the two have
  not been measured together.
- **Queue order was checked once and changed nothing qualitatively.**
  `JobQueue.next_batch` orders by `priority DESC, record_id ASC`, so which records
  move first is a property of how the ids sort. Re-run with seeded random
  priorities — the column `--priority access` uses — the calibrated merge beat RRF
  in 10 of 10 mid-migration cells on NFCorpus and SciFact, and RRF fell below
  `bridged only` in 2. One alternative ordering is not a distribution over them.
- **No significance test, and nothing about latency.** The differences in section 2
  are large and consistent in direction; none is accompanied by a paired
  randomisation test, and the spike writes per-configuration aggregates rather than
  the per-query arrays one would need. Section 5 reports how much was retrieved,
  not what it cost in time — `MixedSpaceSearch` reports depth rather than duration
  for exactly that reason.
- **The calibrated path through `MixedSpaceSearch` has no test.** Every case in
  `tests/unit/test_mixed_search.py` builds its `Bridge` without a calibrator, so
  the whole suite exercises the RRF branch. Not a finding of this measurement, but
  it is why the branch sections 3 and 6 are about could behave this way without
  anything noticing.

## Reproducing

```bash
~/rebasis/.venv/bin/python spikes/mixed_fusion.py \
    --corpus beir/nfcorpus/test --corpus beir/scifact/test \
    --corpus beir/cqadupstack/android --corpus beir/trec-covid \
    --cache-dir ~/band-cache --out reports/mixed-fusion.json
```

The three controls, each on NFCorpus and SciFact, and the row not yet measured:

```bash
# section 6: what `rebasis migrate` writes today, rather than a re-embed
... --migrated-vectors adapter --out reports/mixed-fusion-adapter-migration.json
# section 5: what the MAX_OVER_FETCH ceiling costs
... --over-fetch-ceiling 32    --out reports/mixed-fusion-ceiling32.json
# does the finding survive a different set of records moving first?
... --queue-order random       --out reports/mixed-fusion-random-queue.json
# not measured
... --corpus beir/fiqa/test    --out reports/mixed-fusion-fiqa.json
```

Embeddings come from the same `~/band-cache` the band harness fills, so a corpus
already measured there costs no GPU time here.
