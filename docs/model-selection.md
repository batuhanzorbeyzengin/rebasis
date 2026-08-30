# Which model, on your corpus

`rebasis compare` ranks candidate models against the one already in your index,
from a sample, without embedding the corpus once per candidate and without
rebuilding anything.

```bash
rebasis compare --store chroma:///path/db#docs \
  --old sentence-transformers/all-MiniLM-L6-v2 \
  --candidates BAAI/bge-small-en-v1.5,BAAI/bge-base-en-v1.5 \
  --queries queries.jsonl --report compare.html
```

This page is what that ordering is worth. **Read it before gating anything on
the table**, because the headline is a loss.

---

## The result, first

Sixteen corpora — the twelve CQADupStack forums, FiQA, SciFact, NFCorpus and
ArguAna — with `potion-base-8M` as the index's model and three candidates
against it: `all-MiniLM-L6-v2`, `bge-small-en-v1.5` and `bge-base-en-v1.5`.

The estimate comes from a **sample** of each corpus, scored on recall against
the corpus' own judged queries. The truth is each candidate's nDCG@10 over the
**whole** corpus, its own vectors against its own index, scored by `ranx`
against the human judgements — what a full reindex to that candidate would
deliver.

**The null is what everybody actually does: pick whatever tops the published
MTEB table.** MTEB is an average over 56 tasks and it is nobody's corpus, which
is the objection this measurement exists to test. Scores are taken from the
models' own cards, so the null's prediction is the same order on every corpus.

| rule | named the genuinely best candidate |
|---|---|
| published MTEB order | **14 / 16** |
| `rebasis compare`, sample 4,000 | **9 / 16** |

**The null wins.** On this candidate set, reading the leaderboard is better than
running the command, and that is the first thing this page should say because it
is what a reader would otherwise have to discover for themselves.

Two things make the null unusually strong here, and neither excuses the result:

- **The candidate set is a quality ladder.** These three models are 5, 6 and 12
  MTEB points apart and are in a known order. A published table is right about a
  known order almost by construction, and it is right here on 14 of 16 corpora.
- **The truth is one-sided.** `bge-base` is genuinely best on 14 of the 16, so a
  rule that always answered "the biggest model" scores 14 too. Section 9's
  lesson applies to this table as much as to the one it was written about: an
  accuracy cannot separate a real rule from a constant when the outcome is this
  one-sided.

What that means is that this measurement is a weak test *of the null*, not that
it is a weak test of the command. The command was given sixteen chances to
disagree with the table and be right, and took two of them — `android` and
`gis`, where `all-MiniLM-L6-v2` genuinely beats `bge-base` and the leaderboard
says otherwise. Those two are the whole case for running it, and two is not
many.

---

## What the ordering does carry

The proportion above is not the whole measurement and it is not the most
informative part of it. Two things survive.

**It orders the candidates, positively but not strongly.** Mean Spearman
ρ = **+0.469** across the sixteen corpora at a 4,000-document sample, Kendall
τ = +0.417. Eleven of sixteen corpora are positive, four are ties at +0.500, and
**three are negative** — `unix`, `webmasters` and `fiqa` all come back at −0.500.
A correlation with three sign flips in sixteen is real and is not something to
gate a decision on.

**It is per corpus, which is the only thing the leaderboard cannot be.** The two
corpora where the command beats the table are the two where the truth departs
from the published order, and it found both. That is the shape of what this is
for: not a better global ranking — the table has more evidence behind it than
any single corpus can — but a signal about *yours* to weigh against it.

---

## How it moves with the sample

The estimate is taken on a sample and the truth on the whole corpus, so the
obvious suspect for the gap is the sample. A 4,000-document mini-index is an
easier retrieval problem than a 60,000-document one — `docs/access-weighting.md`
already measured a 4,000-document mini-index sitting +0.048 above the
whole-corpus quantity — and an easier problem compresses the gap between a
strong candidate and a weaker one.

The whole grid re-run at four sample sizes, sixteen corpora each:

| documents sampled | mean Spearman ρ | mean Kendall τ | top-1 |
|---|---|---|---|
| 1,000 | +0.429 | +0.356 | 6 / 16 |
| 2,000 | +0.433 | +0.356 | 6 / 16 |
| 4,000 | +0.469 | +0.417 | 9 / 16 |
| 8,000 | **+0.688** | **+0.625** | 8 / 16 |
| *published MTEB order* | — | — | *14 / 16* |

**The ordering does stabilise, and the correlation is what shows it.** ρ moves
+0.429 → +0.433 → +0.469 → **+0.688** as the sample grows eightfold, and τ
follows. That is the mechanism the section above suspected, measured: a
1,000-document mini-index is an easy retrieval problem and an easy problem
compresses the gap between a strong candidate and a weaker one, so the ordering
it produces is noisier.

**Top-1 does not follow it, and that is the honest reading.** 6, 6, 9, 8 — the
correlation improves smoothly while the count wanders, because top-1 over three
candidates on sixteen corpora is a coarse statistic and the truth is one-sided.
Nothing here reaches the published order's 14, and nothing suggests a larger
sample would: at 8,000 documents most of these corpora are more than a third
sampled and the trend has most of its room behind it.

**What this sets.** `--sample` below about 4,000 is not worth running for an
ordering — the two smallest rows are indistinguishable from each other. Where
`--tiered`'s first round should sit is a measurement this table does not
settle, and `TIERED_FIRST_ROUND` says so in the code rather than implying it was
derived here.

A fifth point at 16,000 was started and stopped: it was competing for the same
host as three other measurements, and four points already show the trend. That
is the reason, rather than a result.

---

## The identity check

[Section 9](bridge-band.md#9-what-the-counting-is-worth) found the headline count
of an earlier measurement to be an identity, and the rule since is that a
quantity is checked for degeneracy before it is scored.

`upgrade_gain` is the oracle's recall over the incumbent's, on a **sample**. The
outcome is nDCG@10 over the **whole corpus** divided by the incumbent's. Two
metrics, two cut-offs, two populations.

Measured at a 4,000-document sample, `|upgrade_gain − (true nDCG ratio)|` has a
maximum of **0.982** and a mean of **0.344** over the 48 candidate/corpus pairs.
The two are not the same number and nothing cancels. Whatever else is wrong with
the ordering above, it is a prediction that could have been right.

---

## What the command does about it

The table ships with the caveat attached rather than as a footnote, in the
terminal, in the report and in `--json` as `ranking_caveat`. That is the whole
response to the result above: the command reports an ordering and says what the
ordering is worth, and does not report a winner.

Three design choices follow from the same place.

**One sample, one split, one reference.** Every candidate is scored on the same
drawn sample, the same fit/held-out split and the same queries. Redrawing per
candidate would introduce a shift larger than several of the gaps being
compared. Consistency across rows is bought at the cost of a little absolute
accuracy, which is the right trade for a comparison and the wrong one for a
single measurement.

**The index's model is the reference, not a row.** It is already in the index, so
its vectors are read rather than recomputed — which is what makes the whole
comparison cheap, and what makes passing it as a candidate meaningless.

**The cost is printed before the run.** N candidates is N embedding passes.
`--tiered` scores everything on a small sample first and carries through only
what that round could not separate, on the same ±0.025 band the decision rule
already reports its own borderline cases at.

**A candidate evaluated once is free the next time.** The embedding cache is
keyed on the encoding profile's fingerprint — one SQLite file per profile under
`.rebasis/cache/embeddings/` — so a second comparison over the same sample
embeds nothing, and two candidates cannot read each other's vectors however
similar their names. Both are asserted on the cache's own counters rather than
on a clock (`tests/unit/test_compare.py`), because a wall-clock assertion on a
shared runner is noise wearing a red X. That cache was built for this use and
had nothing using it until now.

---

## Two things the plan asked for that this did not settle

Both are recorded because a checklist item quietly dropped is worse than one
answered "no".

**`--tiered`'s first round rests on no measurement.** The plan asked for the
sweep above to set it. It does not: what the sweep shows is that the *ordering*
improves with the sample, not where a cheap first round should sit, and those
are different questions — the first round only has to separate what it can, and
what it can separate depends on the candidates. `TIERED_FIRST_ROUND` carries
that in the code rather than implying it was derived here.

**The identity check lives in `tools/model_selection.py`, not
`tools/band_stats.py`.** The plan asked for it as a mode of the second, so that
the same mistake could not be made twice. It is in the first because the two
tools read different files — `band_stats.py` reads the band harness's rows and
this measurement writes its own shape — and a mode that could not be run on the
file it is about would be a checkbox rather than a check.

## What this does not establish

- **One incumbent, three candidates, one language.** The candidates span 12
  MTEB points, which is a wide enough spread for a published table to have a
  confident opinion. A set of genuinely close candidates — the case where a
  leaderboard is least useful and this command would be most useful — is not
  measured here.
- **The ladder is a quality ladder.** These four models are in a known order and
  the published table is right about that order almost everywhere, which makes
  the null unusually strong. A candidate set assembled from models the table
  cannot separate would be a different and fairer test, and it is the one this
  measurement should be repeated on.
- **English, technical Q&A and BEIR.** `docs/golden-findings.md` section 7's
  warning applies unchanged.
- **Nothing here measures an API model.** The `openai_compat` backend sends
  document text off the machine, and `compare` names such a candidate before the
  run; what it costs in quality was not measured.

## Reproducing

```bash
uv run --extra sentence-transformers --with ir-datasets --with ranx \
    --with model2vec python tools/model_selection.py \
    --corpora heldout --corpora beir --cache-dir ~/band-cache \
    --out reports/band/selection.jsonl

uv run python tools/model_selection.py --summarise reports/band/selection.jsonl
```
