# What a completed migration is worth

`rebasis migrate` rewrites the vectors an index holds. Until this release it
applied the wrong map to do it, and left an index no query could answer — that is
[its own story](guides/migration.md). With the right map it works. This is what
the right map buys, measured the same way everything else here is measured: real
corpora, real user questions, human relevance judgements, scored with
[ranx](https://github.com/AmenRa/ranx) rather than with rebasis' own metric code.

**51 cells — seventeen corpora, three ladder rungs.** Four configurations, all
against the same index:

| | |
|---|---|
| `status_quo` | old query → old index. What you have. |
| `bridged` | adapted new query → old index. What [`Bridge`](concepts/adapters.md) serves. |
| `migrated` | **raw** new query → forward-mapped index. What `migrate` leaves. |
| `reindexed` | new query → new index. The ceiling. |

The third is the new one. It is the only row here where no adapter sits on the
query path: after a migration finishes there is nothing to bridge with, the new
model queries an index that is supposed to be in its own space, and whether that
works is the question the command exists to answer.

There is deliberately no fifth row bounding `migrated` from above. For a
*document* map the best achievable result is the documents' own new-model
vectors — no map of the old ones can beat having actually re-embedded them — so
the ceiling on `migrated` **is** `reindexed`, as an identity rather than a
measurement. What is worth reading is the gap.

---

## 1. It works, and it delivers about three quarters of a reindex

| | mean | min | max |
|---|---|---|---|
| `migrated` ÷ `reindexed` | **0.727** | 0.366 | 0.971 |

Before the direction was fixed this number was **0.000** — not low, but
unanswerable: the index was in neither model's space and every query type
returned nothing. Three quarters of a reindex, for a map that fits in seconds
against an embedding pass that does not, is the trade the command was always
supposed to offer. It offers it now.

## 2. It is not distinguishable from bridging, and that is the finding

| | mean of ÷ `reindexed` |
|---|---|
| `migrated` | 0.727 |
| `bridged` | 0.719 |

Across the 51 cells the two track each other at **Spearman ρ = 0.993**
(p ≈ 1e-46). The paired difference has a median of **+0.0039** in favour of
migrating — detectable (Wilcoxon p = 0.018) and, at four thousandths of an
nDCG point, not a reason to do anything.

That is not a disappointment; it is
[ADR 10](adr/0010-retention-is-bounded-by-the-source.md) arriving from a new
direction. The ADR says retention is bounded by what the source space carries.
Both rows here are the same source space under the same family of map. Whether
the map is applied to the *query* or to the *documents* turns out not to change
how much survives it — which is a stronger statement than the ADR made, because
the ADR was measured entirely on the query side and could not have said it.

**The practical consequence.** Migrating buys you the adapter leaving the hot
path, and it costs you a rewrite of every vector in the index plus the shadow
copy behind it. What it does not buy is retrieval quality. Anyone choosing
between the two should choose on operational grounds, and this page is the
evidence that quality is not one of them.

## 3. Both are usually worse than doing nothing

| | beat `status_quo` |
|---|---|
| `migrated` | **5 / 51** |
| `bridged` | 2 / 51 |

Median `migrated` − `status_quo` is **−0.0446** (Wilcoxon p = 2.2e-08). In 46 of
51 cells, a completed migration leaves the user worse off than never having
started.

This is the same shape [`bridge-band.md`](bridge-band.md) reports for bridging
and the reason `probe` recommends against itself four times in five. It is worth
restating here only because `migrate` is the command that *writes*: a bridge that
was not worth building can be dropped, and an index that was not worth migrating
has to be rolled back.

## 4. When it does win, the upgrade was large

The five cells where migrating beat doing nothing:

| corpus | pair | status quo | migrated | gain | upgrade gain |
|---|---|---|---|---|---|
| trec-covid | MiniLM → bge-small | 0.472 | **0.546** | +0.073 | 1.69 |
| cqadupstack/mathematica | potion → MiniLM | 0.121 | **0.140** | +0.019 | 2.41 |
| cqadupstack/android | potion → MiniLM | 0.318 | **0.328** | +0.010 | 1.70 |
| arguana | MiniLM → bge-small | 0.506 | **0.515** | +0.009 | 1.20 |
| cqadupstack/webmasters | potion → MiniLM | 0.237 | **0.240** | +0.003 | 1.60 |

Every one has an upgrade gain of 1.2 or more, against a median of 1.09 across
all 51. The break-even that governs bridging governs migrating too.

**One thing this table cannot be used for, and the reason is worth writing
down.** It is tempting to score a `migration_advantage = retention × gain` the
way `probe` scores bridging, and report how often its sign was right. That
number would be 51 out of 51, and it would mean nothing: read off one cell's own
scores, `(migrated ÷ reindexed) × (reindexed ÷ status quo)` is
`migrated ÷ status quo`, which is the same inequality as the outcome being
predicted. [`bridge-band.md` section 9](bridge-band.md#9-what-the-counting-is-worth)
found exactly that identity hiding under a published count, and this page is not
going to reintroduce it one document later.

What is reportable is the relationship between the *upgrade* and the *advantage*,
which is not an identity: Spearman **ρ = +0.800** (p = 1.8e-12) across the 51.
The two quantities share a denominator, so some of that is structural; it is
offered as the shape of the effect rather than as a rule.

---

## What this does not establish

- **One adapter family.** Everything here is `procrustes_centered` in both
  directions. `auto` was not used, deliberately: it selects by scoring on a
  held-out set, and the two directions are scored on different questions, so an
  `auto` on each side would vary the family and the direction at once. Whether a
  different family closes the gap in section 2 is unmeasured.
- **One seed, one fit budget.** 4,000 pairs, seed 0, no repeats.
- **English, and one ladder.** Three rungs, 256 → 384 → 384 → 768. The same
  limits [`bridge-band.md`](bridge-band.md) states apply here unchanged.
- **Nothing about the write path.** These numbers come from applying the map to
  the vectors in memory. They say what a migration *would be worth*; they say
  nothing about what `migrate` costs to run, what it does to an index's graph
  ([`index-health.md`](index-health.md) measures that separately), or how it
  behaves when interrupted.
- **No per-query significance on the individual cells.** The paired tests above
  are over the 51 cell means. The harness does not yet write per-query scores
  for these configurations, so a per-corpus claim would not survive the
  correction [`bridge-band.md`](bridge-band.md#9-what-the-counting-is-worth)
  applies to the band.

## Reproducing

```bash
# 17 corpora, three rungs, four configurations each
PYTHONPATH=src python spikes/migration_band.py \
    --corpus heldout --corpus beir --corpus beir/trec-covid \
    --ladder default --k 10,100 \
    --out reports/migration/rows.jsonl --device cuda
```

The adapters come from the same `fit_candidates` call `rebasis fit` makes and are
applied through the same `BaseAdapter.apply` the engine uses, so what is measured
is the tool rather than a reimplementation of it. Corpus loading, the ladders and
the embedding cache are imported from `tools/bridge_band.py` so that a corpus
means the same thing here as it does there — including the self-removal
convention, which moved a published number by 0.2 nDCG the one time it was
missed.
