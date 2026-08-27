# Refitting during a migration, and when it is worth it

A migration can run for hours. `rebasis/migrate/refit.py` has been able to refit
the adapter part-way through since it was written, with a guard that adopts the
result only if it beats the one in use — and nothing has ever called it. This is
what happens when something does.

The number that decides it is not "did a refit ever help". Over enough runs
something always helps. What decides it is whether the help clears
`RefitPolicy.min_improvement` — **0.01**, the threshold the guard already
enforces — often enough to be worth the documents it re-embeds to get there.

**216 cells.** Sixteen corpora, three ladder rungs, two migration orders, two
original fit budgets, and four mixtures of two corpora read as one index. Every
arm scored the way `rebasis probe` scores a forward map: rewrite the slice, send
a **raw** new-model query at it, and count what a full reindex of that slice
would have returned. Computed by
[`tools/refit_stats.py`](https://github.com/batuhanzorbeyzengin/rebasis/blob/main/tools/refit_stats.py),
so every count below is a script's output rather than a reading.

---

## 1. The premise the module was built on was wrong

Its docstring said pairs become available "for free" during a migration, because
records already migrated carry new-model vectors.

They do not. A migrated record carries `A(old)` — the adapter's own image of the
old vector — so fitting on those pairs fits `A` to reproduce `A`. Every real
pair costs a document re-embedded, which is why the engine needs an embedder to
do this at all, and why the question is not *which pairs are free* but **which
pairs are worth paying for**.

## 2. On a corpus that has not changed, it is a pair-count effect and nothing more

Against `rebasis fit`'s default 4,000-pair budget, over 96 cells:

| arm | median gain | clears 0.01 |
|---|---:|---:|
| 12,000 pairs (4,000 kept + 8,000 new) | **+0.0075** | 12 / 72 |
| 8,000 new pairs alone | +0.0049 | 5 / 72 |
| 4,000 new pairs alone | −0.0022 | 1 / 84 |
| 1,000 new pairs alone | −0.0360 | 0 / 96 |

Three times the fit budget moves retention by seven thousandths and clears the
guard in one cell in six. That is
[the squeeze](bridge-band.md) arriving again from a third direction: retention
is not improvable by fitting harder.

**Where the pairs came from does not matter here.** Held at equal pair count,
drawing from the records not yet migrated rather than from the ones already
done is worth −0.0016 to −0.0026 — the wrong sign, and not significant after
Holm on most budgets.

**Neither does the migration order.** `refit.py` carried a caveat that pairs
accumulated during a migration come from records processed in priority order
rather than at random. Measured, `--priority none` and an access-ordered
migration give +0.0073 and +0.0080. The caveat describes something real about
the sample and nothing about the outcome.

## 3. Except when the original fit was under-budgeted

The same grid with the original adapter fitted on 1,000 pairs instead of 4,000:

| arm | median gain | clears 0.01 |
|---|---:|---:|
| 2,000 new pairs alone | **+0.0205** | 78 / 90 |
| 4,000 new pairs alone | +0.0329 | 84 / 84 |
| 1,000 new pairs alone | −0.0024 | 3 / 96 |

So a refit recovers an under-budgeted fit, decisively — and reading the two
tables together says why. `refit:2000` at a 1,000-pair budget scores the same as
`accumulated:1000` at the same budget, which is also 2,000 pairs in total. **Only
the count matters, not which pairs.** Retention saturates near 4,000 and a refit
is simply another way of getting there.

Which is an argument against the feature, not for it: `rebasis fit --pairs 4000`
is the default, it costs less, and it happens before anything is written.

## 4. The case it is actually for: a corpus that grew

Four indexes assembled from two unrelated cqadupstack forums each, migrated in
arrival order — every document of the first forum, then every document of the
second — with the adapter fitted on the **first alone**. That is an index that
gained a domain while the migration was running.

| arm | median gain | clears 0.01 |
|---|---:|---:|
| **1,000 pairs from the new domain** | **+0.1598** | 12 / 12 |
| 8,000 pairs from the new domain | +0.2096 | 12 / 12 |
| 8,000 pairs from the old domain | +0.0077 | 4 / 12 |
| 1,000 pairs from the old domain | −0.0345 | 0 / 12 |

Held at equal pair count, drawing from the remainder rather than from the
migrated half is worth **+0.20 to +0.21**, in every cell, at every budget
tested. The effect is flat in the number of pairs, which is the tell: this is
not the saturation curve of section 2 arriving late. It is a different quantity
entirely — **which distribution the map was fitted to** — and a thousand pairs
of the right kind beat eight thousand of the wrong kind by an order of
magnitude.

**Keeping the original pairs makes it worse.** `refit` alone scores +0.2096
against +0.1911 for the same pairs with the original 4,000 still in the fit.
They pull the map back toward a domain that is no longer what is being written.

## 5. What ships, and why the guard is the whole design

`migrate --refit` samples records **not yet migrated**, re-embeds them, refits on
those pairs alone, and adopts the result only if it beats the adapter in use on a
held-out slice by 0.01.

Both readings above are true at once, and the guard is what lets them be. On an
unchanged corpus a 1,000-pair refit loses to a 4,000-pair adapter and is
declined — section 2 says it would have been worth +0.0049 at best, and buying
that with 8,000 re-embedded documents is not a trade anyone should make silently.
On a drifted one it wins by sixteen times the threshold and is adopted.

Off by default, because the corpus that needs it is the exception and the
documents it re-embeds are not free.

## Not the same thing as Drift-Adapter's continuous adaptation

[Drift-Adapter](https://arxiv.org/abs/2509.23471) §5.6 reports that refitting
hourly keeps ARR above 0.95 over 24 hours where a fixed adapter "trained only at
T=0" degrades to around 0.83. The names collide and the mechanisms do not.

Their adapter maps **queries into the old space**, and their index fills with
items "now purely in the `f_new` space". A query mapped backwards is wrong for a
growing share of that index, so refitting chases a target moving underneath it.
rebasis serves exactly that index with two-space search instead
([the measurement](mixed-space-fusion.md)), which is a structural answer rather
than a moving one — so that half of their scenario does not arise here.

What is left of it once that is removed is the corpus changing *in kind*, which
is section 4, and which their setup does not isolate.

## What this does not establish

- **One adapter family.** Everything here is `procrustes_centered` in both
  directions, for the reason [`migration-band.md`](migration-band.md) gives.
- **One seed, one split point.** The migration is interrupted at 50% (at the
  domain boundary for the arrival arm), seed 0, no repeats. Where the best
  moment to refit is, and whether refitting more than once compounds, is
  unmeasured — the engine allows it, and nothing here says it pays.
- **The drift arm is four mixtures.** Two forums each, assembled. The *corpus*
  is assembled and the *drift* is not — it is whatever two real models do to
  real text — but four pairs of StackExchange forums is not a survey of how
  corpora actually change.
- **No cost model.** The gain is reported; the seconds spent re-embedding are
  a property of the model and the hardware, and `probe`'s reindex estimate is
  the thing that would price them.
- **`refit.py`'s comparator is not the metric above.** The guard scores mean
  cosine similarity on held-out pairs, which needs no index and no search. The
  tables here score retrieval. They agree on these runs, and a case where they
  disagree would be a finding about the guard.

## Reproducing

```bash
# The static grid: 16 corpora, three rungs, two orders, two fit budgets
PYTHONPATH=src python spikes/continuous_refit.py \
    --corpus heldout --corpus beir --corpus beir/trec-covid \
    --ladder default --out reports/refit/rows.jsonl --device cuda

# The corpus that grew: two forums read as one index, adapter fitted on the first
PYTHONPATH=src python spikes/continuous_refit.py \
    --corpus mix:beir/cqadupstack/android+beir/cqadupstack/mathematica \
    --corpus mix:beir/cqadupstack/gaming+beir/cqadupstack/physics \
    --corpus mix:beir/cqadupstack/english+beir/cqadupstack/stats \
    --corpus mix:beir/cqadupstack/unix+beir/cqadupstack/gis \
    --order arrival --fit-scope first \
    --ladder default --out reports/refit/drift.jsonl --device cuda

python tools/refit_stats.py reports/refit/rows.jsonl
python tools/refit_stats.py reports/refit/drift.jsonl
```

Corpus loading, the ladders and the embedding cache are imported from
`tools/bridge_band.py`, and the `mix:` construction from `spikes/per_cluster.py`,
so a corpus means the same thing here as it does there.
