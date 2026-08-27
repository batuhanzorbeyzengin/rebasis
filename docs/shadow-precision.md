# What a float16 shadow gives up

`--keep-original` writes every record's previous vector to a shadow file before
it is overwritten, and `rebasis rollback` reads them back. At the default
`float32` that file holds the bytes that were read, so the restore is
bit-identical. `--shadow-precision float16` halves it.

The plumbing for that has been in `ShadowStore` since it was written and the
option was never exposed, for a reason worth restating: **a half guarantee may
be more dangerous than no guarantee.** A user who believes `rollback` is exact
and gets something else has been told a smaller truth than one who was told
nothing. So the option waited on a measurement of what the smaller truth is.

**68 runs — seventeen corpora, four models, 256 to 768 dimensions.** For each,
the index as it stands and the index a rollback from a float16 shadow would
leave, queried with the corpus' own real questions and scored with
[ranx](https://github.com/AmenRa/ranx) against human judgements.

---

## 1. Nothing leaves the format

float16 tops out at 65504 and flushes below about 6e-8. A unit vector cannot
overflow; an unnormalised one can, and this ladder has both kinds of model on
it — so the question was measured rather than assumed.

| | |
|---|---|
| Runs where any component overflowed to infinity | **0 / 68** |
| Runs whose vectors were not unit vectors | 0 / 68 |
| Most components flushed to zero in one run | 0.59% |
| Largest component error | 2.4e-04 |
| Largest error as a fraction of the vector's norm | 3.3e-04 |

The flushed components are the ones already nearest zero, which is why an error
that reaches 2.4e-04 on a single component moves the whole vector by at most
3.3e-04 of its length.

**This is a property of unit vectors, not of float16.** A store holding
unnormalised vectors with components above 65504 would produce infinities here,
and rebasis has no way to know that before it writes. Nothing in this table
covers that case.

## 2. The ranking survives, and the ordering mostly does

| | min | median | max |
|---|---:|---:|---:|
| Top-10 **set** unchanged | 0.9978 | 0.9995 | 1.0000 |
| First hit unchanged | 0.9200 | 1.0000 | 1.0000 |
| Top-10 **order** unchanged | 0.6800 | 0.9768 | 0.9967 |

Read those three rows together, because they disagree in an informative way. The
documents that come back are all but always the same ones — the worst run keeps
99.78% of top-10 slots. What moves is the order *within* the top ten, on about
2% of queries in the median run.

The 0.68 is TREC-COVID under `potion-base-8M`, where the top-10 **set** was
identical on every one of its 50 queries and the ordering differed on 16 of
them. A float16 step is enough to swap two documents whose scores were already
within 3e-04 of each other, and no adjacent pair being that close is not
something an index promises.

## 3. Quality does not move

| | |
|---|---|
| Median nDCG@10 change | **0.00000** |
| Worst nDCG@10 change | **−0.00170** |
| Runs moving by more than 0.001 | 2 / 68 |
| Runs moving by more than 0.005 | **0 / 68** |

The worst cell is ArguAna under `all-MiniLM-L6-v2`, 0.5057 → 0.5040, over 1,401
queries.

For scale: [`bridge-band.md`](bridge-band.md) reports ARR's bootstrap confidence
interval at ±0.024, and `RefitPolicy` declines to adopt a new adapter below a
0.01 improvement on the grounds that anything smaller is measurement noise.
Every run here is inside both.

## What ships

`--shadow-precision float16` is exposed, and `float32` stays the default. The
default is not a hedge: the space it costs is temporary and a couple of
gigabytes of disk is cheaper than any argument about whether 0.0017 mattered on
somebody's index.

What makes the option safe is that nothing claims bit-identity when it is on:

- The disk-space plan before the confirmation says the rollback becomes
  approximate.
- The shadow's own manifest records the precision, and `rollback` prints it from
  there — off the file being restored from, which is the one record that cannot
  disagree with itself.
- `migrate.job.started` carries it into the audit trail. Nothing in the index
  says which precision a job used, so those two are where it survives.

## What this does not establish

- **Unit vectors only.** Every model on this ladder normalises. A store holding
  unnormalised vectors is the case that could overflow, and it is unmeasured.
- **The store's own round trip is not included.** These numbers are the
  shadow's loss alone. What lands back in the index also goes through the
  store's upsert, which for a normalising store adds one float32 ulp
  ([the shadow's own docstring](https://github.com/batuhanzorbeyzengin/rebasis/blob/main/src/rebasis/storage/shadow.py))
  and for a quantizing one adds its quantizer's error. On a store that
  `doctor --store` reports as quantized, two approximations compose and only one
  of them is rebasis'.
- **One cast, not repeated ones.** A job migrated, rolled back, migrated again
  and rolled back again passes through float16 twice. The second pass is
  idempotent — a value already representable in float16 survives another cast —
  but nothing here runs it.
- **No timing.** Half the bytes is half the writes, and whether that is visible
  next to the embedding pass is not measured.

## Reproducing

```bash
PYTHONPATH=src python spikes/shadow_precision.py \
    --corpus heldout --corpus beir --corpus beir/trec-covid \
    --out reports/shadow/rows.jsonl --device cuda
```

The round trip is `vectors.astype(float16).astype(float32)`, which is exactly
what `ShadowStore` does on write and read; nothing else in the path touches the
values.
