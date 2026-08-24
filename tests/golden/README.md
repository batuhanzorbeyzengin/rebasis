# Golden fixtures

Real embeddings of a real corpus, so the pipeline can be tested on the thing it
is actually for. Every other test in this repository runs on synthetic vectors
with a planted rotation: those prove the mathematics and cannot prove that
prefix handling works or that hard drift is recognised as hard.

## What is here

`manifest.json` — committed. Names the four scenarios, the model pair for each,
and a SHA-256 per fixture file. A substituted or corrupted fixture fails loudly
instead of quietly shifting a band.

`data/*.npz` — **not committed**, about 20 MB each. Regenerate them:

```bash
uv run --extra sentence-transformers --with ir-datasets --with model2vec \
    python tools/make_golden.py --out tests/golden/data
```

Point the tests elsewhere with `REBASIS_GOLDEN_DIR`.

## Why vectors and not models

Each pair is embedded once, by the build script, and the vectors are stored. The
tests then run the whole probe pipeline through `PrecomputedEmbedder` — no model
download, no network, no GPU. That is what makes them runnable in CI, and fast
enough to actually be run.

## The four scenarios

| Scenario | Models | What it is for |
|---|---|---|
| `same_family` | all-MiniLM-L6-v2 → L12-v2 | A consecutive version. The easy case. |
| `different_family` | all-MiniLM-L6-v2 → bge-small-en-v1.5 | Different training, same dimension. The common case. |
| `prefix_trap` | all-MiniLM-L6-v2 → e5-small-v2 | Symmetric to asymmetric. Detects broken prefix handling — **at T1 only**, see below. |
| `hard_drift` | all-MiniLM-L6-v2 → potion-base-8M | Static distilled embeddings: a genuinely different architecture, and a different dimension. |

## What the measurements said

Run on the project's host against BEIR/scifact, 5,000 documents, 295 judged
queries. Recorded in `docs/golden-findings.md`; the short version:

- **The prefix trap only traps at T1.** With the prefixes removed, T0 barely
  moves (0.870 → 0.875) while T1 moves enough to change the recommendation.
  T0's ground truth is the new model's own output, so a consistently wrong
  prefix moves the reference along with the measurement and cancels out.
- **ARR is not comparable across scenarios.** `hard_drift` scores the *highest*
  T1 ARR of the four (0.971) because its oracle is weak — the tool correctly
  answers `no_upgrade_needed` rather than being fooled by the number.

Both are why these tests assert **decisions**, not only numbers.
