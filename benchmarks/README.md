# Benchmarks

The performance harnesses, and the ones used to produce the measurements in
`docs/`.

## What runs where

| | Where | Gate |
|---|---|---|
| `tests/performance/test_hot_path.py` | every PR | relative cost only — never blocks |
| `tests/performance/test_memory_ceiling.py` | every PR | absolute — exceeding blocks |
| `tests/performance/test_macro_budgets.py` | nightly, GPU host | 120% of the performance budget |
| `tests/golden/` | nightly, GPU host | decisions exact, ARR within a band |

Wall-clock benchmarks never block a PR. On a shared runner a wall-clock gate
needs a 7% threshold just to hold false positives at 1%, and a 7% gate hides
exactly the regressions worth catching. `test_hot_path.py` therefore asserts
only what is portable — that Procrustes stays cheaper than the MLP, that the
hot path allocates a bounded amount, that batching amortises. The absolute
figures are measured on the GPU host, where the hardware is known.

Only the absolute-memory ceilings are gates.

## Retrieval-quality harnesses

The numbers in `docs/bridge-band.md` come from harnesses that need real corpora
and real models, so they are not part of the test suite — they are run by hand
on the GPU host and their results recorded.

```bash
# The golden fixtures the test suite uses
uv run --extra sentence-transformers --with ir-datasets --with model2vec \
    python tools/make_golden.py --out tests/golden/data
```

The four-way comparison behind `docs/bridge-band.md` — status quo, naive swap,
bridged, full reindex — scores with `ranx` rather than with rebasis' own metric
code. Grading a tool with its own scorer tests consistency, not correctness, and
the harness is validated by reproducing published BEIR numbers to three decimal
places (`docs/bridge-band.md`, section 6).

## Adding a micro-benchmark

Put it in `tests/performance/`, mark it `perf`, and make it measure one thing.
A benchmark that exercises three subsystems tells you something regressed and
not what.
