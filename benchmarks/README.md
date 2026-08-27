# Benchmarks

The performance harnesses, and the ones used to produce the measurements in
`docs/`.

## What runs where

The marker says what a test **asserts on**, and that decides where it can run.
`memory` asserts peak allocation, which `tracemalloc` measures identically on a
loaded shared runner and on a quiet host — so it gates a merge. `perf` asserts
wall clock, which a shared runner cannot measure — so it does not.

| | Marker | Where | Gate |
|---|---|---|---|
| `test_hot_path.py`, the wall-clock half | `perf` | GPU host | relative cost only — never blocks |
| `test_hot_path.py`, the allocation half | `memory` | every PR | absolute — exceeding blocks |
| `test_memory_ceiling.py`, all but one | `memory` | every PR | absolute — exceeding blocks |
| `test_memory_ceiling.py`, the time curve | `perf` | GPU host | super-linear only, 4x tolerance |
| `test_macro_budgets.py` | `slow`, `perf` | nightly, GPU host | 120% of the performance budget |
| `tests/golden/` | `slow` | nightly, GPU host | decisions exact, ARR within a band |

Wall-clock benchmarks never block a PR. On a shared runner a wall-clock gate
needs a 7% threshold just to hold false positives at 1%, and a 7% gate hides
exactly the regressions worth catching. `test_hot_path.py`'s `perf` half
therefore asserts only what is portable — that Procrustes stays cheaper than the
MLP, that batching amortises — and the absolute figures are measured on the GPU
host, where the hardware is known.

**Only the absolute-memory ceilings are gates, and for a while that sentence was
false.** Both files carried `perf`, CI runs `-m "not perf"`, and so the ceilings
gated nothing while this table went on saying they gated every pull request. The
two markers exist to keep the sentence true: excluding the noisy half no longer
excludes the deterministic half with it.

Locally: `just gate` runs what CI gates on, `just bench` runs the wall-clock
layer. Neither is in the default `just test` loop.

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

Put it in `tests/performance/` and make it measure one thing: a benchmark that
exercises three subsystems tells you something regressed and not what.

Then pick the marker by what you **assert on**, not by where the file lives:

- Asserting bytes — `tracemalloc`, a parameter count, a declared budget — is
  `memory`. It gates every pull request, so the number has to hold on a runner
  you do not control.
- Asserting seconds, or a ratio of seconds, is `perf`. It runs on the host.

Do not give a test both. `perf` is excluded from the merge path, so a test
carrying both is excluded from it too — which is exactly how the memory ceilings
came to gate nothing while the table above claimed they gated everything.
