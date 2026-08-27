`tests/performance/test_macro_budgets.py` is marked `perf` as well as `slow`, which takes its wall-clock gates off the pull-request path.

The file's own docstring said "wall clock never blocks a PR", and it did. CI runs `-m "not network and not perf"`, so `slow` alone was not enough to exclude it, and six budget assertions — 20s, 90s, 180s, 360s, 30s, and 50ms for an adapter load — were gating merges on a shared runner.

`ci.yml` already carries the reasoning, written when the same thing happened to the `perf` layer: *"the perf layer asserts timing, and a shared runner cannot measure timing: this job failed twice on `test_batching_amortises_the_per_call_cost`, by 1% and by 2.6%, which is noise wearing a red X."* That argument is about what a test measures rather than which marker it happens to carry, and it applies here exactly — it was simply not reached, because the exclusion is keyed on `perf` and this file was `slow`.

Observed rather than anticipated: on a host running two other jobs, the residual-MLP fit blew its 180-second budget; measured alone on the same machine minutes later it took **17 seconds**, a tenfold swing with nothing in the code changed. That is the second timing assertion to fail this way in one day — `test_the_centred_adapter_folds_its_offset` was the first, and it has since been split so that the property it protects is checked algebraically instead.

Nothing is lost. `slow` still keeps the file out of the default developer run; `perf` keeps it off the merge path; `scripts/remote.sh test` runs `-m "not network"` and so still executes all six on the project's own host, which is the machine whose hardware their numbers describe.
