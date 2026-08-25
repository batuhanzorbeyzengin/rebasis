# Testing

## The layers

Markers, and what each is for:

| Marker | What it covers | Budget |
|---|---|---|
| `unit` | Fast, isolated, no I/O | < 10 s total |
| `property` | Hypothesis-driven | < 60 s |
| `contract` | Every registered backend runs the same suite | < 3 min |
| `integration` | A real store or embedder | < 10 min |
| `e2e` | The full CLI flow | < 15 min |
| `perf` | Benchmarks and ceilings | excluded by default |
| `gpu` | Needs an accelerator | server only |
| `slow` | Golden corpora, macro benchmarks | server only |
| `network` | Downloads models or datasets | opt-in |

The default `pytest` run executes only the fast layer. That is a decision, not
an oversight: a developer loop longer than ten seconds is a loop that stops
being run.

## Determinism is enforced

An autouse fixture pins every source of randomness before each test, and
`pytest-randomly` shuffles execution order. A test that leaks state therefore
fails *irreproducibly*, which costs more time than the state it was saving — so
the fixture has no teardown, on purpose.

## The contract suite

`tests/contract/test_vector_store.py` runs against every registered backend. Two
of its tests matter more than the rest, because they cover what a backend is
most likely to get wrong quietly:

**Laziness.** A materialising `iter_records` breaks the memory invariant only on
corpora large enough that nobody notices in development.

**Truthful capabilities.** A store that claims more than it can do fails halfway
through a migration instead of at second zero. That now covers
`can_rebuild_index` as well: a backend that declares it has to actually rebuild
when asked, and one that does not has to refuse rather than silently do nothing.
Which of the two a backend is decides whether a migration's cost to the search
structure is recoverable — see [what a migration does to the
index](../index-health.md).

## Performance tests

Three layers:

1. **Memory ceilings** — absolute thresholds. Exceeding one **blocks** a PR:
   unlike a wall-clock comparison, a ceiling has no false-positive trade-off.
2. **The scaling test** — peak memory measured at three corpus sizes and
   asserted *not* to track N. This is the guard on the architecture's central
   invariant.
3. **Macro benchmarks** — end-to-end against the performance budgets, on the
   server, nightly.

Wall-clock benchmarks never block a PR. On a shared runner a wall-clock gate
needs 7% just to keep false positives at 1%, and a 7% gate hides exactly the
regressions worth catching.

## Adding a test that needs a real service

Prefer the embedded mode. Qdrant runs from a path with no server; sqlite-vec is
an extension; LanceDB and Chroma are files. Every integration test in the suite
runs without a container.

Skip honestly when the dependency is genuinely absent:

```python
qdrant_client = pytest.importorskip("qdrant_client", reason="qdrant-client is not installed")
```

## What a clone cannot run

Two markers, `gpu` and `slow`, never run from a clone, and one more, `perf`, is
excluded locally by default.

The reason is the hardware, not the permissions. Device parity has nothing to
compare against on a single-device runner, so a parity suite there is not a
parity suite. The golden corpora and the macro benchmarks have the same problem
in a different form: their numbers only mean something on a machine whose
specification is recorded next to them. Both run nightly on the project's own
GPU host, along with the perf layer. The wrapper that drives that host carries a
real instance id and a real host alias, so it stays on the maintainer's machine
and is not in a clone — as does the workflow that calls it.

This is stated rather than papered over: **a contributor cannot reproduce those
numbers, and a pull request is not expected to.** Everything a review gates on —
`unit`, `property`, `contract`, `integration`, `e2e` — runs from a clone with no
container and no service to install. If a change needs a GPU number to justify
it, say so in the pull request and it will be measured on the host.

## What CI runs, and what it no longer does

One test leg: Ubuntu on the declared floor, 3.12. It was three — two Python
versions and a macOS runner — and collapsing it was a decision about wall clock
rather than compute. A hung macOS job held every merge behind it for half an
hour at a time.

**macOS is not in CI, and the suite still runs there.** It is the maintainer's
own platform, which is where the storage-layer differences that leg existed for
— directory fsync, `os.replace`, a system sqlite3 that cannot load extensions —
actually get exercised. The leg also could not finish: `faiss-cpu` and `torch`
each link their own OpenMP runtime on macOS and a process holding both aborts
with `OMP: Error #15` before either library does any work
([faiss-wheels#40](https://github.com/kyamagu/faiss-wheels/issues/40),
[pytorch#149201](https://github.com/pytorch/pytorch/issues/149201)). That is not
rebasis' bug and there is no fix a caller can apply; the documented workaround
is itself documented as liable to produce wrong results. `rebasis doctor` reports
the pair, and the FAISS tests skip themselves under it with the reason attached.

**The newest Python is not tested separately**, because across this
repository's runs it caught nothing the floor did not. `lowest direct
dependencies` still pins the floor's dependency versions, and that is the check
that has actually failed.

Every job carries a `timeout-minutes`. A job that hangs should cost minutes, not
the six hours a runner will otherwise give it — and `faulthandler_timeout` in
`pyproject.toml` means a hang names the test it happened in rather than the
percentage it stopped at.

Bring a leg back when there is a failure it would have caught.
