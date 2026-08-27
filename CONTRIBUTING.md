# Contributing to rebasis

## Setup

```bash
git clone https://github.com/batuhanzorbeyzengin/rebasis
cd rebasis
uv sync                                    # dependencies and dev tools
uv run pre-commit install --install-hooks --hook-type commit-msg
just check                                 # about 30 seconds
```

`just check` runs exactly what CI runs on the fast path: ruff, mypy `--strict`,
import-linter and the unit and property tests. If it passes locally it will pass
in CI.

## The development loop

1. **Write the test first**, in `tests/unit` or `tests/contract`. Watch it fail.
2. Write the smallest implementation that makes it pass.
3. `just check` — thirty seconds, locally.
4. **Added an error?** The error class, its stable code and its
   `docs/reference/errors.md` entry go in **the same PR**. A code without
   documentation is a code a user will google and not find.
   **Added a log event?** The `Events` entry, the catalogue row and the fields it
   carries, likewise together.
5. **Touched a hot path or memory behaviour?** Run `just gate` — that is the
   layer CI blocks on, and running it locally is how you see it go red before
   pushing rather than after. `just bench` runs the wall-clock layer, which never
   blocks a merge because a shared runner cannot measure timing.
   **Touched numerics?** Run device parity on CPU and at least one accelerator.
6. **Changed adapter or metric behaviour?** Run the golden tests on the GPU host.

The default `pytest` run executes only the fast layer. That is deliberate: a
developer loop longer than ten seconds is a loop that stops being run.

## Adding a store backend — three steps

1. Write `src/rebasis/store/backends/<name>.py` implementing the `VectorStore`
   protocol.
2. Register it under `[project.entry-points."rebasis.stores"]` in
   `pyproject.toml`.
3. Add a builder to `tests/conftest.py` and the backend's name to
   `LIVE_BACKENDS` in `tests/contract/test_vector_store.py`, then make the suite
   pass.

The contract suite runs the same tests against the in-memory store and against
chroma, faiss, lancedb, qdrant and sqlite-vec — every one embedded, none of them
needing a server. It ran against the in-memory store alone for far longer than
it should have: the two things it checks hardest, laziness and truthful
capabilities, are precisely what a real client library gets wrong and a dict
cannot.

**Declare capabilities honestly.** If your store cannot read vectors back out,
say so — `probe` and the bridge phase will still work, and `migrate` will refuse
up front with a clear reason. Partial support is genuinely useful; *silent*
partial support fails in the middle of someone's migration, which is worse than
not supporting it at all.

Adding an embedder follows the same three steps against `rebasis.embedders`.

## Things the codebase enforces

These are checked mechanically, so there is no need to remember them — but
knowing *why* saves a confusing CI failure:

- **No `logging.basicConfig`, `logging.getLogger` or `print()` under `src/`**
  outside `observability/` and `cli/`. Centralised logging has one failure mode —
  somebody attaching a handler off to the side — and that is not left to
  discipline.
- **No direct `open(path, "w")`** outside `storage/atomic.py`. Overwriting in
  place means a full disk deletes the old content and fails to write the new one.
  `atomic_write_*` removes that failure mode entirely.
- **Only registered event names.** Free-text log messages cannot be grepped or
  audited.
- **Only allowlisted log fields.** Anything else is redacted, and the redaction
  counter firing in production code is a **bug signal**, not a safety net
  working.
- **Layer contract.** import-linter enforces it. `core` knows nothing about
  stores, embedders or the CLI.

## Never logged, categorically

Document text, vectors, query text, filesystem paths, API keys and the
credential part of a store URI. Vectors are on that list because text can be
reconstructed from embeddings — a vector in a log is as sensitive as plaintext.

## Commit messages

Conventional Commits. The version number is computed from them, not chosen:

```
feat(store): add sqlite-vec backend             → MINOR
fix(probe): correct the ARR denominator         → PATCH
perf(core): avoid a copy in the Procrustes fit  → PATCH
feat(core)!: change default decision thresholds → MAJOR
docs: ...   test: ...   chore: ...              → no release
```

Merges are squashed, so **the PR title is the commit message**.

## No dangling references

Comments, docstrings and printed output used to cite numbered sections of a
technical design document that is not in the repository. They are gone, and they
do not come back: a reference a reader cannot open is not a reference. Say what
the rule is, in the sentence that needs it.

An output-hygiene test checks every line the CLI prints. Everything a user needs
is at [the docs site](https://batuhanzorbeyzengin.github.io/rebasis/) or in
`docs/`.

## Measurement over assertion

Design claims in this project are expected to carry numbers. `docs/m0-findings.md`
is the worked example: it measures the original design's assumptions and records
where they did not hold — mean centering being worth +0.26 ARR, CSLS
*hurting* strong adapters, the hot-path budget being unreachable on a cloud vCPU.

If you change something on the strength of a claim, measure it. If a measurement
contradicts a design claim, that is a finding worth writing down, not a problem
to route around.
