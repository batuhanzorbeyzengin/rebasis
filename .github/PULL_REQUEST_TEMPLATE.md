## What and why

<!-- What changes, and what problem it solves. Link an issue if there is one. -->

## Checklist

Tick what applies; delete what does not.

- [ ] `just check` passes locally (ruff, mypy, import-linter, fast tests)
- [ ] Behaviour is covered by a test that failed before the change
- [ ] **New error?** Error class + stable code + `docs/reference/errors.md` entry, all in this PR
- [ ] **New log event?** `Events` entry + catalogue row + declared fields
- [ ] **New store or embedder?** Entry point registered and the contract suite passes
- [ ] **Touched a hot path or memory behaviour?** Benchmarks and memory ceilings re-run
- [ ] **Touched numerics?** Device parity checked on CPU and at least one accelerator
- [ ] **Changed decision thresholds or a metric definition?** A changelog fragment under "Behaviour changes", with the rationale

## Commit messages

Conventional Commits — the release version is computed from them, not
chosen by hand. Because merges are squashed, **the PR title is the commit
message**.
