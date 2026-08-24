# Development setup

```bash
git clone https://github.com/batuhanzorbeyzengin/rebasis
cd rebasis
uv sync --all-extras
```

`uv` is the only prerequisite. The version comes from the git tag via
hatch-vcs, so a shallow clone produces `0.0.0.dev0` — clone with history if the
version matters to what you are doing.

## The gates

Every one of these runs in CI, and every one is fast enough to run locally
first:

```bash
uv run ruff check src tests spikes
uv run ruff format --check src tests spikes
uv run mypy src            # --strict
uv run lint-imports        # the layer contract
uv run pytest              # the fast layer only, by default
```

Or `just check` for all of them.

### ruff runs with `ALL` selected

Rules are subtracted, not added, and every ignore in `pyproject.toml` carries a
comment saying why. An opt-in rule list silently misses everything added
upstream; starting from `ALL` means new rules arrive on their own.

If a rule is wrong for one file, add a per-file ignore with a reason. If it is
wrong everywhere, argue it in the PR.

### The layer contract is enforced, not documented

`lint-imports` checks that higher layers import lower ones and never the
reverse. It found two real inversions the moment it was extended to cover every
module — see [the M4 findings](../m4-findings.md).

It cannot express *lazy* imports: a function-body `import torch` counts as a
dependency statically. Those properties are asserted by
`tests/unit/test_lazy_imports.py`, which measures the real behaviour in a
subprocess instead of approximating it.

### Generated documentation

`docs/reference/errors.md`, `events.md` and `profiles.md` are generated. CI
regenerates and diffs them, so a new error code without its documentation fails
the build.

```bash
just docs-gen
```

## The docs site

```bash
uv sync --group docs
uv run mkdocs serve
```

Builds in strict mode: a dead link is a build failure, not a warning.

## The GPU host

Tests that need an accelerator run on the project's own machine rather than a
laptop or a free CI runner. The wrapper that drives it is local to the
maintainer's machine and not in the repository. Nothing in the loop above
depends on it: see
[what a clone cannot run](testing.md#what-a-clone-cannot-run) for which layers
it covers.
