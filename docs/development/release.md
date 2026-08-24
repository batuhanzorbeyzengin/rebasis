# Releasing

The version is not chosen by a person. Commit messages choose it, the git tag
carries it, and `hatch-vcs` derives the package version from the tag — so there
is no version string in any file that can disagree with another.

## Conventional Commits

```
feat(store): add sqlite-vec backend             → MINOR
fix(probe): correct the ARR denominator         → PATCH
perf(core): avoid a copy in the Procrustes fit  → PATCH
feat(core)!: change the decision thresholds     → MAJOR
docs: … test: … chore: … ci: …                  → no release
```

A `!` after the type, or a `BREAKING CHANGE:` footer, triggers MAJOR. The scope
is the module name and it groups the changelog on its own.

Merges are squashed, so **the PR title is the commit message**. CI checks it.

## News fragments

A commit subject says what changed. It does not say what that means for someone
upgrading, and six months later only the second is useful. So every user-visible
change adds a file to `changelog.d/` in the same PR:

```markdown
<!-- changelog.d/142.behaviour.md -->
`probe` now compares bridging against keeping the current model, not only
against a full reindex. Measured on BEIR/scifact: bridging recovered 0.903 of a
reindex while keeping the current model gave 0.944.
```

Types, in the order they appear: `behaviour`, `removed`, `added`, `fixed`,
`performance`, `docs`. **Behaviour changes lead**, because a moved threshold is
the thing a reader needs before anything else.

```bash
uv run towncrier build --draft    # preview without writing
```

## 0.x means MINOR can break

Read `0.1 → 0.2` the way you would read a major bump. The API, the `.rbs`
format and the decision thresholds may all change before 1.0. `major_on_zero`
is off so the release tool cannot promote a breaking change to 1.0 by itself —
that is a decision, not a consequence.

## Four independent version axes

| Axis | Source | Changes when |
|---|---|---|
| Package | git tag | Any release |
| `.rbs` schema | `CURRENT_SCHEMA` | The adapter file format changes |
| Manifest schema | `SCHEMA_VERSION` | The SQLite schema changes |
| Metric version | `METRIC_VERSION` | A metric's definition changes |

They move independently on purpose. A package release does not invalidate a
stored adapter, and a metric change does not require a schema migration — but a
recorded decision has to know which metric definition produced it, or a replay
compares two different quantities and calls the difference a regression.

## Cutting a release

The `Release` workflow is manual (`workflow_dispatch`). Run it with
`dry_run: true` first to see the version the commits imply.

Before anything is tagged it gates on:

1. Every static gate — ruff, format, mypy, import-linter
2. The full suite except what needs the host
3. Generated catalogues match the code
4. The docs build in strict mode
5. `uv build`, and **the sdist installs and imports**

That last one is there because a working wheel and a broken sdist is a real and
common failure, and nothing else in CI exercises the sdist.

Then it tags, pushes, rebuilds at the tagged version — the earlier build predates
the tag, so its version is wrong — and publishes through PyPI Trusted Publishing.
No API token exists to leak.

## Before a release, on the host

Nightly runs are advisory; before a release they block. On the host: the whole
suite, then `-m gpu` for device parity, then `-m slow` for the golden corpora
and the macro budgets.

Device parity must be green on every available device. Numerical differences
between devices are acceptable; **a different decision is not**.

## A deliberate slowdown

Allowed, and written down. A change that trades speed for accuracy goes in the
changelog with its reasoning and the baseline is updated explicitly. A silently
accepted regression is not allowed — every accepted one raises the baseline, and
they accumulate faster than anyone expects.
