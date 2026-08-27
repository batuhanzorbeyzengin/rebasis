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

The `Release` workflow is manual (`workflow_dispatch`) and takes a `mode`.

| mode | What it does | What it can break |
|---|---|---|
| `dry-run` | Prints the version the commits imply and the changelog that would be written | Nothing. It exits before anything moves. |
| `rehearse` | Assembles, commits, tags and builds — all on the runner — then uploads to **TestPyPI** | Nothing outside the runner. The job is not granted `contents: write`. |
| `release` | The same, pushed, against **PyPI** | Everything. A published version cannot be replaced. |

Run them in that order. `dry-run` answers "what version is this", `rehearse`
answers "does the machinery work", and only `release` is irreversible.

**What a rehearsal catches that a dry run cannot:** that towncrier assembles
cleanly, that `hatch-vcs` reads the tag, and that Trusted Publishing is
configured for this repository. It needs a pending publisher registered on
TestPyPI against the `testpypi` environment, the same shape as the PyPI one.

A rehearsal commits before tagging rather than building the tree as it stands.
That is not tidiness: `hatch-vcs` appends a local version segment to a build made
from uncommitted changes, and PyPI rejects local version identifiers — so a
rehearsal from a dirty tree fails at the upload for a reason that has nothing to
do with the release. Both publishing jobs check the built filenames for that
segment and name it.

Before anything is tagged, every mode gates on:

1. Every static gate — ruff, format, mypy, import-linter
2. The full suite except what needs the host, and except `perf`
3. Generated catalogues match the code, and every arXiv citation names its paper
4. The docs build in strict mode
5. `uv build`, and **the sdist installs and imports**
6. `changelog.d/` is not empty

Item 5 is there because a working wheel and a broken sdist is a real and common
failure, and nothing else in CI exercises the sdist.

Item 2 excludes `perf` for the reason `ci.yml` gives: a shared runner cannot
measure wall clock, and this job used to run exactly the assertions CI had
removed for that — including the one that had already gone red twice on noise.
The `memory` layer still runs, because peak allocation is deterministic and a
release is when the `O(batch × d)` invariant is most worth re-checking.

Item 6 exists because an empty `changelog.d/` means one of two things, and the
second is the one worth catching: either nothing user-visible changed, or
somebody forgot the fragment.

### The changelog is assembled by the release

towncrier owns `CHANGELOG.md`, and the release workflow is what runs it. The
assembly **is** the release commit, so the tag lands on a tree that already
contains the changelog rather than one commit behind it.

This step was missing. `semantic-release` was called with `--no-changelog` and a
comment saying towncrier would do it, and then nothing called towncrier — not
the workflow, not this page, and the `justfile` had only `--draft`, which
previews without writing. A release cut that way ships an unchanged changelog and
leaves every fragment where it was.

### Who decides the version, and who writes the tag

The commits decide. `semantic-release version --print` gives the number and
`--print-tag` gives the tag in the shape `tag_format` asks for; both exit without
changing anything. `git` then places that tag on the release commit.

Splitting it that way keeps one source for the version and one for the tag
format, while removing a dependency on how the release tool behaves when there is
no version file to rewrite — this project has none, because `hatch-vcs` derives
the package version from the tag itself.

Then it pushes, rebuilds at the tagged version — the earlier build predates the
tag, so its version is wrong — and publishes through Trusted Publishing. No API
token exists to leak, and since `pypa/gh-action-pypi-publish` v1.11.0 the
published files carry a PEP 740 attestation with no further configuration.

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
