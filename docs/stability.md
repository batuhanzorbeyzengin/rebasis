# Stability and support

What may change, what may not, and how much warning you get. Written down
because "will this break my build" is the first question anybody asks about a
dependency, and the honest answer is longer than yes or no.

## The short version

| | |
|---|---|
| Version today | **0.1** |
| Read `0.1 → 0.2` as | a major bump |
| Python | 3.12 and 3.13 |
| Operating systems | Linux is gated in CI; macOS is the maintainer's own machine; Windows is untested |
| Maintainers | one |
| Support commitment | best effort, stated in [SUPPORT.md](https://github.com/batuhanzorbeyzengin/rebasis/blob/main/SUPPORT.md) |

## What 0.x means here

[Semantic Versioning](https://semver.org/) promises nothing below 1.0 — "anything
MAY change at any time" is the specification's own wording. This project does not
try to promise more than that yet, and says so rather than implying a stability
it has not earned.

What it does commit to, from now:

- **A breaking change gets a MINOR bump**, never a PATCH. `major_on_zero` is off
  in the release configuration so the tooling cannot promote a breaking change to
  1.0 on its own — that is a decision, not a consequence.
- **A breaking change is in the changelog with its reason.** Behaviour changes
  lead every release, above the feature list, because a moved decision threshold
  is the thing a reader upgrading needs first.
- **A moved number is a breaking change.** If a decision threshold or a metric
  definition moves, your `probe` may return a different answer on the same
  corpus. That is treated as breaking even though no signature changed, because
  it is what actually breaks a user.

## The public surface

Three things carry the compatibility promise. Everything else is internal and may
move without notice, whatever its name suggests.

**The Python API.** What `rebasis` and `rebasis.serve` export, and the functions
named on the [API reference](reference/api.md). `rebasis.Bridge` is the one most
people hold, and the one that changes most reluctantly — it sits in the hot path
of somebody's application.

**The CLI.** Command names, option names, exit codes and the shape of `--json`
output. Exit codes are a contract for anyone gating a deploy on them:

| | |
|---|---|
| `0` | Success |
| `1` | Unexpected — a bug in rebasis |
| `2` | Usage — a bad flag or a missing argument |
| `3` | Domain — the run completed and the answer is "no" |
| `130` | Aborted |

Adding a key to `--json` is not breaking. Removing one, renaming one, or changing
its type is.

**The stable error codes.** `RB-E3002` means the same thing forever. A code is
never removed or repurposed, only marked deprecated, because scripts and
dashboards key off them. Every code is in the
[error reference](reference/errors.md).

Not covered: anything under a leading underscore, anything not in this list,
`spikes/`, and `tools/`.

## Deprecation, from 1.0 onwards

Once the API and the `.rbs` format stop moving — see the
[roadmap](https://github.com/batuhanzorbeyzengin/rebasis/blob/main/ROADMAP.md)
for what 1.0 needs — this is the commitment:

1. A deprecated name keeps working for **at least two minor releases**, and never
   less than **six months**, whichever is longer.
2. It raises a `DeprecationWarning` naming the replacement from the release that
   deprecates it.
3. The changelog entry says what to change and why it moved.
4. It is removed no earlier than the next MAJOR release.

The suite already treats a `DeprecationWarning` **from rebasis' own code** as an
error, so the project cannot forget to migrate off something it deprecated
itself.

Before 1.0, that window is an intention rather than a promise, and the reason is
worth stating: two minor releases at 0.x may be two weeks apart. Pin an exact
version if that matters to you.

## Python versions

rebasis follows [SPEC 0](https://scientific-python.org/specs/spec-0000/), the
scientific-Python support window that replaced NEP 29: **a Python release is
dropped three years after its initial release.**

| | Released | Dropped under SPEC 0 |
|---|---|---|
| 3.11 | Oct 2022 | Q4 2025 — already dropped |
| 3.12 | Oct 2023 | **Q4 2026** |
| 3.13 | Oct 2024 | Q4 2027 |
| 3.14 | Oct 2025 | Q4 2028 |

The floor is 3.12 because 3.11 passed its mark in Q4 2025. **3.12 reaches its own
mark in Q4 2026**, and the floor will rise to 3.13 then rather than drifting.

A trove classifier is a claim, so the package declares only the versions CI runs.

## Dependency floors

Every lower bound in `pyproject.toml` is the oldest version the suite actually
passes on, checked by running it there rather than by guessing. Three of them
moved once they were checked, and the comment above each records what it was
that broke.

That makes rebasis' floors **older than SPEC 0 would require** — it asks for a
two-year window on numpy, scipy and scikit-learn, and these go further back. That
is deliberate: a floor that is tested is worth more than a floor that is
fashionable, and raising one withdraws support from users who have no other
reason to upgrade.

The claim is kept true by a CI job. `lowest direct dependencies` resolves with
`--resolution lowest-direct` and runs the suite against the declared bounds, so a
floor that stops working goes red on the pull request that breaks it rather than
in somebody's install.

## What is tested, and what is only claimed

Being specific about this is the point of the page.

| | Gated on every pull request | Run by hand, on the project's own host | Not tested |
|---|---|---|---|
| **OS** | Linux; macOS for the layers a core install can run | Linux | **Windows** |
| **Python** | 3.12 | 3.12 | 3.13 in CI |
| **Stores** | pgvector, Chroma, LanceDB, sqlite-vec, Qdrant, FAISS, in-memory — the same contract suite and a migrate-and-rollback cycle for each | as CI | Qdrant in server mode beyond the local-mode suite |
| **Embedders** | in-memory and precomputed | sentence-transformers, fastembed on real models | ollama, llama-cpp, hosted OpenAI-compatible endpoints |
| **Scale** | hundreds of records | up to 100,000 for index health | **millions — see below** |

**The middle column is run by hand, and used to say "nightly".** It was
describing a workflow that drives the maintainer's GPU host through scripts
which are not in a clone — so it was never committed, and a `schedule:` trigger
only fires from the default branch. Nothing has ever run it. Those numbers exist
because somebody ran them before a release, which is a person and not a gate.

**macOS is back in CI, for part of the suite.** The full suite could not stay
there: `faiss-cpu` and `torch` each link their own OpenMP runtime on macOS, and
a process holding both aborts before either library does any work. The
`core install` job installs neither, so it runs `unit`, `property` and
`contract` on `macos-latest` as well as `ubuntu-latest` — which covers the
storage layer, where the platforms actually differ. What it does not cover is a
real store on macOS: with no extras installed, the five live backends skip and
only the in-memory one runs — pgvector included, which needs a server the macOS
job does not stand up.

**pgvector is gated against a real PostgreSQL.** It is the only backend that
needs something outside the process, and CI runs a `pgvector/pgvector` image
pinned by digest as a service on both the coverage job and the lowest-direct
job — the second so the `pg8000` floor is found by running the suite against
it, the way the chroma, qdrant and faiss floors were. Without the service the
pgvector layer would skip, and a suite that skips a backend reports the same
green summary as one that ran it.

**Windows has never run.** The storage layer has documented Windows-specific
behaviour — `os.replace` is used precisely because `os.rename` is not atomic over
an existing file there — and nothing exercises it.

**Nobody has pointed `migrate` at an index they could not rebuild.** That is the
single largest gap between this and something to trust unsupervised, and no
amount of unit testing closes it. Take a backup rebasis is not part of, and try
`--limit` on a slice first.

## The four version axes

The package version is not the only one, and they move independently on purpose.
The [release page](development/release.md#four-independent-version-axes) has the
table: a package release does not invalidate a stored adapter, and a metric
change does not require a schema migration.

## Getting a warning before it lands

- The [changelog](https://github.com/batuhanzorbeyzengin/rebasis/blob/main/CHANGELOG.md)
  leads with behaviour changes.
- Watch **Releases** on the repository rather than all activity.
- Pin an exact version and read the changelog when you move; that is what the
  changelog is for.
