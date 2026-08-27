# Maintainers

| | | |
|---|---|---|
| Batuhan Zorbey Zengin | [@batuhanzorbeyzengin](https://github.com/batuhanzorbeyzengin) | everything |

That is the whole table, and the rest of this file is about what follows from
that rather than about who is on it.

## One maintainer is the largest risk in depending on this

Stated here rather than left for a procurement review to discover.
[OpenSSF's own guidance](https://github.com/ossf/wg-best-practices-os-developers)
names a single maintainer as a risk signal, and it is right to. It means every
one of these:

- Nobody reviews a change but its author. OpenSSF Scorecard's **Code-Review**
  check scores this low by construction, and no configuration changes that — see
  [SECURITY.md](SECURITY.md).
- A response time depends on one person's week. [SUPPORT.md](SUPPORT.md) says
  what to expect and what is not promised.
- If that person stops, the project stops.

## What reduces it, and what does not

**Apache-2.0.** A fork is always available to you, without asking. That is the
only guarantee here that does not depend on anybody's continued attention.

**The reasoning is written down.** `docs/adr/` holds eleven decisions that would
otherwise be re-argued, each with the measurement behind it, and `docs/`
generally records the runs that produced every number this project claims.
Somebody picking it up does not have to re-derive the decisions — which is the
difference between a project that can be taken over and one that can only be
rewritten.

**The gates are mechanical, not cultural.** `ruff` with every rule enabled,
`mypy --strict`, a layer contract in `import-linter`, docstring coverage, per-module
coverage floors on the modules where a bug costs data, and a contract suite every
backend runs. A new maintainer inherits the standard rather than having to
reconstruct it from taste.

**What does not reduce it:** a governance document. A file describing how
decisions are made between multiple maintainers, written by the only maintainer,
describes nothing. It will be written when there is something to describe.

## Becoming one

Genuinely open, and the fastest route is the `0.2` list in
[ROADMAP.md](ROADMAP.md) — every item there is a connection between two pieces
that already exist and already have tests, so the work is bounded and the tests
tell you when you are done.

What a maintainer is expected to hold to is already in
[CONTRIBUTING.md](CONTRIBUTING.md), and one rule matters more than the rest:
**nothing is claimed that has not been measured.** Where something is expected
rather than measured, it says so. That is the project's whole disposition and it
is the one thing a second maintainer would have to share.

## Releases

Cut manually from the `Release` workflow, by a maintainer. The commits choose the
version; nobody types one. See
[the release page](https://batuhanzorbeyzengin.github.io/rebasis/development/release/)
for the three modes and what each can break.

There is no fixed cadence. A release happens when there is something in
`changelog.d/` worth a reader's time.
