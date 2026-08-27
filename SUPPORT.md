# Support

## Where to ask

| | |
|---|---|
| **Something is broken** | [Open an issue](https://github.com/batuhanzorbeyzengin/rebasis/issues/new/choose) — there is a bug template, a feature template and one for requesting a store backend |
| **Something is confusing** | [Start a discussion](https://github.com/batuhanzorbeyzengin/rebasis/discussions), or open an issue; a question that needed asking is usually a documentation bug |
| **A security problem** | **Not an issue.** Follow [SECURITY.md](SECURITY.md) — private advisory, and please leave corpus content out of it |
| **You ran `migrate` on something real** | An issue, either way. That is the most useful thing anybody can contribute right now |

## Attach this

`rebasis doctor --json` is the thing to put in a bug report. It lists the
backends, embedders and devices rebasis can see on your machine, and with
`--store <uri>` it adds what it can learn about a live index — read-only in every
path. It carries no document text, no vectors and no credentials, by
construction.

Also useful: what you ran, what you expected, and the error code if there was
one. Every error carries a stable `RB-Exxxx` code, and the
[error reference](https://batuhanzorbeyzengin.github.io/rebasis/reference/errors/)
explains each.

## What to expect

**There is one maintainer.** What follows is an intention so that silence can be
read correctly. It is not a service-level agreement, and nothing here is
contractual.

| | |
|---|---|
| First response to an issue | Usually within a week |
| A security report | Acknowledged within 3 working days — see [SECURITY.md](SECURITY.md) |
| A fix | No date is promised. Data-loss bugs come first, then correctness, then everything else |
| A pull request | Reviewed within two weeks, or you get told why not |

If an issue goes quiet for two weeks, a comment saying so is welcome rather than
rude. It has most likely been missed.

## What this project does not have

Stated plainly, because finding out later is worse.

- **No commercial support, and no paid tier.** There is nobody to escalate to.
- **No SLA.** See above.
- **No long-term support branch.** While the project is 0.x, only the latest
  release receives fixes. See [stability and support](https://batuhanzorbeyzengin.github.io/rebasis/stability/).
- **A bus factor of one.** This is the honest risk of depending on rebasis, and
  it is not a risk any amount of documentation removes. Two things reduce it: the
  licence is Apache-2.0, so a fork is always available to you; and the design
  decisions are written down in `docs/adr/` with the measurement behind each,
  so somebody picking it up does not have to re-derive them.

  **A co-maintainer would be welcome.** The `0.2` section of the
  [roadmap](ROADMAP.md) is the best place to start: every item there is a
  connection between two pieces that already exist and already have tests.

## Paying for a feature

There is no arrangement for this and no invoice to send. What does move work
forward, in order:

1. **A measurement.** This project changes its mind on evidence and has done so
   twice on the decision rule. A run that contradicts something in `docs/` is
   worth more than a feature request.
2. **A pull request**, with the contract suite passing. Adding a store or an
   embedder is three steps — see [CONTRIBUTING.md](CONTRIBUTING.md).
3. **A real corpus somebody actually migrated.** See the roadmap for why this one
   is the bottleneck.
