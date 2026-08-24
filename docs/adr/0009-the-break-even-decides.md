# 9. The break-even decides; the bands describe

**Status:** Accepted · **Date:** 2026-08 · **Evidence:** `docs/bridge-band.md`, sections 2 and 4

## Decision

When `upgrade_gain` is measurable, `bridge_advantage = ARR × upgrade_gain`
decides **whether** to bridge. The ARR bands decide **which** bridging answer to
give. Without a measurable upgrade, the bands decide alone and the result is
marked provisional.

## Context

The decision rule places ARR in one of four bands and recommends from that. ARR
is measured against the oracle — a full reindex — so it answers "how much of a
rebuild does this adapter recover". That is not the question a user is deciding:
they are choosing between bridging and doing nothing.

ADR 5 added the comparison against the current model, and it caught the case
where bridging lost. It did not catch the reverse.

## Evidence

Fifteen runs over five corpora searched with queries real people typed —
StackExchange questions and financial questions, 222,680 documents and 5,761
human-judged queries — scored with ranx.

| criterion | agrees with the measured outcome |
|---|---|
| ARR bands alone | 10/15 |
| `bridge_advantage > 1` | **14/15** |

Every disagreement went one way. The four runs where bridging genuinely helped:

| corpus | ARR | gain | advantage | banded as | measured |
|---|---|---|---|---|---|
| programmers | 0.55 | 1.96 | 1.071 | `caution` | **+4.2%** |
| english | 0.56 | 1.89 | 1.052 | `full_reindex` | **+3.8%** |
| gaming | 0.73 | 1.52 | 1.108 | `caution` | **+7.4%** |
| fiqa | 0.51 | 2.44 | 1.247 | `full_reindex` | **+16.0%** |

The bands rejected all four, including a 16% improvement.

The mechanism is the anti-correlation in `docs/bridge-band.md`, section 3: gain and
retention pull against each other at −0.958. A large upgrade means the old model
was weak, a weak source space carries less recoverable structure, and the adapter
therefore lands at a **low** ARR precisely when the upgrade is worth having. The
bands read low ARR as "an adapter cannot bridge this" when what it actually meant
was "there was a lot to bridge".

## Consequences

- T1 agreement rises from 10/15 to **13/15**, and the four genuine wins go from
  0/4 recommended to **4/4**.
- The two remaining misses are a −0.7% run — a tie in any reading — and one whose
  break-even fell inside the noise band, where the result now says the two
  options cannot be told apart rather than picking one.
- The bands keep a real job: `bridge_sufficient` when little of a reindex is
  left on the table, `bridge_and_migrate` when a lot is. That is what ARR is
  genuinely good at.
- Nothing changes at T0. There is no break-even to consult, the bands decide
  alone, and the result is provisional.

## Alternatives

**Widen the bands.** Rejected: the bands are not mis-tuned, they are answering a
different question. No threshold on "fraction of a reindex recovered" can express
"better than what I have".

**Report both and let the user choose.** Rejected. The report already showed both
numbers and led with the band; a reader who has to multiply two figures in their
head to find out whether the headline is right is not being helped.

**Replace ARR with the advantage everywhere.** Rejected. ARR is what the decision
bands were calibrated against, what every recorded measurement holds, and the
right answer to "how much am I leaving on the table". Both numbers are worth
having; only one of them is the decision.
