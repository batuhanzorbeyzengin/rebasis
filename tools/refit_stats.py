"""What ``spikes/continuous_refit.py`` measured, with the nulls it needs.

The question is not "did a refit ever help". Over enough cells something always
helps; what decides whether `--refit` is worth shipping is whether the help
clears the threshold the adoption guard already enforces —
``RefitPolicy.min_improvement``, 0.01 — often enough to be worth the documents it
re-embeds to get there.

So every arm is reported three ways, and the third is the one that decides:

**Median paired gain.** Each cell contributes one difference against its own
baseline, so a corpus that is simply harder cannot move the summary.

**How often it clears the guard.** Not "how often it was positive" — positive is
free. The guard adopts nothing below 0.01, so an arm that wins by 0.002 in every
cell would ship a flag that never fires, which is worse than not shipping it.

**A paired randomisation test over the cells**, Holm-corrected across the arms,
because a sweep of eight arms tested one at a time will find something.

Rows are grouped by the **original fit budget**, and that grouping is the
finding rather than a convenience: a refit is competing with the adapter it
would replace, so what it has to beat is that adapter's own budget.

    python tools/refit_stats.py reports/refit/rows.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

#: What `RefitPolicy` adopts above. Anything below this is measured and then
#: declined, so it is the line the report is written against.
MIN_IMPROVEMENT = 0.01

PERMUTATIONS = 20_000
PERMUTATION_BLOCK = 2048


class Arm(NamedTuple):
    """One arm's paired differences against its own cells' baselines."""

    name: str
    gains: np.ndarray


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the order given.

    The same implementation `tools/band_stats.py` uses and for the same reason:
    uniformly more powerful than Bonferroni, resting on the same assumption,
    which is none.
    """
    total = len(p_values)
    order = sorted(range(total), key=lambda i: p_values[i])
    adjusted = [0.0] * total
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def randomization_p(difference: np.ndarray, *, permutations: int, seed: int) -> float:
    """Paired Fisher randomisation test over the cells.

    Under the null the refit and the adapter it would replace are the same
    system, so which of a cell's two scores came from which is arbitrary.

    ``(b + 1) / (B + 1)`` rather than ``b / B``: with finitely many draws the
    uncorrected estimator can return zero, which asserts more than the sample
    supports and gives Holm nothing to multiply.
    """
    generator = np.random.default_rng(seed)
    observed = abs(float(difference.mean()))
    size = difference.size
    at_least = 0
    drawn = 0
    while drawn < permutations:
        block = min(PERMUTATION_BLOCK, permutations - drawn)
        signs = generator.integers(0, 2, size=(block, size), dtype=np.int8) * 2.0 - 1.0
        at_least += int((np.abs(signs @ difference) / size >= observed - 1e-12).sum())
        drawn += block
    return (at_least + 1) / (permutations + 1)


def load(path: Path) -> list[dict[str, Any]]:
    """Every row of a JSONL file."""
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def arms_for(rows: list[dict[str, Any]]) -> list[Arm]:
    """Every arm present in every row, as paired differences.

    An arm a cell did not run — a small corpus whose sweep was clamped — is
    absent from that cell rather than zero. Comparing arms over different cell
    sets would make the columns incomparable, so each arm carries its own count
    and the report prints it.
    """
    collected: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for name, gain in row["gains"].items():
            collected[name].append(float(gain))
    return [
        Arm(name, np.array(values, dtype=np.float64)) for name, values in sorted(collected.items())
    ]


def report(rows: list[dict[str, Any]], *, seed: int) -> str:
    """One table per original fit budget."""
    lines: list[str] = []
    by_budget: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_budget[int(row["fit_pairs"])].append(row)

    for budget in sorted(by_budget, reverse=True):
        group = by_budget[budget]
        arms = arms_for(group)
        raw = [randomization_p(arm.gains, permutations=PERMUTATIONS, seed=seed) for arm in arms]
        adjusted = holm(raw)

        lines.append("")
        lines.append(f"## The adapter was fitted on {budget:,} pairs — {len(group)} cells")
        lines.append("")
        lines.append("| arm | cells | median gain | > 0 | clears 0.01 | p (Holm) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for arm, p_value in zip(arms, adjusted, strict=True):
            clears = int((arm.gains > MIN_IMPROVEMENT).sum())
            lines.append(
                f"| `{arm.name}` | {arm.gains.size} | {np.median(arm.gains):+.4f} | "
                f"{int((arm.gains > 0).sum())}/{arm.gains.size} | "
                f"{clears}/{arm.gains.size} | {p_value:.3g} |"
            )

        best = max(arms, key=lambda a: int((a.gains > MIN_IMPROVEMENT).sum()))
        share = int((best.gains > MIN_IMPROVEMENT).sum()) / best.gains.size
        lines.append("")
        lines.append(
            f"Best by the guard's own threshold: `{best.name}`, clearing 0.01 in "
            f"{share:.0%} of cells at a median of {np.median(best.gains):+.4f}."
        )
        # A randomisation test cannot report a p-value below `1 / (B + 1)`, and
        # Holm then multiplies that by the number of arms. Where every arm sits
        # on that product the column is reporting the test's resolution rather
        # than the arms' evidence, and saying so is the difference between a
        # floor and a finding.
        floor = len(arms) / (PERMUTATIONS + 1)
        if all(p <= floor + 1e-12 for p in adjusted):
            lines.append("")
            lines.append(
                f"Every p-value is at the test's resolution floor "
                f"({floor:.2g} = {len(arms)} arms / {PERMUTATIONS:,} permutations). "
                "The differences are real and the test cannot say how real; what "
                "decides here is the effect size beside it, not the p-value."
            )
    return "\n".join(lines)


def by_order(rows: list[dict[str, Any]], *, seed: int) -> str:
    """The same arms split by migration order.

    `refit.py`'s own caveat is about order — pairs accumulated during a
    migration come from records processed in priority order, not at random — so
    the split is the caveat's test rather than an extra cut of the data.
    """
    lines = ["", "## By migration order", ""]
    lines.append("| fit pairs | order | best arm | median gain | clears 0.01 |")
    lines.append("|---|---|---|---:|---:|")
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["fit_pairs"]), str(row["order"]))].append(row)
    for (budget, order), group in sorted(grouped.items(), reverse=True):
        arms = arms_for(group)
        if not arms:
            continue
        best = max(arms, key=lambda a: int((a.gains > MIN_IMPROVEMENT).sum()))
        lines.append(
            f"| {budget:,} | {order} | `{best.name}` | {np.median(best.gains):+.4f} | "
            f"{int((best.gains > MIN_IMPROVEMENT).sum())}/{best.gains.size} |"
        )
    _ = seed
    return "\n".join(lines)


def source_contrast(rows: list[dict[str, Any]], *, seed: int) -> str:
    """``remaining`` against ``migrated`` at the same K, on the same cells.

    This is the contrast the arm names exist for, and reading the two columns of
    the tables above against each other would not give it: they are summarised
    over cells separately, and the difference of two medians is not the median
    of the difference. Held at equal K, the only thing that varies is **which
    documents the pairs came from** — pair count, corpus and model pair are all
    fixed — so a difference here is the corpus' own heterogeneity and nothing
    else.

    Under ``--order arrival`` it is the whole experiment: ``migrated`` is the
    domain the adapter was already fitted on and ``remaining`` is the one that
    arrived afterwards.
    """
    lines = ["", "## Where the pairs came from, at equal K", ""]
    lines.append("| fit pairs | mode | K | cells | median remaining - migrated | p |")
    lines.append("|---|---|---:|---:|---:|---:|")
    grouped: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    for row in rows:
        gains = row["gains"]
        for name, gain in gains.items():
            mode, source, budget = name.split(":")
            if source != "remaining":
                continue
            twin = f"{mode}:migrated:{budget}"
            if twin in gains:
                grouped[(int(row["fit_pairs"]), mode, int(budget))].append(
                    float(gain) - float(gains[twin])
                )
    raw: list[float] = []
    keys = sorted(grouped, reverse=True)
    for key in keys:
        differences = np.array(grouped[key], dtype=np.float64)
        raw.append(randomization_p(differences, permutations=PERMUTATIONS, seed=seed))
    for key, p_value in zip(keys, holm(raw), strict=True):
        budget, mode, k = key
        differences = np.array(grouped[key], dtype=np.float64)
        lines.append(
            f"| {budget:,} | `{mode}` | {k:,} | {differences.size} | "
            f"{np.median(differences):+.4f} | {p_value:.3g} |"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """The command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", type=Path, help="JSONL written by spikes/continuous_refit.py")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print both views."""
    args = build_parser().parse_args(argv)
    rows = load(args.rows)
    if not rows:
        print("no rows")
        return 1
    print(f"{len(rows)} cells from {args.rows}")
    print(report(rows, seed=args.seed))
    print(by_order(rows, seed=args.seed))
    print(source_contrast(rows, seed=args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
