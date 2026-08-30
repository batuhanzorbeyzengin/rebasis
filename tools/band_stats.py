"""What the counts in ``docs/bridge-band.md`` are worth, tested rather than quoted.

That document scores the decision rule by counting: the break-even "predicted the
outcome" 61 times out of 62. A count with no null behind it is not evidence, and
this is the tool that supplies the nulls. Three things it does, in order of how
much they change the reading:

**It refuses to test an identity.** ``bridge_advantage`` is ``ARR x
upgrade_gain``. Read off one run's own scores that is ``(bridged / reindex) x
(reindex / status_quo)``, which is ``bridged / status_quo`` — and the outcome
being predicted is ``bridged > status_quo``. The two are the same inequality
written twice, so agreement is arithmetic and not a result. ``tools/
bridge_band_report.py --view summary`` computes exactly this pair, which is why
it reports a perfect score on every file it is given. The tool detects the
degeneracy and prints the count with no p-value beside it, because a p-value
there would launder an identity into a finding. ADR 4 is the precedent: a
degenerate case made the wrong code look right, and the fix was to test on data
where the two sides can differ.

**It supplies the informative null.** Against ``H0: p = 0.5`` almost any count
looks spectacular, but a coin is not the alternative anyone would use. The rule
emits a binary call about an outcome that is overwhelmingly one-sided — on these
corpora bridging beats doing nothing in a small minority of runs — so the
baseline to beat is a rule that ignores its inputs and always answers with the
majority class. That base rate is derived from the same rows rather than assumed,
and both nulls are reported so the difference between them is visible.

**It replaces the proportion where a proportion is the wrong summary.** The rule
thresholds a continuous quantity and already flags its own borderline cases, so a
single accuracy throws away most of what was measured. Two things are reported
beside it: the split between runs the rule calls borderline and runs it calls
decisive, and the rank correlation between the estimate and the margin it was
predicting. The correlation is the one that survives an outcome this one-sided —
accuracy cannot separate a good rule from a constant when 95% of outcomes agree,
and a rank correlation can.

``--view arrangement`` is M2 of `docs/_local` item 1, and it exists because the
same mistake can be made twice. ``probe`` now recommends a two-stage arrangement
when ``cascade_advantage`` clears the band, and that quantity has to be shown to
be a *prediction* rather than the outcome written twice — the identity check is
printed first for that reason. The rest is the null that matters: the
arrangement wins on most runs, so "always recommend cascade" is a strong
baseline, and the tests that survive a one-sided outcome are a rank correlation
and a permutation test over which runs the rule selected.

``--view paired`` is the other half. Means cannot be tested against each other,
so the harness now writes one JSON sidecar of per-query scores per run and the
row names it in ``per_query``. Given those, each run gets a paired randomisation
test of ``bridged`` against ``status_quo`` on the same queries, and the per-corpus
p-values are corrected across the whole table with Holm. Rows without a sidecar
are named and skipped rather than dropped silently.

**Why the randomisation test is implemented here and not called from ranx.**
``ranx.compare`` is the obvious route and does not fit: it takes ``Qrels`` and
``Run`` objects — full ranked lists and judgements — and a sidecar carries
per-query metric values, which is the thing being tested and not the material to
rebuild a run from. The function that does take score arrays,
``ranx.statistical_tests.fisher_randomization_test``, is the right test with the
right pairing, and it is ``@njit(parallel=True)`` over a single global seed: the
same call with the same ``random_seed`` returns 0.948, 0.945, 0.948, 0.942 on
four consecutive runs. A p-value that moves in the third decimal between runs
cannot be corrected across a table of thirty or quoted in a document. The test
below is the same statistic — permute each query's pair independently, count
permuted mean differences at least as large as the observed one — drawn from a
seeded ``numpy`` generator, and it agrees with ranx to Monte Carlo error. The
seed and the permutation count are printed with the results because they are part
of the number.

Two conventions worth naming, both chosen against overclaiming:

*Clopper-Pearson leads, Wilson is printed beside it.* At 61 of 62 the normal
approximation is worthless — at 62 of 62 it produces an interval of zero width —
so neither is a candidate. Wilson's coverage is closer to nominal on average;
Clopper-Pearson's is never *below* nominal, at the cost of being wider. A
document whose borderline band was widened from +-0.005 to +-0.025 rather than
claim a precision the measurement lacked should take the interval that cannot
under-cover, and see the other one next to it.

*The randomisation p-value is ``(b + 1) / (B + 1)``.* ranx reports ``b / B``,
which can return exactly zero — a claim no finite number of permutations
supports. The added pseudo-count is the standard correction and it keeps every
p-value strictly positive, which Holm needs.

Usage::

    python tools/band_stats.py reports/band/cascade.jsonl reports/band/mmteb.jsonl
    python tools/band_stats.py reports/band/protocol.jsonl --view paired
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Default seed for the randomisation test. Printed with every result: a
#: randomisation p-value without its seed is not reproducible.
DEFAULT_SEED = 20260825

#: Permutations per test. 10,000 puts the resolution floor at 1e-4, which is
#: below anything Holm over a table of fifty will leave significant.
DEFAULT_PERMUTATIONS = 10_000

#: Permutations drawn at once. Fixed, because the result depends on it: the
#: generator is consumed in this order and a different block size is a different
#: draw. 256 x 3,000 queries is a 6 MB array, which is the point of chunking.
PERMUTATION_BLOCK = 256

#: Mirrors ``rebasis.probe.decision.BORDERLINE_BAND``. Restated rather than
#: imported because this tool reads report files and must not depend on the
#: package; if that constant moves, this one has to be moved with it.
BORDERLINE_BAND = 0.025

#: How close two ratios have to be before the tool calls them the same number.
#: Scores are rounded to 4 decimals in the rows, so ratios of them cannot be
#: compared any tighter than this.
IDENTITY_TOLERANCE = 5e-4


def load(path: Path) -> list[dict[str, Any]]:
    """Read the harness's append-only rows."""
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def protocol_of(row: dict[str, Any]) -> str:
    """Which protocol produced a row; rows predating the flag are ``t1-judged``."""
    return str(row.get("protocol", "t1-judged"))


def short(model_id: str) -> str:
    """The model name a table column has room for."""
    return model_id.rpartition("/")[2].removesuffix("-en-v1.5").removesuffix("-v2")


def corpus_label(name: str) -> str:
    """Short, stable name for a corpus, whichever loader it came from."""
    label = name.removeprefix("mmteb:mteb/").removeprefix("beir/")
    label = label.removeprefix("cqadupstack/").removesuffix("/test")
    return label.replace("_test_top_250_only_w_correct-v2", "-HN")


def identity(row: dict[str, Any]) -> tuple[str, str, str, str, bool]:
    """What makes two rows the same run.

    Needed because the published files overlap: ``cascade.jsonl`` contains every
    row of ``heldout.jsonl`` and every row of ``beir.jsonl``. Counting a run
    twice would narrow every interval in this file by a factor it has not earned.
    """
    return (
        protocol_of(row),
        row["corpus"],
        row["old_model"],
        row["new_model"],
        bool(row.get("self_removal", False)),
    )


def _disagrees(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Do two rows for the same run report different numbers.

    Only the configurations and metrics both rows measured are compared. A run
    re-measured with an extra configuration is still the same run, and a file
    that added ``cascade@200`` to it has not contradicted anything.
    """
    left, right = first["scores"], second["scores"]
    for configuration in set(left) & set(right):
        shared = set(left[configuration]) & set(right[configuration])
        if any(left[configuration][name] != right[configuration][name] for name in shared):
            return True
    return False


def dedupe(
    sourced: list[tuple[Path, dict[str, Any]]],
) -> tuple[list[tuple[Path, dict[str, Any]]], int, list[str]]:
    """One entry per run, plus how many repeats were dropped and which disagreed.

    Needed because the published files overlap. A repeat that disagrees is not a
    repeat — it is two measurements of the same thing, and which one the tables
    are built from is a question for whoever made them rather than something to
    settle here by taking the first.
    """
    kept: dict[tuple[str, str, str, str, bool], tuple[Path, dict[str, Any]]] = {}
    dropped = 0
    conflicts: list[str] = []
    for path, row in sourced:
        key = identity(row)
        if key not in kept:
            kept[key] = (path, row)
            continue
        dropped += 1
        if _disagrees(kept[key][1], row):
            conflicts.append(
                f"{corpus_label(row['corpus'])} {short(row['old_model'])}→{short(row['new_model'])}"
            )
    return list(kept.values()), dropped, conflicts


def score(row: dict[str, Any], configuration: str, name: str) -> float | None:
    """One measured number, or ``None`` where the configuration was not run."""
    scores = row["scores"].get(configuration)
    if scores is None:
        return None
    value = scores.get(name)
    return None if value is None else float(value)


@dataclass(slots=True)
class Run:
    """One run, reduced to what a significance test needs from it."""

    corpus: str
    rung: str
    #: Did bridging actually beat keeping the current model, on the headline
    #: metric. This is the thing every predictor below is scored against.
    won: bool
    #: By how much, as a share of the status quo. The continuous quantity the
    #: binary outcome throws away.
    margin: float
    #: ``ARR x upgrade_gain`` with both factors read off this run's own scores.
    #: Algebraically ``bridged / status_quo``; see the module docstring.
    measured: float
    #: The same product as the tool reports it: retention from the adapter fit's
    #: own kNN ground truth, gain from recall rather than from the graded metric.
    #: ``None`` where the row carries no fit summary.
    probe: float | None


def runs_from(rows: list[dict[str, Any]], *, metric: str, k: int) -> list[Run]:
    """Reduce rows to runs, skipping any that did not measure all three configurations."""
    name = f"{metric}@{k}"
    out: list[Run] = []
    for row in rows:
        status_quo = score(row, "status_quo", name)
        bridged = score(row, "bridged", name)
        reindex = score(row, "full_reindex", name)
        if not status_quo or not reindex or bridged is None:
            continue
        recall_quo = score(row, "status_quo", f"recall@{k}")
        recall_reindex = score(row, "full_reindex", f"recall@{k}")
        arr = (row.get("fit") or {}).get(f"arr_r{k}")
        probe = (
            float(arr) * (recall_reindex / recall_quo)
            if arr is not None and recall_quo and recall_reindex is not None
            else None
        )
        out.append(
            Run(
                corpus=corpus_label(row["corpus"]),
                rung=f"{short(row['old_model'])}→{short(row['new_model'])}",
                won=bridged > status_quo,
                margin=(bridged - status_quo) / status_quo,
                measured=(reindex / status_quo) * (bridged / reindex),
                probe=probe,
            )
        )
    return out


def clopper_pearson(successes: int, total: int, *, alpha: float) -> tuple[float, float]:
    """The exact binomial interval — never narrower than its coverage guarantee."""
    low = (
        0.0
        if successes == 0
        else float(stats.beta.ppf(alpha / 2, successes, total - successes + 1))
    )
    high = (
        1.0
        if successes == total
        else float(stats.beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    )
    return low, high


def wilson(successes: int, total: int, *, alpha: float) -> tuple[float, float]:
    """The score interval — closer to nominal coverage on average, and narrower."""
    z = float(stats.norm.ppf(1 - alpha / 2))
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * float(np.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denominator
    return centre - half, centre + half


def binomial_p(successes: int, total: int, p_null: float) -> float:
    """One-sided exact binomial test: is the rule doing better than ``p_null``."""
    return float(stats.binomtest(successes, total, p_null, alternative="greater").pvalue)


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the order given.

    Holm rather than Bonferroni because it is uniformly more powerful and rests
    on the same assumption — none. Bonferroni multiplies every p-value by the
    number of tests; Holm multiplies the smallest by the full count and each
    later one by what is left, so it controls the same family-wise error rate and
    rejects at least as much. There is no case for paying Bonferroni's extra
    conservatism here.
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
    """Paired Fisher randomisation test on per-query differences.

    Under the null the two configurations are the same system, so which of a
    query's two scores came from which is arbitrary and every one of the 2^n sign
    assignments is equally likely. The statistic is the absolute mean difference;
    the p-value is how often a random reassignment reaches the observed one.

    ``(b + 1) / (B + 1)`` rather than ``b / B``: with a finite number of draws
    the uncorrected estimator can return zero, which asserts more than any finite
    sample can support and gives Holm nothing to multiply.
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


def _proportion_row(label: str, successes: int, total: int, *, alpha: float, base: float) -> str:
    low, high = clopper_pearson(successes, total, alpha=alpha)
    w_low, w_high = wilson(successes, total, alpha=alpha)
    coin = binomial_p(successes, total, 0.5)
    informed = binomial_p(successes, total, base)
    return (
        f"| {label:38s} | {successes:2d}/{total:2d} | {successes / total:.4f} "
        f"| {low:.3f}-{high:.3f} | {w_low:.3f}-{w_high:.3f} "
        f"| {coin:9.2e} | {informed:.4f} |"
    )


def outcome_view(runs: list[Run], *, alpha: float, metric: str, k: int) -> str:
    """The proportions, against both nulls, plus the two summaries that beat a proportion."""
    total = len(runs)
    wins = sum(run.won for run in runs)
    majority = max(wins, total - wins)
    base = majority / total
    side = "bridging beats doing nothing" if wins * 2 > total else "bridging loses"

    lines = [
        f"outcome: bridged beat status_quo on {metric}@{k} in {wins} of {total} runs",
        "",
        f"The majority class is “{side}”, at {majority}/{total} = {base:.4f}. A rule that",
        "ignored every input and always answered with it would score exactly that, so it",
        "is the null a decision rule has to beat. H0: p = 0.5 is printed beside it to show",
        "what the choice of null is worth, not because a coin is anyone's alternative.",
        "",
        (
            "| predictor | agrees | share | 95% Clopper-Pearson | 95% Wilson "
            f"| p (H0: 0.5) | p (H0: {base:.4f}) |"
        ),
        "|" + "---|" * 7,
    ]

    measured = sum((run.measured > 1.0) == run.won for run in runs)
    lines.append(
        _proportion_row("break-even, measured (identity)", measured, total, alpha=alpha, base=base)
    )

    scored = [run for run in runs if run.probe is not None]
    if scored:
        agreed = sum(((run.probe or 0.0) > 1.0) == run.won for run in scored)
        lines.append(
            _proportion_row(
                "break-even, probe's estimate", agreed, len(scored), alpha=alpha, base=base
            )
        )
    lines.append(
        _proportion_row("majority-class baseline", majority, total, alpha=alpha, base=base)
    )

    degenerate = sum(abs(run.measured - (1.0 + run.margin)) <= IDENTITY_TOLERANCE for run in runs)
    if degenerate == total:
        lines += [
            "",
            "**The first row is an identity and its p-values are not evidence.** ARR times",
            "upgrade_gain, with both factors read off this run's own scores, is",
            "bridged/status_quo — the same inequality as the outcome it is scored against.",
            f"It matched on {degenerate} of {total} runs because it could not do otherwise.",
            "Read the second row: that break-even is built from the adapter fit's own kNN",
            "retention and from recall, so it can disagree with a graded nDCG outcome, and",
            "does. It is still not independent — its gain comes from the same evaluation —",
            "so treat it as an upper bound on how well the shipped rule does here.",
        ]

    inside = [run for run in runs if abs(run.measured - 1.0) <= BORDERLINE_BAND]
    outside = [run for run in runs if abs(run.measured - 1.0) > BORDERLINE_BAND]
    lines += [
        "",
        f"borderline split, on the ±{BORDERLINE_BAND} band the rule already reports:",
        (
            f"  the rule calls {len(inside)} of {total} borderline; "
            f"{sum(r.won for r in inside)} of those won"
        ),
        f"  it calls {len(outside)} decisive; {sum(r.won for r in outside)} of those won",
    ]

    if scored and len({run.margin for run in scored}) > 1:
        estimate = np.array([run.probe for run in scored], dtype=float)
        margin = np.array([run.margin for run in scored], dtype=float)
        spearman = stats.spearmanr(estimate, margin)
        kendall = stats.kendalltau(estimate, margin)
        lines += [
            "",
            "what the proportion throws away — the estimate against the margin it predicts:",
            f"  Spearman rho = {spearman.statistic:+.3f}  p = {spearman.pvalue:.3g}",
            f"  Kendall tau  = {kendall.statistic:+.3f}  p = {kendall.pvalue:.3g}",
            f"  over {len(scored)} runs. An accuracy cannot separate a real rule from a",
            "  constant when the outcome is this one-sided; a rank correlation can, and it",
            "  is the honest summary of a threshold on a continuous quantity.",
        ]
    return "\n".join(lines)


@dataclass(slots=True)
class Arrangement:
    """One run, reduced to what the arrangement rule needs and is scored on."""

    corpus: str
    rung: str
    #: ``cascade_arr x upgrade_gain`` — the quantity the rule thresholds.
    cascade_advantage: float
    #: ``ARR x upgrade_gain`` at ``k`` — the gate that says the single stage did
    #: not already win. Algebraically ``bridged / status_quo``; that is fine
    #: here, because it is used as a gate and never scored against an outcome.
    bridge_advantage: float
    #: Did the reranked result beat keeping the current model, on the graded
    #: metric at ``k``. The outcome every rule below is scored against.
    won: bool
    #: By how much, as a share of the status quo.
    margin: float


def arrangements_from(
    rows: list[dict[str, Any]], *, metric: str, k: int, depth: int
) -> list[Arrangement]:
    """Reduce rows to the inputs and the outcome of the arrangement rule.

    Every quantity is rebuilt from the harness' own scores rather than read from
    a ``probe`` result, so that the rule is scored against a measurement it did
    not produce. Rows that did not run ``cascade@depth`` are skipped rather than
    defaulted: a missing configuration is not a losing one.
    """
    graded = f"{metric}@{k}"
    out: list[Arrangement] = []
    for row in rows:
        status_quo = score(row, "status_quo", graded)
        cascaded = score(row, f"cascade@{depth}", graded)
        recall_quo = score(row, "status_quo", f"recall@{k}")
        recall_reindex = score(row, "full_reindex", f"recall@{k}")
        recall_bridged = score(row, "bridged", f"recall@{k}")
        deep_reindex = score(row, "full_reindex", f"recall@{depth}")
        deep_bridged = score(row, "bridged", f"recall@{depth}")
        if not status_quo or cascaded is None or not recall_quo or not recall_reindex:
            continue
        if not deep_reindex or deep_bridged is None or recall_bridged is None:
            continue
        upgrade_gain = recall_reindex / recall_quo
        out.append(
            Arrangement(
                corpus=corpus_label(row["corpus"]),
                rung=f"{short(row['old_model'])}→{short(row['new_model'])}",
                cascade_advantage=(deep_bridged / deep_reindex) * upgrade_gain,
                bridge_advantage=(recall_bridged / recall_reindex) * upgrade_gain,
                won=cascaded > status_quo,
                margin=(cascaded - status_quo) / status_quo,
            )
        )
    return out


def selects(run: Arrangement, *, single_stage_gate: float) -> bool:
    """Whether the shipped rule would put the arrangement beside the decision.

    Two of the rule's four conditions are constant over this table and are not
    modelled: every corpus here carries its documents' text, and every run has a
    judged query log, so ``can_read_text`` and ``candidate_reuse`` are satisfied
    on all of them. What varies is the pair of thresholds.
    """
    return run.cascade_advantage > 1 + BORDERLINE_BAND and run.bridge_advantage <= single_stage_gate


def selection_p(margins: np.ndarray, chosen: np.ndarray, *, permutations: int, seed: int) -> float:
    """Does the rule pick runs with a larger margin than chance would?

    The null is not a coin and not the majority class: it is **this rule's own
    selection count, drawn at random**. If the arrangement wins on most runs,
    almost any selection of them looks good, and the only question worth asking
    is whether the runs the rule picked did better than the same number picked
    without looking.

    Exchange the labels rather than the values: under the null the rule carries
    no information about the margin, so which runs it named is arbitrary and
    every subset of that size is equally likely. ``(b + 1) / (B + 1)`` for the
    reason :func:`randomization_p` gives.
    """
    size = int(chosen.sum())
    if size in {0, margins.size}:
        return float("nan")
    observed = float(margins[chosen].mean() - margins[~chosen].mean())
    generator = np.random.default_rng(seed)
    at_least = 0
    for _ in range(permutations):
        drawn = generator.permutation(margins.size)[:size]
        mask = np.zeros(margins.size, dtype=bool)
        mask[drawn] = True
        if float(margins[mask].mean() - margins[~mask].mean()) >= observed - 1e-12:
            at_least += 1
    return (at_least + 1) / (permutations + 1)


def arrangement_view(  # noqa: PLR0913 - one argument per knob the test has
    runs: list[Arrangement],
    *,
    alpha: float,
    metric: str,
    k: int,
    depth: int,
    permutations: int,
    seed: int,
) -> str:
    """M2 — is the arrangement rule better than always recommending the arrangement?

    The constant rule is a strong baseline here and that is the point. If the
    two-stage arrangement wins 36 of 48, "always recommend cascade" is 75%
    accurate, and a rule that scores 75% has measured nothing. So the accuracy
    table is printed and then set aside for two things it cannot say: whether
    the quantity the rule thresholds **orders** the runs by the margin they
    returned, and whether the runs it **selected** did better than the same
    number selected blind.
    """
    total = len(runs)
    if not total:
        return "arrangement: no run measured cascade at this depth"

    wins = sum(run.won for run in runs)
    base = max(wins, total - wins) / total
    margins = np.array([run.margin for run in runs], dtype=float)
    advantage = np.array([run.cascade_advantage for run in runs], dtype=float)

    lines = [
        (f"arrangement: cascade@{depth} beat status_quo on {metric}@{k} in {wins} of {total} runs"),
        "",
        "The constant rule — always recommend the arrangement — therefore scores",
        f"{wins}/{total} = {wins / total:.4f}. That is the number the shipped rule has to",
        "beat, and an accuracy that merely matches it has measured nothing.",
        "",
        (
            "| rule | agrees | share | 95% Clopper-Pearson | 95% Wilson "
            f"| p (H0: 0.5) | p (H0: {base:.4f}) |"
        ),
        "|" + "---|" * 7,
    ]

    variants = {
        "shipped (single stage <= 1+band)": 1 + BORDERLINE_BAND,
        "plan text (single stage <= 1)": 1.0,
    }
    chosen_by: dict[str, np.ndarray] = {}
    for label, gate in variants.items():
        chosen = np.array([selects(run, single_stage_gate=gate) for run in runs], dtype=bool)
        chosen_by[label] = chosen
        agrees = int(((chosen) == np.array([run.won for run in runs])).sum())
        lines.append(_proportion_row(label, agrees, total, alpha=alpha, base=base))
    lines.append(_proportion_row("always recommend cascade", wins, total, alpha=alpha, base=base))

    lines += [
        "",
        "what an accuracy hides — the rule fires on a subset, the constant rule on all:",
        "| rule | fires | of those, won | precision | of the winners, caught |",
        "|" + "---|" * 5,
    ]
    for label, chosen in chosen_by.items():
        fires = int(chosen.sum())
        correct = int(sum(run.won for run, pick in zip(runs, chosen, strict=True) if pick))
        precision = correct / fires if fires else float("nan")
        lines.append(
            f"| {label} | {fires}/{total} | {correct} | {precision:.4f} | {correct}/{wins} |"
        )
    lines.append(
        f"| always recommend cascade | {total}/{total} | {wins} "
        f"| {wins / total:.4f} | {wins}/{wins} |"
    )
    lines += [
        "",
        "  Accuracy scores a conservative rule as if silence were a wrong answer. A rule",
        "  that declines to recommend an arrangement is not asserting the arrangement",
        "  loses — `arrangement` sits beside a decision rather than replacing it — so the",
        "  number that describes it is how often the runs it *named* actually won.",
        "",
        "identity check — is the rule's quantity the outcome, written twice?",
    ]
    gap = np.abs(advantage - (1.0 + margins))
    lines += [
        f"  max |cascade_advantage - (1 + margin)| = {gap.max():.4f} over {total} runs",
        f"  mean {gap.mean():.4f}, and {int((gap <= IDENTITY_TOLERANCE).sum())} of {total} runs",
        f"  sit inside the {IDENTITY_TOLERANCE} tolerance two rounded ratios can be compared at.",
        "  Retention is read at the candidate depth on recall, the upgrade at k on recall,",
        "  and the outcome is the graded metric of a *reranked* list at k. Three quantities,",
        "  none of which cancels — unlike bridge_advantage, which section 9 found to be one.",
    ]

    if len(set(margins.tolist())) > 1:
        spearman = stats.spearmanr(advantage, margins)
        kendall = stats.kendalltau(advantage, margins)
        lines += [
            "",
            "ranking — does the quantity order the runs by the margin they returned?",
            f"  Spearman rho = {spearman.statistic:+.3f}  p = {spearman.pvalue:.3g}",
            f"  Kendall tau  = {kendall.statistic:+.3f}  p = {kendall.pvalue:.3g}",
        ]

    lines += [
        "",
        "selection — did the runs the rule named do better than the same number drawn blind?",
        f"  {permutations} permutations, seed {seed}",
    ]
    for label, chosen in chosen_by.items():
        picked = int(chosen.sum())
        if picked in {0, total}:
            lines.append(f"  {label}: selected {picked}/{total}; no contrast to test")
            continue
        p_value = selection_p(margins, chosen, permutations=permutations, seed=seed)
        lines.append(
            f"  {label}: selected {picked}/{total}, "
            f"mean margin {margins[chosen].mean():+.4f} against "
            f"{margins[~chosen].mean():+.4f}, p = {p_value:.4f}"
        )

    lines += [
        "",
        "per run, ordered by what the rule predicted:",
        "| corpus | rung | cascade adv. | single stage | margin | won | rule |",
        "|" + "---|" * 7,
    ]
    shipped = chosen_by["shipped (single stage <= 1+band)"]
    for run, picked in sorted(
        zip(runs, shipped, strict=True), key=lambda pair: -pair[0].cascade_advantage
    ):
        lines.append(
            f"| {run.corpus} | {run.rung} | {run.cascade_advantage:.3f} "
            f"| {run.bridge_advantage:.3f} | {run.margin:+.4f} "
            f"| {'yes' if run.won else 'no':3s} | {'cascade' if picked else '—'} |"
        )
    return "\n".join(lines)


def _sidecar(path: Path, row: dict[str, Any]) -> dict[str, Any] | None:
    """The per-query scores a row names, or ``None`` if it names none or none is there."""
    relative = row.get("per_query")
    if not relative:
        return None
    target = path.parent / str(relative)
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


def paired_view(  # noqa: PLR0913 - one argument per knob the test has
    sourced: list[tuple[Path, dict[str, Any]]],
    *,
    metric: str,
    k: int,
    against: str,
    configuration: str,
    permutations: int,
    seed: int,
    alpha: float,
) -> str:
    """Per-run paired tests on the per-query scores, corrected across the table."""
    name = f"{metric}@{k}"
    tested: list[tuple[str, str, int, float, float, float]] = []
    absent: list[str] = []

    for path, row in sourced:
        label = f"{corpus_label(row['corpus'])} {short(row['old_model'])}→{short(row['new_model'])}"
        payload = _sidecar(path, row)
        if payload is None:
            absent.append(label)
            continue
        scores = payload["scores"]
        if configuration not in scores or against not in scores:
            absent.append(f"{label} (no {configuration} or {against} in the sidecar)")
            continue
        treatment = np.asarray(scores[configuration][name], dtype=float)
        control = np.asarray(scores[against][name], dtype=float)
        difference = treatment - control
        if difference.size < 2 or not np.any(difference):  # noqa: PLR2004 - a test needs two
            absent.append(f"{label} (no measurable difference to test)")
            continue
        student = float(stats.ttest_rel(treatment, control).pvalue)
        tested.append(
            (
                label,
                f"{(treatment.mean() - control.mean()) / control.mean():+.2%}",
                difference.size,
                randomization_p(difference, permutations=permutations, seed=seed),
                student,
                float(treatment.mean() - control.mean()),
            )
        )

    if not tested:
        lines = [
            f"no per-query data for any of the {len(sourced)} rows given.",
            "",
            "A paired test needs one score per query and these rows carry only means. Re-run",
            "the harness without --no-per-query; every row it writes then names a sidecar in",
            "its `per_query` field and this view reads them.",
        ]
        lines += ["", "rows without a sidecar:"] + [f"  {label}" for label in absent]
        return "\n".join(lines)

    adjusted = holm([row[3] for row in tested])
    ordered = sorted(zip(tested, adjusted, strict=True), key=lambda pair: pair[0][3])

    lines = [
        f"paired randomisation test, {configuration} against {against} on {name}",
        f"{permutations} permutations, seed {seed}, Holm over the {len(tested)} runs tested",
        "",
        "| corpus | rung | queries | Δ | p | p (Holm) | p (Student) |",
        "|" + "---|" * 7,
    ]
    for (label, delta, queries, raw, student, _), holm_p in ordered:
        corpus, _, rung = label.partition(" ")
        lines.append(
            f"| {corpus} | {rung} | {queries} | {delta} | {raw:.4f} | {holm_p:.4f} "
            f"| {student:.3g} |"
        )

    raw_significant = sum(1 for row in tested if row[3] < alpha)
    holm_significant = sum(1 for p in adjusted if p < alpha)
    gains = [(row, p) for row, p in zip(tested, adjusted, strict=True) if row[5] > 0]
    lines += [
        "",
        (
            f"{raw_significant} of {len(tested)} differ from zero at raw p < {alpha}; "
            f"{holm_significant} survive Holm."
        ),
        f"Holm cost {raw_significant - holm_significant} of them.",
        (
            f"{len(gains)} of {len(tested)} runs came out positive; "
            f"{sum(1 for _, p in gains if p < alpha)} of those survive Holm."
        ),
    ]
    if absent:
        lines += ["", f"{len(absent)} rows had no per-query data and were not tested:"]
        lines += [f"  {label}" for label in absent]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Every knob, in one place."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("rows", type=Path, nargs="+", help="One or more .jsonl the harness wrote")
    parser.add_argument(
        "--view", default="all", choices=("outcome", "paired", "arrangement", "all")
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=200,
        help="Candidate depth the arrangement view reads cascade@N at",
    )
    parser.add_argument("--k", type=int, default=10, help="Cut-off everything is measured at")
    parser.add_argument("--metric", default="ndcg", choices=("ndcg", "recall", "mrr"))
    parser.add_argument(
        "--protocol",
        default="t1-judged",
        help="Only rows measured under this protocol. Mixing protocols averages two "
        "different questions together",
    )
    parser.add_argument("--configuration", default="bridged", help="The configuration under test")
    parser.add_argument(
        "--against",
        default="status_quo",
        help="What it is tested against — the alternative a user actually has",
    )
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Read the rows, print the statistics."""
    args = build_parser().parse_args(argv)

    sourced: list[tuple[Path, dict[str, Any]]] = []
    for path in args.rows:
        sourced.extend((path, row) for row in load(path))
    sourced = [(path, row) for path, row in sourced if protocol_of(row) == args.protocol]
    if not sourced:
        print(f"no {args.protocol} rows in {', '.join(str(p) for p in args.rows)}", file=sys.stderr)
        return 1

    unique, dropped, conflicts = dedupe(sourced)
    rows = [row for _, row in unique]
    print(f"{len(rows)} distinct {args.protocol} runs from {len(args.rows)} file(s)")
    if dropped:
        print(f"  {dropped} duplicate rows dropped — the published files overlap")
    if conflicts:
        print(f"  {len(conflicts)} duplicates DISAGREE and the first was kept: {conflicts}")
    print()

    if args.view in ("outcome", "all"):
        print(
            outcome_view(
                runs_from(rows, metric=args.metric, k=args.k),
                alpha=args.alpha,
                metric=args.metric,
                k=args.k,
            )
        )
    if args.view == "all":
        print("\n" + "-" * 96 + "\n")
    if args.view in ("arrangement", "all"):
        print(
            arrangement_view(
                arrangements_from(rows, metric=args.metric, k=args.k, depth=args.depth),
                alpha=args.alpha,
                metric=args.metric,
                k=args.k,
                depth=args.depth,
                permutations=args.permutations,
                seed=args.seed,
            )
        )
    if args.view == "all":
        print("\n" + "-" * 96 + "\n")
    if args.view in ("paired", "all"):
        print(
            paired_view(
                unique,
                metric=args.metric,
                k=args.k,
                against=args.against,
                configuration=args.configuration,
                permutations=args.permutations,
                seed=args.seed,
                alpha=args.alpha,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
