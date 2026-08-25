"""Turn the band harness's rows into the tables the documents quote.

Separate from `bridge_band.py` because the two have different lifetimes: the
measurement is hours of GPU time and is run once, the reading of it is run every
time somebody asks a different question of the same numbers. Keeping them apart
means a new question costs a second of CPU rather than a repeat of the run.

Three views, all from the same file:

``--view band``
    One row per run at a single cut-off: the four configurations, the
    break-even, and whether bridging beat doing nothing. This is what
    `docs/bridge-band.md` is built from.

``--view cascade``
    The question the cut-off list exists for. For each run, the measured
    two-stage result at k=10 against both alternatives a user actually has —
    keeping the current model, or bridging in one stage — plus the recall@N that
    bounds it.

``--view geometry``
    The pre-fit bound against the retention it bounds. δ costs one Gram-matrix
    difference and no fit; whether it tracks what the fit then achieves is the
    question this view is for.

``--view summary``
    The aggregate claims: how often the break-even predicted the outcome, the
    correlation between upgrade gain and retention, and the same for recall at
    every cut-off measured.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

CONFIGURATIONS = ("status_quo", "naive_swap", "bridged", "full_reindex")


def load(path: Path) -> list[dict[str, Any]]:
    """Read the harness's append-only rows."""
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def short(model_id: str) -> str:
    return model_id.rpartition("/")[2].removesuffix("-en-v1.5").removesuffix("-v2")


def corpus_label(name: str) -> str:
    """Short, stable name for a corpus, whichever loader it came from."""
    label = name.removeprefix("mmteb:mteb/").removeprefix("beir/")
    label = label.removeprefix("cqadupstack/").removesuffix("/test")
    # MMTEB's hard-negative variants carry their construction in the name;
    # a table column is not the place for it.
    return label.replace("_test_top_250_only_w_correct-v2", "-HN")


def _metric(row: dict[str, Any], configuration: str, metric: str) -> float | None:
    scores = row["scores"].get(configuration)
    if scores is None:
        return None
    value = scores.get(metric)
    return None if value is None else float(value)


def band_view(rows: list[dict[str, Any]], *, k: int, metric: str) -> str:
    """The four configurations, the break-even, and what actually happened."""
    header = (
        "| corpus | rung | status quo | naive swap | bridged | reindex | "
        "gain | retention | product | vs. doing nothing |"
    )
    lines = [header, "|" + "---|" * 10]
    for row in rows:
        name = f"{metric}@{k}"
        status_quo = _metric(row, "status_quo", name)
        naive = _metric(row, "naive_swap", name)
        bridged = _metric(row, "bridged", name)
        reindex = _metric(row, "full_reindex", name)
        if status_quo is None or bridged is None or reindex is None:
            continue
        gain = reindex / status_quo if status_quo else float("nan")
        retention = bridged / reindex if reindex else float("nan")
        delta = (bridged - status_quo) / status_quo if status_quo else float("nan")
        lines.append(
            f"| {corpus_label(row['corpus'])} "
            f"| {short(row['old_model'])}→{short(row['new_model'])} "
            f"| {status_quo:.3f} "
            f"| {'—' if naive is None else f'{naive:.3f}'} "
            f"| {bridged:.3f} | {reindex:.3f} "
            f"| {gain:.2f} | {retention:.2f} | {gain * retention:.2f} "
            f"| {delta:+.1%} |"
        )
    return "\n".join(lines)


def cascade_view(rows: list[dict[str, Any]], *, metric: str = "ndcg", k: int = 10) -> str:
    """The two-stage result against the two alternatives a user actually has.

    Everything is at k=10, because that is what a RAG pipeline consumes and what
    every other number in `docs/bridge-band.md` is measured at. The candidate
    size N only decides how wide the first stage casts; the answer is still ten
    documents.

    `recall@N held` is the quantity that bounds the whole arrangement — the
    share of the relevant documents a full reindex would have found that the
    bridge managed to put in the candidate set. Reported beside the outcome so
    the two can be read against each other.
    """
    stages = sorted({n for row in rows for n in row.get("cascade", [])})
    if not stages:
        return "no cascade measurements in these rows"

    name = f"{metric}@{k}"
    header = ["corpus", "rung", "status quo", "bridged", "reindex"]
    for n in stages:
        header += [f"cascade@{n}", "vs. today", f"R@{n} held"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    wins = dict.fromkeys(stages, 0)
    counted = dict.fromkeys(stages, 0)

    for row in rows:
        status_quo = _metric(row, "status_quo", name)
        bridged = _metric(row, "bridged", name)
        reindex = _metric(row, "full_reindex", name)
        if not status_quo or bridged is None or reindex is None:
            continue
        cells = [
            corpus_label(row["corpus"]),
            f"{short(row['old_model'])}→{short(row['new_model'])}",
            f"{status_quo:.3f}",
            f"{bridged:.3f}",
            f"{reindex:.3f}",
        ]
        for n in stages:
            cascade = _metric(row, f"cascade@{n}", name)
            recall_bridged = _metric(row, "bridged", f"recall@{n}")
            recall_reindex = _metric(row, "full_reindex", f"recall@{n}")
            if cascade is None:
                cells += ["—", "—", "—"]
                continue
            counted[n] += 1
            wins[n] += cascade > status_quo
            held = (
                recall_bridged / recall_reindex
                if recall_bridged is not None and recall_reindex
                else float("nan")
            )
            cells += [
                f"{cascade:.3f}",
                f"{(cascade - status_quo) / status_quo:+.1%}",
                f"{held:.2f}",
            ]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.extend(
        f"cascade@{n} beat keeping the current model in {wins[n]}/{counted[n]} runs"
        for n in stages
        if counted[n]
    )
    return "\n".join(lines)


def geometry_view(rows: list[dict[str, Any]], *, k: int) -> str:
    """The pre-fit bound beside the retention it turned out to bound.

    δ is available in seconds and the retention costs a full fit and evaluation,
    so a δ that ordered the runs the way retention does would be worth having.
    Whether it does is a measurement, not an assumption — the correlation is
    printed rather than claimed.
    """
    lines = [
        f"| corpus | rung | dim | δ | bound | cosine floor | retention nDCG@{k} | ARR |",
        "|" + "---|" * 8,
    ]
    deltas, retentions = [], []
    for row in rows:
        geometry = row.get("geometry")
        if not geometry or geometry.get("geometry_delta") is None:
            continue
        bridged = _metric(row, "bridged", f"ndcg@{k}")
        reindex = _metric(row, "full_reindex", f"ndcg@{k}")
        if bridged is None or not reindex:
            continue
        retention = bridged / reindex
        floor = geometry.get("cosine_floor")
        deltas.append(geometry["geometry_delta"])
        retentions.append(retention)
        lines.append(
            f"| {corpus_label(row['corpus'])} "
            f"| {short(row['old_model'])}→{short(row['new_model'])} "
            f"| {geometry['dim']} | {geometry['geometry_delta']:.4f} "
            f"| {geometry['alignment_bound']:.2f} "
            f"| {'—' if floor is None else f'{floor:.3f}'} "
            f"| {retention:.3f} | {row['fit']['arr_r10']:.3f} |"
        )

    if len(deltas) >= 2:  # noqa: PLR2004 - a correlation needs two points
        correlation = float(np.corrcoef(deltas, retentions)[0, 1])
        lines += [
            "",
            f"corr(δ, retention) = {correlation:+.3f} over {len(deltas)} runs",
        ]
    return "\n".join(lines)


def _predicted(rows: list[dict[str, Any]], *, k: int, metric: str) -> tuple[int, int]:
    """How often ``gain × retention > 1`` agreed with the measured outcome."""
    agreed = total = 0
    for row in rows:
        name = f"{metric}@{k}"
        status_quo = _metric(row, "status_quo", name)
        bridged = _metric(row, "bridged", name)
        reindex = _metric(row, "full_reindex", name)
        if not status_quo or not reindex or bridged is None:
            continue
        product = (reindex / status_quo) * (bridged / reindex)
        total += 1
        agreed += (product > 1.0) == (bridged > status_quo)
    return agreed, total


def summary_view(rows: list[dict[str, Any]], cutoffs: Sequence[int]) -> str:
    """The aggregate claims, recomputed rather than quoted."""
    out: list[str] = [f"runs: {len(rows)}", ""]

    for metric in ("ndcg", "recall"):
        for k in cutoffs:
            name = f"{metric}@{k}"
            gains, retentions, wins = [], [], 0
            counted = 0
            for row in rows:
                status_quo = _metric(row, "status_quo", name)
                bridged = _metric(row, "bridged", name)
                reindex = _metric(row, "full_reindex", name)
                if not status_quo or not reindex or bridged is None:
                    continue
                gains.append(reindex / status_quo)
                retentions.append(bridged / reindex)
                wins += bridged > status_quo
                counted += 1
            if counted < 2:  # noqa: PLR2004 - a correlation needs two points
                continue
            agreed, total = _predicted(rows, k=k, metric=metric)
            correlation = float(np.corrcoef(gains, retentions)[0, 1])
            out.append(
                f"{name:12s}  gain {np.mean(gains):.3f}  retention {np.mean(retentions):.3f}  "
                f"corr(gain, retention) {correlation:+.3f}  "
                f"bridging beat doing nothing {wins}/{counted}  "
                f"break-even predicted {agreed}/{total}"
            )
        out.append("")

    headline = f"ndcg@{cutoffs[0]}"
    pairs = [
        (_metric(row, "naive_swap", headline), _metric(row, "status_quo", headline)) for row in rows
    ]
    naive = [swap / quo for swap, quo in pairs if swap is not None and quo]
    if naive:
        out.append(f"naive swap retains {np.mean(naive):.3f} of the status quo ({len(naive)} runs)")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", type=Path, help="The .jsonl the harness wrote")
    parser.add_argument(
        "--view", default="summary", choices=("band", "cascade", "geometry", "summary")
    )
    parser.add_argument("--k", type=int, default=10, help="Cut-off for the band view")
    parser.add_argument("--metric", default="ndcg", choices=("ndcg", "recall", "mrr"))
    parser.add_argument(
        "--rung", default=None, help="Only runs whose new model ends with this string"
    )
    args = parser.parse_args(argv)

    rows = load(args.rows)
    if args.rung:
        rows = [r for r in rows if r["new_model"].endswith(args.rung)]
    if not rows:
        print("no rows", file=sys.stderr)
        return 1

    cutoffs = sorted({k for row in rows for k in row.get("cutoffs", [])})

    if args.view == "band":
        print(band_view(rows, k=args.k, metric=args.metric))
    elif args.view == "cascade":
        print(cascade_view(rows, metric=args.metric, k=args.k))
    elif args.view == "geometry":
        print(geometry_view(rows, k=args.k))
    else:
        print(summary_view(rows, cutoffs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
