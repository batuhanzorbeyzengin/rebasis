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

``--view protocol``
    The same adapter, the same corpus, the same models, under each evaluation
    protocol the harness can run. Three protocols vary two things between them —
    who asks the question and what counts as a right answer — so the view walks
    them one step at a time and prints what each step is worth. This is what
    `docs/vs-drift-adapter.md` is built from.

Every view except ``protocol`` reads **one** protocol's rows, ``t1-judged`` by
default, because averaging a ratio against human judgements together with a
ratio against a model's own neighbours produces a number that is not about
anything. Rows written before the protocol flag existed carry no protocol field
and are read as ``t1-judged``, which is what they are.
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

CONFIGURATIONS = (
    "status_quo",
    "naive_swap",
    "naive_swap_padded",
    "bridged",
    "ceiling_old_space",
    "full_reindex",
)

#: How the protocol view orders its columns: the published protocol first, then
#: one change at a time towards rebasis' own. Rows are grouped by *tag* rather
#: than by protocol, so `t0-knn@10` and `t0-knn@1` are two columns and never one
#: — they are two different ground truths, and averaging them together produces
#: a number that is not about anything.
PROTOCOL_RANK = {"t0-knn": 0, "t0-knn-real-queries": 1, "t1-judged": 2}


def load(path: Path) -> list[dict[str, Any]]:
    """Read the harness's append-only rows."""
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def protocol_of(row: dict[str, Any]) -> str:
    """Which protocol produced a row, at which ground-truth depth.

    The **tag**, not the bare protocol name. A file can hold `t0-knn` rows at
    two ground-truth depths, and those are two measurements that differ by 0.26
    on average (`docs/m0-findings.md`, section 3) — grouping them under one name
    would average a top-10 set overlap together with a nearest-neighbour hit
    rate. This function is the only place that decides, so there is one way for
    it to be wrong rather than five.

    Rows predating the flag have neither field and are ``t1-judged`` — that is
    what the harness measured before there was anything else to measure.
    """
    return str(row.get("protocol_tag", row.get("protocol", "t1-judged")))


def base_protocol(tag: str) -> str:
    """The protocol a tag belongs to, with any depth or replication stripped."""
    return tag.partition("@")[0]


def tag_order(tags: set[str]) -> list[str]:
    """Order tags for display: by protocol, then deepest ground truth first."""

    def key(tag: str) -> tuple[int, int, str]:
        depth = tag.partition("@")[2].partition("x")[0]
        return (PROTOCOL_RANK.get(base_protocol(tag), 9), -int(depth or 0), tag)

    return sorted(tags, key=key)


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


def _retention(row: dict[str, Any], metric: str) -> float | None:
    """Bridged over a full reindex, the quantity every protocol reports.

    Under a kNN ground truth the reindex scores exactly 1.0 by construction — it
    is the computation that produced the judgements — so this reduces to the
    bridged score itself, which is what arXiv:2509.23471 calls ARR. Under human
    judgements it is a real ratio. Writing it the same way in both places is
    what makes the columns comparable.
    """
    bridged = _metric(row, "bridged", metric)
    reindex = _metric(row, "full_reindex", metric)
    if bridged is None or not reindex:
        return None
    return bridged / reindex


def _misaligned(row: dict[str, Any], metric: str) -> float | None:
    """The naive swap's retention, under whichever convention made it definable."""
    for configuration in ("naive_swap", "naive_swap_padded"):
        value = _metric(row, configuration, metric)
        reindex = _metric(row, "full_reindex", metric)
        if value is not None and reindex:
            return value / reindex
    return None


#: The walk from the published protocol to rebasis' own, one change per step.
#: Each pair differs in exactly one thing, which is what makes the size of the
#: step attributable to that thing.
PROTOCOL_STEPS = (
    ("published", "recall", "published", "ndcg", "the metric, at the published protocol"),
    ("published", "recall", "hybrid", "recall", "the query distribution"),
    ("hybrid", "recall", "t1-judged", "recall", "the ground truth"),
    ("t1-judged", "recall", "t1-judged", "ndcg", "the metric, at rebasis' protocol"),
)


def walk_tags(present: list[str]) -> dict[str, str]:
    """Which three tags the decomposition walks between.

    The walk needs one `t0-knn` column and one `t0-knn-real-queries` column at
    the **same** ground-truth depth, or a step would move two things at once and
    stop being attributable. The deepest available depth is chosen, because that
    is the one the published protocol is being read as.
    """
    deepest = [tag for tag in present if base_protocol(tag) == "t0-knn"]
    if not deepest:
        return {}
    published = deepest[0]
    depth = published.partition("@")[2]
    hybrid = f"t0-knn-real-queries@{depth}"
    chosen = {"published": published, "t1-judged": "t1-judged"}
    if hybrid in present:
        chosen["hybrid"] = hybrid
    return chosen


def _protocol_table(
    runs: list[tuple[str, str, str]],
    indexed: dict[tuple[str, str, str, str], dict[str, Any]],
    present: list[str],
    *,
    k: int,
) -> tuple[list[str], list[tuple[str, str, str]]]:
    """The per-run table, and the runs that every protocol measured.

    Only the complete runs feed the means underneath. A step is a difference,
    and a difference taken over two different sets of runs is not one.
    """
    header = ["corpus", "rung"]
    for name in present:
        header += [f"{name} R@{k}", f"{name} nDCG@{k}"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    complete: list[tuple[str, str, str]] = []
    for run in runs:
        values = [
            _retention(indexed[name, *run], f"{metric}@{k}") if (name, *run) in indexed else None
            for name in present
            for metric in ("recall", "ndcg")
        ]
        lines.append(
            "| "
            + " | ".join(
                [corpus_label(run[0]), f"{short(run[1])}→{short(run[2])}"]
                + ["—" if value is None else f"{value:.3f}" for value in values]
            )
            + " |"
        )
        if all(value is not None for value in values):
            complete.append(run)
    return lines, complete


def protocol_view(rows: list[dict[str, Any]], *, k: int) -> str:
    """One line per run, one column per protocol, and the delta itemised.

    The columns are retention — bridged over a full reindex — because that is
    the one quantity all three protocols define, and because under a kNN ground
    truth it is exactly the ARR a published result reports.

    The summary underneath walks from the published protocol to rebasis' own one
    change at a time. Each step moves exactly one thing, so the size of each step
    is what that thing is worth.
    """
    present = tag_order({protocol_of(row) for row in rows})
    if len(present) < 2:  # noqa: PLR2004 - a comparison needs two protocols
        return f"only one protocol in these rows ({present or ['none']})"
    walk = walk_tags(present)

    indexed: dict[tuple[str, str, str, str], dict[str, Any]] = {
        (protocol_of(row), row["corpus"], row["old_model"], row["new_model"]): row for row in rows
    }
    runs = sorted({key[1:] for key in indexed})
    lines, complete = _protocol_table(runs, indexed, present, k=k)

    mean: dict[tuple[str, str], float] = {}
    for name in present:
        for metric in ("recall", "ndcg"):
            values = [_retention(indexed[name, *run], f"{metric}@{k}") for run in complete]
            if values:
                mean[name, metric] = float(np.mean(values))

    counted = (
        f"retention, mean over the {len(complete)} runs measured under all "
        f"{len(present)} protocols:"
    )
    lines += ["", counted, ""]
    lines += [
        f"  {name + ' ' + metric + '@' + str(k):34s} {value:.3f}"
        for (name, metric), value in mean.items()
    ]

    lines += ["", f"what each step is worth (walking {walk.get('published', '?')} -> t1-judged):"]
    lines += [
        f"  {what:38s} {mean[walk[to_role], to_metric] - mean[walk[from_role], from_metric]:+.3f}"
        for from_role, from_metric, to_role, to_metric, what in PROTOCOL_STEPS
        if from_role in walk
        and to_role in walk
        and (walk[from_role], from_metric) in mean
        and (walk[to_role], to_metric) in mean
    ]

    lines += ["", f"the misaligned baseline, as a share of a full reindex, nDCG@{k}:"]
    for name in present:
        values = [
            value
            for row in rows
            if protocol_of(row) == name
            if (value := _misaligned(row, f"ndcg@{k}")) is not None
        ]
        if values:
            lines.append(f"  {name:34s} {np.mean(values):.3f} over {len(values)} runs")

    # The ceiling only exists where the ground truth is a set of documents, so
    # it appears under the kNN protocols and nowhere else. It is the number that
    # says whether a low retention is the adapter's fault or the old space's.
    lines += ["", f"what the old space could hold at all, recall@{k}:"]
    for name in present:
        values = [
            value
            for row in rows
            if protocol_of(row) == name
            if (value := _metric(row, "ceiling_old_space", f"recall@{k}")) is not None
        ]
        if values:
            bridged = [
                value
                for row in rows
                if protocol_of(row) == name
                if (value := _metric(row, "bridged", f"recall@{k}")) is not None
            ]
            lines.append(
                f"  {name:34s} ceiling {np.mean(values):.3f}  "
                f"bridged {np.mean(bridged):.3f}  over {len(values)} runs"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", type=Path, help="The .jsonl the harness wrote")
    parser.add_argument(
        "--view", default="summary", choices=("band", "cascade", "geometry", "summary", "protocol")
    )
    parser.add_argument("--k", type=int, default=10, help="Cut-off for the band view")
    parser.add_argument("--metric", default="ndcg", choices=("ndcg", "recall", "mrr"))
    parser.add_argument(
        "--rung", default=None, help="Only runs whose new model ends with this string"
    )
    parser.add_argument(
        "--protocol",
        default="t1-judged",
        help=(
            "Which protocol's rows the single-protocol views read. The `protocol` "
            "view always reads all of them, because comparing them is what it is for"
        ),
    )
    args = parser.parse_args(argv)

    rows = load(args.rows)
    if args.rung:
        rows = [r for r in rows if r["new_model"].endswith(args.rung)]
    if not rows:
        print("no rows", file=sys.stderr)
        return 1

    if args.view == "protocol":
        print(protocol_view(rows, k=args.k))
        return 0

    # Matches either the full tag or the bare protocol name, so `--protocol
    # t0-knn` selects every depth and `--protocol t0-knn@10` selects one.
    rows = [
        row for row in rows if args.protocol in {protocol_of(row), base_protocol(protocol_of(row))}
    ]
    if not rows:
        print(f"no {args.protocol} rows", file=sys.stderr)
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
