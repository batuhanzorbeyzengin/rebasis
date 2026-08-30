"""Turning CLI flags into a running pipeline.

``probe``, ``fit`` and ``eval`` all need the same four things: a store, two
embedders, an optional query log, and somewhere to put the report. Doing that
once here keeps the three commands consistent — a flag means the same thing in
all of them — and keeps each command file about its own job.

Imports of the heavy machinery are deferred into the functions on purpose:
``rebasis --help`` must not pay for numpy, torch or a store client, and
``tests/unit/test_lazy_imports.py`` fails the build if it does.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from rebasis.cli._common import console, err_console
from rebasis.cli._profiles import ProfileOverrides, resolve_profile
from rebasis.errors import MalformedQueryLog
from rebasis.storage.atomic import atomic_write_text

if TYPE_CHECKING:
    from pathlib import Path

    from rebasis.audit import AuditWriter
    from rebasis.probe.session import QueryLog
    from rebasis.store.base import VectorStore
    from rebasis.types import Embedder

__all__ = [
    "ProfileOverrides",
    "audit_writer_for",
    "load_query_log",
    "open_embedders",
    "open_target_store",
    "print_result",
    "read_access_log",
    "write_reports",
]


def audit_writer_for(state_dir: Path | None) -> AuditWriter:
    """Open the audit trail beside the project.

    Constructed here rather than inside ``probe`` so that layer never reaches
    down into ``manifest``: the decision is written by ``probe``, but where it
    is written is a CLI concern.
    """
    import uuid

    from rebasis.audit import AuditWriter
    from rebasis.manifest import ManifestDB, default_state_dir, manifest_path

    directory = state_dir or default_state_dir()
    return AuditWriter(ManifestDB(manifest_path(directory)), run_id=uuid.uuid4().hex[:16])


def open_target_store(uri: str) -> VectorStore:
    """Open the store named by ``--store``."""
    from rebasis.store import open_store

    return open_store(uri)


def open_embedders(  # noqa: PLR0913 - each profile override is a documented flag
    old: str | None,
    new: str,
    *,
    device: str = "auto",
    backend: str = "sentence-transformers",
    old_overrides: ProfileOverrides | None = None,
    new_overrides: ProfileOverrides | None = None,
    store_dim: int | None = None,
) -> tuple[Embedder | None, Embedder]:
    """Construct the candidate embedder, and the current one when asked for.

    ``old`` is optional because the T0 tier does not need it: the old-model
    vectors are already in the index.

    ``store_dim`` is the index's own dimension, used as the fallback for the
    **old** model only. The index is authoritative about the model it was built
    with, so an unregistered old model needs no ``--old-dim``. The new model
    gets no such fallback: nothing has measured it yet, and assuming it matches
    the index would silently produce a wrong profile for the common case where
    the whole point is that the dimensions differ.
    """
    from rebasis.embed import open_embedder

    device_arg = None if device == "auto" else device
    new_embedder = open_embedder(
        new,
        backend=backend,
        device=device_arg,
        profile=resolve_profile(new, new_overrides),
    )
    old_embedder = None
    if old is not None:
        old_embedder = open_embedder(
            old,
            backend=backend,
            device=device_arg,
            profile=resolve_profile(old, old_overrides, fallback_dim=store_dim),
        )
    return old_embedder, new_embedder


def read_access_log(path: Path | None) -> dict[str, float] | None:
    """Read access counts from a JSONL log: ``{"id": ..., "count": ...}``.

    Two commands take one and they mean different things by it, which is why it
    is parsed in one place and interpreted in two. `migrate --priority access`
    orders the queue, so hot records are rewritten first and quality improves
    where somebody will notice. `probe --access-log` weights which sampled
    records become **query proxies**, so ARR describes the questions people
    actually send.

    ``count`` defaults to 1 for a line that omits it, and ``record_id`` is
    accepted beside ``id`` because these logs are usually exported from
    somewhere else and a rigid schema means everyone writes a converter first.

    Returns ``None`` for no path and for a log naming nothing, so a caller can
    tell "no log" from "a log with no usable lines" only by what it passed —
    which is right: both mean the same thing downstream, and reporting an empty
    log as a weighting that happened would be worse.
    """
    if path is None:
        return None

    import json

    counts: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            record_id = payload.get("id") or payload.get("record_id")
            if record_id is not None:
                counts[str(record_id)] = float(payload.get("count", 1))
    return counts or None


def load_query_log(path: Path) -> QueryLog:
    """Read a JSONL query log.

    One object per line, ``{"query": "...", "relevant": ["id", ...]}``. Keys are
    accepted under a couple of common names because query logs are usually
    exported from somewhere else, and a rigid schema here would mean every user
    writes a converter first.

    Raises:
        ConfigInvalid: When a line is not JSON, or carries no query text.
    """
    from rebasis.probe.session import QueryLog

    queries: list[str] = []
    qrels: list[set[str]] = []

    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise MalformedQueryLog(
                    f"{path.name} line {number} is not valid JSON.",
                    hint='Expected one object per line: {"query": "...", "relevant": ["id"]}.',
                    context={"path": str(path), "count": number},
                    cause=exc,
                ) from exc

            text = _first(payload, "query", "text", "question")
            if not text:
                raise MalformedQueryLog(
                    f"{path.name} line {number} has no query text.",
                    hint='Each line needs a "query" field.',
                    context={"path": str(path), "count": number},
                )
            relevant = _first(payload, "relevant", "relevant_ids", "qrels", "doc_ids") or []
            queries.append(str(text))
            qrels.append({str(r) for r in relevant})

    if not queries:
        raise MalformedQueryLog(
            f"{path.name} contains no queries.",
            hint="An empty query log falls back to nothing; omit --queries instead.",
            context={"path": str(path)},
        )
    return QueryLog(queries=queries, qrels=qrels, metadata={"source": path.name})


def _first(payload: dict[str, Any], *names: str) -> Any:
    for name in names:
        if payload.get(name) is not None:
            return payload[name]
    return None


def write_reports(result: Any, *, store_uri: str, report: Path | None) -> None:
    """Write the report in the format the extension asks for."""
    if report is None:
        return

    from rebasis.report import render_html, render_markdown

    suffix = report.suffix.lower()
    rendered = (
        render_html(result, store_uri=store_uri)
        if suffix in {".html", ".htm"}
        else render_markdown(result, store_uri=store_uri)
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    # Atomic: `write_text` truncates first, so a full disk or a Ctrl-C between
    # the two takes the previous report with it — and a probe run is minutes of
    # embedding to reproduce. `private=False` because the user named
    # this file: it is theirs to read, serve or send, and 0600 inherited from a
    # temporary file would be an accident rather than a decision.
    atomic_write_text(report, rendered, private=False)
    # stderr: this is messaging about the run, not the run's output. On stdout
    # it would sit inside the JSON that `--json` is piping to something.
    err_console.print(f"[dim]Report written to {report}[/dim]")


def print_json(result: Any) -> None:
    """The machine-readable form: the whole result, on stdout, and nothing else.

    The same dict the report and the audit record are built from, so a script
    branching on `decision` reads exactly what the panel above it would say.
    """
    import json

    console.print_json(json.dumps(result.to_dict(), default=str))


#: What to run next, per decision. A three-step tool that never names step two
#: makes the user go back to the README to find a command it could have printed.
_NEXT_STEP: dict[str, str] = {
    "bridge_sufficient": "Fit the adapter and put it in front of your queries:",
    "bridge_and_migrate": "Fit the adapter, then migrate in the background:",
    "caution": "If you want to try it anyway, fit the adapter and measure again:",
}


def _command(*lines: str) -> None:
    """Print a command the reader is meant to copy.

    No markup: a store URI can hold a bracket, and a trailing backslash directly
    before one is Rich's own escape — which turned the line continuation into a
    literal `[/bold]` on screen. No wrapping either: a break landing mid-flag
    produces a second line that does not run.
    """
    for line in lines:
        console.print(f"  {line}", style="bold", markup=False, soft_wrap=True, highlight=False)


def print_cascade_setup(result: Any, *, store: str, old: str, new: str) -> None:
    """Print how to build the two-stage arrangement, where it is the answer.

    Printed as `fit` plus the `Cascade(...)` call rather than as prose. The
    decision this follows is often ``full_reindex``, whose next-step text says
    an adapter cannot close the gap — true of the single stage and not of this
    one, so leaving the reader to assemble the arrangement from the API
    reference is how a recommendation becomes an unread paragraph.
    """
    decision = result.decision
    if decision.arrangement != "cascade":
        return
    depth = decision.cascade_n
    console.print()
    console.print(
        "[dim]Fit the adapter, then serve it as a recall stage — the index is not touched:[/dim]"
    )
    _command(
        f"rebasis fit --store {store} \\",
        f"    --old {old} --new {new} \\",
        "    --out adapter.rbs",
    )
    _command(
        "from rebasis.serve import Bridge, Cascade",
        "from rebasis.store import open_store",
        "",
        "cascade = Cascade(",
        f"    open_store({store!r}),",
        '    Bridge.load("adapter.rbs"),',
        "    new_embedder,",
        *([] if depth is None else [f"    candidates={depth},"]),
        ")",
        'hits = cascade.search(new_embedder.encode([q], kind="query")[0], k=10)',
    )


def print_next_step_after_probe(result: Any, *, store: str, old: str, new: str) -> None:
    """Print the command this decision implies."""
    if result.decision.arrangement == "cascade":
        # The arrangement outranks the decision's own next step here: it is the
        # answer the numbers support, and printing "reindex" underneath it would
        # send the reader to the more expensive of two measured options.
        print_cascade_setup(result, store=store, old=old, new=new)
        return
    decision = result.decision.decision
    if decision == "no_upgrade_needed":
        console.print()
        console.print("[dim]Nothing to do: the current model is not measurably behind.[/dim]")
        return
    if decision == "full_reindex":
        console.print()
        console.print(
            "[dim]An adapter cannot close this gap. Reindexing with the new "
            "model is the honest option.[/dim]"
        )
        return

    lead = _NEXT_STEP.get(decision)
    if lead is None:
        return
    console.print()
    console.print(f"[dim]{lead}[/dim]")
    _command(
        f"rebasis fit --store {store} \\",
        f"    --old {old} --new {new} \\",
        "    --out adapter.rbs",
    )


def write_comparison_report(result: Any, *, store_uri: str, report: Path | None) -> None:
    """Write the comparison as HTML or Markdown, by the path's suffix."""
    if report is None:
        return
    from rebasis.report import render_comparison_html, render_comparison_markdown

    render = (
        render_comparison_html if report.suffix.lower() == ".html" else render_comparison_markdown
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    # Through `atomic_write_text` for the reason `write_reports` gives, and more
    # so here: a comparison is N embedding passes, and `write_text` truncates
    # first, so a full disk or a Ctrl-C between the two takes the previous
    # report with it. `tests/unit/test_write_discipline.py` caught this one.
    atomic_write_text(report, render(result, store_uri=store_uri), private=False)
    err_console.print(f"[dim]Report written to {report}[/dim]")


def print_comparison_json(result: Any) -> None:
    """The ranking on stdout, and nothing else."""
    import json

    console.print_json(json.dumps(result.to_dict(), default=str))


def print_comparison(result: Any) -> None:
    """The ranking as a table, with what the ordering is worth underneath it.

    The caveat is not a footnote and is not optional. `compare` makes a stronger
    claim than `probe` — which model is better, rather than whether bridging
    pays — and the evidence for it is a rank correlation rather than an
    accuracy. A table printed without that reads as a leaderboard.
    """
    from rich.table import Table

    table = Table(title="Candidates", title_justify="left", header_style="bold")
    table.add_column("Model")
    table.add_column("vs. current", justify="right")
    table.add_column("ARR", justify="right")
    table.add_column("bridge", justify="right")
    table.add_column("cascade", justify="right")
    table.add_column("reindex", justify="right")
    table.add_column("Decision")

    for candidate in result.ranked():
        decision = candidate.result.decision
        marker = " [dim](round 1)[/dim]" if candidate.eliminated else ""
        table.add_row(
            f"{candidate.model}{marker}",
            "—" if candidate.upgrade_gain is None else f"{candidate.upgrade_gain:.2f}x",
            f"{decision.arr_at_k:.3f}",
            _ratio(decision.bridge_advantage),
            _ratio(decision.cascade_advantage),
            _hours(candidate.result.reindex_cost),
            (
                f"{decision.decision}"
                if decision.arrangement == "single_stage"
                else f"{decision.decision} [green]+cascade[/green]"
            ),
        )

    console.print()
    console.print(table)
    console.print()
    reference = result.reference.get("model") or "the index's own model"
    console.print(
        f"[dim]'vs. current' is how much better each candidate retrieves than "
        f"{reference}, which is the model already in the index and is the "
        f"reference rather than a candidate.[/dim]"
    )
    console.print(f"[yellow]{result.ranking_caveat}[/yellow]")
    if result.remote_candidates:
        console.print(
            f"[yellow]Document text was sent off this machine for: "
            f"{', '.join(result.remote_candidates)}.[/yellow]"
        )


def _ratio(value: float | None) -> str:
    """A break-even, or an em dash where it could not be computed."""
    return "—" if value is None else f"{value:.2f}x"


def _hours(cost: dict[str, Any] | None) -> str:
    """A reindex estimate as something a person reads."""
    seconds = (cost or {}).get("seconds")
    if not seconds:
        return "—"
    if seconds < 5400:  # noqa: PLR2004 - the point at which hours read better
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def write_grid_report(grid: Any, *, store_uri: str, report: Path | None) -> None:
    """Write the truncation grid as HTML or Markdown, by the path's suffix."""
    if report is None:
        return
    from rebasis.report import render_grid_html, render_grid_markdown

    render = render_grid_html if report.suffix.lower() == ".html" else render_grid_markdown
    report.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(report, render(grid, store_uri=store_uri), private=False)
    err_console.print(f"[dim]Report written to {report}[/dim]")


def print_grid_json(grid: Any) -> None:
    """The grid on stdout, and nothing else."""
    import json

    console.print_json(json.dumps(grid.to_dict(), default=str))


def print_grid(grid: Any) -> None:
    """The grid as a table, retention over storage, cheapest acceptable cell last.

    Two numbers per cell rather than one. Retention alone invites reading a
    binary row as a disaster when the pattern that makes it useful — candidates
    from the cheap codes, reordered by the full-precision vectors — is the
    second number and is usually much higher.
    """
    from rich.table import Table

    precisions = list(dict.fromkeys(cell.precision for cell in grid.cells))
    by_dim: dict[int, dict[str, Any]] = {}
    for cell in grid.cells:
        by_dim.setdefault(cell.dim, {})[cell.precision] = cell

    table = Table(
        title=f"Retained of nDCG@{grid.k}  ·  rescored at {grid.rescore_at}  ·  storage",
        title_justify="left",
        header_style="bold",
    )
    table.add_column("dim")
    for precision in precisions:
        table.add_column(precision, justify="right")
    for dim in sorted(by_dim, reverse=True):
        label = f"{dim}" + (" (full)" if dim == grid.full_dim else "")
        row = [label]
        for precision in precisions:
            cell = by_dim[dim].get(precision)
            row.append(
                "—"
                if cell is None
                else (
                    f"{cell.retained:.3f} / {cell.retained_rescored:.3f}\n"
                    f"[dim]{cell.storage:.3f}x[/dim]"
                )
            )
        table.add_row(*row)

    console.print()
    console.print(table)
    console.print(
        f"[dim]Each cell: retained / retained after a full-precision rescore of the "
        f"top {grid.rescore_at}, and the storage it costs as a fraction of today's.[/dim]"
    )
    console.print(f"[dim]{grid.simulation_note}[/dim]")

    if grid.floor is not None:
        chosen = grid.cheapest_above(grid.floor)
        if chosen is None:
            console.print(
                f"[yellow]No cell retains {grid.floor:.0%}. The best is "
                f"{max(c.retained for c in grid.cells):.3f}.[/yellow]"
            )
        else:
            low, high = chosen.interval
            straddles = low < grid.floor <= high
            console.print()
            console.print(
                f"[bold green]Cheapest cell above {grid.floor:.0%}: "
                f"{chosen.dim} dimensions at {chosen.precision}[/bold green] — "
                f"retains {chosen.retained:.3f} "
                f"[dim](95% {low:.3f}–{high:.3f})[/dim] for {chosen.storage:.3f}x "
                f"the storage."
            )
            if straddles:
                console.print(
                    "[yellow]Its interval spans the floor, so this run cannot "
                    "settle whether that cell clears it. Increase --sample.[/yellow]"
                )
    for warning in grid.warnings:
        console.print(f"[yellow]{warning}[/yellow]")


def print_exposure_json(result: Any) -> None:
    """The measurement on stdout, and nothing else."""
    import json

    console.print_json(json.dumps(result.to_dict(), default=str))


def print_exposure(result: Any) -> None:
    """One number, the pool it is relative to, and what it does not say.

    No band. low/medium/high would be a classifier and the evidence does not
    support one — `docs/exposure.md` says so at length and this print stays
    short, because a caveat nobody reads is a caveat that is not there.
    """
    console.print()
    attempts = ", ".join(f"{value:.3f}" for value in result.per_seed)
    console.print(
        f"  Alignability   [bold]{result.alignability:.3f}[/bold]  "
        f"[dim]({result.pool:,} documents, mean rank {result.mean_rank:.1f})[/dim]"
    )
    if len(result.per_seed) > 1:
        console.print(
            f"  Attempts       {attempts}  [dim](the best of them is the figure "
            f"above; the method is stochastic)[/dim]"
        )
    console.print(f"  Reference      {result.reference_model} [dim](local)[/dim]")
    console.print(
        f"  Sample         {result.n_sampled:,} of {result.n_total:,}  "
        f"[dim]seed {result.seed}[/dim]"
    )
    console.print()
    console.print(
        f"An adversary holding only these vectors, plus a public embedding model "
        f"over their own documents, aligned this index well enough to identify "
        f"[bold]{result.alignability:.0%}[/bold] of a {result.pool:,}-document "
        f"hold-out by its vector alone."
    )
    for warning in result.warnings:
        console.print(f"  [yellow]{warning}[/yellow]")
    console.print(
        "[dim]There is no low/medium/high here: banding this number would be a "
        "classifier, and the evidence does not support one. What the number means "
        "— and the four things it does not say — are in docs/exposure.md.[/dim]"
    )


def print_next_step_after_fit(out: Any, *, store: str) -> None:
    """Print what to do with the adapter that was just written."""
    console.print()
    console.print("[dim]Use it in front of your queries:[/dim]")
    _command(f"Bridge.load({str(out)!r}).to_index_space(vectors)")
    console.print("[dim]Or rewrite the index in the background, a batch at a time:[/dim]")
    _command(f"rebasis migrate --adapter {out} --store {store}")


def print_result(result: Any) -> None:
    """Print the decision and the numbers behind it.

    The decision leads. A user who reads one line should read the one that says
    what to do, not the one that says 0.937.
    """
    from rich.table import Table

    decision = result.decision
    colour = {
        "bridge": "green",
        "bridge_with_caution": "yellow",
        "investigate": "yellow",
        "reindex": "red",
        "no_upgrade_needed": "cyan",
    }.get(decision.decision, "white")

    console.print()
    console.print(f"[bold {colour}]{decision.decision.replace('_', ' ')}[/bold {colour}]")
    console.print(f"{decision.rationale}")
    console.print()

    low, high = result.best.arr_ci
    console.print(
        f"  ARR@{result.k}      [bold]{result.best.arr:.3f}[/bold]  "
        f"[dim](95% CI {low:.3f}–{high:.3f})[/dim]"
    )
    console.print(f"  adapter     {result.best.name} ({result.best.n_params:,} parameters)")
    console.print(f"  ground truth {result.ground_truth_tier.upper()}")
    # A ceiling, printed under the measurement it is a ceiling on. Not shown
    # when it is vacuous: "≥ -0.4 cosine" is not a fact anybody can use.
    cascade = decision.cascade_advantage
    if cascade is not None and decision.cascade_arr is not None:
        depth = "" if decision.cascade_n is None else f"@{decision.cascade_n}"
        served = (
            "[green]recommended[/green]"
            if decision.arrangement == "cascade"
            else "measured, not recommended"
        )
        console.print(
            f"  cascade{depth:<5} [bold]{decision.cascade_arr:.3f}[/bold] retained at candidate "
            f"depth  [dim](break-even {cascade:.2f}x if the new model reranks; "
            f"{served})[/dim]"
        )
        per_query = decision.cascade_embeddings_per_query
        if per_query is not None and decision.candidate_reuse is not None:
            console.print(
                f"              [dim]~{per_query:.0f} documents embedded per query "
                f"({decision.candidate_reuse:.0%} of each candidate set is already "
                f"cached; a lower bound)[/dim]"
            )
    geometry = getattr(result, "geometry", None)
    if geometry is not None and geometry.informative:
        console.print(
            f"  geometry    δ {geometry.delta:.4f}  "
            f"[dim]any orthogonal alignment lands within {geometry.bound:.3f} "
            f"(cosine ≥ {geometry.cosine_floor:.3f})[/dim]"
        )
    if decision.borderline:
        console.print("  [yellow]borderline — the interval spans a decision boundary[/yellow]")
    for warning in decision.warnings:
        console.print(f"  [yellow]{warning}[/yellow]")

    table = Table(title="Candidates", title_justify="left", header_style="bold")
    table.add_column("Adapter")
    table.add_column("ARR", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("Params", justify="right")
    for candidate in sorted(result.candidates, key=lambda c: c.arr, reverse=True):
        marker = " [green]*[/green]" if candidate.name == result.best.name else ""
        low, high = candidate.arr_ci
        table.add_row(
            f"{candidate.name}{marker}",
            f"{candidate.arr:.3f}",
            f"{low:.3f}–{high:.3f}",
            f"{candidate.n_params:,}",
        )
    console.print()
    console.print(table)

    if result.baselines:
        console.print()
        for name, value in sorted(result.baselines.items()):
            console.print(f"  [dim]{name.replace('_', ' ')}[/dim]  {value:.3f}")
