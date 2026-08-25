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


def print_next_step_after_probe(result: Any, *, store: str, old: str, new: str) -> None:
    """Print the command this decision implies."""
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
        console.print(
            f"  cascade     [bold]{decision.cascade_arr:.3f}[/bold] retained at candidate "
            f"depth  [dim](break-even {cascade:.2f}x if the new model reranks; "
            f"measured, not served)[/dim]"
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
