"""``rebasis audit`` — inspect and verify the decision record."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 - typer resolves annotations at runtime
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.table import Table

from rebasis.cli._common import console, handle_errors
from rebasis.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rebasis.audit import AuditRecord
    from rebasis.cli._pipeline import ProfileOverrides
    from rebasis.probe import ProbeResult

__all__ = ["audit_app"]

audit_app = typer.Typer(
    name="audit",
    help="Inspect the decision record. Audit entries are never deleted automatically.",
    no_args_is_help=True,
)


def _reader(state_dir: Path | None) -> object:
    from rebasis.audit import AuditReader
    from rebasis.manifest import ManifestDB, default_state_dir, manifest_path

    directory = state_dir or default_state_dir()
    return AuditReader(ManifestDB(manifest_path(directory)))


@audit_app.command("list")
@handle_errors
def list_records(
    action: Annotated[str | None, typer.Option("--action", help="Filter by event name")] = None,
    since: Annotated[str | None, typer.Option("--since", help="ISO date lower bound")] = None,
    limit: Annotated[int, typer.Option("--limit")] = 20,
    state_dir: Annotated[Path | None, typer.Option("--state-dir")] = None,
) -> None:
    """List audit records, most recent first."""
    reader = _reader(state_dir)
    records = reader.list_records(action=action, since=since, limit=limit)  # type: ignore[attr-defined]

    if not records:
        console.print("[dim]No audit records match.[/dim]")
        return

    table = Table(title="Audit record")
    table.add_column("seq", justify="right")
    table.add_column("when")
    table.add_column("action")
    table.add_column("subject")
    table.add_column("outcome")

    for record in records:
        outcome = record.outputs.get("decision") or record.outputs.get("state") or ""
        if not outcome and "count" in record.outputs:
            outcome = f"{record.outputs['count']} records"
        table.add_row(
            str(record.seq),
            record.ts_utc[:19],
            record.action,
            (record.subject or "")[:34],
            str(outcome),
        )
    console.print(table)


@audit_app.command("show")
@handle_errors
def show_record(
    seq: Annotated[int, typer.Argument(help="Sequence number")],
    state_dir: Annotated[Path | None, typer.Option("--state-dir")] = None,
) -> None:
    """Show one record in full: everything needed to defend the decision."""
    reader = _reader(state_dir)
    record = reader.get(seq)  # type: ignore[attr-defined]
    if record is None:
        console.print(f"[yellow]No record with seq {seq}.[/yellow]")
        raise typer.Exit(code=1)

    console.print(f"[bold]seq {record.seq}[/bold]  {record.ts_utc}")
    console.print(f"  action     {record.action}")
    console.print(f"  actor      {record.actor}")
    console.print(f"  run_id     {record.run_id}")
    console.print(f"  subject    {record.subject or '—'}")
    console.print(f"  version    {record.rebasis_version}")
    console.print("\n  [bold]inputs[/bold]")
    console.print(json.dumps(dict(record.inputs), indent=4))
    console.print("  [bold]outputs[/bold]")
    console.print(json.dumps(dict(record.outputs), indent=4))
    console.print("  [bold]environment[/bold]")
    console.print(json.dumps(dict(record.env), indent=4))
    console.print(f"\n  hash       {record.hash}")
    console.print(f"  prev_hash  {record.prev_hash}")


@audit_app.command("verify")
@handle_errors
def verify_chain_command(
    state_dir: Annotated[Path | None, typer.Option("--state-dir")] = None,
) -> None:
    """Verify the hash chain end to end.

    Tamper-evident, not tamper-proof. This catches accidental
    corruption and silent loss; it is not a defence against a determined local
    attacker, and claiming otherwise would be false.
    """
    reader = _reader(state_dir)
    result = reader.verify()  # type: ignore[attr-defined]

    if result.valid:
        console.print(f"[green]OK[/green]  {result.describe()}")
        return
    console.print(f"[red]BROKEN[/red]  {result.describe()}")
    console.print(
        "\n[dim]A record was altered or removed. The chain is tamper-evident, so "
        "this tells you something changed — not who changed it.[/dim]"
    )
    raise typer.Exit(code=3)


@audit_app.command("replay")
@handle_errors
def replay_record(
    seq: Annotated[int, typer.Argument(help="Sequence number of a probe decision")],
    state_dir: Annotated[Path | None, typer.Option("--state-dir")] = None,
    store: Annotated[
        str | None,
        typer.Option("--store", help="Override the store URI the record names"),
    ] = None,
    device: Annotated[str, typer.Option("--device", help="auto|cpu|cuda|cuda:N|mps")] = "auto",
) -> None:
    """Re-run a recorded decision and compare it with what was recorded.

    A difference means either a regression or a changed corpus, and both are
    worth knowing. Across devices an equivalence band is used rather than exact
    agreement, and the result says which was applied.
    """
    from rebasis.audit import compare_replay, replayable_inputs

    reader = _reader(state_dir)
    record = reader.get(seq)  # type: ignore[attr-defined]
    if record is None:
        console.print(f"[yellow]No record with seq {seq}.[/yellow]")
        raise typer.Exit(code=1)

    replayable, reason = replayable_inputs(record)
    if not replayable:
        console.print(f"[yellow]Not replayable:[/yellow] {reason}")
        raise typer.Exit(code=2)

    store_uri = store or record.subject
    if not store_uri:
        raise ConfigError(
            f"Record {seq} does not name a store, so there is nothing to replay against.",
            hint="Pass --store with the URI the original run used.",
            context={"count": seq},
        )

    console.print(f"[bold]Replaying seq {seq}[/bold]  ({record.action})")
    console.print(f"  recorded decision  {record.outputs.get('decision')}")
    console.print(f"  recorded ARR       {record.outputs.get('arr_r10')}")
    console.print(f"  recorded device    {record.env.get('device')}")
    console.print(f"  store              {store_uri}")

    result = _rerun(record, store_uri=store_uri, device=device)
    comparison = compare_replay(
        record,
        replayed_decision=result.decision.decision,
        replayed_arr=result.best.arr,
    )

    console.print()
    console.print(f"  replayed decision  {result.decision.decision}")
    console.print(f"  replayed ARR       {result.best.arr:.4f}")
    console.print()
    colour = {"matched": "green", "matched_cross_device": "green"}.get(comparison.verdict, "yellow")
    console.print(f"[bold {colour}]{comparison.verdict}[/bold {colour}]")
    for note in comparison.notes:
        console.print(f"  [dim]{note}[/dim]")

    if comparison.verdict == "differed":
        # Exit 3, not 0: a difference is a domain result the caller should be
        # able to branch on. It means either a regression or a changed corpus,
        # and both are things you want a script to notice.
        raise typer.Exit(code=3)


def _rerun(record: AuditRecord, *, store_uri: str, device: str) -> ProbeResult:
    """Re-run a recorded probe with the inputs the record carries.

    Only the inputs are replayed, never the outputs. Everything the decision
    depended on — the two models, the sampling strategy and seed, the sample
    size, the cut-off — is in the record; anything not in the record is not
    allowed to influence the result, which is what makes the comparison mean
    something.
    """
    from rebasis.cli._pipeline import open_embedders, open_target_store
    from rebasis.probe.session import probe_store

    inputs = record.inputs
    new_model = str(inputs.get("new_model", ""))
    if not new_model:
        raise ConfigError(
            "The record does not name the candidate model.",
            hint="Records written before this field existed cannot be replayed.",
        )

    # Rebuilt from the record, never re-resolved from the profile table: the
    # table may have gained an entry since, and a replay that quietly used a
    # different encoding would report a difference that is not a regression.
    overrides = _profile_overrides(inputs)
    _, new_embedder = open_embedders(None, new_model, device=device, new_overrides=overrides)
    result, _ = probe_store(
        open_target_store(store_uri),
        new_embedder,
        size=int(inputs.get("count", 10_000)),
        heldout=int(inputs.get("n_queries", 1_000)),
        strategy=str(inputs.get("strategy", "stratified")),
        k=int(inputs.get("k", 10)),
        seed=int(inputs.get("seed", 0)),
    )
    return result


@audit_app.command("export")
@handle_errors
def export_records(
    out: Annotated[Path | None, typer.Option("--out", help="Write here instead of stdout")] = None,
    state_dir: Annotated[Path | None, typer.Option("--state-dir")] = None,
) -> None:
    """Export the whole trail as JSON Lines."""
    reader = _reader(state_dir)
    payload = reader.export_jsonl()  # type: ignore[attr-defined]

    if out is None:
        console.print(payload)
        return
    from rebasis.storage import atomic_write_text

    atomic_write_text(out, payload)
    console.print(f"[green]Written[/green] {out}")


@audit_app.command("prune")
@handle_errors
def prune_records(
    before: Annotated[
        str, typer.Option("--before", help="Remove records older than this date (YYYY-MM-DD)")
    ],
    i_understand: Annotated[
        bool,
        typer.Option("--i-understand", help="Required: pruning is not reversible"),
    ] = False,
    state_dir: Annotated[Path | None, typer.Option("--state-dir")] = None,
) -> None:
    """Remove audit records older than a date, leaving a gap record behind.

    An audit trail that can only grow eventually gets deleted wholesale, which
    is worse than pruning it deliberately. So this exists — and it writes a
    record of its own saying where the gap is, so `audit verify` reads a prune
    as a prune rather than as tampering.

    A dry run first. `--i-understand` is required because the records that go
    are gone: they are the evidence, and there is no copy.
    """
    import datetime

    from rebasis.audit import AuditWriter
    from rebasis.errors import ConfigError
    from rebasis.manifest import ManifestDB, default_state_dir, manifest_path
    from rebasis.observability import Events

    try:
        cutoff = datetime.datetime.strptime(before, "%Y-%m-%d").replace(tzinfo=datetime.UTC)
    except ValueError as exc:
        raise ConfigError(
            f"{before!r} is not a date.",
            hint="Use YYYY-MM-DD, for example --before 2026-01-01.",
            context={"before": before},
        ) from exc

    directory = state_dir or default_state_dir()
    db = ManifestDB(manifest_path(directory))
    boundary = cutoff.isoformat()
    doomed = db.query(
        "SELECT seq FROM audit_records WHERE ts_utc < ? ORDER BY seq ASC", (boundary,)
    )
    if not doomed:
        console.print(f"[dim]No audit records older than {before}.[/dim]")
        return

    last_seq = int(doomed[-1]["seq"])
    console.print(f"[bold]prune[/bold]  {len(doomed):,} records older than {before}")
    console.print(f"  highest sequence removed  {last_seq}")
    console.print("  [dim]the remaining chain stays verifiable across the gap[/dim]")

    if not i_understand:
        console.print(
            "\n[yellow]These records are the evidence, and there is no copy. "
            "Pass --i-understand to proceed.[/yellow]"
        )
        raise typer.Exit(code=2)

    # The gap record goes in *first*: it has to be inside the chain, linked to
    # the last surviving record, or it is just a claim about a chain it is not
    # part of.
    writer = AuditWriter(db, run_id=f"prune-{last_seq}")
    writer.write(
        Events.AUDIT_CHAIN_VERIFIED,
        inputs={"before": before},
        outputs={"gap_after_seq": last_seq, "count": len(doomed)},
        subject="audit",
        force=True,
    )
    db.execute("DELETE FROM audit_records WHERE ts_utc < ? AND seq <= ?", (boundary, last_seq))
    console.print(f"\n[green]Pruned[/green] {len(doomed):,} records")


def _profile_overrides(inputs: Mapping[str, Any]) -> ProfileOverrides:
    """Rebuild the recorded encoding profile as override flags.

    Records written before the profile was stored carry only a fingerprint. For
    those, the overrides come back empty and resolution falls through to the
    table — which works for a known model and fails clearly for one that was
    described by flags at the time.
    """
    from rebasis.cli._pipeline import ProfileOverrides

    profile = inputs.get("new_profile")
    if not isinstance(profile, dict):
        return ProfileOverrides()
    return ProfileOverrides(
        dim=profile.get("dim"),
        query_prefix=profile.get("query_prefix"),
        document_prefix=profile.get("document_prefix"),
        pooling=profile.get("pooling"),
        matryoshka_dim=profile.get("matryoshka_dim"),
    )
