"""``rebasis migrate``, ``status``, ``rollback`` and ``gc``.

These are the commands that write. Every one of them takes the state lock, shows
what it will do before doing it, and records what it did.
"""

from __future__ import annotations

# Runtime import: typer resolves annotations at runtime, and _resume_defaults
# constructs a Path from the job row.
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.table import Table

from rebasis.cli._common import confirm, console, count_progress, handle_errors

if TYPE_CHECKING:
    from rebasis.core.base import BaseAdapter
    from rebasis.core.serialization import AdapterManifest
    from rebasis.manifest import JobRow
    from rebasis.migrate.engine import MigrationEngine, MigrationResult
    from rebasis.store.base import VectorStore

__all__ = ["gc_command", "migrate_command", "rollback_command", "status_command"]

#: Ids are streamed into the queue in chunks this size. Large enough that the
#: transaction overhead disappears, small enough that a huge corpus never has
#: its whole id list in memory.
ENQUEUE_CHUNK = 50_000


@handle_errors
def migrate_command(  # noqa: PLR0913, PLR0917 - each option is a documented CLI flag
    adapter: Annotated[
        Path | None,
        typer.Option("--adapter", help="Path to a .rbs adapter; --resume recovers it"),
    ] = None,
    store: Annotated[
        str | None, typer.Option("--store", help="Store URI; --resume recovers it")
    ] = None,
    priority: Annotated[
        str,
        typer.Option(
            "--priority",
            help=(
                "access = migrate the records you actually read first, so quality "
                "improves where you will notice it"
            ),
        ),
    ] = "none",
    access_log: Annotated[
        Path | None,
        typer.Option("--access-log", help='JSONL of {"id": ..., "count": ...}; --priority access'),
    ] = None,
    batch: Annotated[int, typer.Option("--batch", help="Records per batch")] = 256,
    limit: Annotated[int | None, typer.Option("--limit", help="Stop after this many")] = None,
    power_aware: Annotated[
        bool, typer.Option("--power-aware/--no-power-aware", help="Pause on low battery")
    ] = True,
    keep_original: Annotated[
        bool,
        typer.Option(
            "--keep-original/--no-keep-original",
            help="Keep a shadow copy so the migration can be rolled back",
        ),
    ] = True,
    max_memory: Annotated[
        str | None, typer.Option("--max-memory", help="Ceiling, e.g. 2GB")
    ] = None,
    resume: Annotated[
        str | None, typer.Option("--resume", help="Continue an existing job id")
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", "-n", help="Show the plan and stop, writing nothing"),
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation")] = False,
    no_input: Annotated[
        bool, typer.Option("--no-input", help="Never prompt; fail instead of asking")
    ] = False,
    state_dir: Annotated[Path | None, typer.Option("--state-dir")] = None,
) -> None:
    """Gradually rewrite the index with the new model's vectors.

    [EXPERIMENTAL] This is the only command that writes to your index, and it
    only upserts -- it never deletes. Every guarantee it makes is covered by a
    test against a real store, but none of them has been proved on a production
    index nobody could rebuild. Take a backup you can restore without rebasis,
    and try `--limit` on a slice first.
    """
    from rebasis.cli._pipeline import audit_writer_for, open_target_store
    from rebasis.core import load_adapter
    from rebasis.errors import ConfigError
    from rebasis.manifest import ADAPTERS_DIR, SHADOW_DIR, default_state_dir
    from rebasis.migrate import MigrationEngine
    from rebasis.storage import state_lock
    from rebasis.storage.budget import enforce_budget, estimate_budget
    from rebasis.store.base import require_capability

    directory = state_dir or default_state_dir()

    # The job already recorded both of these when it was created. Making the
    # user retype them is asking for the one thing they are least likely to
    # still have: a migration is resumed after an interruption, and the point of
    # the queue being the checkpoint is that nothing else has to survive it.
    if resume is not None:
        adapter, store = _resume_defaults(directory, resume, adapter, store)

    if adapter is None:
        raise ConfigError(
            "`migrate` needs --adapter.",
            hint="`rebasis migrate --adapter a.rbs --store chroma:///db#docs`",
        )
    if store is None:
        raise ConfigError(
            "`migrate` needs --store.",
            hint="`rebasis migrate --adapter a.rbs --store chroma:///db#docs`",
        )

    loaded, manifest, _ = load_adapter(adapter)
    opened = open_target_store(store)
    require_capability(opened, "can_upsert_vectors", operation="migrate")
    _check_dimensions(opened, loaded)
    # The lock is what keeps two of these out of one manifest.
    with state_lock(directory, operation="migrate"):
        writer = audit_writer_for(directory)
        engine = MigrationEngine(
            db=writer.db,
            store=opened,
            adapter=loaded,
            shadow_root=directory / SHADOW_DIR,
            job_id=resume,
            keep_original=keep_original,
            batch_size=batch,
            max_memory_bytes=_memory_ceiling(max_memory),
            power_aware=power_aware,
            audit=writer,
            store_uri=store,
            adapter_path=str(adapter),
        )

        if not keep_original:
            # Disabling rollback cannot be done quietly.
            console.print(
                "[red bold]Rollback is disabled for this job.[/red bold] If the result "
                "is not what you expected, the original vectors cannot be restored. "
                "This is recorded in the audit trail."
            )

        if resume is None:
            priorities = _read_access_log(access_log) if priority == "access" else None
            if priority == "access" and priorities is None:
                raise ConfigError(
                    "`--priority access` needs an access log.",
                    hint='Pass --access-log with one {"id": ..., "count": ...} object per line.',
                )
            queued = _enqueue_all(engine, opened, priorities=priorities)
        else:
            queued = engine.queue.stats().pending

        _preview(
            adapter=adapter,
            manifest=manifest,
            store=store,
            job_id=engine.job_id,
            queued=queued,
            keep_original=keep_original,
            state_dir=directory / ADAPTERS_DIR,
        )
        if not queued:
            console.print("[dim]Nothing to migrate.[/dim]")
            return

        # What it will cost, before it costs it. A migration that fills the
        # disk halfway through is not an error to handle but a design flaw
        # to prevent — and the shadow copy, which is what makes the job
        # reversible, is the first thing a full disk takes away.
        budget = estimate_budget(
            record_count=queued,
            dim=loaded.output_dim,
            state_dir=directory,
            keep_original=keep_original,
        )
        console.print()
        console.print(budget.render())
        console.print()
        enforce_budget(budget, directory)

        if dry_run:
            console.print("[dim]--dry-run: the plan above is all that happened.[/dim]")
            raise typer.Exit(code=0)

        if not confirm("Proceed?", assume_yes=yes, no_input=no_input):
            console.print("[yellow]Nothing was written.[/yellow]")
            raise typer.Exit(code=0)

        # X of Y over the queue: the total is known before the first batch, so
        # there is no reason to show a spinner that cannot say how far in it is.
        with count_progress(limit if limit is not None else queued, "migrating") as counter:
            result = engine.run(limit=limit, on_batch=counter.advance)
        _report_run(result)


def _emit_jobs(jobs: list[tuple[JobRow, Any]], *, as_json: bool) -> None:
    """`status` for something other than a person.

    A Rich table renders box-drawing characters and truncates ids with an
    ellipsis, so the human view is actively hostile to `grep` and `cut`. These
    two carry the full id and no formatting.
    """
    payload = [
        {
            "job_id": job.job_id,
            "state": job.state,
            "adapter_type": job.adapter_type,
            "adapter_path": job.adapter_path,
            "store_uri": job.store_uri,
            "progress": round(stats.completed_fraction, 4),
            "done": stats.done,
            "failed": stats.failed,
            "total": stats.total,
            "rollback": "available" if job.reversible else job.state,
            "created_utc": job.created_utc,
            "updated_utc": job.updated_utc,
        }
        for job, stats in jobs
    ]
    if as_json:
        import json

        console.print_json(json.dumps(payload))
        return
    for row in payload:
        console.print(
            "\t".join(
                str(row[key])
                for key in ("job_id", "state", "progress", "done", "failed", "total", "rollback")
            ),
            highlight=False,
            markup=False,
        )


def _rollback_column(job: JobRow) -> str:
    """What `status` says in the rollback column.

    Three outcomes, not two. "disabled" is right for a job that never kept a
    shadow; for one that kept a shadow and already used it, the copy is spent,
    and saying it was disabled describes a choice the user did not make.
    """
    if job.reversible:
        return "available"
    if job.state == "rolled_back":
        return "[dim]spent[/dim]"
    return "[yellow]disabled[/yellow]"


def _rolled_back_jobs(directory: Path) -> list[str]:
    """Jobs whose shadow copy has already been spent.

    A read with no lock, and a missing or unreadable manifest answers "none":
    `gc` listing less than it could is a worse outcome than `gc` refusing to
    run at all.
    """
    from rebasis.manifest import ManifestDB, manifest_path

    path = manifest_path(directory)
    if not path.exists():
        return []
    db = ManifestDB(path)
    try:
        rows = db.query("SELECT job_id FROM jobs WHERE state = ?", ("rolled_back",))
    except Exception:  # noqa: BLE001 - see the docstring; gc must still run
        return []
    finally:
        db.close()
    return [str(row["job_id"]) for row in rows]


def _resume_defaults(
    directory: Path, job_id: str, adapter: Path | None, store: str | None
) -> tuple[Path | None, str | None]:
    """Fill in ``--adapter`` and ``--store`` from the job being resumed.

    Anything passed explicitly wins: resuming with a different adapter is a
    mistake worth making loudly rather than one to silently override.
    """
    from rebasis.errors import ConfigError
    from rebasis.manifest import JobRow, ManifestDB, manifest_path

    path = manifest_path(directory)
    if not path.exists():
        raise ConfigError(
            f"No rebasis state at {directory}.",
            hint="`rebasis status` lists the jobs in a state directory.",
            context={"job_id": job_id},
        )

    # A read, so no lock: `status` does the same, and blocking behind a
    # migration to answer "what was this job?" would be the worse trade.
    db = ManifestDB(path)
    try:
        rows = db.query("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
    finally:
        db.close()
    if not rows:
        raise ConfigError(
            f"No migration job named {job_id!r}.",
            hint="`rebasis status` lists the jobs in this state directory.",
            context={"job_id": job_id},
        )

    job = JobRow.from_row(rows[0])
    recovered_adapter = adapter or (Path(job.adapter_path) if job.adapter_path else None)
    recovered_store = store or job.store_uri or None
    if recovered_adapter is None:
        raise ConfigError(
            f"Job {job_id} did not record which adapter it used.",
            hint="Pass the same --adapter the migration used.",
            context={"job_id": job_id},
        )
    if recovered_store is None:
        raise ConfigError(
            f"Job {job_id} did not record which store it wrote to.",
            hint="Pass the same --store the migration used.",
            context={"job_id": job_id},
        )
    return recovered_adapter, recovered_store


def _check_dimensions(store: VectorStore, adapter: BaseAdapter) -> None:
    """Refuse a dimension mismatch before writing anything.

    A store with a locked dimension rejects the write anyway, but it rejects it
    halfway through — after the shadow copy exists and part of the index has
    changed. Checking first turns that into an error message.
    """
    from rebasis.errors import StoreDimensionMismatch

    store_dim = store.dimension()
    output_dim = adapter.output_dim
    if store_dim and output_dim and store_dim != output_dim:
        raise StoreDimensionMismatch(
            f"The adapter writes {output_dim}-dimensional vectors, but the "
            f"collection holds {store_dim}-dimensional ones.",
            hint=(
                "This adapter was fitted against a different index. Check the "
                "collection in the store URI, or re-fit against this one."
            ),
            context={"dim": store_dim},
        )


def _enqueue_all(
    engine: MigrationEngine, store: VectorStore, *, priorities: dict[str, float] | None
) -> int:
    """Stream every id into the queue, a chunk at a time.

    Never builds the full id list: on a five-million-record index that alone is
    hundreds of megabytes of Python strings, and peak memory has to stay
    ``O(batch × d)`` regardless of corpus size.
    """
    total = store.count()
    queued = 0
    chunk: list[str] = []
    for record in store.iter_records(with_vectors=False, with_text=False):
        chunk.append(record.id)
        if len(chunk) >= ENQUEUE_CHUNK:
            queued += engine.prepare(chunk, priorities=priorities, total=total)
            chunk = []
    if chunk:
        queued += engine.prepare(chunk, priorities=priorities, total=total)
    return queued


def _read_access_log(path: Path | None) -> dict[str, float] | None:
    """Read access counts, so hot records migrate first."""
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


def _memory_ceiling(value: str | None) -> int | None:
    """The ceiling this run should respect.

    `--max-memory` when given, otherwise `REBASIS_MAX_MEMORY`. The environment
    variable has to work as well as the flag; it was parsed into settings,
    printed by `doctor`, and consulted nowhere.
    """
    if value is not None:
        return _parse_memory(value)
    from rebasis.config import settings

    return settings().max_memory_bytes


def _parse_memory(value: str | None) -> int | None:
    """Parse ``2GB`` / ``512MB`` / a plain byte count."""
    if value is None:
        return None

    from rebasis.errors import ConfigError

    text = value.strip().upper().removesuffix("B")
    units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    multiplier = units.get(text[-1:], 1)
    number = text[:-1] if multiplier > 1 else text
    try:
        return int(float(number) * multiplier)
    except ValueError as exc:
        raise ConfigError(
            f"Could not read {value!r} as a memory ceiling.",
            hint="Use a form like 2GB, 512MB, or a plain number of bytes.",
            cause=exc,
        ) from exc


def _preview(  # noqa: PLR0913 - the preview names every input it shows
    *,
    adapter: Path,
    manifest: AdapterManifest,
    store: str,
    job_id: str,
    queued: int,
    keep_original: bool,
    state_dir: Path,
) -> None:
    """Show what will happen before it happens."""
    del state_dir
    console.print()
    console.print(
        "[yellow]experimental[/yellow] [dim]— tested against every backend, not yet "
        "proved at production scale. Take a backup rebasis is not part of.[/dim]"
    )
    console.print(f"[bold]migrate[/bold]  {queued:,} records")
    console.print(f"  store       {store}")
    console.print(f"  adapter     {adapter.name} ({manifest.adapter_type})")
    console.print(f"  models      {manifest.old_model_id} → {manifest.new_model_id}")
    console.print(f"  job         {job_id}")
    console.print(f"  rollback    {'available' if keep_original else '[red]disabled[/red]'}")
    console.print()


def _report_run(result: MigrationResult) -> None:
    """Print what the run actually did."""
    console.print()
    console.print(f"[bold]{result.state}[/bold]  {result.processed:,} records")
    if result.failed:
        console.print(f"  [red]{result.failed:,} failed[/red]")
    if result.pause_reason:
        console.print(f"  [yellow]paused: {result.pause_reason}[/yellow]")
        console.print(f"  [dim]resume with `rebasis migrate --resume {result.job_id}`[/dim]")
    console.print(f"  [dim]{result.duration_seconds:.1f}s[/dim]")


@handle_errors
def status_command(
    job_id: Annotated[str | None, typer.Argument(help="A job id; omit to list all")] = None,
    state_dir: Annotated[Path | None, typer.Option("--state-dir")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit the jobs as JSON")] = False,
    plain: Annotated[
        bool, typer.Option("--plain", help="One job per line, tab separated, no table")
    ] = False,
) -> None:
    """Show migration progress.

    Takes no lock, so it can be run while a migration is in flight —
    which is exactly when it is wanted.
    """
    from rebasis.manifest import JobRow, ManifestDB, default_state_dir, manifest_path
    from rebasis.migrate import JobQueue

    directory = state_dir or default_state_dir()
    path = manifest_path(directory)
    if not path.exists():
        console.print(f"[dim]No rebasis state at {directory}.[/dim]")
        return

    db = ManifestDB(path)
    rows = db.query(
        "SELECT * FROM jobs WHERE (? IS NULL OR job_id = ?) ORDER BY created_utc DESC",
        (job_id, job_id),
    )
    if not rows:
        console.print("[dim]No migration jobs.[/dim]")
        return

    parsed = [JobRow.from_row(raw) for raw in rows]
    jobs = [(job, JobQueue(db, job.job_id).stats()) for job in parsed]

    if as_json or plain:
        _emit_jobs(jobs, as_json=as_json)
        db.close()
        return

    table = Table(title="Migration jobs")
    table.add_column("job")
    table.add_column("state")
    table.add_column("adapter")
    table.add_column("progress", justify="right")
    table.add_column("done", justify="right")
    table.add_column("failed", justify="right")
    table.add_column("rollback")

    for job, stats in jobs:
        table.add_row(
            job.job_id,
            job.state,
            job.adapter_type,
            f"{stats.completed_fraction:.0%}",
            f"{stats.done:,}",
            f"{stats.failed:,}" if stats.failed else "—",
            _rollback_column(job),
        )
    console.print(table)
    db.close()


@handle_errors
def rollback_command(
    job_id: Annotated[str, typer.Argument(help="The job to undo")],
    state_dir: Annotated[Path | None, typer.Option("--state-dir")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation")] = False,
    no_input: Annotated[
        bool, typer.Option("--no-input", help="Never prompt; fail instead of asking")
    ] = False,
) -> None:
    """Restore the vectors a migration replaced, from its shadow copy.

    The shadow is bit-identical when it was written at float32, which is the
    default. What lands back in the index is that, put through the store's own
    upsert — exact for a store that stores what it is given, and within one
    float32 ulp for one that normalises on write, such as Chroma in cosine
    space.
    """
    from rebasis.cli._pipeline import audit_writer_for, open_target_store
    from rebasis.errors import ConfigError
    from rebasis.manifest import SHADOW_DIR, JobRow, default_state_dir
    from rebasis.migrate import MigrationEngine
    from rebasis.storage import state_lock

    directory = state_dir or default_state_dir()
    # Two writers in one manifest is the failure this prevents.
    with state_lock(directory, operation="rollback"):
        writer = audit_writer_for(directory)
        rows = writer.db.query("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        if not rows:
            raise ConfigError(
                f"No migration job named {job_id!r}.",
                hint="`rebasis status` lists the jobs in this state directory.",
                context={"job_id": job_id},
            )
        job = JobRow.from_row(rows[0])
        if not job.reversible:
            raise ConfigError(
                f"Job {job_id} ran with --no-keep-original, so there is no shadow copy.",
                hint="Nothing can be restored. The original vectors were not kept.",
                context={"job_id": job_id},
            )

        store_uri = job.store_uri
        if not store_uri:
            raise ConfigError(
                f"Job {job_id} did not record which store it wrote to.",
                hint="Pass the same --store the migration used.",
                context={"job_id": job_id},
            )

        console.print(f"[bold]rollback[/bold]  job {job_id}")
        console.print(f"  store     {store_uri}")
        console.print("  restores  the original vectors from the shadow copy")
        console.print()
        if not confirm("Proceed?", assume_yes=yes, no_input=no_input):
            console.print("[yellow]Nothing was written.[/yellow]")
            raise typer.Exit(code=0)

        opened = open_target_store(store_uri)
        engine = MigrationEngine(
            db=writer.db,
            store=opened,
            adapter=_noop_adapter(opened.dimension()),
            shadow_root=directory / SHADOW_DIR,
            job_id=job_id,
            audit=writer,
            store_uri=store_uri,
        )
        restored = engine.rollback()
        console.print(f"[green]Restored[/green] {restored:,} records")


def _noop_adapter(dim: int) -> BaseAdapter:
    """Rollback writes the shadow back verbatim; no adapter is applied.

    The engine still wants one, because every other path needs it — so this is
    the identity, which is exactly what "restore what was there" means.
    """
    from rebasis.core import IdentityAdapter

    return IdentityAdapter(input_dim=dim, output_dim=dim)


@handle_errors
def gc_command(  # noqa: PLR0913, PLR0917 - each option is a documented CLI flag
    apply: Annotated[
        bool, typer.Option("--apply", help="Actually remove; default is a dry run")
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            "-n",
            help="Force the dry run. Already the default; here because -n is what people type",
        ),
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Emit the plan as JSON")] = False,
    job: Annotated[
        str | None, typer.Option("--job", help="Also remove this job's shadow copy")
    ] = None,
    i_understand: Annotated[
        bool,
        typer.Option(
            "--i-understand",
            help="Required to remove a shadow copy: the job becomes irreversible",
        ),
    ] = False,
    state_dir: Annotated[Path | None, typer.Option("--state-dir")] = None,
) -> None:
    """List what can be cleaned up; pass --apply to remove it.

    A dry run by default. A garbage collector that deletes without being
    asked is the data-loss class it exists to prevent.
    """
    from rebasis.errors import ConfigError
    from rebasis.manifest import default_state_dir
    from rebasis.storage import apply_gc, plan_gc

    if apply and dry_run:
        raise ConfigError(
            "`--apply` and `--dry-run` ask for opposite things.",
            hint="Drop one. Without `--apply`, `gc` is already a dry run.",
        )

    directory = state_dir or default_state_dir()
    plan = plan_gc(
        directory,
        include_shadows=[job] if job else [],
        spent_shadows=_rolled_back_jobs(directory),
    )

    if as_json:
        import json

        console.print_json(json.dumps(plan.to_dict()))
        if not apply:
            return
    else:
        console.print(plan.render())
        if not apply:
            return

    needs_confirmation = any(c.requires_confirmation for c in plan.candidates)
    if needs_confirmation and not i_understand:
        console.print(
            "\n[yellow]Removing a shadow copy makes that migration permanently "
            "irreversible. Pass --i-understand to proceed.[/yellow]"
        )
        raise typer.Exit(code=2)

    # Only the destructive half takes the lock. The dry run above is a
    # read, and making "what would you delete?" wait behind a running migration
    # would be a worse answer than showing it.
    from rebasis.storage import state_lock

    with state_lock(directory, operation="gc"):
        freed = apply_gc(plan, confirmed=i_understand)
    console.print(f"\n[green]Freed[/green] {freed / 1024**2:.1f} MB")
