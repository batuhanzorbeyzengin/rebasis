"""``rebasis migrate``, ``pause``, ``resume``, ``status``, ``rollback`` and ``gc``.

These are the commands that write. Every one that touches the index takes the
state lock, shows what it will do before doing it, and records what it did.

``status`` and ``pause`` deliberately take no lock, because a running migration
holds it for its whole run and both of them exist to be used *while* one is
running. ``status`` only reads. ``pause`` writes one column — ``pause_requested``
— that nothing else writes, so there is no second writer to serialise against;
what state a job is in remains the engine's word alone.
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
    from rebasis.types import Embedder, EncodingProfile

__all__ = [
    "gc_command",
    "migrate_command",
    "pause_command",
    "resume_command",
    "rollback_command",
    "status_command",
]

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
    health_check: Annotated[
        bool,
        typer.Option(
            "--health-check/--no-health-check",
            help=(
                "Measure what the index's own search returns against exact kNN, "
                "before and after. Costs two scans of the collection"
            ),
        ),
    ] = True,
    rebuild_index: Annotated[
        bool,
        typer.Option(
            "--rebuild-index",
            help=(
                "When the run finishes, ask the store to rebuild its search "
                "structure. Only where the backend supports it"
            ),
        ),
    ] = False,
    refit: Annotated[
        bool,
        typer.Option(
            "--refit",
            help=(
                "Periodically refit the adapter on records not yet migrated, "
                "adopting the result only if it wins. Re-embeds documents"
            ),
        ),
    ] = False,
    refit_every: Annotated[
        int, typer.Option("--refit-every", help="Records between refit attempts")
    ] = 50_000,
    refit_pairs: Annotated[
        int, typer.Option("--refit-pairs", help="Records sampled and re-embedded per attempt")
    ] = 1000,
    device: Annotated[
        str, typer.Option("--device", help="Where to run the embedder --refit needs")
    ] = "auto",
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
    from rebasis.migrate import MigrationEngine, RefitPolicy
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
    # Before the store is even opened: this one needs nothing but the manifest,
    # and the cheapest refusal is the one that happens first.
    _check_direction(manifest, adapter)
    opened = open_target_store(store)
    require_capability(opened, "can_upsert_vectors", operation="migrate")
    _check_dimensions(opened, loaded)
    # The lock is what keeps two of these out of one manifest.
    with state_lock(directory, operation="migrate"):
        writer = audit_writer_for(directory)
        embedder, profiles = _refit_collaborators(manifest, opened, enabled=refit, device=device)
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
            refit=RefitPolicy(
                enabled=refit,
                every_n_records=refit_every,
                sample_size=refit_pairs,
            ),
            embedder=embedder,
            adapter_root=directory / ADAPTERS_DIR,
            profiles=profiles,
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
            partial=limit is not None and limit < queued,
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
        _note_background_reindex(opened)
        _note_quantized_store(opened, dry_run=dry_run)
        console.print()
        enforce_budget(budget, directory)

        if dry_run:
            console.print("[dim]--dry-run: the plan above is all that happened.[/dim]")
            raise typer.Exit(code=0)

        if not confirm("Proceed?", assume_yes=yes, no_input=no_input):
            console.print("[yellow]Nothing was written.[/yellow]")
            raise typer.Exit(code=0)

        # Measured before a single vector changes, so the comparison afterwards
        # is against this index rather than against an assumption about it.
        # Some collections start below 1.0 — an HNSW index is approximate by
        # construction — and a drop only means something against where it began.
        health_before = _measure_health(opened, enabled=health_check)

        # X of Y over the queue: the total is known before the first batch, so
        # there is no reason to show a spinner that cannot say how far in it is.
        with count_progress(limit if limit is not None else queued, "migrating") as counter:
            result = engine.run(limit=limit, on_batch=counter.advance)

        _report_aftermath(
            engine,
            result,
            store=store,
            db=writer.db,
            health_before=health_before,
            health_check=health_check,
            rebuild_index=rebuild_index,
        )


def _emit_jobs(
    jobs: list[tuple[JobRow, Any]], *, as_json: bool, mixed: dict[str, Any] | None = None
) -> None:
    """`status` for something other than a person.

    A Rich table renders box-drawing characters and truncates ids with an
    ellipsis, so the human view is actively hostile to `grep` and `cut`. These
    two carry the full id and no formatting.

    ``mixed_space`` is in the payload rather than only in the human view because
    it is the field a script most needs to branch on: a CI job that queries an
    index after migrating a slice of it should be able to fail rather than
    quietly measure the wrong thing.
    """
    mixed = mixed or {}
    payload = [
        {
            "job_id": job.job_id,
            "state": job.state,
            # Separate from `state` rather than folded into it: a script that
            # branches on "running" must keep working, and this is a second fact
            # about a running job rather than a different state.
            "pause_requested": job.pause_requested,
            "adapter_type": job.adapter_type,
            "adapter_path": job.adapter_path,
            "store_uri": job.store_uri,
            "progress": round(stats.completed_fraction, 4),
            "done": stats.done,
            "failed": stats.failed,
            "total": stats.total,
            "rollback": "available" if job.reversible else job.state,
            "mixed_space": (mixed[job.job_id].to_dict() if job.job_id in mixed else None),
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
            )
            + f"\t{'mixed' if row['mixed_space'] else 'single'}",
            highlight=False,
            markup=False,
        )


def _measure_health(store: VectorStore, *, enabled: bool) -> Any:
    """The index's recall against exact kNN, or ``None`` when not asked for.

    Never fatal. This is a diagnostic on top of a migration, and a backend whose
    `search` behaves unexpectedly must not be the reason a migration does not
    happen — the vectors are what matter, and every guarantee about them is
    checked elsewhere.
    """
    if not enabled:
        return None

    from rebasis.migrate import measure_index_health

    with console.status("Measuring what the index finds…"):
        try:
            return measure_index_health(store)
        except Exception as exc:  # noqa: BLE001 - see the docstring
            console.print(f"[dim]index health check skipped: {exc}[/dim]")
            return None


def _report_health(store: VectorStore, before: Any, *, enabled: bool) -> None:
    """Measure again and say what changed.

    The one thing the rest of `migrate` cannot see. The read-back proves the
    store took the write and the fresh-connection check proves it kept it;
    neither proves the record can still be *retrieved*. A graph index chose its
    edges from the geometry the old vectors had, and rewriting the vectors does
    not rewrite the graph.
    """
    if not enabled or before is None:
        return

    after = _measure_health(store, enabled=True)
    if after is None:
        return

    from rebasis.migrate import HealthComparison

    comparison = HealthComparison(before=before, after=after)
    console.print()
    if comparison.delta < 0:
        console.print(f"[yellow]index[/yellow]  {comparison.explain()}")
        # Measured both ways: on a 100,000-record collection a rebuild
        # recovered the whole loss from an unconstrained affine adapter, and
        # none of it from a low-rank one, because those are different failures
        # wearing the same number (docs/index-health.md). So this offers the
        # move rather than promising the outcome.
        if store.capabilities.can_rebuild_index:
            console.print(
                "       [dim]this backend can rebuild its index: re-run with "
                "--rebuild-index, or trigger it yourself.[/dim]"
            )
    else:
        console.print(f"[dim]index  {comparison.explain()}[/dim]")


def _note_background_reindex(store: VectorStore) -> None:
    """Say that the plan above is missing a line, on the backends where it is.

    Measured: on a Qdrant server the same 100,000-record migration took 181-199
    seconds against Chroma's 89-145 on the same host, and the collection kept
    reindexing in the background after the run returned — `status: yellow` with
    `indexed_vectors_count` above the point count, which is a rebuild over
    overlapping segments. The estimate comes from throughput and has no line for
    that work.

    Not a warning. A backend that rebuilds is doing the right thing: it is why
    Qdrant loses almost nothing where Chroma loses five to eight points of
    recall. The cost simply lands in wall-clock rather than in quality, and the
    plan should say which.
    """
    if not store.capabilities.can_rebuild_index:
        return
    console.print(
        "  [dim]This backend rebuilds its search index as vectors change. That work "
        "is not in the estimate above, and some of it continues after the run "
        "finishes.[/dim]"
    )


def _note_quantized_store(store: VectorStore, *, dry_run: bool) -> None:
    """Say what rollback is worth here, on a store that does not keep what it is given.

    It does not refuse, and it is not asking a question. A quantized index is a
    deliberate engineering choice and its owner has as much right to migrate it
    as anyone; what they do not have without this is a correct reading of the
    sentence `rollback` is sold on. So the plan states the difference and the
    run continues.

    **What changes.** rebasis shadows what the store *returns*, and a store that
    quantizes returns a value decoded from its stored code rather than the value
    that was written to it. The shadow is still bit-identical — to that decoded
    view. `rollback` therefore restores the vectors this collection reads back
    today, which is the state the migration replaced; it does not recover
    precision the collection had already spent before rebasis was involved.

    Three states, and the third is handled differently on purpose.
    ``False`` says nothing, because there is nothing to say. ``True`` says it
    every time. ``None`` — the store could not be asked — is not a finding, and
    a caveat printed on every unknown is a caveat nobody reads; it appears only
    under `--dry-run`, which is where the user has explicitly asked for
    everything the plan knows.
    """
    quantized = store.capabilities.quantized
    if quantized is None:
        if dry_run:
            console.print(
                "  [dim]Whether this backend stores its vectors quantized could not be "
                "determined, so the rollback guarantee above is the one for a store "
                "that keeps what it is given.[/dim]"
            )
        return
    if not quantized:
        return

    console.print(
        "  [yellow]This collection stores its vectors quantized.[/yellow] [dim]rebasis "
        "reads the store's decoded view of them, and that view is what the shadow "
        "copy holds — so `rollback` restores what this collection reads back today, "
        "not what your embedding model produced. That gap was there before rebasis "
        "ran.[/dim]"
    )
    if dry_run:
        # The tolerance is interpolated, never spelled out. It is a constant in
        # the engine, and a second copy of it here is the copy nobody updates —
        # the message would go on quoting a number the check no longer uses.
        from rebasis.migrate.engine import VERIFY_ATOL

        console.print(
            "  [dim]The same applies going the other way: each migrated vector is "
            "re-encoded on write, so what the index ends up holding is an "
            "approximation of what the adapter produced. `migrate` re-reads a sample "
            "of every batch and compares it to what it sent, to a tolerance of "
            f"{VERIFY_ATOL:.0e} — a store whose codec is coarser than that will fail "
            "the check and stop the job, with the shadow copy already written.[/dim]"
        )


def _report_aftermath(  # noqa: PLR0913 - one argument per thing the run left behind
    engine: MigrationEngine,
    result: MigrationResult,
    *,
    store: str,
    db: Any,
    health_before: Any,
    health_check: bool,
    rebuild_index: bool,
) -> None:
    """Everything a finished run has to say, in the order it has to say it.

    Split out of the command because it is the part that grew: what the run did,
    whether the index can still find things, and whether the collection is now
    holding two embedding spaces. The order is not arbitrary — a rebuild has to
    happen before the second measurement, or the measurement reports a problem
    the user has already asked to fix.
    """
    from rebasis.migrate import mixed_spaces_for

    _report_run(result)
    if rebuild_index:
        _rebuild_index(engine.store)
    # The engine may have reopened the store for the durability check, so this
    # measures through whatever handle it is holding now.
    _report_health(engine.store, health_before, enabled=health_check)
    # A run that stopped short — `--limit`, a pause, a failed batch — leaves the
    # collection holding both models' vectors. Said while the person who did it
    # is still looking at the terminal.
    _print_mixed_space(mixed_spaces_for(db, store))


def _rebuild_index(store: VectorStore) -> None:
    """Ask the store to rebuild its search structure, if it can.

    Not the default, and not automatic on a measured drop. Rebuilding changes
    the collection's own configuration — on Qdrant it is a bump to
    `ef_construct` — and that is the user's index, not rebasis'. What is
    automatic is *saying* the drop happened; acting on it is asked for.
    """
    from rebasis.errors import RebasisError

    try:
        store.rebuild_index()
    except RebasisError as exc:
        console.print(f"[yellow]index rebuild not available:[/yellow] [dim]{exc.message}[/dim]")
        return
    console.print(
        "[dim]index  rebuild requested; the store builds it in the background and "
        "keeps serving from the old one meanwhile.[/dim]"
    )


def _print_mixed_space(states: list[Any]) -> None:
    """Say, unprompted, that an index is holding two embedding spaces.

    Printed by `status` and again by `migrate` on the way out of a run that
    stopped short. Twice rather than once because they are read at different
    moments: `migrate` catches the person who just did it, `status` the person
    who comes back tomorrow wondering why search got worse.

    Escaped rather than interpolated raw — a store URI carries `[` and `]` often
    enough that Rich would eat part of it as markup.
    """
    from rich.markup import escape

    for state in states:
        console.print()
        console.print(
            "[red bold]This index holds two embedding spaces.[/red bold] "
            "[dim]Search results are not correct until the migration finishes "
            "or is rolled back.[/dim]"
        )
        console.print(f"  {escape(state.explain())}")
        for step in state.next_steps():
            console.print(f"    [dim]{escape(step)}[/dim]")


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


def _refit_collaborators(
    manifest: AdapterManifest,
    store: VectorStore,
    *,
    enabled: bool,
    device: str,
) -> tuple[Embedder | None, tuple[EncodingProfile, EncodingProfile] | None]:
    """The embedder and profiles ``--refit`` needs, or ``(None, None)``.

    `migrate` normally opens no model at all — it applies a matrix to vectors it
    reads. A refit has to produce *real* target vectors, because a migrated
    record carries the adapter's own image rather than the new model's output,
    so it needs the new model. Which model that is comes off the adapter's
    manifest rather than off a flag: the adapter records what it was fitted
    against, and asking the user to retype it is asking them to get it wrong.

    Both profiles are resolved because an adopted adapter is written back as a
    `.rbs`, and a `.rbs` records the profiles it maps between. The **old** one
    is resolved rather than opened — nothing here runs the old model — with the
    index's own dimension as the fallback, which is the same arrangement
    `open_embedders` uses and for the same reason: the index is authoritative
    about the model it was built with.

    Raises:
        ConfigError: When the opened model's profile is not the one the adapter
            was fitted against. A refit under a different prefix scheme would
            produce pairs the adapter never saw and adopt a map fitted to them.
    """
    if not enabled:
        return None, None

    from rebasis.cli._profiles import resolve_profile
    from rebasis.embed import open_embedder
    from rebasis.errors import ConfigError

    new_profile = resolve_profile(manifest.new_model_id, None)
    embedder = open_embedder(
        manifest.new_model_id,
        device=None if device == "auto" else device,
        profile=new_profile,
    )
    fingerprint = embedder.profile.fingerprint()
    if fingerprint != manifest.new_profile_fingerprint:
        raise ConfigError(
            f"{manifest.new_model_id} encodes differently now than when the adapter "
            "was fitted, so a refit would fit against pairs the adapter never saw.",
            hint=(
                "Drop --refit, or re-fit the adapter with `rebasis fit "
                "--direction old_to_new` against the current profile."
            ),
            context={
                "model_id": manifest.new_model_id,
                "expected": manifest.new_profile_fingerprint,
                "actual": fingerprint,
            },
        )

    old_profile = resolve_profile(manifest.old_model_id, None, fallback_dim=store.dimension())
    return embedder, (old_profile, embedder.profile)


def _check_direction(manifest: AdapterManifest, adapter_path: Path) -> None:
    """Refuse an adapter that maps the wrong way, before anything is written.

    An adapter has a direction and `migrate` needs the one that is not produced.
    `fit` writes ``query_to_old``: a map from the **new** model's space into the
    index's, which is what lets `Bridge` send a new-model query at an untouched
    index. `migrate` does the opposite job — it rewrites the **indexed document
    vectors** — and for that it needs ``old_to_new``, a map out of the index's
    space and into the new model's.

    Handing it the query map applies a function outside its domain, and nothing
    downstream notices: the write succeeds, the count is right, the text
    survives, the read-back verifies (it compares what was written against what
    comes back, not against anything meaningful), and `migrate`'s own index
    health check measures the store's search against exact kNN *over the vectors
    it now holds*, which is a property of the index structure rather than of the
    vectors' meaning. Every existing guard passes. The index is destroyed.

    **Measured**, on 4,000 synthetic documents where both spaces are known
    exactly and the bridge itself scores 1.000 against the untouched index: the
    index a completed migration leaves behind answers at recall@1 **0.000** to a
    raw new-model query, **0.000** to a bridged query and **0.000** to an
    old-model query. There is no query that is correct against it. For an
    orthogonal adapter the arithmetic says why — ``A(q)·A(d) = q·d``, so a
    bridged query against a fully migrated index reduces to the naive swap.

    ``rebasis fit --direction old_to_new`` produces the map this needs, and the
    check stays because the two files are indistinguishable from the outside: an
    `.rbs` is an `.rbs`, both directions are the same shape, and the wrong one
    fails silently rather than loudly. The direction is recorded in the manifest
    precisely so that something can read it before the write.
    """
    from rebasis.errors import ConfigError

    if manifest.direction == "old_to_new":
        return
    raise ConfigError(
        f"{adapter_path.name} maps queries into the index's space "
        f"(direction={manifest.direction!r}); `migrate` rewrites the indexed "
        "vectors and needs a map in the opposite direction.",
        hint=(
            "Re-fit with `rebasis fit --direction old_to_new`, which produces "
            "the map `migrate` needs. This adapter is the one `rebasis.Bridge` "
            "serves with — it maps queries, and it leaves the index untouched."
        ),
        context={"direction": manifest.direction, "adapter": adapter_path.name},
    )


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
    partial: bool = False,
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
    if partial:
        # Said before the confirmation rather than after the run, because this
        # is the point at which the user can still decide not to. `--limit` is
        # recommended in the guide as the safe way to try a migration, and it
        # is safe for the *data* — the shadow copy is intact either way. What it
        # is not is safe for *queries* in the window before the job finishes.
        console.print()
        console.print(
            "  [yellow]--limit stops this run short.[/yellow] [dim]Until the job "
            "finishes, the index holds both models' vectors and no single query is "
            "correct against all of it. Plan to finish or roll back before "
            "serving from it.[/dim]"
        )
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
    from rebasis.migrate import JobQueue, mixed_spaces

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
    # Every unfinished job, then narrowed to the ones being shown: `status
    # <job-id>` should report on that job and not on somebody else's.
    shown = {job.job_id for job in parsed}
    mixed = {state.job_id: state for state in mixed_spaces(db) if state.job_id in shown}

    if as_json or plain:
        _emit_jobs(jobs, as_json=as_json, mixed=mixed)
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
    table.add_column("index")

    for job, stats in jobs:
        table.add_row(
            job.job_id,
            # A job that has been asked to stop and has not stopped yet is still
            # running, and saying only "running" hides the one fact the person
            # who just asked is waiting on.
            f"{job.state} [yellow](pausing)[/yellow]" if job.pause_requested else job.state,
            job.adapter_type,
            f"{stats.completed_fraction:.0%}",
            f"{stats.done:,}",
            f"{stats.failed:,}" if stats.failed else "—",
            _rollback_column(job),
            "[red]mixed[/red]" if job.job_id in mixed else "[dim]single[/dim]",
        )
    console.print(table)
    _print_mixed_space(list(mixed.values()))
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

    On a store that quantizes, "the original" means something narrower and
    `migrate` says so before it writes: the shadow holds the store's own decoded
    view of its vectors, because that is what it returned when they were read,
    so this restores the state the migration replaced rather than the vectors
    the embedding model produced.
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


@handle_errors
def pause_command(
    job_id: Annotated[str, typer.Argument(help="The running job to stop")],
    state_dir: Annotated[Path | None, typer.Option("--state-dir")] = None,
) -> None:
    """Ask a running migration to stop after its current batch.

    The engine reads the request at the top of every batch and finishes the one
    it is in, so this returns immediately and the job stops a moment later. That
    is deliberate: killing the process mid-batch is already safe — the queue is
    the checkpoint and a shadow is always written before the vector it copies is
    overwritten — but it leaves the store holding a batch nobody has verified,
    and a clean stop at a boundary does not.

    Takes no lock. The migration it is talking to holds the state lock for its
    whole run, so a command that waited for it would wait for the thing it is
    trying to interrupt. What makes that safe is that this writes one column no
    other process writes: `pause_requested` is a *request*, and only the engine
    ever says what state a job is in.

    Resume with `rebasis resume <job-id>`, which clears the request.
    """
    from rebasis.cli._pipeline import audit_writer_for
    from rebasis.errors import ConfigError
    from rebasis.manifest import JobRow, ManifestDB, default_state_dir, manifest_path
    from rebasis.migrate import JobState, request_pause
    from rebasis.observability import Events

    directory = state_dir or default_state_dir()
    path = manifest_path(directory)
    if not path.exists():
        raise ConfigError(
            f"There is no rebasis state at {directory}.",
            hint="Point --state-dir at the directory the migration was started from.",
        )

    with ManifestDB(path) as db:
        row = db.query_one("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        if row is None:
            raise ConfigError(
                f"No migration job {job_id!r}.",
                hint="`rebasis status` lists every job this state directory knows.",
                context={"job_id": job_id},
            )
        job = JobRow.from_row(row)
        if not request_pause(db, job_id):
            # The guard is in the statement, so reaching here means the state
            # changed under us or was never `running`. Either way the row that
            # was read is what the user needs to see.
            raise ConfigError(
                f"Job {job_id} is {job.state}, not running.",
                hint=(
                    "Only a running job can be asked to pause. "
                    f"`rebasis resume {job_id}` continues one that stopped."
                ),
                context={"job_id": job_id, "state": job.state},
            )

        writer = audit_writer_for(directory)
        writer.write(
            Events.MIGRATE_PAUSE_REQUESTED,
            inputs={"job_id": job_id},
            outputs={"job_id": job_id, "state": str(JobState.RUNNING)},
            subject=job_id,
        )

    console.print(f"[yellow]Pause requested[/yellow] for {job_id}.")
    console.print("  [dim]it stops at the end of the batch it is in[/dim]")
    console.print(f"  [dim]resume with `rebasis resume {job_id}`[/dim]")


@handle_errors
def resume_command(  # noqa: PLR0913, PLR0917 - each option is a documented CLI flag
    job_id: Annotated[str, typer.Argument(help="The job to continue")],
    batch: Annotated[
        int | None, typer.Option("--batch", help="Records per batch; omit to keep migrate's")
    ] = None,
    limit: Annotated[int | None, typer.Option("--limit", help="Stop after this many")] = None,
    power_aware: Annotated[
        bool | None, typer.Option("--power-aware/--no-power-aware", help="Pause on low battery")
    ] = None,
    max_memory: Annotated[
        str | None, typer.Option("--max-memory", help="Ceiling, e.g. 2GB")
    ] = None,
    health_check: Annotated[
        bool | None,
        typer.Option("--health-check/--no-health-check", help="Measure the index either side"),
    ] = None,
    rebuild_index: Annotated[
        bool, typer.Option("--rebuild-index", help="Rebuild the search structure at the end")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation")] = False,
    no_input: Annotated[
        bool, typer.Option("--no-input", help="Never prompt; fail instead of asking")
    ] = False,
    state_dir: Annotated[Path | None, typer.Option("--state-dir")] = None,
) -> None:
    """Continue a migration that stopped, from where it stopped.

    The same thing as `rebasis migrate --resume <job-id>`, and it forwards to
    it: the adapter and the store URI come off the job row, the queue is the
    checkpoint, and any outstanding pause request is cleared as the engine
    starts. It exists because "pause" and "resume" is the pair a person reaches
    for, and because `migrate` is the command that starts something new.

    Only the flags that describe *this run* are here. `--priority` and
    `--access-log` are not: they order the queue, the queue was ordered when the
    job was created, and re-ordering half a migration would be a different job.
    Everything left out keeps whatever `migrate` defaults to — the defaults live
    in one place, and passing them on from here would be a second copy of them.
    """
    overrides = {
        name: value
        for name, value in (
            ("batch", batch),
            ("limit", limit),
            ("power_aware", power_aware),
            ("max_memory", max_memory),
            ("health_check", health_check),
        )
        if value is not None
    }
    migrate_command(
        resume=job_id,
        rebuild_index=rebuild_index,
        yes=yes,
        no_input=no_input,
        state_dir=state_dir,
        **overrides,
    )
