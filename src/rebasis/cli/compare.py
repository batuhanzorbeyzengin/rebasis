"""``rebasis compare`` — rank several candidate models on your own corpus."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - typer resolves annotations at runtime
from typing import TYPE_CHECKING, Annotated, Any

import typer

from rebasis.cli._common import handle_errors, step_progress

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["compare_command"]


@handle_errors
def compare_command(  # noqa: PLR0913, PLR0917 - each option is a documented CLI flag
    store: Annotated[
        str, typer.Option("--store", help="Store URI, e.g. chroma:///path/db#collection")
    ],
    old: Annotated[str, typer.Option("--old", help="Model the index was built with")],
    candidates: Annotated[
        str,
        typer.Option(
            "--candidates",
            help="Comma-separated model ids to rank against the index's own model",
        ),
    ],
    sample: Annotated[
        int, typer.Option("--sample", help="Documents every candidate is scored on")
    ] = 10_000,
    heldout: Annotated[
        int, typer.Option("--heldout", help="Documents held out as query proxies")
    ] = 1_000,
    k: Annotated[int, typer.Option("--k", help="Cut-off for every metric")] = 10,
    queries: Annotated[
        Path | None,
        typer.Option("--queries", help="Real query log (JSONL). Always preferred"),
    ] = None,
    synth_queries: Annotated[
        str | None,
        typer.Option(
            "--synth-queries",
            help="Estimate the upgrade from the documents: title|lead|keywords",
        ),
    ] = None,
    tiered: Annotated[
        bool,
        typer.Option(
            "--tiered",
            help=(
                "Score every candidate on a small sample first and carry only "
                "what it could not separate through to the full one"
            ),
        ),
    ] = False,
    report: Annotated[
        Path | None,
        typer.Option("--report", help="Write a report here; .html for HTML, otherwise Markdown"),
    ] = None,
    strategy: Annotated[
        str, typer.Option("--strategy", help="stratified|random sampling of the corpus")
    ] = "stratified",
    seed: Annotated[int, typer.Option("--seed", help="Recorded so the run can be replayed")] = 0,
    device: Annotated[str, typer.Option("--device", help="auto|cpu|cuda|cuda:N|mps")] = "auto",
    state_dir: Annotated[
        Path | None,
        typer.Option("--state-dir", help="Where the audit trail lives; defaults to ./.rebasis"),
    ] = None,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the cost estimate's confirmation")
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the ranking as JSON on stdout")
    ] = False,
) -> None:
    """Rank candidate models on your own corpus, without rebuilding the index.

    Reads the index; never writes to it. Every candidate is scored on the same
    sample, the same split and the same queries — a redraw per candidate would
    make the rows incomparable, which is the one thing a ranking cannot survive.
    """
    from rebasis.cli._common import console
    from rebasis.cli._pipeline import (
        ProfileOverrides,
        audit_writer_for,
        load_query_log,
        open_embedders,
        open_target_store,
        print_comparison,
        print_comparison_json,
        write_comparison_report,
    )
    from rebasis.errors import UserAbort
    from rebasis.probe.comparison import compare_models
    from rebasis.storage import default_embedding_cache_dir

    names = [name.strip() for name in candidates.split(",") if name.strip()]
    if not names:
        message = "--candidates listed no models."
        raise UserAbort(message, hint="Pass a comma-separated list of model ids.")

    query_log = load_query_log(queries) if queries is not None else None
    opened = open_target_store(store)

    embedders = {}
    for name in names:
        # `open_embedders` is reused rather than reimplemented: it resolves the
        # profile the same way `probe` does, and a candidate resolved
        # differently from the way `probe` would resolve it is a candidate whose
        # number does not transfer to the `probe` the user runs next.
        _, embedder = open_embedders(
            None, name, device=device, new_overrides=ProfileOverrides(), store_dim=None
        )
        embedders[name] = embedder
    old_embedder, _ = open_embedders(
        old,
        old,
        device=device,
        old_overrides=ProfileOverrides(),
        store_dim=opened.dimension(),
    )

    _warn_about_remote(embedders)
    if not _confirm_cost(opened, embedders, sample=sample, assume_yes=yes):
        console.print("[dim]Nothing was run.[/dim]")
        return

    audit = audit_writer_for(state_dir)
    with step_progress("Sampling the index") as steps:
        result = compare_models(
            opened,
            embedders,
            old_embedder=old_embedder,
            query_log=query_log,
            size=sample,
            heldout=heldout,
            strategy=strategy,
            k=k,
            seed=seed,
            tiered=tiered,
            synth_queries=synth_queries,
            cache_dir=default_embedding_cache_dir(state_dir),
            audit=audit,
            store_uri=store,
            old_model=old,
            device=device,
            on_stage=steps.stage,
        )

    write_comparison_report(result, store_uri=store, report=report)
    if as_json:
        print_comparison_json(result)
        return
    print_comparison(result)


def _warn_about_remote(embedders: Mapping[str, Any]) -> None:
    """Say which candidates would send document text off the machine, first.

    Before the run rather than after it, and per candidate rather than once: a
    comparison embeds the same sample once for every candidate, so one hosted
    endpoint in the list means the corpus leaves the machine a whole extra time.
    """
    from rebasis.cli._common import console
    from rebasis.probe.comparison import remote_candidates

    remote = remote_candidates(embedders)
    if not remote:
        return
    console.print(
        f"[yellow]{len(remote)} candidate(s) run on a remote endpoint: "
        f"{', '.join(remote)}. Their backend sends the sampled document text off "
        f"this machine, once per candidate.[/yellow]"
    )


def _confirm_cost(
    store: object, embedders: Mapping[str, Any], *, sample: int, assume_yes: bool
) -> bool:
    """Print what each candidate will cost, measured here, and ask.

    The same shape as `migrate`'s pre-flight plan and for the same reason: N
    candidates is N embedding passes, and the difference between a static 8M
    model and a 300M transformer is two orders of magnitude. A user should meet
    that number before the run rather than during it.
    """
    import itertools

    import typer

    from rebasis.cli._common import console, interactive
    from rebasis.probe.comparison import COST_PROBE_DOCUMENTS, estimate_candidate_cost
    from rebasis.store.base import VectorStore

    if not isinstance(store, VectorStore):  # pragma: no cover - defensive
        return True
    records = itertools.islice(
        store.iter_records(with_vectors=False, batch_size=COST_PROBE_DOCUMENTS),
        COST_PROBE_DOCUMENTS,
    )
    texts = [record.text or "" for record in records]
    if not texts:
        return True

    console.print()
    console.print(f"[dim]Measured on this machine, on {len(texts)} of your own documents:[/dim]")
    total = 0.0
    for model, embedder in embedders.items():
        cost = estimate_candidate_cost(embedder, texts, total=sample)
        total += cost["seconds"]
        console.print(f"  {model:44s} ~{_duration(cost['seconds'])}")
    console.print(f"  [bold]{'total':44s} ~{_duration(total)}[/bold]")
    console.print(
        "[dim]A warm embedding cache makes a re-run of the same candidate free; "
        "--tiered scores everything on a small sample first.[/dim]"
    )
    # Not `confirm`: that one raises on a non-terminal, which is right for a
    # command that writes and wrong for one that does not. `compare` is
    # read-only, so a script gets the estimate printed and the run performed;
    # only a person at a terminal is asked, and only because the wait is long
    # enough that finding out afterwards is a poor way to learn it.
    if assume_yes or not interactive():
        return True
    return bool(typer.confirm("Run the comparison?", default=True))


def _duration(seconds: float) -> str:
    """Seconds as something a person reads without dividing."""
    if seconds < 90:  # noqa: PLR2004 - the point at which minutes read better
        return f"{seconds:.0f}s"
    if seconds < 5400:  # noqa: PLR2004 - and hours after ninety minutes
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"
