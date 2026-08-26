"""``rebasis probe`` — measure the drift and recommend."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - typer resolves annotations at runtime
from typing import Annotated

import typer

from rebasis.cli._common import handle_errors, step_progress

__all__ = ["probe_command"]


@handle_errors
def probe_command(  # noqa: PLR0913, PLR0917 - each option is a documented CLI flag
    store: Annotated[
        str, typer.Option("--store", help="Store URI, e.g. chroma:///path/db#collection")
    ],
    old: Annotated[str, typer.Option("--old", help="Model the index was built with")],
    new: Annotated[str, typer.Option("--new", help="Model you are considering")],
    sample: Annotated[
        int, typer.Option("--sample", help="Documents to embed with the new model")
    ] = 10_000,
    heldout: Annotated[
        int, typer.Option("--heldout", help="Documents held out as query proxies")
    ] = 1_000,
    k: Annotated[int, typer.Option("--k", help="Cut-off for every metric")] = 10,
    queries: Annotated[
        Path | None,
        typer.Option(
            "--queries",
            help="Real query log (JSONL). Always preferred when available",
        ),
    ] = None,
    access_log: Annotated[
        Path | None,
        typer.Option(
            "--access-log",
            help=(
                'JSONL of {"id": ..., "count": ...}. Weights which sampled '
                "records become query proxies, so ARR describes what is read"
            ),
        ),
    ] = None,
    report: Annotated[
        Path | None,
        typer.Option("--report", help="Write a report here; .html for HTML, otherwise Markdown"),
    ] = None,
    synth_queries: Annotated[
        str | None,
        typer.Option(
            "--synth-queries",
            help=(
                "Estimate the upgrade from the documents when you have no query "
                "log: title|lead|keywords"
            ),
        ),
    ] = None,
    strategy: Annotated[
        str, typer.Option("--strategy", help="stratified|random sampling of the corpus")
    ] = "stratified",
    old_dim: Annotated[
        int | None,
        typer.Option("--old-dim", help="Dimension of the old model, if rebasis does not know it"),
    ] = None,
    new_dim: Annotated[
        int | None,
        typer.Option("--new-dim", help="Dimension of the new model, if rebasis does not know it"),
    ] = None,
    query_prefix: Annotated[
        str | None,
        typer.Option("--query-prefix", help="New model's query prefix, e.g. 'query: '"),
    ] = None,
    document_prefix: Annotated[
        str | None,
        typer.Option("--document-prefix", help="New model's document prefix, e.g. 'passage: '"),
    ] = None,
    old_query_prefix: Annotated[
        str | None, typer.Option("--old-query-prefix", help="Old model's query prefix")
    ] = None,
    old_document_prefix: Annotated[
        str | None, typer.Option("--old-document-prefix", help="Old model's document prefix")
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Recorded so the run can be replayed")] = 0,
    device: Annotated[str, typer.Option("--device", help="auto|cpu|cuda|cuda:N|mps")] = "auto",
    state_dir: Annotated[
        Path | None,
        typer.Option("--state-dir", help="Where the audit trail lives; defaults to ./.rebasis"),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit the decision as JSON on stdout, for scripts and CI"),
    ] = False,
) -> None:
    """Measure what switching embedding models would cost, and recommend.

    Reads the index; never writes to it.
    """
    from rebasis.cli._pipeline import (
        ProfileOverrides,
        audit_writer_for,
        load_query_log,
        open_embedders,
        open_target_store,
        print_json,
        print_next_step_after_probe,
        print_result,
        read_access_log,
        write_reports,
    )
    from rebasis.probe.session import probe_store
    from rebasis.storage import default_embedding_cache_dir

    query_log = load_query_log(queries) if queries is not None else None
    opened = open_target_store(store)

    # The old model is needed whenever the upgrade is being measured — with a
    # real query log, or with synthesised ones. Only T0 can skip it, and only
    # because it cannot measure the upgrade at all.
    needs_old = query_log is not None or synth_queries is not None
    old_embedder, new_embedder = open_embedders(
        old if needs_old else None,
        new,
        device=device,
        old_overrides=ProfileOverrides(
            dim=old_dim, query_prefix=old_query_prefix, document_prefix=old_document_prefix
        ),
        new_overrides=ProfileOverrides(
            dim=new_dim, query_prefix=query_prefix, document_prefix=document_prefix
        ),
        store_dim=opened.dimension(),
    )
    audit = audit_writer_for(state_dir)
    with step_progress("Sampling the index") as steps:
        result, _ = probe_store(
            opened,
            new_embedder,
            old_embedder=old_embedder,
            query_log=query_log,
            size=sample,
            heldout=heldout,
            strategy=strategy,
            k=k,
            seed=seed,
            synth_queries=synth_queries,
            audit=audit,
            store_uri=store,
            old_model=old,
            device=device,
            # Beside the audit trail rather than in whatever directory the
            # command was run from, so `--state-dir` moves both together.
            cache_dir=default_embedding_cache_dir(state_dir),
            access_counts=read_access_log(access_log),
            on_stage=steps.stage,
        )

    # `--report` and `--json` answer different questions and are independent:
    # the report is for a person to read later, the JSON is for the script that
    # is about to branch on the decision. Writing the report first means asking
    # for both gets both.
    write_reports(result, store_uri=store, report=report)

    if as_json:
        print_json(result)
        return

    print_result(result)
    print_next_step_after_probe(result, store=store, old=old, new=new)
