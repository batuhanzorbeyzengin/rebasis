"""``rebasis probe`` — measure the drift and recommend."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - typer resolves annotations at runtime
from typing import Annotated

import typer

from rebasis.cli._common import handle_errors, step_progress
from rebasis.probe.runner import CASCADE_N

__all__ = ["probe_command"]


@handle_errors
def probe_command(  # noqa: PLR0913, PLR0917 - each option is a documented CLI flag
    store: Annotated[
        str, typer.Option("--store", help="Store URI, e.g. chroma:///path/db#collection")
    ],
    old: Annotated[
        str | None,
        typer.Option(
            "--old",
            help=(
                "Model the index was built with. Required, except in a "
                "--truncate/--quantize run with no --queries, where nothing is "
                "encoded at all"
            ),
        ),
    ] = None,
    new: Annotated[
        str | None,
        typer.Option(
            "--new",
            help="Model you are considering. Not used by --truncate/--quantize",
        ),
    ] = None,
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
    truncate: Annotated[
        str | None,
        typer.Option(
            "--truncate",
            help=(
                "Comma-separated dimensions to measure this index at, e.g. "
                "1024,512,256. Switches to the grid: same model, cheaper vectors"
            ),
        ),
    ] = None,
    quantize: Annotated[
        str | None,
        typer.Option(
            "--quantize",
            help="Comma-separated precisions: float32,float16,int8,binary",
        ),
    ] = None,
    floor: Annotated[
        float | None,
        typer.Option(
            "--floor",
            help=(
                "Quality floor, e.g. 0.95. Names the cheapest cell that clears "
                "it — a Pareto choice, not a break-even"
            ),
        ),
    ] = None,
    cascade_n: Annotated[
        int,
        typer.Option(
            "--cascade-n",
            help=(
                "Candidate depth the two-stage arrangement is measured at. 0 "
                "skips it. Bind it to your reranking budget"
            ),
        ),
    ] = CASCADE_N,
    seed: Annotated[int, typer.Option("--seed", help="Recorded so the run can be replayed")] = 0,
    device: Annotated[str, typer.Option("--device", help="auto|cpu|cuda|cuda:N|mps")] = "auto",
    state_dir: Annotated[
        Path | None,
        typer.Option(
            "--state-dir",
            help=(
                "Where the audit trail lives; defaults to ./.rebasis. Unused by "
                "--truncate/--quantize, which decides nothing and records nothing"
            ),
        ),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit the decision as JSON on stdout, for scripts and CI"),
    ] = False,
) -> None:
    """Measure what switching embedding models would cost, and recommend.

    Reads the index; never writes to it.

    With --truncate or --quantize it answers a different question about the same
    model: what a cheaper representation of this index would cost. No candidate,
    no adapter, and no squeeze — only a measurement.

    (No RST markup here, and that is a rule rather than a habit: a command's
    docstring is its --help text, and a terminal renders backticks and asterisks
    as backticks and asterisks. `tests/unit/test_cli_output_hygiene.py` checks
    it, and caught this paragraph.)
    """
    if truncate is not None or quantize is not None:
        _truncation_run(
            store=store,
            old=old,
            new=new,
            truncate=truncate,
            quantize=quantize,
            floor=floor,
            queries=queries,
            sample=sample,
            heldout=heldout,
            k=k,
            strategy=strategy,
            seed=seed,
            report=report,
            as_json=as_json,
            old_dim=old_dim,
            old_query_prefix=old_query_prefix,
            old_document_prefix=old_document_prefix,
            device=device,
        )
        return

    from rebasis.errors import ConfigError

    if old is None or new is None:
        message = "`probe` needs --old and --new."
        raise ConfigError(
            message,
            hint=(
                "They are only optional for a --truncate/--quantize run, which "
                "measures the model already in the index rather than a new one."
            ),
        )

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
            # 0 rather than a separate --no-cascade: the depth and the switch
            # are one choice, and a flag that can contradict another flag is a
            # state somebody has to reason about.
            cascade_k=cascade_n if cascade_n > 0 else None,
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


def _truncation_run(  # noqa: PLR0913 - it forwards one CLI flag per argument
    *,
    store: str,
    old: str | None,
    new: str | None,
    truncate: str | None,
    quantize: str | None,
    floor: float | None,
    queries: Path | None,
    sample: int,
    heldout: int,
    k: int,
    strategy: str,
    seed: int,
    report: Path | None,
    as_json: bool,
    old_dim: int | None,
    old_query_prefix: str | None,
    old_document_prefix: str | None,
    device: str,
) -> None:
    """The grid: what a cheaper representation of this index would cost.

    Split out of ``probe_command`` rather than branched inside it, because the
    two share their flags and share nothing else — a different reference, a
    different ground truth, a different output shape and no adapter anywhere.
    """
    from rebasis.cli._pipeline import (
        ProfileOverrides,
        load_query_log,
        open_embedders,
        open_target_store,
        print_grid,
        print_grid_json,
        write_grid_report,
    )
    from rebasis.errors import ConfigError
    from rebasis.probe.session import probe_truncation
    from rebasis.probe.truncation import PRECISIONS

    if new is not None:
        message = "--new has no meaning in a --truncate/--quantize run."
        raise ConfigError(
            message,
            hint=(
                "The grid measures the model already in the index at a cheaper "
                "representation. To compare a different model, drop --truncate."
            ),
        )

    opened = open_target_store(store)
    dims = (
        [int(part) for part in truncate.split(",") if part.strip()]
        if truncate
        else [opened.dimension()]
    )
    precisions = (
        [part.strip() for part in quantize.split(",") if part.strip()] if quantize else ["float32"]
    )
    unknown = [name for name in precisions if name not in PRECISIONS]
    if unknown:
        message = f"--quantize does not know {', '.join(unknown)}."
        raise ConfigError(message, hint=f"Choose from {', '.join(PRECISIONS)}.")

    query_log = load_query_log(queries) if queries is not None else None
    old_embedder = None
    if query_log is not None:
        if old is None:
            message = "--queries needs --old to encode the queries with."
            raise ConfigError(
                message,
                hint=(
                    "The queries have to be encoded with the model the index was "
                    "built with. Drop --queries to measure against held-out "
                    "documents instead, which needs no model at all."
                ),
            )
        old_embedder, _ = open_embedders(
            old,
            old,
            device=device,
            old_overrides=ProfileOverrides(
                dim=old_dim,
                query_prefix=old_query_prefix,
                document_prefix=old_document_prefix,
            ),
            store_dim=opened.dimension(),
        )

    with step_progress("Sampling the index") as steps:
        grid, _ = probe_truncation(
            opened,
            dims=dims,
            precisions=precisions,
            old_embedder=old_embedder,
            query_log=query_log,
            size=sample,
            heldout=heldout,
            strategy=strategy,
            k=k,
            seed=seed,
            floor=floor,
            device=device,
            on_stage=steps.stage,
        )

    write_grid_report(grid, store_uri=store, report=report)
    if as_json:
        print_grid_json(grid)
        return
    print_grid(grid)
