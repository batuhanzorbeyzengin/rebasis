"""``rebasis eval`` — score an existing adapter against a query set."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - typer resolves annotations at runtime
from typing import TYPE_CHECKING, Annotated

import typer

from rebasis.cli._common import console, handle_errors, step_progress
from rebasis.errors import ConfigError

if TYPE_CHECKING:
    from rebasis.core.base import BaseAdapter
    from rebasis.core.calibration import ScoreCalibrator
    from rebasis.core.serialization import AdapterManifest

__all__ = ["eval_command"]


@handle_errors
def eval_command(  # noqa: PLR0913, PLR0917 - each option is a documented CLI flag
    adapter: Annotated[Path, typer.Argument(help="Path to a .rbs adapter")],
    queries: Annotated[
        Path | None,
        typer.Option("--queries", help="Real query log (JSONL). Preferred when available"),
    ] = None,
    store: Annotated[str | None, typer.Option("--store", help="Store URI")] = None,
    sample: Annotated[
        int, typer.Option("--sample", help="Documents to embed with the new model")
    ] = 10_000,
    heldout: Annotated[
        int, typer.Option("--heldout", help="Documents held out as query proxies")
    ] = 1_000,
    k: Annotated[int, typer.Option("--k", help="Cut-off for every metric")] = 10,
    seed: Annotated[int, typer.Option("--seed", help="Recorded so the run can be replayed")] = 0,
    device: Annotated[str, typer.Option("--device", help="auto|cpu|cuda|cuda:N|mps")] = "auto",
    query_prefix: Annotated[
        str | None,
        typer.Option("--query-prefix", help="New model's query prefix, if not in the table"),
    ] = None,
    document_prefix: Annotated[
        str | None,
        typer.Option("--document-prefix", help="New model's document prefix, if not in the table"),
    ] = None,
    verify: Annotated[
        bool,
        typer.Option("--verify", help="Recompute every tensor hash before loading"),
    ] = False,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit the result as JSON on stdout, for scripts and CI"),
    ] = False,
) -> None:
    """Measure an adapter that already exists.

    With --verify this also acts as an integrity check: a corrupt adapter
    fails loudly rather than quietly returning worse results.
    """
    from rebasis.core import load_adapter

    loaded, manifest, calibrator = load_adapter(adapter, verify=verify)

    if store is None:
        _describe(adapter, loaded, manifest, calibrator, verify=verify)
        return

    _score(
        adapter,
        manifest,
        store=store,
        queries=queries,
        sample=sample,
        heldout=heldout,
        k=k,
        seed=seed,
        device=device,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
        as_json=as_json,
    )


def _describe(
    path: Path,
    loaded: BaseAdapter,
    manifest: AdapterManifest,
    calibrator: ScoreCalibrator | None,
    *,
    verify: bool,
) -> None:
    """What the file contains, without touching a store."""
    console.print(f"[bold]{path.name}[/bold]")
    console.print(f"  schema        {manifest.schema}")
    console.print(f"  adapter       {manifest.adapter_type}")
    console.print(f"  direction     {manifest.direction}")
    console.print(f"  dimensions    {manifest.input_dim} → {manifest.output_dim}")
    console.print(f"  old model     {manifest.old_model_id}")
    console.print(f"  new model     {manifest.new_model_id}")
    console.print(f"  parameters    {loaded.n_params():,} ({loaded.memory_bytes / 1024**2:.2f} MB)")
    console.print(f"  calibrator    {'present' if calibrator else 'none'}")
    integrity = "verified" if verify else "header only — pass --verify for a full tensor hash check"
    console.print(f"  integrity     {integrity}")


def _score(  # noqa: PLR0913 - one argument per measurement input
    path: Path,
    manifest: AdapterManifest,
    *,
    store: str,
    queries: Path | None,
    sample: int,
    heldout: int,
    k: int,
    seed: int,
    device: str,
    query_prefix: str | None = None,
    document_prefix: str | None = None,
    as_json: bool = False,
) -> None:
    """Re-measure a stored adapter against a live store.

    The adapter is scored the same way ``fit`` scored it, restricted to the one
    method it holds — so the number here is comparable to the one in its
    ``eval.json`` and a drop means the corpus moved, not that the metric changed.

    Raises:
        ConfigError: When the adapter records no adapter type to re-fit.
    """
    from rebasis.cli._pipeline import (
        ProfileOverrides,
        load_query_log,
        open_embedders,
        open_target_store,
        print_json,
        print_result,
    )
    from rebasis.probe.session import probe_store
    from rebasis.storage import default_embedding_cache_dir

    adapter_type = manifest.adapter_type
    if not adapter_type:
        raise ConfigError(
            f"{path.name} does not record which adapter it holds.",
            hint="Re-fit it with `rebasis fit`; adapters written before schema 1 lack this.",
        )

    query_log = load_query_log(queries) if queries is not None else None
    opened = open_target_store(store)
    old_embedder, new_embedder = open_embedders(
        manifest.old_model_id if query_log is not None else None,
        manifest.new_model_id,
        device=device,
        new_overrides=ProfileOverrides(
            dim=manifest.input_dim,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
        ),
        store_dim=opened.dimension(),
    )
    with step_progress("Scoring the adapter against the store") as steps:
        result, _ = probe_store(
            opened,
            new_embedder,
            old_embedder=old_embedder,
            query_log=query_log,
            size=sample,
            heldout=heldout,
            k=k,
            seed=seed,
            methods=[adapter_type],
            device=device,
            # `eval` takes no `--state-dir`, so this resolves the default —
            # which is the same place `probe` and `fit` put it, and scoring an
            # adapter re-embeds the collection they already embedded.
            cache_dir=default_embedding_cache_dir(),
            on_stage=steps.stage,
        )

    if as_json:
        print_json(result)
        return

    print_result(result)
