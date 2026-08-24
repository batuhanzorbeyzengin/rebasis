"""``rebasis fit`` — fit an adapter and write a ``.rbs`` file."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - typer resolves annotations at runtime
from typing import Annotated

import typer

from rebasis.cli._common import console, handle_errors
from rebasis.errors import ConfigError
from rebasis.observability import Events

__all__ = ["fit_command"]


@handle_errors
def fit_command(  # noqa: PLR0913, PLR0917 - each option is a documented CLI flag
    out: Annotated[Path, typer.Option("--out", help="Where to write the .rbs adapter")],
    store: Annotated[str | None, typer.Option("--store", help="Store URI")] = None,
    old: Annotated[str | None, typer.Option("--old", help="Model the index was built with")] = None,
    new: Annotated[str | None, typer.Option("--new", help="Model you are adopting")] = None,
    method: Annotated[
        str,
        typer.Option(
            "--method",
            help=(
                "auto, or a specific adapter. auto fits every candidate and picks "
                "the best on a held-out set"
            ),
        ),
    ] = "auto",
    pairs: Annotated[
        int,
        typer.Option(
            "--pairs",
            help=(
                "Matched pairs to fit on. Measured to saturate near 4000; the "
                "next 20000 gained +0.001"
            ),
        ),
    ] = 4000,
    heldout: Annotated[
        int, typer.Option("--heldout", help="Documents held out to score candidates on")
    ] = 1_000,
    k: Annotated[int, typer.Option("--k", help="Cut-off used to score candidates")] = 10,
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
    seed: Annotated[int, typer.Option("--seed", help="Recorded so the fit can be replayed")] = 0,
    device: Annotated[str, typer.Option("--device", help="auto|cpu|cuda|cuda:N|mps")] = "auto",
    state_dir: Annotated[Path | None, typer.Option("--state-dir")] = None,
) -> None:
    """Fit an adapter that maps the new model's space onto the existing index.

    Reads the index; never writes to it.
    """
    from rebasis.cli._pipeline import (
        ProfileOverrides,
        audit_writer_for,
        open_embedders,
        open_target_store,
        print_next_step_after_fit,
    )
    from rebasis.cli._profiles import resolve_profile
    from rebasis.core import save_adapter
    from rebasis.probe.session import probe_store

    if store is None or old is None or new is None:
        raise ConfigError(
            "`fit` needs --store, --old and --new.",
            hint="`rebasis fit --store chroma:///db#docs --old <model> --new <model> --out a.rbs`",
        )

    methods = None if method == "auto" else [method]
    opened = open_target_store(store)

    _, new_embedder = open_embedders(
        None,
        new,
        device=device,
        new_overrides=ProfileOverrides(
            dim=new_dim, query_prefix=query_prefix, document_prefix=document_prefix
        ),
    )
    # The index is authoritative about the old model's dimension, so an
    # unregistered model id does not need `--old-dim` here the way it does when
    # the model has to be loaded.
    old_profile = resolve_profile(
        old,
        ProfileOverrides(
            dim=old_dim, query_prefix=old_query_prefix, document_prefix=old_document_prefix
        ),
        fallback_dim=opened.dimension(),
    )
    with console.status("Fitting the adapter…"):
        result, _ = probe_store(
            opened,
            new_embedder,
            # `pairs` is the fitting budget, so the sample must hold that many
            # documents plus the held-out set they are scored against: the two
            # have to stay disjoint, or the score means nothing.
            size=pairs + heldout,
            heldout=heldout,
            k=k,
            seed=seed,
            methods=methods,
            device=device,
        )

    if result.adapter is None:
        raise ConfigError(
            "No adapter could be fitted from this collection.",
            hint="Try `--method procrustes`, which needs the fewest pairs.",
            context={"count": pairs},
        )

    written = save_adapter(
        result.adapter,
        out,
        direction="query_to_old",
        old_profile=old_profile,
        new_profile=new_embedder.profile,
        calibrator=result.calibrator,
        evaluation=result.to_dict(),
    )

    # `fit.adapter.selected` is on the audited list, and it is the record that
    # matters most — the `.rbs` file says which adapter won, and the audit trail
    # says on what evidence, against which models, from how many pairs.
    # `migrate` writes an audit record; the artefact it migrates with did not.
    audit = audit_writer_for(state_dir)
    audit.write(
        Events.FIT_ADAPTER_SELECTED,
        inputs={
            "old_model": old,
            "new_model": new,
            "count": pairs,
            "k": k,
            "seed": seed,
            "method": method,
        },
        outputs={
            "adapter_type": result.best.name,
            "arr_r10": round(result.best.arr, 4),
            "n_params": result.best.n_params,
            "path": str(written),
        },
        subject=str(written),
    )

    console.print()
    console.print(f"[bold green]{result.best.name}[/bold green]  ARR@{k} {result.best.arr:.3f}")
    console.print(f"  parameters  {result.best.n_params:,}")
    console.print(f"  fit pairs   {result.n_fit_pairs:,}")
    console.print(f"  calibrator  {'fitted' if result.calibrator else 'none'}")
    console.print(f"  written to  {written}")
    print_next_step_after_fit(written, store=store)
