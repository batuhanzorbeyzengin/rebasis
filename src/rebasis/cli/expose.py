"""``rebasis expose`` — measure how alignable an index is.

A defensive diagnostic. It returns a number and nothing else: no aligned
vectors, no reconstructed text, no inversion. `docs/exposure.md` says what the
number means and, at more length, what it does not.
"""

from __future__ import annotations

from typing import Annotated

import typer

from rebasis.cli._common import handle_errors, step_progress
from rebasis.probe.exposure import SEEDS

__all__ = ["expose_command"]

#: The reference model, when the user names none.
#:
#: A small, public, local model — which is exactly the adversary's position.
#: Naming a default at all is a choice: the alternative is to require a flag,
#: and a security diagnostic that is awkward to run is a security diagnostic
#: nobody runs.
DEFAULT_REFERENCE = "sentence-transformers/all-MiniLM-L6-v2"


@handle_errors
def expose_command(  # noqa: PLR0913, PLR0917 - each option is a documented CLI flag
    store: Annotated[
        str, typer.Option("--store", help="Store URI, e.g. chroma:///path/db#collection")
    ],
    reference: Annotated[
        str,
        typer.Option(
            "--reference",
            help=(
                "A local model to align against — the public one an adversary "
                "would reach for. A hosted endpoint is refused"
            ),
        ),
    ] = DEFAULT_REFERENCE,
    sample: Annotated[
        int, typer.Option("--sample", help="Documents drawn from the index")
    ] = 20_000,
    heldout: Annotated[
        int,
        typer.Option(
            "--heldout",
            help=(
                "Documents kept out of the fit and used to measure it. They are "
                "ranked against each other, so this is the pool the number means"
            ),
        ),
    ] = 1_000,
    seeds: Annotated[
        int,
        typer.Option(
            "--seeds",
            help=(
                "Independent alignments to run. The method is stochastic and a "
                "single attempt can be off by 0.9 on the same index; the answer "
                "is the best of them"
            ),
        ),
    ] = SEEDS,
    strategy: Annotated[
        str, typer.Option("--strategy", help="stratified|random sampling of the corpus")
    ] = "stratified",
    seed: Annotated[int, typer.Option("--seed", help="Recorded so the run can be replayed")] = 0,
    device: Annotated[str, typer.Option("--device", help="auto|cpu|cuda|cuda:N|mps")] = "auto",
    reference_dim: Annotated[
        int | None,
        typer.Option("--reference-dim", help="Dimension, for a model rebasis does not know"),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the measurement as JSON on stdout")
    ] = False,
) -> None:
    """Measure how well this index aligns to a space somebody else already has.

    Reads the index; never writes to it, and returns no vector and no text.

    An adversary holding only your vectors, plus a public embedding model over
    their own documents, can fit a map between the two spaces without any paired
    data at all — published, and cheap since mini-vec2vec reduced it to an
    orthogonal solve. This measures how well that works on your index.
    """
    from rebasis.cli._common import console
    from rebasis.cli._pipeline import (
        ProfileOverrides,
        open_embedders,
        open_target_store,
        print_exposure,
        print_exposure_json,
    )
    from rebasis.probe.exposure import measure_exposure

    opened = open_target_store(store)
    _, model = open_embedders(
        None,
        reference,
        device=device,
        new_overrides=ProfileOverrides(dim=reference_dim),
        store_dim=None,
    )

    if not as_json:
        console.print(
            "[dim]Read-only. This command returns a number: no vectors, no text, "
            "no translation.[/dim]"
        )
    with step_progress("Sampling the index") as steps:
        result = measure_exposure(
            opened,
            model,
            size=sample,
            heldout=heldout,
            strategy=strategy,
            seed=seed,
            seeds=seeds,
            on_stage=steps.stage,
        )

    if as_json:
        print_exposure_json(result)
        return
    print_exposure(result)
