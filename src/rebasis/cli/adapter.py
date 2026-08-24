"""``rebasis adapter`` — inspect, verify, upgrade and list.

The commands that are about the ``.rbs`` file itself rather than about an
index. `upgrade` is the one that matters: it is referenced from the error a
user sees when a file is too old to read, so it has to exist and it has to
never destroy the file it was pointed at.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rebasis.cli._common import console, handle_errors
from rebasis.errors import ConfigError

__all__ = ["adapter_app"]

adapter_app = typer.Typer(
    name="adapter",
    help="Inspect (--verify checks integrity), upgrade and list profiles for .rbs files.",
    no_args_is_help=True,
)


@adapter_app.command("upgrade")
@handle_errors
def upgrade_adapter(
    adapter: Annotated[Path, typer.Argument(help="Path to a .rbs adapter")],
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Where to write the upgraded copy"),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite --out if it already exists")
    ] = False,
) -> None:
    """Convert an adapter to the current schema.

    The original file is never modified: an upgrade writes a new directory, so a
    migration that turns out to be wrong costs nothing but disk. Schema
    migrations are forward-only, and that copy is the only way back.
    """
    import json

    from rebasis.core.migrations import upgrade_manifest
    from rebasis.core.serialization import CURRENT_SCHEMA

    manifest_path = Path(adapter) / "manifest.json"
    if not manifest_path.exists():
        raise ConfigError(
            f"{adapter} does not look like a .rbs adapter.",
            hint="An adapter is a directory containing manifest.json.",
            context={"path": str(adapter)},
        )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = int(payload.get("schema", 0))

    if current == CURRENT_SCHEMA:
        console.print(f"[green]{adapter.name}[/green] is already at schema {CURRENT_SCHEMA}.")
        return
    if current > CURRENT_SCHEMA:
        raise ConfigError(
            f"{adapter.name} uses schema {current}, newer than this release reads "
            f"({CURRENT_SCHEMA}).",
            hint="Upgrade rebasis: `pip install --upgrade rebasis`.",
            context={"count": current},
        )

    destination = out or adapter.with_name(f"{adapter.stem}-schema{CURRENT_SCHEMA}.rbs")
    if destination.exists() and not force:
        raise ConfigError(
            f"{destination} already exists.",
            hint="Pass --force to overwrite it, or --out to write elsewhere.",
            context={"path": str(destination)},
        )

    _copy_upgraded(adapter, destination, upgrade_manifest(payload, to_schema=CURRENT_SCHEMA))

    console.print(f"[green]upgraded[/green] schema {current} → {CURRENT_SCHEMA}")
    console.print(f"  original  {adapter}  [dim](untouched)[/dim]")
    console.print(f"  upgraded  {destination}")
    console.print("\n[dim]Verify it before removing the original:[/dim]")
    console.print(f"[dim]  rebasis eval {destination} --verify[/dim]")


def _copy_upgraded(source: Path, destination: Path, manifest: dict[str, object]) -> None:
    """Write the upgraded adapter, manifest last.

    Manifest last because it is the file that declares the schema: if the copy
    is interrupted, a directory with tensors and no manifest is obviously
    incomplete, while one with a manifest and missing tensors looks valid.
    """
    import shutil

    from rebasis.storage.atomic import atomic_write_json

    destination.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.iterdir()):
        if item.name == "manifest.json" or item.is_dir():
            continue
        shutil.copy2(item, destination / item.name)
    atomic_write_json(destination / "manifest.json", manifest)


@adapter_app.command("inspect")
@handle_errors
def inspect_adapter(
    adapter: Annotated[Path, typer.Argument(help="Path to a .rbs adapter")],
    verify: Annotated[bool, typer.Option("--verify", help="Recompute every tensor hash")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output")] = False,
) -> None:
    """Show what an adapter file contains, without touching a store."""
    import json

    from rebasis.core import load_adapter

    loaded, manifest, calibrator = load_adapter(adapter, verify=verify)

    if json_output:
        console.print_json(
            json.dumps(
                {
                    **manifest.to_json(),
                    "n_params": loaded.n_params(),
                    "memory_bytes": loaded.memory_bytes,
                    "calibrator": calibrator is not None,
                    "integrity": "verified" if verify else "header_only",
                }
            )
        )
        return

    console.print(f"[bold]{adapter.name}[/bold]")
    console.print(f"  schema        {manifest.schema}")
    console.print(f"  adapter       {manifest.adapter_type}")
    console.print(f"  direction     {manifest.direction}")
    console.print(f"  dimensions    {manifest.input_dim} → {manifest.output_dim}")
    console.print(f"  old model     {manifest.old_model_id}")
    console.print(f"  new model     {manifest.new_model_id}")
    console.print(f"  symmetric     {manifest.symmetric}")
    console.print(f"  built with    rebasis {manifest.rebasis_version}")
    console.print(f"  created       {manifest.created_utc or '—'}")
    console.print(f"  parameters    {loaded.n_params():,} ({loaded.memory_bytes / 1024**2:.2f} MB)")
    console.print(f"  calibrator    {'present' if calibrator else 'none'}")
    console.print(f"  integrity     {'verified' if verify else 'header only — pass --verify'}")


@adapter_app.command("profiles")
@handle_errors
def list_profiles(
    search: Annotated[
        str | None, typer.Argument(help="Only show models whose id contains this")
    ] = None,
) -> None:
    """List the encoding profiles rebasis knows.

    A model not listed here still works: pass its dimension, and its prefixes
    if it is asymmetric. rebasis will not guess a prefix, because guessing wrong
    lowers quality with no error to explain it.
    """
    from rich.table import Table

    from rebasis.embed import known_profiles

    profiles = known_profiles()
    if search:
        profiles = {k: v for k, v in profiles.items() if search.lower() in k.lower()}

    if not profiles:
        console.print(f"[yellow]No profile matches {search!r}.[/yellow]")
        raise typer.Exit(code=1)

    table = Table(title=f"{len(profiles)} known encoding profiles", title_justify="left")
    table.add_column("Model")
    table.add_column("Dim", justify="right")
    table.add_column("Asymmetric")
    table.add_column("Query prefix")
    table.add_column("Document prefix")

    for model_id, profile in sorted(profiles.items()):
        table.add_row(
            model_id,
            str(profile.dim),
            "[yellow]yes[/yellow]" if not profile.symmetric else "no",
            profile.query_prefix or "—",
            profile.document_prefix or "—",
        )
    console.print(table)
    console.print(
        "\n[dim]For a model not listed: --new-dim, plus --query-prefix / "
        "--document-prefix if it is asymmetric.[/dim]"
    )
