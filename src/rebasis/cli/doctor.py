"""``rebasis doctor`` — report the environment.

The first thing to run when something is slow or a GPU is not being used. Every
line here answers a question that is otherwise expensive to investigate:

* **BLAS threads** — the most-missed performance trap in scientific Python.
  numpy is single-threaded, but its BLAS backend is not; combined with worker
  processes the thread count multiplies, and 64 threads on 8 cores is slower
  than 8.
* **torch built without CUDA** — the most common cause of "I have a GPU but it
  is not being used", and hard to spot otherwise.
* **TF32** — silently lowers matmul precision on Ampere and later, which is not
  acceptable while computing a decision metric.
"""

from __future__ import annotations

import platform
import sys
from typing import Annotated, Any

import typer
from rich.table import Table

from rebasis.cli._common import console, handle_errors

__all__ = ["doctor_command"]


@handle_errors
def doctor_command(
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit the environment as JSON, for bug reports and CI"),
    ] = False,
) -> None:
    """Report the environment, devices and configuration."""
    if as_json:
        _print_json()
        return

    from rebasis import __version__
    from rebasis.compute import (
        blas_info,
        describe_devices,
        resolve_device,
        should_use_accelerator,
        torch_available,
        user_set_thread_limits,
    )
    from rebasis.embed import available_embedders, known_profiles
    from rebasis.observability import current_settings, telemetry_status
    from rebasis.store import available_backends

    table = Table(title="rebasis doctor", show_header=False, box=None)
    table.add_column(style="dim", width=22)
    table.add_column()

    table.add_row("rebasis", __version__)
    table.add_row("python", sys.version.split()[0])
    table.add_row("platform", platform.platform())
    table.add_row("", "")

    if torch_available():
        import torch

        table.add_row("torch", torch.__version__)
        if not torch.cuda.is_available():
            # The single most common "my GPU is not being used" cause.
            table.add_row(
                "",
                "[yellow]torch reports no CUDA device. If this machine has an "
                "NVIDIA GPU, the CPU-only wheel is probably installed. Reinstall "
                "with the CUDA index URL.[/yellow]",
            )
        tf32 = torch.backends.cuda.matmul.allow_tf32
        table.add_row(
            "TF32 (matmul)",
            "[green]off[/green]" if not tf32 else "[yellow]on — lowers precision[/yellow]",
        )
    else:
        table.add_row("torch", "[dim]not installed — the core path does not need it[/dim]")

    table.add_row("", "")
    for capability in describe_devices():
        memory = (
            f"{capability.total_memory_bytes / 1024**3:.1f} GB"
            if capability.total_memory_bytes
            else ""
        )
        table.add_row(f"device {capability.device}", f"{capability.name} {memory}".strip())
    selected = resolve_device("auto")
    has_accelerator = not selected.is_cpu

    # The per-sub-job assignment, not one device for the whole run: `auto`
    # decides each operation independently, so embedding may run on the GPU
    # while Procrustes and the hot path stay on the CPU.
    def placement(job: str, size: int) -> str:
        on_device = should_use_accelerator(job, size, device_available=has_accelerator)
        return f"{job}→{selected if on_device else 'cpu'}"

    assignments = " · ".join(
        placement(job, size)
        for job, size in (("embedding", 10_000), ("knn", 10_000), ("procrustes_fit", 10_000))
    )
    table.add_row("selected (auto)", assignments)
    table.add_row("hot path", "cpu [dim](always — transfer exceeds the budget)[/dim]")

    table.add_row("", "")
    status, threads = blas_info()
    if status == "unavailable":
        table.add_row("BLAS", "[dim]threadpoolctl not installed[/dim]")
    elif status == "opaque":
        # Apple Accelerate is the common case: threadpoolctl looks for
        # OpenBLAS, MKL or BLIS shared libraries and Accelerate is none of them.
        # Saying "not installed" here would send the user down the wrong path.
        table.add_row(
            "BLAS",
            "[dim]present but not introspectable (this backend reports no thread "
            "pools). Oversubscription cannot be detected here.[/dim]",
        )
    else:
        note = (
            " [dim](set by you; rebasis will not override it)[/dim]"
            if user_set_thread_limits()
            else ""
        )
        table.add_row("BLAS threads", f"{threads}{note}")

    table.add_row("", "")
    table.add_row("store backends", ", ".join(sorted(available_backends())))
    table.add_row("embed backends", ", ".join(sorted(available_embedders())))
    table.add_row("known profiles", str(len(known_profiles())))

    from rebasis.compute import deterministic_mode_enabled

    table.add_row(
        "determinism",
        "on"
        if deterministic_mode_enabled()
        else "off [dim](REBASIS_DETERMINISTIC=1 enables it)[/dim]",
    )

    settings = current_settings()
    table.add_row("log level", settings.level_name if settings else "[dim]library mode[/dim]")
    table.add_row("telemetry", _telemetry_row(telemetry_status()))

    console.print(table)
    _print_environment()
    _warn_about_openmp_conflict()


def openmp_conflict() -> str:
    """Two OpenMP runtimes in one process, or "" when there is only one.

    `faiss-cpu` and `torch` each ship their own on macOS, and the second to
    initialise aborts the process with `OMP: Error #15` — before either library
    has done any work. Measured directly: faiss alone runs a reconstruct and
    sixty searches without complaint, and the same script with `import torch`
    in front of it dies before the first call.

    Neither wheel can be told not to link it, and the workaround the error
    itself suggests, `KMP_DUPLICATE_LIB_OK=TRUE`, is documented as liable to
    "silently produce incorrect results".

    Reported rather than worked around. `doctor` exists to say what this machine
    can do, and "your FAISS index and your torch embeddings cannot be used from
    one process" is exactly that kind of fact — better learned here than from an
    abort halfway through a migration.
    """
    import importlib.util

    if sys.platform != "darwin":
        return ""
    if importlib.util.find_spec("faiss") is None or importlib.util.find_spec("torch") is None:
        return ""
    return (
        "faiss and torch are both installed, and on macOS each links its own "
        "OpenMP runtime: loading both in one process aborts it. Use one or the "
        "other per process — `rebasis[faiss]` without the torch extra, or a "
        "torch-free embedding backend such as fastembed. Upstream: "
        "faiss-wheels#40, pytorch#149201."
    )


def _warn_about_openmp_conflict() -> None:
    """Say so when this environment cannot load both libraries."""
    conflict = openmp_conflict()
    if conflict:
        console.print()
        console.print(f"[yellow]incompatible pair[/yellow]  {conflict}")


def _print_json() -> None:
    """The same facts, structured.

    `doctor` is the command a user is asked to run when they open an issue, and
    a screenshot of a Rich table is a poor attachment. This is the version of it
    that can be pasted, diffed and asserted on in CI.
    """
    import json

    from rebasis import __version__
    from rebasis.compute import (
        blas_info,
        describe_devices,
        deterministic_mode_enabled,
        resolve_device,
        torch_available,
    )
    from rebasis.embed import available_embedders, known_profiles
    from rebasis.observability import current_settings, telemetry_status
    from rebasis.store import available_backends

    status, threads = blas_info()
    settings = current_settings()
    payload: dict[str, Any] = {
        "rebasis": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": _torch_version() if torch_available() else None,
        "devices": [
            {
                "device": str(capability.device),
                "name": capability.name,
                "total_memory_bytes": capability.total_memory_bytes,
            }
            for capability in describe_devices()
        ],
        "selected_device": str(resolve_device("auto")),
        "blas": {"status": status, "threads": threads},
        "store_backends": sorted(available_backends()),
        "embed_backends": sorted(available_embedders()),
        "known_profiles": len(known_profiles()),
        "deterministic": deterministic_mode_enabled(),
        "log_level": settings.level_name if settings else None,
        "telemetry": telemetry_status(),
    }
    payload["openmp_conflict"] = openmp_conflict() or None
    console.print_json(json.dumps(payload, default=str))


def _torch_version() -> str:
    import torch

    return str(torch.__version__)


def _print_environment() -> None:
    """List the environment variables rebasis reads.

    Knobs that exist only in call sites are knobs nobody can find. This is the
    discoverable copy, and `config.py` is the single place they are read.
    """
    from rebasis.config import ENVIRONMENT_VARIABLES, settings

    resolved = settings()
    active = {
        "REBASIS_DEVICE": resolved.device if resolved.device != "auto" else "",
        "REBASIS_STATE_DIR": resolved.state_dir or "",
        "REBASIS_MAX_MEMORY": (
            f"{resolved.max_memory_bytes / 1024**3:.1f}GB" if resolved.max_memory_bytes else ""
        ),
        "REBASIS_DETERMINISTIC": "1" if resolved.deterministic else "",
        "REBASIS_OTEL_ENABLED": "1" if resolved.otel_enabled else "",
    }

    table = Table(title="environment", show_header=False, box=None, title_justify="left")
    table.add_column(style="dim", width=26)
    table.add_column(width=10)
    table.add_column(style="dim")
    for name, description in ENVIRONMENT_VARIABLES.items():
        value = active.get(name, "")
        table.add_row(name, f"[green]{value}[/green]" if value else "[dim]unset[/dim]", description)
    console.print()
    console.print(table)


def _telemetry_row(status: dict[str, object]) -> str:
    """Say which of the three off-states this is.

    "off" alone sends a user who set `REBASIS_OTEL_ENABLED=1` and saw nothing
    looking in the wrong place. The three cases have three different fixes.
    """
    if not status["available"]:
        # `\[` escapes the bracket: rich would otherwise read `[otel]` as a
        # style tag and strip it, leaving `pip install "rebasis"` — a hint that
        # tells the user to install what they already have.
        return 'off [dim](install `pip install "rebasis\\[otel]"`)[/dim]'
    if not status["enabled"]:
        return "off [dim](REBASIS_OTEL_ENABLED=1 enables it)[/dim]"
    endpoint = str(status["endpoint"] or "")
    if not endpoint:
        return "on [yellow](no OTEL_EXPORTER_OTLP_ENDPOINT — nothing is exported)[/yellow]"
    return f"on [dim]→ {endpoint}[/dim]"
