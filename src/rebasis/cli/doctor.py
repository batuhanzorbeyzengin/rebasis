"""``rebasis doctor`` — report the environment, and one index when asked.

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

``--store`` adds the half that needs a URI: the same command pointed at a live
index. Three rules shape it.

**It is read-only in every path, including the local one.** ``doctor`` is what a
user runs when they are already confused, so it is the last command that should
be able to change anything. Nothing here opens a store for writing, and the
manifest is opened only when this release would not migrate it — migrating is a
write, and a diagnostic that upgrades a schema behind the user's back has
altered the thing it was asked to describe.

``--calibrate`` is the one exception, and it is named as one. It times this
machine and writes ``calibration.json`` into the state directory — never near
the index, never a store, and only when it is asked for by name. Anything that
writes should be impossible to trigger by accident from a command whose whole
promise is that it does not.

**A check that fails does not take the report with it.** A store that will not
open is the most likely reason someone is running this at all, and the reason it
would not open is the answer they came for. Every check returns a verdict,
including "could not be determined", and the rest of the report still prints.

**Nothing is claimed that has not been measured.** The checks below either read a
fact out of the index or compare it against something rebasis itself recorded
earlier. Where nothing was recorded, the output says so — that is a different
answer from a clean bill of health, and the two are never rendered the same way.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass, field

# A runtime import, not a TYPE_CHECKING one: typer resolves the annotation of
# every option at runtime to decide how to parse it, so `--state-dir` needs
# `Path` to exist when the command is registered.
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.markup import escape
from rich.table import Table

from rebasis.cli._common import console, err_console, handle_errors

if TYPE_CHECKING:
    from collections.abc import Callable

    from rebasis.core.serialization import AdapterManifest
    from rebasis.manifest import ManifestDB
    from rebasis.migrate import MixedSpace
    from rebasis.store.base import VectorStore
    from rebasis.store.uri import StoreURI

__all__ = ["StoreCheck", "StoreReport", "doctor_command"]

#: The first sixteen bytes of every SQLite database ever written.
_SQLITE_MAGIC = b"SQLite format 3\x00"

#: `user_version` lives at this offset in the SQLite header, big-endian, four
#: bytes wide. The header is 100 bytes and its layout has been frozen since the
#: format was defined.
_USER_VERSION_OFFSET = 60
_SQLITE_HEADER_BYTES = 100

#: What a table cell says where the index would not answer.
_UNKNOWN = "[dim]unknown[/dim]"

#: The three verdicts, padded to one width so the column reads down.
#:
#: "unknown" is deliberately not styled as a warning. A check that could not run
#: is not a problem with the user's index, and colouring it like one sends
#: somebody looking for a fault that was never reported.
_VERDICTS: dict[bool | None, str] = {
    True: "[green]ok     [/green]",
    False: "[red]problem[/red]",
    None: "[dim]unknown[/dim]",
}

#: Chroma's persistent client keeps its metadata in this file inside the
#: directory a `chroma://` URI names. Checked by header magic like any other
#: candidate, so a layout change downgrades the check to "not reachable" rather
#: than producing a wrong answer.
_CHROMA_SQLITE = "chroma.sqlite3"


@dataclass(frozen=True, slots=True)
class StoreCheck:
    """One question asked of an index, and the answer.

    ``ok`` is three-valued on purpose. ``True`` and ``False`` are verdicts;
    ``None`` means the check could not be run — no baseline was recorded, the
    store never opened, the backend keeps nothing this can read. Collapsing that
    third state into either of the others is how a diagnostic starts lying: a
    comparison that was never made is not a comparison that passed.
    """

    name: str
    ok: bool | None
    detail: str
    hint: str | None = None


@dataclass(frozen=True, slots=True)
class StoreReport:
    """Everything ``doctor --store`` found, in the shape ``--json`` emits.

    Built before anything is printed, so the human report and the JSON one are
    the same findings rendered twice rather than two code paths that have to be
    kept in agreement.
    """

    uri: str
    backend: str | None
    opened: bool
    records: int | None
    dimension: int | None
    capabilities: dict[str, Any] | None
    recorded: dict[str, Any] | None
    mixed_spaces: list[MixedSpace] = field(default_factory=list)
    checks: list[StoreCheck] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form, for ``doctor --json``.

        The URI a mixed-space record carries is replaced with the redacted one.
        `status` emits it as stored, and `status` is not the command the README
        tells people to paste into a public issue; this one is.
        """
        return {
            "uri": self.uri,
            "backend": self.backend,
            "opened": self.opened,
            "records": self.records,
            "dimension": self.dimension,
            "capabilities": self.capabilities,
            "recorded": self.recorded,
            "mixed_spaces": [
                {**state.to_dict(), "store_uri": self.uri} for state in self.mixed_spaces
            ],
            "checks": [asdict(check) for check in self.checks],
        }


@handle_errors
def doctor_command(
    store: Annotated[
        str | None,
        typer.Option("--store", help="Also check this index. Read-only; nothing is written"),
    ] = None,
    state_dir: Annotated[
        Path | None,
        typer.Option("--state-dir", help="Where rebasis state lives; defaults to ./.rebasis"),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit the environment as JSON, for bug reports and CI"),
    ] = False,
    calibrate: Annotated[
        bool,
        typer.Option(
            "--calibrate",
            help=(
                "Time this machine and record the result. The only path that "
                "writes, and it writes only into the state directory"
            ),
        ),
    ] = False,
) -> None:
    """Report the environment, devices and configuration.

    Pass --store to check a live index as well: whether it opens and why not if
    it does not, what it holds, whether its file is intact, and whether a
    half-finished migration has left it holding two embedding spaces. Every one
    of those reads; none of them writes.

    Pass --calibrate to replace the reference speedups with this machine's. The
    numbers in `rebasis.compute.thresholds` were measured on one GPU against one
    CPU, and a faster host narrows every one of them — so the honest thing for a
    diagnostic to do is measure rather than repeat. It records only what it can
    time without downloading anything, and says which keys those were.
    """
    report = _inspect_store(store, state_dir) if store is not None else None
    measured = _run_calibration(state_dir) if calibrate else None

    if as_json:
        _print_json(report, measured)
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
    # Off the file rather than off `measured`, so a calibration taken on an
    # earlier run shows up too. A number measured last week on this machine is
    # more use than one measured last year on somebody else's.
    _add_calibration_row(table, state_dir)

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
    if measured is not None:
        console.print()
        _print_calibration(measured)
    _print_environment()
    _warn_about_openmp_conflict()
    # Last, and deliberately: the environment is the same on every run, and the
    # index is the thing the person passing `--store` came to read. It should be
    # what is still on screen when the command finishes.
    if report is not None:
        _print_store(report)


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

    # `platform.system()`, not `sys.platform`: mypy narrows the latter to the
    # platform it is running on, so on Linux this function's body becomes
    # unreachable and the check that reports a macOS-only conflict fails to
    # type-check on the only platform CI runs.
    if platform.system() != "Darwin":
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


# ── local calibration ────────────────────────────────────────────────────────
#
# `rebasis.compute.thresholds` records speedups measured on one GPU against one
# CPU, and says so: a faster host narrows every one of them. These functions
# measure the machine in front of them instead.
#
# Two rules, and they are the same two the rest of the project runs on.
#
# **Measured or omitted, never estimated.** `embed` dominates a probe and needs a
# model, which needs a download; a diagnostic that pulled 400 MB to answer a
# question nobody asked would be a bad citizen, so it is not measured and not
# guessed. `worth_accelerating` falls back per key, so an absent one costs
# nothing.
#
# **A failure is a finding, not a crash.** `doctor` is what somebody runs when
# their environment is already broken. Every measurement is wrapped: one that
# raises records nothing and the report still prints.

#: kNN benchmark shape. Chosen to be a realistic probe's ground-truth search
#: rather than a microbenchmark: 20k documents is a small index, 512 queries is
#: a probe's sample, 768 is the dimensionality of the models the ladder ends on.
KNN_SHAPE = {"queries": 512, "documents": 20_000, "dim": 768, "k": 10}

#: MLP benchmark shape. `epochs` is far below the 60 a real fit runs, because
#: what is wanted is the per-epoch ratio and not the wall clock of a fit nobody
#: asked for. Recorded in `notes` so the number is never read as a fit time.
MLP_SHAPE = {"pairs": 2000, "dim": 384, "epochs": 3}

#: Timed runs per measurement, after one warm-up. The median is taken: an
#: accelerator's first call pays for kernel loading and allocation, and a mean
#: over three would carry a third of that.
REPEATS = 3


def _median_seconds(work: Callable[[], object], *, repeats: int = REPEATS) -> float:
    """Run ``work`` once to warm up, then ``repeats`` times, and take the median."""
    import statistics
    import time

    work()
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        work()
        timings.append(time.perf_counter() - started)
    return statistics.median(timings)


def _time_knn(device: Any) -> float:
    """CPU seconds divided by accelerator seconds for a ground-truth kNN.

    The same :func:`~rebasis.compute.search.top_k_search` a probe calls, with
    the device passed explicitly rather than through the ambient context, so the
    two arms differ in the device and in nothing else.
    """
    import numpy as np

    from rebasis.compute import l2_normalize, top_k_search
    from rebasis.compute.device import resolve_device

    rng = np.random.default_rng(0)
    shape = KNN_SHAPE
    queries = l2_normalize(rng.standard_normal((shape["queries"], shape["dim"])).astype(np.float32))
    documents = l2_normalize(
        rng.standard_normal((shape["documents"], shape["dim"])).astype(np.float32)
    )
    cpu = resolve_device("cpu")

    on_cpu = _median_seconds(lambda: top_k_search(queries, documents, k=shape["k"], device=cpu))
    on_device = _median_seconds(
        lambda: top_k_search(queries, documents, k=shape["k"], device=device)
    )
    return on_cpu / on_device if on_device > 0 else 0.0


def _time_mlp(device: Any) -> float:
    """CPU seconds divided by accelerator seconds for the residual MLP's fit."""
    import numpy as np

    from rebasis.compute import l2_normalize
    from rebasis.core.residual_mlp import ResidualMLPAdapter

    rng = np.random.default_rng(0)
    shape = MLP_SHAPE
    src = l2_normalize(rng.standard_normal((shape["pairs"], shape["dim"])).astype(np.float32))
    rotation = np.linalg.qr(rng.standard_normal((shape["dim"], shape["dim"])))[0]
    dst = l2_normalize(src @ rotation.T.astype(np.float32))

    def fit(where: str) -> object:
        return ResidualMLPAdapter.fit(src, dst, epochs=shape["epochs"], device=where)

    on_cpu = _median_seconds(lambda: fit("cpu"), repeats=1)
    on_device = _median_seconds(lambda: fit(str(device)), repeats=1)
    return on_cpu / on_device if on_device > 0 else 0.0


def _run_calibration(state_dir: Path | None) -> Any:
    """Time this machine, write the result, and return it.

    Returns ``None`` when there is no accelerator to compare against — a
    calibration of a CPU against itself is 1.0 by construction and would
    overwrite the reference table with an arithmetic identity.
    """
    import datetime
    import socket

    from rebasis.compute import resolve_device, torch_available
    from rebasis.compute.thresholds import Calibration, calibration_path
    from rebasis.manifest import default_state_dir
    from rebasis.storage.atomic import atomic_write_json

    device = resolve_device("auto")
    if device.is_cpu:
        console.print(
            "[yellow]No accelerator to calibrate against.[/yellow] The recorded "
            "speedups are ratios of CPU time to device time, and there is no "
            "device here; nothing was written."
        )
        return None

    speedups: dict[str, float] = {}
    notes: dict[str, Any] = {}
    console.print(f"[dim]timing this machine against {device}...[/dim]")

    measurements: list[tuple[str, Callable[[], float], dict[str, Any]]] = [
        ("knn", lambda: _time_knn(device), dict(KNN_SHAPE)),
    ]
    if torch_available():
        measurements.append(("fit_mlp", lambda: _time_mlp(device), dict(MLP_SHAPE)))

    for name, measure, shape in measurements:
        try:
            speedups[name] = round(measure(), 2)
        except Exception as exc:  # noqa: BLE001 - a broken environment is the usual caller
            console.print(f"  [yellow]{name}: not measured — {type(exc).__name__}[/yellow]")
            continue
        notes[name] = shape

    if not speedups:
        console.print("[yellow]Nothing could be measured; nothing was written.[/yellow]")
        return None

    calibration = Calibration(
        device=str(device),
        speedups=speedups,
        measured_utc=datetime.datetime.now(tz=datetime.UTC).isoformat(),
        host=socket.gethostname(),
        notes=notes,
    )
    directory = state_dir or default_state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(calibration_path(directory), calibration.to_dict())
    return calibration


def _add_calibration_row(table: Table, state_dir: Path | None) -> None:
    """Name the calibration this machine already has, if it has one.

    Read off the file rather than off this run's measurement, so a calibration
    taken on an earlier run shows up too: a number measured last week on this
    machine is more use than one measured last year on somebody else's.
    """
    from rebasis.compute import load_calibration
    from rebasis.manifest import default_state_dir

    stored = load_calibration(state_dir or default_state_dir())
    if stored is None:
        return
    table.add_row(
        "calibrated",
        f"{stored.device} on {stored.host or 'this machine'}, "
        f"{stored.measured_utc[:10] or 'date not recorded'}",
    )


def _print_calibration(calibration: Any) -> None:
    """Show what was measured here against what was measured on the reference host."""
    from rebasis.compute.thresholds import MEASURED_SPEEDUPS, WORTH_IT

    table = Table(title="calibration", box=None)
    table.add_column("operation", style="dim")
    table.add_column("here", justify="right")
    table.add_column("reference", justify="right")
    table.add_column("")
    for name in sorted(MEASURED_SPEEDUPS):
        local = calibration.speedups.get(name)
        reference = MEASURED_SPEEDUPS[name]
        if local is None:
            table.add_row(name, "[dim]not measured[/dim]", f"{reference:.1f}x", "")
            continue
        verdict = (
            "[green]worth the accelerator[/green]"
            if local >= WORTH_IT
            else "[yellow]not worth it here[/yellow]"
        )
        table.add_row(name, f"{local:.1f}x", f"{reference:.1f}x", verdict)
    console.print(table)
    console.print(
        "[dim]`embed` needs a model, so it is not measured rather than guessed. "
        "A key with no local number falls back to the reference one.[/dim]"
    )
    console.print(
        "[dim]This is a diagnostic: nothing in the runtime dispatches per "
        "operation, so it changes what is reported and not where work runs.[/dim]"
    )


def _print_json(store: StoreReport | None, calibration: Any = None) -> None:
    """The same facts, structured.

    `doctor` is the command a user is asked to run when they open an issue, and
    a screenshot of a Rich table is a poor attachment. This is the version of it
    that can be pasted, diffed and asserted on in CI.

    The `store` key is `null` without `--store`, so every key that was here
    before is still here and still means what it meant: a script reading
    `.blas.threads` out of a bug report does not have to know which flags the
    reporter passed.
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
    payload["store"] = store.to_dict() if store is not None else None
    # `null` without `--calibrate`, and null when there was nothing to measure
    # against. Present either way, for the same reason `store` is: a script
    # reading a bug report should not have to know which flags were passed.
    payload["calibration"] = calibration.to_dict() if calibration is not None else None
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


# ── the index ─────────────────────────────────────────────────────────
#
# Everything below is read-only. The store is opened through the same
# `open_store` a real command uses — going around it would test a path nobody
# takes — and closed again on the way out, because a local Qdrant or LanceDB
# holds an exclusive lock on its directory while a handle is open.


def _redact(text: str) -> str:
    """Remove anything credential-shaped from a string bound for the report.

    `StoreURI.redacted` is the route wherever the URI parsed; this is the one
    for wherever it did not, and `--json` is what the README tells people to
    attach to a public issue. The rule itself lives in `store.uri`, which now
    applies it to its own `InvalidStoreURI` messages as well — two copies of a
    redaction rule is one copy that stops being applied.
    """
    from rebasis.store.uri import redact_credentials

    return redact_credentials(text)


def _inspect_store(uri: str, state_dir: Path | None) -> StoreReport:
    """Run every store check, and let none of them stop the others.

    The order is the order a person needs the answers in: does the string
    parse, does the index open, what is in it, is its file intact, and — last,
    because it is the one that explains a quality collapse nothing else
    explains — is the collection holding two embedding spaces at once.
    """
    parsed, uri_check = _check_uri(uri)
    store, open_check = _open_store(parsed)
    try:
        facts = _read_facts(store)
        manifest_checks, recorded, mixed = _inspect_manifest(uri, state_dir, facts.dimension)
    finally:
        _close(store)

    return StoreReport(
        uri=parsed.redacted() if parsed is not None else _redact(uri),
        backend=facts.backend or (parsed.backend if parsed is not None else None),
        opened=store is not None,
        records=facts.records,
        dimension=facts.dimension,
        capabilities=facts.capabilities,
        recorded=recorded,
        mixed_spaces=mixed,
        checks=[uri_check, open_check, facts.text, _check_sqlite(parsed), *manifest_checks],
    )


def _check_uri(uri: str) -> tuple[StoreURI | None, StoreCheck]:
    """Whether the string is a store URI at all.

    First because a typo here is the cheapest possible failure to diagnose and
    the most common one a new user hits: `parse_store_uri` already names the
    shape it wanted, so this passes that message straight through.
    """
    from rebasis.errors import RebasisError
    from rebasis.store import parse_store_uri

    try:
        parsed = parse_store_uri(uri)
    except RebasisError as exc:
        return None, StoreCheck(
            "uri", ok=False, detail=f"{exc.code} {_redact(exc.message)}", hint=exc.hint
        )
    collection = parsed.collection or "(none)"
    return parsed, StoreCheck(
        "uri", ok=True, detail=f"backend {parsed.backend}, collection {collection}"
    )


def _open_store(parsed: StoreURI | None) -> tuple[VectorStore | None, StoreCheck]:
    """Open the index, and treat failing to as the finding it usually is.

    A raw third-party exception reaching here is a defect: the contract is that
    every backend converts its client library's exceptions into a
    `RebasisError`. It is still caught, because `doctor` refusing to print the
    rest of its report is not an improvement on a backend that leaked.
    """
    if parsed is None:
        return None, StoreCheck("open", ok=None, detail="not attempted: the URI did not parse")

    from rebasis.errors import RebasisError
    from rebasis.store import open_store

    try:
        store = open_store(parsed)
    except RebasisError as exc:
        return None, StoreCheck(
            "open", ok=False, detail=f"{exc.code} {_redact(exc.message)}", hint=exc.hint
        )
    except Exception as exc:  # noqa: BLE001 - a broken backend must not silence the report
        return None, StoreCheck(
            "open",
            ok=False,
            detail=_redact(f"{type(exc).__name__}: {exc}"),
            hint=(
                "A backend is meant to convert its client library's exceptions into a "
                "rebasis error, so this one is a bug. Please report it with the output "
                "of `rebasis doctor --json`."
            ),
        )
    return store, StoreCheck("open", ok=True, detail=f"opened by the {parsed.backend} backend")


@dataclass(frozen=True, slots=True)
class _Facts:
    """What the open index says about itself."""

    backend: str | None
    records: int | None
    dimension: int | None
    capabilities: dict[str, Any] | None
    text: StoreCheck


def _read_facts(store: VectorStore | None) -> _Facts:
    """Count, dimension, capabilities and whether text comes back."""
    if store is None:
        return _Facts(
            backend=None,
            records=None,
            dimension=None,
            capabilities=None,
            text=StoreCheck("text", ok=None, detail="not attempted: the store did not open"),
        )
    capabilities = _capabilities(store)
    name = capabilities.get("name")
    return _Facts(
        backend=str(name) if name else None,
        records=_reading(store.count),
        dimension=_reading(store.dimension),
        capabilities=capabilities,
        text=_check_text(store, capabilities),
    )


def _capabilities(store: VectorStore) -> dict[str, Any]:
    """Every declared capability, read off the dataclass rather than listed.

    `StoreCapabilities` gains fields. A renderer with a hand-written line per
    field reports an index against last month's list and says nothing about the
    newest one — silently, which is the failure mode this whole file exists to
    avoid.
    """
    from dataclasses import fields

    try:
        declared = store.capabilities
    except Exception:  # noqa: BLE001 - a backend that cannot describe itself still gets reported
        return {}
    return {entry.name: getattr(declared, entry.name) for entry in fields(declared)}


def _reading(read: Callable[[], int]) -> int | None:
    """A number the index can supply, or ``None`` when it cannot.

    `sqlite-vec` raises rather than guessing the dimensionality of an empty
    table, and several bridge backends cannot count. Neither is a reason to
    abandon the rest of the report.
    """
    try:
        return int(read())
    except Exception:  # noqa: BLE001 - an unanswerable fact is reported as unanswered
        return None


def _check_text(store: VectorStore, capabilities: dict[str, Any]) -> StoreCheck:
    """Whether document text can actually be read back.

    Two different answers, kept apart. A backend that declares it cannot return
    text has ruled out `probe` entirely — `probe` asserts that capability before
    it starts. A backend that declares it can and then hands back a record with
    no text has not: one empty document proves nothing about the rest, so that
    is reported as an observation and not as a verdict.

    Costs one record. The text itself is never printed: it is corpus content,
    and corpus content does not belong in a diagnostic a user pastes into an
    issue.
    """
    if not capabilities.get("can_read_text"):
        return StoreCheck(
            "text",
            ok=False,
            detail="this backend declares it cannot return document text",
            hint="`rebasis probe` requires it and refuses a collection without it.",
        )
    try:
        records = store.iter_records(with_vectors=False, with_text=True, batch_size=1)
        first = next(iter(records), None)
    except Exception as exc:  # noqa: BLE001 - the reason it could not be read is the finding
        return StoreCheck("text", ok=None, detail=f"could not be read: {type(exc).__name__}: {exc}")
    if first is None:
        return StoreCheck(
            "text", ok=None, detail="the collection is empty, so there is none to read"
        )
    if not first.text:
        return StoreCheck(
            "text",
            ok=None,
            detail="declared readable, but the first record carries none",
        )
    return StoreCheck("text", ok=True, detail="readable")


def _close(store: VectorStore | None) -> None:
    """Give the handle back, where the backend has one to give."""
    import contextlib

    close = getattr(store, "close", None)
    if callable(close):
        with contextlib.suppress(Exception):
            close()


# ── the SQLite file under the index ───────────────────────────────────


def _sqlite_header(path: Path) -> tuple[bool, int]:
    """Whether a file is a SQLite database, and the ``user_version`` it carries.

    Read out of the header rather than over a connection, because deciding
    whether to *open* the manifest is precisely what one of the two answers is
    for: `ManifestDB` migrates on connect, and a migration is a write. Both
    fields sit at fixed offsets in a file format that has been backwards
    compatible since it was defined, and reading a hundred bytes cannot change
    anything — which a connection, in the general case, can.

    The manifest runs in WAL mode, so a schema change committed by another
    process and not yet checkpointed is not in this header yet. That errs the
    safe way: the answer is stale in the direction of "older than it is", and an
    older schema is the case where this declines to open the file at all.
    """
    try:
        with path.open("rb") as handle:
            header = handle.read(_SQLITE_HEADER_BYTES)
    except OSError:
        return False, 0
    if not header.startswith(_SQLITE_MAGIC):
        return False, 0
    end = _USER_VERSION_OFFSET + 4
    if len(header) < end:
        return True, 0
    return True, int.from_bytes(header[_USER_VERSION_OFFSET:end], "big")


def _sqlite_file(parsed: StoreURI) -> Path | None:
    """The SQLite database behind this URI, when there is one to reach.

    Which backends this covers is a question about the backends, not a
    judgement: `sqlite-vec` *is* a SQLite file, and a Chroma directory holds
    one. Everything is confirmed by header magic before it is treated as a
    database, so a wrong guess reports "not reachable" rather than a result.
    """
    if not parsed.path:
        return None
    path = Path(parsed.path)
    for candidate in (path, path / _CHROMA_SQLITE):
        if candidate.is_file() and _sqlite_header(candidate)[0]:
            return candidate
    return None


def _check_sqlite(parsed: StoreURI | None) -> StoreCheck:
    """Run SQLite's own integrity check against the index's database.

    A real check with a real answer: it either passes or names the damage, and
    it cannot write — the connection is opened `mode=ro`, which SQLite enforces
    rather than merely promising.

    It reads **every page**, so on a large index it is not free. That is why the
    size is announced on stderr before it starts, and why the whole store half
    of `doctor` is behind an explicit `--store` rather than running by default.
    """
    if parsed is None:
        return StoreCheck("sqlite", ok=None, detail="not attempted: the URI did not parse")

    path = _sqlite_file(parsed)
    if path is None:
        return StoreCheck(
            "sqlite",
            ok=None,
            detail=f"the {parsed.backend} backend keeps no SQLite database this can reach",
        )

    import sqlite3

    try:
        megabytes = path.stat().st_size / 1024**2
        # stderr: this is progress, not output. On stdout it would land inside
        # the document `--json` is piping into something.
        err_console.print(f"[dim]checking {megabytes:,.0f} MB of SQLite in {path.name}…[/dim]")
        # `as_uri` rather than an f-string: a path holding `?` or `#` would
        # otherwise be read as the query and fragment of the URI, and the
        # connection would silently be to a different file.
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            problems = [
                str(row[0])
                for row in connection.execute("PRAGMA integrity_check")
                if row[0] != "ok"
            ]
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as exc:
        return StoreCheck(
            "sqlite",
            ok=None,
            detail=f"{path.name} could not be opened read-only: {exc}",
            hint=(
                "The file may not be readable by this user, or it may be in WAL mode "
                "with no shared-memory file for a read-only connection to attach to — "
                "which usually means another process has it open."
            ),
        )

    if problems:
        return StoreCheck(
            "sqlite",
            ok=False,
            detail=f"{path.name}: {'; '.join(problems[:5])}",
            hint=(
                "SQLite reports this database as damaged. Restore it from a backup; "
                "rebasis will not write to a store in this state."
            ),
        )
    return StoreCheck("sqlite", ok=True, detail=f"{path.name} passes PRAGMA integrity_check")


# ── what rebasis recorded about this collection ───────────────────────


def _inspect_manifest(
    uri: str, state_dir: Path | None, dimension: int | None
) -> tuple[list[StoreCheck], dict[str, Any] | None, list[MixedSpace]]:
    """The three checks that need rebasis' own records, not the index.

    The manifest is opened only when this release would leave it exactly as it
    found it. `ManifestDB` runs its forward-only migrations on connect and takes
    a `VACUUM INTO` backup on the way — correct for `status` or `migrate`, and
    wrong for a command whose entire promise is that it changes nothing. So the
    schema is read out of the file header first, and a manifest this release
    would upgrade is reported rather than upgraded.
    """
    from rebasis.errors import RebasisError
    from rebasis.manifest import SCHEMA_VERSION, ManifestDB, default_state_dir, manifest_path

    directory = state_dir or default_state_dir()
    path = manifest_path(directory)
    if not path.exists():
        return _no_manifest(f"no rebasis state at {directory}"), None, []

    is_sqlite, schema = _sqlite_header(path)
    if not is_sqlite:
        return _no_manifest(f"{path} is not a SQLite database"), None, []
    if schema != SCHEMA_VERSION:
        return (
            _no_manifest(
                f"the manifest is at schema {schema} and this release uses {SCHEMA_VERSION}; "
                f"opening it would migrate it, and `doctor` does not write",
                hint="Run `rebasis status` to upgrade it, then run this again.",
            ),
            None,
            [],
        )

    try:
        db = ManifestDB(path)
        try:
            return _read_manifest(db, uri, dimension)
        finally:
            db.close()
    except RebasisError as exc:
        return _no_manifest(f"{exc.code} {exc.message}", hint=exc.hint), None, []


def _no_manifest(reason: str, hint: str | None = None) -> list[StoreCheck]:
    """The three manifest-backed checks, all unanswerable for the same reason.

    Said three times rather than once because they are three different
    questions, and a reader scanning for "spaces" should find the reason it is
    blank next to it rather than have to infer it from a line further up.
    """
    return [
        StoreCheck("manifest", ok=None, detail=reason, hint=hint),
        StoreCheck(
            "profile", ok=None, detail=f"no recorded profile for this collection — {reason}"
        ),
        StoreCheck("spaces", ok=None, detail=f"cannot be determined — {reason}"),
    ]


def _read_manifest(
    db: ManifestDB, uri: str, dimension: int | None
) -> tuple[list[StoreCheck], dict[str, Any] | None, list[MixedSpace]]:
    """Manifest integrity, the recorded profile, and the embedding spaces."""
    recorded, profile = _check_recorded(db, uri, dimension)
    mixed, spaces = _check_spaces(db, uri)
    return [_check_manifest_integrity(db), profile, spaces], recorded, mixed


def _check_manifest_integrity(db: ManifestDB) -> StoreCheck:
    """SQLite's checks against rebasis' own database.

    `ManifestDB.integrity_check` was written for this and had no caller: the
    file holds the audit trail, which is the one thing in rebasis that is never
    supposed to disappear, so damage to it is worth reporting even to a user who
    came here about something else entirely.
    """
    try:
        problems = db.integrity_check()
    except Exception as exc:  # noqa: BLE001 - an unreadable manifest is a finding, not a crash
        return StoreCheck("manifest", ok=None, detail=f"could not be checked: {exc}")
    if problems:
        return StoreCheck(
            "manifest",
            ok=False,
            detail="; ".join(problems[:5]),
            hint=(
                "The manifest holds the audit trail and every migration checkpoint. "
                "Restore it from a backup before running anything that writes."
            ),
        )
    settings = db.pragma_settings()
    return StoreCheck(
        "manifest",
        ok=True,
        detail=f"intact (schema {settings.get('user_version')}, {settings.get('journal_mode')})",
    )


def _check_recorded(
    db: ManifestDB, uri: str, dimension: int | None
) -> tuple[dict[str, Any] | None, StoreCheck]:
    """Compare the index against the encoding profile rebasis recorded for it.

    Two places hold one, and neither is guaranteed to. A `probe` writes its
    decision into the audit trail under the store URI as its subject, carrying
    both model ids and the candidate's whole profile. A `migrate` records the
    URI on the job together with the path of the ``.rbs`` it ran with, and that
    file's manifest carries the fingerprints of *both* profiles, both model ids
    and both dimensions — the format refuses to load on a fingerprint mismatch,
    which is what makes those fields trustworthy.

    Only one comparison is available against the live index, and it is the
    dimension. rebasis produces `query_to_old` adapters only, so the adapter's
    output dimension is the index's own space; if the collection does not have
    that dimensionality, that adapter was not fitted against this collection.
    The profile *fingerprints* cannot be checked here, because computing one
    needs the model, and `doctor` does not load models.

    Where neither record exists the answer is "no recorded profile for this
    collection". That is the honest output, and it is not the same as a pass.
    """
    probe = _last_probe(db, uri)
    adapter = _last_adapter(db, uri)
    if probe is None and adapter is None:
        return None, StoreCheck(
            "profile",
            ok=None,
            detail="no recorded profile for this collection",
            hint=(
                "rebasis records one when `probe` or `migrate` runs against this URI. "
                "A URI spelled differently is a different collection as far as this is "
                "concerned."
            ),
        )
    recorded: dict[str, Any] = {"probe": probe, "adapter": adapter}
    return recorded, _compare_dimension(adapter, dimension)


def _last_probe(db: ManifestDB, uri: str) -> dict[str, Any] | None:
    """The most recent probe decision recorded against this exact URI.

    Matched on the URI as the user spelled it, the same way `mixed_spaces_for`
    matches. Two spellings of one collection are two collections here, which
    errs toward reporting nothing rather than toward reporting somebody else's
    profile as this collection's.
    """
    import json

    from rebasis.observability import Events

    row = db.query_one(
        "SELECT ts_utc, inputs_json FROM audit_records "
        "WHERE subject = ? AND action = ? ORDER BY seq DESC LIMIT 1",
        (uri, str(Events.PROBE_DECISION_MADE)),
    )
    if row is None:
        return None
    try:
        inputs = json.loads(row["inputs_json"])
    except (TypeError, ValueError):
        return None
    if not isinstance(inputs, dict):
        return None
    return {
        "ts_utc": str(row["ts_utc"]),
        "old_model": inputs.get("old_model"),
        "new_model": inputs.get("new_model"),
        "new_profile": inputs.get("new_profile"),
        "new_profile_fingerprint": inputs.get("new_profile_fingerprint"),
        "records_at_probe": inputs.get("n_total"),
    }


def _last_adapter(db: ManifestDB, uri: str) -> dict[str, Any] | None:
    """The newest ``.rbs`` a migration recorded against this URI and still on disk.

    Newest first, and the first one whose file can still be read wins: an
    adapter that has been deleted or moved since the job ran says nothing about
    the index, and skipping past it finds the most recent one that does.
    """
    from rebasis.manifest import JobRow

    rows = db.query("SELECT * FROM jobs WHERE store_uri = ? ORDER BY created_utc DESC", (uri,))
    for raw in rows:
        job = JobRow.from_row(raw)
        if not job.adapter_path:
            continue
        manifest = _read_adapter_manifest(Path(job.adapter_path))
        if manifest is None:
            continue
        return {
            "job_id": job.job_id,
            "path": job.adapter_path,
            "adapter_type": manifest.adapter_type,
            "direction": manifest.direction,
            "input_dim": manifest.input_dim,
            "output_dim": manifest.output_dim,
            "old_model_id": manifest.old_model_id,
            "new_model_id": manifest.new_model_id,
            "old_profile_fingerprint": manifest.old_profile_fingerprint,
            "new_profile_fingerprint": manifest.new_profile_fingerprint,
        }
    return None


def _read_adapter_manifest(path: Path) -> AdapterManifest | None:
    """Read a ``.rbs`` manifest, without loading the adapter.

    The weights are not wanted: every field this needs is in `manifest.json`,
    and `load_adapter` would map a safetensors file and verify hashes to reach
    the same dictionary.
    """
    import json

    from rebasis.core.serialization import AdapterManifest
    from rebasis.errors import RebasisError

    manifest_file = path / "manifest.json"
    if not manifest_file.is_file():
        return None
    try:
        return AdapterManifest.from_json(json.loads(manifest_file.read_text(encoding="utf-8")))
    except (OSError, TypeError, ValueError, RebasisError):
        return None


def _compare_dimension(adapter: dict[str, Any] | None, dimension: int | None) -> StoreCheck:
    """The one comparison the recorded profile supports against a live index."""
    if adapter is None:
        return StoreCheck(
            "profile",
            ok=None,
            detail=(
                "a probe of this collection is recorded, but no adapter is, and a probe "
                "records the candidate model's profile rather than the index's"
            ),
        )
    if dimension is None:
        return StoreCheck(
            "profile",
            ok=None,
            detail="an adapter is recorded, but the index would not report its dimensionality",
        )
    direction = str(adapter["direction"])
    if direction != "query_to_old":
        return StoreCheck(
            "profile",
            ok=None,
            detail=f"the recorded adapter runs {direction}, which does not map into the index",
        )

    indexed = int(adapter["output_dim"])
    models = f"{adapter['new_model_id']} → {adapter['old_model_id']}"
    if indexed == dimension:
        return StoreCheck(
            "profile",
            ok=True,
            detail=(
                f"the recorded adapter ({models}) maps into {indexed} dimensions, "
                f"and so does this index"
            ),
        )
    return StoreCheck(
        "profile",
        ok=False,
        detail=(
            f"the recorded adapter ({models}) maps into {indexed} dimensions, "
            f"but this index is {dimension}-dimensional"
        ),
        hint=(
            "That adapter was not fitted against this collection. Check the store URI, "
            "or refit against this index with `rebasis fit`."
        ),
    )


def _check_spaces(db: ManifestDB, uri: str) -> tuple[list[MixedSpace], StoreCheck]:
    """Whether this collection is holding two embedding spaces at once.

    The check that earns its place most, and the reason `--store` is worth
    having at all: a half-finished migration answers a share of its queries
    wrongly, raises nothing, and changes no count. `status` reports it and
    `doctor` could not, because `doctor` had no store — and `doctor` is what a
    user runs when retrieval has quietly got worse.

    Run unconditionally, on terms worth stating. `mixed_spaces_for` reads the
    manifest and nothing else: no store is opened, no vector is touched, no lock
    is taken, nothing goes over a network. Its cost is one indexed aggregate per
    unfinished job over that job's own rows in the local queue table — it scales
    with what a migration enqueued, not with the collection, and the index
    `idx_items_pending` covers it. There is no size of corpus at which that
    becomes a surprise.
    """
    from rebasis.migrate import mixed_spaces_for

    try:
        mixed = mixed_spaces_for(db, uri)
    except Exception as exc:  # noqa: BLE001 - an unreadable queue is a finding, not a crash
        return [], StoreCheck("spaces", ok=None, detail=f"could not be determined: {exc}")
    if not mixed:
        return [], StoreCheck(
            "spaces", ok=True, detail="one embedding space: no unfinished migration of this index"
        )
    jobs = (
        "one unfinished migration has"
        if len(mixed) == 1
        else f"{len(mixed)} unfinished migrations have"
    )
    return mixed, StoreCheck(
        "spaces",
        ok=False,
        detail=f"{jobs} left this index holding two embedding spaces",
        hint="Search results are not correct until it finishes or is rolled back.",
    )


# ── rendering ─────────────────────────────────────────────────────────
#
# Everything taken from the store, the manifest or an exception is escaped
# before it goes inside markup. A store URI, a model id, a collection name and
# a SQLite error message all contain square brackets often enough that rich
# would eat part of them as a style tag.


def _print_store(report: StoreReport) -> None:
    """The index half of the report."""
    table = Table(title="store", show_header=False, box=None, title_justify="left")
    table.add_column(style="dim", width=22)
    table.add_column()

    table.add_row("uri", escape(report.uri))
    table.add_row("backend", escape(report.backend) if report.backend else _UNKNOWN)
    table.add_row("records", f"{report.records:,}" if report.records is not None else _UNKNOWN)
    table.add_row("dimension", str(report.dimension) if report.dimension is not None else _UNKNOWN)
    for name, value in _capability_rows(report.capabilities or {}):
        table.add_row(name, value)
    for name, value in _recorded_rows(report.recorded):
        table.add_row(name, value)

    table.add_row("", "")
    for check in report.checks:
        table.add_row(check.name, f"{_VERDICTS[check.ok]} {escape(check.detail)}")
        if check.hint:
            table.add_row("", f"[cyan]{escape(check.hint)}[/cyan]")

    console.print()
    console.print(table)
    _print_mixed_spaces(report)


def _capability_rows(capabilities: dict[str, Any]) -> list[tuple[str, str]]:
    """Render what a backend declares without naming any of it.

    `StoreCapabilities` gains fields, and a renderer with a hand-written line
    per field describes an index against last month's list while saying nothing
    at all about the newest one. Booleans collapse into two lines because that
    is the question being asked — what can this backend do — and anything that
    is not a boolean gets its own line, because there is no way to guess how a
    field this code has never seen ought to be summarised.
    """
    if not capabilities:
        return []
    booleans = {k: v for k, v in capabilities.items() if isinstance(v, bool)}
    supported = ", ".join(sorted(k for k, v in booleans.items() if v))
    rows = [("supports", supported or "[dim]nothing declared[/dim]")]
    withheld = sorted(k for k, v in booleans.items() if not v)
    if withheld:
        rows.append(("does not support", ", ".join(withheld)))
    rows.extend(
        (key, escape(str(value)))
        for key, value in sorted(capabilities.items())
        if key != "name" and not isinstance(value, bool) and value not in (None, "")
    )
    return rows


def _recorded_rows(recorded: dict[str, Any] | None) -> list[tuple[str, str]]:
    """What rebasis has on file for this collection, as fact and nothing more.

    The record count at probe time is printed beside the count now because both
    are facts and the reader is the one who knows whether the difference means
    anything. No verdict is attached to it: a corpus that has grown is the
    normal case, and rebasis has measured nothing that would let it say when a
    change is large enough to matter.
    """
    if recorded is None:
        return []
    rows: list[tuple[str, str]] = []
    probe = recorded.get("probe")
    if isinstance(probe, dict):
        seen = probe.get("records_at_probe")
        counted = f", {seen:,} records then" if isinstance(seen, int) else ""
        models = f"{probe['old_model']} → {probe['new_model']}"
        rows.append(("last probe", escape(f"{probe['ts_utc'][:10]}  {models}{counted}")))
    adapter = recorded.get("adapter")
    if isinstance(adapter, dict):
        rows.append(
            (
                "last adapter",
                escape(
                    f"{adapter['adapter_type']}  {adapter['input_dim']} → "
                    f"{adapter['output_dim']}  ({adapter['direction']})"
                ),
            )
        )
    return rows


def _print_mixed_spaces(report: StoreReport) -> None:
    """Say, unprompted, that this index is holding two embedding spaces.

    The same paragraph `status` prints, from the same `MixedSpace.explain` and
    `next_steps`, because it is read at a third moment those two do not cover:
    by somebody who has noticed retrieval get worse and has run `doctor` to find
    out why.
    """
    for state in report.mixed_spaces:
        console.print()
        console.print(
            "[red bold]This index holds two embedding spaces.[/red bold] "
            "[dim]Search results are not correct until the migration finishes "
            "or is rolled back.[/dim]"
        )
        console.print(f"  {escape(state.explain())}")
        for step in state.next_steps():
            console.print(f"    [dim]{escape(step)}[/dim]")
