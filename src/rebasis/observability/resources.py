"""Resource accounting for a run.

Every ``probe`` and ``migrate`` ends with a summary that goes into both the
report and the audit record: wall time, CPU time, peak RSS, batch size, throttle
events, BLAS threads, device, and energy where it can be measured.

Two reasons this exists, and the second is the one that pays.

It **calibrates the tool's own estimates.** ``cost_full_reindex`` predicts how
long a reindex would take; the only honest way to improve that prediction is to
compare it against runs that actually happened.

It **makes "it got slower" a fact.** A user who can say "last month this took
four minutes, now it takes eleven" can be helped. Without a recorded number that
sentence is a feeling, and a feeling cannot be diagnosed.

**Energy is measured or absent, never estimated.** A fabricated watt-hour figure
would be worse than none: it reads as a measurement and answers a question
nobody actually asked the hardware.
"""

from __future__ import annotations

import contextlib
import io
import os
import resource
import time
from dataclasses import dataclass, field
from typing import Any, Self

__all__ = ["ResourceSummary", "measure_resources", "peak_rss_bytes"]


def peak_rss_bytes() -> int:
    """Peak resident set size for this process, in bytes.

    ``ru_maxrss`` is kilobytes on Linux and bytes on macOS — a difference that
    silently produces a 1024x error in one direction or the other, so it is
    handled here rather than at three call sites.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) if os.uname().sysname == "Darwin" else int(peak) * 1024


def cpu_seconds() -> float:
    """User plus system CPU time, this process and its children.

    Reported alongside wall time because their ratio is the useful signal: a
    run at 4x wall clock is using the cores, a run at 1x is waiting on
    something.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(usage.ru_utime + usage.ru_stime + children.ru_utime + children.ru_stime)


@dataclass(slots=True)
class ResourceSummary:
    """What a run cost.

    Field names are part of the audit schema, so they are stable.
    """

    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    peak_rss_bytes: int = 0
    device: str = "cpu"
    blas_threads: int | None = None
    batch_size_final: int | None = None
    throttle_events: int = 0
    #: Watt-hours, when the hardware could be asked. ``None`` means it could
    #: not; it never means zero and it is never estimated.
    energy_wh: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form, for the report and the audit record."""
        payload: dict[str, Any] = {
            "wall_seconds": round(self.wall_seconds, 2),
            "cpu_seconds": round(self.cpu_seconds, 2),
            "peak_rss_bytes": self.peak_rss_bytes,
            "device": self.device,
            "throttle_events": self.throttle_events,
            "energy_wh": None if self.energy_wh is None else round(self.energy_wh, 3),
        }
        if self.blas_threads is not None:
            payload["blas_threads"] = self.blas_threads
        if self.batch_size_final is not None:
            payload["batch_size_final"] = self.batch_size_final
        return {**payload, **self.extras}

    @property
    def parallelism(self) -> float:
        """CPU seconds per wall second — how much of the machine was used."""
        return self.cpu_seconds / self.wall_seconds if self.wall_seconds > 0 else 0.0


class measure_resources:  # noqa: N801 - a context manager used as a verb reads better
    """Context manager that fills in a :class:`ResourceSummary`.

    ::

        with measure_resources(device="cuda") as usage:
            ...
        record["resources"] = usage.summary.to_dict()

    ``device`` and ``blas_threads`` are supplied by the caller rather than
    detected here. Both are compute-layer facts, and the layer contract puts
    ``observability`` below ``compute`` — reaching up for them would invert that
    contract and, in this case, put torch on the import path of the logging
    package.

    Peak RSS is process-wide rather than block-scoped: ``ru_maxrss`` is a
    high-water mark that never falls, so a second measurement in one process
    inherits the first's peak. That is stated rather than worked around,
    because the alternative — sampling RSS on a timer — costs a thread and
    still misses a spike between samples.
    """

    def __init__(self, *, device: str = "cpu", blas_threads: int | None = None) -> None:
        self.summary = ResourceSummary(device=device, blas_threads=blas_threads)
        self._started = 0.0
        self._cpu_started = 0.0
        self._energy: Any = None

    def __enter__(self) -> Self:
        self._started = time.perf_counter()
        self._cpu_started = cpu_seconds()
        self._energy = _start_energy_measurement()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.summary.wall_seconds = time.perf_counter() - self._started
        self.summary.cpu_seconds = cpu_seconds() - self._cpu_started
        self.summary.peak_rss_bytes = peak_rss_bytes()
        self.summary.energy_wh = _stop_energy_measurement(self._energy)


def _start_energy_measurement() -> Any:
    """Begin an energy window, if the hardware exposes one.

    Zeus is the measurement path: it reads the GPU's own energy counters.
    Absent it, this returns ``None`` and the summary reports no energy — which
    is the honest answer, not zero.
    """
    # Zeus narrates its own startup, and not only through logging: probing for
    # an AMD SMI library prints the dlopen failure straight to stdout. That
    # landed in front of `probe --json`, so `rebasis probe --json | jq` received
    # "/opt/rocm/lib/libamd_smi.so: cannot open shared object file" and then the
    # JSON. Silencing the loggers was never enough on its own; stdout has to be
    # taken away from it for the whole window, import included.
    #
    # Discarded rather than moved to stderr: this package already decided this
    # narration is unwanted, and a user running `probe` did not ask for a report
    # on which SMI libraries are present.
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            from zeus.monitor import ZeusMonitor
        except ImportError:
            return None

        import logging

        for name in ("zeus", "zeus.device", "zeus.monitor", "zeus.utils"):
            logging.getLogger(name).setLevel(logging.ERROR)

        try:
            monitor = ZeusMonitor()
            monitor.begin_window("rebasis")
        except Exception:  # noqa: BLE001 - no GPU, no permissions, no driver
            return None
    return monitor


def _stop_energy_measurement(monitor: Any) -> float | None:
    """Close the window and return watt-hours, or ``None``.

    A window shorter than the GPU's energy-counter update period reads as
    exactly zero. That is the counter saying "I could not measure this", not
    the hardware saying it used no power — so it becomes ``None``.

    Zeus offers ``approx_instant_energy`` for this case, which multiplies
    instantaneous power by the window length. That is declined: it would produce
    a number indistinguishable from a measurement, for a quantity nothing
    actually measured.
    """
    import warnings

    if monitor is None:
        return None
    try:
        # Same guard as the start: closing the window can narrate too, and
        # anything it prints would land in the middle of `--json` output.
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            # The zero-energy warning is Zeus recommending the approximation
            # we have deliberately declined. The zero itself is handled below.
            warnings.simplefilter("ignore", UserWarning)
            measurement = monitor.end_window("rebasis")
    except Exception:  # noqa: BLE001 - a failed read is not a failed run
        return None

    joules = getattr(measurement, "total_energy", None)
    if joules is None or float(joules) <= 0.0:
        return None
    # Zeus reports joules; 1 Wh is 3600 J.
    return float(joules) / 3600.0
