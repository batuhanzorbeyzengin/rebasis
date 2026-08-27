"""Where the accelerator pays, and where it does not.

The open question was where the size threshold sits above which kNN should move
to the GPU. **M0 measured it and there is no threshold** — see section 8 of
`docs/m0-findings.md`. On an A10G against a 4-vCPU host, chunked top-k was
faster on the accelerator at every size tested, including 2,000 documents where
the reference work predicted the CPU would win, and by 22x once transfer was
counted.

So this module does not hold a crossover point. It holds the shape of the
answer that was actually measured: which operations are worth moving, with the
speedups behind each, so a caller can decide without re-deriving it.

The one honest caveat is that these are ratios between *this* GPU and *that*
CPU. A faster host narrows every one of them, which is why a local calibration
can override the table — ``rebasis doctor --calibrate`` measures the machine in
front of it and writes the result down, and :func:`load_calibration` reads it
back.

**What that calibration is, and is not.** It is a diagnostic. Nothing in the
runtime dispatches per operation: `probe` and `fit` run under one ambient device
for the whole session and :func:`~rebasis.compute.search.top_k_search` uses it
without consulting a size threshold. So a calibration changes what ``doctor``
reports about this machine; it does not change where work runs. Saying otherwise
would be the more flattering claim and the false one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

__all__ = [
    "MEASURED_SPEEDUPS",
    "Calibration",
    "calibration_path",
    "load_calibration",
    "worth_accelerating",
]

#: Measured on AWS g5.xlarge (A10G, 4 vCPU) during M0. Ratios of CPU time to
#: accelerator time, including host-to-device transfer.
MEASURED_SPEEDUPS: Final[dict[str, float]] = {
    # 80-90% of a probe's wall clock. The one that matters.
    "embed": 25.4,
    "embed_large": 40.0,
    # Real but modest; the fit is already in BLAS.
    "fit_mlp": 5.9,
    "fit_linear": 1.0,
    # Expected to be borderline at our sample sizes. It is not.
    "knn": 22.0,
}

#: Below this ratio, moving an operation to the accelerator is not worth the
#: transfer and the complexity. Set at 2x rather than 1x because a marginal
#: win on one machine is a loss on another.
WORTH_IT = 2.0


@dataclass(frozen=True, slots=True)
class Calibration:
    """Speedups measured on *this* machine.

    Written by ``rebasis doctor --calibrate``, which times the operations it can
    reach without downloading anything and records **only those**. A key that is
    absent was not measured, and :func:`worth_accelerating` falls back to
    :data:`MEASURED_SPEEDUPS` for it rather than reading the absence as a zero.
    That is the same rule the rest of the project applies to energy and to the
    reindex estimate: measured or omitted, never estimated.

    ``notes`` records what each measurement was taken on — sizes, repeats, the
    dimensionality — because a ratio without its configuration is not
    reproducible and this file outlives the terminal it was printed in.
    """

    device: str
    speedups: dict[str, float]
    measured_utc: str = ""
    host: str = ""
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "device": self.device,
            "speedups": self.speedups,
            "measured_utc": self.measured_utc,
            "host": self.host,
            "notes": self.notes,
        }


def worth_accelerating(operation: str, calibration: Calibration | None = None) -> bool:
    """Whether ``operation`` is worth moving off the CPU.

    Uses a local calibration where it has an entry, falling back to the M0
    measurements where it does not. **Per key, not per table**, and the
    distinction is the whole safety of a partial calibration: `doctor
    --calibrate` measures what it can reach without downloading a model, which
    is not every key here, and replacing the whole table with a partial one
    would read a missing ``embed`` as *not worth accelerating* — turning off the
    one operation that dominates a `probe`, on a machine that had just been
    measured and found fast.

    Unknown operations return ``False``: the burden is on an operation to show
    it benefits, not on the CPU to defend itself.
    """
    table = (
        MEASURED_SPEEDUPS if calibration is None else {**MEASURED_SPEEDUPS, **calibration.speedups}
    )
    return table.get(operation, 0.0) >= WORTH_IT


def calibration_path(state_dir: Path | str) -> Path:
    """Where a local calibration is kept."""
    return Path(state_dir) / "calibration.json"


def load_calibration(state_dir: Path | str) -> Calibration | None:
    """Read a local calibration, or ``None`` when there is not one.

    A malformed file is treated as absent rather than fatal: it is a cache of a
    measurement, and the measurement can be taken again.
    """
    path = calibration_path(state_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return Calibration(
            device=str(payload["device"]),
            speedups={str(k): float(v) for k, v in payload["speedups"].items()},
            measured_utc=str(payload.get("measured_utc", "")),
            host=str(payload.get("host", "")),
            notes=dict(payload.get("notes", {})),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


# Writing a calibration lives in the CLI, not here. The layer contract makes
# `compute` and `storage` siblings, so reaching for the atomic writer would
# cross a layer; and `doctor` is the only thing that produces one anyway.
# Reading is a plain parse and stays where the readers are.
