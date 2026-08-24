"""Space and time budgeting before a migration starts.

**A migration that fills the disk halfway through is not an error to handle but a
design flaw to prevent.** Record count, target dimension, shadow copy size,
checkpoint size and the store's own growth are estimated up front, shown to the
user, and checked with a 20% safety margin before the first record moves. Too
little room and the job never starts.

**No intermediate vector file is written.** ``migrate`` goes embed → upsert
directly rather than writing ``N × d × 4`` bytes and reading them back. The only
persistent artefact is the checkpoint, which is a list of ids and states — a few
megabytes.

Estimates improve over time: each run records what it actually cost, and the next
pre-check calibrates against that instead of guessing again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rebasis.storage.atomic import free_bytes, require_free_space

__all__ = ["MigrationBudget", "enforce_budget", "estimate_budget"]

#: Margin on the space requirement.
SAFETY_MARGIN = 0.20

#: Bytes of checkpoint state per record — an id, a state and a priority.
CHECKPOINT_BYTES_PER_RECORD = 24

#: Fallback throughput when nothing has been measured yet, in records per second.
#: Deliberately pessimistic: an estimate that turns out short is a pleasant
#: surprise, while one that turns out long erodes trust in every later estimate.
FALLBACK_RECORDS_PER_SECOND = 40.0

#: Fallback energy per record, in watt-hours. Only used when Zeus is unavailable.
#: A *reported* energy figure is measured or left null, never guessed, so this
#: feeds the pre-check estimate and nothing else.
FALLBACK_WH_PER_RECORD = 1e-4


@dataclass(slots=True)
class MigrationBudget:
    """What a migration will cost, estimated before it starts."""

    record_count: int
    dim: int
    keep_original: bool
    shadow_bytes: int
    checkpoint_bytes: int
    required_bytes: int
    available_bytes: int
    estimated_seconds: float
    estimated_wh: float | None
    warnings: list[str] = field(default_factory=list)

    @property
    def fits(self) -> bool:
        """Whether there is enough free space."""
        return self.available_bytes >= self.required_bytes

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form, for the report and the audit record."""
        return {
            "count": self.record_count,
            "dim": self.dim,
            "keep_original": self.keep_original,
            "shadow_bytes": self.shadow_bytes,
            "checkpoint_bytes": self.checkpoint_bytes,
            "required_bytes": self.required_bytes,
            "available_bytes": self.available_bytes,
            "estimated_seconds": round(self.estimated_seconds, 1),
            "estimated_wh": round(self.estimated_wh, 2) if self.estimated_wh else None,
            "fits": self.fits,
            "warnings": list(self.warnings),
        }

    def render(self) -> str:
        """The human-readable plan shown before a migration."""
        gib = 1024**3
        lines = [
            "Migration plan",
            f"  Records                 {self.record_count:,}",
            f"  Target dimension        {self.dim} (float32)",
        ]
        if self.keep_original:
            lines.append(
                f"  Shadow copy             {self.shadow_bytes / gib:.2f} GB   "
                "(for rollback; --no-keep-original skips it)"
            )
        else:
            lines.append(
                "  Shadow copy             disabled — this migration cannot be rolled back"
            )
        lines.extend(
            [
                f"  Checkpoint and state    ~{self.checkpoint_bytes / 1024**2:.0f} MB",
                "  " + "─" * 33,
                (
                    f"  Free space needed       {self.required_bytes / gib:.2f} GB   "
                    f"({SAFETY_MARGIN:.0%} margin included)"
                ),
                (
                    f"  Free space available    {self.available_bytes / gib:.2f} GB   "
                    f"{'OK' if self.fits else 'NOT ENOUGH'}"
                ),
                (
                    f"  Estimated time          ~{_duration(self.estimated_seconds)}   "
                    "(depends on the embedding backend)"
                ),
            ]
        )
        if self.estimated_wh is not None:
            lines.append(f"  Estimated energy        ~{self.estimated_wh:.0f} Wh")
        lines.extend(f"  ! {warning}" for warning in self.warnings)
        return "\n".join(lines)


def estimate_budget(  # noqa: PLR0913 - one argument per component of the estimate
    *,
    record_count: int,
    dim: int,
    state_dir: Path | str,
    keep_original: bool = True,
    shadow_precision: str = "float32",
    records_per_second: float | None = None,
    wh_per_record: float | None = None,
    store_growth_bytes: int = 0,
) -> MigrationBudget:
    """Estimate the cost of a migration.

    Args:
        record_count: How many records will move.
        dim: Target dimensionality.
        state_dir: Where rebasis writes, used to measure free space.
        keep_original: Whether a shadow copy will be kept.
        shadow_precision: ``float32`` for bit-identical rollback, ``float16`` for
            half the space and an approximate one.
        records_per_second: Measured throughput, when a previous run recorded it.
        wh_per_record: Measured energy per record, when Zeus was available.
        store_growth_bytes: Extra space the store itself needs. Non-zero for
            backends that cannot update in place and must build a new collection
            — the user should not discover that halfway through.
    """
    from rebasis.storage.shadow import ShadowStore

    shadow_bytes = (
        ShadowStore.estimate_bytes(record_count, dim, precision=shadow_precision)
        if keep_original
        else 0
    )
    checkpoint_bytes = record_count * CHECKPOINT_BYTES_PER_RECORD
    raw_total = shadow_bytes + checkpoint_bytes + store_growth_bytes
    required = int(raw_total * (1 + SAFETY_MARGIN))
    available = free_bytes(Path(state_dir))

    throughput = records_per_second or FALLBACK_RECORDS_PER_SECOND
    seconds = record_count / throughput if throughput > 0 else 0.0

    energy = None
    if wh_per_record is not None:
        energy = record_count * wh_per_record
    elif record_count:
        energy = record_count * FALLBACK_WH_PER_RECORD

    warnings: list[str] = []
    if not keep_original:
        warnings.append(
            "Rollback is disabled for this job. If the result is not what you "
            "expected, the original vectors cannot be restored."
        )
    if shadow_precision == "float16" and keep_original:
        warnings.append(
            "float16 shadow halves the space but makes rollback approximate "
            "rather than bit-identical."
        )
    if store_growth_bytes:
        warnings.append(
            "This backend cannot update in place, so the collection is rebuilt "
            "and both copies exist for a while. That space is included above."
        )

    return MigrationBudget(
        record_count=record_count,
        dim=dim,
        keep_original=keep_original,
        shadow_bytes=shadow_bytes,
        checkpoint_bytes=checkpoint_bytes,
        required_bytes=required,
        available_bytes=available,
        estimated_seconds=seconds,
        estimated_wh=energy,
        warnings=warnings,
    )


def enforce_budget(budget: MigrationBudget, state_dir: Path | str) -> None:
    """Refuse to start when the space is not there.

    Raises:
        InsufficientDiskSpace: With the shortfall and what can be freed.
    """
    require_free_space(
        Path(state_dir),
        budget.shadow_bytes + budget.checkpoint_bytes,
        safety_margin=SAFETY_MARGIN,
    )


_MINUTE = 60
_HOUR = 3600


def _duration(seconds: float) -> str:
    if seconds < _MINUTE:
        return f"{seconds:.0f}s"
    if seconds < _HOUR:
        return f"{seconds / _MINUTE:.0f}m"
    hours, remainder = divmod(seconds, _HOUR)
    return f"{hours:.0f}h {remainder / _MINUTE:.0f}m"
