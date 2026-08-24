"""Power and resource awareness.

The target user is migrating their own vault on their own laptop. A tool that
flattens the battery or makes the machine unusable will simply be turned off —
so ``--power-aware`` is not a nicety, it is what makes the tool usable at all.
Being laptop-friendly — running on CPU, pausable, resumable, power-aware — is a
goal of the project in its own right.

``psutil.sensors_battery()`` answers "is it plugged in" on every platform, and
that single question covers most of what power awareness needs. Zeus is an
optional extra for actual watt-hours.

The memory watchdog: at 90% of the ceiling the batch halves and emits
``migrate.batch.throttled``; at 100% the job **pauses rather than dying**,
because a checkpointed job can always be resumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["PowerState", "ResourceMonitor", "power_state"]

#: Battery percentage below which an unplugged migration pauses.
LOW_BATTERY_PERCENT = 20.0

#: Fraction of the memory ceiling at which the batch size halves.
THROTTLE_AT = 0.90

#: Fraction at which the job pauses. Pauses, never dies — checkpointing is what
#: makes that the right response.
PAUSE_AT = 1.00


@dataclass(frozen=True, slots=True)
class PowerState:
    """Whether it is currently reasonable to keep working."""

    on_battery: bool
    battery_percent: float | None
    should_pause: bool
    reason: str = ""


def power_state(*, power_aware: bool = True) -> PowerState:
    """Read the current power situation.

    Args:
        power_aware: When ``False`` this always reports "keep going" — a server
            has no battery and should not inherit laptop behaviour.
    """
    if not power_aware:
        return PowerState(on_battery=False, battery_percent=None, should_pause=False)

    try:
        import psutil

        battery = psutil.sensors_battery()
    except Exception:  # noqa: BLE001 - absence of a battery is not an error
        battery = None

    if battery is None:
        # Desktop, server, or a platform that does not report. Not a reason to
        # stop.
        return PowerState(on_battery=False, battery_percent=None, should_pause=False)

    on_battery = not battery.power_plugged
    percent = float(battery.percent)
    if on_battery and percent <= LOW_BATTERY_PERCENT:
        return PowerState(
            on_battery=True,
            battery_percent=percent,
            should_pause=True,
            reason=(
                f"running on battery at {percent:.0f}%. The job is checkpointed; "
                "plug in and resume."
            ),
        )
    return PowerState(on_battery=on_battery, battery_percent=percent, should_pause=False)


class ResourceMonitor:
    """Watches memory and adapts the batch size.

    Telling a user to "reduce the batch size" is a poor interface — the tool
    should work it out. The batch is derived from the ceiling to begin with, and
    adjusted from measured RSS as the job runs.
    """

    def __init__(
        self,
        *,
        max_memory_bytes: int | None = None,
        dim: int = 768,
        initial_batch: int | None = None,
        min_batch: int = 32,
    ) -> None:
        self.max_memory_bytes = max_memory_bytes
        self.dim = dim
        self.min_batch = min_batch
        self.batch_size = initial_batch or self.suggested_batch()
        self.throttle_events = 0
        self.peak_rss = 0

    def suggested_batch(self, *, safety_factor: float = 4.0) -> int:
        """Batch size derived from the memory ceiling.

        ``batch = max(32, floor(budget / (d × 4 × safety)))``. The safety factor
        accounts for the copies a batch actually makes: input, mapped output and
        whatever the embedding backend holds.
        """
        if not self.max_memory_bytes:
            return 256
        per_record = self.dim * 4 * safety_factor
        return max(self.min_batch, int(self.max_memory_bytes / per_record))

    def current_rss(self) -> int:
        """Resident memory of this process, in bytes."""
        try:
            import psutil

            rss = int(psutil.Process().memory_info().rss)
        except Exception:  # noqa: BLE001 - measurement must never break a job
            return 0
        self.peak_rss = max(self.peak_rss, rss)
        return rss

    def check(self) -> tuple[bool, str]:
        """Decide after a batch whether to keep going.

        Returns ``(should_pause, reason)``. Halving the batch happens here as a
        side effect, because that is the cheaper remedy and it is tried first.
        """
        if not self.max_memory_bytes:
            self.current_rss()
            return False, ""

        rss = self.current_rss()
        ratio = rss / self.max_memory_bytes

        if ratio >= PAUSE_AT:
            return True, (
                f"memory reached {rss / 1024**3:.2f} GB against a "
                f"{self.max_memory_bytes / 1024**3:.2f} GB ceiling. The job is "
                "checkpointed; raise --max-memory and resume."
            )

        if ratio >= THROTTLE_AT and self.batch_size > self.min_batch:
            self.batch_size = max(self.min_batch, self.batch_size // 2)
            self.throttle_events += 1
            return False, f"batch halved to {self.batch_size}"

        return False, ""

    def summary(self) -> dict[str, Any]:
        """The resource summary written to the report and the audit record."""
        return {
            "peak_rss_bytes": self.peak_rss,
            "batch_size_final": self.batch_size,
            "throttle_events": self.throttle_events,
        }
