"""Reproducing a recorded decision.

``rebasis audit replay <seq>`` re-runs a decision with the inputs the record
kept, and compares. If it differs, either there is a regression or the corpus has
changed — and both are things you want to know.

**The comparison depends on the hardware, and saying so matters:**

* **Same device** — bit-level agreement is expected. A difference is a
  regression.
* **Different device** — an equivalence band is used instead (ARR within 0.002),
  and the report states plainly that the comparison crossed devices. PyTorch says
  outright that CPU and GPU results may differ with identical seeds; presenting a
  cross-device comparison as if it were same-device would lead the reader to
  exactly the wrong conclusion.

The decision itself is what actually has to match. Small numerical drift is
tolerable; a different recommendation is not — the same reasoning that governs
the device-parity tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rebasis.audit.record import EnvironmentSnapshot

if TYPE_CHECKING:
    from rebasis.audit.record import AuditRecord

__all__ = ["ReplayComparison", "ReplayVerdict", "compare_replay", "replayable_inputs"]

#: ARR difference accepted across devices.
CROSS_DEVICE_ARR_TOLERANCE = 0.002

#: Fields a probe decision must carry to be replayable at all.
REQUIRED_INPUTS = ("seed", "count")


class ReplayVerdict:
    """The outcome names, kept stable because scripts branch on them."""

    MATCHED = "matched"
    MATCHED_CROSS_DEVICE = "matched_cross_device"
    DIFFERED = "differed"
    NOT_REPLAYABLE = "not_replayable"


@dataclass(slots=True)
class ReplayComparison:
    """What a replay found."""

    verdict: str
    recorded_decision: str | None = None
    replayed_decision: str | None = None
    recorded_arr: float | None = None
    replayed_arr: float | None = None
    same_device: bool = True
    recorded_device: str = ""
    current_device: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def matched(self) -> bool:
        """Whether the replay agrees with the record."""
        return self.verdict in {ReplayVerdict.MATCHED, ReplayVerdict.MATCHED_CROSS_DEVICE}

    def describe(self) -> str:
        """A one-line summary for the CLI."""
        if self.verdict == ReplayVerdict.NOT_REPLAYABLE:
            return "not replayable: " + (self.notes[0] if self.notes else "insufficient inputs")
        if self.verdict == ReplayVerdict.MATCHED:
            return f"matched ({self.recorded_decision}, ARR {self.recorded_arr:.4f})"
        if self.verdict == ReplayVerdict.MATCHED_CROSS_DEVICE:
            return (
                f"matched within the cross-device band "
                f"({self.recorded_device} → {self.current_device})"
            )
        return (
            f"DIFFERED: recorded {self.recorded_decision} "
            f"(ARR {self.recorded_arr}), replayed {self.replayed_decision} "
            f"(ARR {self.replayed_arr})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "verdict": self.verdict,
            "recorded_decision": self.recorded_decision,
            "replayed_decision": self.replayed_decision,
            "recorded_arr": self.recorded_arr,
            "replayed_arr": self.replayed_arr,
            "same_device": self.same_device,
            "notes": list(self.notes),
        }


def replayable_inputs(record: AuditRecord) -> tuple[bool, str]:
    """Whether a record carries enough to be replayed.

    Returns ``(replayable, reason)``. A record that cannot be replayed is not a
    failure — most audited events are not decisions — but the user should be told
    which it is rather than shown a confusing null result.
    """
    if record.action != "probe.decision.made":
        return False, f"{record.action} is not a decision; only probe decisions replay"
    missing = [field for field in REQUIRED_INPUTS if field not in record.inputs]
    if missing:
        return False, f"the record is missing {', '.join(missing)}"
    return True, ""


def compare_replay(
    record: AuditRecord,
    *,
    replayed_decision: str,
    replayed_arr: float,
    current_env: dict[str, Any] | None = None,
) -> ReplayComparison:
    """Compare a fresh result against a recorded one.

    Args:
        record: The recorded decision.
        replayed_decision: The decision the replay produced.
        replayed_arr: The ARR the replay produced.
        current_env: The current environment; captured when omitted.
    """
    replayable, reason = replayable_inputs(record)
    if not replayable:
        return ReplayComparison(verdict=ReplayVerdict.NOT_REPLAYABLE, notes=[reason])

    environment = current_env or EnvironmentSnapshot.capture().to_dict()
    recorded_device = str(record.env.get("device", "unknown"))
    current_device = str(environment.get("device", "unknown"))
    same_device = recorded_device == current_device

    recorded_decision = record.outputs.get("decision")
    recorded_arr = record.outputs.get("arr_r10")

    notes: list[str] = []
    if not same_device:
        notes.append(
            f"This comparison crossed devices ({recorded_device} → {current_device}). "
            "Identical seeds do not guarantee identical results between CPU and "
            "GPU, so an equivalence band is used rather than exact agreement."
        )
    if record.env.get("tf32_matmul") and not environment.get("tf32_matmul"):
        notes.append(
            "The recorded run had TF32 enabled and this one does not, which alone "
            "can move the numbers."
        )

    comparison = ReplayComparison(
        recorded_decision=str(recorded_decision) if recorded_decision else None,
        replayed_decision=replayed_decision,
        recorded_arr=float(recorded_arr) if recorded_arr is not None else None,
        replayed_arr=replayed_arr,
        same_device=same_device,
        recorded_device=recorded_device,
        current_device=current_device,
        notes=notes,
        verdict=ReplayVerdict.DIFFERED,
    )

    # The decision is what has to match. Numerical drift is tolerable; a
    # different recommendation is not.
    if recorded_decision != replayed_decision:
        comparison.notes.append(
            "The recommendation changed. Either rebasis regressed, or the corpus "
            "is no longer the one that was measured — both worth investigating."
        )
        return comparison

    if same_device:
        comparison.verdict = ReplayVerdict.MATCHED
        return comparison

    drift = abs((comparison.recorded_arr or 0.0) - replayed_arr)
    if drift <= CROSS_DEVICE_ARR_TOLERANCE:
        comparison.verdict = ReplayVerdict.MATCHED_CROSS_DEVICE
    else:
        comparison.notes.append(
            f"The decision agrees, but ARR moved by {drift:.4f}, beyond the "
            f"{CROSS_DEVICE_ARR_TOLERANCE} cross-device band."
        )
    return comparison
