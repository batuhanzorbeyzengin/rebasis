"""Replay comparison.

Replay is idempotent: the same corpus and seed must produce the same decision.
What is tested here is the **comparison logic** — including the part that
matters most, that a cross-device comparison is reported as such rather than
presented as if it were exact.
"""

from __future__ import annotations

from typing import Any

import pytest

from rebasis.audit import ReplayVerdict, compare_replay, replayable_inputs
from rebasis.audit.record import AuditRecord
from rebasis.audit.replay import CROSS_DEVICE_ARR_TOLERANCE


def _decision_record(
    *, decision: str = "bridge_sufficient", arr: float = 0.96, device: str = "cpu"
) -> AuditRecord:
    return AuditRecord(
        ts_utc="2026-08-23T00:00:00+00:00",
        run_id="01JTEST",
        actor="cli",
        action="probe.decision.made",
        inputs={"seed": 0, "count": 10_000, "strategy": "stratified"},
        outputs={"decision": decision, "arr_r10": arr},
        rebasis_version="0.1.0",
        env={"device": device, "python": "3.12.3", "tf32_matmul": False},
        hash="x" * 64,
    )


@pytest.mark.unit
def test_same_device_agreement_matches() -> None:
    """The straightforward case: same machine, same answer."""
    result = compare_replay(
        _decision_record(),
        replayed_decision="bridge_sufficient",
        replayed_arr=0.96,
        current_env={"device": "cpu"},
    )
    assert result.verdict == ReplayVerdict.MATCHED
    assert result.matched


@pytest.mark.unit
def test_a_changed_decision_is_reported_however_close_the_numbers() -> None:
    """Numerical drift is tolerable; a different recommendation is not.

    A user whose corpus now says "reindex" where it once said "bridge" needs to
    know that, even if ARR barely moved.
    """
    result = compare_replay(
        _decision_record(decision="bridge_sufficient", arr=0.951),
        replayed_decision="bridge_and_migrate",
        replayed_arr=0.949,
        current_env={"device": "cpu"},
    )
    assert result.verdict == ReplayVerdict.DIFFERED
    assert not result.matched
    assert any("recommendation changed" in n for n in result.notes)


@pytest.mark.unit
def test_cross_device_agreement_is_labelled_as_such() -> None:
    """Presenting a cross-device comparison as exact misleads the reader.

    PyTorch states that identical seeds do not guarantee identical results
    between CPU and GPU, so the comparison band and the fact of it are both
    reported.
    """
    result = compare_replay(
        _decision_record(device="cpu", arr=0.960),
        replayed_decision="bridge_sufficient",
        replayed_arr=0.9605,
        current_env={"device": "cuda:0"},
    )
    assert result.verdict == ReplayVerdict.MATCHED_CROSS_DEVICE
    assert result.matched
    assert not result.same_device
    assert any("crossed devices" in n for n in result.notes)


@pytest.mark.unit
def test_cross_device_drift_beyond_the_band_is_flagged() -> None:
    """Agreeing on the decision is not enough if the numbers moved too far."""
    result = compare_replay(
        _decision_record(arr=0.960),
        replayed_decision="bridge_sufficient",
        replayed_arr=0.960 + CROSS_DEVICE_ARR_TOLERANCE * 5,
        current_env={"device": "mps"},
    )
    assert result.verdict == ReplayVerdict.DIFFERED
    assert any("beyond the" in n for n in result.notes)


@pytest.mark.unit
def test_a_tf32_difference_is_called_out() -> None:
    """TF32 alone can move the numbers, so it is named explicitly.

    M0 measured the size of that effect: with TF32 on, a 4096³ matmul deviated
    from CPU by 0.106 against 0.001 with it off.
    """
    record = _decision_record()
    record.env["tf32_matmul"] = True  # type: ignore[index]

    result = compare_replay(
        record,
        replayed_decision="bridge_sufficient",
        replayed_arr=0.96,
        current_env={"device": "cpu", "tf32_matmul": False},
    )
    assert any("TF32" in n for n in result.notes)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("action", "inputs", "expected"),
    [
        ("probe.decision.made", {"seed": 0, "count": 100}, True),
        ("probe.decision.made", {"seed": 0}, False),
        ("migrate.job.started", {"seed": 0, "count": 100}, False),
        ("store.write.performed", {}, False),
    ],
)
def test_only_complete_decisions_are_replayable(
    action: str, inputs: dict[str, Any], expected: bool
) -> None:
    """A record that cannot be replayed says which it is, rather than failing oddly."""
    record = AuditRecord(
        ts_utc="2026-08-23T00:00:00+00:00",
        run_id="r",
        actor="cli",
        action=action,
        inputs=inputs,
        outputs={},
        rebasis_version="0.1.0",
        env={},
    )
    replayable, reason = replayable_inputs(record)
    assert replayable is expected
    if not expected:
        assert reason


@pytest.mark.unit
def test_a_non_replayable_record_produces_a_clear_verdict() -> None:
    """Most audited events are not decisions; that is not an error."""
    record = AuditRecord(
        ts_utc="2026-08-23T00:00:00+00:00",
        run_id="r",
        actor="cli",
        action="store.write.performed",
        inputs={},
        outputs={"count": 500},
        rebasis_version="0.1.0",
        env={},
    )
    result = compare_replay(record, replayed_decision="", replayed_arr=0.0)
    assert result.verdict == ReplayVerdict.NOT_REPLAYABLE
    assert "not a decision" in result.describe()
