"""The decision rule, including the amendments M0 forced.

`probe`'s output is a recommendation, not a number, and it has to stay
defensible months later. These tests pin the bands, the uncertainty handling
and the three changes measurement made to the original rule.
"""

from __future__ import annotations

import pytest

from rebasis.probe.decision import (
    BORDERLINE_BAND,
    SCORE_SHIFT_LIMIT,
    UPGRADE_GAIN_FLOOR,
    decide,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        (0.99, "bridge_sufficient"),
        (0.95, "bridge_sufficient"),
        (0.90, "bridge_and_migrate"),
        (0.85, "bridge_and_migrate"),
        (0.75, "caution"),
        (0.70, "caution"),
        (0.50, "full_reindex"),
        (0.10, "full_reindex"),
    ],
)
def test_bands(arr: float, expected: str) -> None:
    """The decision table, boundaries included."""
    assert decide(arr).decision == expected


@pytest.mark.unit
def test_upgrade_gain_overrides_every_band() -> None:
    """The fifth outcome, added after M0 (`docs/m0-findings.md`, section 12).

    Across four corpora, staying on the old model retained a mean ARR of 0.983
    and sometimes beat the new model outright. If the new model is not better on
    this corpus, how well an adapter bridges to it is beside the point — so this
    check comes first.
    """
    result = decide(0.30, upgrade_gain=1.00)
    assert result.decision == "no_upgrade_needed"
    assert "does not retrieve measurably better" in result.rationale


@pytest.mark.unit
def test_a_real_upgrade_gain_does_not_override() -> None:
    """When the new model genuinely is better, the ARR bands take over again."""
    assert decide(0.97, upgrade_gain=1.30).decision == "bridge_sufficient"


@pytest.mark.unit
def test_borderline_band_matches_measured_uncertainty() -> None:
    """The original ±0.005 band was narrower than the measured ±0.024.

    Claiming a ±0.005 resolution would assert a precision the sample cannot
    support, so the band was widened to ±0.025 (`docs/m0-findings.md`,
    section 3.4).
    """
    assert BORDERLINE_BAND >= 0.02, "narrower than the measured confidence interval"
    assert decide(0.955).borderline
    assert decide(0.94).borderline
    assert not decide(0.99).borderline


@pytest.mark.unit
def test_interval_straddling_a_threshold_is_reported() -> None:
    """A point estimate invites over-reading; the interval is what decides."""
    result = decide(0.94, confidence_interval=(0.91, 0.97))
    assert result.interval_straddles_threshold
    assert any("not resolved by the current sample" in w for w in result.warnings)


@pytest.mark.unit
def test_a_tight_interval_away_from_a_threshold_is_quiet() -> None:
    """No warning when the measurement genuinely settles the question.

    ``upgrade_gain`` is supplied so this isolates the interval. Without it the
    result is provisional and warns about *that* instead, which is a different
    question and has its own test.
    """
    result = decide(0.99, confidence_interval=(0.985, 0.995), upgrade_gain=1.3, old_model_arr=0.70)
    assert not result.interval_straddles_threshold
    assert not result.warnings


@pytest.mark.unit
def test_score_shift_warning_is_separate_from_the_band() -> None:
    """Score shift warns independently of the decision.

    Ranking can be perfect while every fixed similarity threshold breaks, so
    this cannot be folded into ARR.
    """
    result = decide(0.98, score_shift=SCORE_SHIFT_LIMIT + 0.2)
    assert result.decision == "bridge_sufficient"
    assert any("fixed similarity threshold" in w for w in result.warnings)


@pytest.mark.unit
def test_heterogeneous_drift_is_surfaced() -> None:
    """A healthy average can hide a badly served subset.

    One global adapter measured 0.85 on a corpus where two domain-specific ones
    reached 0.94 — invisible unless the tail is reported.
    """
    result = decide(0.92, tail_arr=0.60)
    assert any("heterogeneous" in w for w in result.warnings)


@pytest.mark.unit
def test_every_decision_carries_a_rationale() -> None:
    """The decision must be defensible later, so it explains itself."""
    for arr in (0.99, 0.90, 0.75, 0.40):
        assert decide(arr).rationale.strip()


@pytest.mark.unit
def test_result_serialises_for_the_audit_record() -> None:
    """Everything the audit trail needs must survive `to_dict`."""
    payload = decide(
        0.93, confidence_interval=(0.90, 0.96), upgrade_gain=1.4, score_shift=0.05
    ).to_dict()
    for key in ("decision", "arr_r10", "borderline", "confidence_interval", "rationale"):
        assert key in payload


@pytest.mark.unit
def test_upgrade_gain_floor_is_above_one() -> None:
    """A gain of exactly 1.0 is no gain; the floor has to leave room for noise."""
    assert UPGRADE_GAIN_FLOOR > 1.0
