"""ARR's confidence interval must be an interval for ARR.

ARR is ``mean(candidate recall) / mean(oracle recall)``. Bootstrapping only the
numerator gives the interval of the numerator — a different quantity. At T0 the
oracle is perfect by construction so the two coincide, which is why every
synthetic test passed while the first real-corpus run reported

    ARR@10  0.908  (95% CI 0.712-0.808)

with the point estimate outside its own range.
"""

from __future__ import annotations

import numpy as np
import pytest

from rebasis.probe.metrics import bootstrap_ci, bootstrap_ratio_ci

pytestmark = pytest.mark.unit


@pytest.fixture
def paired(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Per-query recall for a candidate and for an imperfect oracle.

    Correlated on purpose: a query that is hard for one is usually hard for the
    other, and that correlation is what the paired bootstrap exists to keep.
    """
    difficulty = rng.random(400).astype(np.float32)
    oracle = np.clip(0.95 - 0.5 * difficulty, 0, 1).astype(np.float32)
    candidate = np.clip(oracle - 0.08 * rng.random(400), 0, 1).astype(np.float32)
    return candidate, oracle


class TestTheIntervalMatchesThePointEstimate:
    def test_the_point_estimate_lies_inside_its_own_interval(
        self, paired: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """The defect, stated as the property it violated."""
        candidate, oracle = paired
        arr = float(candidate.mean() / oracle.mean())

        low, high = bootstrap_ratio_ci(candidate, oracle)

        assert low <= arr <= high, f"ARR {arr:.4f} outside [{low:.4f}, {high:.4f}]"

    def test_the_numerator_only_interval_does_not_contain_it(
        self, paired: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """The old behaviour, kept as a test so the difference stays visible.

        With an imperfect oracle the two intervals are on different scales, and
        this asserts the gap rather than leaving it as a claim in a comment.
        """
        candidate, oracle = paired
        arr = float(candidate.mean() / oracle.mean())

        low, high = bootstrap_ci(candidate)

        assert not (low <= arr <= high)

    def test_a_perfect_oracle_makes_the_two_agree(self, rng: np.random.Generator) -> None:
        """Why T0 never caught it: with oracle recall 1.0 the ratio is the mean."""
        candidate = rng.random(300).astype(np.float32)
        perfect = np.ones(300, dtype=np.float32)

        ratio_low, ratio_high = bootstrap_ratio_ci(candidate, perfect)
        plain_low, plain_high = bootstrap_ci(candidate)

        assert ratio_low == pytest.approx(plain_low, abs=1e-6)
        assert ratio_high == pytest.approx(plain_high, abs=1e-6)


class TestPairing:
    def test_pairing_narrows_the_interval(self, paired: tuple[np.ndarray, np.ndarray]) -> None:
        """Resampling the two series independently would pretend the correlation
        away and report a wider interval than the data supports."""
        candidate, oracle = paired
        low, high = bootstrap_ratio_ci(candidate, oracle)

        rng = np.random.default_rng(0)
        shuffled = oracle[rng.permutation(oracle.size)]
        unpaired_low, unpaired_high = bootstrap_ratio_ci(candidate, shuffled)

        assert (high - low) <= (unpaired_high - unpaired_low)


class TestEdges:
    def test_an_empty_series_is_nan_not_a_crash(self) -> None:
        empty = np.empty(0, dtype=np.float32)

        low, high = bootstrap_ratio_ci(empty, empty)

        assert np.isnan(low)
        assert np.isnan(high)

    def test_mismatched_lengths_are_nan(self, rng: np.random.Generator) -> None:
        """Unequal series cannot be paired, and silently truncating would pair
        the wrong queries with each other."""
        low, high = bootstrap_ratio_ci(rng.random(10).astype(np.float32), np.ones(4, np.float32))

        assert np.isnan(low)
        assert np.isnan(high)

    def test_an_oracle_that_retrieves_nothing_is_nan(self) -> None:
        """A ratio with a zero denominator carries no information about ARR."""
        low, high = bootstrap_ratio_ci(np.zeros(50, np.float32), np.zeros(50, np.float32))

        assert np.isnan(low)
        assert np.isnan(high)
