"""Continuous re-fitting (K1).

A refit that is adopted without being better is a silent quality regression —
the failure mode this project keeps designing against. These tests pin the two
guards that prevent it: adopt only on a measured improvement, and measure on data
the candidate was not fitted to.
"""

from __future__ import annotations

import numpy as np
import pytest

from rebasis.core import CenteredProcrustesAdapter, ProcrustesAdapter, l2_normalize
from rebasis.migrate import MIN_IMPROVEMENT, RefitPolicy, consider_refit

DIM = 48


@pytest.fixture
def good_pairs(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Matched pairs related by a clean rotation — learnable in full."""
    rotation = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype(np.float32)
    src = l2_normalize(rng.standard_normal((4000, DIM)).astype(np.float32))
    return src, l2_normalize(src @ rotation.T)


@pytest.fixture
def weak_adapter(rng: np.random.Generator) -> ProcrustesAdapter:
    """An adapter fitted on unrelated pairs, so it maps nothing usefully."""
    src = l2_normalize(rng.standard_normal((300, DIM)).astype(np.float32))
    dst = l2_normalize(rng.standard_normal((300, DIM)).astype(np.float32))
    return ProcrustesAdapter.fit(src, dst)


@pytest.mark.unit
def test_a_clearly_better_refit_is_adopted(
    weak_adapter: ProcrustesAdapter, good_pairs: tuple[np.ndarray, np.ndarray]
) -> None:
    """The case the feature exists for: better pairs accumulate mid-migration."""
    src, dst = good_pairs
    decision = consider_refit(
        weak_adapter, src=src, dst=dst, policy=RefitPolicy(enabled=True), job_id="j1"
    )

    assert decision.adopted
    assert decision.adapter is not None
    assert decision.improvement > MIN_IMPROVEMENT


@pytest.mark.unit
def test_a_marginal_refit_is_refused(good_pairs: tuple[np.ndarray, np.ndarray]) -> None:
    """An adapter that is already good must not be swapped on noise.

    The threshold is deliberately above nothing: M0 measured ARR's bootstrap
    interval at ±0.024, so a hair's-breadth margin would be reading sampling
    noise as improvement.
    """
    src, dst = good_pairs
    already_good = CenteredProcrustesAdapter.fit(src[:2000], dst[:2000])

    decision = consider_refit(already_good, src=src, dst=dst, policy=RefitPolicy(enabled=True))
    assert not decision.adopted
    assert "below the" in decision.reason


@pytest.mark.unit
def test_too_few_pairs_means_no_attempt(
    weak_adapter: ProcrustesAdapter, good_pairs: tuple[np.ndarray, np.ndarray]
) -> None:
    """Refitting on a handful of pairs fits noise."""
    src, dst = good_pairs
    decision = consider_refit(
        weak_adapter, src=src[:100], dst=dst[:100], policy=RefitPolicy(enabled=True)
    )

    assert not decision.adopted
    assert decision.adapter is None
    assert "accumulated pairs" in decision.reason


@pytest.mark.unit
def test_scoring_uses_held_out_pairs(
    weak_adapter: ProcrustesAdapter, good_pairs: tuple[np.ndarray, np.ndarray]
) -> None:
    """The candidate is judged on data it was not fitted to.

    Scoring on the fitting data favours the candidate automatically, which is how
    an overfitted adapter gets adopted.
    """
    src, dst = good_pairs
    policy = RefitPolicy(enabled=True, holdout_fraction=0.25)
    decision = consider_refit(weak_adapter, src=src, dst=dst, policy=policy)

    assert decision.adopted
    assert "held-out" in decision.reason


@pytest.mark.unit
def test_the_policy_gates_how_often_a_refit_is_tried() -> None:
    """Refitting has a cost; it is attempted on an interval, not every batch."""
    policy = RefitPolicy(enabled=True, every_n_records=50_000)

    assert not policy.due(processed=10_000, last_refit_at=0)
    assert policy.due(processed=50_000, last_refit_at=0)
    assert not policy.due(processed=60_000, last_refit_at=50_000)


@pytest.mark.unit
def test_refitting_is_off_by_default() -> None:
    """Opt-in: a job that swaps adapters mid-run is a job with more to explain."""
    assert not RefitPolicy().enabled
    assert not RefitPolicy().due(processed=10**9, last_refit_at=0)
