"""Synthetic adapter validation — "the project's most valuable test".

These catch implementation errors in milliseconds without downloading a model.
An adapter that fails here makes every number it later produces on a real corpus
untrustworthy, which is why this runs before anything touching real embeddings.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rebasis.core import (
    CenteredProcrustesAdapter,
    IdentityAdapter,
    LinearAdapter,
    LowRankAffineAdapter,
    ProcrustesAdapter,
    ScaledAdapter,
    l2_normalize,
)
from rebasis.errors import FitFailed

FAST = settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def random_orthogonal(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Random orthogonal matrix under the Haar measure."""
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    return (q * np.sign(np.diag(r))).astype(np.float32)


@pytest.mark.property
@FAST
@given(dim=st.integers(16, 128), n=st.integers(200, 800), seed=st.integers(0, 2**31 - 1))
def test_procrustes_recovers_a_known_rotation(dim: int, n: int, seed: int) -> None:
    """Orthogonal Procrustes must reproduce a rotation exactly.

    This is the closed form's defining property. If it fails, the SVD solution
    is being applied in the wrong orientation — an error that would otherwise
    surface as "quality is mysteriously poor" on a real corpus.
    """
    rng = np.random.default_rng(seed)
    a = l2_normalize(rng.standard_normal((n, dim)).astype(np.float32))
    rotation = random_orthogonal(dim, rng)
    b = (a @ rotation.T).astype(np.float32)

    adapter = ProcrustesAdapter.fit(src=b, dst=a)
    assert np.allclose(adapter.apply(b), a, atol=1e-4)


@pytest.mark.property
@FAST
@given(dim=st.integers(16, 96), n=st.integers(300, 800), seed=st.integers(0, 2**31 - 1))
def test_centered_procrustes_recovers_a_rotation_with_translation(
    dim: int, n: int, seed: int
) -> None:
    """Centred Procrustes must handle a rotation *plus* a shift.

    This is the case plain Procrustes cannot express, and the reason M0 measured
    +0.26 ARR from centering: a rotation pinned to the origin cannot absorb a
    difference in means however it turns.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, dim)).astype(np.float32)
    rotation = random_orthogonal(dim, rng)
    offset = (rng.standard_normal(dim) * 2.0).astype(np.float32)
    b = (a @ rotation.T + offset).astype(np.float32)

    centred = CenteredProcrustesAdapter.fit(src=b, dst=a)
    assert np.allclose(centred.apply(b), a, atol=1e-3)


@pytest.mark.property
@FAST
@given(dim=st.integers(16, 96), n=st.integers(400, 900), seed=st.integers(0, 2**31 - 1))
def test_centering_beats_plain_procrustes_on_a_shifted_space(dim: int, n: int, seed: int) -> None:
    """The M0 finding as an executable property (`docs/m0-findings.md`, section 4).

    On a space whose mean is displaced, centring must not merely be no worse —
    it must be strictly better. That is the claim the default rests on.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, dim)).astype(np.float32)
    rotation = random_orthogonal(dim, rng)
    offset = (rng.standard_normal(dim) * 3.0).astype(np.float32)
    b = (a @ rotation.T + offset).astype(np.float32)

    plain_error = float(np.abs(ProcrustesAdapter.fit(b, a).apply(b) - a).mean())
    centred_error = float(np.abs(CenteredProcrustesAdapter.fit(b, a).apply(b) - a).mean())
    assert centred_error < plain_error


@pytest.mark.property
@FAST
@given(
    d_in=st.integers(16, 96),
    d_out=st.integers(16, 96),
    n=st.integers(400, 900),
    seed=st.integers(0, 2**31 - 1),
)
def test_linear_recovers_a_known_affine_including_dimension_change(
    d_in: int, d_out: int, n: int, seed: int
) -> None:
    """The ridge solution must reproduce an affine map, dimensions differing or not.

    Dimension change is the case that lets a bridge sidestep a store's dimension
    lock entirely.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d_in)).astype(np.float32)
    weight = (rng.standard_normal((d_in, d_out)) / np.sqrt(d_in)).astype(np.float32)
    bias = (rng.standard_normal(d_out) * 0.1).astype(np.float32)
    y = x @ weight + bias

    adapter = LinearAdapter.fit(src=x, dst=y, alpha=0.0)
    assert adapter.output_dim == d_out
    assert np.allclose(adapter.apply(x), y, atol=1e-2)


@pytest.mark.property
@FAST
@given(dim=st.integers(32, 96), rank=st.integers(4, 16), seed=st.integers(0, 2**31 - 1))
def test_low_rank_recovers_a_genuinely_low_rank_map(dim: int, rank: int, seed: int) -> None:
    """A rank-r truncation must reproduce a rank-r transform."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1200, dim)).astype(np.float32)
    u = (rng.standard_normal((dim, rank)) / np.sqrt(dim)).astype(np.float32)
    v = (rng.standard_normal((rank, dim)) / np.sqrt(rank)).astype(np.float32)
    y = x @ (u @ v)

    adapter = LowRankAffineAdapter.fit(src=x, dst=y, rank=rank, alpha=0.0)
    relative = float(np.abs(adapter.apply(x) - y).max() / (np.abs(y).max() + 1e-12))
    assert relative < 1e-2


@pytest.mark.property
@FAST
@given(dim=st.integers(16, 128), seed=st.integers(0, 2**31 - 1))
def test_apply_is_batch_invariant(dim: int, seed: int) -> None:
    """One vector at a time must equal the whole batch at once.

    A difference here means the hot path and the migration path disagree, which
    would show up as a quality drop that only appears in production.
    """
    rng = np.random.default_rng(seed)
    x = l2_normalize(rng.standard_normal((64, dim)).astype(np.float32))
    y = l2_normalize(rng.standard_normal((64, dim)).astype(np.float32))
    adapter = CenteredProcrustesAdapter.fit(x, y)

    batch = adapter.apply(x)
    single = np.vstack([adapter.apply(x[i : i + 1]) for i in range(x.shape[0])])
    assert np.allclose(batch, single, atol=1e-5)


@pytest.mark.property
@FAST
@given(dim=st.integers(16, 96), seed=st.integers(0, 2**31 - 1))
def test_output_dimension_is_always_honoured(dim: int, seed: int) -> None:
    """Whatever the input width, the output matches the index's dimension."""
    rng = np.random.default_rng(seed)
    src = l2_normalize(rng.standard_normal((300, dim)).astype(np.float32))
    dst = l2_normalize(rng.standard_normal((300, dim + 17)).astype(np.float32))

    for adapter in (
        ProcrustesAdapter.fit(src, dst),
        CenteredProcrustesAdapter.fit(src, dst),
        LinearAdapter.fit(src, dst),
        IdentityAdapter.fit(src, dst),
    ):
        assert adapter.apply(src).shape == (300, dim + 17), type(adapter).__name__


@pytest.mark.unit
def test_dsm_improves_or_matches_its_base(rng: np.random.Generator) -> None:
    """Diagonal scaling fits the residual, so it cannot be worse than the base."""
    dim = 64
    src = l2_normalize(rng.standard_normal((800, dim)).astype(np.float32))
    scale_true = (1.0 + rng.random(dim)).astype(np.float32)
    dst = (src * scale_true).astype(np.float32)

    base = ProcrustesAdapter.fit(src, dst)
    scaled = ScaledAdapter.fit(base, src, dst)

    base_error = float(np.abs(base.apply(src) - dst).mean())
    scaled_error = float(np.abs(scaled.apply(src) - dst).mean())
    assert scaled_error <= base_error


@pytest.mark.unit
def test_dsm_reports_a_composite_name(rng: np.random.Generator) -> None:
    """Reports and audit records must say what was actually used."""
    src = l2_normalize(rng.standard_normal((200, 32)).astype(np.float32))
    dst = l2_normalize(rng.standard_normal((200, 32)).astype(np.float32))
    scaled = ScaledAdapter.fit(CenteredProcrustesAdapter.fit(src, dst), src, dst)
    assert scaled.type_name == "procrustes_centered+dsm"


@pytest.mark.unit
def test_l2_normalize_leaves_zero_vectors_alone() -> None:
    """A zero row must stay zero rather than becoming NaN.

    One NaN propagates through every downstream metric and turns a data problem
    into an unexplained quality problem.
    """
    x = np.zeros((3, 8), dtype=np.float32)
    x[1] = 1.0
    out = l2_normalize(x)
    assert np.isfinite(out).all()
    assert np.allclose(out[0], 0.0)
    assert np.isclose(np.linalg.norm(out[1]), 1.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("src_shape", "dst_shape", "match"),
    [
        ((10, 8), (10, 8), "at least"),
        ((100, 8), (50, 8), "different pair counts"),
    ],
)
def test_fit_rejects_bad_input(
    src_shape: tuple[int, int], dst_shape: tuple[int, int], match: str
) -> None:
    """Bad pair sets fail with an actionable error rather than a silent bad fit."""
    src = np.zeros(src_shape, dtype=np.float32)
    dst = np.zeros(dst_shape, dtype=np.float32)
    with pytest.raises(FitFailed, match=match):
        ProcrustesAdapter.fit(src, dst)


@pytest.mark.unit
def test_fit_rejects_non_finite_input(rng: np.random.Generator) -> None:
    """A single NaN must be caught at the boundary.

    Left through, it produces an all-NaN adapter that fails later as "quality is
    bad" — a far longer path back to the real cause.
    """
    src = rng.standard_normal((200, 16)).astype(np.float32)
    dst = rng.standard_normal((200, 16)).astype(np.float32)
    src[7, 3] = np.nan
    with pytest.raises(FitFailed, match="NaN or infinity"):
        ProcrustesAdapter.fit(src, dst)
