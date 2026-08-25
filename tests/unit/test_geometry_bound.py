"""The Procrustes bound, checked against the alignment it claims to bound.

`rebasis.core.geometry` implements Corollary 1 of Maystre et al.
(arXiv:2510.13406): if two spaces' pairwise inner products agree to within δ,
the best orthogonal alignment cannot be worse than √(2D)·δ.

A bound is only worth printing if it holds, so the test that matters here is the
one that fits an actual orthogonal map and checks that the error it achieves is
underneath the ceiling. Everything else is the endpoints — identical spaces,
rotated spaces, unrelated spaces — where the right answer is known without
measuring anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from rebasis.core import ProcrustesAdapter, geometry_bound, l2_normalize

pytestmark = pytest.mark.unit

DIM = 48
N = 400


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(19)


@pytest.fixture
def space(rng: np.random.Generator) -> np.ndarray:
    return l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))


class TestTheEndpoints:
    def test_a_space_against_itself_has_no_difference(self, space: np.ndarray) -> None:
        bound = geometry_bound(space, space)

        assert bound.delta == pytest.approx(0.0, abs=1e-6)
        assert bound.bound == pytest.approx(0.0, abs=1e-4)
        assert bound.cosine_floor == pytest.approx(1.0, abs=1e-4)

    def test_a_rotation_preserves_every_inner_product(
        self, space: np.ndarray, rng: np.random.Generator
    ) -> None:
        """The property the whole bound rests on.

        An orthogonal map leaves every pairwise similarity exactly where it was,
        so δ is zero however far the vectors themselves moved. A δ that reacted
        to rotation would be measuring the coordinate system rather than the
        geometry, and the bound built on it would mean nothing.
        """
        rotation = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype(np.float32)

        bound = geometry_bound(space @ rotation.T, space)

        assert bound.delta == pytest.approx(0.0, abs=1e-5)
        assert bound.informative

    def test_unrelated_spaces_give_a_bound_that_says_nothing(
        self, space: np.ndarray, rng: np.random.Generator
    ) -> None:
        """Reported as uninformative rather than as a floor of zero.

        Two unit vectors are never more than 2 apart, so a ceiling at or above 2
        permits every outcome. Printing "cosine ≥ -0.4" would invite a reader to
        compare it with a real floor as if the two were the same kind of thing.
        """
        unrelated = l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))

        bound = geometry_bound(unrelated, space)

        assert bound.delta > 0
        assert not bound.informative
        assert bound.cosine_floor is None
        assert "says nothing" in bound.explain()


class TestTheBoundHolds:
    """Fit the alignment the bound is about, and check it stays underneath."""

    @pytest.mark.parametrize("noise", [0.0, 0.05, 0.15, 0.4, 1.0])
    def test_the_measured_alignment_error_is_under_the_ceiling(
        self, space: np.ndarray, rng: np.random.Generator, noise: float
    ) -> None:
        """Across the whole range from "identical" to "unrelated".

        The target is a rotation of the source plus noise, which is the shape of
        a real model pair: the geometry mostly survives and partly does not.
        Sweeping the noise sweeps δ, and the inequality has to hold at every
        point — a bound that only holds where the spaces are already close is
        not a bound.
        """
        rotation = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype(np.float32)
        target = l2_normalize(
            space @ rotation.T + rng.standard_normal(space.shape).astype(np.float32) * noise
        )

        bound = geometry_bound(space, target)

        # `x̄` in the corollary: the source under the *best* orthogonal map,
        # which is exactly what orthogonal Procrustes solves for.
        aligned = ProcrustesAdapter.fit(space, target).apply(space)
        measured = float(np.mean(np.sum((aligned - target) ** 2, axis=1)))

        assert measured <= bound.bound + 1e-6

    def test_the_bound_tightens_as_the_spaces_agree(
        self, space: np.ndarray, rng: np.random.Generator
    ) -> None:
        """Monotone in the right direction, which is what makes it readable.

        A bound that held but moved arbitrarily would be true and useless: the
        reason it is worth printing is that a smaller number means a better
        alignment is available.
        """
        rotation = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype(np.float32)
        bounds = []
        for noise in (0.05, 0.2, 0.5):
            target = l2_normalize(
                space @ rotation.T + rng.standard_normal(space.shape).astype(np.float32) * noise
            )
            bounds.append(geometry_bound(space, target).bound)

        assert bounds == sorted(bounds)


class TestTheMechanics:
    def test_the_diagonal_is_excluded(self, space: np.ndarray, rng: np.random.Generator) -> None:
        """Every row's similarity with itself is 1 on both sides.

        Those n guaranteed zeros would pull the mean down by 1/n, which flatters
        δ on exactly the small samples where it is least reliable.
        """
        target = l2_normalize(space + rng.standard_normal(space.shape).astype(np.float32) * 0.3)

        small = geometry_bound(space[:20], target[:20])
        large = geometry_bound(space[:400], target[:400])

        # With the diagonal included the small sample would report a visibly
        # lower delta than the large one on the same distribution.
        assert small.delta == pytest.approx(large.delta, rel=0.25)

    def test_the_larger_dimension_is_used(self, rng: np.random.Generator) -> None:
        """√(2D) grows with D, so the wider space gives the looser ceiling.

        That is the conservative choice, and it matches the paper's own handling
        of a dimension mismatch — zero-pad the smaller side, which preserves its
        geometry and leaves it living in the larger space.
        """
        wide = l2_normalize(rng.standard_normal((N, 768)).astype(np.float32))
        narrow = l2_normalize(rng.standard_normal((N, 256)).astype(np.float32))

        assert geometry_bound(wide, narrow).dim == 768
        assert geometry_bound(narrow, wide).dim == 768

    def test_the_sample_is_capped(self, rng: np.random.Generator) -> None:
        """The Gram matrix is n x n, so an uncapped sample is the one way this
        check could cost more than the fit it precedes."""
        big = l2_normalize(rng.standard_normal((5000, DIM)).astype(np.float32))

        bound = geometry_bound(big, big, sample=500)

        assert bound.n_pairs == 500

    def test_the_same_seed_picks_the_same_rows(self, rng: np.random.Generator) -> None:
        big = l2_normalize(rng.standard_normal((5000, DIM)).astype(np.float32))
        other = l2_normalize(rng.standard_normal((5000, DIM)).astype(np.float32))

        first = geometry_bound(big, other, sample=300, seed=7)
        second = geometry_bound(big, other, sample=300, seed=7)

        assert first.delta == second.delta

    def test_too_few_rows_say_nothing(self, rng: np.random.Generator) -> None:
        one = l2_normalize(rng.standard_normal((1, DIM)).astype(np.float32))

        bound = geometry_bound(one, one)

        assert np.isnan(bound.delta)

    def test_it_serialises(self, space: np.ndarray) -> None:
        payload = geometry_bound(space, space).to_dict()

        assert payload["geometry_delta"] == pytest.approx(0.0, abs=1e-4)
        assert payload["cosine_floor"] == pytest.approx(1.0, abs=1e-4)
        assert payload["dim"] == DIM
