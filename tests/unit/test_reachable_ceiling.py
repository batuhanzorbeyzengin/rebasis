"""The bound that says whether a better adapter was available.

`arr` answers "how much of a reindex did this adapter recover". It cannot answer
"was more on offer", and those are different questions with opposite responses: a
low `arr` under a high ceiling is a reason to keep looking for a better map, and
a low `arr` under a low ceiling is ADR 10 — retention is bounded by the source
space — and a reason to reindex instead.

Measured over 144 runs, the ceiling ranks runs by the retention they eventually
returned at Spearman 0.90. It is reported and deliberately does **not** enter the
decision rule: a number that predicts well is not the same as a threshold that
has been validated, and the second measurement has not been taken.

The tests below pin the two properties that make it safe to print — it never
promises more than an adapter could deliver, and it declines to answer where the
answer would carry no information.
"""

from __future__ import annotations

import numpy as np
import pytest

from rebasis.core import l2_normalize
from rebasis.probe.groundtruth import GroundTruth, build_tier0
from rebasis.probe.runner import _reachable_ceiling

pytestmark = pytest.mark.unit

DIM = 32
N = 400
K = 10


def _spaces(seed: int, *, noise: float) -> tuple[np.ndarray, np.ndarray]:
    """An old space and a new one that is a rotation of it, plus noise.

    ``noise`` is the knob the tests turn: at zero the two spaces hold the same
    neighbourhoods and everything is reachable, and as it grows the old space
    stops holding what the new model considers close.
    """
    rng = np.random.default_rng(seed)
    old = l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))
    rotation = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype(np.float32)
    new = l2_normalize(old @ rotation.T + rng.standard_normal((N, DIM)).astype(np.float32) * noise)
    return old, new


def _truth(new: np.ndarray, queries: np.ndarray) -> GroundTruth:
    return build_tier0(new, new[queries], queries, k=K)


class TestItDeclinesWhereItWouldSayNothing:
    def test_a_single_relevant_document_returns_none(self) -> None:
        """The centroid of one document is that document, which retrieves itself.

        A ceiling of 1.0 by construction is not a finding, and printing it beside
        an `arr` a reader is trying to interpret would be worse than printing
        nothing. `probe`'s own T0 uses one relevant document per query
        (`SPARSE_RELEVANT`), so this is the ordinary case rather than an edge one.
        """
        old, new = _spaces(0, noise=0.05)
        queries = np.arange(0, N, 8, dtype=np.int64)
        truth = _truth(new, queries)
        single = GroundTruth(
            tier=truth.tier,
            query_indices=truth.query_indices,
            relevant=[{next(iter(r))} for r in truth.relevant],
            relevant_sparse=truth.relevant_sparse,
            oracle_indices=truth.oracle_indices,
            oracle_scores=truth.oracle_scores,
            self_mask=truth.self_mask,
            oracle_recall=truth.oracle_recall,
        )

        assert _reachable_ceiling(old, single, K) is None

    def test_an_empty_relevant_set_returns_none(self) -> None:
        old, new = _spaces(1, noise=0.05)
        queries = np.arange(0, N, 8, dtype=np.int64)
        truth = _truth(new, queries)
        empty = GroundTruth(
            tier=truth.tier,
            query_indices=truth.query_indices,
            relevant=[set() for _ in truth.relevant],
            relevant_sparse=truth.relevant_sparse,
            oracle_indices=truth.oracle_indices,
            oracle_scores=truth.oracle_scores,
            self_mask=truth.self_mask,
            oracle_recall=truth.oracle_recall,
        )

        assert _reachable_ceiling(old, empty, K) is None


class TestItBoundsWhatAnAdapterCanDo:
    def test_it_falls_as_the_two_spaces_diverge(self) -> None:
        """The property the whole number rests on.

        As the new model's neighbourhoods stop existing in the old space, the
        best a single query point could do there falls — which is what makes a
        low ceiling mean "no adapter will find them here" rather than "this
        adapter was weak".
        """
        queries = np.arange(0, N, 4, dtype=np.int64)
        ceilings = []
        for noise in (0.0, 0.4, 1.2):
            old, new = _spaces(7, noise=noise)
            ceiling = _reachable_ceiling(old, _truth(new, queries), K)
            assert ceiling is not None
            ceilings.append(ceiling)

        assert ceilings == sorted(ceilings, reverse=True), ceilings
        assert ceilings[0] > ceilings[-1] + 0.1, ceilings

    def test_an_identical_space_is_fully_reachable(self) -> None:
        """With no drift at all the old index holds exactly what the new model
        would return, so the ceiling is at its maximum."""
        rng = np.random.default_rng(3)
        space = l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))
        queries = np.arange(0, N, 4, dtype=np.int64)

        ceiling = _reachable_ceiling(space, _truth(space, queries), K)

        assert ceiling is not None
        assert ceiling > 0.9, ceiling

    def test_it_is_at_least_as_high_as_a_fitted_adapter(self) -> None:
        """A ceiling an adapter passes is not a ceiling.

        The construction bounds by estimate rather than by proof — the centroid
        maximises summed similarity to the target set, not membership in the top
        ``k`` — so this is the property that would fail first if the estimate
        were too loose to be useful.
        """
        from rebasis.compute import top_k_search
        from rebasis.core import fit_candidates
        from rebasis.probe.metrics import recall_at_k

        old, new = _spaces(11, noise=0.3)
        queries = np.arange(0, N, 4, dtype=np.int64)
        truth = _truth(new, queries)

        fit = np.setdiff1d(np.arange(N), queries)
        adapter = fit_candidates(new[fit], old[fit], normalize=False, methods=["procrustes"])[
            0
        ].adapter
        mapped = l2_normalize(adapter.apply(new[queries]), copy=False)
        indices, _ = top_k_search(mapped, old, k=K, self_mask=truth.self_mask)
        fitted = recall_at_k(indices, truth.relevant_sparse, K)

        ceiling = _reachable_ceiling(old, truth, K)
        assert ceiling is not None
        assert ceiling >= fitted, (ceiling, fitted)
