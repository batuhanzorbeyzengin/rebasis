"""Retention at candidate-set depth, and the promise that it changes nothing.

`probe` now reports what an adapter would retain if it produced a *candidate
set* for the new model to rerank rather than the final ranking. That number is
systematically higher than ARR — measured, 0.697 against 0.889 over 24 runs —
which makes it exactly the kind of addition that could quietly move a decision.

So the property tested hardest here is a negative one: widening the search to
reach candidate depth must leave every metric that decides anything **bit for
bit** where it was. The rest is that the new number is on ARR's scale, and that
it is absent rather than invented when there is nothing to compute it from.
"""

from __future__ import annotations

import numpy as np
import pytest

from rebasis.core import l2_normalize
from rebasis.probe.decision import decide
from rebasis.probe.groundtruth import build_tier0
from rebasis.probe.runner import CASCADE_N, run_probe

pytestmark = pytest.mark.unit

DIM = 32
N_DOCS = 600
N_QUERIES = 60


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(23)


@pytest.fixture
def spaces(rng: np.random.Generator):  # type: ignore[no-untyped-def]
    """An old space and a rotated new one, with a held-out query set."""
    centers = (rng.standard_normal((20, DIM)) * 3.0).astype(np.float32)
    assignment = rng.integers(0, 20, size=N_DOCS)
    old = l2_normalize(
        centers[assignment] + rng.standard_normal((N_DOCS, DIM)).astype(np.float32) * 1.4
    )
    rotation = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype(np.float32)
    new = l2_normalize(old @ rotation.T + rng.standard_normal(old.shape).astype(np.float32) * 0.2)

    query_indices = np.arange(N_QUERIES)
    fit_indices = np.arange(N_QUERIES, N_DOCS)
    ground_truth = build_tier0(new, new[query_indices], query_indices, k=10)
    return {
        "old": old,
        "new": new,
        "query_indices": query_indices,
        "fit_indices": fit_indices,
        "ground_truth": ground_truth,
    }


def probe(spaces, **kwargs):  # type: ignore[no-untyped-def]
    return run_probe(
        old_doc_vectors=spaces["old"],
        new_doc_vectors=spaces["new"],
        fit_indices=spaces["fit_indices"],
        ground_truth=spaces["ground_truth"],
        old_query_vectors=spaces["old"][spaces["query_indices"]],
        new_query_vectors=spaces["new"][spaces["query_indices"]],
        k=10,
        methods=["procrustes"],
        with_csls=False,
        **kwargs,
    )


class TestItChangesNothing:
    """The property that matters more than the new number itself."""

    def test_every_decision_metric_is_identical_with_and_without_it(self, spaces) -> None:  # type: ignore[no-untyped-def]
        """Widening the search may not move ARR, its interval, nDCG or the
        decision. If it can, then the number that decides depends on how deep an
        unrelated diagnostic happened to look."""
        without = probe(spaces, cascade_k=None)
        with_cascade = probe(spaces, cascade_k=CASCADE_N)

        assert with_cascade.best.arr == without.best.arr
        assert with_cascade.best.arr_ci == without.best.arr_ci
        assert with_cascade.best.ndcg == without.best.ndcg
        assert with_cascade.best.mrr == without.best.mrr
        assert with_cascade.best.overlap == without.best.overlap
        assert with_cascade.decision.decision == without.decision.decision

    def test_the_winning_candidate_is_the_same(self, spaces) -> None:  # type: ignore[no-untyped-def]
        """Selection runs on ARR at k, so a wider search must not reorder it."""
        without = probe(spaces, cascade_k=None)
        with_cascade = probe(spaces, cascade_k=CASCADE_N)

        assert with_cascade.best.name == without.best.name


class TestTheNumber:
    def test_it_is_reported_and_is_at_least_arr(self, spaces) -> None:  # type: ignore[no-untyped-def]
        """Reaching the top 100 is a weaker requirement than ranking in the top
        10, so retention at depth cannot be lower — on the same retrieval, by
        the same adapter, against the same judgements."""
        result = probe(spaces, cascade_k=CASCADE_N)

        assert result.best.cascade_arr is not None
        assert result.best.cascade_arr >= result.best.arr - 1e-6

    def test_it_is_absent_when_not_asked_for(self, spaces) -> None:  # type: ignore[no-untyped-def]
        result = probe(spaces, cascade_k=None)

        assert result.best.cascade_arr is None
        assert result.decision.cascade_advantage is None

    def test_it_is_absent_when_the_depth_adds_nothing(self, spaces) -> None:  # type: ignore[no-untyped-def]
        """A candidate depth at or below `k` is the number ARR already reports.
        Publishing it twice under two names invites a reader to compare a
        quantity with itself."""
        result = probe(spaces, cascade_k=5)

        assert result.best.cascade_arr is None

    def test_it_serialises(self, spaces) -> None:  # type: ignore[no-untyped-def]
        payload = probe(spaces, cascade_k=CASCADE_N).to_dict()

        assert payload["cascade_arr"] is not None
        assert payload["best_adapter"]["cascade_arr"] is not None


class TestTheBreakEven:
    def test_it_is_the_product_with_the_upgrade(self) -> None:
        result = decide(0.70, upgrade_gain=1.5, cascade_arr=0.90, old_model_arr=0.95)

        assert result.bridge_advantage == pytest.approx(0.70 * 1.5)
        assert result.cascade_advantage == pytest.approx(0.90 * 1.5)

    def test_it_needs_both_halves(self) -> None:
        """Retention without an upgrade estimate answers a different question,
        and multiplying by an absent number is not a way to answer this one."""
        assert decide(0.70, cascade_arr=0.90).cascade_advantage is None
        assert decide(0.70, upgrade_gain=1.5).cascade_advantage is None

    def test_it_does_not_move_the_decision(self) -> None:
        """The arrangement it describes is one rebasis measures and does not
        serve. A rule built on it would recommend something the tool cannot do."""
        without = decide(0.70, upgrade_gain=1.05, old_model_arr=0.95)
        with_cascade = decide(0.70, upgrade_gain=1.05, old_model_arr=0.95, cascade_arr=0.99)

        assert with_cascade.decision == without.decision
        assert with_cascade.rationale == without.rationale
