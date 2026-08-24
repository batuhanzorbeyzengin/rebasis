"""A query log with no relevance judgements.

The common case, and the one rebasis used to reject. A search box exports the
queries people typed; almost nobody has someone's judgement of which document
was right. Such a log still answers the retention half of the question — how
much of what a full reindex would return does the adapter return, on the
queries people actually type — and cannot answer the upgrade half, because the
oracle cannot grade itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from rebasis.core import l2_normalize
from rebasis.probe.groundtruth import build_tier1_unjudged
from rebasis.probe.session import QueryLog

pytestmark = pytest.mark.unit

DIM = 24
N = 200
K = 10


@pytest.fixture
def vectors(rng: np.random.Generator) -> np.ndarray:
    return l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))


class TestTheLogItself:
    def test_a_log_without_judgements_says_so(self) -> None:
        log = QueryLog(queries=["a", "b"], qrels=[set(), set()])

        assert not log.judged

    def test_one_judgement_is_enough_to_count_as_judged(self) -> None:
        log = QueryLog(queries=["a", "b"], qrels=[set(), {"doc-3"}])

        assert log.judged


class TestTheGroundTruth:
    def test_the_oracle_defines_the_answer(self, vectors, rng) -> None:  # type: ignore[no-untyped-def]
        """ARR against this is pure retention: the denominator is a full
        reindex, by construction, so it is exactly 1.0."""
        queries = l2_normalize(rng.standard_normal((30, DIM)).astype(np.float32))

        truth = build_tier1_unjudged(vectors, queries, k=K)

        assert truth.tier == "t1u"
        assert truth.oracle_recall == 1.0
        assert np.allclose(truth.oracle_recall_per_query, 1.0)

    def test_every_query_has_exactly_k_relevant_documents(self, vectors, rng) -> None:  # type: ignore[no-untyped-def]
        queries = l2_normalize(rng.standard_normal((30, DIM)).astype(np.float32))

        truth = build_tier1_unjudged(vectors, queries, k=K)

        assert all(len(judged) == K for judged in truth.relevant)

    def test_nothing_is_masked_out(self, vectors, rng) -> None:  # type: ignore[no-untyped-def]
        """A real query is text that was never indexed, so unlike T0 there is no
        document standing in for it that would retrieve itself."""
        queries = l2_normalize(rng.standard_normal((5, DIM)).astype(np.float32))

        assert build_tier1_unjudged(vectors, queries, k=K).self_mask is None


class TestTheDecisionItLeadsTo:
    def test_it_is_provisional_because_the_upgrade_is_unmeasurable(self) -> None:
        """The oracle cannot grade itself, so there is no upgrade estimate — and
        `decide` already treats a missing estimate as grounds for saying so."""
        from rebasis.probe.decision import decide

        result = decide(0.91, upgrade_gain=None, tier="t1u")

        assert result.provisional
        assert result.bridge_advantage is None

    def test_it_is_not_treated_as_a_synthesised_estimate(self) -> None:
        """T2's `unsettled_estimate` rule keys on a gain near 1.0 from
        synthesised queries. There is no gain here at all, so that rule must not
        be the reason this run is provisional."""
        from rebasis.probe.decision import decide

        result = decide(0.91, upgrade_gain=None, tier="t1u")

        assert result.tier == "t1u"
