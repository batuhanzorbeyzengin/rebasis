"""nDCG@k, and its agreement with ranx.

nDCG lives in the core path rather than behind the ``[eval]`` extra because it
changes decisions. Measured on BEIR/scifact, MiniLM to bge-base: the adapter
improved recall@10 from 0.7833 to 0.7867 while nDCG@10 fell from 0.6451 to
0.6294 — the same documents, ordered worse, which a recall-only rule read as an
improvement.

A second implementation is only worth having if it agrees with the first, so the
last test checks ours against ranx on the same inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from rebasis.probe.metrics import ndcg_at_k, recall_at_k

pytestmark = pytest.mark.unit


class TestOrdering:
    def test_a_perfect_ranking_scores_one(self) -> None:
        retrieved = np.array([[7, 1, 2], [3, 9, 4]])

        assert ndcg_at_k(retrieved, [{7}, {3}], 3) == pytest.approx(1.0)

    def test_the_same_documents_lower_down_score_less(self) -> None:
        """The property recall cannot express."""
        first = np.array([[7, 1, 2]])
        third = np.array([[1, 2, 7]])

        assert recall_at_k(first, [{7}], 3) == recall_at_k(third, [{7}], 3)
        assert ndcg_at_k(first, [{7}], 3) > ndcg_at_k(third, [{7}], 3)

    def test_missing_everything_scores_zero(self) -> None:
        assert ndcg_at_k(np.array([[1, 2, 3]]), [{9}], 3) == pytest.approx(0.0)

    def test_the_discount_is_logarithmic(self) -> None:
        """Rank 2 is worth 1/log2(3) of rank 1, which is the standard."""
        second = ndcg_at_k(np.array([[1, 7]]), [{7}], 2)

        assert second == pytest.approx(1.0 / np.log2(3))


class TestMultipleRelevant:
    def test_finding_more_scores_higher(self) -> None:
        one = np.array([[7, 1, 2]])
        both = np.array([[7, 8, 2]])

        assert ndcg_at_k(both, [{7, 8}], 3) > ndcg_at_k(one, [{7, 8}], 3)

    def test_the_ideal_ranking_is_the_denominator(self) -> None:
        """Two relevant documents at ranks 1 and 2 is perfect, not partial."""
        assert ndcg_at_k(np.array([[7, 8, 1]]), [{7, 8}], 3) == pytest.approx(1.0)


class TestEdges:
    def test_queries_with_no_relevant_documents_are_skipped(self) -> None:
        """Matches recall_at_k: scoring them zero understates every system
        equally, changing absolute numbers without changing any comparison."""
        retrieved = np.array([[7, 1], [1, 2]])

        assert ndcg_at_k(retrieved, [{7}, set()], 2) == pytest.approx(1.0)

    def test_an_empty_result_is_zero(self) -> None:
        assert ndcg_at_k(np.empty((0, 0), dtype=np.int64), [], 10) == 0.0

    def test_graded_relevance_outranks_binary(self) -> None:
        """A highly relevant document at rank 1 beats a marginal one there."""
        grades = [{7: 3.0, 8: 1.0}]
        good = ndcg_at_k(np.array([[7, 8]]), [{7, 8}], 2, grades=grades)
        worse = ndcg_at_k(np.array([[8, 7]]), [{7, 8}], 2, grades=grades)

        assert good > worse
        assert good == pytest.approx(1.0)


class TestAgreementWithRanx:
    def test_it_matches_ranx_on_the_same_inputs(self, rng: np.random.Generator) -> None:
        ranx = pytest.importorskip("ranx", reason="the [eval] extra is not installed")

        n_queries, k, n_docs = 60, 10, 200
        retrieved = np.array([rng.choice(n_docs, size=k, replace=False) for _ in range(n_queries)])
        relevant = [
            set(rng.choice(n_docs, size=rng.integers(1, 4), replace=False).tolist())
            for _ in range(n_queries)
        ]

        ours = ndcg_at_k(retrieved, relevant, k)

        qrels = ranx.Qrels({str(i): {str(d): 1 for d in rel} for i, rel in enumerate(relevant)})
        run = ranx.Run(
            {
                str(i): {str(d): float(k - rank) for rank, d in enumerate(row.tolist())}
                for i, row in enumerate(retrieved)
            }
        )
        theirs = float(ranx.evaluate(qrels, run, f"ndcg@{k}"))

        assert ours == pytest.approx(theirs, abs=1e-6)


class TestTheDecisionSurfacesDisagreement:
    """Recall decides; ranking gets a hearing.

    The decision bands were calibrated on recall and every recorded ARR is a
    recall figure, so nDCG does not override them. But an adapter that returns
    the same documents in a worse order is a real outcome the recall number
    cannot express, and staying silent about it is how the original defect went
    unnoticed.
    """

    def test_it_warns_when_the_two_metrics_disagree(self) -> None:
        """The measured case: BEIR/scifact, MiniLM to bge-base."""
        from rebasis.probe.decision import decide

        result = decide(0.906, upgrade_gain=1.113, old_model_arr=0.898, ndcg_advantage=0.9757)

        assert any("Recall and ranking disagree" in w for w in result.warnings)

    def test_it_says_which_way_round(self) -> None:
        from rebasis.probe.decision import decide

        joined = " ".join(
            decide(0.906, upgrade_gain=1.113, old_model_arr=0.898, ndcg_advantage=0.9757).warnings
        )

        assert "improves recall and degrades ranking" in joined

    def test_agreement_is_quiet(self) -> None:
        from rebasis.probe.decision import decide

        result = decide(0.906, upgrade_gain=1.113, old_model_arr=0.898, ndcg_advantage=1.05)

        assert not any("disagree" in w for w in result.warnings)

    def test_it_does_not_override_the_decision(self) -> None:
        """nDCG informs; it does not decide. Every recorded ARR is a recall
        figure and the bands were calibrated against those."""
        from rebasis.probe.decision import decide

        with_ndcg = decide(0.906, upgrade_gain=1.113, old_model_arr=0.898, ndcg_advantage=0.20)
        without = decide(0.906, upgrade_gain=1.113, old_model_arr=0.898)

        assert with_ndcg.decision == without.decision

    def test_it_is_absent_when_not_measured(self) -> None:
        from rebasis.probe.decision import decide

        assert decide(0.93, upgrade_gain=1.2, old_model_arr=0.8).ndcg_advantage is None
