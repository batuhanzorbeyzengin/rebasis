"""The two-stage arrangement, priced and recommended.

`docs/cascade-band.md` measures single-stage bridging beating the status quo in
1 run of 48 and the two-stage arrangement beating it in 36. Until now the tool
measured that and did not recommend it, because the arrangement's cost turns on
a cache hit rate and a hit rate is a property of traffic rather than of a corpus.

This file tests the thing that changed: given ``--queries`` the traffic *is*
sampled, the overlap between the candidate sets prices the cache, and the rule
fires. Four conditions gate it and each one is tested for its veto separately —
a rule that recommends an arrangement the store cannot run, or one whose price
was assumed rather than counted, is worse than one that stays quiet.
"""

from __future__ import annotations

import numpy as np
import pytest

from rebasis.core import l2_normalize
from rebasis.probe.decision import BORDERLINE_BAND, decide
from rebasis.probe.groundtruth import build_tier0, build_tier1
from rebasis.probe.metrics import candidate_reuse
from rebasis.probe.runner import run_probe

pytestmark = pytest.mark.unit

DIM = 32
N_DOCS = 800
N_QUERIES = 60


def priced(**overrides: object):  # type: ignore[no-untyped-def]
    """A run where every gate is open, so a test can close exactly one.

    ARR 0.70 against an upgrade of 1.35 puts the single stage at 0.945 — below
    the break-even, which is the run the arrangement exists for — while
    retention at depth of 0.90 puts the two-stage figure at 1.215.
    """
    arguments: dict[str, object] = {
        "upgrade_gain": 1.35,
        "old_model_arr": 0.72,
        "cascade_arr": 0.90,
        "candidate_reuse": 0.6,
        "cascade_n": 200,
        "can_read_text": True,
    }
    arguments.update(overrides)
    return decide(0.70, **arguments)  # type: ignore[arg-type]


class TestTheCount:
    """What ``candidate_reuse`` counts, on arrays whose answer is known."""

    def test_disjoint_candidate_sets_reuse_nothing(self) -> None:
        indices = np.arange(400).reshape(4, 100)

        assert candidate_reuse(indices) == pytest.approx(0.0)

    def test_identical_candidate_sets_reuse_all_but_the_first(self) -> None:
        """Four queries over the same 100 documents embed 100, not 400."""
        indices = np.tile(np.arange(100), (4, 1))

        assert candidate_reuse(indices) == pytest.approx(1 - 1 / 4)

    def test_it_is_absent_rather_than_zero_on_nothing(self) -> None:
        """Zero would read as "your queries share no documents", which is a
        finding. No candidate sets is the absence of one."""
        assert candidate_reuse(np.empty((0, 0), dtype=np.int64)) is None


class TestTheRule:
    def test_it_recommends_the_arrangement_when_every_gate_is_open(self) -> None:
        result = priced()

        assert result.arrangement == "cascade"
        assert result.cascade_advantage is not None
        assert result.cascade_advantage > 1 + BORDERLINE_BAND

    def test_the_decision_itself_is_untouched(self) -> None:
        """Different axes. One says what to do with the index, the other says
        what to put in front of it, and a run can honestly be told both."""
        with_arrangement = priced()
        without = priced(cascade_arr=None)

        assert with_arrangement.decision == without.decision

    def test_a_store_with_no_text_blocks_it(self) -> None:
        """`Cascade` refuses at construction on such a store, so recommending
        the arrangement would name something that cannot be run."""
        result = priced(can_read_text=False)

        assert result.arrangement == "single_stage"
        assert any("text of a record" in w for w in result.warnings)

    def test_an_unasked_store_blocks_it_too(self) -> None:
        result = priced(can_read_text=None)

        assert result.arrangement == "single_stage"

    def test_an_unpriced_cache_blocks_it(self) -> None:
        """No real query log, no measured overlap. Assuming a hit rate is the
        one thing this project does not do."""
        result = priced(candidate_reuse=None)

        assert result.arrangement == "single_stage"
        assert any("--queries" in w for w in result.warnings)

    def test_a_single_stage_that_already_wins_blocks_it(self) -> None:
        """Nobody needs both. Where bridging alone pays, a rerank stage is cost
        for nothing."""
        result = priced(upgrade_gain=1.8)

        assert result.bridge_advantage is not None
        assert result.bridge_advantage > 1 + BORDERLINE_BAND
        assert result.arrangement == "single_stage"

    def test_a_two_stage_figure_inside_the_noise_band_is_not_a_win(self) -> None:
        """1.01x is a coin flip at this measurement's precision, and every other
        threshold in the module reads it that way."""
        result = priced(cascade_arr=0.75)

        assert result.cascade_advantage == pytest.approx(0.75 * 1.35)
        assert result.arrangement == "single_stage"

    def test_no_upgrade_leaves_nothing_for_either_arrangement(self) -> None:
        """The settled path. Nothing to deliver, so neither stage invents any."""
        result = priced(upgrade_gain=1.0, cascade_arr=0.99)

        assert result.decision == "no_upgrade_needed"
        assert result.arrangement == "single_stage"

    def test_it_reaches_the_run_that_was_told_to_reindex(self) -> None:
        """The case the whole item exists for: bridging lost to doing nothing,
        the decision short-circuits to `full_reindex` before the bands are
        consulted at all, and the arrangement still has to be offered there."""
        result = decide(
            0.60,
            upgrade_gain=1.40,
            old_model_arr=0.90,
            cascade_arr=0.92,
            candidate_reuse=0.5,
            cascade_n=200,
            can_read_text=True,
        )

        assert result.decision == "full_reindex"
        assert result.arrangement == "cascade"


class TestThePrice:
    def test_it_is_the_share_of_each_candidate_set_that_is_not_cached(self) -> None:
        result = priced(candidate_reuse=0.6, cascade_n=200)

        assert result.cascade_embeddings_per_query == pytest.approx(200 * 0.4)

    def test_it_is_reported_even_where_the_arrangement_is_not_recommended(self) -> None:
        """A reader who is weighing the arrangement anyway needs its cost; the
        rule declining to recommend it is not a reason to withhold the number."""
        result = priced(cascade_arr=0.75)

        assert result.arrangement == "single_stage"
        assert result.cascade_embeddings_per_query == pytest.approx(80.0)

    def test_it_is_absent_when_the_cache_was_never_priced(self) -> None:
        assert priced(candidate_reuse=None).cascade_embeddings_per_query is None


class TestTheContract:
    def test_the_new_fields_serialise(self) -> None:
        payload = priced().to_dict()

        assert payload["arrangement"] == "cascade"
        assert payload["candidate_reuse"] == pytest.approx(0.6)
        assert payload["cascade_embeddings_per_query"] == pytest.approx(80.0)
        assert payload["cascade_n"] == 200

    def test_the_decision_field_still_holds_only_its_five_values(self) -> None:
        """`arrangement` is beside the decision rather than a sixth value in it,
        because a new value in that Literal breaks every script branching on it
        while a new key does not (`docs/stability.md`)."""
        assert priced().to_dict()["decision"] in {
            "no_upgrade_needed",
            "bridge_sufficient",
            "bridge_and_migrate",
            "caution",
            "full_reindex",
        }

    def test_the_default_is_the_single_stage(self) -> None:
        """A caller that measured no cascade gets the arrangement it measured."""
        assert decide(0.9, upgrade_gain=1.1).arrangement == "single_stage"


@pytest.fixture
def spaces():  # type: ignore[no-untyped-def]
    """An old space, a rotated new one, and a query set outside the index."""
    rng = np.random.default_rng(11)
    centers = (rng.standard_normal((20, DIM)) * 3.0).astype(np.float32)
    assignment = rng.integers(0, 20, size=N_DOCS)
    old = l2_normalize(
        centers[assignment] + rng.standard_normal((N_DOCS, DIM)).astype(np.float32) * 1.4
    )
    rotation = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype(np.float32)
    new = l2_normalize(old @ rotation.T + rng.standard_normal(old.shape).astype(np.float32) * 0.2)
    queries_new = l2_normalize(
        new[:N_QUERIES] + rng.standard_normal((N_QUERIES, DIM)).astype(np.float32) * 0.3
    )
    return {"old": old, "new": new, "queries_new": queries_new}


def _probe(spaces, ground_truth, **kwargs):  # type: ignore[no-untyped-def]
    n_queries = ground_truth.query_indices.size
    return run_probe(
        old_doc_vectors=spaces["old"],
        new_doc_vectors=spaces["new"],
        fit_indices=np.arange(N_QUERIES, N_DOCS),
        ground_truth=ground_truth,
        old_query_vectors=spaces["old"][:n_queries],
        new_query_vectors=(
            spaces["queries_new"] if ground_truth.tier == "t1" else spaces["new"][:n_queries]
        ),
        k=10,
        methods=["procrustes"],
        with_csls=False,
        cascade_k=200,
        **kwargs,
    )


class TestWhoseQueriesCanPriceACache:
    """The gate that keeps the lower bound a bound on something.

    The overlap between candidate sets prices a cache only when the queries it
    was counted over are a sample of somebody's traffic. Held-out documents and
    synthesised questions produce candidate sets too, and their overlap
    describes the sampling scheme.
    """

    def test_held_out_documents_do_not_price_it(self, spaces) -> None:  # type: ignore[no-untyped-def]
        query_indices = np.arange(N_QUERIES)
        truth = build_tier0(spaces["new"], spaces["new"][query_indices], query_indices, k=10)

        result = _probe(spaces, truth)

        assert result.best.cascade_arr is not None
        assert result.best.candidate_reuse is None
        assert result.decision.arrangement == "single_stage"

    def test_a_real_query_log_prices_it(self, spaces) -> None:  # type: ignore[no-untyped-def]
        qrels = [{int(i)} for i in range(N_QUERIES)]
        truth = build_tier1(spaces["new"], spaces["queries_new"], qrels, k=10)

        result = _probe(spaces, truth, can_read_text=True)

        assert result.best.candidate_reuse is not None
        assert 0.0 <= result.best.candidate_reuse <= 1.0
        assert result.decision.candidate_reuse == result.best.candidate_reuse
        assert result.decision.cascade_n == 200

    def test_the_count_never_exceeds_what_a_perfect_cache_could_give(
        self,
        spaces,  # type: ignore[no-untyped-def]
    ) -> None:
        """A lower bound is only useful if it is one. The ceiling on this run is
        the share of each candidate set beyond the first distinct document."""
        qrels = [{int(i)} for i in range(N_QUERIES)]
        truth = build_tier1(spaces["new"], spaces["queries_new"], qrels, k=10)

        result = _probe(spaces, truth, can_read_text=True)

        assert result.best.candidate_reuse is not None
        assert result.best.candidate_reuse <= 1 - 1 / N_QUERIES + 1e-9
