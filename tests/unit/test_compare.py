"""Ranking several candidates on one corpus.

The property tested hardest is the one a comparison cannot survive losing:
**every candidate is scored on the same sample, the same split and the same
queries**. Redraw per candidate and the rows stop being comparable — and the
failure is invisible, because each row on its own still looks like a valid
measurement.

The rest is what the command has to say honestly rather than what it computes:
the reference is the index's own model and not a row, a candidate with no
upgrade estimate is ordered last rather than as a low score, and the tiered
round eliminates on the same +-0.025 band the decision rule already reports its
own borderline cases at.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from rebasis.core import l2_normalize
from rebasis.embed import PrecomputedEmbedder
from rebasis.probe.comparison import (
    CandidateComparison,
    ComparisonResult,
    compare_models,
    remote_candidates,
)
from rebasis.probe.decision import BORDERLINE_BAND
from rebasis.store import MemoryStore
from rebasis.types import EncodingProfile

if TYPE_CHECKING:
    from rebasis.probe.session import CorpusSample

pytestmark = pytest.mark.unit

DIM = 24
#: Large enough that a sample can hold `sample.MIN_SAMPLE` fit records and a
#: held-out query set beside them; below that the draw refuses, correctly.
N = 1200


def _corpus(rng: np.random.Generator) -> tuple[list[str], list[str], Any]:
    centers = (rng.standard_normal((12, DIM)) * 3.0).astype(np.float32)
    assignment = rng.integers(0, 12, size=N)
    vectors = l2_normalize(
        centers[assignment] + rng.standard_normal((N, DIM)).astype(np.float32) * 1.2
    )
    ids = [f"doc-{i}" for i in range(N)]
    texts = [f"document number {i}" for i in range(N)]
    return ids, texts, vectors


def _candidate(texts: list[str], vectors: Any, rng: np.random.Generator, noise: float) -> Any:
    """A model that rotates the old space and adds noise — a candidate."""
    rotation = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype(np.float32)
    moved = l2_normalize(
        vectors @ rotation.T + rng.standard_normal(vectors.shape).astype(np.float32) * noise
    )
    profile = EncodingProfile(model_id=f"candidate/noise-{noise}", dim=DIM)
    return PrecomputedEmbedder(profile, dict(zip(texts, moved, strict=True)))


@pytest.fixture
def world(rng: np.random.Generator):  # type: ignore[no-untyped-def]
    ids, texts, vectors = _corpus(rng)
    store = MemoryStore(ids, vectors, texts)
    candidates = {
        "candidate/a": _candidate(texts, vectors, rng, 0.10),
        "candidate/b": _candidate(texts, vectors, rng, 0.35),
    }
    return {"store": store, "candidates": candidates, "texts": texts, "vectors": vectors}


def _compare(world: dict[str, Any], **kwargs: Any) -> ComparisonResult:
    return compare_models(
        world["store"],
        world["candidates"],
        size=600,
        heldout=120,
        k=10,
        seed=7,
        old_model="incumbent/model",
        **kwargs,
    )


class TestOneSampleForEveryCandidate:
    """The guarantee the whole comparison rests on."""

    def test_the_sample_is_drawn_once(self, world: dict[str, Any], monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """Two candidates, one draw. A second draw would be a second corpus, and
        the rows would then differ by which documents each one happened to see."""
        from rebasis.probe import comparison

        draws: list[CorpusSample] = []
        original = comparison.draw_corpus_sample

        def counted(*args: Any, **kwargs: Any) -> Any:
            drawn = original(*args, **kwargs)
            draws.append(drawn)
            return drawn

        monkeypatch.setattr(comparison, "draw_corpus_sample", counted)
        _compare(world)

        assert len(draws) == 1

    def test_every_candidate_is_scored_on_the_same_documents(self, world: dict[str, Any]) -> None:
        """Not merely the same size — the same rows, in the same split."""
        result = _compare(world)

        assert len({c.result.n_documents for c in result.candidates}) == 1
        assert len({c.result.n_fit_pairs for c in result.candidates}) == 1
        assert len({c.result.n_queries for c in result.candidates}) == 1
        assert len({c.result.seed for c in result.candidates}) == 1


class TestTheOrdering:
    def test_the_better_candidate_ranks_first(self, world: dict[str, Any]) -> None:
        """`candidate/a` sees a tenth of the noise `candidate/b` does, so it
        retains more of the old space's neighbourhoods and must come first."""
        result = _compare(world)

        assert result.ranked()[0].model == "candidate/a"

    def test_a_candidate_with_no_estimate_sorts_last(self) -> None:
        """The absence of a measurement is not a low score, and putting it in
        the middle of an order would read as one."""
        scored = _fake("scored", upgrade_gain=1.2)
        unscored = _fake("unscored", upgrade_gain=None)
        result = ComparisonResult(candidates=[unscored, scored], reference={}, sample={})

        assert [c.model for c in result.ranked()] == ["scored", "unscored"]

    def test_every_row_carries_the_ranking_caveat(self, world: dict[str, Any]) -> None:
        """`compare` makes a stronger claim than `probe`, on weaker-shaped
        evidence: a rank correlation rather than an accuracy. The table is not
        publishable without the sentence that says so."""
        result = _compare(world)

        caveat = result.ranking_caveat.lower()

        assert "spearman" in caveat
        # The measurement that scored *this command* lost to the published
        # leaderboard, and the caveat has to say so — quoting a related
        # measurement instead is how a table becomes a leaderboard.
        assert "mteb" in caveat
        assert "did not beat" in caveat
        assert result.to_dict()["ranking_caveat"] == result.ranking_caveat


class TestTheReference:
    def test_the_index_model_is_the_reference_and_not_a_row(self, world: dict[str, Any]) -> None:
        result = _compare(world)

        assert result.reference["model"] == "incumbent/model"
        assert result.reference["source"] == "index"
        assert "incumbent/model" not in {c.model for c in result.candidates}


class TestTieredElimination:
    def test_it_carries_through_what_the_small_round_could_not_separate(
        self, world: dict[str, Any]
    ) -> None:
        """Every candidate is still reported; the eliminated ones say which
        round measured them, because a smaller sample is a different precision
        and a row that hid that would be comparing two things."""
        result = _compare(world, tiered=True, first_round=400)

        assert {c.model for c in result.candidates} == set(world["candidates"])
        assert result.sample["tiered"] is True
        for candidate in result.candidates:
            assert candidate.round_number in {1, 2}
            assert candidate.eliminated == (candidate.round_number == 1)

    def test_survivors_are_within_the_band_of_the_leader(self) -> None:
        """The same +-0.025 the decision rule reports its own borderline cases
        at: below it, two numbers from one sample are not two numbers."""
        from rebasis.probe.comparison import _survivors

        leader = _fake("leader", upgrade_gain=1.40)
        close = _fake("close", upgrade_gain=1.40 - BORDERLINE_BAND / 2)
        far = _fake("far", upgrade_gain=1.10)

        assert _survivors([leader, close, far]) == {"leader", "close"}

    def test_a_candidate_the_round_could_not_score_is_not_a_candidate_it_rejected(
        self,
    ) -> None:
        unscored = _fake("unscored", upgrade_gain=None)
        leader = _fake("leader", upgrade_gain=1.40)

        from rebasis.probe.comparison import _survivors

        assert "unscored" in _survivors([leader, unscored])


class TestWhatLeavesTheMachine:
    def test_a_local_backend_is_not_reported_as_remote(self, world: dict[str, Any]) -> None:
        assert remote_candidates(world["candidates"]) == []

    def test_the_backend_decides_rather_than_the_model_id(self) -> None:
        """The same name can be a local checkpoint or a hosted endpoint, so the
        module the embedder came from is what is asked."""

        class _Hosted:
            __module__ = "rebasis.embed.backends.openai_compat"

        assert remote_candidates({"gpt/whatever": _Hosted()}) == ["gpt/whatever"]  # type: ignore[dict-item]


def _fake(model: str, *, upgrade_gain: float | None) -> CandidateComparison:
    """A comparison row with only the field under test filled in."""
    from rebasis.probe.decision import DecisionResult
    from rebasis.probe.runner import CandidateMetrics, ProbeResult

    metrics = CandidateMetrics(
        name="procrustes",
        arr=0.9,
        arr_sparse=0.9,
        arr_ci=(0.85, 0.95),
        mrr=0.8,
        overlap=0.8,
        spearman=0.8,
        score_shift_raw=0.0,
    )
    decision = DecisionResult(
        decision="bridge_sufficient",
        arr_at_k=0.9,
        borderline=False,
        nearest_threshold=0.95,
        distance_to_threshold=0.05,
        upgrade_gain=upgrade_gain,
    )
    result = ProbeResult(
        decision=decision,
        best=metrics,
        candidates=[metrics],
        baselines={},
        ground_truth_tier="t1",
        n_documents=1,
        n_queries=1,
        n_fit_pairs=1,
        k=10,
        seed=0,
        duration_seconds=0.0,
    )
    return CandidateComparison(model=model, result=result, duration_seconds=0.0)
