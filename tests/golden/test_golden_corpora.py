"""Golden tests: the whole pipeline on real embeddings of a real corpus.

Everything else in this suite runs on synthetic vectors with a planted rotation.
That proves the mathematics and cannot prove the two things that actually break
in practice: that a model's prefix is applied correctly, and that the decision
rule reaches the right *answer* rather than a plausible number.

**Numerical differences are acceptable; a different decision is not.**
So the bands below are wide and the decisions are exact. A band exists to absorb
a legitimate small change in an adapter; a decision is what the user acts on.

Every expected value here was **measured**, not predicted, on BEIR/scifact with
5,000 documents and 295 human-judged queries. Where the measurement disagreed
with what was expected, the measurement is what is written down and the
disagreement is recorded in `docs/golden-findings.md`.
"""

from __future__ import annotations

from typing import Any

import pytest

from rebasis.embed import PrecomputedEmbedder
from rebasis.probe.session import QueryLog, probe_store
from rebasis.store import MemoryStore
from rebasis.types import EncodingProfile

pytestmark = [pytest.mark.slow]

#: How far a measured ARR may move before it counts as a regression. Wide
#: enough that a legitimate refinement to an adapter does not fail the build,
#: narrow enough that a broken one does. The decisions are asserted exactly.
BAND = 0.04

#: Share of the available headroom an adapter must recover, measured as
#: ``(arr - unadapted) / (1 - unadapted)``. Across the three comparable
#: scenarios this sits between 0.81 and 0.89, so 0.70 is a floor rather than a
#: fitted threshold.
GAP_CLOSED_FLOOR = 0.70

#: Measured on the project's host. `t0` and `t1` differ because they answer
#: different questions, not because one is wrong.
EXPECTED: dict[str, dict[str, Any]] = {
    "same_family": {
        "t0": {"arr": 0.942, "decision": "bridge_and_migrate"},
        # L12 is not better than L6 on this corpus: gain 0.993, below the floor.
        "t1": {"arr": 0.946, "decision": "no_upgrade_needed"},
        "unadapted_below": 0.80,
    },
    "different_family": {
        "t0": {"arr": 0.911, "decision": "bridge_and_migrate"},
        # bge IS better here (gain 1.060), but bridging retrieves 0.903 against
        # 0.944 for simply keeping MiniLM — so only a reindex delivers the gain.
        "t1": {"arr": 0.903, "decision": "full_reindex"},
        "unadapted_below": 0.40,
    },
    "prefix_trap": {
        "t0": {"arr": 0.870, "decision": "bridge_and_migrate"},
        "t1": {"arr": 0.921, "decision": "no_upgrade_needed"},
        "unadapted_below": 0.30,
    },
    "large_upgrade": {
        # The path ADR 9 changed. ARR 0.82 bands as `caution`; the break-even
        # (0.824 x 1.32 = 1.087) says bridging beats doing nothing, and the
        # decision follows the break-even. Nothing else in this suite exercises
        # it — every other scenario's advantage sits inside the noise band.
        "t0": {"arr": 0.825, "decision": "caution"},
        "t1": {"arr": 0.824, "decision": "bridge_and_migrate"},
        "unadapted_below": None,
    },
    "hard_drift": {
        "t0": {"arr": 0.762, "decision": "caution"},
        # The highest T1 ARR of the four, against the *weakest* oracle: potion
        # retrieves worse than MiniLM (gain 0.845), so a high ratio against it
        # means little. The decision is what carries the answer.
        "t1": {"arr": 0.971, "decision": "no_upgrade_needed"},
        # Not measurable: 384 to 256 means the new vector cannot enter the old
        # index at all.
        "unadapted_below": None,
    },
}


def build(scenario: dict[str, Any]) -> tuple[MemoryStore, PrecomputedEmbedder, Any, QueryLog]:
    """A store holding the old model's vectors, plus both embedders.

    The store is the real index: the old model's actual embeddings of actual
    documents. Nothing here is synthetic except the absence of a model download.
    """
    spec, texts, queries = scenario["spec"], scenario["texts"], scenario["queries"]
    store = MemoryStore(scenario["ids"], scenario["old_documents"], texts)

    new_embedder = PrecomputedEmbedder(
        EncodingProfile(model_id=spec["new"], dim=int(spec["new_dim"])),
        dict(zip(texts, scenario["new_documents"], strict=True)),
        query_vectors=dict(zip(queries, scenario["new_queries"], strict=True)),
    )
    old_embedder = PrecomputedEmbedder(
        EncodingProfile(model_id=spec["old"], dim=int(spec["old_dim"])),
        dict(zip(texts, scenario["old_documents"], strict=True)),
        query_vectors=dict(zip(queries, scenario["old_queries"], strict=True)),
    )
    log = QueryLog(queries=queries, qrels=[set(r) for r in scenario["relevant"]])
    return store, new_embedder, old_embedder, log


@pytest.fixture(params=sorted(EXPECTED), ids=lambda n: n)
def scenario(request: pytest.FixtureRequest, load_scenario: Any) -> dict[str, Any]:
    loaded = load_scenario(request.param)
    loaded["name"] = request.param
    return loaded


@pytest.fixture
def t0(scenario: dict[str, Any]) -> Any:
    store, new_embedder, _, _ = build(scenario)
    result, _ = probe_store(store, new_embedder, size=5000, heldout=1000, seed=0)
    return result


@pytest.fixture
def t1(scenario: dict[str, Any]) -> Any:
    store, new_embedder, old_embedder, log = build(scenario)
    result, _ = probe_store(
        store,
        new_embedder,
        old_embedder=old_embedder,
        query_log=log,
        size=5000,
        heldout=1000,
        seed=0,
    )
    return result


class TestDecisions:
    """What the user acts on. Exact, not banded."""

    def test_the_t0_decision_is_stable(self, scenario: dict[str, Any], t0: Any) -> None:
        assert t0.decision.decision == EXPECTED[scenario["name"]]["t0"]["decision"]

    def test_the_t1_decision_is_stable(self, scenario: dict[str, Any], t1: Any) -> None:
        """The tier that matters: real queries with human judgements."""
        assert t1.decision.decision == EXPECTED[scenario["name"]]["t1"]["decision"]


class TestBands:
    """The numbers behind the decision, within a band that absorbs refinement."""

    def test_t0_arr_is_within_band(self, scenario: dict[str, Any], t0: Any) -> None:
        expected = EXPECTED[scenario["name"]]["t0"]["arr"]
        assert t0.best.arr == pytest.approx(expected, abs=BAND)

    def test_t1_arr_is_within_band(self, scenario: dict[str, Any], t1: Any) -> None:
        expected = EXPECTED[scenario["name"]]["t1"]["arr"]
        assert t1.best.arr == pytest.approx(expected, abs=BAND)

    def test_the_adapter_closes_most_of_the_gap_it_could(
        self, scenario: dict[str, Any], t0: Any
    ) -> None:
        """The premise of the whole tool, stated as a fraction rather than a factor.

        How much better than nothing an adapter *can* be depends on how far
        apart the two spaces are: between MiniLM-L6 and L12 the unadapted
        vectors already retrieve 0.694, so "1.5x better" is arithmetically
        impossible. What is comparable across scenarios is the share of the
        available headroom the adapter recovers — and that stays high whether
        the spaces start 0.69 apart or 0.08 apart.
        """
        ceiling = EXPECTED[scenario["name"]]["unadapted_below"]
        if ceiling is None:
            # Different dimensions: the new vector cannot enter the index at all,
            # so there is no unadapted baseline to compare against.
            assert "unadapted" not in t0.baselines
            return

        unadapted = t0.baselines["unadapted"]
        assert unadapted < ceiling
        closed = (t0.best.arr - unadapted) / (1.0 - unadapted)
        assert closed > GAP_CLOSED_FLOOR, (
            f"the adapter closed {closed:.0%} of the gap between {unadapted:.3f} "
            f"and a full reindex; below {GAP_CLOSED_FLOOR:.0%} it is not earning its place"
        )


class TestTheIntervalIsHonest:
    """The defect the first real-corpus run exposed.

    ARR is a ratio of means; its interval has to be an interval for that ratio.
    Bootstrapping only the numerator put the point estimate outside its own
    range — visible only at T1, where the oracle is imperfect.
    """

    def test_the_point_estimate_lies_inside_its_interval_at_t1(self, t1: Any) -> None:
        low, high = t1.best.arr_ci
        assert low <= t1.best.arr <= high, (
            f"ARR {t1.best.arr:.4f} outside its own interval [{low:.4f}, {high:.4f}]"
        )

    def test_the_point_estimate_lies_inside_its_interval_at_t0(self, t0: Any) -> None:
        low, high = t0.best.arr_ci
        assert low <= t0.best.arr <= high


class TestTheBreakEvenPath:
    """The scenario ADR 9 exists for.

    A weak old model and a strong new one: the upgrade is large, the adapter
    bridges it imperfectly, and ARR lands in a band that reads as "do not
    bridge" while the break-even says the opposite. Measured, the break-even is
    right — 14 of 15 real-query runs against the bands' 10.
    """

    def test_the_break_even_overrides_the_band(self, scenario, t1) -> None:  # type: ignore[no-untyped-def]
        if scenario["name"] != "large_upgrade":
            pytest.skip("only the large-upgrade scenario reaches this path")

        assert t1.best.arr < 0.85, "this scenario is meant to band as caution or worse"
        assert t1.decision.bridge_advantage is not None
        assert t1.decision.bridge_advantage > 1.0
        assert t1.decision.decision == "bridge_and_migrate"

    def test_without_the_upgrade_the_band_decides(self, scenario, t0) -> None:  # type: ignore[no-untyped-def]
        """T0 has no break-even to consult, so the band stands — and the result
        says it is provisional rather than presenting it as an answer."""
        if scenario["name"] != "large_upgrade":
            pytest.skip("only the large-upgrade scenario reaches this path")

        assert t0.decision.decision == "caution"
        assert t0.decision.provisional


class TestTheUpgradeQuestion:
    """`no_upgrade_needed` is only reachable with a real query log."""

    def test_upgrade_gain_is_measured_at_t1(self, t1: Any) -> None:
        assert t1.decision.upgrade_gain is not None

    def test_upgrade_gain_is_not_measurable_at_t0(self, t0: Any) -> None:
        """At T0 the ground truth is the new model's own output, so it would
        score perfectly against itself."""
        assert t0.decision.upgrade_gain is None
