"""The comparisons that outrank the ARR bands.

ARR is measured against the oracle — what a full reindex would retrieve. But
the alternative a user actually has is *keeping the current model*, and those
are not the same question. An adapter can recover 0.91 of what a reindex would
give and still retrieve worse than changing nothing.

That is not hypothetical. The first run of this package against a real corpus
(BEIR/scifact, MiniLM to bge-small, 300 human-judged queries) produced exactly
it: ARR 0.908, old model 0.940, and a recommendation to migrate. These tests
exist so that cannot come back.
"""

from __future__ import annotations

import pytest

from rebasis.probe.decision import BORDERLINE_BAND, UPGRADE_GAIN_FLOOR, decide

pytestmark = pytest.mark.unit

#: The numbers from the real scifact run, kept as a named case rather than
#: inlined: they are evidence, not an arbitrary fixture.
SCIFACT_BRIDGED = 0.908
SCIFACT_OLD_MODEL = 0.940
SCIFACT_UPGRADE_GAIN = 1.064


class TestBridgingBelowTheCurrentModel:
    def test_the_real_case_no_longer_recommends_migrating(self) -> None:
        """The regression this whole module exists for."""
        result = decide(
            SCIFACT_BRIDGED,
            upgrade_gain=SCIFACT_UPGRADE_GAIN,
            old_model_arr=SCIFACT_OLD_MODEL,
        )

        assert result.decision != "bridge_and_migrate"
        assert result.decision == "full_reindex"

    def test_it_says_why_in_terms_of_the_alternative(self) -> None:
        """A user told to reindex deserves the comparison that decided it."""
        result = decide(
            SCIFACT_BRIDGED,
            upgrade_gain=SCIFACT_UPGRADE_GAIN,
            old_model_arr=SCIFACT_OLD_MODEL,
        )

        assert "0.940" in " ".join(result.warnings)
        assert "current model" in result.rationale

    def test_a_worthless_upgrade_that_also_bridges_badly_is_not_worth_reindexing(self) -> None:
        """Reindexing is only the answer when the new model is actually better.

        Below the upgrade floor, a reindex would spend hours to arrive at a
        model that is not an improvement.
        """
        result = decide(0.60, upgrade_gain=1.00, old_model_arr=0.95)

        assert result.decision == "no_upgrade_needed"

    def test_a_loss_inside_the_noise_band_warns_but_does_not_overrule(self) -> None:
        """A difference smaller than the measurement cannot carry a decision.

        It can still be said out loud: a user about to spend an afternoon
        migrating should know the gain was not measurable.
        """
        result = decide(0.96, old_model_arr=0.96 + BORDERLINE_BAND / 2)

        assert result.decision == "bridge_sufficient"
        assert any("not measurably better" in w for w in result.warnings)

    def test_bridging_ahead_of_the_current_model_is_unaffected(self) -> None:
        """The ordinary case has to keep behaving ordinarily."""
        result = decide(0.97, upgrade_gain=1.20, old_model_arr=0.80)

        assert result.decision == "bridge_sufficient"
        assert not any("current model" in w for w in result.warnings)

    def test_without_the_baseline_the_rule_is_the_old_one(self) -> None:
        """T0 runs may not have the comparison; they must not be penalised."""
        result = decide(0.908, upgrade_gain=1.064)

        assert result.decision == "bridge_and_migrate"

    def test_the_upgrade_question_still_comes_first(self) -> None:
        """A model that is not better is answered before anything else."""
        result = decide(0.99, upgrade_gain=UPGRADE_GAIN_FLOOR - 0.01, old_model_arr=0.10)

        assert result.decision == "no_upgrade_needed"


class TestBreakEven:
    """`bridge_advantage` — the number a user is actually deciding on.

    Measured across 14 corpus/model pairs, its sign predicted whether bridging
    beat doing nothing in **14 of 14** (`docs/bridge-band.md`). Neither factor
    settles it alone, which is the whole reason it exists as one figure.
    """

    def test_it_is_arr_times_the_upgrade_gain(self) -> None:
        result = decide(0.908, upgrade_gain=1.064, old_model_arr=0.940)

        assert result.bridge_advantage == pytest.approx(0.908 * 1.064)

    def test_below_one_means_bridging_loses_ground(self) -> None:
        """The scifact case: a good adapter against a small upgrade."""
        result = decide(
            SCIFACT_BRIDGED, upgrade_gain=SCIFACT_UPGRADE_GAIN, old_model_arr=SCIFACT_OLD_MODEL
        )

        assert result.bridge_advantage is not None
        assert result.bridge_advantage < 1.0
        assert result.decision == "full_reindex"

    def test_above_one_means_bridging_gains_ground(self) -> None:
        """potion to bge-base: a mediocre adapter against a large upgrade."""
        result = decide(0.737, upgrade_gain=1.462, old_model_arr=0.684)

        assert result.bridge_advantage is not None
        assert result.bridge_advantage > 1.0
        assert result.decision not in {"full_reindex", "no_upgrade_needed"}

    def test_a_high_arr_does_not_rescue_a_small_upgrade(self) -> None:
        """The trap the single figure exists to expose."""
        result = decide(0.97, upgrade_gain=1.01, old_model_arr=0.99)

        assert result.bridge_advantage is not None
        assert result.bridge_advantage < 1.0

    def test_a_large_upgrade_does_not_rescue_a_poor_adapter(self) -> None:
        result = decide(0.55, upgrade_gain=1.50, old_model_arr=0.70)

        assert result.bridge_advantage is not None
        assert result.bridge_advantage < 1.0

    def test_it_is_absent_without_a_query_log(self) -> None:
        """`upgrade_gain` needs real queries, so the break-even does too. Absent
        is the honest answer; a figure computed from a missing factor is not."""
        assert decide(0.93).bridge_advantage is None

    def test_it_agrees_with_the_baseline_comparison(self) -> None:
        """The rule compares ARR against `old_model_arr`; the figure is the same
        comparison as one number. They must never disagree."""
        for arr, gain, old in (
            (0.908, 1.064, 0.940),
            (0.737, 1.462, 0.684),
            (0.95, 1.20, 0.80),
            (0.60, 1.05, 0.95),
        ):
            result = decide(arr, upgrade_gain=gain, old_model_arr=old)
            assert (result.bridge_advantage > 1.0) == (arr > old)

    def test_it_is_serialised_for_the_audit_record(self) -> None:
        payload = decide(0.908, upgrade_gain=1.064, old_model_arr=0.940).to_dict()

        assert payload["bridge_advantage"] == pytest.approx(0.9661, abs=1e-4)
        assert payload["old_model_arr"] == pytest.approx(0.940)


class TestProvisionalWithoutAnUpgradeEstimate:
    """A run that cannot measure the upgrade must not sound like it decided.

    Measured on six real corpus/model pairs, a run without an upgrade estimate
    recommended bridging in **six of six** cases where bridging actually lost
    ground. The ARR band was accurate every time; the recommendation built on it
    was not, and nothing in the output said so.
    """

    def test_no_upgrade_estimate_makes_the_result_provisional(self) -> None:
        result = decide(0.93)

        assert result.provisional
        assert result.bridge_advantage is None

    def test_it_says_which_measurement_is_missing(self) -> None:
        result = decide(0.93)

        assert any(
            "could not measure whether the new model is better" in w for w in result.warnings
        )

    def test_it_names_both_ways_to_supply_it(self) -> None:
        """A warning that does not say what to do is a warning nobody acts on."""
        joined = " ".join(decide(0.93).warnings)

        assert "--queries" in joined
        assert "--synth-queries" in joined

    def test_the_rationale_is_marked_too(self) -> None:
        """The rationale is the sentence people quote to each other."""
        assert decide(0.93).rationale.startswith("Provisional")

    def test_an_upgrade_estimate_clears_it(self) -> None:
        result = decide(0.93, upgrade_gain=1.3, old_model_arr=0.70)

        assert not result.provisional
        assert not result.rationale.startswith("Provisional")

    def test_the_band_itself_is_still_reported(self) -> None:
        """Provisional is not "unknown". How well the adapter bridges is
        measured and useful; only the decision built on it is withheld."""
        result = decide(0.93)

        assert result.arr_at_k == 0.93
        assert result.decision == "bridge_and_migrate"

    def test_it_is_serialised(self) -> None:
        assert decide(0.93).to_dict()["provisional"] is True


class TestUninformativeSynthesisedEstimate:
    """A synthesised task every model solves cannot separate two models.

    Measured: `--synth-queries lead` on BEIR/scifact produced an oracle recall of
    0.998 and an upgrade gain of 1.002 — both models found the document each
    lead sentence was cut from, every time. A confident answer built on that is
    worse than the honest "cannot tell" it replaced, because it looks like a
    measurement. `keywords` on the same corpus gave oracle recall 0.85-0.97 and
    a gain that varied by pair, which is a real signal.
    """

    def test_a_trivial_task_makes_the_result_provisional(self) -> None:
        result = decide(
            0.93,
            upgrade_gain=1.002,
            old_model_arr=0.93,
            tier="t2",
            estimate_uninformative=True,
        )

        assert result.provisional
        assert result.estimate_uninformative

    def test_it_says_what_to_try_instead(self) -> None:
        result = decide(
            0.93,
            upgrade_gain=1.002,
            old_model_arr=0.93,
            tier="t2",
            estimate_uninformative=True,
        )
        joined = " ".join(result.warnings)

        assert "too easy" in joined
        assert "keywords" in joined

    def test_it_survives_the_short_circuit_paths(self) -> None:
        """A gain below the floor returns early. That path answers a different
        question and is built on the same estimate, so it carries the same
        caveat — it did not, and produced a confident `no_upgrade_needed`."""
        result = decide(
            0.93,
            upgrade_gain=1.002,
            old_model_arr=0.93,
            tier="t2",
            estimate_uninformative=True,
        )

        assert result.decision == "no_upgrade_needed"
        assert result.provisional
        assert result.rationale.startswith("Provisional")
        assert any("too easy" in w for w in result.warnings)

    def test_an_informative_estimate_is_not_provisional(self) -> None:
        result = decide(0.93, upgrade_gain=1.30, old_model_arr=0.72, tier="t2")

        assert not result.provisional
        assert result.bridge_advantage == pytest.approx(0.93 * 1.30)

    def test_the_tier_is_recorded(self) -> None:
        """`audit replay` compares like with like; a T2 answer and a T1 answer
        are not the same measurement."""
        assert decide(0.93, upgrade_gain=1.3, old_model_arr=0.7, tier="t2").tier == "t2"
        assert decide(0.93, upgrade_gain=1.3, old_model_arr=0.7).to_dict()["tier"] == "t1"


class TestTheBreakEvenOverridesTheBands:
    """The bands describe the adapter; the break-even decides whether to use it.

    Measured over 15 real-query runs (StackExchange and financial questions,
    222,680 documents, 5,761 human-judged queries): `bridge_advantage > 1`
    matched the independent nDCG outcome 14 times, the ARR bands alone 10.

    Every disagreement went the same way. A large upgrade bridged imperfectly
    lands at a low ARR — 0.47 to 0.71 in the four runs where bridging genuinely
    helped — which the bands read as `full_reindex` or `caution`. Letting the
    bands decide rejected all four.
    """

    @pytest.mark.parametrize(
        ("corpus", "arr", "gain", "measured"),
        [
            # The four real-query runs where bridging beat the status quo. ARR
            # and gain are the values probe recorded; their product is the
            # advantage it reported, and every one of these was rejected by the
            # bands alone.
            ("programmers", 0.547, 1.958, "+4.2%"),
            ("english", 0.557, 1.888, "+3.8%"),
            ("gaming", 0.728, 1.521, "+7.4%"),
            ("fiqa", 0.510, 2.445, "+16.0%"),
        ],
    )
    def test_a_genuine_win_is_recommended(
        self, corpus: str, arr: float, gain: float, measured: str
    ) -> None:
        result = decide(arr, upgrade_gain=gain, old_model_arr=arr / gain)

        assert result.bridge_advantage is not None
        assert result.bridge_advantage > 1.0, f"{corpus}: {result.bridge_advantage}"
        assert result.decision in {"bridge_sufficient", "bridge_and_migrate"}, (
            f"{corpus}: ARR {arr} bands as {result.decision}, but bridging measured {measured}"
        )

    def test_a_low_arr_does_not_veto_a_measured_win(self) -> None:
        """The exact disagreement: ARR 0.557 bands as `full_reindex`, while the
        break-even says bridging beats doing nothing — and it measured +3.8%."""
        banded_alone = decide(0.557)
        with_the_upgrade = decide(0.557, upgrade_gain=1.888, old_model_arr=0.295)

        assert banded_alone.decision == "full_reindex"
        assert with_the_upgrade.decision == "bridge_and_migrate"

    def test_the_bands_still_choose_between_the_two_bridging_answers(self) -> None:
        """Whether to bridge is settled; how much a reindex would still add is
        what separates `bridge_sufficient` from `bridge_and_migrate`."""
        low = decide(0.60, upgrade_gain=1.90, old_model_arr=0.55)
        high = decide(0.97, upgrade_gain=1.20, old_model_arr=0.60)

        assert low.decision == "bridge_and_migrate"
        assert high.decision == "bridge_sufficient"

    def test_a_break_even_inside_the_noise_band_decides_nothing(self) -> None:
        """At 0.977 the two options are indistinguishable. The bands describe
        the adapter and the report says they cannot settle the question."""
        result = decide(0.887, upgrade_gain=1.102, old_model_arr=0.90)

        assert result.bridge_advantage == pytest.approx(0.977, abs=1e-3)
        assert any("within measurement noise" in w for w in result.warnings)

    def test_without_an_upgrade_estimate_the_bands_still_decide(self) -> None:
        """T0 has no break-even to consult, so nothing changes there — except
        that the result is marked provisional."""
        result = decide(0.53)

        assert result.decision == "full_reindex"
        assert result.provisional

    def test_an_uninformative_estimate_does_not_override_anything(self) -> None:
        """A trivial synthesised task produces an advantage near 1.0 that means
        nothing. Letting it drive the decision would be worse than the bands."""
        result = decide(
            0.53, upgrade_gain=1.002, old_model_arr=0.53, tier="t2", estimate_uninformative=True
        )

        assert result.provisional


class TestSynthesisedEstimatesCarryTheirError:
    """A synthesised estimate is compared against a threshold it cannot resolve.

    Measured over 60 runs — five corpora, three strategies, four held-out sizes
    — the mean absolute error of a synthesised `upgrade_gain` against a real
    query log's figure is 0.14 (`lead`), 0.17 (`title`) and 0.22 (`keywords`).
    The break-even threshold is 1.0. An estimate within that error of the line
    could fall on either side of it.

    The guard uses the **worst** of the three, because one set to the best would
    pass estimates the other strategies cannot support.
    """

    def test_an_estimate_near_the_threshold_is_provisional(self) -> None:
        result = decide(0.75, upgrade_gain=1.05, old_model_arr=0.60, tier="t2")

        assert result.provisional
        assert any("too close to the break-even" in w for w in result.warnings)

    def test_an_estimate_far_above_it_settles(self) -> None:
        result = decide(0.75, upgrade_gain=1.60, old_model_arr=0.47, tier="t2")

        assert not result.provisional

    def test_an_estimate_far_below_it_settles(self) -> None:
        result = decide(0.75, upgrade_gain=0.70, old_model_arr=1.07, tier="t2")

        assert not result.provisional

    def test_a_real_query_log_is_not_penalised(self) -> None:
        """T1 measures the gain rather than estimating it, so the error bar that
        makes a synthesised figure unusable does not apply."""
        result = decide(0.75, upgrade_gain=1.05, old_model_arr=0.60, tier="t1")

        assert not result.provisional

    def test_the_warning_quotes_both_numbers(self) -> None:
        """A user told the estimate is unusable needs to see how unusable."""
        joined = " ".join(decide(0.75, upgrade_gain=1.05, old_model_arr=0.60, tier="t2").warnings)

        assert "1.05" in joined
        assert "0.22" in joined
