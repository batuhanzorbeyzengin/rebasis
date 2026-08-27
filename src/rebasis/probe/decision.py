"""The decision rule.

``probe`` does not output a number, it outputs a **decision** — and that decision
has to remain defensible six months later, which is what the audit record is for.
Three M0 findings changed the four-band rule as first written:

**A fifth outcome: "no upgrade needed."** All four bands assume the upgrade is
going ahead. Across four corpora, staying on the old model retained a mean ARR of
**0.983**, with an upper bound of **1.238** — on some corpora the old model is
simply better. The first thing a user needs to know is whether the new model is an
improvement on *their* corpus at all; if it is not, neither bridging nor
reindexing is worth doing (section 12 of ``docs/m0-findings.md``).

**The borderline band widens from ±0.005 to ±0.025.** The band exists so that one
measurement gives the same decision on any device, and ±0.005 was the original
figure; the measured bootstrap 95% CI half-width for ARR is **±0.024** on ~1,000
held-out queries and **±0.042** with real queries. A ±0.005 band would declare
confidence the measurement cannot support.

**ARR is reported with its interval.** Point estimates invite over-reading. When
the interval straddles a threshold, the report says so rather than picking a side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rebasis.probe.groundtruth import SYNTH_GAIN_ERROR
from rebasis.types import Decision  # noqa: TC001 - used at runtime in DECISION_BANDS

__all__ = [
    "BORDERLINE_BAND",
    "DECISION_BANDS",
    "DecisionResult",
    "decide",
]

#: (threshold, decision) in descending order.
DECISION_BANDS: tuple[tuple[float, Decision], ...] = (
    (0.95, "bridge_sufficient"),
    (0.85, "bridge_and_migrate"),
    (0.70, "caution"),
    (0.0, "full_reindex"),
)

#: Distance from a threshold within which a result is reported as borderline.
#:
#: ±0.005 was the original figure. M0 measured the ARR confidence interval at
#: ±0.024 (T0) and ±0.042 (real queries), so ±0.005 would claim a precision the
#: measurement does not have. ±0.025 matches what can actually be resolved.
BORDERLINE_BAND = 0.025

#: Below this, the new model is not enough of an improvement to justify any
#: migration work. Expressed as a ratio of oracle recall to old-model recall.
UPGRADE_GAIN_FLOOR = 1.02

#: A gap this large between overall ARR and the sparsest clusters means drift
#: is heterogeneous: one global adapter is leaving quality on the table in
#: part of the corpus.
TAIL_GAP_LIMIT = 0.1

#: `score_shift` above this warns that fixed-threshold filters will break.
#: Evaluated AFTER calibration; before it, the check fires every time.
SCORE_SHIFT_LIMIT = 0.1


@dataclass(slots=True)
class DecisionResult:
    """A decision together with everything needed to defend it later."""

    decision: Decision
    arr_at_k: float
    borderline: bool
    nearest_threshold: float
    distance_to_threshold: float
    confidence_interval: tuple[float, float] | None = None
    interval_straddles_threshold: bool = False
    upgrade_gain: float | None = None
    score_shift: float | None = None
    #: What the **current** model retrieves, on ARR's scale. The alternative
    #: the user actually has.
    old_model_arr: float | None = None
    #: Set when the upgrade could not be measured. The ARR band is real; the
    #: recommendation built on it is not yet an answer.
    provisional: bool = False
    #: Which ground truth produced this.
    tier: str = "t1"
    #: The upgrade estimate exists but separates nothing.
    estimate_uninformative: bool = False
    #: The break-even measured on nDCG rather than recall. Reported, and used
    #: to warn when the two metrics disagree — never to override the decision,
    #: which the bands were calibrated against on recall.
    ndcg_advantage: float | None = None
    #: Retention when the bridge feeds a rerank instead of producing the final
    #: ranking. Reported, never decisive: it describes an arrangement the tool
    #: measures and does not yet serve.
    cascade_arr: float | None = None
    warnings: list[str] = field(default_factory=list)
    rationale: str = ""

    @property
    def cascade_advantage(self) -> float | None:
        """The break-even for a **two-stage** arrangement.

        The same product as :attr:`bridge_advantage`, with retention measured at
        candidate-set depth instead of at ``k``. That is the right retention for
        an arrangement where the bridge produces candidates and the new model
        ranks them in its own space: what can be lost is a relevant document
        that never reached the candidate set, and nothing after that.

        Systematically higher than ``bridge_advantage``, and measured to be
        higher in a way that changes the answer — 1 of 48 runs against 36 of 48
        (`docs/cascade-band.md`).

        Reported and **not** acted on, which is a narrower statement than it was.
        :class:`~rebasis.serve.cascade.Cascade` serves this arrangement, so the
        objection is no longer that the tool cannot do it. It is that the
        decision rule weighs quality against cost, and this arrangement's cost
        depends on a **query distribution** — how often a candidate is already
        cached — which is a property of a running system and not of the corpus a
        probe can see. ``bridge_advantage`` costs a matrix multiply whatever the
        traffic. So the rule runs on the number it can price, and this one is
        reported for a reader who can price the other.
        """
        if self.upgrade_gain is None or self.cascade_arr is None:
            return None
        return self.cascade_arr * self.upgrade_gain

    @property
    def bridge_advantage(self) -> float | None:
        """Bridged quality divided by the current model's — the break-even.

        ``ARR x upgrade_gain``. Above 1.0 bridging retrieves better than
        changing nothing; below it, worse. Neither factor answers it alone: a
        high ARR against a small upgrade still loses, and so does a large upgrade
        bridged by a poor adapter. It is the comparison the decision rule makes
        against ``old_model_arr``, stated as one figure because two numbers a
        reader has to divide in their head is not a number they will use.

        **The count this docstring used to quote was an identity.** Read off one
        run's own scores, ``ARR x upgrade_gain`` is ``(bridged / reindex) x
        (reindex / status quo)``, which is ``bridged / status quo`` — the same
        inequality as "did bridging beat doing nothing". Scored that way it
        agrees always, and "14 of 14" measured nothing.

        What is measured, over the 57 runs `reports/band/` still holds: the
        estimate ranks runs by the margin they actually returned at **Spearman
        rho = 0.60, p ~ 1e-6**, so it carries real information about the *size*
        of the effect. Scored as a threshold against the outcome it agrees in
        37 of 57, below the 54 of 57 a rule that always answered "do not bridge"
        would score — because 95% of those outcomes fall on one side, and an
        accuracy cannot separate a real rule from a constant there
        (`docs/bridge-band.md`, section 9).

        The quantity is still what decides, and the bands are still where they
        were: what changed is the confidence a reader should take from the
        count, not the rule.
        """
        if self.upgrade_gain is None:
            return None
        return self.arr_at_k * self.upgrade_gain

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form, for the report and the audit record."""
        return {
            "decision": self.decision,
            "arr_r10": round(self.arr_at_k, 4),
            "borderline": self.borderline,
            "nearest_threshold": self.nearest_threshold,
            "distance_to_threshold": round(self.distance_to_threshold, 4),
            "confidence_interval": (
                [round(v, 4) for v in self.confidence_interval]
                if self.confidence_interval
                else None
            ),
            "interval_straddles_threshold": self.interval_straddles_threshold,
            "provisional": self.provisional,
            "tier": self.tier,
            "estimate_uninformative": self.estimate_uninformative,
            "ndcg_advantage": (
                round(self.ndcg_advantage, 4) if self.ndcg_advantage is not None else None
            ),
            "upgrade_gain": round(self.upgrade_gain, 4) if self.upgrade_gain else None,
            "old_model_arr": round(self.old_model_arr, 4) if self.old_model_arr else None,
            "bridge_advantage": (
                round(self.bridge_advantage, 4) if self.bridge_advantage is not None else None
            ),
            "cascade_arr": round(self.cascade_arr, 4) if self.cascade_arr is not None else None,
            "cascade_advantage": (
                round(self.cascade_advantage, 4) if self.cascade_advantage is not None else None
            ),
            "score_shift": round(self.score_shift, 4) if self.score_shift else None,
            "warnings": list(self.warnings),
            "rationale": self.rationale,
        }


_RATIONALE: dict[Decision, str] = {
    "no_upgrade_needed": (
        "The new model does not retrieve measurably better than the current one on "
        "this corpus, so neither bridging nor reindexing would buy anything."
    ),
    "bridge_sufficient": (
        "The adapter recovers essentially everything a full reindex would give. "
        "Use the new model now and leave the index alone."
    ),
    "bridge_and_migrate": (
        "The adapter recovers most of the gap. Bridge today, then migrate "
        "gradually in the background; quality only improves as records move."
    ),
    "caution": (
        "The adapter closes part of the gap but leaves a real loss. Check tail_arr "
        "in the report: if drift is concentrated in a few clusters, per-cluster "
        "adapters may help where one global adapter cannot."
    ),
    "full_reindex": (
        "The drift is too large for an adapter to bridge. Reindexing is the "
        "honest answer; the report shows what bridging would and would not recover."
    ),
}


#: Used in place of the ``full_reindex`` rationale when the reason for
#: reindexing is specifically that bridging lost to doing nothing. Not a
#: :data:`~rebasis.types.Decision` — the decision is still ``full_reindex``,
#: only the explanation differs.
_BRIDGING_BELOW_CURRENT = (
    "The new model is better, but the adapter does not carry enough of that "
    "advantage across: bridging retrieves less than keeping the current model. "
    "A full reindex is what would actually deliver the improvement."
)


#: Prefixed to a rationale when the upgrade could not be measured. The band is
#: still worth knowing; the recommendation is not yet one.
_PROVISIONAL_PREFIX = "Provisional, and not yet a recommendation. "

_NO_UPGRADE_ESTIMATE = (
    "This run could not measure whether the new model is better on your corpus, "
    "so it cannot say whether bridging is worth doing — only how well an adapter "
    "bridges. Add --queries with a real query log, or --synth-queries to estimate "
    "it from the documents."
)

_UNSETTLED_SYNTHESIS = (
    "The synthesised queries put the upgrade at {gain:.2f}x, and an estimate "
    "from synthesised queries carries about {error:.2f} of error against what a "
    "real query log gives. That is too close to the break-even to settle the "
    "question either way. A real query log with --queries would."
)

_TRIVIAL_SYNTHESIS = (
    "The synthesised queries were too easy: every model found the document each "
    "query was cut from, so the upgrade estimate cannot tell the two models "
    "apart. Try --synth-queries keywords, which does not hand the answer to the "
    "retriever, or supply a real query log with --queries."
)


def decide(  # noqa: PLR0913 - one keyword argument per measurement the rule uses
    arr_at_k: float,
    *,
    confidence_interval: tuple[float, float] | None = None,
    upgrade_gain: float | None = None,
    score_shift: float | None = None,
    tail_arr: float | None = None,
    old_model_arr: float | None = None,
    tier: str = "t1",
    estimate_uninformative: bool = False,
    ndcg_advantage: float | None = None,
    cascade_arr: float | None = None,
) -> DecisionResult:
    """Turn measurements into a recommendation.

    Args:
        arr_at_k: The main decision metric.
        confidence_interval: Bootstrap interval for ``arr_at_k``. Supplying it
            lets the result say when the interval straddles a threshold.
        upgrade_gain: Oracle recall over old-model recall. Below
            :data:`UPGRADE_GAIN_FLOOR` the answer is "do not upgrade", whatever
            ARR says — measurable only with a real query log.
        score_shift: KS distance **after** calibration.
        tail_arr: ARR in the sparsest decile of clusters — the signal for
            heterogeneous drift.
        old_model_arr: What the **current** model retrieves, on the same scale.
            The user's real alternative to bridging is doing nothing, so an
            adapter that scores below this buys nothing however good its ARR
            looks against the oracle.
        tier: Which ground truth produced these numbers. At T0 the upgrade
            cannot be measured at all, and the result is marked provisional
            rather than presented as an answer.
        estimate_uninformative: The upgrade estimate exists but carries no
            information — a synthesised task every model solves cannot separate
            two models. Treated the same as no estimate at all, because a
            confident wrong answer is worse than an honest missing one.
        ndcg_advantage: The same break-even on a rank-sensitive metric. The
            decision runs on recall, which cannot see an adapter that returns
            the same documents in a worse order; when the two disagree the
            report says so rather than picking one silently.
        cascade_arr: Retention at candidate-set depth — what the bridge would
            retain if it fed a rerank instead of producing the final ranking.
            Carried onto the result and reported; it decides nothing, because
            what that arrangement costs depends on a query distribution rather
            than on this corpus. See :attr:`DecisionResult.cascade_advantage`.
    """
    warnings: list[str] = []
    bridging_loss = None if old_model_arr is None else old_model_arr - arr_at_k

    # Without an upgrade estimate the tool can say how well an adapter bridges and
    # not whether bridging is worth doing: on six real corpus/model pairs, T0
    # recommended bridging in 6 of 6 cases where it actually lost ground. A
    # synthesised estimate carries about 0.22 of measured error against a real
    # query log's figure, against a threshold of 1.0 — near that line it settles
    # nothing, whatever the number says.
    unsettled_estimate = (
        tier == "t2" and upgrade_gain is not None and abs(upgrade_gain - 1.0) <= SYNTH_GAIN_ERROR
    )
    provisional = upgrade_gain is None or estimate_uninformative or unsettled_estimate
    warnings.extend(
        _provisional_notes(upgrade_gain, estimate_uninformative, unsettled=unsettled_estimate)
    )

    settled = _settled_before_the_bands(
        arr_at_k,
        confidence_interval=confidence_interval,
        upgrade_gain=upgrade_gain,
        score_shift=score_shift,
        old_model_arr=old_model_arr,
        bridging_loss=bridging_loss,
    )
    if settled is not None:
        # The short-circuit answers a different question, but it is still built
        # on the same estimate — so it inherits the same caveats.
        settled.provisional = provisional
        settled.tier = tier
        settled.estimate_uninformative = estimate_uninformative
        settled.cascade_arr = cascade_arr
        settled.warnings = warnings + settled.warnings
        if provisional:
            settled.rationale = _PROVISIONAL_PREFIX + settled.rationale
        return settled

    decision = _band_for(arr_at_k)

    # When the break-even is measurable it decides, and the bands only say how.
    # Measured over 15 real-query runs, `bridge_advantage > 1` matched the
    # independent nDCG outcome 14 times; the bands alone matched 10. The four
    # runs they disagreed on were all genuine wins that the bands rejected,
    # because a large upgrade bridged imperfectly still lands at a low ARR.
    advantage = None if upgrade_gain is None else arr_at_k * upgrade_gain
    if advantage is not None and not estimate_uninformative:
        if advantage > 1 + BORDERLINE_BAND:
            # Bridging beats doing nothing. How much of a reindex is still on
            # the table is what separates the two bridging answers.
            decision = (
                "bridge_sufficient" if arr_at_k >= DECISION_BANDS[0][0] else ("bridge_and_migrate")
            )
        elif abs(advantage - 1.0) <= BORDERLINE_BAND:
            # The break-even is the thing that decides, and here it lands inside
            # its own noise band. Saying so in a warning while the headline still
            # reads as a recommendation is the contradiction a user hits first:
            # "bridge and migrate" over "they cannot settle whether to use it".
            # It is the same situation `unsettled_estimate` already covers for a
            # synthesised log, reached from a measured one, so it takes the same
            # route — the band stays, the recommendation does not.
            provisional = True
            warnings.append(
                f"Bridging and keeping the current model are within measurement "
                f"noise of each other ({advantage:.3f}x). The bands below "
                f"describe the adapter; they cannot settle whether to use it."
            )

    distance, nearest = min((abs(arr_at_k - t), t) for t, _ in DECISION_BANDS if t > 0)
    borderline = distance <= BORDERLINE_BAND
    straddles, certainty_notes = _certainty(
        arr_at_k, confidence_interval, nearest, borderline, tier=tier
    )
    warnings.extend(certainty_notes)

    if score_shift is not None and score_shift > SCORE_SHIFT_LIMIT:
        warnings.append(
            f"Score distribution shifted by {score_shift:.2f} even after "
            "calibration. If your pipeline filters on a fixed similarity "
            "threshold, that filter will need retuning — ranking is preserved, "
            "absolute scores are not."
        )

    if old_model_arr is not None and bridging_loss is not None and bridging_loss > 0:
        # Inside the noise band: not enough to change the recommendation, but a
        # user about to migrate deserves to know the gain was not measurable.
        warnings.append(
            f"Bridging ({arr_at_k:.3f}) is not measurably better than keeping the "
            f"current model ({old_model_arr:.3f}). The difference is within "
            "sampling noise, so the upgrade may buy nothing on this corpus."
        )

    if tail_arr is not None and arr_at_k - tail_arr > TAIL_GAP_LIMIT:
        warnings.append(
            f"The sparsest clusters retain only {tail_arr:.2f} against an overall "
            f"{arr_at_k:.2f}. Drift is heterogeneous, so a single global adapter "
            "is leaving quality on the table in part of the corpus."
        )

    warnings.extend(_metric_disagreement(advantage, ndcg_advantage))

    return DecisionResult(
        decision=decision,
        arr_at_k=arr_at_k,
        borderline=borderline,
        nearest_threshold=nearest,
        distance_to_threshold=distance,
        confidence_interval=confidence_interval,
        interval_straddles_threshold=straddles,
        upgrade_gain=upgrade_gain,
        score_shift=score_shift,
        old_model_arr=old_model_arr,
        provisional=provisional,
        tier=tier,
        estimate_uninformative=estimate_uninformative,
        ndcg_advantage=ndcg_advantage,
        cascade_arr=cascade_arr,
        warnings=warnings,
        rationale=(
            _PROVISIONAL_PREFIX + _RATIONALE[decision] if provisional else _RATIONALE[decision]
        ),
    )


def _settled_before_the_bands(  # noqa: PLR0913 - one argument per input it consults
    arr_at_k: float,
    *,
    confidence_interval: tuple[float, float] | None,
    upgrade_gain: float | None,
    score_shift: float | None,
    old_model_arr: float | None,
    bridging_loss: float | None,
) -> DecisionResult | None:
    """The two questions that outrank the ARR bands, or ``None`` if neither fires.

    Both are comparisons against the alternatives the user actually has, and
    both make the bands irrelevant when they fire — so they are answered first
    rather than folded in as a warning on a recommendation that already went out.
    """
    # If the new model is not better here, how well an adapter bridges to it is
    # beside the point.
    if upgrade_gain is not None and upgrade_gain < UPGRADE_GAIN_FLOOR:
        return DecisionResult(
            decision="no_upgrade_needed",
            arr_at_k=arr_at_k,
            borderline=False,
            nearest_threshold=UPGRADE_GAIN_FLOOR,
            distance_to_threshold=abs(upgrade_gain - UPGRADE_GAIN_FLOOR),
            confidence_interval=confidence_interval,
            upgrade_gain=upgrade_gain,
            score_shift=score_shift,
            old_model_arr=old_model_arr,
            rationale=_RATIONALE["no_upgrade_needed"],
        )

    # ARR is measured against the oracle, but the alternative a user actually
    # has is keeping the current model. An adapter can recover 0.91 of what a
    # reindex would give and still retrieve *worse* than changing nothing — the
    # first real-corpus run did exactly that: bridged 0.908 against 0.940 for
    # the old model. Recommending a migration there recommends a downgrade.
    if old_model_arr is None or bridging_loss is None or bridging_loss <= BORDERLINE_BAND:
        return None

    # Beyond measurement noise, so this is a finding rather than a wobble. Which
    # way out depends on whether the new model is worth having at all: if it is,
    # only a reindex delivers it; if it is not, nothing does.
    beyond_reach: Decision = (
        "full_reindex"
        if upgrade_gain is not None and upgrade_gain >= UPGRADE_GAIN_FLOOR
        else "no_upgrade_needed"
    )
    return DecisionResult(
        decision=beyond_reach,
        arr_at_k=arr_at_k,
        borderline=False,
        nearest_threshold=old_model_arr,
        distance_to_threshold=bridging_loss,
        confidence_interval=confidence_interval,
        upgrade_gain=upgrade_gain,
        score_shift=score_shift,
        old_model_arr=old_model_arr,
        warnings=[
            (
                f"Bridging retrieves {arr_at_k:.3f} where keeping the current "
                f"model retrieves {old_model_arr:.3f}. The adapter recovers much "
                f"of what a reindex would give, but not enough to beat changing "
                f"nothing."
            )
        ],
        rationale=(
            _BRIDGING_BELOW_CURRENT
            if beyond_reach == "full_reindex"
            else _RATIONALE["no_upgrade_needed"]
        ),
    )


def _metric_disagreement(advantage: float | None, ndcg_advantage: float | None) -> list[str]:
    """Say when recall and nDCG reach opposite verdicts.

    An adapter can retrieve the same documents and order them worse. Measured on
    BEIR/scifact, MiniLM to bge-base: recall@10 rose from 0.7833 to 0.7867 while
    nDCG@10 fell from 0.6451 to 0.6294. The decision runs on recall, so it read
    that as an improvement and said nothing.

    Which metric is right depends on what consumes the results — a pipeline that
    hands ten chunks to a model re-ranks them implicitly, a list shown to a
    person does not. That is the user's call, so the disagreement is surfaced
    rather than resolved here.
    """
    if advantage is None or ndcg_advantage is None:
        return []
    if (advantage > 1.0) == (ndcg_advantage > 1.0):
        return []
    better, worse = ("recall", "ranking") if advantage > 1.0 else ("ranking", "recall")
    return [
        (
            f"Recall and ranking disagree: bridging scores {advantage:.3f}x on "
            f"recall and {ndcg_advantage:.3f}x on nDCG, so it improves {better} "
            f"and degrades {worse}. If your pipeline hands the top-k to a model "
            f"that re-ranks them, recall is the number that matters; if it shows "
            f"a ranked list to a person, nDCG is."
        )
    ]


def _provisional_notes(
    upgrade_gain: float | None, uninformative: bool, *, unsettled: bool = False
) -> list[str]:
    """Why a run cannot say whether bridging is worth doing.

    Three failures with one consequence: no estimate at all, an estimate from a
    task every model solves, and an estimate too close to the threshold to
    survive its own error bar. Each leaves the tool able to describe the adapter
    and unable to recommend it.
    """
    if upgrade_gain is None:
        return [_NO_UPGRADE_ESTIMATE]
    if uninformative:
        return [_TRIVIAL_SYNTHESIS]
    if unsettled:
        return [_UNSETTLED_SYNTHESIS.format(gain=upgrade_gain, error=SYNTH_GAIN_ERROR)]
    return []


def _decide_on_the_break_even(
    advantage: float, arr_at_k: float, banded: Decision
) -> tuple[Decision, list[str]]:
    """Let the break-even override the ARR band, where it is decisive.

    Above the break-even bridging beats doing nothing whatever the band says, and
    the band can say something quite different, because a large upgrade bridged
    imperfectly lands at a low ARR: four of the five real-query runs where
    bridging genuinely helped scored ARR between 0.47 and 0.71, which the bands
    read as ``full_reindex`` or ``caution``. Below the break-even the caller has
    already returned — that case is settled before the bands are consulted at all.
    Within a band-width of 1.0 the two are indistinguishable and neither can
    settle it; saying so beats picking one.
    """
    if advantage > 1 + BORDERLINE_BAND:
        # How much of a reindex is still on the table separates the two
        # bridging answers; whether to bridge at all is already decided.
        return (
            "bridge_sufficient" if arr_at_k >= DECISION_BANDS[0][0] else "bridge_and_migrate",
            [],
        )
    if abs(advantage - 1.0) <= BORDERLINE_BAND:
        return banded, [
            (
                f"Bridging and keeping the current model are within measurement "
                f"noise of each other ({advantage:.3f}x). The band below describes "
                f"the adapter; it cannot settle whether using it is worth doing."
            )
        ]
    return banded, []


def _band_for(arr_at_k: float) -> Decision:
    """The ARR band a score falls in."""
    for threshold, label in DECISION_BANDS:
        if arr_at_k >= threshold:
            return label
    return "full_reindex"


def _certainty(
    arr_at_k: float,
    confidence_interval: tuple[float, float] | None,
    nearest: float,
    borderline: bool,
    *,
    tier: str = "t1",
) -> tuple[bool, list[str]]:
    """How settled the band placement is, and what to say about it.

    Two related observations, kept together because the stronger one supersedes
    the weaker: an interval that spans a threshold already says the result is
    unresolved, and adding "it is also close to one" underneath is noise.

    ``tier`` is consulted only to keep the remedy honest. Telling a run that was
    given ``--queries`` to supply ``--queries`` is advice it has already taken,
    and it reads as the tool not knowing what it was handed.
    """
    del arr_at_k
    notes: list[str] = []
    straddles = False

    if confidence_interval is not None:
        low, high = confidence_interval
        straddles = any(low <= t <= high for t, _ in DECISION_BANDS if t > 0)
        if straddles:
            remedy = (
                "Increase --sample."
                if tier == "t1"
                else "Increase --sample, or supply a real query log with --queries."
            )
            notes.append(
                "The confidence interval spans a decision threshold: this "
                "recommendation is not resolved by the current sample. " + remedy
            )

    if borderline and not straddles:
        notes.append(
            f"ARR is within {BORDERLINE_BAND} of the {nearest} threshold. "
            "Measurement uncertainty at typical sample sizes is about the same "
            "size, so treat the band as approximate."
        )
    return straddles, notes


def _advisories(
    arr_at_k: float,
    *,
    old_model_arr: float | None,
    bridging_loss: float | None,
    score_shift: float | None,
    tail_arr: float | None,
) -> list[str]:
    """Observations that qualify a recommendation without changing it.

    Separate from :func:`decide` because they are independent checks rather
    than branches of one decision: each answers "what else should this user
    know", and none of them alters which band the result falls in.
    """
    notes: list[str] = []

    if old_model_arr is not None and bridging_loss is not None and bridging_loss > 0:
        # Inside the noise band: not enough to change the recommendation, but a
        # user about to migrate deserves to know the gain was not measurable.
        notes.append(
            f"Bridging ({arr_at_k:.3f}) is not measurably better than keeping the "
            f"current model ({old_model_arr:.3f}). The difference is within "
            "sampling noise, so the upgrade may buy nothing on this corpus."
        )

    if score_shift is not None and score_shift > SCORE_SHIFT_LIMIT:
        notes.append(
            f"Score distribution shifted by {score_shift:.2f} even after "
            "calibration. If your pipeline filters on a fixed similarity "
            "threshold, that filter will need retuning — ranking is preserved, "
            "absolute scores are not."
        )

    if tail_arr is not None and arr_at_k - tail_arr > TAIL_GAP_LIMIT:
        notes.append(
            f"The sparsest clusters retain only {tail_arr:.2f} against an overall "
            f"{arr_at_k:.2f}. Drift is heterogeneous, so a single global adapter "
            "is leaving quality on the table in part of the corpus."
        )

    return notes
