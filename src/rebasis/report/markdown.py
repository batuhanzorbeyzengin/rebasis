"""The probe report.

This is what the user actually reads, and the only place the tool's reasoning
becomes visible. Three rules shape it:

**Lead with the decision, not the numbers.** A user opening this wants to know
whether to reindex. The metrics justify the answer; they are not the answer.

**State the limits alongside the result.** The caveat that held-out documents
are standing in for real queries, the confidence interval, and whether the
comparison could even be made — all of it goes in the report rather than in
documentation nobody opens. A number without its uncertainty invites
over-reading, and M0 measured that uncertainty at ±0.024, which is the same
order as the decision bands themselves.

**No content, ever.** Reports carry chunk ids and counts, never text. The single
exception is ``--include-samples``, which the user asks for explicitly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rebasis.probe.runner import ProbeResult

__all__ = ["render_markdown"]

# ── prose blocks ─────────────────────────────────────────────────────
#
# Kept as module constants rather than inline: the render functions stay
# readable, and a long string split across lines inside a list literal reads
# like a missing comma.

_PROVISIONAL_BANNER = (
    "> **This run cannot tell you whether the upgrade is worth making.** It "
    "measured how well an adapter bridges to the new model, which is a real "
    "number, and it could not measure whether the new model retrieves better on "
    "your corpus — which is the other half of the decision.\n>\n"
    "> Across six real corpus/model pairs, a run without that second number "
    "recommended bridging in **six of six** cases where bridging actually lost "
    "ground against changing nothing. Add `--queries` with a real query log, or "
    "`--synth-queries lead` to estimate it from the documents."
)

_BREAK_EVEN_EXPLAINER = (
    "That figure is the adapter's quality multiplied by how much better the new "
    "model is on this corpus. Both matter and neither settles it alone: a good "
    "adapter bridging to a barely-better model still loses ground, and so does a "
    "poor adapter bridging to a much better one."
)

_ARR_EXPLAINER = (
    "ARR is the fraction of what a full reindex would give you that the adapter "
    "recovers. 1.00 would mean losing nothing."
)

_ALTERNATIVES_INTRO = (
    "An ARR on its own is hard to place. These are the two comparisons that make it meaningful."
)

_CANDIDATES_INTRO = (
    "`auto` fits each of these and keeps the best on a held-out set. Where the top "
    "few are within one another's confidence intervals, the cheapest wins."
)

_BANDS_EXPLAINER = (
    "**The decision bands.** ARR at or above 0.95 means bridging recovers "
    "essentially everything; 0.85-0.95 means bridge now and migrate in the "
    "background; 0.70-0.85 warrants a look at whether the loss is concentrated in "
    "part of the corpus; below 0.70 an adapter will not close the gap."
)

_INTERVAL_EXPLAINER = (
    "**The interval matters as much as the number.** ARR is computed from a "
    "sample, and the interval above is its 95% range. Where that range spans a "
    "band boundary, this measurement does not settle the question - a larger "
    "sample, or a real query log, would."
)

_T0_CAVEAT = (
    "**This run used document proxies, not real queries.** Held-out documents "
    "stood in for queries because no query log was supplied. That assumes your "
    "queries resemble your documents; it is a reasonable default and a real "
    "limitation. If you have a query log, `--queries` will give you a firmer "
    "answer - and it is the only way to tell whether the new model is actually "
    "better on your corpus at all.\n\n"
    "This tier also cannot tell you anything about the **query encoding**. Its "
    "ground truth is defined by the new model's document space, so a wrong "
    "query prefix does not show up here - measured, a misconfigured model "
    "scores *higher* at this tier, because sharing one encoding makes the "
    "mapping easier. With a real query log the same error moves the answer "
    "enough to change the recommendation."
)

_PROVENANCE_INTRO = (
    "Everything needed to reproduce this run. The same values are in the audit "
    "record, and `rebasis audit replay` re-runs it."
)

_SCORE_SCALE_WARNING = (
    "**Score scale.** After calibration the score distribution differs from the "
    "original by {shift:.2f} (KS). Ranking is preserved exactly, but absolute "
    "similarity scores move. If your pipeline filters on a fixed threshold such "
    "as `similarity > 0.7`, that threshold will need retuning."
)

_NO_CONTENT = "_This report contains no document text, queries or vectors._"


_DECISION_HEADLINE = {
    "no_upgrade_needed": "No upgrade needed",
    "bridge_sufficient": "Bridging is enough — leave the index alone",
    "bridge_and_migrate": "Bridge now, migrate gradually",
    "caution": "Caution — check the tail before deciding",
    "full_reindex": "Reindexing is the honest answer",
}

_DECISION_COLOUR = {
    "no_upgrade_needed": "grey",
    "bridge_sufficient": "green",
    "bridge_and_migrate": "green",
    "caution": "amber",
    "full_reindex": "red",
}


#: What the cascade figure is, and what rebasis does not do with it.
_CASCADE_EXPLAINER = (
    "A two-stage arrangement is bounded by whether the relevant document reached "
    "the candidate set, not by where the bridge ranked it — which is a weaker "
    "requirement, and measurably so: across 24 runs single-stage bridging beat "
    "keeping the current model in 0, and a two-stage arrangement in 16 "
    "([the measurement](../cascade-band.md)). **rebasis measures this and does not "
    "serve it.** Building it means re-embedding N documents per query, which is a "
    "different cost class from the mapping, and needs a cache that does not exist "
    "yet. Treated as a reason to look, not as a recommendation."
)

#: Why the bound is in the report at all, and what it is not.
_GEOMETRY_EXPLAINER = (
    "δ compares the two models' pairwise similarities on the same documents; the "
    "bound is Corollary 1 of [Maystre et al., *When Embedding Models Meet*]"
    "(https://arxiv.org/abs/2510.13406). It costs one Gram-matrix difference and "
    "no fit, so it is known before the adapter search starts. It runs one way: "
    "geometry preserved means an alignment exists, not that retrieval will find "
    "it. Where a low bound sits next to a low ARR, the alignment was available "
    "and something else lost it — a prefix, an encoding mismatch, or a corpus "
    "the fit sample did not cover."
)


def render_markdown(result: ProbeResult, *, store_uri: str = "", title: str = "") -> str:
    """Render a probe result as Markdown."""
    decision = result.decision
    lines: list[str] = [
        f"# {title or 'rebasis probe report'}",
        "",
        f"## {_DECISION_HEADLINE.get(decision.decision, decision.decision)}",
        "",
        decision.rationale,
        "",
    ]

    lines.extend(_headline_numbers(result))
    lines.extend(_warnings(result))
    lines.extend(_cascade(result))
    lines.extend(_geometry(result))
    lines.extend(_baselines(result))
    lines.extend(_candidates(result))
    lines.extend(_how_to_read(result))
    lines.extend(_provenance(result, store_uri))
    return "\n".join(lines)


def _headline_numbers(result: ProbeResult) -> list[str]:
    decision = result.decision
    interval = decision.confidence_interval
    lines = ["### What was measured", ""]

    # The break-even leads when it can be computed. ARR alone does not answer
    # the question a user is deciding: a high ARR against a small upgrade still
    # loses to changing nothing, and this is the number that says so.
    if decision.provisional:
        lines.extend([_PROVISIONAL_BANNER, ""])

    advantage = decision.bridge_advantage
    if advantage is not None:
        verdict = (
            "**better** than keeping the current model"
            if advantage > 1
            else "**worse** than keeping the current model"
        )
        lines.extend(
            [
                (
                    f"**Bridging would retrieve {advantage:.2f}x what you have "
                    f"today** — that is {verdict}."
                ),
                "",
                _BREAK_EVEN_EXPLAINER,
                "",
            ]
        )

    if interval:
        lines.append(
            f"**ARR@{result.k} = {decision.arr_at_k:.3f}**  "
            f"(95% confidence: {interval[0]:.3f} – {interval[1]:.3f})"
        )
    else:
        lines.append(f"**ARR@{result.k} = {decision.arr_at_k:.3f}**")

    lines.extend(
        [
            "",
            _ARR_EXPLAINER,
            "",
            f"- Best adapter: **{result.best.name}** ({result.best.n_params:,} parameters)",
            f"- Ground truth: {_tier_label(result.ground_truth_tier)}",
            (
                f"- Documents: {result.n_documents:,} · "
                f"queries: {result.n_queries:,} · "
                f"fit pairs: {result.n_fit_pairs:,}"
            ),
            "",
        ]
    )

    if decision.upgrade_gain is not None:
        lines.extend(
            [
                (
                    f"- Upgrade gain: **{decision.upgrade_gain:.3f}** — how much "
                    "better the new model retrieves than the current one on this corpus."
                ),
                "",
            ]
        )
    return lines


def _warnings(result: ProbeResult) -> list[str]:
    if not result.decision.warnings:
        return []
    lines = ["### Worth knowing", ""]
    lines.extend(f"- {warning}" for warning in result.decision.warnings)
    lines.append("")
    return lines


def _cascade(result: ProbeResult) -> list[str]:
    """What the same adapter would be worth feeding a rerank.

    Reported after the decision, never inside it. `rebasis.serve.Cascade` does
    serve this arrangement — what keeps it out of the decision is that its cost
    depends on how often a candidate is already cached, which is a property of a
    query log rather than of the corpus this run measured.
    """
    decision = result.decision
    advantage = decision.cascade_advantage
    if advantage is None or decision.cascade_arr is None:
        return []

    single = decision.bridge_advantage
    comparison = (
        f" — against {single:.2f}x when the bridge produces the final ranking"
        if single is not None
        else ""
    )
    return [
        "### If the bridge fed a rerank instead",
        "",
        (
            f"Used as a **recall stage** — the adapter retrieves a candidate set and "
            f"the new model reorders it in its own space — this adapter retains "
            f"**{decision.cascade_arr:.3f}** of what a full reindex finds at that depth, "
            f"putting the break-even at **{advantage:.2f}x**{comparison}."
        ),
        "",
        _CASCADE_EXPLAINER,
        "",
    ]


def _geometry(result: ProbeResult) -> list[str]:
    """The bound that was available before anything was fitted.

    Placed after the measurement rather than before it, deliberately. It is a
    ceiling on what an orthogonal alignment could do, and a reader who meets a
    ceiling first will read the measurement as a shortfall against it. The
    measurement is the answer; this says what the geometry permitted.
    """
    geometry = result.geometry
    if geometry is None or geometry.n_pairs == 0 or geometry.delta != geometry.delta:
        return []
    return [
        "### What the geometry allowed",
        "",
        geometry.explain(),
        "",
        _GEOMETRY_EXPLAINER,
        "",
    ]


def _baselines(result: ProbeResult) -> list[str]:
    if not result.baselines:
        return []

    lines = [
        "### What the alternatives look like",
        "",
        _ALTERNATIVES_INTRO,
        "",
        "| Approach | ARR | What it means |",
        "|---|---|---|",
        (
            f"| **Bridge with {result.best.name}** | {result.best.arr:.3f} | "
            "Use the new model now; the index is untouched |"
        ),
    ]

    if "old_model_only" in result.baselines:
        lines.append(
            f"| Do nothing | {result.baselines['old_model_only']:.3f} | "
            "Stay on the current model entirely |"
        )
    if "unadapted" in result.baselines:
        lines.append(
            f"| New model, no adapter | {result.baselines['unadapted']:.3f} | "
            "Feed new vectors straight into the old index |"
        )
    if "reachable_ceiling" in result.baselines:
        lines.append(
            f"| *Ceiling — nothing reaches this* | *{result.baselines['reachable_ceiling']:.3f}* | "
            "*The best any query-side map could do in this index* |"
        )
    lines.append("")
    if "reachable_ceiling" in result.baselines:
        gap = result.baselines["reachable_ceiling"] - result.best.arr
        lines.extend(
            [
                (
                    f"The ceiling is what a query would score if it already knew the answer, "
                    f"so no adapter can pass it — it is here to say whether a better one is "
                    f"available. This adapter sits **{gap:+.3f}** from it."
                ),
                "",
            ]
        )
    return lines


def _candidates(result: ProbeResult) -> list[str]:
    lines = [
        "### Every adapter that was tried",
        "",
        _CANDIDATES_INTRO,
        "",
        "| Adapter | ARR | 95% interval | MRR | Parameters | CSLS |",
        "|---|---|---|---|---|---|",
    ]
    for candidate in sorted(result.candidates, key=lambda c: -c.arr):
        marker = " ←" if candidate.name == result.best.name else ""
        lines.append(
            f"| {candidate.name}{marker} | {candidate.arr:.3f} | "
            f"{candidate.arr_ci[0]:.3f} – {candidate.arr_ci[1]:.3f} | "
            f"{candidate.mrr:.3f} | {candidate.n_params:,} | "
            f"{'used' if candidate.used_csls else '—'} |"
        )
    lines.append("")
    return lines


def _how_to_read(result: ProbeResult) -> list[str]:
    lines = [
        "### How to read this",
        "",
        _BANDS_EXPLAINER,
        "",
        _INTERVAL_EXPLAINER,
        "",
    ]

    if result.ground_truth_tier == "t0":
        lines.extend([_T0_CAVEAT, ""])

    if result.best.score_shift_calibrated is not None:
        shift = result.best.score_shift_calibrated
        lines.extend([_SCORE_SCALE_WARNING.format(shift=shift), ""])
    return lines


def _provenance(result: ProbeResult, store_uri: str) -> list[str]:
    from rebasis.__about__ import __version__

    lines = [
        "---",
        "",
        "### Provenance",
        "",
        _PROVENANCE_INTRO,
        "",
        f"- rebasis {__version__}",
        f"- seed {result.seed} · k={result.k} · {result.duration_seconds:.1f}s",
    ]
    if store_uri:
        lines.append(f"- store `{store_uri}`")
    lines.extend(
        [
            "",
            _NO_CONTENT,
            "",
        ]
    )
    return lines


def _tier_label(tier: str) -> str:
    return {
        "t0": "held-out documents as query proxies (no query log supplied)",
        "t1": "a real query log with relevance judgements",
        "t2": "synthesised queries",
    }.get(tier, tier)


def summary_dict(result: ProbeResult) -> dict[str, Any]:
    """A compact machine-readable summary, for scripts and the audit record."""
    return result.to_dict()
