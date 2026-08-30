"""The HTML probe report.

Same content as the Markdown report, in a form that can be opened in a browser
and shared. Deliberately a **single self-contained file** with no external
requests: a report about somebody's private corpus should not phone anywhere when
opened, and the same reasoning that keeps telemetry off by default applies here.

No templating dependency either. The report is one page with a fixed structure,
and pulling in Jinja to build it would add a dependency to the core install for
something a formatted string does well enough.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

from rebasis.probe.decision import CASCADE_RATIONALE

if TYPE_CHECKING:
    from rebasis.probe.runner import ProbeResult

__all__ = ["render_html"]

_HEADLINE = {
    "no_upgrade_needed": ("No upgrade needed", "neutral"),
    "bridge_sufficient": ("Bridging is enough", "good"),
    "bridge_and_migrate": ("Bridge now, migrate gradually", "good"),
    "caution": ("Caution", "warn"),
    "full_reindex": ("Reindex", "bad"),
}

_CSS = """
:root {
  --ink: #1a1a1a; --muted: #666; --line: #e2e2e2; --bg: #fff;
  --good: #1a7f37; --warn: #9a6700; --bad: #cf222e; --neutral: #57606a;
}
@media (prefers-color-scheme: dark) {
  :root {
    --ink: #e6e6e6; --muted: #9aa0a6; --line: #30363d; --bg: #0d1117;
    --good: #3fb950; --warn: #d29922; --bad: #f85149; --neutral: #8b949e;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2.5rem 1.5rem; background: var(--bg); color: var(--ink);
  font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
}
main { max-width: 52rem; margin: 0 auto; }
h1 { font-size: 1.1rem; font-weight: 500; color: var(--muted); margin: 0 0 2rem; }
h2 { font-size: 1.9rem; line-height: 1.25; margin: 0 0 .5rem; }
h3 { font-size: 1.05rem; margin: 2.5rem 0 .75rem; }
.verdict { border-left: 4px solid var(--line); padding-left: 1.25rem; margin-bottom: 2.5rem; }
.lede { font-size: 1.15rem; margin: 0 0 .4rem; }
.lede .good { color: var(--good); } .lede .bad { color: var(--bad); }
.verdict.good { border-color: var(--good); } .verdict.good h2 { color: var(--good); }
.verdict.warn { border-color: var(--warn); } .verdict.warn h2 { color: var(--warn); }
.verdict.bad  { border-color: var(--bad);  } .verdict.bad h2  { color: var(--bad); }
.verdict.neutral { border-color: var(--neutral); } .verdict.neutral h2 { color: var(--neutral); }
.rationale { color: var(--muted); margin: 0; }
.metric { display: flex; align-items: baseline; gap: .75rem; margin: 1.5rem 0; }
.metric .value { font-size: 2.6rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.metric .range { color: var(--muted); font-size: .95rem; font-variant-numeric: tabular-nums; }
table { width: 100%; border-collapse: collapse; margin: .5rem 0 1rem; font-size: .93rem; }
th, td { text-align: left; padding: .55rem .7rem; border-bottom: 1px solid var(--line); }
th { font-weight: 600; color: var(--muted); font-size: .82rem;
     text-transform: uppercase; letter-spacing: .04em; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.chosen td { font-weight: 600; }
.note { background: color-mix(in srgb, var(--warn) 8%, transparent);
        border-left: 3px solid var(--warn); padding: .8rem 1rem; margin: .6rem 0;
        font-size: .93rem; }
.wrap { overflow-x: auto; }
footer { margin-top: 3rem; padding-top: 1.25rem; border-top: 1px solid var(--line);
         color: var(--muted); font-size: .85rem; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .9em; }
"""


_BANDS_HTML = (
    "<p><strong>The decision bands.</strong> ARR at or above 0.95 means bridging "
    "recovers essentially everything; 0.85–0.95 means bridge now and migrate in "
    "the background; 0.70–0.85 warrants checking whether the loss is concentrated "
    "in part of the corpus; below 0.70 an adapter will not close the gap.</p>"
)

_PROVISIONAL_HTML = (
    '<div class="note"><p><strong>This run cannot tell you whether the upgrade '
    "is worth making.</strong> It measured how well an adapter bridges to the "
    "new model, which is a real number, and it could not measure whether the "
    "new model retrieves better on your corpus — which is the other half of the "
    "decision.</p>"
    "<p>Across six real corpus/model pairs, a run without that second number "
    "recommended bridging in <strong>six of six</strong> cases where bridging "
    "actually lost ground against changing nothing. Add <code>--queries</code> "
    "with a real query log, or <code>--synth-queries lead</code> to estimate it "
    "from the documents.</p></div>"
)

_T0_BLIND_SPOT_HTML = (
    "<p>This tier also cannot tell you anything about the <strong>query "
    "encoding</strong>. Its ground truth is defined by the new model's document "
    "space, so a wrong query prefix does not show up here — measured, a "
    "misconfigured model scores <em>higher</em> at this tier, because sharing "
    "one encoding makes the mapping easier. With a real query log the same "
    "error moves the answer enough to change the recommendation.</p>"
)

_INTERVAL_HTML = (
    "<p><strong>The interval matters as much as the number.</strong> ARR is "
    "computed from a sample. Where its range spans a band boundary, this "
    "measurement does not settle the question — a larger sample, or a real query "
    "log, would.</p>"
)


def render_html(result: ProbeResult, *, store_uri: str = "", title: str = "") -> str:
    """Render a probe result as a single self-contained HTML page."""
    from rebasis.__about__ import __version__

    decision = result.decision
    headline, tone = _HEADLINE.get(decision.decision, (decision.decision, "neutral"))
    interval = decision.confidence_interval

    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{html.escape(title or 'rebasis probe report')}</title>",
        f"<style>{_CSS}</style></head><body><main>",
        f"<h1>{html.escape(title or 'rebasis probe report')}</h1>",
        f'<div class="verdict {tone}"><h2>{html.escape(headline)}</h2>',
        f'<p class="rationale">{html.escape(decision.rationale)}</p></div>',
        '<div class="metric">',
        f'<span class="value">{decision.arr_at_k:.3f}</span>',
        f'<span class="range">ARR@{result.k}',
    ]
    if interval:
        parts.append(f" · 95% {interval[0]:.3f}–{interval[1]:.3f}")
    parts.append("</span></div>")

    if decision.provisional:
        parts.append(_PROVISIONAL_HTML)

    advantage = decision.bridge_advantage
    if advantage is not None:
        tone = "good" if advantage > 1 else "bad"
        verdict = "better" if advantage > 1 else "worse"
        parts.append(
            f'<p class="lede">Bridging would retrieve '
            f'<strong class="{tone}">{advantage:.2f}x</strong> what you have today '
            f"— that is <strong>{verdict}</strong> than keeping the current model.</p>"
        )
        parts.append(
            "<p>That figure is the adapter's quality multiplied by how much better "
            "the new model is on this corpus. Both matter and neither settles it "
            "alone: a good adapter bridging to a barely-better model still loses "
            "ground, and so does a poor adapter bridging to a much better one.</p>"
        )

    parts.append(
        "<p>ARR is the fraction of what a full reindex would give you that the "
        "adapter recovers. 1.000 would mean losing nothing.</p>"
    )

    parts.extend(f'<div class="note">{html.escape(w)}</div>' for w in decision.warnings)

    parts.extend(_cascade_html(result))

    if result.baselines:
        parts.extend(["<h3>What the alternatives look like</h3>", '<div class="wrap"><table>'])
        parts.append("<tr><th>Approach</th><th>ARR</th><th>What it means</th></tr>")
        parts.append(
            f'<tr class="chosen"><td>Bridge with {html.escape(result.best.name)}</td>'
            f'<td class="num">{result.best.arr:.3f}</td>'
            "<td>Use the new model now; the index is untouched</td></tr>"
        )
        if "old_model_only" in result.baselines:
            parts.append(
                f"<tr><td>Do nothing</td>"
                f'<td class="num">{result.baselines["old_model_only"]:.3f}</td>'
                "<td>Stay on the current model entirely</td></tr>"
            )
        if "unadapted" in result.baselines:
            parts.append(
                f"<tr><td>New model, no adapter</td>"
                f'<td class="num">{result.baselines["unadapted"]:.3f}</td>'
                "<td>Feed new vectors straight into the old index</td></tr>"
            )
        if "reachable_ceiling" in result.baselines:
            parts.append(
                f"<tr><td><em>Ceiling &mdash; nothing reaches this</em></td>"
                f'<td class="num"><em>{result.baselines["reachable_ceiling"]:.3f}</em></td>'
                "<td><em>The best any query-side map could do in this index</em>"
                "</td></tr>"
            )
        parts.append("</table></div>")

    parts.extend(["<h3>Every adapter that was tried</h3>", '<div class="wrap"><table>'])
    parts.append(
        "<tr><th>Adapter</th><th>ARR</th><th>95% interval</th>"
        "<th>MRR</th><th>Parameters</th><th>CSLS</th></tr>"
    )
    for candidate in sorted(result.candidates, key=lambda c: -c.arr):
        chosen = ' class="chosen"' if candidate.name == result.best.name else ""
        parts.append(
            f"<tr{chosen}><td>{html.escape(candidate.name)}</td>"
            f'<td class="num">{candidate.arr:.3f}</td>'
            f'<td class="num">{candidate.arr_ci[0]:.3f}–{candidate.arr_ci[1]:.3f}</td>'
            f'<td class="num">{candidate.mrr:.3f}</td>'
            f'<td class="num">{candidate.n_params:,}</td>'
            f"<td>{'used' if candidate.used_csls else '—'}</td></tr>"
        )
    parts.append("</table></div>")

    parts.extend(["<h3>How to read this</h3>", _how_to_read_html(result)])

    parts.extend(
        [
            "<footer>",
            (
                f"rebasis {html.escape(__version__)} · seed {result.seed} · "
                f"k={result.k} · {result.duration_seconds:.1f}s"
            ),
        ]
    )
    if store_uri:
        parts.append(f" · <code>{html.escape(store_uri)}</code>")
    parts.extend(
        [
            "<br>This report contains no document text, queries or vectors.",
            "</footer></main></body></html>",
        ]
    )
    return "".join(parts)


def _cascade_html(result: ProbeResult) -> list[str]:
    """The two-stage arrangement, with its price beside it.

    Rendered as a verdict block rather than a paragraph when it is recommended,
    for the same reason the Markdown report gives it its own heading: the run
    this matters most on is one whose decision reads ``full_reindex``, and a
    win printed as a footnote to "rebuild your index" is a win nobody acts on.
    """
    decision = result.decision
    advantage = decision.cascade_advantage
    if advantage is None or decision.cascade_arr is None:
        return []

    recommended = decision.arrangement == "cascade"
    depth = "N" if decision.cascade_n is None else str(decision.cascade_n)
    single = decision.bridge_advantage
    comparison = (
        f" — against {single:.2f}x when the bridge produces the final ranking"
        if single is not None
        else ""
    )
    parts = []
    if recommended:
        parts.append(
            '<div class="verdict good">'
            "<h2>Recommended: keep the index, add a rerank stage</h2>"
            f'<p class="rationale">{html.escape(CASCADE_RATIONALE)}</p></div>'
        )
    else:
        parts.append("<h3>If the bridge fed a rerank instead</h3>")
    parts.append(
        f"<p>Used as a <strong>recall stage</strong> — the adapter retrieves a "
        f"candidate set of {html.escape(depth)} and the new model reorders it in "
        f"its own space — this adapter retains <strong>{decision.cascade_arr:.3f}"
        f"</strong> of what a full reindex finds at that depth, putting the "
        f"break-even at <strong>{advantage:.2f}x</strong>{html.escape(comparison)}.</p>"
    )
    reuse = decision.candidate_reuse
    per_query = decision.cascade_embeddings_per_query
    if reuse is not None and per_query is not None:
        parts.append(
            f"<p><strong>It costs about {per_query:.0f} documents embedded per "
            f"query.</strong> Your query log's candidate sets overlap by "
            f"{reuse:.0%}, so that share of each set is already cached and only "
            f"the rest is re-embedded. The overlap is a <em>lower bound</em> — a "
            f"running cache accumulates across queries this sample does not "
            f"contain — so the real figure is at or below this one, "
            f"<em>provided the cache can hold the working set</em>. Three runs "
            f"in 48 ran past the default in-memory cache's 50,000 entries and "
            f"paid up to 6 points more; a disk cache has no such ceiling.</p>"
        )
    else:
        parts.append(
            '<div class="note">Not recommended on this run: what the arrangement '
            "costs turns on how often a candidate is already cached, and only a "
            "real query log samples that. Re-run with --queries.</div>"
        )
    return parts


def _how_to_read_html(result: ProbeResult) -> str:
    blocks = [_BANDS_HTML, _INTERVAL_HTML]
    if result.access_weighted:
        blocks.append(
            "<p><strong>Queries were drawn weighted by an access log.</strong> "
            "Every number here describes retention on the questions people "
            "actually send, not on a uniform draw over the corpus. Those are "
            "different quantities and this one is usually the higher of the two, "
            "because a frequently-read document is one the index already handles "
            "well. Compare it against another weighted run, not against an "
            "unweighted one.</p>"
        )
    if result.ground_truth_tier == "t0":
        blocks.append(
            "<p><strong>This run used document proxies, not real queries.</strong> "
            "Held-out documents stood in for queries because no query log was "
            "supplied. That assumes your queries resemble your documents — a "
            "reasonable default and a real limitation. A query log passed with "
            "<code>--queries</code> gives a firmer answer, and is the only way to "
            "tell whether the new model is actually better on your corpus.</p>"
        )
        blocks.append(_T0_BLIND_SPOT_HTML)
    if result.best.score_shift_calibrated is not None:
        blocks.append(
            "<p><strong>Score scale.</strong> After calibration the score "
            f"distribution differs by {result.best.score_shift_calibrated:.2f} (KS). "
            "Ranking is preserved exactly, but absolute similarity scores move. If "
            "your pipeline filters on a fixed threshold such as "
            "<code>similarity &gt; 0.7</code>, that threshold will need retuning.</p>"
        )
    return "".join(blocks)
