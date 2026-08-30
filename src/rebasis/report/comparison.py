"""Rendering a multi-candidate comparison.

Separate from `report/markdown.py` because it renders a different shape: one
decision there, an **ordering** here. That difference is the whole design — a
`probe` report answers "should I do this", a `compare` report answers "which of
these", and the second is only useful if a reader can tell how much to trust the
order.

So the caveat travels with the table rather than under it. `probe`'s estimate is
weak as a threshold and carries real information as a ranker
([section 9](../bridge-band.md#9-what-the-counting-is-worth)), and a leaderboard
printed without that sentence is a stronger claim than the measurement supports.

**No content, ever**, the same as every other report: model ids, counts and
ratios. Never a document, never a query.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rebasis.probe.comparison import CandidateComparison, ComparisonResult

__all__ = ["render_comparison_html", "render_comparison_markdown"]

_INTRO = (
    "Every candidate below was scored on the **same** sample of your index, the "
    "same fit/held-out split and the same queries. Only the embedding pass "
    "differs. A redraw per candidate would introduce a shift larger than some "
    "of the gaps being compared, so consistency across rows is bought at the "
    "cost of a little absolute accuracy — which is the right trade for a "
    "comparison and the wrong one for a single measurement."
)

_COLUMNS = "| candidate | vs. current | ARR | bridge adv. | cascade adv. | reindex | decision |"


def _ratio(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}x"


def _duration(cost: dict[str, Any] | None) -> str:
    seconds = (cost or {}).get("seconds")
    if not seconds:
        return "—"
    if seconds < 5400:  # noqa: PLR2004 - the point at which hours read better
        return f"~{seconds / 60:.0f}m"
    return f"~{seconds / 3600:.1f}h"


def _decision(candidate: CandidateComparison) -> str:
    decision = candidate.result.decision
    if decision.arrangement == "cascade":
        return f"`{decision.decision}` + `cascade`"
    return f"`{decision.decision}`"


def render_comparison_markdown(result: ComparisonResult, *, store_uri: str = "") -> str:
    """Render a comparison as Markdown."""
    reference = result.reference.get("model") or "the index's own model"
    lines = [
        "# Which model, on your corpus",
        "",
        (
            f"**{reference}** is the reference — it is already in the index, so its "
            f"vectors are read rather than recomputed and it is not one of the rows."
        ),
        "",
        _INTRO,
        "",
        _COLUMNS,
        "|" + "---|" * 7,
    ]
    for candidate in result.ranked():
        decision = candidate.result.decision
        name = f"`{candidate.model}`" + (" *(round 1)*" if candidate.eliminated else "")
        lines.append(
            f"| {name} | {_ratio(candidate.upgrade_gain)} | {decision.arr_at_k:.3f} "
            f"| {_ratio(decision.bridge_advantage)} | {_ratio(decision.cascade_advantage)} "
            f"| {_duration(candidate.result.reindex_cost)} | {_decision(candidate)} |"
        )

    lines += [
        "",
        "### What the ordering is worth",
        "",
        result.ranking_caveat,
        "",
    ]
    if result.remote_candidates:
        lines += [
            "> **Document text left this machine.** These candidates run on a "
            "remote endpoint: " + ", ".join(f"`{m}`" for m in result.remote_candidates) + ".",
            "",
        ]
    lines += _provenance(result, store_uri)
    return "\n".join(lines)


def _provenance(result: ComparisonResult, store_uri: str) -> list[str]:
    """What was drawn, once, and what it was drawn from."""
    sample = result.sample
    lines = [
        "### The sample every row shares",
        "",
        f"- {sample.get('size', 0):,} documents of {sample.get('n_total', 0):,}",
        f"- {sample.get('n_queries', 0):,} query proxies",
        f"- strategy `{sample.get('strategy')}`, seed {sample.get('seed')}",
    ]
    if sample.get("tiered"):
        lines.append(
            f"- tiered: every candidate scored on {sample.get('first_round'):,} first, "
            f"and only what that round could not separate carried through"
        )
    if store_uri:
        lines.append(f"- index `{store_uri}`")
    lines += ["", "This report contains no document text, queries or vectors."]
    return lines


_CSS = """
:root { color-scheme: light dark; --line: #d0d7de; --muted: #57606a; --good: #1a7f37;
        --bad: #cf222e; --bg: Canvas; --fg: CanvasText; }
body { margin: 0; background: var(--bg); color: var(--fg);
       font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
main { max-width: 60rem; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }
h1 { font-size: 1.7rem; margin: 0 0 .5rem; }
h3 { margin-top: 2.5rem; }
p.lede { color: var(--muted); margin: 0 0 2rem; }
.wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { text-align: left; padding: .5rem .75rem; border-bottom: 1px solid var(--line); }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.best td { font-weight: 600; }
.note { background: color-mix(in srgb, var(--bad) 8%, transparent);
        border-left: 3px solid var(--bad); padding: .75rem 1rem; margin: 1.5rem 0; }
.caveat { background: color-mix(in srgb, var(--muted) 10%, transparent);
          border-left: 3px solid var(--muted); padding: .75rem 1rem; margin: 1.5rem 0; }
footer { margin-top: 3rem; color: var(--muted); font-size: .85rem; }
"""


def render_comparison_html(result: ComparisonResult, *, store_uri: str = "") -> str:
    """Render a comparison as a single self-contained HTML page."""
    from rebasis.__about__ import __version__

    reference = str(result.reference.get("model") or "the index's own model")
    sample = result.sample
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Which model, on your corpus</title>",
        f"<style>{_CSS}</style></head><body><main>",
        "<h1>Which model, on your corpus</h1>",
        (
            f'<p class="lede"><strong>{html.escape(reference)}</strong> is the reference '
            f"— already in the index, so its vectors are read rather than recomputed, "
            f"and it is not one of the rows.</p>"
        ),
        f"<p>{html.escape(_INTRO).replace('**', '')}</p>",
        '<div class="wrap"><table>',
        (
            "<tr><th>Candidate</th><th>vs. current</th><th>ARR</th><th>bridge adv.</th>"
            "<th>cascade adv.</th><th>reindex</th><th>Decision</th></tr>"
        ),
    ]
    for position, candidate in enumerate(result.ranked()):
        decision = candidate.result.decision
        best = ' class="best"' if position == 0 and not candidate.eliminated else ""
        name = html.escape(candidate.model) + (
            " <em>(round 1)</em>" if candidate.eliminated else ""
        )
        parts.append(
            f"<tr{best}><td>{name}</td>"
            f'<td class="num">{_ratio(candidate.upgrade_gain)}</td>'
            f'<td class="num">{decision.arr_at_k:.3f}</td>'
            f'<td class="num">{_ratio(decision.bridge_advantage)}</td>'
            f'<td class="num">{_ratio(decision.cascade_advantage)}</td>'
            f'<td class="num">{_duration(candidate.result.reindex_cost)}</td>'
            f"<td>{html.escape(_decision(candidate).replace('`', ''))}</td></tr>"
        )
    parts.append("</table></div>")

    parts.append("<h3>What the ordering is worth</h3>")
    parts.append(f'<div class="caveat">{html.escape(result.ranking_caveat)}</div>')
    if result.remote_candidates:
        parts.append(
            '<div class="note"><strong>Document text left this machine.</strong> '
            "These candidates run on a remote endpoint: "
            + html.escape(", ".join(result.remote_candidates))
            + ".</div>"
        )

    parts.append("<h3>The sample every row shares</h3><ul>")
    parts.append(
        f"<li>{sample.get('size', 0):,} documents of {sample.get('n_total', 0):,}</li>"
        f"<li>{sample.get('n_queries', 0):,} query proxies</li>"
        f"<li>strategy {html.escape(str(sample.get('strategy')))}, "
        f"seed {sample.get('seed')}</li>"
    )
    if sample.get("tiered"):
        parts.append(
            f"<li>tiered: every candidate scored on {sample.get('first_round'):,} first</li>"
        )
    parts.append("</ul>")

    parts.append(f"<footer>rebasis {html.escape(__version__)}")
    if store_uri:
        parts.append(f" · <code>{html.escape(store_uri)}</code>")
    parts.append(
        "<br>This report contains no document text, queries or vectors."
        "</footer></main></body></html>"
    )
    return "".join(parts)
