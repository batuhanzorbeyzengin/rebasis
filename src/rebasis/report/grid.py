"""Rendering the truncation and quantization grid.

A different shape again from the other two reports: `probe` renders one
decision, `compare` renders an ordering, and this renders a **frontier**. Two
axes, and which of them matters more is the reader's call rather than the tool's
— so what a report can do is put quality and cost in the same cell and name the
cheapest point above a floor the reader chose.

Every cell carries two retentions. The single-stage number is what the cheap
representation returns on its own; the rescored one is what it returns when the
full-precision vectors reorder its candidates, which costs no embedding at all
because those vectors are the ones the index already holds. On the binary row
the two are far apart, and a report that printed only the first would be
describing an arrangement nobody would deploy.

**No content, ever** — dimensions, precisions, ratios.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rebasis.probe.truncation import GridCell, TruncationGrid

__all__ = ["render_grid_html", "render_grid_markdown"]

_INTRO = (
    "The model does not change and the space does not change. Every cell below "
    "is the **same** vectors held more cheaply — truncated, rounded, or both — "
    "so there is no adapter here and none of the squeeze that bounds one. The "
    "whole grid costs what a single probe costs: the vectors are already in "
    "memory and cutting them is free."
)

_CELL_LEGEND = (
    "Each cell reads **retained / retained after a rescore**, over the storage it "
    "costs as a fraction of today's. The second number is what the cell returns "
    "when the full-precision vectors reorder its top candidates — a pattern that "
    "costs no embedding at all, because those vectors are the ones the index "
    "already holds."
)


def _cells_by_dim(grid: TruncationGrid) -> tuple[list[str], dict[int, dict[str, GridCell]]]:
    """Precisions in the order they were asked for, and the cells by dimension."""
    precisions = list(dict.fromkeys(cell.precision for cell in grid.cells))
    by_dim: dict[int, dict[str, GridCell]] = {}
    for cell in grid.cells:
        by_dim.setdefault(cell.dim, {})[cell.precision] = cell
    return precisions, by_dim


def _ground_truth_note(grid: TruncationGrid) -> str:
    """Which of the two questions this grid answered.

    The distinction decides how the reference cell reads. Against human
    judgements it is a real measurement and is below 1.000; against the index's
    own neighbours it is 1.000 by construction, and the retentions below it are
    agreement with today's results rather than quality against an answer key.
    """
    if grid.reference_ndcg >= 1.0 - 1e-9:
        return (
            "**This grid measures agreement, not quality.** No query log was "
            "given, so held-out documents stood in for queries and the index's "
            "own exact neighbours were the answer. A retention here is the "
            "fraction of *today's results* a cheaper index would still return. "
            "With `--queries` and human judgements it would be quality instead, "
            "and the reference cell would not be 1.000."
        )
    return (
        f"Measured against human judgements: the index retrieves "
        f"**{grid.reference_ndcg:.3f}** nDCG@{grid.k} today, and every cell below "
        f"is a fraction of that. A retention here is quality, not agreement."
    )


def render_grid_markdown(grid: TruncationGrid, *, store_uri: str = "") -> str:
    """Render the grid as Markdown."""
    precisions, by_dim = _cells_by_dim(grid)
    lines = [
        "# What a cheaper index would cost",
        "",
        _INTRO,
        "",
        _ground_truth_note(grid),
        "",
        "| dimensions | " + " | ".join(precisions) + " |",
        "|" + "---|" * (len(precisions) + 1),
    ]
    for dim in sorted(by_dim, reverse=True):
        label = f"**{dim}**" + (" (full)" if dim == grid.full_dim else "")
        row = [label]
        for precision in precisions:
            cell = by_dim[dim].get(precision)
            row.append(
                "—"
                if cell is None
                else (f"{cell.retained:.3f} / {cell.retained_rescored:.3f}<br>{cell.storage:.3f}x")
            )
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", _CELL_LEGEND, "", grid.simulation_note, ""]
    lines.extend(_frontier(grid))
    lines.extend(_intervals(precisions, by_dim))
    if grid.warnings:
        lines += ["### Worth knowing", ""]
        lines += [f"- {warning}" for warning in grid.warnings]
        lines.append("")
    lines += [
        "### Writing it back is not this tool's job",
        "",
        (
            "Going from `vector(1024)` to `vector(256)`, or from float32 to a half "
            "type, means recreating the column. That is DDL, and `migrate` changes "
            "vectors rather than schemas — the line that keeps rebasis from becoming "
            "a vector database. This says what the change is worth; performing it is "
            "yours."
        ),
        "",
        f"Sampled {grid.n_queries:,} queries at k={grid.k}"
        + (f" against `{store_uri}`" if store_uri else "")
        + ".",
        "",
        "This report contains no document text, queries or vectors.",
    ]
    return "\n".join(lines)


def _frontier(grid: TruncationGrid) -> list[str]:
    """The cheapest cell above the floor, and whether the run can settle it."""
    if grid.floor is None:
        return []
    chosen = grid.cheapest_above(grid.floor)
    if chosen is None:
        best = max(cell.retained for cell in grid.cells)
        return [
            "### Nothing clears the floor",
            "",
            (
                f"No cell retains {grid.floor:.0%} of what this index does today; "
                f"the best is {best:.3f}. Either the floor is above what a cheaper "
                f"representation of this corpus can deliver, or the grid needs a "
                f"row between the ones it was given."
            ),
            "",
        ]
    low, high = chosen.interval
    lines = [
        f"### Cheapest cell above {grid.floor:.0%}",
        "",
        (
            f"**{chosen.dim} dimensions at {chosen.precision}** — retains "
            f"{chosen.retained:.3f} (95% {low:.3f}–{high:.3f}) for "
            f"{chosen.storage:.3f}x the storage."
        ),
        "",
    ]
    if low < grid.floor <= high:
        lines += [
            (
                "> Its interval spans the floor, so this run cannot settle "
                "whether that cell clears it. Increase `--sample`."
            ),
            "",
        ]
    return lines


def _intervals(precisions: list[str], by_dim: dict[int, dict[str, GridCell]]) -> list[str]:
    """Every cell's interval, because a point estimate invites over-reading.

    In its own table rather than in the grid: six numbers to a cell is a table
    nobody reads, and the intervals are what a reader consults once they have a
    candidate rather than while scanning.
    """
    lines = [
        "### The intervals",
        "",
        (
            "Paired bootstrap against the reference cell, on the same queries. "
            "Where two cells' intervals overlap, this run did not separate them."
        ),
        "",
        "| dimensions | " + " | ".join(precisions) + " |",
        "|" + "---|" * (len(precisions) + 1),
    ]
    for dim in sorted(by_dim, reverse=True):
        row = [str(dim)]
        for precision in precisions:
            cell = by_dim[dim].get(precision)
            row.append("—" if cell is None else f"{cell.interval[0]:.3f}–{cell.interval[1]:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


_CSS = """
:root { color-scheme: light dark; --line: #d0d7de; --muted: #57606a; --good: #1a7f37;
        --warn: #9a6700; --bg: Canvas; --fg: CanvasText; }
body { margin: 0; background: var(--bg); color: var(--fg);
       font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
main { max-width: 62rem; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }
h1 { font-size: 1.7rem; margin: 0 0 .5rem; }
h3 { margin-top: 2.5rem; }
p.lede { color: var(--muted); margin: 0 0 2rem; }
.wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { text-align: left; padding: .5rem .75rem; border-bottom: 1px solid var(--line); }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
td .cost { color: var(--muted); font-size: .85em; display: block; }
td.chosen { outline: 2px solid var(--good); }
.note { background: color-mix(in srgb, var(--warn) 8%, transparent);
        border-left: 3px solid var(--warn); padding: .75rem 1rem; margin: 1.5rem 0; }
footer { margin-top: 3rem; color: var(--muted); font-size: .85rem; }
"""


def render_grid_html(grid: TruncationGrid, *, store_uri: str = "") -> str:
    """Render the grid as a single self-contained HTML page."""
    from rebasis.__about__ import __version__

    precisions, by_dim = _cells_by_dim(grid)
    chosen = None if grid.floor is None else grid.cheapest_above(grid.floor)
    parts: list[Any] = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>What a cheaper index would cost</title>",
        f"<style>{_CSS}</style></head><body><main>",
        "<h1>What a cheaper index would cost</h1>",
        f'<p class="lede">{html.escape(_INTRO).replace("**", "")}</p>',
        f"<p>{html.escape(_ground_truth_note(grid)).replace('**', '')}</p>",
        '<div class="wrap"><table>',
        "<tr><th>dimensions</th>"
        + "".join(f"<th>{html.escape(p)}</th>" for p in precisions)
        + "</tr>",
    ]
    for dim in sorted(by_dim, reverse=True):
        label = f"{dim}" + (" (full)" if dim == grid.full_dim else "")
        cells = [f"<td>{html.escape(label)}</td>"]
        for precision in precisions:
            cell = by_dim[dim].get(precision)
            if cell is None:
                cells.append('<td class="num">—</td>')
                continue
            marker = " chosen" if chosen is not None and cell is chosen else ""
            cells.append(
                f'<td class="num{marker}">{cell.retained:.3f} / '
                f'{cell.retained_rescored:.3f}<span class="cost">{cell.storage:.3f}x</span></td>'
            )
        parts.append("<tr>" + "".join(cells) + "</tr>")
    parts.append("</table></div>")
    parts.append(f"<p>{html.escape(_CELL_LEGEND).replace('**', '')}</p>")
    parts.append(f'<div class="note">{html.escape(grid.simulation_note)}</div>')

    if grid.floor is not None:
        parts.append(f"<h3>Cheapest cell above {grid.floor:.0%}</h3>")
        if chosen is None:
            best = max(cell.retained for cell in grid.cells)
            parts.append(
                f'<div class="note">No cell retains {grid.floor:.0%}; the best is {best:.3f}.</div>'
            )
        else:
            low, high = chosen.interval
            parts.append(
                f"<p><strong>{chosen.dim} dimensions at {html.escape(chosen.precision)}"
                f"</strong> — retains {chosen.retained:.3f} (95% {low:.3f}–{high:.3f}) "
                f"for {chosen.storage:.3f}x the storage.</p>"
            )
            if low < grid.floor <= high:
                parts.append(
                    '<div class="note">Its interval spans the floor, so this run '
                    "cannot settle whether that cell clears it. Increase "
                    "--sample.</div>"
                )

    parts.extend(f'<div class="note">{html.escape(w)}</div>' for w in grid.warnings)
    parts.append(
        "<h3>Writing it back is not this tool's job</h3><p>Going from "
        "<code>vector(1024)</code> to <code>vector(256)</code>, or from float32 "
        "to a half type, means recreating the column. That is DDL, and "
        "<code>migrate</code> changes vectors rather than schemas.</p>"
    )
    parts.append(
        f"<footer>rebasis {html.escape(__version__)} · {grid.n_queries:,} queries · k={grid.k}"
    )
    if store_uri:
        parts.append(f" · <code>{html.escape(store_uri)}</code>")
    parts.append(
        "<br>This report contains no document text, queries or vectors."
        "</footer></main></body></html>"
    )
    return "".join(parts)
