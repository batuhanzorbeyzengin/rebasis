"""Report rendering.

Under the layer contract, ``report`` is used by ``cli`` and depends on
``probe``'s result types only.

The rule that governs everything here: **reports carry ids and counts, never
content**. The single exception is ``--include-samples``, which the user asks
for explicitly.
"""

from __future__ import annotations

from rebasis.report.html import render_html
from rebasis.report.markdown import render_markdown, summary_dict

__all__ = ["render_html", "render_markdown", "summary_dict"]
