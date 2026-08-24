"""The report.

A report is not a data dump; it is an argument. These tests assert the things
that make it one: the decision comes before the numbers, a measurement's
caveats travel with it, and the HTML page reaches nowhere. The last is not a
nicety — a report about a private corpus that fetches a font is a report that
tells a third party the corpus was probed.
"""

from __future__ import annotations

import re

import pytest

from rebasis.report import render_html, render_markdown, summary_dict

pytestmark = pytest.mark.unit


class TestMarkdown:
    def test_the_decision_comes_before_the_numbers(self, probe_result) -> None:
        """A reader who reads one line should read the one that says what to do."""
        rendered = render_markdown(probe_result())

        # Case-insensitive: the headline is prose ("Bridge now, ..."), and what
        # is being asserted is its position, not its capitalisation.
        assert rendered.lower().index("bridge") < rendered.index("0.93")

    def test_the_alternatives_are_shown_beside_the_result(self, probe_result) -> None:
        """An ARR alone is uninterpretable; the comparison is what makes it mean something."""
        rendered = render_markdown(probe_result())

        assert "0.710" in rendered
        assert "0.040" in rendered

    def test_the_confidence_interval_travels_with_the_number(self, probe_result) -> None:
        rendered = render_markdown(probe_result(arr=0.93))

        assert "0.910" in rendered
        assert "0.950" in rendered

    def test_the_t0_caveat_appears_only_at_t0(self, probe_result) -> None:
        """T0 substitutes documents for queries. A reader who is not told that
        will read the number as stronger evidence than it is."""
        assert "document proxies" in render_markdown(probe_result(tier="t0"))
        assert "document proxies" not in render_markdown(probe_result(tier="t1"))

    def test_a_score_shift_warns_about_fixed_thresholds(self, probe_result) -> None:
        """Ranking survives calibration; absolute scores do not."""
        rendered = render_markdown(probe_result(score_shift=0.31))

        assert "similarity > 0.7" in rendered

    def test_no_corpus_content_reaches_the_report(self, probe_result) -> None:
        """Reports carry parameters and metrics, never document text."""
        rendered = render_markdown(probe_result(), store_uri="chroma:///vault/db#notes")

        assert "no document text" in rendered

    def test_summary_dict_is_serialisable(self, probe_result) -> None:
        import json

        assert json.loads(json.dumps(summary_dict(probe_result())))


class TestHtml:
    def test_the_page_is_self_contained(self, probe_result) -> None:
        """No external host may be contacted.

        Checked as "no absolute URL in a fetching attribute" rather than "no
        http anywhere", so a URL printed as text stays allowed.
        """
        rendered = render_html(probe_result(), store_uri="chroma:///vault/db#notes")

        fetching = re.findall(r'(?:src|href|action)\s*=\s*"([^"]+)"', rendered)
        assert not [url for url in fetching if url.startswith(("http://", "https://", "//"))]
        assert "@import" not in rendered

    def test_the_page_adapts_to_a_dark_theme(self, probe_result) -> None:
        assert "prefers-color-scheme" in render_html(probe_result())

    def test_the_store_uri_is_escaped(self, probe_result) -> None:
        """A collection name is user input and lands in the page."""
        rendered = render_html(probe_result(), store_uri='chroma:///db#<script>alert("x")</script>')

        assert "<script>alert" not in rendered
        assert "&lt;script&gt;" in rendered

    def test_the_decision_is_in_the_page(self, probe_result) -> None:
        assert "bridge" in render_html(probe_result()).lower()
