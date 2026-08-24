"""CLI plumbing: query logs, memory ceilings, report formats.

Small, unglamorous functions that stand between a user's flags and the pipeline.
They are tested for the same reason they exist: each one is a place where a
typo in a file the user exported from somewhere else can turn into a confusing
failure three steps later.
"""

from __future__ import annotations

import json

import pytest

from rebasis.cli._pipeline import load_query_log, write_reports
from rebasis.cli.migrate import _parse_memory, _read_access_log
from rebasis.errors import ConfigError, MalformedQueryLog

pytestmark = pytest.mark.unit


def write_jsonl(path, rows) -> object:  # type: ignore[no-untyped-def]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


class TestQueryLog:
    def test_reads_the_documented_shape(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = write_jsonl(
            tmp_path / "q.jsonl",
            [{"query": "what is x", "relevant": ["a", "b"]}, {"query": "y?", "relevant": ["c"]}],
        )

        log = load_query_log(path)

        assert log.queries == ["what is x", "y?"]
        assert log.qrels == [{"a", "b"}, {"c"}]

    def test_accepts_the_common_alternative_keys(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Query logs are exported from elsewhere. Insisting on one spelling
        means every user writes a converter before they can try the tool."""
        path = write_jsonl(tmp_path / "q.jsonl", [{"text": "hello", "doc_ids": ["a"]}])

        assert load_query_log(path).queries == ["hello"]

    def test_a_query_with_no_judgements_is_kept(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Dropping it is the pipeline's decision, made against the drawn
        sample. The reader's job is to report the file faithfully."""
        path = write_jsonl(tmp_path / "q.jsonl", [{"query": "orphan"}])

        assert load_query_log(path).qrels == [set()]

    def test_blank_lines_are_skipped(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "q.jsonl"
        path.write_text('\n{"query": "a"}\n\n{"query": "b"}\n\n', encoding="utf-8")

        assert len(load_query_log(path)) == 2

    def test_broken_json_names_the_line(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """ "Invalid JSON" in a 40,000-line export is not an actionable message."""
        path = tmp_path / "q.jsonl"
        path.write_text('{"query": "ok"}\nnot json at all\n', encoding="utf-8")

        with pytest.raises(MalformedQueryLog, match="line 2"):
            load_query_log(path)

    def test_a_line_without_a_query_names_the_line(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = write_jsonl(tmp_path / "q.jsonl", [{"relevant": ["a"]}])

        with pytest.raises(MalformedQueryLog, match="line 1"):
            load_query_log(path)

    def test_an_empty_file_is_an_error_not_an_empty_run(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "q.jsonl"
        path.write_text("\n\n", encoding="utf-8")

        with pytest.raises(MalformedQueryLog, match="no queries"):
            load_query_log(path)


class TestMemoryCeiling:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("2GB", 2 * 1024**3),
            ("512MB", 512 * 1024**2),
            ("1500", 1500),
            ("1.5GB", int(1.5 * 1024**3)),
            ("4g", 4 * 1024**3),
        ],
    )
    def test_parses_the_forms_a_user_would_write(self, text: str, expected: int) -> None:
        assert _parse_memory(text) == expected

    def test_none_means_no_ceiling(self) -> None:
        assert _parse_memory(None) is None

    def test_nonsense_is_rejected_rather_than_silently_ignored(self) -> None:
        """Silently treating `--max-memory lots` as "no ceiling" would remove
        the protection the user asked for at the moment they asked for it."""
        with pytest.raises(ConfigError, match="memory ceiling"):
            _parse_memory("lots")


class TestAccessLog:
    def test_reads_counts(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = write_jsonl(tmp_path / "a.jsonl", [{"id": "a", "count": 9}, {"id": "b"}])

        assert _read_access_log(path) == {"a": 9.0, "b": 1.0}

    def test_no_file_means_no_priorities(self) -> None:
        assert _read_access_log(None) is None


class TestReportFormat:
    def test_the_suffix_chooses_the_renderer(self, tmp_path, probe_result) -> None:  # type: ignore[no-untyped-def]
        html, markdown = tmp_path / "r.html", tmp_path / "r.md"
        write_reports(probe_result(), store_uri="memory://", report=html)
        write_reports(probe_result(), store_uri="memory://", report=markdown)

        assert html.read_text(encoding="utf-8").lstrip().startswith("<")
        assert markdown.read_text(encoding="utf-8").lstrip().startswith("#")

    def test_no_path_writes_nothing(self, tmp_path, probe_result) -> None:  # type: ignore[no-untyped-def]
        write_reports(probe_result(), store_uri="memory://", report=None)

        assert not list(tmp_path.iterdir())
