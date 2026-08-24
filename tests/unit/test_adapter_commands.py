"""``rebasis adapter``.

`upgrade` is here because an error message referenced it before it existed. The
test that matters most is not that it upgrades — there are no migrations yet —
but that it **never touches the file it was pointed at**. Schema migrations are
forward-only, so the original copy is the only way back.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from rebasis.cli import app
from rebasis.core import ProcrustesAdapter, l2_normalize, save_adapter
from rebasis.errors import EXIT_OK, EXIT_USAGE
from rebasis.types import EncodingProfile

pytestmark = pytest.mark.unit

runner = CliRunner()
DIM = 16


@pytest.fixture
def adapter(tmp_path, rng):  # type: ignore[no-untyped-def]
    src = l2_normalize(rng.standard_normal((60, DIM)).astype(np.float32))
    dst = l2_normalize(rng.standard_normal((60, DIM)).astype(np.float32))
    profile = EncodingProfile(model_id="a-model", dim=DIM)
    return save_adapter(
        ProcrustesAdapter.fit(src, dst),
        tmp_path / "a.rbs",
        direction="query_to_old",
        old_profile=profile,
        new_profile=profile,
    )


class TestUpgrade:
    def test_a_current_adapter_is_left_alone(self, adapter) -> None:  # type: ignore[no-untyped-def]
        before = (adapter / "manifest.json").read_bytes()

        result = runner.invoke(app, ["adapter", "upgrade", str(adapter)])

        assert result.exit_code == EXIT_OK, result.output
        assert "already at schema" in result.output
        assert (adapter / "manifest.json").read_bytes() == before

    def test_a_newer_schema_is_refused_with_the_fix(self, adapter) -> None:  # type: ignore[no-untyped-def]
        """A file from a future release cannot be downgraded, and saying
        "upgrade rebasis" is the only useful answer."""
        path = adapter / "manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["schema"] = 99
        path.write_text(json.dumps(payload), encoding="utf-8")

        result = runner.invoke(app, ["adapter", "upgrade", str(adapter)])

        assert result.exit_code == EXIT_USAGE
        assert "pip install --upgrade" in result.output

    def test_an_unmigratable_old_schema_says_to_refit(self, adapter, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """There are no migrations registered yet, so schema 0 cannot be
        upgraded. Refusing beats producing a file that claims a version whose
        invariants it does not hold."""
        path = adapter / "manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["schema"] = 0
        path.write_text(json.dumps(payload), encoding="utf-8")

        result = runner.invoke(
            app, ["adapter", "upgrade", str(adapter), "--out", str(tmp_path / "up.rbs")]
        )

        assert result.exit_code != EXIT_OK
        assert "rebasis fit" in result.output

    def test_a_directory_that_is_not_an_adapter_says_so(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        result = runner.invoke(app, ["adapter", "upgrade", str(tmp_path)])

        assert result.exit_code == EXIT_USAGE
        assert "manifest.json" in result.output


class TestInspect:
    def test_it_reports_the_file_without_a_store(self, adapter) -> None:  # type: ignore[no-untyped-def]
        result = runner.invoke(app, ["adapter", "inspect", str(adapter)])

        assert result.exit_code == EXIT_OK, result.output
        assert "procrustes" in result.output
        assert "a-model" in result.output

    def test_verify_recomputes_the_hashes(self, adapter) -> None:  # type: ignore[no-untyped-def]
        result = runner.invoke(app, ["adapter", "inspect", str(adapter), "--verify"])

        assert result.exit_code == EXIT_OK, result.output
        assert "verified" in result.output

    def test_json_output_is_machine_readable(self, adapter) -> None:  # type: ignore[no-untyped-def]
        """Parsed from stdout, not from `result.output`.

        `output` is stdout and stderr interleaved, and every rebasis log line
        goes to stderr — including `runtime.detected`, which is emitted on every
        run. A script doing `rebasis adapter inspect --json | jq` reads stdout
        alone, so that is what this asserts.
        """
        result = runner.invoke(app, ["adapter", "inspect", str(adapter), "--json"])

        assert result.exit_code == EXIT_OK, result.output
        payload = json.loads(result.stdout)
        assert payload["adapter_type"] == "procrustes"
        assert payload["n_params"] > 0

    def test_nothing_but_json_reaches_stdout(self, adapter) -> None:  # type: ignore[no-untyped-def]
        """The invariant behind the test above: diagnostics belong on stderr.

        One log line on stdout breaks every pipe into `jq`, and it breaks it in
        a way that looks like malformed JSON rather than like a logging bug.
        """
        result = runner.invoke(app, ["adapter", "inspect", str(adapter), "--json"])

        assert result.stdout.lstrip().startswith("{")
        assert result.stdout.rstrip().endswith("}")


class TestProfiles:
    def test_it_lists_what_rebasis_knows(self) -> None:
        result = runner.invoke(app, ["adapter", "profiles"], env={"COLUMNS": "220"})

        assert result.exit_code == EXIT_OK, result.output
        assert "BAAI/bge-small-en-v1.5" in result.output

    def test_it_marks_asymmetric_models(self) -> None:
        """The distinction the profile table exists for."""
        result = runner.invoke(
            app, ["adapter", "profiles", "intfloat/e5-base-v2"], env={"COLUMNS": "220"}
        )

        assert result.exit_code == EXIT_OK, result.output
        assert "query: " in result.output
        assert "passage: " in result.output

    def test_a_search_with_no_match_exits_nonzero(self) -> None:
        result = runner.invoke(app, ["adapter", "profiles", "not-a-real-model"])

        assert result.exit_code != EXIT_OK

    def test_it_points_at_the_flags_for_unknown_models(self) -> None:
        """The listing is where a user lands after RB-E2003, so it has to close
        the loop rather than only saying what is absent."""
        result = runner.invoke(app, ["adapter", "profiles"], env={"COLUMNS": "220"})

        assert "--new-dim" in result.output
