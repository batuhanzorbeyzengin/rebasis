"""`rebasis gc` — the one command whose whole purpose is to delete.

The library beneath it is covered by `test_storage_lifecycle.py`. What is
covered here is the command: that a bare invocation deletes nothing, that
`--apply` deletes only what the plan listed, and that a shadow copy — the thing
that makes a migration reversible — cannot be removed by a user who has not
said in as many words that they understand what that costs.
"""

from __future__ import annotations

import os
import time
from pathlib import Path  # noqa: TC003 - a runtime annotation on a fixture

import pytest
from typer.testing import CliRunner

from rebasis.cli import app
from rebasis.manifest import CACHE_DIR, REPORTS_DIR, SHADOW_DIR

pytestmark = pytest.mark.unit

runner = CliRunner()
OLD = 200 * 24 * 3600
WIDE = {"COLUMNS": "200"}


@pytest.fixture
def state(tmp_path: Path) -> Path:
    """A state directory holding one of everything `gc` knows about."""
    root = tmp_path / ".rebasis"
    stale = time.time() - OLD

    for directory, name in ((REPORTS_DIR, "probe-2025.html"), (CACHE_DIR, "sample.npz")):
        path = root / directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x" * 1024, encoding="utf-8")
        os.utime(path, (stale, stale))

    shadow = root / SHADOW_DIR / "job-abc123" / "batch-0.npz"
    shadow.parent.mkdir(parents=True, exist_ok=True)
    shadow.write_text("originals", encoding="utf-8")
    return root


def gc(state: Path, *args: str):  # type: ignore[no-untyped-def]
    return runner.invoke(app, ["gc", "--state-dir", str(state), *args], env=WIDE)


class TestItDeletesNothingUnlessAsked:
    def test_a_bare_run_is_a_dry_run(self, state: Path) -> None:
        """A garbage collector that deletes without being asked is the
        data-loss class it exists to prevent."""
        result = gc(state)

        assert result.exit_code == 0
        assert (state / REPORTS_DIR / "probe-2025.html").exists()
        assert (state / CACHE_DIR / "sample.npz").exists()

    def test_it_says_what_it_would_remove(self, state: Path) -> None:
        result = gc(state)

        assert "reports" in result.output
        assert "cache" in result.output

    def test_it_names_what_it_will_never_remove(self, state: Path) -> None:
        """Audit records and adapters are protected, and saying so is the
        difference between a tool you trust with `--apply` and one you do not."""
        result = gc(state)

        assert "audit" in result.output.lower()
        assert ".rbs" in result.output


class TestApply:
    def test_apply_removes_the_stale_files(self, state: Path) -> None:
        result = gc(state, "--apply")

        assert result.exit_code == 0
        assert not (state / REPORTS_DIR / "probe-2025.html").exists()
        assert not (state / CACHE_DIR / "sample.npz").exists()

    def test_apply_leaves_the_shadow_copy_alone(self, state: Path) -> None:
        """A shadow copy is only ever removed when named with `--job`."""
        gc(state, "--apply")

        assert (state / SHADOW_DIR / "job-abc123" / "batch-0.npz").exists()

    def test_a_missing_state_directory_is_not_an_error(self, tmp_path: Path) -> None:
        """Nothing to collect is a normal outcome, not a failure."""
        result = gc(tmp_path / "never-used", "--apply")

        assert result.exit_code == 0


class TestTheShadowCopyNeedsAnExplicitAcknowledgement:
    def test_naming_a_job_is_not_enough(self, state: Path) -> None:
        result = gc(state, "--job", "job-abc123", "--apply")

        assert result.exit_code == 2
        assert "irreversible" in result.output
        assert (state / SHADOW_DIR / "job-abc123" / "batch-0.npz").exists()

    def test_with_the_acknowledgement_it_goes(self, state: Path) -> None:
        result = gc(state, "--job", "job-abc123", "--apply", "--i-understand")

        assert result.exit_code == 0
        assert not (state / SHADOW_DIR / "job-abc123").exists()

    def test_an_unknown_job_removes_nothing_and_does_not_fail(self, state: Path) -> None:
        result = gc(state, "--job", "job-does-not-exist", "--apply", "--i-understand")

        assert result.exit_code == 0
        assert (state / SHADOW_DIR / "job-abc123" / "batch-0.npz").exists()
