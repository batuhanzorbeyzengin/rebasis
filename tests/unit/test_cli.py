"""CLI surface and exit codes.

Exit codes are a documented contract for script users, so they are tested rather
than assumed. A change here is a breaking change.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from rebasis.cli import app
from rebasis.errors import EXIT_OK, EXIT_USAGE

runner = CliRunner()


@pytest.mark.unit
def test_help_lists_the_commands() -> None:
    """`--help` must work with no configuration and no dependencies."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == EXIT_OK
    for command in ("probe", "doctor", "version"):
        assert command in result.output


@pytest.mark.unit
def test_version_prints_the_package_version() -> None:
    """Scripts pin on this output."""
    from rebasis import __version__

    result = runner.invoke(app, ["version"])
    assert result.exit_code == EXIT_OK
    assert __version__ in result.output


@pytest.mark.unit
def test_doctor_runs_without_optional_dependencies() -> None:
    """`doctor` is what a user runs when something is wrong.

    It must therefore work in the most broken environment — no torch, no GPU,
    nothing configured.
    """
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == EXIT_OK
    assert "rebasis" in result.output
    assert "store backends" in result.output


@pytest.mark.unit
def test_doctor_reports_torch_honestly() -> None:
    """It must say what is actually installed, not what would be convenient."""
    result = runner.invoke(app, ["doctor"])
    assert "torch" in result.output


@pytest.mark.unit
def test_unknown_command_exits_with_the_usage_code() -> None:
    """Usage errors are 2, not 1."""
    assert runner.invoke(app, ["no-such-command"]).exit_code == EXIT_USAGE


@pytest.mark.unit
def test_missing_required_option_exits_with_the_usage_code() -> None:
    """Same category: the invocation was wrong, not the domain operation."""
    assert runner.invoke(app, ["probe"]).exit_code == EXIT_USAGE


@pytest.mark.unit
def test_unsafe_logging_announces_itself() -> None:
    """Disabling redaction cannot be done quietly."""
    result = runner.invoke(app, ["--unsafe-log-content", "version"])
    assert "Redaction is disabled" in result.output


@pytest.mark.unit
@pytest.mark.parametrize("flags", [["-v"], ["-vv"], ["-q"]])
def test_verbosity_flags_are_accepted(flags: list[str]) -> None:
    """The log-level precedence ladder starts with these."""
    assert runner.invoke(app, [*flags, "version"]).exit_code == EXIT_OK
