"""Level matrix and precedence order.

Parametrised across every environment and every precedence level, because the
one failure mode of a configuration system is ambiguity about which input wins —
and that produces bug reports nobody can reproduce.
"""

from __future__ import annotations

import logging

import pytest

from rebasis.observability.env import (
    LEVEL_MATRIX,
    Runtime,
    detect_runtime,
    parse_level_spec,
    resolve_log_settings,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"PYTEST_CURRENT_TEST": "x"}, Runtime.TEST),
        ({"CI": "true"}, Runtime.CI),
        ({"GITHUB_ACTIONS": "true"}, Runtime.CI),
        ({"GITLAB_CI": "true"}, Runtime.CI),
        ({"REBASIS_ENV": "server"}, Runtime.SERVER),
        # Precedence: test beats ci, so a logging assertion behaves the same on
        # a laptop and on a runner. If ci won, the same test could pass locally
        # and fail in CI — the hardest class of failure to reproduce.
        ({"PYTEST_CURRENT_TEST": "x", "CI": "1"}, Runtime.TEST),
        # An explicit REBASIS_ENV beats autodetection: a daemon may well have a
        # TTY attached while being debugged.
        ({"REBASIS_ENV": "server", "CI": "1"}, Runtime.SERVER),
    ],
)
def test_runtime_detection(env: dict[str, str], expected: Runtime) -> None:
    """Each marker resolves to the documented environment."""
    assert detect_runtime(env) is expected


@pytest.mark.unit
@pytest.mark.parametrize("runtime", [r for r in Runtime if r in LEVEL_MATRIX], ids=str)
def test_level_matrix_matches_the_spec(runtime: Runtime) -> None:
    """The documented level table, asserted row by row."""
    expected = {
        Runtime.CLI_INTERACTIVE: (logging.WARNING, "console"),
        Runtime.CLI_PIPED: (logging.INFO, "json"),
        Runtime.CI: (logging.INFO, "json"),
        Runtime.TEST: (logging.DEBUG, "console"),
        Runtime.SERVER: (logging.INFO, "json"),
    }[runtime]
    profile = LEVEL_MATRIX[runtime]
    assert (profile.level, profile.fmt) == expected


@pytest.mark.unit
def test_library_mode_has_no_matrix_entry() -> None:
    """Embedded as a library, rebasis emits nothing until asked."""
    assert Runtime.LIBRARY not in LEVEL_MATRIX


@pytest.mark.unit
def test_interactive_cli_defaults_to_warning() -> None:
    """The CLI's primary output is the report, not logs.

    Log noise corrupts the rich-rendered output; a user who wants detail types
    `-v`.
    """
    assert LEVEL_MATRIX[Runtime.CLI_INTERACTIVE].level == logging.WARNING


@pytest.mark.unit
@pytest.mark.parametrize(
    ("cli", "env_var", "config", "expected"),
    [
        # 4. environment default
        (None, None, None, logging.INFO),
        # 3. configuration file beats the default
        (None, None, logging.ERROR, logging.ERROR),
        # 2. environment variable beats the config file
        (None, "debug", logging.ERROR, logging.DEBUG),
        # 1. CLI flag beats everything
        (logging.CRITICAL, "debug", logging.ERROR, logging.CRITICAL),
    ],
)
def test_level_precedence(
    cli: int | None, env_var: str | None, config: int | None, expected: int
) -> None:
    """CLI > environment > configuration file > matrix default."""
    env = {"CI": "1"}
    if env_var:
        env["REBASIS_LOG_LEVEL"] = env_var
    settings = resolve_log_settings(cli_level=cli, config_level=config, env=env)
    assert settings.level == expected


@pytest.mark.unit
def test_format_precedence_and_auto() -> None:
    """`auto` undoes an override rather than acting as a third format."""
    base = {"CI": "1"}
    assert resolve_log_settings(env=base).fmt == "json"
    assert resolve_log_settings(env={**base, "REBASIS_LOG_FORMAT": "console"}).fmt == "console"
    assert resolve_log_settings(env={**base, "REBASIS_LOG_FORMAT": "auto"}).fmt == "json"
    assert resolve_log_settings(cli_format="console", env=base).fmt == "console"


@pytest.mark.unit
def test_module_level_tuning() -> None:
    """Per-module levels let one subsystem be debugged without drowning in the rest."""
    level, modules = parse_level_spec("info,rebasis.migrate=debug,rebasis.embed=warning")
    assert level == logging.INFO
    assert modules == {"rebasis.migrate": logging.DEBUG, "rebasis.embed": logging.WARNING}


@pytest.mark.unit
def test_module_only_spec_leaves_global_unset() -> None:
    """A spec with only module overrides must not silently change the global level."""
    level, modules = parse_level_spec("rebasis.migrate=debug")
    assert level is None
    assert modules == {"rebasis.migrate": logging.DEBUG}


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["nonsense", "info,rebasis.x=nonsense", "verbose"])
def test_unknown_level_is_rejected(bad: str) -> None:
    """A typo must fail loudly.

    Silently ignoring it leaves the user staring at the wrong verbosity with no
    explanation of why.
    """
    with pytest.raises(ValueError, match="unknown log level"):
        parse_level_spec(bad)


@pytest.mark.unit
def test_unknown_format_is_rejected() -> None:
    """Same reasoning for the renderer."""
    with pytest.raises(ValueError, match="unknown log format"):
        resolve_log_settings(env={"REBASIS_LOG_FORMAT": "yaml"})


@pytest.mark.unit
def test_caller_info_only_at_debug() -> None:
    """Stack inspection is expensive and only enabled at DEBUG.

    In a hot loop such as batch embedding it produces measurable slowdown.
    """
    assert not resolve_log_settings(env={"CI": "1"}).caller_info
    assert resolve_log_settings(cli_level=logging.DEBUG, env={"CI": "1"}).caller_info


@pytest.mark.unit
def test_colour_is_off_when_writing_to_a_file() -> None:
    """ANSI codes in a log file help nobody."""
    settings = resolve_log_settings(cli_log_file="/tmp/x.log", env={})  # noqa: S108
    assert not settings.colour
