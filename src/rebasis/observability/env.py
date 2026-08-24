"""Runtime detection and the log level matrix.

Log level is decided in exactly one place. Every other module asks; none
decides. The resolved configuration is emitted as the first ``INFO`` event of
each run (``runtime.detected``), so "why is it logging at this level" is always
answerable.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "LEVEL_MATRIX",
    "LogFormat",
    "LogSettings",
    "Runtime",
    "RuntimeProfile",
    "detect_runtime",
    "parse_level_spec",
    "resolve_log_settings",
]

LogFormat = Literal["console", "json"]


class Runtime(StrEnum):
    """Detected execution environment."""

    CLI_INTERACTIVE = "cli_interactive"
    CLI_PIPED = "cli_piped"
    CI = "ci"
    LIBRARY = "library"
    TEST = "test"
    SERVER = "server"


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """Defaults for one runtime in the level matrix."""

    level: int
    fmt: LogFormat
    colour: bool
    utc_timestamps: bool
    caller_info: bool


#: The level matrix: one set of defaults per detected runtime.
#:
#: ``cli_interactive`` defaults to WARNING because the CLI's primary output is
#: the rich-rendered progress and report, not logs. Log noise corrupts that
#: output; a user who wants detail types ``-v``.
LEVEL_MATRIX: Mapping[Runtime, RuntimeProfile] = {
    Runtime.CLI_INTERACTIVE: RuntimeProfile(
        logging.WARNING, "console", colour=True, utc_timestamps=False, caller_info=False
    ),
    Runtime.CLI_PIPED: RuntimeProfile(
        logging.INFO, "json", colour=False, utc_timestamps=True, caller_info=False
    ),
    Runtime.CI: RuntimeProfile(
        logging.INFO, "json", colour=False, utc_timestamps=True, caller_info=False
    ),
    Runtime.TEST: RuntimeProfile(
        logging.DEBUG, "console", colour=False, utc_timestamps=False, caller_info=True
    ),
    Runtime.SERVER: RuntimeProfile(
        logging.INFO, "json", colour=False, utc_timestamps=True, caller_info=False
    ),
    # LIBRARY is intentionally absent: as a library, rebasis attaches a
    # NullHandler and emits nothing until the host application asks.
}

_CI_MARKERS = ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "BUILDKITE", "CIRCLECI")


def detect_runtime(env: Mapping[str, str] | None = None) -> Runtime:
    """Determine the execution environment.

    Precedence is deliberate and worth stating, because two markers can be
    present at once:

    1. ``test`` wins over ``ci``. Under pytest the same test must behave
       identically on a laptop and on a CI runner; if ``ci`` won, a logging
       assertion could pass locally and fail in CI — precisely the class of
       failure that is hardest to reproduce.
    2. An explicit ``REBASIS_ENV=server`` beats autodetection: a long-running
       daemon may well have a TTY attached during debugging.
    """
    env = os.environ if env is None else env

    if env.get("PYTEST_CURRENT_TEST"):
        return Runtime.TEST
    if env.get("REBASIS_ENV") == "server":
        return Runtime.SERVER
    if any(env.get(marker) for marker in _CI_MARKERS):
        return Runtime.CI
    if _stderr_is_tty():
        return Runtime.CLI_INTERACTIVE
    return Runtime.CLI_PIPED


def _stderr_is_tty() -> bool:
    """Whether stderr is a terminal, tolerating detached streams."""
    try:
        return bool(sys.stderr.isatty())
    except (AttributeError, ValueError):
        # A closed or replaced stream raises rather than returning False.
        return False


@dataclass(frozen=True, slots=True)
class LogSettings:
    """Fully resolved logging configuration for one run."""

    runtime: Runtime
    level: int
    fmt: LogFormat
    colour: bool
    utc_timestamps: bool
    caller_info: bool
    module_levels: Mapping[str, int] = field(default_factory=dict)
    log_file: str | None = None
    redact: bool = True

    @property
    def level_name(self) -> str:
        """Level as its canonical name, for the ``runtime.detected`` event."""
        return logging.getLevelName(self.level)


_LEVEL_NAMES: Mapping[str, int] = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "warn": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "trace": logging.DEBUG,
    "notset": logging.NOTSET,
}


def parse_level_spec(spec: str) -> tuple[int | None, dict[str, int]]:
    """Parse ``REBASIS_LOG_LEVEL`` syntax.

    Supports a bare level (``debug``) and per-module overrides
    (``info,rebasis.migrate=debug``). Module-level tuning exists so that
    debugging a migration does not drown the operator in embedding batch logs.

    Returns:
        The global level (``None`` if only module overrides were given) and a
        mapping of logger name to level.

    Raises:
        ValueError: If a level name is not recognised. Silently ignoring a typo
            here would leave the user staring at the wrong verbosity with no
            explanation.
    """
    global_level: int | None = None
    modules: dict[str, int] = {}

    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        if "=" in part:
            name, _, value = part.partition("=")
            modules[name.strip()] = _level_value(value.strip())
        else:
            global_level = _level_value(part)
    return global_level, modules


def _level_value(name: str) -> int:
    key = name.strip().lower()
    if key not in _LEVEL_NAMES:
        options = ", ".join(sorted(_LEVEL_NAMES))
        msg = f"unknown log level {name!r}; expected one of: {options}"
        raise ValueError(msg)
    return _LEVEL_NAMES[key]


def _parse_format(raw: str | None) -> LogFormat | None:
    """Validate ``REBASIS_LOG_FORMAT``.

    ``auto`` returns ``None``, meaning "defer to the environment default"; that
    is what makes ``REBASIS_LOG_FORMAT=auto`` a way to *undo* an override rather
    than a third format.

    Raises:
        ValueError: On an unrecognised value. Silently ignoring a typo here
            leaves the user staring at the wrong renderer with no explanation.
    """
    if raw is None or raw == "auto":
        return None
    # Explicit comparisons rather than a membership test: only this form lets a
    # type checker narrow `str` down to the Literal, and a silently widened
    # return type here would defeat the point of typing LogFormat at all.
    if raw == "console":
        return "console"
    if raw == "json":
        return "json"
    msg = f"unknown log format {raw!r}; expected one of: console, json, auto"
    raise ValueError(msg)


def resolve_log_settings(  # noqa: PLR0913 - one input per precedence level
    *,
    cli_level: int | None = None,
    cli_format: LogFormat | None = None,
    cli_log_file: str | None = None,
    config_level: int | None = None,
    config_format: LogFormat | None = None,
    redact: bool = True,
    env: Mapping[str, str] | None = None,
) -> LogSettings:
    """Resolve logging configuration in precedence order.

    Highest to lowest:

    1. CLI flag (``--log-level``, ``-v``, ``-vv``, ``-q``)
    2. Environment variable (``REBASIS_LOG_LEVEL``)
    3. Configuration file (``[logging] level``)
    4. Environment default (:data:`LEVEL_MATRIX`)

    One rule, no ambiguity — which is the point: a configuration system whose
    precedence is unclear produces bug reports nobody can reproduce.
    """
    env = os.environ if env is None else env
    runtime = detect_runtime(env)
    profile = LEVEL_MATRIX.get(runtime, LEVEL_MATRIX[Runtime.CLI_INTERACTIVE])

    env_level: int | None = None
    module_levels: dict[str, int] = {}
    if raw := env.get("REBASIS_LOG_LEVEL"):
        env_level, module_levels = parse_level_spec(raw)

    level: int = _first_not_none(cli_level, env_level, config_level, profile.level)

    env_format = _parse_format(env.get("REBASIS_LOG_FORMAT"))

    # Annotated explicitly: joining two Literal members widens to `str` during
    # inference, which would silently defeat the LogFormat type.
    fmt: LogFormat = _first_not_none(cli_format, env_format, config_format, profile.fmt)
    log_file = cli_log_file or env.get("REBASIS_LOG_FILE")

    return LogSettings(
        runtime=runtime,
        level=level,
        fmt=fmt,
        # Colour only ever makes sense on a console renderer writing to a TTY.
        colour=profile.colour and fmt == "console" and not log_file,
        utc_timestamps=profile.utc_timestamps,
        # Caller info walks the stack, which is measurably slow in hot loops
        # (batch embedding). Only enabled at DEBUG.
        caller_info=profile.caller_info or level <= logging.DEBUG,
        module_levels=module_levels,
        log_file=log_file,
        redact=redact,
    )


def _first_not_none[T](*values: T | None) -> T:
    """First non-``None`` value; the mechanism behind the precedence order."""
    for value in values:
        if value is not None:
            return value
    msg = "no value supplied and no default available"
    raise ValueError(msg)
