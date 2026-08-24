"""Configuration, read from the environment in one place.

Every ``REBASIS_*`` variable is declared here, with its default and what it is
for. Scattering ``os.environ.get`` across the codebase produces two failures
that are hard to see: a variable that is read in one place and ignored in
another, and a set of knobs no user can discover because they exist only in
call sites.

Precedence is always: **explicit argument, then environment, then default.** A
flag the user typed outranks a variable they set months ago.

Reading is deliberately cheap and uncached — these are consulted once per run,
and a cache would mean a test that sets a variable has to know to clear it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Final

__all__ = ["ENVIRONMENT_VARIABLES", "Settings", "env_bool", "env_int", "env_path", "settings"]

#: Every variable rebasis reads, with what it does. Printed by
#: ``rebasis doctor`` so the knobs are discoverable without grepping.
ENVIRONMENT_VARIABLES: Final[dict[str, str]] = {
    "REBASIS_LOG_LEVEL": "Log level: DEBUG, INFO, WARNING, ERROR.",
    "REBASIS_LOG_FORMAT": "auto, console or json. Defaults to json when piped.",
    "REBASIS_LOG_FILE": "Also write logs to this path.",
    "REBASIS_STATE_DIR": "Where state lives. Defaults to ./.rebasis",
    "REBASIS_MAX_MEMORY": "Memory ceiling, e.g. 2GB. The batch size is derived from it.",
    "REBASIS_DEVICE": "auto, cpu, cuda, cuda:N or mps.",
    "REBASIS_DETERMINISTIC": "1 to force deterministic kernels and disable TF32.",
    "REBASIS_OTEL_ENABLED": "1 to send telemetry to YOUR OWN collector. Off by default.",
    "REBASIS_OTEL_CONSOLE": "1 to print spans to the console instead of exporting.",
    "REBASIS_CACHE_DIR": "Sample and embedding cache. Defaults to <state dir>/cache.",
    "REBASIS_NO_COLOR": "1 to disable colour. NO_COLOR is honoured too.",
}

_TRUE = frozenset({"1", "true", "yes", "on"})


def env_bool(name: str, *, default: bool = False) -> bool:
    """Read a boolean variable. Anything not in the true-set is false."""
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in _TRUE


def env_int(name: str, *, default: int | None = None) -> int | None:
    """Read an integer variable, ignoring an unparseable value.

    Ignoring rather than raising: a malformed tuning knob should not stop a run
    that would otherwise work, and the resolved value is reported by ``doctor``.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def env_path(name: str, *, default: str | None = None) -> str | None:
    """Read a path variable, treating empty as unset."""
    raw = os.environ.get(name, "").strip()
    return raw or default


def parse_memory(value: str | None) -> int | None:
    """Parse ``2GB`` / ``512MB`` / a plain byte count into a memory ceiling.

    Raises:
        ConfigError: When the string is not a size. Silently treating
            ``--max-memory lots`` as "no ceiling" would remove the protection
            the user asked for at the moment they asked for it.
    """
    if value is None:
        return None

    from rebasis.errors import ConfigError

    text = value.strip().upper().removesuffix("B")
    units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    multiplier = units.get(text[-1:], 1)
    number = text[:-1] if multiplier > 1 else text
    try:
        return int(float(number) * multiplier)
    except ValueError as exc:
        raise ConfigError(
            f"Could not read {value!r} as a memory ceiling.",
            hint="Use a form like 2GB, 512MB, or a plain number of bytes.",
            cause=exc,
        ) from exc


@dataclass(frozen=True, slots=True)
class Settings:
    """Resolved configuration for one run."""

    device: str = "auto"
    state_dir: str | None = None
    cache_dir: str | None = None
    max_memory_bytes: int | None = None
    deterministic: bool = False
    otel_enabled: bool = False
    no_color: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        """Read every variable once, at the entry point."""
        return cls(
            device=os.environ.get("REBASIS_DEVICE", "auto").strip() or "auto",
            state_dir=env_path("REBASIS_STATE_DIR"),
            cache_dir=env_path("REBASIS_CACHE_DIR"),
            max_memory_bytes=parse_memory(env_path("REBASIS_MAX_MEMORY")),
            deterministic=env_bool("REBASIS_DETERMINISTIC"),
            otel_enabled=env_bool("REBASIS_OTEL_ENABLED"),
            # NO_COLOR is a cross-tool convention; honouring only our own
            # variable would make rebasis the one tool that ignores it.
            no_color=env_bool("REBASIS_NO_COLOR") or bool(os.environ.get("NO_COLOR")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form, for `doctor` and the audit record."""
        return {
            "device": self.device,
            "state_dir": self.state_dir,
            "cache_dir": self.cache_dir,
            "max_memory_bytes": self.max_memory_bytes,
            "deterministic": self.deterministic,
            "otel_enabled": self.otel_enabled,
            "no_color": self.no_color,
        }


def settings() -> Settings:
    """The current settings. Read fresh, so a test that sets a variable works."""
    return Settings.from_env()
