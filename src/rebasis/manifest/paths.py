"""Where rebasis keeps its state.

**Project-local by default:** ``.rebasis/`` next to the index, so a vault's
rebasis state moves with the vault rather than being stranded in a home directory
when the vault is copied to another machine.

``--state-dir`` overrides it, and a global location under ``platformdirs`` is
available for users who prefer one.
"""

from __future__ import annotations

from pathlib import Path

from rebasis.storage.layout import (
    ADAPTERS_DIR,
    CACHE_DIR,
    REPORTS_DIR,
    SHADOW_DIR,
    STATE_DIR_NAME,
)

__all__ = [
    "ADAPTERS_DIR",
    "CACHE_DIR",
    "REPORTS_DIR",
    "SHADOW_DIR",
    "STATE_DIR_NAME",
    "default_state_dir",
    "global_state_dir",
    "manifest_path",
]


def default_state_dir(near: Path | str | None = None) -> Path:
    """The state directory for an index at ``near``.

    ``REBASIS_STATE_DIR`` wins when it is set and no explicit path is given: a
    user who has said once where rebasis keeps its state should not have to
    repeat it on every command. Otherwise the current working directory, which
    is what someone running ``rebasis probe`` inside their project expects.
    """
    if near is None:
        from rebasis.config import settings

        configured = settings().state_dir
        if configured is not None:
            return Path(configured)
    base = Path(near) if near is not None else Path.cwd()
    if base.is_file():
        base = base.parent
    return base / STATE_DIR_NAME


def global_state_dir() -> Path:
    """A platform-appropriate shared location, for users who prefer one."""
    from platformdirs import user_data_dir

    return Path(user_data_dir("rebasis", appauthor=False))


def manifest_path(state_dir: Path | str) -> Path:
    """Path to the manifest database inside a state directory."""
    return Path(state_dir) / "manifest.db"
