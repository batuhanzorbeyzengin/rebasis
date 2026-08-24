"""The names inside a state directory.

These live in ``storage`` rather than ``manifest`` because ``storage.gc`` needs
them and sits below ``manifest`` in the layer contract. ``manifest.paths``
re-exports them, so the state directory still reads as one idea from above.
"""

from __future__ import annotations

__all__ = ["ADAPTERS_DIR", "CACHE_DIR", "REPORTS_DIR", "SHADOW_DIR", "STATE_DIR_NAME"]

#: Project-local by default: ``.rebasis/`` next to the index. A vault's rebasis
#: state should move with the vault rather than be stranded in a home directory.
STATE_DIR_NAME = ".rebasis"

ADAPTERS_DIR = "adapters"
SHADOW_DIR = "shadow"
REPORTS_DIR = "reports"
CACHE_DIR = "cache"
