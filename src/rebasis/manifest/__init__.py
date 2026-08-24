"""Local state: migration jobs, checkpoints and the audit trail.

Layer contract: ``manifest`` is used by ``audit``, ``migrate`` and ``serve``, and
depends on ``errors``, ``types`` and ``observability``.

The state directory is **project-local** by default — ``.rebasis/`` beside the
index. A vault's rebasis state should travel with the vault rather than getting
lost in a home directory.
"""

from __future__ import annotations

from rebasis.manifest.db import SCHEMA_VERSION, ManifestDB
from rebasis.manifest.models import AuditRow, JobItemRow, JobRow, ShadowRow
from rebasis.manifest.paths import (
    ADAPTERS_DIR,
    CACHE_DIR,
    REPORTS_DIR,
    SHADOW_DIR,
    STATE_DIR_NAME,
    default_state_dir,
    global_state_dir,
    manifest_path,
)

__all__ = [
    "ADAPTERS_DIR",
    "CACHE_DIR",
    "REPORTS_DIR",
    "SCHEMA_VERSION",
    "SHADOW_DIR",
    "STATE_DIR_NAME",
    "AuditRow",
    "JobItemRow",
    "JobRow",
    "ManifestDB",
    "ShadowRow",
    "default_state_dir",
    "global_state_dir",
    "manifest_path",
]
