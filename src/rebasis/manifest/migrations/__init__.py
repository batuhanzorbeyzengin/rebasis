"""Numbered SQL migrations.

Forward-only, tracked in ``PRAGMA user_version``. The statements themselves live
in ``db.py`` so that a migration and the schema it produces are read together;
this package exists for migrations that need Python rather than SQL.
"""

from __future__ import annotations

__all__: list[str] = []
