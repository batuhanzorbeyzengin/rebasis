"""``.rbs`` schema migrations.

Transformations from schema N-1 to N live here as pure functions on the
manifest payload. The compatibility promise is explicit: a release producing
schema N must read N and N-1. ``rebasis adapter upgrade`` applies them in order
and **never touches the original file** — it writes a new directory, so a
migration that turns out to be wrong costs nothing.

Every migration requires two tests: that a fixed golden file in the old format
upgrades, and that the upgraded adapter's ``apply()`` output is numerically
identical to the original's. Without the second, a migration can appear to work
while silently degrading quality.

There are no migrations yet — schema 1 is the first. The machinery is here
because the first migration is the worst possible time to design it: by then
there are files in the wild and no way to test the design against them.
"""

from __future__ import annotations

from typing import Any, Protocol

__all__ = ["MIGRATIONS", "Migration", "migration_path", "upgrade_manifest"]


class Migration(Protocol):
    """One step from schema N-1 to N."""

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return the manifest at the next schema version."""
        ...


#: Keyed by the version each migration *produces*. Empty until schema 2.
MIGRATIONS: dict[int, Migration] = {}


def migration_path(from_schema: int, to_schema: int) -> list[int]:
    """Which migrations run, in order, to get from one version to another.

    Raises:
        AdapterSchemaVersion: When a step in the chain is missing. A partial
            upgrade is worse than a refused one: it produces a file that claims
            a version whose invariants it does not hold.
    """
    from rebasis.errors import AdapterSchemaVersion

    if to_schema <= from_schema:
        return []

    steps = list(range(from_schema + 1, to_schema + 1))
    missing = [step for step in steps if step not in MIGRATIONS]
    if missing:
        raise AdapterSchemaVersion(
            f"No migration is registered for schema {missing[0]}.",
            hint=(
                "This release cannot upgrade the file. Re-fit the adapter with "
                "`rebasis fit`, which always writes the current schema."
            ),
            context={"count": missing[0]},
        )
    return steps


def upgrade_manifest(payload: dict[str, Any], *, to_schema: int) -> dict[str, Any]:
    """Apply every migration between the payload's schema and ``to_schema``.

    Pure: the input is not modified, so a failed migration leaves the caller
    holding exactly what it had.
    """
    current = int(payload.get("schema", 0))
    upgraded = dict(payload)
    for step in migration_path(current, to_schema):
        upgraded = MIGRATIONS[step](upgraded)
        upgraded["schema"] = step
    return upgraded
