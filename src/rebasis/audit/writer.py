"""Writing audit records.

Which events produce a record is a **small subset** of those that log. The event
catalogue is the source of truth: a record is written when the event's
``audited`` flag is set.

Appending is serialised through the manifest's write transaction. Two processes
appending concurrently would produce two records claiming the same predecessor,
and the chain would break for a reason that has nothing to do with tampering —
``BEGIN IMMEDIATE`` takes the write lock before the previous hash is read, which
closes that window.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

from rebasis.audit.record import GENESIS_HASH, AuditRecord, EnvironmentSnapshot, canonical_json
from rebasis.observability import EVENT_CATALOG, Events, get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rebasis.manifest import ManifestDB

__all__ = ["AuditWriter", "is_audited"]

log = get_logger(__name__)


def is_audited(action: Events | str) -> bool:
    """Whether this event produces an audit record — the event catalogue decides."""
    try:
        event = Events(action)
    except ValueError:
        return False
    spec = EVENT_CATALOG.get(event)
    return bool(spec and spec.audited)


class AuditWriter:
    """Appends records to the audit trail."""

    def __init__(self, db: ManifestDB, *, run_id: str, actor: str = "cli") -> None:
        self.db = db
        self.run_id = run_id
        self.actor = actor
        self._environment: dict[str, Any] | None = None

    @property
    def environment(self) -> dict[str, Any]:
        """The environment snapshot, captured once per run."""
        if self._environment is None:
            self._environment = EnvironmentSnapshot.capture().to_dict()
        return self._environment

    def write(
        self,
        action: Events | str,
        *,
        inputs: Mapping[str, Any],
        outputs: Mapping[str, Any],
        subject: str | None = None,
        force: bool = False,
    ) -> AuditRecord | None:
        """Append a record, if this action is audited.

        Args:
            action: An event name from :class:`~rebasis.observability.Events`.
            inputs: Everything needed to reproduce the decision — model ids and
                profile fingerprints, sampling strategy and seed, sample size,
                thresholds, metric version. **No document text.**
            outputs: The decision and its metrics.
            subject: The adapter, store or job the record is about.
            force: Write even when the catalogue does not mark this action
                audited. Used by tests; not by production code.

        Returns:
            The sealed record, or ``None`` when the action is not audited.
        """
        if not force and not is_audited(action):
            return None

        from rebasis.__about__ import __version__

        with self.db.transaction() as connection:
            # Read the tip inside the write transaction. Reading it outside
            # would let two processes seal against the same predecessor.
            row = connection.execute(
                "SELECT hash FROM audit_records ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            previous = row["hash"] if row else GENESIS_HASH

            record = AuditRecord(
                ts_utc=datetime.datetime.now(tz=datetime.UTC).isoformat(),
                run_id=self.run_id,
                actor=self.actor,
                action=str(action),
                inputs=dict(inputs),
                outputs=dict(outputs),
                rebasis_version=__version__,
                env=self.environment,
                subject=subject,
            ).sealed(previous)

            cursor = connection.execute(
                """
                INSERT INTO audit_records (
                    ts_utc, run_id, actor, action, subject,
                    inputs_json, outputs_json, rebasis_version, env_json,
                    prev_hash, hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.ts_utc,
                    record.run_id,
                    record.actor,
                    record.action,
                    record.subject,
                    canonical_json(dict(record.inputs)),
                    canonical_json(dict(record.outputs)),
                    record.rebasis_version,
                    canonical_json(dict(record.env)),
                    record.prev_hash,
                    record.hash,
                ),
            )
            seq = int(cursor.lastrowid or 0)

        log.debug(Events.AUDIT_RECORD_WRITTEN, audit_seq=seq, action=str(action))
        from dataclasses import replace

        return replace(record, seq=seq)
