"""Reading the audit trail."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from rebasis.audit.chain import ChainVerification, verify_chain
from rebasis.audit.record import AuditRecord

if TYPE_CHECKING:
    import sqlite3

    from rebasis.manifest import ManifestDB

__all__ = ["AuditReader"]


class AuditReader:
    """Queries the audit trail."""

    def __init__(self, db: ManifestDB) -> None:
        self.db = db

    def list_records(
        self,
        *,
        action: str | None = None,
        since: str | None = None,
        run_id: str | None = None,
        limit: int = 50,
    ) -> list[AuditRecord]:
        """List records, most recent first."""
        clauses: list[str] = []
        parameters: list[Any] = []
        if action:
            clauses.append("action = ?")
            parameters.append(action)
        if since:
            clauses.append("ts_utc >= ?")
            parameters.append(since)
        if run_id:
            clauses.append("run_id = ?")
            parameters.append(run_id)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.db.query(
            f"SELECT * FROM audit_records {where} ORDER BY seq DESC LIMIT ?",  # noqa: S608 - clauses are literals, values are bound
            (*parameters, limit),
        )
        return [_row_to_record(row) for row in rows]

    def get(self, seq: int) -> AuditRecord | None:
        """One record by sequence number."""
        row = self.db.query_one("SELECT * FROM audit_records WHERE seq = ?", (seq,))
        return _row_to_record(row) if row else None

    def all_records(self) -> list[AuditRecord]:
        """Every record, in chain order."""
        return [
            _row_to_record(row)
            for row in self.db.query("SELECT * FROM audit_records ORDER BY seq ASC")
        ]

    def verify(self) -> ChainVerification:
        """Verify the whole chain.

        Gaps recorded by ``audit prune`` are read from the trail itself, so a
        deliberate prune does not read as tampering.
        """
        records = self.all_records()
        gaps = [
            int(record.outputs.get("gap_after_seq", 0))
            for record in records
            if record.action == "audit.chain.verified" and "gap_after_seq" in record.outputs
        ]
        return verify_chain(records, gap_seqs=gaps)

    def export_jsonl(self) -> str:
        """The whole trail as JSON Lines, for archiving or external analysis."""
        from rebasis.audit.record import canonical_json

        return "\n".join(
            canonical_json(
                {
                    "seq": record.seq,
                    **record.payload(),
                    "hash": record.hash,
                }
            )
            for record in self.all_records()
        )

    def count(self) -> int:
        """Number of records in the trail."""
        row = self.db.query_one("SELECT COUNT(*) AS n FROM audit_records")
        return int(row["n"]) if row else 0

    def actions(self) -> dict[str, int]:
        """Record count per action — what ``audit list`` summarises."""
        return {
            str(row["action"]): int(row["n"])
            for row in self.db.query(
                "SELECT action, COUNT(*) AS n FROM audit_records GROUP BY action ORDER BY n DESC"
            )
        }


def _row_to_record(row: sqlite3.Row) -> AuditRecord:
    return AuditRecord(
        seq=int(row["seq"]),
        ts_utc=str(row["ts_utc"]),
        run_id=str(row["run_id"]),
        actor=str(row["actor"]),
        action=str(row["action"]),
        subject=row["subject"],
        inputs=json.loads(row["inputs_json"]),
        outputs=json.loads(row["outputs_json"]),
        rebasis_version=str(row["rebasis_version"]),
        env=json.loads(row["env_json"]),
        prev_hash=str(row["prev_hash"]),
        hash=str(row["hash"]),
    )
