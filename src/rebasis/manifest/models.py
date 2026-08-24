"""Typed rows for the manifest tables.

``sqlite3.Row`` is a mapping of strings to whatever SQLite stored, so every
caller ends up writing ``str(row["job_id"])`` and ``bool(row["keep_original"])``
by hand. That is fine once and a source of drift by the fifth time: a column
renamed in ``db.py`` breaks at whichever call site runs first, with a
``KeyError`` that names the column but not the schema it came from.

These dataclasses do the conversion in one place and give mypy something to
check. They are deliberately **not** an ORM: the queries stay hand-written in
``db.py`` where they are read together with the schema they run against, and
these only describe a row that has already been fetched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import sqlite3

__all__ = ["AuditRow", "JobItemRow", "JobRow", "ShadowRow"]


def _get(row: sqlite3.Row, key: str) -> Any:
    """Read a column, or ``None`` when the row does not have it.

    Rows come from schemas of different ages: a job written before schema 2 has
    no ``store_uri``, and a listing that raised on it would make old jobs
    invisible rather than incomplete.
    """
    return row[key] if key in row.keys() else None  # noqa: SIM118 - sqlite3.Row has no __contains__


def _text(row: sqlite3.Row, key: str, default: str = "") -> str:
    value = _get(row, key)
    return default if value is None else str(value)


def _number(row: sqlite3.Row, key: str, default: int = 0) -> int:
    value = _get(row, key)
    return default if value is None else int(value)


def _payload(row: sqlite3.Row, key: str) -> dict[str, Any]:
    """Parse a JSON column, treating unreadable content as empty.

    Not fatal, on purpose: these columns carry context around a record whose
    identity and hash are in real columns. A corrupted `config_json` should not
    make a migration job unlistable.
    """
    raw = _text(row, key, "{}")
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


#: ``JobState.ROLLED_BACK``, as a literal. ``manifest`` sits below ``migrate``
#: in the layer contract, so the enum cannot be imported here; the value is
#: pinned by ``tests/unit/test_manifest_schema.py`` instead.
_ROLLED_BACK = "rolled_back"


@dataclass(frozen=True, slots=True)
class JobRow:
    """One row of ``jobs`` — a migration and its state."""

    job_id: str
    created_utc: str
    updated_utc: str
    state: str
    store_backend: str
    store_uri: str
    adapter_path: str
    adapter_type: str
    total_records: int
    batch_size: int
    keep_original: bool
    config: dict[str, Any]
    error_code: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> JobRow:
        """Build from a fetched row."""
        return cls(
            job_id=_text(row, "job_id"),
            created_utc=_text(row, "created_utc"),
            updated_utc=_text(row, "updated_utc"),
            state=_text(row, "state"),
            store_backend=_text(row, "store_backend"),
            # Added in schema 2, so an older job has none. The default keeps a
            # pre-upgrade job listable instead of raising on a missing column.
            store_uri=_text(row, "store_uri"),
            adapter_path=_text(row, "adapter_path"),
            adapter_type=_text(row, "adapter_type"),
            total_records=_number(row, "total_records"),
            batch_size=_number(row, "batch_size"),
            keep_original=bool(_number(row, "keep_original", 1)),
            config=_payload(row, "config_json"),
            error_code=_text(row, "error_code"),
        )

    @property
    def reversible(self) -> bool:
        """Whether a rollback is possible — the question `status` answers.

        Two ways to lose it: the job never kept a shadow, or it already spent
        one. ``rolled_back`` is terminal in ``JOB_TRANSITIONS``, so a job that
        reached it cannot be rolled back again and reporting "available" for it
        offers something that is not there.
        """
        return self.keep_original and self.state != _ROLLED_BACK


@dataclass(frozen=True, slots=True)
class JobItemRow:
    """One row of ``job_items`` — a record's place in the queue."""

    job_id: str
    record_id: str
    state: str
    priority: float
    attempts: int
    error_code: str
    updated_utc: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> JobItemRow:
        """Build from a fetched row."""
        return cls(
            job_id=_text(row, "job_id"),
            record_id=_text(row, "record_id"),
            state=_text(row, "state"),
            priority=float(_get(row, "priority") or 0.0),
            attempts=_number(row, "attempts"),
            error_code=_text(row, "error_code"),
            updated_utc=_text(row, "updated_utc"),
        )


@dataclass(frozen=True, slots=True)
class ShadowRow:
    """One row of ``shadows`` — a saved copy of what a batch overwrote."""

    job_id: str
    path: str
    record_count: int
    dim: int
    precision: str
    block_hashes: list[str]
    created_utc: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> ShadowRow:
        """Build from a fetched row."""
        raw = _text(row, "block_hashes", "[]")
        try:
            hashes = json.loads(raw)
        except ValueError:
            hashes = []
        return cls(
            job_id=_text(row, "job_id"),
            path=_text(row, "path"),
            record_count=_number(row, "record_count"),
            dim=_number(row, "dim"),
            precision=_text(row, "precision", "float32"),
            block_hashes=[str(h) for h in hashes] if isinstance(hashes, list) else [],
            created_utc=_text(row, "created_utc"),
        )

    @property
    def bit_identical_restore(self) -> bool:
        """Whether a rollback returns the original bytes exactly.

        Only float32 does. A quantised shadow restores something close, and
        "close" is not what rollback promises.
        """
        return self.precision == "float32"


@dataclass(frozen=True, slots=True)
class AuditRow:
    """One row of ``audit_records``."""

    seq: int
    ts_utc: str
    run_id: str
    actor: str
    action: str
    subject: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    rebasis_version: str
    env: dict[str, Any]
    prev_hash: str
    hash: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> AuditRow:
        """Build from a fetched row."""
        return cls(
            seq=_number(row, "seq"),
            ts_utc=_text(row, "ts_utc"),
            run_id=_text(row, "run_id"),
            actor=_text(row, "actor"),
            action=_text(row, "action"),
            subject=_text(row, "subject"),
            inputs=_payload(row, "inputs_json"),
            outputs=_payload(row, "outputs_json"),
            rebasis_version=_text(row, "rebasis_version"),
            env=_payload(row, "env_json"),
            prev_hash=_text(row, "prev_hash"),
            hash=_text(row, "hash"),
        )
