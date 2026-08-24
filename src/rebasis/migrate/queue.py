"""The migration work queue.

A purpose-built SQLite table rather than a general-purpose queue: the custom
state machine, the priority score and ``rebasis status``'s queries all need
columns a generic queue does not have, and bending one into shape would take
longer than the ~150 lines of SQL it costs to write.

**Priority is what makes gradual migration worth doing.** Records the user
actually reads move first, so quality improves where they will notice it while
the long tail catches up in the background. With ``--priority access`` the score
comes from access frequency; otherwise insertion order is used, which at least
keeps the job deterministic and resumable.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rebasis.migrate.states import ItemState, JobState

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from rebasis.manifest import ManifestDB

__all__ = ["JobQueue", "QueueStats"]


@dataclass(slots=True)
class QueueStats:
    """A snapshot of progress — what ``rebasis status`` prints."""

    total: int
    pending: int
    shadowed: int
    done: int
    failed: int
    skipped: int

    @property
    def completed_fraction(self) -> float:
        """Fraction finished, counting skips as resolved."""
        if self.total == 0:
            return 1.0
        return (self.done + self.skipped) / self.total

    @property
    def remaining(self) -> int:
        """Records still to process."""
        return self.pending + self.shadowed


class JobQueue:
    """Durable work queue for one migration job."""

    def __init__(self, db: ManifestDB, job_id: str) -> None:
        self.db = db
        self.job_id = job_id

    def enqueue(self, ids: Sequence[str], priorities: Mapping[str, float] | None = None) -> int:
        """Add records to the queue.

        ``INSERT OR IGNORE`` so that re-enqueuing after an interruption is
        harmless: the queue is the checkpoint, and re-running the setup step must
        not reset progress.
        """
        now = _now()
        rows = [
            (self.job_id, record_id, ItemState.PENDING, (priorities or {}).get(record_id, 0.0), now)
            for record_id in ids
        ]
        with self.db.transaction() as connection:
            connection.executemany(
                """
                INSERT OR IGNORE INTO job_items
                    (job_id, record_id, state, priority, updated_utc)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def next_batch(self, size: int) -> list[str]:
        """The next records to process, highest priority first."""
        rows = self.db.query(
            """
            SELECT record_id FROM job_items
            WHERE job_id = ? AND state IN (?, ?)
            ORDER BY priority DESC, record_id ASC
            LIMIT ?
            """,
            (self.job_id, ItemState.PENDING, ItemState.SHADOWED, size),
        )
        return [str(row["record_id"]) for row in rows]

    def mark(self, ids: Sequence[str], state: ItemState, *, error_code: str | None = None) -> None:
        """Move records to a state. One transaction per batch, not per record.

        Per-record transactions would put an fsync in the inner loop — the same
        hot-loop mistake as logging once per record instead of once per batch —
        and batching them is why the durability settings can afford to be strict.
        """
        now = _now()
        with self.db.transaction() as connection:
            connection.executemany(
                """
                UPDATE job_items
                SET state = ?, updated_utc = ?, error_code = ?,
                    attempts = attempts + CASE WHEN ? = 'failed' THEN 1 ELSE 0 END
                WHERE job_id = ? AND record_id = ?
                """,
                [(state, now, error_code, state, self.job_id, record_id) for record_id in ids],
            )

    def stats(self) -> QueueStats:
        """Current progress."""
        rows = self.db.query(
            "SELECT state, COUNT(*) AS n FROM job_items WHERE job_id = ? GROUP BY state",
            (self.job_id,),
        )
        counts = {str(row["state"]): int(row["n"]) for row in rows}
        return QueueStats(
            total=sum(counts.values()),
            pending=counts.get(ItemState.PENDING, 0),
            shadowed=counts.get(ItemState.SHADOWED, 0),
            done=counts.get(ItemState.DONE, 0),
            failed=counts.get(ItemState.FAILED, 0),
            skipped=counts.get(ItemState.SKIPPED, 0),
        )

    def failed_records(self, limit: int = 100) -> list[tuple[str, str | None, int]]:
        """Records that failed, with their error code and attempt count."""
        rows = self.db.query(
            """
            SELECT record_id, error_code, attempts FROM job_items
            WHERE job_id = ? AND state = ?
            ORDER BY record_id LIMIT ?
            """,
            (self.job_id, ItemState.FAILED, limit),
        )
        return [(str(r["record_id"]), r["error_code"], int(r["attempts"])) for r in rows]

    def reset_failed(self) -> int:
        """Return failed records to pending, so a resume retries them."""
        with self.db.transaction() as connection:
            cursor = connection.execute(
                "UPDATE job_items SET state = ?, updated_utc = ? WHERE job_id = ? AND state = ?",
                (ItemState.PENDING, _now(), self.job_id, ItemState.FAILED),
            )
            return int(cursor.rowcount)

    def iter_done(self, batch_size: int = 1000) -> Iterator[list[str]]:
        """Stream the completed records — what rollback walks."""
        offset = 0
        while True:
            rows = self.db.query(
                """
                SELECT record_id FROM job_items
                WHERE job_id = ? AND state = ?
                ORDER BY record_id LIMIT ? OFFSET ?
                """,
                (self.job_id, ItemState.DONE, batch_size, offset),
            )
            if not rows:
                return
            yield [str(row["record_id"]) for row in rows]
            offset += len(rows)


def set_job_state(
    db: ManifestDB, job_id: str, state: JobState, *, error_code: str | None = None
) -> None:
    """Record a job's state transition."""
    with db.transaction() as connection:
        connection.execute(
            "UPDATE jobs SET state = ?, updated_utc = ?, error_code = ? WHERE job_id = ?",
            (state, _now(), error_code, job_id),
        )


def _now() -> str:
    return datetime.datetime.now(tz=datetime.UTC).isoformat()
