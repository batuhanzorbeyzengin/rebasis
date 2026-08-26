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

    import numpy as np

    from rebasis.manifest import ManifestDB

__all__ = ["JobQueue", "QueueStats", "clear_pause_request", "pause_requested", "request_pause"]


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

    def sample_pending(self, size: int, rng: np.random.Generator) -> list[str]:
        """A uniform sample of the records still to be migrated.

        Uniform over what is *left*, not over the queue's head. A refit fitted
        on the highest-priority records still pending would be fitted on a
        slice of a slice — and what it is about to be applied to is the whole
        remainder.

        Reservoir sampling over a streamed scan rather than ``ORDER BY
        RANDOM()``: SQLite's ``RANDOM()`` cannot be seeded, and a migration that
        made a different decision on a re-run of the same job would not be
        reproducible from the audit trail. One pass of record ids, and only when
        a refit is due — every 50,000 records by default, against a scan of a
        column that is already indexed.
        """
        reservoir: list[str] = []
        seen = 0
        for chunk in self._iter_pending():
            for record_id in chunk:
                if len(reservoir) < size:
                    reservoir.append(record_id)
                else:
                    position = int(rng.integers(0, seen + 1))
                    if position < size:
                        reservoir[position] = record_id
                seen += 1
        return reservoir

    def _iter_pending(self, batch_size: int = 1000) -> Iterator[list[str]]:
        """Stream the records still to be migrated, in a stable order.

        ``SHADOWED`` counts as pending for the same reason ``next_batch`` takes
        it: a record whose original was copied but whose write never landed is
        still carrying its old vector.
        """
        offset = 0
        while True:
            rows = self.db.query(
                """
                SELECT record_id FROM job_items
                WHERE job_id = ? AND state IN (?, ?)
                ORDER BY record_id LIMIT ? OFFSET ?
                """,
                (self.job_id, ItemState.PENDING, ItemState.SHADOWED, batch_size, offset),
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


# ── pause requests ───────────────────────────────────────────────────────────
#
# A request is not a state, and keeping them apart is what makes this safe to
# write from a second process.
#
# `state` says where the job *is*, and the engine is the only thing that knows:
# between `rebasis pause` returning and the current batch finishing, the job is
# still RUNNING and writing PAUSED from outside would claim a stop that has not
# happened. Worse, both processes would then be writing one column — and the
# engine's own `_finish` would overwrite whatever the other wrote.
#
# So this is a separate column with one writer and one reader in each direction:
# the CLI sets it, the engine reads it, the engine clears it. `status` reads it
# too, because a job that has been asked to stop and has not stopped yet is a
# thing a user needs to be able to see.


def request_pause(db: ManifestDB, job_id: str) -> bool:
    """Ask a running job to stop after its current batch.

    Guarded on ``state = 'running'`` in the statement rather than checked first,
    so a job that finishes between the check and the write is not left carrying
    a request nothing will ever read. Returns whether the request was recorded;
    ``False`` means the job does not exist or is not running, and the caller has
    the row it needs to say which.
    """
    with db.transaction() as connection:
        cursor = connection.execute(
            "UPDATE jobs SET pause_requested = 1, updated_utc = ? WHERE job_id = ? AND state = ?",
            (_now(), job_id, JobState.RUNNING),
        )
        return cursor.rowcount > 0


def clear_pause_request(db: ManifestDB, job_id: str) -> None:
    """Forget a pause request, so a resumed job does not stop immediately.

    Called by the engine as it starts, not by whatever asked it to start: a
    request that outlived the run it was meant for would pause the next one, and
    the engine is the only place that knows a run has actually begun.
    """
    with db.transaction() as connection:
        connection.execute(
            "UPDATE jobs SET pause_requested = 0, updated_utc = ? WHERE job_id = ?",
            (_now(), job_id),
        )


def pause_requested(db: ManifestDB, job_id: str) -> bool:
    """Whether someone has asked this job to stop.

    Read once per batch, so it is a primary-key lookup and nothing more. A job
    row that has gone missing reads as no request: the engine's business is to
    finish, and a vanished row is not an instruction to stop.
    """
    row = db.query_one("SELECT pause_requested FROM jobs WHERE job_id = ?", (job_id,))
    return bool(row["pause_requested"]) if row is not None else False


def _now() -> str:
    return datetime.datetime.now(tz=datetime.UTC).isoformat()
