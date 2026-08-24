"""The manifest database.

One SQLite file holds migration state **and the audit trail**. The audit records
are what make the durability settings non-negotiable: a log may be lost, an audit
record may not.

**The pragma choice, stated explicitly**, because this area is genuinely
confusing and the folklore is wrong in both directions:

* ``journal_mode=DELETE`` (the default) with ``synchronous=FULL`` keeps the
  database consistent but **can still lose the last commits** to a power cut;
  real durability there needs ``EXTRA``.
* ``journal_mode=WAL`` with ``synchronous=NORMAL`` survives an application crash
  and an OS crash, but a power cut at the wrong moment can lose the last
  transactions.
* ``journal_mode=WAL`` with ``synchronous=FULL`` syncs the WAL after every
  commit. In WAL mode ``EXTRA`` and ``FULL`` are the same thing.

rebasis takes **WAL + FULL**. The cost argument that usually justifies weaker
settings does not apply here: the hot-loop rule keeps manifest writes per-batch
and per-decision rather than per-record, so an fsync divided across a 256-record
batch's embedding time is unmeasurable.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rebasis.errors import StorageError
from rebasis.observability import Events, get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Self

__all__ = ["SCHEMA_VERSION", "ManifestDB"]

log = get_logger(__name__)

#: Current manifest schema. Forward-only migrations, tracked in
#: ``PRAGMA user_version``.
SCHEMA_VERSION = 2

_PRAGMAS = (
    # Persistent: set once and stored in the file header.
    ("journal_mode", "WAL"),
    # Per connection.
    ("synchronous", "FULL"),
    # SQLite leaves this off for backwards compatibility; rebasis wants it on.
    ("foreign_keys", "ON"),
    # Wait rather than fail immediately when another process holds a write lock.
    ("busy_timeout", "5000"),
)


class ManifestDB:
    """Local state: migration jobs, checkpoints and the audit trail."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: sqlite3.Connection | None = None

    # ── connection ────────────────────────────────────────────────────

    @property
    def connection(self) -> sqlite3.Connection:
        """The open connection, created and configured on first use."""
        if self._connection is None:
            self._connection = self._connect()
        return self._connection

    def _connect(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(self.path, isolation_level=None)
        except sqlite3.Error as exc:
            raise StorageError(
                f"Could not open the manifest at {self.path.name}.",
                hint="Check that the directory is writable.",
                cause=exc,
            ) from exc

        connection.row_factory = sqlite3.Row
        for pragma, value in _PRAGMAS:
            connection.execute(f"PRAGMA {pragma} = {value}")

        self._warn_on_network_filesystem()
        self._migrate(connection)
        return connection

    def _warn_on_network_filesystem(self) -> None:
        """WAL needs shared memory between processes, which network shares lack.

        Detection is a best effort — there is no portable way to ask — so this
        warns rather than refusing. A wrong guess must not block a legitimate run.
        """
        try:
            resolved = str(self.path.resolve())
        except OSError:
            return
        for marker in ("/net/", "/mnt/nfs", "//"):
            if marker in resolved:
                log.warning(
                    Events.STORAGE_WRITE_COMPLETED,
                    duration_ms=0,
                )
                return

    def close(self) -> None:
        """Close the connection. Safe to call more than once."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ── schema ────────────────────────────────────────────────────────

    def _migrate(self, connection: sqlite3.Connection) -> None:
        """Apply forward-only migrations, backing up first.

        The backup exists because this file holds audit records, which are never
        supposed to disappear. ``VACUUM INTO`` produces a consistent copy without
        holding a lock.
        """
        current = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if current > SCHEMA_VERSION:
            raise StorageError(
                f"The manifest uses schema {current}, but this release understands "
                f"at most {SCHEMA_VERSION}.",
                hint="Upgrade rebasis: `pip install --upgrade rebasis`.",
            )
        if current == SCHEMA_VERSION:
            return

        if current > 0:
            backup = self.path.with_suffix(f"{self.path.suffix}.bak.{current}")
            connection.execute("VACUUM INTO ?", (str(backup),))

        for version in range(current + 1, SCHEMA_VERSION + 1):
            connection.executescript(_MIGRATIONS[version])
            connection.execute(f"PRAGMA user_version = {version}")

    # ── helpers ───────────────────────────────────────────────────────

    def execute(self, sql: str, parameters: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        """Run one statement."""
        return self.connection.execute(sql, parameters)

    def query(self, sql: str, parameters: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        """Run a query and return every row."""
        return list(self.connection.execute(sql, parameters).fetchall())

    def query_one(self, sql: str, parameters: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        """Run a query and return the first row, if any."""
        row: sqlite3.Row | None = self.connection.execute(sql, parameters).fetchone()
        return row

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """A write transaction, committed on success and rolled back on failure.

        ``BEGIN IMMEDIATE`` rather than the default deferred begin: it takes the
        write lock up front, so a conflict surfaces here instead of at commit
        time when work has already been done.
        """
        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        else:
            connection.execute("COMMIT")

    def integrity_check(self) -> list[str]:
        """Run SQLite's own checks — what ``rebasis doctor`` reports."""
        problems: list[str] = [
            str(row[0])
            for row in self.connection.execute("PRAGMA integrity_check")
            if row[0] != "ok"
        ]
        problems.extend(
            f"foreign key violation in {row[0]}"
            for row in self.connection.execute("PRAGMA foreign_key_check")
        )
        return problems

    def pragma_settings(self) -> dict[str, Any]:
        """The durability settings actually in force, for diagnostics."""
        return {
            name: self.connection.execute(f"PRAGMA {name}").fetchone()[0]
            for name in ("journal_mode", "synchronous", "foreign_keys", "user_version")
        }


#: Numbered, forward-only migrations.
_MIGRATIONS: dict[int, str] = {
    1: """
    -- The audit trail. Distinct from logs: lossless, never sampled,
    -- schema-versioned, never deleted automatically.
    CREATE TABLE IF NOT EXISTS audit_records (
        seq             INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_utc          TEXT    NOT NULL,
        run_id          TEXT    NOT NULL,
        actor           TEXT    NOT NULL,
        action          TEXT    NOT NULL,
        subject         TEXT,
        inputs_json     TEXT    NOT NULL,
        outputs_json    TEXT    NOT NULL,
        rebasis_version TEXT    NOT NULL,
        env_json        TEXT    NOT NULL,
        prev_hash       TEXT    NOT NULL,
        hash            TEXT    NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_records(action);
    CREATE INDEX IF NOT EXISTS idx_audit_run ON audit_records(run_id);
    CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_records(ts_utc);

    -- Migration jobs and their state machine.
    CREATE TABLE IF NOT EXISTS jobs (
        job_id          TEXT PRIMARY KEY,
        created_utc     TEXT NOT NULL,
        updated_utc     TEXT NOT NULL,
        state           TEXT NOT NULL,
        store_backend   TEXT NOT NULL,
        adapter_path    TEXT NOT NULL,
        adapter_type    TEXT NOT NULL,
        total_records   INTEGER NOT NULL,
        batch_size      INTEGER NOT NULL,
        keep_original   INTEGER NOT NULL DEFAULT 1,
        config_json     TEXT NOT NULL DEFAULT '{}',
        error_code      TEXT
    );

    -- The work queue. A purpose-built table rather than a general queue: the
    -- state machine, the priority score and `rebasis status` all need columns a
    -- generic queue does not have.
    CREATE TABLE IF NOT EXISTS job_items (
        job_id      TEXT    NOT NULL,
        record_id   TEXT    NOT NULL,
        state       TEXT    NOT NULL DEFAULT 'pending',
        priority    REAL    NOT NULL DEFAULT 0.0,
        attempts    INTEGER NOT NULL DEFAULT 0,
        error_code  TEXT,
        updated_utc TEXT,
        PRIMARY KEY (job_id, record_id),
        FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_items_pending
        ON job_items(job_id, state, priority DESC);

    -- Shadow copies, for rollback.
    CREATE TABLE IF NOT EXISTS shadows (
        job_id       TEXT NOT NULL,
        path         TEXT NOT NULL,
        record_count INTEGER NOT NULL,
        dim          INTEGER NOT NULL,
        precision    TEXT NOT NULL DEFAULT 'float32',
        block_hashes TEXT NOT NULL,
        created_utc  TEXT NOT NULL,
        PRIMARY KEY (job_id, path),
        FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
    );
    """,
    2: """
    -- `rollback` needs to reopen the store the job wrote to. Recording the URI
    -- with the job means the user does not have to remember it months later,
    -- which is exactly when a rollback is wanted.
    ALTER TABLE jobs ADD COLUMN store_uri TEXT NOT NULL DEFAULT '';
    """,
}
