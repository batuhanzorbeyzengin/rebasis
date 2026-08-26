"""Stopping a migration from outside it, and starting it again.

Killing the process was always safe — the queue is the checkpoint and a shadow
is written before the vector it copies is overwritten — but it leaves the store
holding a batch nobody verified. `rebasis pause` stops at a batch boundary
instead, which is a different guarantee and the reason the command exists.

The mechanism is one column, `jobs.pause_requested`, and everything below is
about the two properties that make writing it from a second process safe:

* **A request is not a state.** Only the engine says where a job *is*. Between
  the request and the engine reading it the job is still running, and a second
  process writing `state` would both claim a stop that had not happened and race
  the engine over the same column.
* **A request never outlives the run it was meant for.** It is cleared when a
  run ends and again when one starts, so a request left behind by a crash cannot
  silently pause the next run.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import numpy as np
import pytest
from typer.testing import CliRunner

from rebasis.cli import app
from rebasis.core import ProcrustesAdapter, l2_normalize
from rebasis.manifest import JobRow, ManifestDB, manifest_path
from rebasis.migrate import (
    JobState,
    MigrationEngine,
    clear_pause_request,
    pause_requested,
    request_pause,
    set_job_state,
)
from rebasis.store import MemoryStore

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

runner = CliRunner()
WIDE = {"COLUMNS": "220"}

DIM = 16
N = 64
BATCH = 8


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(5)


def build_engine(tmp_path: Path, rng: np.random.Generator) -> MigrationEngine:
    """A job over an in-memory store, queued and ready to run.

    No store URI: the engine reopens the store from its URI to re-verify
    durability when a job completes, and a fresh ``memory://`` is a fresh empty
    one, which would fail a check none of these tests is about.
    """
    vectors = l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))
    rotation = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype(np.float32)
    adapter = ProcrustesAdapter.fit(vectors, l2_normalize(vectors @ rotation.T))

    ids = [f"doc-{i:04d}" for i in range(N)]
    engine = MigrationEngine(
        db=ManifestDB(manifest_path(tmp_path / "state")),
        store=MemoryStore(ids, vectors, [f"text {i}" for i in range(N)]),
        adapter=adapter,
        shadow_root=tmp_path / "shadow",
        batch_size=BATCH,
        power_aware=False,
    )
    engine.prepare(ids)
    return engine


@pytest.fixture
def engine(tmp_path: Path, rng: np.random.Generator) -> MigrationEngine:
    return build_engine(tmp_path, rng)


def force_flag(db: ManifestDB, job_id: str, value: int) -> None:
    """Write the column directly, bypassing the ``state = 'running'`` guard.

    This is what a crash between the request and the engine reading it leaves
    behind, and there is no supported call that produces it — which is exactly
    why the state it creates needs a test.
    """
    with db.transaction() as connection:
        connection.execute("UPDATE jobs SET pause_requested = ? WHERE job_id = ?", (value, job_id))


class TestTheEngineHonoursIt:
    def test_it_stops_at_the_next_batch_boundary(self, engine: MigrationEngine) -> None:
        """Requested during the first batch, honoured before the second.

        `on_batch` fires once a batch has finished, so a request made from it is
        exactly the case the design is for: made while a batch was in flight,
        and honoured at the boundary rather than one batch later.
        """
        batches = 0

        def request_after_the_first(_: int) -> None:
            nonlocal batches
            batches += 1
            if batches == 1:
                request_pause(engine.db, engine.job_id)

        result = engine.run(on_batch=request_after_the_first)

        assert result.state is JobState.PAUSED
        assert result.processed == BATCH, "it ran a batch it had been told to stop before"
        assert result.pause_reason == "a pause was requested"

    def test_the_rest_of_the_queue_is_still_there(self, engine: MigrationEngine) -> None:
        """A pause is not a cancellation: what was not migrated is still queued."""

        def request_immediately(_: int) -> None:
            request_pause(engine.db, engine.job_id)

        engine.run(on_batch=request_immediately)

        assert engine.queue.stats().pending == N - BATCH

    def test_it_finishes_normally_when_nobody_asks(self, engine: MigrationEngine) -> None:
        result = engine.run()

        assert result.state is JobState.COMPLETED
        assert result.processed == N


class TestTheRequestNeverOutlivesItsRun:
    def test_a_honoured_request_is_cleared(self, engine: MigrationEngine) -> None:
        """Otherwise `status` would keep showing a stop that already happened."""

        def request_immediately(_: int) -> None:
            request_pause(engine.db, engine.job_id)

        engine.run(on_batch=request_immediately)

        assert pause_requested(engine.db, engine.job_id) is False

    def test_a_paused_job_resumes_all_the_way(self, engine: MigrationEngine) -> None:
        def request_immediately(_: int) -> None:
            request_pause(engine.db, engine.job_id)

        engine.run(on_batch=request_immediately)
        second = engine.run()

        assert second.state is JobState.COMPLETED
        assert second.processed == N - BATCH

    def test_a_request_left_by_a_crash_does_not_pause_the_next_run(
        self, engine: MigrationEngine
    ) -> None:
        """The engine clears the column as it starts, not as it is asked to start.

        A process killed between `rebasis pause` and the engine reading it
        leaves the flag set. Clearing it in the engine rather than in `resume`
        is what makes that recoverable by any path that runs the job again —
        including `migrate --resume`, which never went through `resume`.
        """
        force_flag(engine.db, engine.job_id, 1)

        result = engine.run()

        assert result.state is JobState.COMPLETED
        assert result.processed == N


class TestTheRequestIsGuardedOnState:
    def test_a_pending_job_cannot_be_paused(self, engine: MigrationEngine) -> None:
        """There is nothing to interrupt, and a flag set now would fire later."""
        assert request_pause(engine.db, engine.job_id) is False
        assert pause_requested(engine.db, engine.job_id) is False

    def test_a_running_job_can(self, engine: MigrationEngine) -> None:
        set_job_state(engine.db, engine.job_id, JobState.RUNNING)

        assert request_pause(engine.db, engine.job_id) is True
        assert pause_requested(engine.db, engine.job_id) is True

    def test_a_job_that_does_not_exist_is_not_an_exception(self, engine: MigrationEngine) -> None:
        """The CLI turns this into a message; the helper just says no."""
        assert request_pause(engine.db, "job-nonexistent") is False
        assert pause_requested(engine.db, "job-nonexistent") is False

    def test_clearing_is_idempotent(self, engine: MigrationEngine) -> None:
        clear_pause_request(engine.db, engine.job_id)
        clear_pause_request(engine.db, engine.job_id)

        assert pause_requested(engine.db, engine.job_id) is False


class TestTheColumnSurvivesAnUpgrade:
    def test_a_schema_two_job_reads_as_not_requested(self, tmp_path: Path) -> None:
        """A job written before the column existed was never asked to stop."""
        path = tmp_path / "manifest.db"
        connection = sqlite3.connect(path)
        connection.executescript("""
            CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY, created_utc TEXT NOT NULL, updated_utc TEXT NOT NULL,
                state TEXT NOT NULL, store_backend TEXT NOT NULL, adapter_path TEXT NOT NULL,
                adapter_type TEXT NOT NULL, total_records INTEGER NOT NULL,
                batch_size INTEGER NOT NULL, keep_original INTEGER NOT NULL DEFAULT 1,
                config_json TEXT NOT NULL DEFAULT '{}', error_code TEXT,
                store_uri TEXT NOT NULL DEFAULT ''
            );
            INSERT INTO jobs VALUES
                ('job-old', 'x', 'x', 'paused', 'chroma', '', 'procrustes', 5, 8, 1, '{}',
                 NULL, 'memory://');
            PRAGMA user_version = 2;
        """)
        connection.commit()
        connection.close()

        with ManifestDB(path) as db:
            row = db.query_one("SELECT * FROM jobs WHERE job_id = ?", ("job-old",))
            assert row is not None
            assert JobRow.from_row(row).pause_requested is False
            assert pause_requested(db, "job-old") is False


class TestTheCommand:
    def test_it_records_the_request(self, engine: MigrationEngine, tmp_path: Path) -> None:
        set_job_state(engine.db, engine.job_id, JobState.RUNNING)
        engine.db.close()

        result = runner.invoke(
            app, ["pause", engine.job_id, "--state-dir", str(tmp_path / "state")]
        )

        assert result.exit_code == 0, result.output
        assert "Pause requested" in result.output
        with ManifestDB(manifest_path(tmp_path / "state")) as db:
            assert pause_requested(db, engine.job_id) is True

    def test_it_writes_an_audit_record(self, engine: MigrationEngine, tmp_path: Path) -> None:
        """A job that stopped because somebody asked is not the same as one that
        stopped for memory pressure, and the audit trail is where that lives."""
        set_job_state(engine.db, engine.job_id, JobState.RUNNING)
        engine.db.close()

        runner.invoke(app, ["pause", engine.job_id, "--state-dir", str(tmp_path / "state")])

        with ManifestDB(manifest_path(tmp_path / "state")) as db:
            actions = [
                row["action"]
                for row in db.query(
                    "SELECT action FROM audit_records WHERE subject = ?", (engine.job_id,)
                )
            ]
        assert "migrate.pause.requested" in actions

    def test_a_job_that_is_not_running_is_refused(
        self, engine: MigrationEngine, tmp_path: Path
    ) -> None:
        engine.db.close()

        result = runner.invoke(
            app, ["pause", engine.job_id, "--state-dir", str(tmp_path / "state")]
        )

        assert result.exit_code != 0
        assert "not running" in result.output

    def test_an_unknown_job_says_which_one(self, engine: MigrationEngine, tmp_path: Path) -> None:
        engine.db.close()

        result = runner.invoke(app, ["pause", "job-nope", "--state-dir", str(tmp_path / "state")])

        assert result.exit_code != 0
        assert "job-nope" in result.output

    def test_a_state_directory_that_is_not_one_says_so(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["pause", "job-x", "--state-dir", str(tmp_path / "nothing")])

        assert result.exit_code != 0
        assert "no rebasis state" in result.output.lower()


class TestStatusShowsIt:
    def test_the_table_says_pausing(self, engine: MigrationEngine, tmp_path: Path) -> None:
        """A job asked to stop is still running, and saying only "running" hides
        the one fact the person who just asked is waiting on."""
        set_job_state(engine.db, engine.job_id, JobState.RUNNING)
        request_pause(engine.db, engine.job_id)
        engine.db.close()

        result = runner.invoke(app, ["status", "--state-dir", str(tmp_path / "state")], env=WIDE)

        assert result.exit_code == 0, result.output
        assert "pausing" in result.output

    def test_json_carries_it_as_its_own_field(
        self, engine: MigrationEngine, tmp_path: Path
    ) -> None:
        """A script branching on `state == "running"` must keep working: this is
        a second fact about a running job, not a different state."""
        import json

        set_job_state(engine.db, engine.job_id, JobState.RUNNING)
        request_pause(engine.db, engine.job_id)
        engine.db.close()

        result = runner.invoke(
            app, ["status", "--state-dir", str(tmp_path / "state"), "--json"], env=WIDE
        )

        payload = json.loads(result.stdout)
        assert payload[0]["state"] == "running"
        assert payload[0]["pause_requested"] is True

    def test_an_untouched_job_reports_false(self, engine: MigrationEngine, tmp_path: Path) -> None:
        import json

        engine.db.close()

        result = runner.invoke(
            app, ["status", "--state-dir", str(tmp_path / "state"), "--json"], env=WIDE
        )

        payload = json.loads(result.stdout)
        assert payload[0]["pause_requested"] is False


class TestResume:
    """`resume` is `migrate --resume` under the name people reach for.

    That it actually finishes a job is proved end to end against a real store in
    `tests/e2e/test_cli_flow.py`; what is here is the argument handling, which
    is where a forwarding command goes wrong.
    """

    def test_an_unknown_job_is_refused(self, engine: MigrationEngine, tmp_path: Path) -> None:
        engine.db.close()

        result = runner.invoke(
            app, ["resume", "job-nope", "--state-dir", str(tmp_path / "state"), "--yes"]
        )

        assert result.exit_code != 0
        assert "job-nope" in result.output
