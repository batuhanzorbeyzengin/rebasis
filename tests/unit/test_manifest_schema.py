"""Forward-only manifest migrations.

A schema migration that loses a row loses a migration job's checkpoint, and the
checkpoint is the only thing standing between an interrupted migration and
starting over. So the test that matters is not "does the new column exist" but
"is everything that was there before still there afterwards".
"""

from __future__ import annotations

import sqlite3

import pytest

from rebasis.manifest import SCHEMA_VERSION, ManifestDB

pytestmark = pytest.mark.unit


def v1_database(path) -> None:  # type: ignore[no-untyped-def]
    """A manifest as schema 1 wrote it, before `jobs.store_uri` existed."""
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY, created_utc TEXT NOT NULL, updated_utc TEXT NOT NULL,
            state TEXT NOT NULL, store_backend TEXT NOT NULL, adapter_path TEXT NOT NULL,
            adapter_type TEXT NOT NULL, total_records INTEGER NOT NULL,
            batch_size INTEGER NOT NULL, keep_original INTEGER NOT NULL DEFAULT 1,
            config_json TEXT NOT NULL DEFAULT '{}', error_code TEXT
        );
        CREATE TABLE audit_records (
            seq INTEGER PRIMARY KEY AUTOINCREMENT, ts_utc TEXT NOT NULL, run_id TEXT NOT NULL,
            actor TEXT NOT NULL, action TEXT NOT NULL, subject TEXT, inputs_json TEXT NOT NULL,
            outputs_json TEXT NOT NULL, rebasis_version TEXT NOT NULL, env_json TEXT NOT NULL,
            prev_hash TEXT NOT NULL, hash TEXT NOT NULL
        );
        INSERT INTO jobs VALUES
            ('job-1', 'x', 'x', 'running', 'chroma', '', 'procrustes', 500, 256, 1, '{}', NULL);
        INSERT INTO audit_records
            (ts_utc, run_id, actor, action, inputs_json, outputs_json, rebasis_version,
             env_json, prev_hash, hash)
        VALUES ('x', 'r', 'cli', 'probe.decision.made', '{}', '{}', '0', '{}', 'g', 'h');
        PRAGMA user_version = 1;
    """)
    connection.commit()
    connection.close()


class TestUpgrade:
    def test_an_old_manifest_is_upgraded_in_place(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "manifest.db"
        v1_database(path)

        db = ManifestDB(path)
        assert db.pragma_settings()["user_version"] == SCHEMA_VERSION
        db.close()

    def test_the_rows_survive_the_upgrade(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The queue is the checkpoint. Losing it means starting over."""
        path = tmp_path / "manifest.db"
        v1_database(path)

        db = ManifestDB(path)
        jobs = db.query("SELECT * FROM jobs")
        records = db.query("SELECT * FROM audit_records")

        assert [row["job_id"] for row in jobs] == ["job-1"]
        assert jobs[0]["adapter_type"] == "procrustes"
        assert len(records) == 1
        db.close()

    def test_the_new_column_has_a_usable_default(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """A job written before the column existed has no URI, not a null that
        blows up the first time `rollback` reads it."""
        path = tmp_path / "manifest.db"
        v1_database(path)

        db = ManifestDB(path)
        assert db.query("SELECT store_uri FROM jobs")[0]["store_uri"] == ""
        db.close()

    def test_a_backup_is_taken_before_migrating(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Forward-only means there is no way back except the copy."""
        path = tmp_path / "manifest.db"
        v1_database(path)

        ManifestDB(path).pragma_settings()

        assert list(tmp_path.glob("manifest.db.bak.*"))

    def test_a_fresh_database_is_created_at_the_current_version(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        db = ManifestDB(tmp_path / "new.db")

        assert db.pragma_settings()["user_version"] == SCHEMA_VERSION
        assert db.query("SELECT store_uri FROM jobs") == []
        db.close()


class TestReversibility:
    """`status`'s rollback column, and the literal it rests on."""

    def test_the_rolled_back_literal_matches_the_enum(self) -> None:
        """`manifest` sits below `migrate`, so `JobRow` cannot import JobState
        and spells the value out. Nothing but this test keeps the two together
        if the enum is ever renamed."""
        from rebasis.manifest.models import _ROLLED_BACK
        from rebasis.migrate.states import JobState

        assert JobState.ROLLED_BACK.value == _ROLLED_BACK

    def test_a_spent_shadow_is_not_offered_again(self) -> None:
        """A rolled-back job kept a shadow and already used it. `rolled_back` is
        terminal, so reporting the rollback as available offers what is gone."""
        from rebasis.manifest import JobRow

        done = _job(state="completed", keep_original=True)
        spent = _job(state="rolled_back", keep_original=True)
        never = _job(state="completed", keep_original=False)

        assert JobRow(**done).reversible
        assert not JobRow(**spent).reversible
        assert not JobRow(**never).reversible


def _job(*, state: str, keep_original: bool) -> dict[str, object]:
    return {
        "job_id": "job-1",
        "created_utc": "2026-01-01T00:00:00Z",
        "updated_utc": "2026-01-01T00:00:00Z",
        "state": state,
        "store_backend": "memory",
        "store_uri": "memory://x",
        "adapter_path": "a.rbs",
        "adapter_type": "procrustes",
        "total_records": 1,
        "batch_size": 1,
        "keep_original": keep_original,
        "config": {},
        "error_code": "",
    }
