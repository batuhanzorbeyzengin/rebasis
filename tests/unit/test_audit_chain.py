"""The audit hash chain.

Three cases carry it: append and verify passes, alter a record and verify
fails, delete a record and verify fails. All three are here, because a chain
that only passes the happy case detects nothing.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - pytest resolves fixture annotations at runtime

import pytest

from rebasis.audit import GENESIS_HASH, AuditReader, AuditWriter, is_audited, verify_chain
from rebasis.manifest import ManifestDB, manifest_path
from rebasis.observability import Events


@pytest.fixture
def db(tmp_path: Path) -> ManifestDB:
    """A fresh manifest."""
    database = ManifestDB(manifest_path(tmp_path / ".rebasis"))
    yield database  # type: ignore[misc]
    database.close()


@pytest.fixture
def writer(db: ManifestDB) -> AuditWriter:
    """A writer bound to that manifest."""
    return AuditWriter(db, run_id="01JTEST0000000000000000000", actor="cli")


def _write_three(writer: AuditWriter) -> None:
    writer.write(
        Events.PROBE_DECISION_MADE,
        inputs={"seed": 0, "count": 10_000},
        outputs={"decision": "bridge_sufficient", "arr_r10": 0.96},
    )
    writer.write(
        Events.FIT_ADAPTER_SELECTED,
        inputs={"method": "auto"},
        outputs={"adapter_type": "procrustes_centered"},
    )
    writer.write(Events.STORE_WRITE_PERFORMED, inputs={"job_id": "j1"}, outputs={"count": 500})


@pytest.mark.unit
def test_only_audited_events_produce_records(writer: AuditWriter, db: ManifestDB) -> None:
    """The audited set is a small subset of what logs.

    If everything were audited, the trail would just be a second log — and the
    distinction that makes it valuable would be gone.
    """
    assert is_audited(Events.PROBE_DECISION_MADE)
    assert not is_audited(Events.EMBED_BATCH_COMPLETED)

    assert writer.write(Events.EMBED_BATCH_COMPLETED, inputs={}, outputs={}) is None
    assert AuditReader(db).count() == 0


@pytest.mark.unit
def test_chain_verifies_after_appending(writer: AuditWriter, db: ManifestDB) -> None:
    """The happy path, and the links really are links."""
    _write_three(writer)
    reader = AuditReader(db)
    records = reader.all_records()

    assert len(records) == 3
    assert records[0].prev_hash == GENESIS_HASH
    assert records[1].prev_hash == records[0].hash
    assert records[2].prev_hash == records[1].hash

    result = reader.verify()
    assert result.valid
    assert result.records_checked == 3


@pytest.mark.unit
def test_altering_a_record_is_detected(writer: AuditWriter, db: ManifestDB) -> None:
    """Change a record by hand, verification must fail."""
    _write_three(writer)
    db.execute(
        "UPDATE audit_records SET outputs_json = ? WHERE seq = 1",
        ('{"decision":"full_reindex","arr_r10":0.2}',),
    )

    result = AuditReader(db).verify()
    assert not result.valid
    assert result.broken_at == 1
    assert "altered" in result.reason


@pytest.mark.unit
def test_deleting_a_record_is_detected(writer: AuditWriter, db: ManifestDB) -> None:
    """Remove a record, the following link no longer matches."""
    _write_three(writer)
    db.execute("DELETE FROM audit_records WHERE seq = 2")

    result = AuditReader(db).verify()
    assert not result.valid
    assert result.broken_at == 3
    assert "deleted" in result.reason


@pytest.mark.unit
def test_an_empty_chain_is_valid() -> None:
    """Nothing to verify is not a failure."""
    assert verify_chain([]).valid


@pytest.mark.unit
def test_records_carry_what_a_replay_needs(writer: AuditWriter, db: ManifestDB) -> None:
    """The inputs must be enough to reproduce the decision."""
    writer.write(
        Events.PROBE_DECISION_MADE,
        inputs={
            "old_model": "sentence-transformers/all-MiniLM-L6-v2",
            "new_model": "BAAI/bge-small-en-v1.5",
            "seed": 0,
            "count": 10_000,
            "strategy": "stratified",
        },
        outputs={"decision": "bridge_sufficient", "arr_r10": 0.96},
        subject="chroma:///vault#docs",
    )
    record = AuditReader(db).all_records()[0]

    assert record.inputs["seed"] == 0
    assert record.inputs["old_model"]
    assert record.rebasis_version
    assert record.env["python"]
    # Without the device, a differing replay cannot be told apart from a
    # regression.
    assert "device" in record.env


@pytest.mark.unit
def test_records_carry_no_content(writer: AuditWriter, db: ManifestDB) -> None:
    """Auditability costs nothing in privacy.

    Reconstructing a decision needs the parameters, not the corpus.
    """
    writer.write(
        Events.PROBE_DECISION_MADE,
        inputs={"count": 10_000, "seed": 0},
        outputs={"decision": "bridge_sufficient"},
    )
    export = AuditReader(db).export_jsonl()
    for forbidden in ("chunk_text", "document", "query_text", "embedding"):
        assert forbidden not in export


@pytest.mark.unit
def test_export_is_one_json_object_per_record(writer: AuditWriter, db: ManifestDB) -> None:
    """JSON Lines, so the trail can be archived or analysed elsewhere."""
    import json

    _write_three(writer)
    lines = AuditReader(db).export_jsonl().splitlines()
    assert len(lines) == 3
    for line in lines:
        assert "hash" in json.loads(line)


@pytest.mark.unit
def test_pragmas_are_durable(db: ManifestDB) -> None:
    """WAL + FULL, chosen because audit records may not be lost.

    ``synchronous`` reads back as 2, which is FULL.
    """
    settings = db.pragma_settings()
    assert settings["journal_mode"].lower() == "wal"
    assert settings["synchronous"] == 2
    assert settings["foreign_keys"] == 1
