"""``--shadow-precision float16``: half the disk, and what that costs.

The option existed in `ShadowStore` from the beginning and was never exposed,
because a **half** guarantee may be more dangerous than no guarantee and nobody
had measured which this is. `spikes/shadow_precision.py` measured it over 68
corpus/model runs: no vector overflows the format, the top-10 set survives on
99.8% of queries, and nDCG@10 moves by at most 0.0017.

So it ships, and what these tests hold is the half that makes it safe rather
than merely small — that the tool never claims bit-identity when it is on, and
that the choice is recoverable afterwards from something other than memory.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from typer.testing import CliRunner

from rebasis.cli import app
from rebasis.core import ProcrustesAdapter, l2_normalize
from rebasis.manifest import ManifestDB, manifest_path
from rebasis.migrate import MigrationEngine
from rebasis.storage.shadow import ShadowStore
from rebasis.store import MemoryStore

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

runner = CliRunner()
DIM = 16
N = 64


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(23)


def build(tmp_path: Path, rng: np.random.Generator, *, precision: str) -> MigrationEngine:
    vectors = l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))
    rotation = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype(np.float32)
    ids = [f"doc-{i:04d}" for i in range(N)]
    engine = MigrationEngine(
        db=ManifestDB(manifest_path(tmp_path / "state")),
        store=MemoryStore(ids, vectors, [f"text {i}" for i in range(N)]),
        adapter=ProcrustesAdapter.fit(vectors, l2_normalize(vectors @ rotation.T)),
        shadow_root=tmp_path / "shadow",
        batch_size=16,
        power_aware=False,
        shadow_precision=precision,
    )
    engine.prepare(ids)
    return engine


class TestItHalvesTheShadow:
    def test_float16_is_half_the_bytes(self, tmp_path: Path, rng: np.random.Generator) -> None:
        """The whole reason to want it, and arithmetic rather than a measurement
        — which is why it is asserted rather than described."""
        wide = build(tmp_path / "wide", rng, precision="float32")
        narrow = build(tmp_path / "narrow", np.random.default_rng(23), precision="float16")
        wide.run()
        narrow.run()

        assert narrow.shadow.size_bytes() * 2 == pytest.approx(wide.shadow.size_bytes(), rel=0.05)

    def test_the_estimate_agrees_with_what_gets_written(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """`migrate` refuses to start when the disk cannot take the shadow, so an
        estimate that disagreed with the file would refuse the wrong jobs."""
        engine = build(tmp_path, rng, precision="float16")
        engine.run()

        assert ShadowStore.estimate_bytes(N, DIM, precision="float16") == pytest.approx(
            engine.shadow.size_bytes(), rel=0.05
        )


class TestTheChoiceIsRecoverable:
    def test_the_shadow_records_it(self, tmp_path: Path, rng: np.random.Generator) -> None:
        """Nothing in the index says which precision a job used. The shadow
        manifest and the audit trail are the only two places it survives, and
        `rollback` reads the first of them."""
        engine = build(tmp_path, rng, precision="float16")
        engine.run()

        assert engine.shadow.manifest().precision == "float16"

    def test_the_audit_trail_records_it(self, tmp_path: Path, rng: np.random.Generator) -> None:
        from rebasis.audit import AuditWriter

        db = ManifestDB(manifest_path(tmp_path / "state"))
        writer = AuditWriter(db, run_id="run-precision")
        vectors = l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))
        ids = [f"doc-{i:04d}" for i in range(N)]
        engine = MigrationEngine(
            db=db,
            store=MemoryStore(ids, vectors, [f"text {i}" for i in range(N)]),
            adapter=ProcrustesAdapter.fit(vectors, vectors),
            shadow_root=tmp_path / "shadow",
            audit=writer,
            shadow_precision="float16",
        )
        engine.prepare(ids)

        recorded = [
            json.loads(row["inputs_json"])
            for row in db.query(
                "SELECT inputs_json FROM audit_records WHERE action = ?",
                ("migrate.job.started",),
            )
        ]
        assert recorded[0]["shadow_precision"] == "float16"


class TestWhatItCosts:
    def test_a_rollback_restores_close_but_not_exactly(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """The guarantee that is given up, asserted as given up.

        float32 restores the bytes that were read. float16 restores a value
        within the format's own step — measured over 68 corpus/model runs at a
        maximum component error of 2.4e-4 against a unit vector.
        """
        engine = build(tmp_path, rng, precision="float16")
        # `.copy()`, and it is not defensive: an in-memory store hands back a
        # view of the array the migration is about to overwrite in place, so a
        # reference here would compare the index against itself and pass no
        # matter what the shadow held.
        before = {
            record.id: record.vector.copy()
            for record in engine.store.iter_records(with_vectors=True, with_text=False)
            if record.vector is not None
        }
        engine.run()
        engine.rollback()
        after = {
            record.id: record.vector.copy()
            for record in engine.store.iter_records(with_vectors=True, with_text=False)
            if record.vector is not None
        }

        errors = np.array(
            [float(np.abs(before[i] - after[i]).max()) for i in sorted(before)], dtype=np.float64
        )
        assert errors.max() > 0, "float16 that round-tripped exactly is not float16"
        assert errors.max() < 1e-3

    def test_float32_restores_exactly(self, tmp_path: Path, rng: np.random.Generator) -> None:
        """The default, and the thing the other option is measured against."""
        engine = build(tmp_path, rng, precision="float32")
        # `.copy()` for the reason the float16 test above gives.
        before = {
            record.id: record.vector.copy()
            for record in engine.store.iter_records(with_vectors=True, with_text=False)
            if record.vector is not None
        }
        engine.run()
        engine.rollback()

        for record in engine.store.iter_records(with_vectors=True, with_text=False):
            assert record.vector is not None
            np.testing.assert_array_equal(record.vector, before[record.id])


class TestTheCommandRefusesNonsense:
    def test_an_unknown_precision_is_a_usage_error(self, tmp_path: Path) -> None:
        """`ShadowStore` treats anything that is not `float16` as `float32`,
        which is right for a library and wrong for a flag: a typo would silently
        give the safe behaviour while the user believed otherwise."""
        result = runner.invoke(
            app,
            [
                "migrate",
                "--adapter",
                str(tmp_path / "nothing.rbs"),
                "--store",
                "memory://",
                "--shadow-precision",
                "float8",
                "--yes",
            ],
        )

        assert result.exit_code != 0
        assert "float8" in result.output

    def test_it_is_refused_before_the_adapter_is_even_read(self, tmp_path: Path) -> None:
        """The adapter path above does not exist. Reaching the precision error
        rather than a missing-file error is what says the check runs first —
        the cheapest refusal happening first is the rule everywhere else here."""
        result = runner.invoke(
            app,
            [
                "migrate",
                "--adapter",
                str(tmp_path / "nothing.rbs"),
                "--store",
                "memory://",
                "--shadow-precision",
                "float8",
                "--yes",
            ],
        )

        assert "shadow-precision" in result.output
