"""A half-migrated index has to say so.

`migrate --limit`, `--priority access` and every pause are documented, supported
ways of stopping short, and each leaves the collection holding both models'
vectors. Nothing raises when it happens and nothing in a search result shows it:
the record count is right, the text is right, the ranking is wrong. So the whole
protection is that the tool says so — which makes "does it say so" a test rather
than a nicety.

Covered here: the manifest query that decides it, and both places a user meets
it — `status` and the end of a `migrate` run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from typer.testing import CliRunner

from rebasis.cli import app
from rebasis.core import ProcrustesAdapter, l2_normalize
from rebasis.manifest import ManifestDB, manifest_path
from rebasis.migrate import MigrationEngine, mixed_spaces, mixed_spaces_for
from rebasis.migrate.states import ItemState, JobState
from rebasis.store import MemoryStore

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

runner = CliRunner()
WIDE = {"COLUMNS": "220"}

DIM = 16
N = 64


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(11)


def build_engine(tmp_path: Path, rng: np.random.Generator, *, uri: str = "") -> MigrationEngine:
    """A job over an in-memory store, ready to be run for as long as we like.

    ``uri`` is empty by default. The engine reopens the store from its URI when
    a job completes, to re-check durability on a fresh connection — and a fresh
    ``memory://`` store is a fresh empty one, so a URI here would make every
    completed run fail on a check that is not what these tests are about.
    """
    vectors = l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))
    rotation = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype(np.float32)
    adapter = ProcrustesAdapter.fit(vectors, l2_normalize(vectors @ rotation.T))

    ids = [f"doc-{i:04d}" for i in range(N)]
    built = MigrationEngine(
        db=ManifestDB(manifest_path(tmp_path / "state")),
        store=MemoryStore(ids, vectors, [f"text {i}" for i in range(N)]),
        adapter=adapter,
        shadow_root=tmp_path / "shadow",
        batch_size=8,
        power_aware=False,
        store_uri=uri,
    )
    built.prepare(ids)
    return built


@pytest.fixture
def engine(tmp_path: Path, rng: np.random.Generator) -> MigrationEngine:
    return build_engine(tmp_path, rng)


class TestTheManifestQuery:
    """What `mixed_spaces` counts, and what it deliberately does not."""

    def test_a_job_stopped_short_leaves_the_index_mixed(self, engine: MigrationEngine) -> None:
        engine.run(limit=16)

        states = mixed_spaces(engine.db)

        assert len(states) == 1
        assert states[0].migrated == 16
        assert states[0].unmigrated == N - 16
        assert states[0].fraction == pytest.approx(0.25)

    def test_a_finished_job_leaves_it_single(self, engine: MigrationEngine) -> None:
        engine.run()

        assert mixed_spaces(engine.db) == []

    def test_a_job_that_has_not_started_leaves_it_single(self, engine: MigrationEngine) -> None:
        """Nothing was written, so nothing is in the new space.

        Worth its own test: the naive query — "is this job unfinished?" — would
        report a mixture on every queued job that has not run yet, and a
        warning that fires when there is nothing wrong is a warning people
        learn to skip.
        """
        assert mixed_spaces(engine.db) == []

    def test_a_rolled_back_job_leaves_it_single(self, engine: MigrationEngine) -> None:
        """The shadow copy put every vector back, so there is one space again."""
        engine.run(limit=16)
        engine.rollback()

        assert mixed_spaces(engine.db) == []

    def test_failed_records_count_as_unmigrated(self, engine: MigrationEngine) -> None:
        """A record the store refused is still in the old space.

        Leaving failures out of the figure would report a clean index on the
        one path where the mixture was not even intentional.
        """
        engine.run(limit=16)
        engine.queue.mark(["doc-0050", "doc-0051"], ItemState.FAILED, error_code="RB-E6001")

        state = mixed_spaces(engine.db)[0]

        assert state.unmigrated == N - 16
        assert state.migrated == 16

    def test_skipped_records_do_not(self, engine: MigrationEngine) -> None:
        """A record with no vector is in neither space.

        `iter_records` returns it, the engine skips it, and counting it as
        un-migrated would report a mixture that a completed job never resolves.
        """
        engine.run(limit=16)
        engine.queue.mark([f"doc-{i:04d}" for i in range(16, N)], ItemState.SKIPPED)

        assert mixed_spaces(engine.db) == []

    def test_it_is_matched_to_the_store_that_holds_it(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        engine = build_engine(tmp_path, rng, uri="memory://test#documents")
        engine.run(limit=8)

        assert mixed_spaces_for(engine.db, "memory://test#documents")
        assert mixed_spaces_for(engine.db, "chroma:///somewhere/else#documents") == []

    def test_it_takes_no_lock_and_opens_no_store(self, engine: MigrationEngine) -> None:
        """`status` is run *while* a migration is in flight — that is when the
        answer is wanted — so this may not block behind one."""
        engine.run(limit=8)
        # The engine still holds its handle; a second reader must not need one.
        assert mixed_spaces(engine.db)[0].state == str(JobState.PAUSED)


class TestWhatTheUserSees:
    """The two moments it reaches a person."""

    def test_status_says_so_unprompted(self, engine: MigrationEngine, tmp_path: Path) -> None:
        engine.run(limit=16)
        engine.db.close()

        result = runner.invoke(app, ["status", "--state-dir", str(tmp_path / "state")], env=WIDE)

        assert result.exit_code == 0
        assert "two embedding spaces" in result.output
        assert "mixed" in result.output

    def test_status_names_both_ways_out(self, engine: MigrationEngine, tmp_path: Path) -> None:
        engine.run(limit=16)
        engine.db.close()

        result = runner.invoke(app, ["status", "--state-dir", str(tmp_path / "state")], env=WIDE)

        assert f"migrate --resume {engine.job_id}" in result.output
        assert f"rollback {engine.job_id}" in result.output

    def test_status_json_carries_the_field(self, engine: MigrationEngine, tmp_path: Path) -> None:
        """The field a script branches on.

        A CI job that queries an index after migrating a slice of it should be
        able to fail rather than quietly measure the wrong thing, and it cannot
        do that against a Rich table.
        """
        import json

        engine.run(limit=16)
        engine.db.close()

        result = runner.invoke(
            app, ["status", "--state-dir", str(tmp_path / "state"), "--json"], env=WIDE
        )

        payload = json.loads(result.stdout)
        assert payload[0]["mixed_space"]["migrated"] == 16
        assert payload[0]["mixed_space"]["unmigrated"] == N - 16

    def test_a_finished_job_says_nothing(self, engine: MigrationEngine, tmp_path: Path) -> None:
        engine.run()
        engine.db.close()

        result = runner.invoke(app, ["status", "--state-dir", str(tmp_path / "state")], env=WIDE)

        assert "two embedding spaces" not in result.output
        assert "single" in result.output
