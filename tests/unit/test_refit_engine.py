"""Refitting the adapter part-way through a migration, and what stops it.

`migrate/refit.py` decides whether a candidate wins. This is the half that
produces the candidate: sampling records from the queue, re-embedding them,
swapping the adapter when the guard says so, and writing the result somewhere a
`--resume` can find it.

The sample comes from records **not yet migrated**, and that is the whole design
rather than a detail. `spikes/continuous_refit.py` measured both sources over 216
cells: on an unchanged corpus they are indistinguishable, and on a corpus that
grew into a domain the adapter never saw, drawing from the remainder is worth
+0.20 nDCG at every pair budget tested. The drift fixture below is that case in
miniature — two halves under two different rotations — because it is the only
one where a refit has anything to learn.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from rebasis.core import ProcrustesAdapter, l2_normalize
from rebasis.embed import PrecomputedEmbedder
from rebasis.manifest import ManifestDB, manifest_path
from rebasis.migrate import JobState, MigrationEngine, RefitPolicy
from rebasis.store import MemoryStore
from rebasis.types import EncodingProfile

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

DIM = 16
N = 400
BATCH = 100
#: Small enough that a unit test can afford the pairs, large enough that a
#: Procrustes fit on `DIM` dimensions is determined rather than degenerate.
SAMPLE = 120
FLOOR = 60


def _rotation(rng: np.random.Generator) -> np.ndarray:
    return np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype(np.float32)


class Corpus:
    """An index, the new model's view of it, and an adapter fitted on part of it."""

    def __init__(self, rng: np.random.Generator, *, drifts: bool) -> None:
        self.ids = [f"doc-{i:04d}" for i in range(N)]
        self.texts = [f"text {i}" for i in range(N)]
        self.old = l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))

        # The first batch's records and the rest, under one rotation or two.
        # Two is the corpus that grew into a domain the adapter never saw; the
        # queue is drained in id order, so the split lines up with what has been
        # migrated by the time the first refit is due.
        first, second = _rotation(rng), _rotation(rng)
        head = l2_normalize(self.old[:BATCH] @ first.T)
        tail = l2_normalize(self.old[BATCH:] @ (second if drifts else first).T)
        self.new = np.vstack([head, tail])

        # Fitted on what existed when `fit` ran — the head — which is what makes
        # it wrong for the tail when the tail is a different rotation.
        self.adapter = ProcrustesAdapter.fit(self.old[:BATCH], self.new[:BATCH])

    def store(self) -> MemoryStore:
        return MemoryStore(self.ids, self.old.copy(), list(self.texts))

    def embedder(self) -> PrecomputedEmbedder:
        return PrecomputedEmbedder(
            EncodingProfile(model_id="new-model", dim=DIM),
            dict(zip(self.texts, self.new, strict=True)),
        )


def build(
    tmp_path: Path,
    corpus: Corpus,
    **overrides: Any,
) -> MigrationEngine:
    """An engine wired for refitting, unless a keyword takes a collaborator away."""
    profiles = (
        EncodingProfile(model_id="old-model", dim=DIM),
        EncodingProfile(model_id="new-model", dim=DIM),
    )
    settings: dict[str, Any] = {
        "embedder": corpus.embedder(),
        "adapter_root": tmp_path / "adapters",
        "profiles": profiles,
        "store": corpus.store(),
        "refit": RefitPolicy(
            enabled=True, every_n_records=BATCH, sample_size=SAMPLE, min_pairs=FLOOR
        ),
    }
    settings.update(overrides)
    engine = MigrationEngine(
        db=ManifestDB(manifest_path(tmp_path / "state")),
        adapter=corpus.adapter,
        shadow_root=tmp_path / "shadow",
        batch_size=BATCH,
        power_aware=False,
        **settings,
    )
    engine.prepare(corpus.ids)
    return engine


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(17)


@pytest.fixture
def drifted(rng: np.random.Generator) -> Corpus:
    return Corpus(rng, drifts=True)


@pytest.fixture
def stable(rng: np.random.Generator) -> Corpus:
    return Corpus(rng, drifts=False)


class TestItSamplesWhatIsLeft:
    def test_the_pairs_come_from_records_not_yet_migrated(
        self, tmp_path: Path, drifted: Corpus
    ) -> None:
        """The measured design, asserted where it can be: every sampled record is
        still pending, so none of them carries the adapter's own image."""
        engine = build(tmp_path, drifted)
        engine.queue.mark(drifted.ids[:BATCH], "done")  # type: ignore[arg-type]

        drawn = set(engine.queue.sample_pending(SAMPLE, np.random.default_rng(0)))

        assert drawn
        assert drawn.isdisjoint(drifted.ids[:BATCH])

    def test_it_asks_for_no_more_than_the_sample_size(
        self, tmp_path: Path, drifted: Corpus
    ) -> None:
        engine = build(tmp_path, drifted)

        assert len(engine.queue.sample_pending(SAMPLE, np.random.default_rng(0))) == SAMPLE

    def test_the_sample_is_reproducible_from_the_seed(
        self, tmp_path: Path, drifted: Corpus
    ) -> None:
        """A migration that made a different decision on a re-run of the same job
        could not be reproduced from its audit trail, which is why this does not
        use SQLite's unseedable `RANDOM()`."""
        engine = build(tmp_path, drifted)

        first = engine.queue.sample_pending(SAMPLE, np.random.default_rng(3))
        second = engine.queue.sample_pending(SAMPLE, np.random.default_rng(3))

        assert first == second


class TestItAdoptsAWinner:
    def test_a_drifted_corpus_gets_a_new_adapter(self, tmp_path: Path, drifted: Corpus) -> None:
        """The case the feature exists for: the tail is a different rotation, so
        the adapter fitted on the head is wrong for everything left."""
        engine = build(tmp_path, drifted)
        before = engine.adapter

        result = engine.run()

        assert result.state is JobState.COMPLETED
        assert engine.adapter is not before

    def test_the_records_it_writes_afterwards_are_closer_to_the_new_model(
        self, tmp_path: Path, drifted: Corpus
    ) -> None:
        """The only assertion here that is about the *outcome* rather than the
        mechanism, and the one the feature is for.

        Everything else says the adapter was swapped. This says the swap was
        worth making: the records migrated after it land nearer the vectors a
        full reindex would have written than the same records do under a run
        with refitting off. The corpus drifts, so the original adapter is
        mapping the tail with a rotation that does not describe it.
        """
        with_refit = build(tmp_path / "on", drifted)
        with_refit.run()
        without = build(tmp_path / "off", drifted, refit=None)
        without.run()

        def closeness(engine: MigrationEngine) -> float:
            written = {
                record.id: record.vector
                for record in engine.store.iter_records(
                    drifted.ids[BATCH:], with_vectors=True, with_text=False
                )
                if record.vector is not None
            }
            got = l2_normalize(np.vstack([written[i] for i in drifted.ids[BATCH:]]))
            return float(np.einsum("ij,ij->i", got, drifted.new[BATCH:]).mean())

        assert closeness(with_refit) > closeness(without) + 0.05

    def test_the_adopted_adapter_is_written_where_resume_looks(
        self, tmp_path: Path, drifted: Corpus
    ) -> None:
        """Without this a `--resume` reloads the file the job started with and
        silently gives back everything the refit gained."""
        engine = build(tmp_path, drifted)

        engine.run()

        written = sorted((tmp_path / "adapters").glob("*.rbs"))
        assert written, "nothing was persisted"
        row = engine.db.query_one(
            "SELECT adapter_path FROM jobs WHERE job_id = ?", (engine.job_id,)
        )
        assert row is not None
        assert row["adapter_path"] == str(max(written))

    def test_what_was_written_loads_back(self, tmp_path: Path, drifted: Corpus) -> None:
        from rebasis.core import load_adapter

        engine = build(tmp_path, drifted)
        engine.run()

        path = max((tmp_path / "adapters").glob("*.rbs"))
        loaded, manifest, _ = load_adapter(path)

        assert manifest.direction == "old_to_new"
        assert loaded.input_dim == DIM
        assert loaded.output_dim == DIM

    def test_the_adoption_is_audited(self, tmp_path: Path, drifted: Corpus) -> None:
        """A job that finished having used two adapters is a job whose results
        need that fact recorded."""
        from rebasis.audit import AuditWriter

        db = ManifestDB(manifest_path(tmp_path / "state"))
        writer = AuditWriter(db, run_id="run-refit")
        engine = build(tmp_path, drifted, audit=writer)
        engine.db = db

        engine.run()

        adopted = [
            json.loads(row["outputs_json"])
            for row in db.query(
                "SELECT outputs_json FROM audit_records WHERE action = ?",
                ("migrate.adapter.refitted",),
            )
        ]
        assert any(entry["adopted"] for entry in adopted)


class TestItDeclinesTheRest:
    def test_an_unchanged_corpus_keeps_its_adapter(self, tmp_path: Path, stable: Corpus) -> None:
        """Both halves are one rotation, so the adapter already fits what is
        left and a smaller sample has nothing to add. This is the majority case
        in the measurement, and the guard is what makes shipping it safe."""
        engine = build(tmp_path, stable)
        before = engine.adapter

        engine.run()

        assert engine.adapter is before
        assert not list((tmp_path / "adapters").glob("*.rbs"))

    def test_too_few_usable_pairs_declines_rather_than_fits(self, tmp_path: Path) -> None:
        """A store that returns text for only some records produces fewer pairs
        than were asked for, and a fit on those is a fit on noise."""
        rng = np.random.default_rng(4)
        corpus = Corpus(rng, drifts=True)
        textless = MemoryStore(corpus.ids, corpus.old.copy(), ["" for _ in corpus.ids])
        engine = build(tmp_path, corpus, store=textless)
        before = engine.adapter

        engine.run()

        assert engine.adapter is before


class TestItSaysWhenItCannot:
    def test_no_embedder_turns_it_off_at_the_start(self, tmp_path: Path, drifted: Corpus) -> None:
        """Not at the first checkpoint an hour in, by which point restarting
        costs everything already migrated."""
        engine = build(tmp_path, drifted, embedder=None)

        engine.run()

        assert engine.refit.enabled is False

    def test_nowhere_to_write_turns_it_off(self, tmp_path: Path, drifted: Corpus) -> None:
        engine = build(tmp_path, drifted, adapter_root=None)

        engine.run()

        assert engine.refit.enabled is False

    def test_the_reason_reaches_the_audit_trail(self, tmp_path: Path, drifted: Corpus) -> None:
        from rebasis.audit import AuditWriter

        db = ManifestDB(manifest_path(tmp_path / "state"))
        writer = AuditWriter(db, run_id="run-refit")
        engine = build(tmp_path, drifted, embedder=None, audit=writer)
        engine.db = db

        engine.run()

        reasons = [
            json.loads(row["outputs_json"])["reason"]
            for row in db.query(
                "SELECT outputs_json FROM audit_records WHERE action = ?",
                ("migrate.adapter.refitted",),
            )
        ]
        assert any("embedder" in reason for reason in reasons)

    def test_off_by_default(self, tmp_path: Path, drifted: Corpus) -> None:
        engine = build(tmp_path, drifted, refit=None)
        before = engine.adapter

        engine.run()

        assert engine.adapter is before
