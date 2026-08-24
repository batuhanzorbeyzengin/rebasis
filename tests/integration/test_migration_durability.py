"""The end-of-job durability check.

The per-batch read-back goes through the handle that wrote, which is the one
handle a caching client will answer from memory. This check reopens the store
and asks again. It exists because that failure mode was mistaken for a rebasis
bug once already: a Chroma `PersistentClient` is cached per path within a
process, so a fingerprint read after a subprocess wrote returned stale data and
looked exactly like a `migrate` that had done nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from rebasis.core import ProcrustesAdapter, l2_normalize
from rebasis.errors import WritesDidNotSurvive
from rebasis.manifest import ManifestDB
from rebasis.migrate.engine import DURABILITY_SAMPLE, MigrationEngine
from rebasis.store import open_store

pytestmark = [pytest.mark.integration]

DIM = 16
N = 200


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(7)


@pytest.fixture
def chroma_uri(tmp_path, rng):  # type: ignore[no-untyped-def]
    """A real Chroma collection — the backend that motivated this check."""
    chromadb = pytest.importorskip("chromadb", reason="the chroma extra is not installed")
    path = tmp_path / "chroma"
    vectors = l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))
    client = chromadb.PersistentClient(path=str(path))
    collection = client.create_collection("documents", metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[f"doc-{i:04d}" for i in range(N)],
        embeddings=vectors.tolist(),
        documents=[f"text of document {i}" for i in range(N)],
    )
    del client, collection
    return f"chroma://{path}#documents"


def make_engine(uri: str, tmp_path, rng):  # type: ignore[no-untyped-def]
    from rebasis.manifest import manifest_path

    vectors = l2_normalize(rng.standard_normal((256, DIM)).astype(np.float32))
    rotation = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype(np.float32)
    adapter = ProcrustesAdapter.fit(vectors, l2_normalize(vectors @ rotation.T))
    store = open_store(uri)
    engine = MigrationEngine(
        db=ManifestDB(manifest_path(tmp_path / "state")),
        store=store,
        adapter=adapter,
        shadow_root=tmp_path / "shadow",
        batch_size=32,
        power_aware=False,
        store_uri=uri,
    )
    engine.prepare([f"doc-{i:04d}" for i in range(N)])
    return engine


class TestTheCheckRuns:
    def test_a_completed_job_re_checks_on_a_fresh_connection(
        self, chroma_uri, tmp_path, rng
    ) -> None:  # type: ignore[no-untyped-def]
        engine = make_engine(chroma_uri, tmp_path, rng)

        engine.run()

        assert engine.verify_durability() == DURABILITY_SAMPLE

    def test_the_sample_is_spread_across_the_run(self, chroma_uri, tmp_path, rng) -> None:  # type: ignore[no-untyped-def]
        """Reservoir, not "the last batch": a store that stops persisting halfway
        through still looks healthy in the batch that is still in its cache."""
        engine = make_engine(chroma_uri, tmp_path, rng)

        engine.run()

        kept = sorted(int(record_id.split("-")[1]) for record_id in engine.durability_sample_ids())
        assert len(kept) == DURABILITY_SAMPLE
        assert min(kept) < N // 4, "the sample never reached the early batches"
        assert max(kept) > 3 * N // 4, "the sample never reached the late batches"

    def test_it_is_a_no_op_without_a_uri_to_reopen(self, tmp_path, rng) -> None:  # type: ignore[no-untyped-def]
        """An engine handed a live store object has nothing to reopen, and
        saying so beats inventing a URI that might name a different store."""
        from rebasis.manifest import manifest_path
        from rebasis.store.backends.memory import MemoryStore

        vectors = l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))
        store = MemoryStore([f"doc-{i:04d}" for i in range(N)], vectors)
        rotation = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype(np.float32)
        engine = MigrationEngine(
            db=ManifestDB(manifest_path(tmp_path / "state")),
            store=store,
            adapter=ProcrustesAdapter.fit(vectors, l2_normalize(vectors @ rotation.T)),
            shadow_root=tmp_path / "shadow",
            power_aware=False,
        )
        engine.prepare([f"doc-{i:04d}" for i in range(N)])
        engine.run()

        assert engine.verify_durability() == 0


class TestTheCheckCatchesWhatItIsFor:
    def test_a_store_that_forgets_its_writes_is_caught(self, chroma_uri, tmp_path, rng) -> None:  # type: ignore[no-untyped-def]
        """Simulated by rolling the store back underneath the engine: the writes
        happened, passed their per-batch check, and are no longer there."""
        engine = make_engine(chroma_uri, tmp_path, rng)
        engine.run()
        engine.rollback()

        with pytest.raises(WritesDidNotSurvive) as caught:
            engine.verify_durability()

        assert "rollback" in (caught.value.hint or "")
        assert caught.value.code == "RB-E6005"

    def test_something_else_overwriting_a_record_is_caught(self, chroma_uri, tmp_path, rng) -> None:  # type: ignore[no-untyped-def]
        """Not the failure the check is named for, but the same evidence: what
        the store holds is not what the migration put there."""
        engine = make_engine(chroma_uri, tmp_path, rng)
        engine.run()

        watched = engine.durability_sample_ids()[:4]
        intruder = open_store(chroma_uri)
        try:
            intruder.upsert_vectors(
                watched, l2_normalize(np.ones((len(watched), DIM), dtype=np.float32))
            )
        finally:
            close = getattr(intruder, "close", None)
            if callable(close):
                close()

        with pytest.raises(WritesDidNotSurvive) as caught:
            engine.verify_durability()

        assert caught.value.context.get("record_id") in watched
