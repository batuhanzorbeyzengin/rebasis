"""Migrate and roll back against every backend that claims it can write.

The durability suite exercises the engine thoroughly, on the in-memory store.
That proves the state machine, the shadow copy and the checkpointing. It does
not prove that a given backend actually persists what it was told to write —
and a store that accepts a write and forgets it is silent data loss. It has
already happened once here, to the file-backed memory store.

So this runs the whole cycle against each real backend: read the vectors,
migrate, confirm the index changed *from a fresh connection*, roll back, confirm
it came back. Backends that declare they cannot write are checked for refusing
rather than skipped.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from rebasis.core import ProcrustesAdapter, l2_normalize
from rebasis.errors import CapabilityMissing
from rebasis.store import open_store
from rebasis.store.base import require_capability

pytestmark = [pytest.mark.integration]

DIM = 32
N = 300


#: Written out rather than imported from the conftest that builds them:
#: `@pytest.fixture(params=...)` runs at collection, and a conftest is not
#: importable by name. The builders live in `tests/conftest.py`, shared with
#: the contract suite.
BACKENDS = ("chroma", "faiss", "lancedb", "qdrant", "sqlite-vec")


@pytest.fixture(params=BACKENDS, ids=lambda n: n)
def live_store(request, tmp_path, rng, make_store):  # type: ignore[no-untyped-def]
    """A real store of each kind, populated identically."""
    vectors = l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))
    ids = [f"doc-{i:04d}" for i in range(N)]
    texts = [f"text of document {i}" for i in range(N)]
    uri = make_store(request.param, tmp_path, ids, vectors, texts)
    return {"uri": uri, "ids": ids, "vectors": vectors, "backend": request.param}


def closing(store: object) -> object:
    """Release a backend's handle if it holds one.

    Qdrant's local mode takes an exclusive lock on its storage folder and a
    second client raises rather than waiting, so a handle left open makes the
    next read fail. Most backends have nothing to release.
    """
    close = getattr(store, "close", None)
    if callable(close):
        with contextlib.suppress(Exception):
            close()
    return store


def writable(uri: str) -> bool:
    """Whether a backend declares it can write, releasing the handle after."""
    store = open_store(uri)
    try:
        return bool(store.capabilities.can_upsert_vectors)
    finally:
        closing(store)


def read_vectors(uri: str, ids: list[str]) -> np.ndarray:
    """Read through a fresh connection.

    Not an optimisation to skip: Chroma caches its client per path, so reusing a
    handle opened before a write returns the state it already had. A real
    migration looked like a no-op here until this was fixed.
    """
    store = open_store(uri)
    try:
        by_id = {r.id: r.vector for r in store.iter_records(with_text=False)}
    finally:
        closing(store)
    return np.vstack([by_id[i] for i in ids])


def build_adapter(vectors: np.ndarray, rng: np.random.Generator) -> ProcrustesAdapter:
    """An adapter that genuinely moves the vectors, so a no-op is detectable."""
    rotation = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype(np.float32)
    return ProcrustesAdapter.fit(vectors, l2_normalize(vectors @ rotation.T))


def run_migration(live_store, tmp_path, rng, *, keep_original: bool = True):  # type: ignore[no-untyped-def]
    from rebasis.manifest import ManifestDB, manifest_path
    from rebasis.migrate import MigrationEngine

    store = open_store(live_store["uri"])
    engine = MigrationEngine(
        db=ManifestDB(manifest_path(tmp_path / "state")),
        store=store,
        adapter=build_adapter(live_store["vectors"], rng),
        shadow_root=tmp_path / "shadow",
        keep_original=keep_original,
        batch_size=64,
        store_uri=live_store["uri"],
    )
    engine.prepare(list(live_store["ids"]))
    result = engine.run()
    return engine, result


def release(engine: object) -> None:
    """Release the engine's store handle once its work is done."""
    closing(getattr(engine, "store", None))


class TestEveryWritableBackend:
    def test_migrate_actually_writes(self, live_store, tmp_path, rng) -> None:  # type: ignore[no-untyped-def]
        """The failure this suite exists for: a store that accepts a write and
        forgets it, reported as success."""
        if not writable(live_store["uri"]):
            pytest.skip(f"{live_store['backend']} declares it cannot write")

        before = read_vectors(live_store["uri"], live_store["ids"])
        engine, result = run_migration(live_store, tmp_path, rng)
        release(engine)
        after = read_vectors(live_store["uri"], live_store["ids"])

        assert result.failed == 0, f"{result.failed} records failed"
        assert result.processed == N
        assert not np.allclose(before, after), "migrate reported success and wrote nothing"

    def test_the_record_count_is_unchanged(self, live_store, tmp_path, rng) -> None:  # type: ignore[no-untyped-def]
        """Migrate upserts and never deletes."""
        if not writable(live_store["uri"]):
            pytest.skip(f"{live_store['backend']} declares it cannot write")

        engine, _ = run_migration(live_store, tmp_path, rng)
        release(engine)

        store = open_store(live_store["uri"])
        try:
            assert store.count() == N
        finally:
            closing(store)

    def test_rollback_restores_the_vectors(self, live_store, tmp_path, rng) -> None:  # type: ignore[no-untyped-def]
        """Bit-identical at the shadow; end to end, within the store's own
        float32 round trip — Chroma's is 3e-08 with rebasis nowhere in it."""
        if not writable(live_store["uri"]):
            pytest.skip(f"{live_store['backend']} declares it cannot write")

        before = read_vectors(live_store["uri"], live_store["ids"])
        engine, _ = run_migration(live_store, tmp_path, rng)
        restored_count = engine.rollback()
        release(engine)
        restored = read_vectors(live_store["uri"], live_store["ids"])

        assert restored_count == N
        np.testing.assert_allclose(restored, before, atol=1e-6)

    def test_text_survives_the_migration(self, live_store, tmp_path, rng) -> None:  # type: ignore[no-untyped-def]
        """rebasis writes vectors and nothing else. A backend that replaced the
        whole record would drop everything not resent."""
        store = open_store(live_store["uri"])
        try:
            capable = store.capabilities.can_upsert_vectors and store.capabilities.can_read_text
        finally:
            closing(store)
        if not capable:
            pytest.skip(f"{live_store['backend']} cannot both write and read text")

        engine, _ = run_migration(live_store, tmp_path, rng)
        release(engine)

        reopened = open_store(live_store["uri"])
        try:
            record = next(iter(reopened.iter_records(["doc-0007"])))
        finally:
            closing(reopened)
        assert record.text == "text of document 7"

    def test_the_declared_write_capability_is_the_truth(self, live_store, rng) -> None:  # type: ignore[no-untyped-def]
        """Partial support beats none; silent partial support does not.

        A capability flag is a promise `migrate` makes on the backend's behalf
        before it touches anything, so the flag is checked against a real
        write rather than against the backend's own opinion of itself.
        """
        store = open_store(live_store["uri"])
        try:
            declared = store.capabilities.can_upsert_vectors
            if not declared:
                with pytest.raises(CapabilityMissing):
                    require_capability(store, "can_upsert_vectors", operation="migrate")
                return
            require_capability(store, "can_upsert_vectors", operation="migrate")
            replacement = l2_normalize(rng.standard_normal((1, DIM)).astype(np.float32))
            store.upsert_vectors(["doc-0011"], replacement)
        finally:
            closing(store)

        written = read_vectors(live_store["uri"], ["doc-0011"])
        assert np.allclose(written[0], replacement[0], atol=1e-5)
