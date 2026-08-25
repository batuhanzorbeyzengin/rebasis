"""Does rewriting every vector break the index that was built around them?

The question `migrate` has never asked. A graph index picks a record's edges
from the geometry of its neighbours at insert time; an in-place vector update
changes the geometry and leaves the edges. Qdrant says so in as many words —
a changed vector value discards the graph the same way a deletion does
(`qdrant/qdrant#6325`) — and a production report measured search quality at 34%
until the collection was force-reindexed (`qdrant/qdrant#7147`). rebasis writes
to five backends and has measured none of them.

This spike measures all of them, on the same corpus, with the same adapter:

    recall@10 of the store's own search against exact kNN, before and after a
    full migration of every record.

A backend that scores 1.000 both sides is doing exact search and has no graph to
break. A backend that starts near 1.000 and falls has exactly the failure this
was written to look for. A backend that starts *below* 1.000 is approximate by
construction, which is not a defect — it is why the before-measurement exists.

    .venv/bin/python spikes/index_health.py --n 20000 --dim 384

Numbers, not adjectives: whatever it prints is what goes in the docs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from rebasis.core.base import l2_normalize
from rebasis.manifest import ManifestDB, manifest_path
from rebasis.migrate import MigrationEngine, measure_index_health
from rebasis.store import open_store

#: Backends that can take an in-place vector update, which is what `migrate`
#: needs. Every one of them is in the README's support table.
BACKENDS = ("chroma", "qdrant", "lancedb", "sqlite-vec", "faiss")

#: A Qdrant *server*, which is a different backend from `qdrant` above in the
#: only way that matters here: the embedded mode scans, and the server builds an
#: HNSW graph once a segment passes its indexing threshold. Everything Qdrant
#: documents about a changed vector value invalidating the graph is about this
#: one. Needs `qdrant` listening on 6333; the spike does not start it.
QDRANT_SERVER = "qdrant-server"


def _await_qdrant(client: Any, name: str, *, timeout: float = 900.0) -> int:
    """Block until the collection is green, and report how much it indexed.

    Qdrant indexes in the background, so a measurement taken straight after the
    last upsert would be measuring a half-built graph.

    The return value is the point of this function as much as the waiting is.
    Qdrant only builds an HNSW graph for a segment once that segment passes
    `indexing_threshold`, so a small collection — or a large one split across
    enough segments — is served by brute force and has **no graph to break**.
    A run that did not check would report "no degradation" about an index that
    was never approximate, which is a true sentence and a useless one.
    """
    deadline = time.monotonic() + timeout
    indexed = 0
    while time.monotonic() < deadline:
        info = client.get_collection(name)
        indexed = int(getattr(info, "indexed_vectors_count", 0) or 0)
        if str(getattr(info, "status", "")).endswith("green"):
            return indexed
        time.sleep(2)
    return indexed


def build(backend: str, root: Path, ids: list[str], vectors: np.ndarray) -> str:
    """Create a collection of ``vectors`` in ``backend`` and return its URI.

    Built through each client's own API rather than through rebasis, because
    what is being measured is the index those clients build — a collection
    rebasis created would not be the collection a user has.
    """
    path = root / backend
    texts = [f"document number {i}" for i in range(len(ids))]

    if backend == "chroma":
        import chromadb

        client = chromadb.PersistentClient(path=str(path))
        collection = client.create_collection("documents", metadata={"hnsw:space": "cosine"})
        for start in range(0, len(ids), 2000):
            stop = start + 2000
            collection.add(
                ids=ids[start:stop],
                embeddings=vectors[start:stop].tolist(),
                documents=texts[start:stop],
            )
        del collection, client
        return f"chroma://{path}#documents"

    if backend == "qdrant":
        from qdrant_client import QdrantClient, models

        client = QdrantClient(path=str(path))
        client.create_collection(
            "documents",
            vectors_config=models.VectorParams(
                size=vectors.shape[1], distance=models.Distance.COSINE
            ),
        )
        for start in range(0, len(ids), 2000):
            stop = min(start + 2000, len(ids))
            client.upsert(
                "documents",
                points=[
                    models.PointStruct(
                        id=i + 1,
                        vector=vectors[i].tolist(),
                        payload={"id": ids[i], "text": texts[i]},
                    )
                    for i in range(start, stop)
                ],
            )
        client.close()
        return f"qdrant://{path}#documents"

    if backend == QDRANT_SERVER:
        from qdrant_client import QdrantClient, models

        client = QdrantClient(url="http://localhost:6333")
        name = "health_documents"
        if client.collection_exists(name):
            client.delete_collection(name)
        # `indexing_threshold` is per segment, and the default (10,000) leaves a
        # collection of this size served by brute force — measured: 20,000
        # points across 4 segments indexed 0 vectors. Lowered so the graph this
        # spike is about actually exists.
        client.create_collection(
            name,
            vectors_config=models.VectorParams(
                size=vectors.shape[1], distance=models.Distance.COSINE
            ),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1000),
        )
        for start in range(0, len(ids), 1000):
            stop = min(start + 1000, len(ids))
            client.upsert(
                name,
                wait=True,
                points=[
                    models.PointStruct(
                        id=i + 1,
                        vector=vectors[i].tolist(),
                        payload={"id": ids[i], "text": texts[i]},
                    )
                    for i in range(start, stop)
                ],
            )
        indexed = _await_qdrant(client, name)
        client.close()
        print(f"    qdrant indexed {indexed:,} of {len(ids):,} vectors into HNSW", flush=True)
        return f"qdrant://localhost:6333#{name}"

    if backend == "lancedb":
        import lancedb

        connection = lancedb.connect(str(path))
        connection.create_table(
            "documents",
            data=[
                {"id": ids[i], "vector": vectors[i].tolist(), "text": texts[i]}
                for i in range(len(ids))
            ],
        )
        return f"lancedb://{path}#documents"

    if backend == "sqlite-vec":
        import sqlite3

        import sqlite_vec

        path.mkdir(parents=True, exist_ok=True)
        database = path / "index.db"
        connection = sqlite3.connect(database)
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
        connection.execute("CREATE TABLE documents (id TEXT NOT NULL, text TEXT)")
        connection.execute(
            f"CREATE VIRTUAL TABLE vec_documents USING vec0(embedding float[{vectors.shape[1]}])"
        )
        for n, record_id in enumerate(ids, start=1):
            connection.execute(
                "INSERT INTO documents(rowid, id, text) VALUES (?, ?, ?)",
                (n, record_id, texts[n - 1]),
            )
            connection.execute(
                "INSERT INTO vec_documents(rowid, embedding) VALUES (?, ?)",
                (n, vectors[n - 1].astype("<f4").tobytes()),
            )
        connection.commit()
        connection.close()
        return f"sqlite-vec://{database}#vec_documents"

    if backend == "faiss":
        import faiss

        path.mkdir(parents=True, exist_ok=True)
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(vectors.shape[1]))
        index.add_with_ids(vectors, np.arange(1000, 1000 + len(ids) * 3, 3, dtype=np.int64))
        database = path / "vectors.faiss"
        faiss.write_index(index, str(database))
        database.with_suffix(".faiss.meta.json").write_text(
            json.dumps({"ids": ids, "texts": texts}), encoding="utf-8"
        )
        return f"faiss://{database}"

    msg = f"unknown backend {backend}"
    raise ValueError(msg)


#: How much a transform is allowed to disturb the geometry the graph was built
#: from. Each is a real thing `auto` can select, except `shuffle`, which is the
#: positive control: it moves every vector somewhere unrelated, so a measurement
#: that cannot see *that* cannot see anything.
ADAPTERS = ("procrustes", "procrustes_centered", "linear", "low_rank_affine", "shuffle")


class _Shuffle:
    """Assign each record another record's vector. Not an adapter — a control.

    Every vector in the collection stays a real vector of the collection, so
    nothing about the distribution changes and no norm check or dimension check
    can notice. What changes is which record holds which, which is precisely the
    thing an HNSW graph encodes and cannot re-derive.
    """

    kind = "shuffle"
    type_name = "shuffle"

    def __init__(self, permutation: np.ndarray, vectors: np.ndarray) -> None:
        self._by_row = {tuple(v.tolist()): i for i, v in enumerate(vectors)}
        self._vectors = vectors
        self._permutation = permutation
        self.input_dim = vectors.shape[1]
        self.output_dim = vectors.shape[1]

    def apply(self, x: np.ndarray) -> np.ndarray:
        rows = [self._by_row.get(tuple(v.tolist()), 0) for v in np.atleast_2d(x)]
        return self._vectors[self._permutation[rows]]

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"permutation": self._permutation.astype(np.float32)}

    def n_params(self) -> int:
        return int(self._permutation.size)


def build_adapter(kind: str, vectors: np.ndarray, rng: np.random.Generator, *, dim: int) -> Any:
    """The transform the migration will apply to every vector.

    The axis this spike exists to vary. An orthogonal map preserves every inner
    product, so the neighbour structure the graph encodes is still true after
    it — which is why measuring only that would report "no problem" about a
    problem that depends entirely on how much the geometry moved.
    """
    if kind == "shuffle":
        return _Shuffle(rng.permutation(len(vectors)), vectors)

    from rebasis.core import fit_candidates

    # Fitted against a target that is a rotation *plus noise*: a pure rotation
    # is recoverable exactly by every candidate here, and an adapter that fits
    # its target exactly is an orthogonal map whatever family it came from.
    rotation = np.linalg.qr(rng.standard_normal((dim, dim)))[0].astype(np.float32)
    source = vectors[:4000]
    target = l2_normalize(
        source @ rotation.T + rng.standard_normal(source.shape).astype(np.float32) * 0.25
    )
    candidates = fit_candidates(source, target, normalize=False, methods=[kind])
    if not candidates:
        msg = f"no candidate fitted for {kind}"
        raise RuntimeError(msg)
    return candidates[0].adapter


def _qdrant_reindex(*, probes: int, seed: int) -> float:
    """Rebuild through the shipped `QdrantStore.rebuild_index`.

    Deliberately the real code path rather than a client call written for the
    spike: what is being measured is whether *rebasis'* reindex recovers the
    loss, and a second implementation here could recover it while the one that
    ships does not.

    The rebuild is scheduled, not synchronous — Qdrant keeps serving from the
    old index while it builds the new one, which is the property that makes it
    worth exposing — so this waits for green before measuring.
    """
    name = "health_documents"
    uri = f"qdrant://localhost:6333#{name}"
    store = open_store(uri)
    try:
        store.rebuild_index()
    finally:
        closer = getattr(store, "close", None)
        if callable(closer):
            closer()

    from qdrant_client import QdrantClient

    client = QdrantClient(url="http://localhost:6333")
    indexed = _await_qdrant(client, name)
    client.close()
    print(f"    qdrant reindexed {indexed:,} vectors", flush=True)

    reopened = open_store(uri)
    try:
        return measure_index_health(reopened, sample=probes, seed=seed).recall
    finally:
        closer = getattr(reopened, "close", None)
        if callable(closer):
            closer()


def _rebuild(backend: str, root: Path, store: Any, *, probes: int, seed: int) -> float:
    """Insert the migrated vectors into a fresh collection and measure again.

    Not something rebasis does or should do — recreating a user's collection is
    exactly the ownership this tool does not take. It is done here to answer
    whether the loss is in the graph (recoverable by rebuilding it) or in the
    vectors (not recoverable at all), because those call for different things
    from the tool.
    """
    if backend == QDRANT_SERVER:
        return _qdrant_reindex(probes=probes, seed=seed)

    records = [(r.id, r.vector) for r in store.iter_records(with_text=False)]
    ids = [record_id for record_id, _ in records]
    vectors = np.vstack([vector for _, vector in records])
    closer = getattr(store, "close", None)
    if callable(closer):
        closer()

    fresh = root / "rebuilt"
    uri = build(backend, fresh, ids, vectors)
    reopened = open_store(uri)
    try:
        return measure_index_health(reopened, sample=probes, seed=seed).recall
    finally:
        closer = getattr(reopened, "close", None)
        if callable(closer):
            closer()


def run_backend(
    backend: str,
    root: Path,
    *,
    n: int,
    dim: int,
    probes: int,
    seed: int,
    adapter_kind: str,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Measure one backend before and after a full migration."""
    rng = np.random.default_rng(seed)
    # A clustered corpus rather than uniform noise: on uniform vectors every
    # neighbour is equidistant, HNSW has nothing to exploit, and both recalls
    # land wherever the tie-breaking does.
    centers = (rng.standard_normal((64, dim)) * 3.0).astype(np.float32)
    assignment = rng.integers(0, 64, size=n)
    vectors = l2_normalize(
        centers[assignment] + rng.standard_normal((n, dim)).astype(np.float32) * 1.5
    )
    ids = [f"doc-{i:06d}" for i in range(n)]

    started = time.perf_counter()
    uri = build(backend, root, ids, vectors)
    built = time.perf_counter() - started

    store = open_store(uri)
    before = measure_index_health(store, sample=probes, seed=seed)

    adapter = build_adapter(adapter_kind, vectors, rng, dim=dim)

    engine = MigrationEngine(
        db=ManifestDB(manifest_path(root / f"state-{backend}")),
        store=store,
        adapter=adapter,
        shadow_root=root / f"shadow-{backend}",
        batch_size=512,
        power_aware=False,
    )
    engine.prepare(ids)
    migrate_started = time.perf_counter()
    result = engine.run()
    migrated = time.perf_counter() - migrate_started

    after = measure_index_health(engine.store, sample=probes, seed=seed)

    # The follow-up question, and the one that decides whether a `rebuild_index`
    # capability is worth having: is the loss recoverable? Rebuilt by reading
    # the migrated vectors back out and inserting them into a fresh collection,
    # which is what any backend's own reindex does — the graph is constructed
    # from the geometry that is actually there now.
    rebuilt = None
    if rebuild:
        rebuilt = _rebuild(backend, root, engine.store, probes=probes, seed=seed)

    return {
        "backend": backend,
        "adapter": adapter_kind,
        "n": n,
        "dim": dim,
        "probes": probes,
        "build_seconds": round(built, 1),
        "migrate_seconds": round(migrated, 1),
        "migrated": result.processed,
        "recall_before": round(before.recall, 4),
        "recall_after": round(after.recall, 4),
        "delta": round(after.recall - before.recall, 4),
        "recall_rebuilt": None if rebuilt is None else round(rebuilt, 4),
        "delta_after_rebuild": (None if rebuilt is None else round(rebuilt - before.recall, 4)),
        "health_seconds": round(before.duration_seconds, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--probes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="After migrating, rebuild the collection from scratch and measure again",
    )
    parser.add_argument("--backend", action="append", default=None)
    parser.add_argument("--adapter", action="append", default=None, choices=ADAPTERS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    backends = args.backend or list(BACKENDS)
    adapters = args.adapter or list(ADAPTERS)
    rows = []
    for backend in backends:
        for adapter_kind in adapters:
            root = Path(tempfile.mkdtemp(prefix=f"rebasis-health-{backend}-"))
            try:
                row = run_backend(
                    backend,
                    root,
                    n=args.n,
                    dim=args.dim,
                    probes=args.probes,
                    seed=args.seed,
                    adapter_kind=adapter_kind,
                    rebuild=args.rebuild,
                )
            except Exception as exc:
                row = {
                    "backend": backend,
                    "adapter": adapter_kind,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                shutil.rmtree(root, ignore_errors=True)
            rows.append(row)
            print(json.dumps(row), flush=True)

    print()
    header = f"{'backend':12s} {'adapter':20s} {'before':>8s} {'after':>8s} {'delta':>8s}"
    print(header)
    for row in rows:
        if "error" in row:
            print(f"{row['backend']:12s} {row['adapter']:20s} {row['error']}")
            continue
        rebuilt = row.get("recall_rebuilt")
        print(
            f"{row['backend']:12s} {row['adapter']:20s} {row['recall_before']:8.3f} "
            f"{row['recall_after']:8.3f} {row['delta']:+8.3f}"
            + ("" if rebuilt is None else f"   rebuilt {rebuilt:.3f}")
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
