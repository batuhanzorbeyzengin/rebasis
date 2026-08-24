"""Shared fixtures.

The determinism fixture is autouse: a non-deterministic test is a broken test.
With ``pytest-randomly`` shuffling execution order, a test that leaks state fails
irreproducibly — which costs more time than the state it was saving.
"""

from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from rebasis.observability import redaction_counter, reset_logging
from rebasis.probe.decision import decide
from rebasis.probe.runner import CandidateMetrics, ProbeResult

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

SEED = 1234


@pytest.fixture(autouse=True)
def deterministic() -> None:
    """Pin every source of randomness before each test.

    No teardown: the next test re-seeds anyway, and restoring the previous
    global state would only reintroduce the order dependence this exists to
    remove.
    """
    random.seed(SEED)
    np.random.seed(SEED)  # noqa: NPY002 - legacy global state must be pinned too
    os.environ["PYTHONHASHSEED"] = str(SEED)
    try:
        import torch

        torch.manual_seed(SEED)
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def _isolate_logging() -> Iterator[None]:
    """Return logging to library mode around every test.

    Without this the first test that configures logging leaks its settings into
    all the others, and the resulting failure depends on execution order.
    """
    reset_logging()
    redaction_counter.reset()
    yield
    reset_logging()
    redaction_counter.reset()


# ── shared corpus builders ────────────────────────────────────────────
#
# Here rather than in a `tests/fixtures/` or `tests/helpers/` module: pytest
# resolves fixtures by name from any conftest above a test, with no import path
# to get wrong. Importable helper modules were tried and reverted — `from
# conftest import ...` works when a file runs alone and collides once the whole
# suite is collected, because several directories contribute a conftest and the
# module name is not unique.

#: Intra-cluster spread for a synthetic corpus. Wide enough that the nearest
#: neighbour is genuinely nearest rather than one of a hundred equidistant
#: points — a densely packed cluster gives the metric a data-dependent ceiling
#: that no adapter, nor the oracle itself, can reach.
CLUSTER_SPREAD = 1.5


def clustered_corpus(
    rng: np.random.Generator,
    *,
    n_docs: int = 1000,
    dim: int = 64,
    n_clusters: int = 20,
    spread: float = CLUSTER_SPREAD,
) -> np.ndarray:
    """A corpus with real cluster structure, ℓ2-normalised.

    Real document collections are clustered, and several properties under test
    depend on that: stratified sampling, `tail_arr`, and the data-dependent
    ceiling the probe tests compare against. Uniform random vectors have none of
    it, and tests written against them pass for the wrong reason.
    """
    from rebasis.core import l2_normalize

    centers = (rng.standard_normal((n_clusters, dim)) * 3.0).astype(np.float32)
    assignment = rng.integers(0, n_clusters, size=n_docs)
    noise = rng.standard_normal((n_docs, dim)).astype(np.float32) * spread
    return l2_normalize(centers[assignment] + noise)


def drifted(
    old: np.ndarray, rng: np.random.Generator, *, noise: float = 0.0, shift: float = 0.05
) -> np.ndarray:
    """A second model's view of the same corpus: rotation, small shift, noise.

    The shift is deliberately small relative to the vector norm. A large one
    collapses every vector onto the same direction and destroys the neighbour
    structure, which measures nothing about the adapter — a mistake made once,
    and recorded in `docs/m0-findings.md`.
    """
    from rebasis.core import l2_normalize

    dim = old.shape[1]
    rotation = np.linalg.qr(rng.standard_normal((dim, dim)))[0].astype(np.float32)
    offset = (rng.standard_normal(dim) * shift).astype(np.float32)
    moved = old @ rotation.T + offset
    if noise:
        moved = moved + rng.standard_normal(old.shape).astype(np.float32) * noise
    return l2_normalize(moved)


@pytest.fixture
def make_corpus() -> Callable[..., np.ndarray]:
    """Build a clustered corpus."""
    return clustered_corpus


@pytest.fixture
def make_drift() -> Callable[..., np.ndarray]:
    """Build a second model's view of a corpus."""
    return drifted


@pytest.fixture
def probe_result() -> Callable[..., ProbeResult]:
    """Build a :class:`ProbeResult` without running a probe.

    The report and the CLI both render results, and neither should need a
    corpus to be tested. Overridable field by field so a test can state the one
    thing it cares about.
    """

    def build(
        *,
        arr: float = 0.93,
        tier: str = "t0",
        score_shift: float | None = 0.02,
        **kwargs: Any,
    ) -> ProbeResult:
        best = _candidate(arr=arr, score_shift_calibrated=score_shift)
        return ProbeResult(
            decision=decide(arr, confidence_interval=best.arr_ci, score_shift=score_shift),
            best=best,
            candidates=[best, _candidate(name="linear", arr=arr - 0.04)],
            baselines={"old_model_only": 0.71, "unadapted": 0.04},
            ground_truth_tier=tier,
            n_documents=10_000,
            n_queries=1_000,
            n_fit_pairs=4_000,
            k=10,
            seed=0,
            duration_seconds=42.5,
            **kwargs,
        )

    return build


def _candidate(*, name: str = "procrustes", arr: float = 0.93, **kwargs: Any) -> CandidateMetrics:
    defaults: dict[str, Any] = {
        "arr_sparse": arr,
        "arr_ci": (arr - 0.02, arr + 0.02),
        "mrr": arr - 0.05,
        "overlap": arr - 0.1,
        "spearman": 0.8,
        "score_shift_raw": 0.9,
        "n_params": 147_456,
    }
    return CandidateMetrics(name=name, arr=arr, **{**defaults, **kwargs})


# ── real store builders ───────────────────────────────────────────────
#
# Here rather than in a helper module for the reason above, and here rather
# than in each suite because two suites need the same five stores: the contract
# suite, which asserts what every backend must do, and the migration suite,
# which asserts what happens when one is written to.

#: Backends that can be stood up from a temporary directory, with no server.
#: Named as a plain tuple because both callers parametrise over it at
#: collection time, which a fixture's return value cannot reach.
LIVE_BACKENDS = ("chroma", "faiss", "lancedb", "qdrant", "sqlite-vec")


def _build_chroma(tmp_path: Any, ids: Any, vectors: Any, texts: Any, dim: int) -> str:
    chromadb = pytest.importorskip("chromadb", reason="the chroma extra is not installed")
    del dim
    path = tmp_path / "chroma"
    client = chromadb.PersistentClient(path=str(path))
    collection = client.create_collection("documents", metadata={"hnsw:space": "cosine"})
    collection.add(ids=list(ids), embeddings=vectors.tolist(), documents=list(texts))
    del client, collection
    return f"chroma://{path}#documents"


def _build_lancedb(tmp_path: Any, ids: Any, vectors: Any, texts: Any, dim: int) -> str:
    lancedb = pytest.importorskip("lancedb", reason="the lancedb extra is not installed")
    del dim
    connection = lancedb.connect(str(tmp_path / "lance"))
    connection.create_table(
        "documents",
        data=[
            {"id": i, "vector": v.tolist(), "text": text}
            for i, v, text in zip(ids, vectors, texts, strict=True)
        ],
    )
    return f"lancedb://{tmp_path / 'lance'}#documents"


def _build_qdrant(tmp_path: Any, ids: Any, vectors: Any, texts: Any, dim: int) -> str:
    pytest.importorskip("qdrant_client", reason="the qdrant extra is not installed")
    from qdrant_client import QdrantClient, models

    path = tmp_path / "qdrant"
    client = QdrantClient(path=str(path))
    client.create_collection(
        "documents",
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )
    client.upsert(
        "documents",
        points=[
            models.PointStruct(id=n + 1, vector=v.tolist(), payload={"id": i, "text": text})
            for n, (i, v, text) in enumerate(zip(ids, vectors, texts, strict=True))
        ],
    )
    # Local mode holds an exclusive lock on the folder; the store under test
    # opens its own client and a handle left here would make that fail.
    client.close()
    return f"qdrant://{path}#documents"


def _build_sqlite_vec(tmp_path: Any, ids: Any, vectors: Any, texts: Any, dim: int) -> str:
    sqlite_vec = pytest.importorskip("sqlite_vec", reason="the sqlite-vec extra is not installed")
    import sqlite3

    path = tmp_path / "index.db"
    connection = sqlite3.connect(path)
    try:
        connection.enable_load_extension(True)  # noqa: FBT003 - sqlite3's own signature
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)  # noqa: FBT003 - sqlite3's own signature
    except (AttributeError, sqlite3.Error):
        pytest.skip("this Python's sqlite3 cannot load extensions")
    connection.execute("CREATE TABLE documents (id TEXT NOT NULL, text TEXT)")
    connection.execute(f"CREATE VIRTUAL TABLE vec_documents USING vec0(embedding float[{dim}])")
    for n, (i, v, text) in enumerate(zip(ids, vectors, texts, strict=True), start=1):
        connection.execute("INSERT INTO documents(rowid, id, text) VALUES (?, ?, ?)", (n, i, text))
        connection.execute(
            "INSERT INTO vec_documents(rowid, embedding) VALUES (?, ?)",
            (n, v.astype("<f4").tobytes()),
        )
    connection.commit()
    connection.close()
    return f"sqlite-vec://{path}#vec_documents"


def _build_faiss(tmp_path: Any, ids: Any, vectors: Any, texts: Any, dim: int) -> str:
    import json

    faiss = pytest.importorskip("faiss", reason="the faiss extra is not installed")
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
    # Labels that are deliberately not row numbers: FAISS addresses an
    # id-mapped index by label, and a fixture where the two coincide cannot
    # tell the difference between a backend that knows that and one that
    # does not.
    index.add_with_ids(vectors, np.arange(1000, 1000 + len(ids) * 3, 3, dtype=np.int64))
    path = tmp_path / "vectors.faiss"
    faiss.write_index(index, str(path))
    path.with_suffix(".faiss.meta.json").write_text(
        json.dumps({"ids": list(ids), "texts": list(texts)}), encoding="utf-8"
    )
    return f"faiss://{path}"


_BUILDERS = {
    "chroma": _build_chroma,
    "faiss": _build_faiss,
    "lancedb": _build_lancedb,
    "qdrant": _build_qdrant,
    "sqlite-vec": _build_sqlite_vec,
}


@pytest.fixture
def make_store() -> Callable[..., str]:
    """Populate a real store of a named backend and return its URI.

    The store is closed before the URI comes back, so the caller opens the only
    live handle. Qdrant's local mode refuses a second one, and a backend that
    caches its client per path would otherwise answer reads from memory rather
    than from what is on disk.
    """

    def build(  # noqa: PLR0913 - a store needs all five, and naming them beats a dict
        backend: str,
        tmp_path: Any,
        ids: Any,
        vectors: Any,
        texts: Any,
        *,
        dim: int | None = None,
    ) -> str:
        return _BUILDERS[backend](
            tmp_path, ids, vectors, texts, dim if dim is not None else vectors.shape[1]
        )

    return build


@pytest.fixture
def release_store() -> Callable[[Any], None]:
    """Release a store handle if this backend has one to release."""
    import contextlib

    def release(store: Any) -> None:
        close = getattr(store, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()

    return release


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator. Preferred over the legacy global numpy random state."""
    return np.random.default_rng(SEED)
