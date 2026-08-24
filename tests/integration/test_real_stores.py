"""Real store backends against their actual client libraries.

Skipped when the client is not installed, so a core-only environment still runs
a green suite. On the GPU host, where the extras are present, these exercise the
paths that matter: reading vectors back out, streaming laziness, and the
dimension lock that motivated the whole project (K13).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from rebasis.core import l2_normalize
from rebasis.errors import EmbeddingDimensionMismatch, RebasisError
from rebasis.store import parse_store_uri

if TYPE_CHECKING:
    from pathlib import Path

chromadb = pytest.importorskip("chromadb", reason="chroma extra not installed")

DIM = 48
N = 600  # above the MIN_SAMPLE floor once a query set is held out


@pytest.fixture
def chroma_uri(tmp_path: Path, rng: np.random.Generator) -> str:
    """A populated Chroma collection on disk."""
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = client.create_collection(name="documents")
    vectors = l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))
    collection.add(
        ids=[f"doc-{i}" for i in range(N)],
        embeddings=vectors.tolist(),
        documents=[f"the text of document {i}" for i in range(N)],
    )
    return f"chroma:///{tmp_path / 'chroma'}#documents"


@pytest.fixture
def chroma_store(chroma_uri: str) -> Any:
    from rebasis.store.backends.chroma import ChromaStore

    return ChromaStore.from_uri(parse_store_uri(chroma_uri))


@pytest.mark.integration
def test_chroma_reports_count_and_dimension(chroma_store: Any) -> None:
    """The two facts every probe starts from."""
    assert chroma_store.count() == N
    assert chroma_store.dimension() == DIM


@pytest.mark.integration
def test_chroma_streams_records_lazily(chroma_store: Any) -> None:
    """Paged, not materialised.

    Chroma's `.get()` with no limit would pull the whole collection into memory,
    which breaks the O(batch × d) invariant on exactly the corpora where it
    matters.
    """
    import types

    stream = chroma_store.iter_records(batch_size=32)
    assert isinstance(stream, types.GeneratorType)

    records = list(stream)
    assert len(records) == N
    assert len({r.id for r in records}) == N
    assert records[0].vector is not None
    assert records[0].vector.shape == (DIM,)
    assert records[0].text


@pytest.mark.integration
def test_chroma_search_finds_an_exact_match(chroma_store: Any) -> None:
    """Searching with a record's own vector returns that record first."""
    record = next(chroma_store.iter_records())
    hits = chroma_store.search(record.vector, k=5)
    assert len(hits) == 5
    assert hits[0].id == record.id


@pytest.mark.integration
def test_chroma_upsert_replaces_without_adding(chroma_store: Any) -> None:
    """The write path upserts. It never creates and never deletes."""
    before = chroma_store.count()
    replacement = l2_normalize(np.ones((2, DIM), dtype=np.float32))
    chroma_store.upsert_vectors(["doc-0", "doc-1"], replacement)

    assert chroma_store.count() == before
    written = {r.id: r.vector for r in chroma_store.iter_records(["doc-0", "doc-1"])}
    assert np.allclose(written["doc-0"], replacement[0], atol=1e-5)


@pytest.mark.integration
def test_chroma_dimension_lock_is_reported_clearly(chroma_store: Any) -> None:
    """K13, the constraint that motivated the project.

    Writing a differently-sized vector must fail with an explanation that points
    at the way around it: bridging needs no dimension change at all. The hint
    used to name a `--direction` flag that does not exist, which is worse than
    no hint — so this asserts the advice, not one spelling of it.
    """
    wrong_size = l2_normalize(np.ones((1, DIM * 2), dtype=np.float32))
    with pytest.raises(EmbeddingDimensionMismatch) as excinfo:
        chroma_store.upsert_vectors(["doc-0"], wrong_size)

    hint = excinfo.value.hint or ""
    assert "no dimension change" in hint
    assert "--" not in hint, f"the hint names a flag: {hint}"
    assert chroma_store.capabilities.dimension_locked


@pytest.mark.integration
def test_chroma_missing_ids_raise(chroma_store: Any) -> None:
    """Silently skipping a missing id would corrupt a migration invisibly."""
    with pytest.raises(RebasisError):
        list(chroma_store.iter_records(["doc-0", "no-such-document"]))


@pytest.mark.integration
def test_chroma_dimension_mismatch_on_search_is_translated(chroma_store: Any) -> None:
    """No chromadb exception crosses the boundary."""
    with pytest.raises(EmbeddingDimensionMismatch):
        chroma_store.search(np.zeros(DIM * 3, dtype=np.float32), k=5)


@pytest.mark.integration
def test_chroma_declares_capabilities_truthfully(chroma_store: Any) -> None:
    """A declared capability must actually work."""
    capabilities = chroma_store.capabilities
    assert capabilities.name == "chroma"
    assert capabilities.can_read_vectors
    assert capabilities.can_upsert_vectors
    assert capabilities.dimension_locked


@pytest.mark.integration
def test_probe_runs_end_to_end_against_chroma(chroma_store: Any, rng: np.random.Generator) -> None:
    """The whole point: measure drift against a real store.

    A rotation with no noise is fully bridgeable, so the recommendation should be
    to leave the index alone — which is the outcome the tool exists to be able to
    give.
    """
    from rebasis.probe import build_tier0, run_probe
    from rebasis.sample import split_disjoint, stratified_sample

    old = np.vstack([r.vector for r in chroma_store.iter_records()])
    rotation = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype(np.float32)
    new = l2_normalize(old @ rotation.T)

    sample = stratified_sample(old, size=N, seed=0)
    query_idx, fit_idx = split_disjoint(sample, query_size=150, seed=0)
    truth = build_tier0(new, new[query_idx], query_idx, k=10)

    result = run_probe(
        old_doc_vectors=old,
        new_doc_vectors=new,
        fit_indices=fit_idx,
        ground_truth=truth,
        old_query_vectors=old[query_idx],
        new_query_vectors=new[query_idx],
        k=10,
        seed=0,
    )
    assert result.decision.decision == "bridge_sufficient"
