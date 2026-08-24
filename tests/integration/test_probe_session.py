"""The store-to-decision plumbing.

``run_probe`` takes vectors; ``probe_store`` takes a store URI and two model
names. Everything between those two is what these tests cover, and most of it is
bookkeeping that fails silently when it is wrong: a sample drawn from a biased
slice, text paired with the wrong document, a query log whose ids do not match.
None of those raise. They produce a plausible ARR that means nothing, which is
why each one is asserted here rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pytest

from rebasis.core import l2_normalize
from rebasis.embed import PrecomputedEmbedder
from rebasis.errors import InsufficientSamples, StoreUnsupported
from rebasis.probe.session import (
    CLUSTER_POOL_MAX,
    CorpusSample,
    QueryLog,
    draw_corpus_sample,
    probe_store,
)
from rebasis.store import MemoryStore
from rebasis.types import EncodingProfile

pytestmark = pytest.mark.integration

DIM = 48
N_DOCS = 900


def corpus(rng: np.random.Generator) -> tuple[list[str], np.ndarray, list[str]]:
    """A clustered corpus with one distinct text per record."""
    centers = (rng.standard_normal((20, DIM)) * 3.0).astype(np.float32)
    assignment = rng.integers(0, 20, size=N_DOCS)
    vectors = l2_normalize(
        centers[assignment] + rng.standard_normal((N_DOCS, DIM)).astype(np.float32) * 1.5
    )
    ids = [f"doc-{i:05d}" for i in range(N_DOCS)]
    texts = [f"text for {i}" for i in range(N_DOCS)]
    return ids, vectors, texts


def new_model(old: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A rotation plus a small shift — a different model, same corpus."""
    dim = old.shape[1]
    rotation = np.linalg.qr(rng.standard_normal((dim, dim)))[0].astype(np.float32)
    return l2_normalize(old @ rotation.T + rng.standard_normal(dim).astype(np.float32) * 0.05)


def embedder(texts: list[str], vectors: np.ndarray, model_id: str) -> PrecomputedEmbedder:
    profile = EncodingProfile(model_id=model_id, dim=DIM)
    return PrecomputedEmbedder(profile, dict(zip(texts, vectors, strict=True)))


class TestDrawCorpusSample:
    def test_vectors_are_the_stores_own(self, rng: np.random.Generator) -> None:
        """The sample carries the index's vectors, not re-derived ones.

        The whole premise is that the index is not touched, so the old-model
        side of every comparison must come out of it verbatim.
        """
        ids, vectors, texts = corpus(rng)
        store = MemoryStore(ids, vectors, texts)

        drawn = draw_corpus_sample(store, size=450, heldout=100, seed=7)

        by_id = dict(zip(ids, vectors, strict=True))
        for position, record_id in enumerate(drawn.ids):
            expected = l2_normalize(by_id[record_id][None, :])[0]
            np.testing.assert_allclose(drawn.old_vectors[position], expected, atol=1e-6)

    def test_text_is_matched_by_id_not_by_position(self, rng: np.random.Generator) -> None:
        """Backends return ids in their own order; the mapping must not assume one.

        A positional zip here would pair each document with someone else's text
        and every downstream number would be quietly meaningless.
        """
        ids, vectors, texts = corpus(rng)
        store = ReversingStore(ids, vectors, texts)

        drawn = draw_corpus_sample(store, size=450, heldout=100, seed=3)

        expected = dict(zip(ids, texts, strict=True))
        assert [expected[i] for i in drawn.ids] == drawn.texts

    def test_query_and_fit_sets_are_disjoint(self, rng: np.random.Generator) -> None:
        """Any overlap invalidates every ARR the tool has ever printed."""
        ids, vectors, texts = corpus(rng)
        drawn = draw_corpus_sample(MemoryStore(ids, vectors, texts), size=450, heldout=100, seed=1)

        assert not set(drawn.query_positions.tolist()) & set(drawn.fit_positions.tolist())

    def test_records_without_text_are_dropped(self, rng: np.random.Generator) -> None:
        """A vector with no text cannot be re-embedded, so it is only noise."""
        ids, vectors, texts = corpus(rng)
        holed = [t if i % 3 else "" for i, t in enumerate(texts)]
        drawn = draw_corpus_sample(MemoryStore(ids, vectors, holed), size=600, heldout=100, seed=5)

        assert len(drawn) == len(drawn.texts) == drawn.old_vectors.shape[0]
        assert all(text.strip() for text in drawn.texts)

    def test_a_store_with_no_text_at_all_says_so(self, rng: np.random.Generator) -> None:
        ids, vectors, texts = corpus(rng)
        with pytest.raises(StoreUnsupported, match="carry text"):
            draw_corpus_sample(MemoryStore(ids, vectors, [""] * len(texts)), size=450, heldout=100)

    def test_the_pool_is_capped_regardless_of_corpus_size(self) -> None:
        """Memory is O(pool × d), never O(N × d)."""
        assert CLUSTER_POOL_MAX == 50_000


class TestProbeStore:
    def test_tier0_reaches_a_decision(self, rng: np.random.Generator) -> None:
        ids, vectors, texts = corpus(rng)
        store = MemoryStore(ids, vectors, texts)
        new = embedder(texts, new_model(vectors, rng), "new-model")

        result, sample = probe_store(store, new, size=600, heldout=150, seed=11)

        assert result.ground_truth_tier == "t0"
        assert 0.0 <= result.best.arr <= 1.0
        assert result.decision.decision
        assert len(sample) <= 600
        # Not measurable at T0: the ground truth is the new model's own output.
        assert result.decision.upgrade_gain is None

    def test_tier1_uses_the_query_log_and_measures_the_upgrade(
        self, rng: np.random.Generator
    ) -> None:
        ids, vectors, texts = corpus(rng)
        new_vectors = new_model(vectors, rng)
        store = MemoryStore(ids, vectors, texts)

        queries = [f"query {i}" for i in range(120)]
        old = embedder(texts + queries, np.vstack([vectors, vectors[:120]]), "old-model")
        new = embedder(texts + queries, np.vstack([new_vectors, new_vectors[:120]]), "new-model")
        log = QueryLog(queries=queries, qrels=[{ids[i]} for i in range(120)])

        result, _ = probe_store(
            store, new, old_embedder=old, query_log=log, size=700, heldout=150, seed=13
        )

        assert result.ground_truth_tier == "t1"
        assert result.decision.upgrade_gain is not None

    def test_tier1_without_the_old_model_refuses(self, rng: np.random.Generator) -> None:
        """The "do nothing" baseline needs the current model; say so, don't guess."""
        ids, vectors, texts = corpus(rng)
        new = embedder(texts, new_model(vectors, rng), "new-model")
        log = QueryLog(queries=["q"], qrels=[{ids[0]}])

        with pytest.raises(StoreUnsupported, match="current model"):
            probe_store(MemoryStore(ids, vectors, texts), new, query_log=log, size=600, heldout=150)

    def test_a_query_log_that_matches_nothing_says_so(self, rng: np.random.Generator) -> None:
        ids, vectors, texts = corpus(rng)
        queries = ["q0"]
        new = embedder(texts + queries, np.vstack([new_model(vectors, rng), vectors[:1]]), "new")
        old = embedder(texts + queries, np.vstack([vectors, vectors[:1]]), "old")
        log = QueryLog(queries=queries, qrels=[{"not-an-id-in-this-store"}])

        with pytest.raises(InsufficientSamples, match="judged document"):
            probe_store(
                MemoryStore(ids, vectors, texts),
                new,
                old_embedder=old,
                query_log=log,
                size=600,
                heldout=150,
            )

    def test_a_prepared_sample_is_reused_rather_than_redrawn(
        self, rng: np.random.Generator
    ) -> None:
        """Drawing twice would embed the corpus twice — the expensive step."""
        ids, vectors, texts = corpus(rng)
        store = MemoryStore(ids, vectors, texts)
        drawn = draw_corpus_sample(store, size=600, heldout=150, seed=17)
        new = embedder(texts, new_model(vectors, rng), "new-model")

        _, used = probe_store(CountingStore(ids, vectors, texts), new, sample=drawn)

        assert isinstance(used, CorpusSample)
        assert used.ids == drawn.ids


class ReversingStore(MemoryStore):
    """A store that returns requested ids in the opposite order.

    Chroma and LanceDB both answer an id filter in their own order. A backend
    that happens to preserve request order would hide a positional-zip bug, so
    one that deliberately does not is what the mapping is tested against.
    """

    def iter_records(self, ids=None, **kwargs):  # type: ignore[no-untyped-def]
        records = list(super().iter_records(ids, **kwargs))
        return iter(reversed(records)) if ids is not None else iter(records)


class CountingStore(MemoryStore):
    """Records how many times the collection was streamed in full."""

    full_scans = 0

    def iter_records(self, ids=None, **kwargs):  # type: ignore[no-untyped-def]
        if ids is None:
            CountingStore.full_scans += 1
        return super().iter_records(ids, **kwargs)
