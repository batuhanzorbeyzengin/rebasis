"""The two-adapter strategy for asymmetric models.

A model like E5 encodes `query: what is x` and `passage: x is ...` differently.
Fitting an adapter on document-encoded pairs and then applying it to query
vectors is a train/serve mismatch — it raises nothing and only costs quality.

Two adapters were the original answer. M0 measured the difference at −0.003 on
average with the sign varying by model pair, so neither strategy wins on
principle. `auto` therefore fits both and keeps whichever scores better.
"""

from __future__ import annotations

import numpy as np
import pytest

from rebasis.core import fit_candidates, l2_normalize
from rebasis.embed import PrecomputedEmbedder
from rebasis.probe.session import probe_store
from rebasis.store import MemoryStore
from rebasis.types import EncodingProfile

pytestmark = pytest.mark.unit

DIM = 32
N = 700


@pytest.fixture
def corpus(rng: np.random.Generator, make_corpus, make_drift) -> dict[str, object]:  # type: ignore[no-untyped-def]
    """A corpus whose new model encodes queries genuinely differently.

    The two encodings are related by their own rotation, so an adapter fitted on
    one is measurably wrong for the other — the mismatch above, made explicit.
    """
    old = make_corpus(rng, n_docs=N, dim=DIM, n_clusters=16)
    as_documents = make_drift(old, rng, shift=0.0)
    skew = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype(np.float32)
    as_queries = l2_normalize(as_documents @ skew.T)

    texts = [f"document {i}" for i in range(N)]
    return {
        "ids": [f"doc-{i}" for i in range(N)],
        "texts": texts,
        "old": old,
        "documents": as_documents,
        "queries": as_queries,
    }


def embedder(corpus: dict[str, object], *, asymmetric: bool) -> PrecomputedEmbedder:
    profile = EncodingProfile(
        model_id="new-model",
        dim=DIM,
        query_prefix="query: " if asymmetric else None,
        document_prefix="passage: " if asymmetric else None,
    )
    texts = corpus["texts"]
    return PrecomputedEmbedder(
        profile,
        dict(zip(texts, corpus["documents"], strict=True)),  # type: ignore[arg-type]
        query_vectors=dict(zip(texts, corpus["queries"], strict=True)),  # type: ignore[arg-type]
    )


class TestCandidateTagging:
    def test_a_candidate_records_which_encoding_it_was_fitted_on(
        self, rng: np.random.Generator
    ) -> None:
        src = l2_normalize(rng.standard_normal((200, DIM)).astype(np.float32))
        dst = l2_normalize(rng.standard_normal((200, DIM)).astype(np.float32))

        document_side = fit_candidates(src, dst, methods=["procrustes"], normalize=False)
        query_side = fit_candidates(
            src, dst, methods=["procrustes"], normalize=False, fit_kind="query"
        )

        assert document_side[0].name == "procrustes"
        assert query_side[0].name == "procrustes@query"

    def test_the_suffix_survives_the_csls_suffix(self, rng: np.random.Generator) -> None:
        """Two candidates of the same type fitted on different encodings are
        different adapters. A report that called them both `procrustes` would
        be hiding the comparison it exists to show."""
        src = l2_normalize(rng.standard_normal((200, DIM)).astype(np.float32))
        dst = l2_normalize(rng.standard_normal((200, DIM)).astype(np.float32))

        candidate = fit_candidates(
            src, dst, methods=["procrustes"], normalize=False, fit_kind="query"
        )[0]
        candidate.use_csls = True

        assert candidate.name == "procrustes+csls@query"


class TestProbeIntegration:
    def test_an_asymmetric_model_gets_both_strategies(self, corpus: dict[str, object]) -> None:
        store = MemoryStore(corpus["ids"], corpus["old"], corpus["texts"])  # type: ignore[arg-type]

        result, _ = probe_store(
            store, embedder(corpus, asymmetric=True), size=600, heldout=150, seed=0
        )

        names = {c.name for c in result.candidates}
        assert any("@query" in n for n in names), names
        assert any("@query" not in n for n in names), names

    def test_a_symmetric_model_does_not_pay_for_the_second_pass(
        self, corpus: dict[str, object]
    ) -> None:
        """The two encodings would be identical, so the extra fit buys nothing."""
        store = MemoryStore(corpus["ids"], corpus["old"], corpus["texts"])  # type: ignore[arg-type]

        result, _ = probe_store(
            store, embedder(corpus, asymmetric=False), size=600, heldout=150, seed=0
        )

        assert not any("@query" in c.name for c in result.candidates)

    def test_the_winner_is_whichever_scored_best(self, corpus: dict[str, object]) -> None:
        """`auto`'s only promise: the highest scorer wins, whichever strategy
        it came from. Nothing about asymmetry decides it in advance."""
        store = MemoryStore(corpus["ids"], corpus["old"], corpus["texts"])  # type: ignore[arg-type]

        result, _ = probe_store(
            store, embedder(corpus, asymmetric=True), size=600, heldout=150, seed=0
        )

        assert result.best.arr == pytest.approx(max(c.arr for c in result.candidates))

    def test_at_t0_the_shared_strategy_is_structurally_favoured(
        self, corpus: dict[str, object]
    ) -> None:
        """A limitation of the tier, recorded rather than worked around.

        T0's ground truth is what the new model retrieves in its **document**
        space. An adapter fitted on document-encoded pairs reproduces exactly
        that geometry, so at T0 it is optimal by construction and the
        query-specific strategy cannot beat it however good it is.

        **T0 cannot evaluate the query encoding at all.** It is the same
        structural blind spot that makes T0 unable to detect a wrong prefix
        (`docs/golden-findings.md`, section 4), and it is why the two strategies
        are only genuinely compared when a real query log supplies queries that
        were never documents.
        """
        store = MemoryStore(corpus["ids"], corpus["old"], corpus["texts"])  # type: ignore[arg-type]

        result, _ = probe_store(
            store, embedder(corpus, asymmetric=True), size=600, heldout=150, seed=0
        )

        shared = max(c.arr for c in result.candidates if "@query" not in c.name)
        query_side = max(c.arr for c in result.candidates if "@query" in c.name)
        assert shared >= query_side
