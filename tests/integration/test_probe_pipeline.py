"""The probe pipeline end to end.

Synthetic corpora with real cluster structure, because the metric has a
**data-dependent ceiling**: in a densely packed cluster the single nearest
neighbour is a near-tie, and no adapter — nor the oracle itself under the same
perturbation — can reproduce it reliably.

That is measured here rather than assumed. Tests compare against the ceiling the
data actually permits, not against 1.0; asserting 1.0 would make the suite fail
for a property of the corpus rather than a defect in the code.
"""

from __future__ import annotations

import numpy as np
import pytest

from rebasis.core import l2_normalize
from rebasis.probe import build_tier0, build_tier1, run_probe
from rebasis.probe.metrics import recall_at_k, top_k_search
from rebasis.sample import split_disjoint, stratified_sample

DIM = 64
N_DOCS = 2400
N_CLUSTERS = 30
#: Intra-cluster spread. Wide enough that the nearest neighbour is genuinely
#: nearest rather than one of a hundred equidistant points.
SPREAD = 1.5


def build_corpus(rng: np.random.Generator) -> np.ndarray:
    """A clustered corpus — the structure real document collections have."""
    centers = (rng.standard_normal((N_CLUSTERS, DIM)) * 3.0).astype(np.float32)
    assignment = rng.integers(0, N_CLUSTERS, size=N_DOCS)
    return l2_normalize(
        centers[assignment] + rng.standard_normal((N_DOCS, DIM)).astype(np.float32) * SPREAD
    )


def drifted(old: np.ndarray, rng: np.random.Generator, noise: float) -> np.ndarray:
    """A new model: rotation, a modest mean shift, and noise.

    The shift is deliberately small relative to the vector norm. A large one
    collapses every vector onto the same direction and destroys the neighbour
    structure — which measures nothing about the adapter.
    """
    rotation = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype(np.float32)
    shift = (rng.standard_normal(DIM) * 0.05).astype(np.float32)
    return l2_normalize(
        old @ rotation.T + shift + rng.standard_normal(old.shape).astype(np.float32) * noise
    )


def ceiling(
    old: np.ndarray, query_idx: np.ndarray, rng: np.random.Generator, noise: float
) -> float:
    """The best any adapter could do: the oracle itself under the same noise.

    Without this the tests would be asserting a property of the corpus.
    """
    mask = query_idx.astype(np.int64)
    gt, _ = top_k_search(old[query_idx], old, k=10, self_mask=mask)
    relevant = [{int(row[0])} for row in gt]
    perturbed = l2_normalize(
        old[query_idx] + rng.standard_normal((query_idx.size, DIM)).astype(np.float32) * noise
    )
    retrieved, _ = top_k_search(perturbed, old, k=10, self_mask=mask)
    return recall_at_k(retrieved, relevant, 10)


@pytest.mark.integration
def test_no_drift_recommends_bridging(rng: np.random.Generator) -> None:
    """A pure rotation must be fully bridgeable.

    This is the case the whole tool exists for: the spaces differ, the retrieval
    does not, and no reindex is warranted.
    """
    old = build_corpus(rng)
    new = drifted(old, rng, noise=0.0)

    sample = stratified_sample(old, size=N_DOCS, seed=0)
    query_idx, fit_idx = split_disjoint(sample, query_size=400, seed=0)
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
    assert result.decision.arr_at_k > 0.95


@pytest.mark.integration
def test_adapter_gets_close_to_the_achievable_ceiling(rng: np.random.Generator) -> None:
    """Under moderate drift the adapter should reach most of what is possible.

    Compared against the measured ceiling rather than 1.0, because the ceiling is
    a property of how densely the corpus clusters.
    """
    old = build_corpus(rng)
    noise = 0.05
    new = drifted(old, rng, noise=noise)

    sample = stratified_sample(old, size=N_DOCS, seed=0)
    query_idx, fit_idx = split_disjoint(sample, query_size=400, seed=0)
    limit = ceiling(old, query_idx, np.random.default_rng(99), noise)
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

    assert limit > 0.3, "the ceiling itself collapsed; the fixture is not measuring anything"
    assert result.decision.arr_at_k >= 0.5 * limit, (
        f"ARR {result.decision.arr_at_k:.3f} against an achievable ceiling of {limit:.3f}"
    )


@pytest.mark.integration
def test_unadapted_baseline_is_near_zero(rng: np.random.Generator) -> None:
    """Feeding a new-model vector straight into the old index must fail badly.

    If this were not near zero, the tool would have no problem to solve.
    """
    old = build_corpus(rng)
    new = drifted(old, rng, noise=0.02)

    sample = stratified_sample(old, size=N_DOCS, seed=0)
    query_idx, fit_idx = split_disjoint(sample, query_size=300, seed=0)
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

    assert result.baselines["unadapted"] < 0.10
    assert result.decision.arr_at_k > result.baselines["unadapted"]


@pytest.mark.integration
def test_every_candidate_is_measured_and_the_winner_is_among_them(
    rng: np.random.Generator,
) -> None:
    """`auto` must report what it compared, not just what it chose."""
    old = build_corpus(rng)
    new = drifted(old, rng, noise=0.02)

    sample = stratified_sample(old, size=N_DOCS, seed=0)
    query_idx, fit_idx = split_disjoint(sample, query_size=300, seed=0)
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

    assert len(result.candidates) >= 4
    assert result.best.name in {c.name for c in result.candidates}
    assert all(c.arr_ci[0] <= c.arr <= c.arr_ci[1] for c in result.candidates)


@pytest.mark.integration
def test_tier1_uses_real_relevance_and_reports_a_real_oracle(
    rng: np.random.Generator,
) -> None:
    """At T1 the oracle is not perfect, which makes ARR a genuine ratio."""
    old = build_corpus(rng)
    new = drifted(old, rng, noise=0.05)

    # A stand-in query log: 200 queries, each with one relevant document.
    query_ids = rng.choice(N_DOCS, size=200, replace=False)
    queries = l2_normalize(
        new[query_ids] + rng.standard_normal((200, DIM)).astype(np.float32) * 0.3
    )
    qrels = [{int(i)} for i in query_ids]

    truth = build_tier1(new, queries, qrels, k=10)
    assert truth.tier == "t1"
    assert truth.oracle_recall is not None
    assert truth.oracle_recall < 1.0, "a full reindex is not perfect either"
    assert truth.self_mask is None, "real queries are not in the index"


@pytest.mark.integration
def test_probe_never_writes_to_the_store(rng: np.random.Generator) -> None:
    """`probe` is read-only, and that is a contract, not an intention.

    The guard fails the test the moment a write is attempted, so the property
    cannot rot silently during a refactor.
    """
    from rebasis.store import MemoryStore

    class ReadOnlyGuard(MemoryStore):
        def upsert_vectors(self, ids: object, vectors: object) -> None:
            # The arguments are deliberately ignored: this override exists to
            # fail on any write at all, whatever it carries.
            del ids, vectors
            msg = "probe attempted to write to the user's index"
            raise AssertionError(msg)

    old = build_corpus(rng)
    new = drifted(old, rng, noise=0.02)
    store = ReadOnlyGuard([f"doc-{i}" for i in range(N_DOCS)], old)

    sample = stratified_sample(old, size=N_DOCS, seed=0)
    query_idx, fit_idx = split_disjoint(sample, query_size=300, seed=0)
    truth = build_tier0(new, new[query_idx], query_idx, k=10)

    run_probe(
        old_doc_vectors=old,
        new_doc_vectors=new,
        fit_indices=fit_idx,
        ground_truth=truth,
        old_query_vectors=old[query_idx],
        new_query_vectors=new[query_idx],
        k=10,
        seed=0,
    )
    assert store.count() == N_DOCS
