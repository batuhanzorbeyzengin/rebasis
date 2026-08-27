"""The serving API.

Two properties matter more than the rest and are asserted first: that loading
validates and serving does not, and that **torch is never imported**. Both are
about the hot path, where a single query's budget is tens of microseconds.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path  # noqa: TC003 - pytest resolves fixture annotations at runtime

import numpy as np
import pytest

from rebasis import Bridge
from rebasis.core import CenteredProcrustesAdapter, ScoreCalibrator, l2_normalize, save_adapter
from rebasis.errors import IncompatibleAdapter
from rebasis.serve import calibrated_merge, reciprocal_rank_fusion, wrap_retriever
from rebasis.types import EncodingProfile, Hit

DIM = 64
OLD = EncodingProfile(model_id="old/model", dim=DIM)
NEW = EncodingProfile(model_id="new/model", dim=DIM, query_prefix="query: ")


@pytest.fixture
def adapter_path(tmp_path: Path, rng: np.random.Generator) -> Path:
    """A saved adapter with a calibrator."""
    src = l2_normalize(rng.standard_normal((400, DIM)).astype(np.float32))
    dst = l2_normalize(rng.standard_normal((400, DIM)).astype(np.float32))
    return save_adapter(
        CenteredProcrustesAdapter.fit(src, dst),
        tmp_path / "bridge.rbs",
        direction="query_to_old",
        old_profile=OLD,
        new_profile=NEW,
        calibrator=ScoreCalibrator.fit(
            rng.random(200).astype(np.float32) * 0.5, rng.random(200).astype(np.float32)
        ),
    )


@pytest.mark.unit
def test_load_validates_and_serving_does_not(adapter_path: Path) -> None:
    """Validation happens once, at load, never per query."""
    bridge = Bridge.load(adapter_path, expected_old=OLD, expected_new=NEW, verify=True)
    assert bridge.input_dim == DIM
    assert bridge.output_dim == DIM
    assert bridge.adapter_type == "procrustes_centered"


@pytest.mark.unit
def test_wrong_profile_refuses_to_load(adapter_path: Path) -> None:
    """A mismatch is a load failure, not degraded retrieval."""
    with pytest.raises(IncompatibleAdapter):
        Bridge.load(adapter_path, expected_old=EncodingProfile(model_id="other", dim=DIM))


@pytest.mark.unit
def test_to_index_space_returns_unit_vectors(adapter_path: Path, rng: np.random.Generator) -> None:
    """The index expects unit vectors, so the bridge renormalises."""
    bridge = Bridge.load(adapter_path)
    query = l2_normalize(rng.standard_normal((5, DIM)).astype(np.float32))
    mapped = bridge.to_index_space(query)

    assert mapped.shape == (5, DIM)
    assert np.allclose(np.linalg.norm(mapped, axis=1), 1.0, atol=1e-5)


@pytest.mark.unit
def test_batch_and_single_query_agree(adapter_path: Path, rng: np.random.Generator) -> None:
    """A divergence here would show up only in production, under load."""
    bridge = Bridge.load(adapter_path)
    queries = l2_normalize(rng.standard_normal((8, DIM)).astype(np.float32))

    batched = bridge.to_index_space(queries)
    singly = np.vstack([bridge.to_index_space(queries[i : i + 1]) for i in range(8)])
    assert np.allclose(batched, singly, atol=1e-5)


@pytest.mark.unit
def test_serving_does_not_import_torch() -> None:
    """The hot path never touches torch.

    A subprocess, because another test in this process may already have imported
    it — which would make this pass for the wrong reason.
    """
    code = (
        "import sys; from rebasis import Bridge; "
        "assert 'torch' not in sys.modules, 'serving imported torch'"
    )
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.unit
def test_calibration_moves_scores_without_reordering(adapter_path: Path) -> None:
    """The calibrator is monotone, so ranking is untouched by construction."""
    bridge = Bridge.load(adapter_path)
    assert bridge.has_calibrator

    raw = np.array([0.10, 0.25, 0.40, 0.55], dtype=np.float32)
    calibrated = bridge.calibrate_scores(raw)

    assert not np.allclose(calibrated, raw), "calibration did nothing"
    assert np.all(np.diff(calibrated) >= -1e-6), "calibration reordered the results"


@pytest.mark.unit
def test_missing_calibrator_passes_scores_through(tmp_path: Path, rng: np.random.Generator) -> None:
    """Without a calibrator the scores are returned untouched, not faked."""
    src = l2_normalize(rng.standard_normal((400, DIM)).astype(np.float32))
    dst = l2_normalize(rng.standard_normal((400, DIM)).astype(np.float32))
    path = save_adapter(
        CenteredProcrustesAdapter.fit(src, dst),
        tmp_path / "plain.rbs",
        direction="query_to_old",
        old_profile=OLD,
        new_profile=NEW,
    )
    bridge = Bridge.load(path)
    raw = np.array([0.1, 0.2], dtype=np.float32)
    assert not bridge.has_calibrator
    assert np.array_equal(bridge.calibrate_scores(raw), raw)


@pytest.mark.unit
def test_rrf_merges_by_rank_alone() -> None:
    """The fallback: without a calibrator, only ranks are comparable.

    Comparing raw scores across two embedding spaces would let the space with the
    wider distribution win for reasons unrelated to relevance.
    """
    old_hits = [Hit(id=f"o{i}", score=0.9 - 0.1 * i, rank=i) for i in range(3)]
    new_hits = [Hit(id=f"n{i}", score=0.5 - 0.1 * i, rank=i) for i in range(3)]

    merged = reciprocal_rank_fusion(old_hits, new_hits, k=4)
    assert len(merged) == 4
    assert [h.rank for h in merged] == [0, 1, 2, 3]
    # Despite much lower raw scores, the new side's top hit ranks second — which
    # is the point: rank, not score.
    assert {"o0", "n0"} <= {h.id for h in merged[:2]}


@pytest.mark.unit
def test_a_document_in_both_sides_is_not_counted_twice(adapter_path: Path) -> None:
    """During migration, overlap is expected rather than exceptional."""
    bridge = Bridge.load(adapter_path)
    shared = [Hit(id="same", score=0.4, rank=0)]
    merged = calibrated_merge(
        shared, [Hit(id="same", score=0.9, rank=0)], k=5, calibrator=bridge._calibrator
    )
    assert len([h for h in merged if h.id == "same"]) == 1


@pytest.mark.unit
def test_one_side_alone_comes_back_in_the_order_it_arrived(adapter_path: Path) -> None:
    """At 0% and 100% migrated the index holds one space and there is one right
    answer — what the store returned. A merge that cannot reproduce it is wrong
    at the endpoints rather than merely worse there.

    It could not. The isotonic calibrator is a step function: ten distinct
    bridged scores land on five to seven distinct calibrated values, and the sort
    then broke every one of those ties on the document id. Measured over four
    corpora, the single-space ranking survived on 4% to 16% of queries.
    """
    bridge = Bridge.load(adapter_path)
    # Ids deliberately anti-sorted against rank: if the id is doing the ordering
    # this comes back reversed, and if the rank is, it comes back unchanged.
    hits = [Hit(id=f"doc-{9 - i}", score=0.9 - 0.001 * i, rank=i) for i in range(10)]

    merged = calibrated_merge(hits, [], k=10, calibrator=bridge._calibrator)

    assert [h.id for h in merged] == [h.id for h in hits]


@pytest.mark.unit
def test_the_new_side_alone_is_unchanged_too(adapter_path: Path) -> None:
    """The same property from the other end, where no calibration happens at
    all — so a failure here would be the merge itself rather than the step
    function feeding it."""
    bridge = Bridge.load(adapter_path)
    hits = [Hit(id=f"doc-{9 - i}", score=0.9 - 0.001 * i, rank=i) for i in range(10)]

    merged = calibrated_merge([], hits, k=10, calibrator=bridge._calibrator)

    assert [h.id for h in merged] == [h.id for h in hits]


@pytest.mark.unit
def test_a_cross_side_tie_is_still_broken_on_the_id(adapter_path: Path) -> None:
    """The property the id tie-break was there for, and which the fix keeps.

    Two hits from different sides can hold the same merged score *and* the same
    rank — one is rank 0 on the old side, the other rank 0 on the new. Resolving
    that by which side was passed first would be a standing bias toward one
    embedding space, which is what the merge exists to avoid, so the id decides.

    The two sides are not symmetric — only the old one goes through the
    calibrator — so the tie has to be constructed rather than assumed: the new
    hit is given exactly the score the calibrator produces for the old one.
    """
    import numpy as np

    bridge = Bridge.load(adapter_path)
    raw = 0.5
    calibrated = float(bridge._calibrator.transform(np.array([raw], dtype=np.float32))[0])

    merged = calibrated_merge(
        [Hit(id="b", score=raw, rank=0)],
        [Hit(id="a", score=calibrated, rank=0)],
        k=2,
        calibrator=bridge._calibrator,
    )

    assert [h.score for h in merged] == pytest.approx([calibrated, calibrated])
    assert [h.id for h in merged] == ["a", "b"]


@pytest.mark.unit
def test_wrap_retriever_routes_embeddings_and_forwards_the_rest(
    adapter_path: Path, rng: np.random.Generator
) -> None:
    """Adopting the bridge is one line, and nothing else changes."""
    bridge = Bridge.load(adapter_path)

    class FakeRetriever:
        def embed_query(self, text: str) -> np.ndarray:
            del text
            return l2_normalize(rng.standard_normal((1, DIM)).astype(np.float32))

        def unrelated_method(self) -> str:
            return "still reachable"

    wrapped = wrap_retriever(FakeRetriever(), bridge)
    assert wrapped.embed_query("how do I deploy?").shape == (1, DIM)
    assert wrapped.unrelated_method() == "still reachable"


@pytest.mark.unit
def test_wrap_retriever_rejects_an_object_without_an_embed_method(
    adapter_path: Path,
) -> None:
    """Failing at wrap time beats failing on the first query in production."""
    bridge = Bridge.load(adapter_path)
    with pytest.raises(AttributeError, match="embed_query"):
        wrap_retriever(object(), bridge)


@pytest.mark.unit
def test_describe_is_serialisable_for_reports(adapter_path: Path) -> None:
    """Diagnostics need the full picture; the hot path never calls this."""
    described = Bridge.load(adapter_path).describe()
    for key in ("adapter_type", "direction", "input_dim", "output_dim", "old_model"):
        assert key in described
