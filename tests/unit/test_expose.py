"""The scalar-only contract, and the two refusals that hold it up.

`expose` is the one command in this tool that is dual-use, and what keeps it
defensive is not its documentation — it is that no path through it produces a
vector or a piece of text. That is asserted here rather than reviewed, because a
reviewer checks it once and a test checks it on every commit.

The second guard is the reference model. Measuring exposure by sending text to a
hosted endpoint creates the exposure being measured, so a remote backend is
**refused** where every other command in this tool would warn.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rebasis.core import l2_normalize
from rebasis.embed import PrecomputedEmbedder
from rebasis.errors import ConfigError, InsufficientSamples
from rebasis.probe.exposure import (
    ExposureResult,
    measure_exposure,
    refuse_remote_reference,
)
from rebasis.store import MemoryStore
from rebasis.types import EncodingProfile

pytestmark = pytest.mark.unit

DIM = 32
N = 2600


@pytest.fixture
def world(rng: np.random.Generator) -> dict[str, Any]:
    """An index, and a local reference model that sees a rotation of its space."""
    centers = (rng.standard_normal((24, DIM)) * 3.0).astype(np.float32)
    assignment = rng.integers(0, 24, size=N)
    indexed = l2_normalize(
        centers[assignment] + rng.standard_normal((N, DIM)).astype(np.float32) * 1.0
    )
    rotation = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype(np.float32)
    reference_space = l2_normalize(
        indexed @ rotation.T + rng.standard_normal(indexed.shape).astype(np.float32) * 0.1
    )
    ids = [f"doc-{i}" for i in range(N)]
    texts = [f"document number {i}" for i in range(N)]
    return {
        "store": MemoryStore(ids, indexed, texts),
        "reference": PrecomputedEmbedder(
            EncodingProfile(model_id="local/reference", dim=DIM),
            dict(zip(texts, reference_space, strict=True)),
        ),
        "texts": texts,
    }


def _measure(world: dict[str, Any], **kwargs: Any) -> ExposureResult:
    return measure_exposure(
        world["store"],
        world["reference"],
        size=2_400,
        heldout=200,
        seed=5,
        # A small, fast configuration. What is under test here is the contract
        # and the refusals; whether the published hyperparameters recover a
        # rotation is `spikes/unpaired_align.py`'s question and is measured
        # there, on real corpora.
        seeds=2,
        config={"runs": 2, "clusters": 8, "qap_restarts": 4, "refine1_iters": 4},
        **kwargs,
    )


class TestItReturnsOnlyAScalar:
    """The hard line the whole command is designed around."""

    def test_the_result_holds_no_array(self, world: dict[str, Any]) -> None:
        """Not "does not print one" — does not hold one. A field that carried a
        vector could be serialised, logged or returned by a caller who never
        read the docstring."""
        import dataclasses

        result = _measure(world)

        # `dataclasses.fields`, not `vars`: the result is slotted and has no
        # `__dict__` — which is itself part of the design, since a slotted class
        # cannot grow a vector-carrying attribute at runtime.
        for field in dataclasses.fields(result):
            assert not isinstance(getattr(result, field.name), np.ndarray)

    def test_the_serialised_form_is_scalars_and_strings(self, world: dict[str, Any]) -> None:
        payload = _measure(world).to_dict()

        _assert_no_content(payload)

    def test_no_document_text_reaches_the_result(self, world: dict[str, Any]) -> None:
        """The command reads text — it has to, to embed the reference half — and
        none of it may come back out."""
        payload = _measure(world).to_dict()
        blob = repr(payload)

        assert "document number" not in blob
        assert "doc-" not in blob

    def test_it_reports_the_pool_the_number_is_relative_to(self, world: dict[str, Any]) -> None:
        """Identifying one document among 200 is a different claim from among
        200,000, so the pool is part of the number rather than a footnote."""
        result = _measure(world)

        assert result.pool == 200
        assert result.to_dict()["pool"] == 200

    def test_it_declares_itself_an_upper_bound(self, world: dict[str, Any]) -> None:
        """The reference half is drawn from the corpus being attacked, which is
        a better position than any adversary has."""
        result = _measure(world)

        assert result.to_dict()["upper_bound"] is True
        assert any("upper bound" in warning for warning in result.warnings)

    def test_every_attempt_is_reported(self, world: dict[str, Any]) -> None:
        """The spread across attempts is part of the answer, not a detail: the
        method is stochastic in three places, so one attempt's number is one
        draw. `docs/exposure.md` reports what that spread is on real corpora."""
        result = _measure(world)
        payload = result.to_dict()

        assert len(result.per_seed) == 2
        assert result.alignability == max(result.per_seed)
        assert payload["alignability_per_seed"] == [
            pytest.approx(round(v, 4)) for v in result.per_seed
        ]
        assert payload["alignability_spread"] == pytest.approx(
            round(max(result.per_seed) - min(result.per_seed), 4)
        )

    def test_it_offers_no_band(self, world: dict[str, Any]) -> None:
        """low/medium/high would be a classifier and the evidence does not
        support one. The absence is the contract, so it is tested."""
        payload = _measure(world).to_dict()

        assert "band" not in payload
        assert not any("band" in str(key).lower() for key in payload)


class TestTheSplitCannotLeak:
    """The unpaired condition is structural here, as it is in the spike."""

    def test_the_three_sets_are_disjoint(self) -> None:
        from rebasis.probe.exposure import _split
        from rebasis.probe.session import CorpusSample

        corpus = CorpusSample(
            ids=[f"d{i}" for i in range(101)],
            texts=[f"t{i}" for i in range(101)],
            old_vectors=np.zeros((101, 4), dtype=np.float32),
            query_positions=np.arange(11),
            fit_positions=np.arange(11, 101),
            n_total=101,
            strategy="random",
            seed=0,
        )
        held, first, second = _split(corpus, seed=3)

        assert set(held) & set(first) == set()
        assert set(held) & set(second) == set()
        assert set(first) & set(second) == set()
        assert len(set(held) | set(first) | set(second)) == 101

    def test_the_two_fit_halves_have_different_row_counts(self) -> None:
        """Equal shapes are the first thing that could be mistaken for a
        correspondence, so the split is uneven on purpose — including when the
        remainder is even, which `n // 2` would have split exactly."""
        from rebasis.probe.exposure import _split
        from rebasis.probe.session import CorpusSample

        for total in (100, 101):
            corpus = CorpusSample(
                ids=[f"d{i}" for i in range(total)],
                texts=[f"t{i}" for i in range(total)],
                old_vectors=np.zeros((total, 4), dtype=np.float32),
                query_positions=np.arange(10),
                fit_positions=np.arange(10, total),
                n_total=total,
                strategy="random",
                seed=0,
            )
            _, first, second = _split(corpus, seed=1)

            assert first.size != second.size, total


class TestTheReferenceMustBeLocal:
    def test_a_hosted_backend_is_refused(self) -> None:
        """Every other command in this tool warns. This one refuses, because
        creating exposure in order to measure it is the failure that would make
        the command worse than not having it."""

        class _Hosted:
            __module__ = "rebasis.embed.backends.openai_compat"

        with pytest.raises(ConfigError) as raised:
            refuse_remote_reference(_Hosted())  # type: ignore[arg-type]

        assert raised.value.code.startswith("RB-E")
        assert "local" in raised.value.hint.lower()

    def test_a_local_backend_passes(self, world: dict[str, Any]) -> None:
        refuse_remote_reference(world["reference"])

    def test_the_result_records_that_it_was_local(self, world: dict[str, Any]) -> None:
        assert _measure(world).to_dict()["reference_is_local"] is True


class TestTheMeasurement:
    def test_alignability_is_a_fraction(self, world: dict[str, Any]) -> None:
        result = _measure(world)

        assert 0.0 <= result.alignability <= 1.0
        assert result.mean_rank >= 1.0

    def test_an_index_too_small_to_cluster_is_refused_rather_than_scored(
        self, rng: np.random.Generator
    ) -> None:
        """Twenty clusters over a few hundred vectors are twenty arbitrary
        partitions of noise, and the number that comes out describes the sample
        rather than the index."""
        vectors = l2_normalize(rng.standard_normal((300, DIM)).astype(np.float32))
        ids = [f"d{i}" for i in range(300)]
        store = MemoryStore(ids, vectors, [f"t{i}" for i in range(300)])
        embedder = PrecomputedEmbedder(
            EncodingProfile(model_id="local/reference", dim=DIM),
            {f"t{i}": vectors[i] for i in range(300)},
        )

        with pytest.raises(InsufficientSamples):
            measure_exposure(store, embedder, size=300, heldout=50, seed=0)

    def test_it_is_deterministic_under_a_seed(self, world: dict[str, Any]) -> None:
        """A security figure that moves between runs is a figure nobody can act
        on, and every stochastic step here takes the seed."""
        assert _measure(world).alignability == _measure(world).alignability


def _assert_no_content(payload: Any) -> None:
    """Every leaf is a scalar, a string, a bool or None — recursively."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            assert isinstance(key, str)
            _assert_no_content(value)
    elif isinstance(payload, list):
        for value in payload:
            _assert_no_content(value)
    else:
        assert isinstance(payload, (str, int, float, bool, type(None))), payload
