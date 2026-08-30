"""What a cheaper representation of the same space costs.

No model change, no adapter, and none of the squeeze ADR 10 measures — the space
is the same one, held more cheaply. So the properties worth testing are the ones
that would make the grid *wrong* rather than merely pessimistic:

**Both sides are cut.** Truncating documents while leaving queries at full width
compares coordinates that no longer correspond. The result is not a cheaper
index, it is a broken one, and it would look like a catastrophic quality loss.

**Truncation renormalises.** Cutting a unit vector's tail leaves it shorter by
an amount that differs per document, so an un-renormalised cut ranks partly by
how much of each document's norm survived.

**The reference cell is measured, not assumed.** Full width at float32 goes
through the same code path as every other cell.
"""

from __future__ import annotations

import numpy as np
import pytest

from rebasis.core import l2_normalize
from rebasis.probe.groundtruth import build_tier0
from rebasis.probe.truncation import (
    PRECISIONS,
    measure_grid,
    quantize,
    truncate,
)

pytestmark = pytest.mark.unit

DIM = 64
N = 600
QUERIES = 60


@pytest.fixture
def vectors(rng: np.random.Generator):  # type: ignore[no-untyped-def]
    centers = (rng.standard_normal((15, DIM)) * 3.0).astype(np.float32)
    assignment = rng.integers(0, 15, size=N)
    return l2_normalize(
        centers[assignment] + rng.standard_normal((N, DIM)).astype(np.float32) * 1.2
    )


@pytest.fixture
def world(vectors):  # type: ignore[no-untyped-def]
    positions = np.arange(QUERIES)
    queries = vectors[positions]
    truth = build_tier0(vectors, queries, positions, k=10)
    return {"documents": vectors, "queries": queries, "truth": truth}


def _grid(world, **kwargs):  # type: ignore[no-untyped-def]
    return measure_grid(
        doc_vectors=world["documents"],
        query_vectors=world["queries"],
        ground_truth=world["truth"],
        dims=kwargs.pop("dims", [DIM, DIM // 2, DIM // 4]),
        precisions=kwargs.pop("precisions", ["float32", "float16", "int8", "binary"]),
        k=10,
        rescore_at=kwargs.pop("rescore_at", 50),
        **kwargs,
    )


class TestQuantize:
    """Each narrowing does what its name says, on arrays whose answer is known."""

    def test_float32_is_the_identity(self, vectors) -> None:  # type: ignore[no-untyped-def]
        """In the table so the grid has a reference cell that is 1.000 by
        construction rather than by accident."""
        assert np.array_equal(quantize(vectors, "float32"), vectors)

    def test_float16_rounds_and_stays_float32(self, vectors) -> None:  # type: ignore[no-untyped-def]
        """Returned as float32 because that is what a store hands back after
        decoding its own codes — what is measured is the information lost, not
        the dtype it was held in."""
        rounded = quantize(vectors, "float16")

        assert rounded.dtype == np.float32
        assert not np.array_equal(rounded, vectors)
        assert np.abs(rounded - vectors).max() < 1e-2

    def test_int8_is_symmetric_and_per_vector(self, vectors) -> None:  # type: ignore[no-untyped-def]
        """A per-corpus scale would let one outlier document coarsen every other
        one, which is not what a backend that quantizes actually does."""
        coded = quantize(vectors, "int8")

        assert np.abs(coded).max() <= np.abs(vectors).max() + 1e-6
        assert np.abs(coded - vectors).max() > 0.0
        # Each vector's own largest component survives exactly: it is the scale.
        largest = np.abs(vectors).argmax(axis=1)
        rows = np.arange(vectors.shape[0])
        assert np.allclose(coded[rows, largest], vectors[rows, largest], atol=1e-6)

    def test_binary_keeps_only_the_sign(self, vectors) -> None:  # type: ignore[no-untyped-def]
        coded = quantize(vectors, "binary")

        assert set(np.unique(coded).tolist()) <= {-1.0, 1.0}
        assert np.array_equal(coded > 0, vectors >= 0)

    def test_a_zero_vector_survives_int8(self) -> None:
        """Its scale is zero, and dividing by it would return NaN and take every
        similarity computed against it along."""
        coded = quantize(np.zeros((2, 8), dtype=np.float32), "int8")

        assert np.isfinite(coded).all()

    def test_an_unknown_precision_is_refused(self, vectors) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="unknown precision"):
            quantize(vectors, "int4")


class TestTruncate:
    def test_it_renormalises(self, vectors) -> None:  # type: ignore[no-untyped-def]
        """Otherwise a cut ranks documents by how much of their norm survived,
        which is a property of the document rather than of the query."""
        cut = truncate(vectors, DIM // 4)

        assert cut.shape == (N, DIM // 4)
        assert np.allclose(np.linalg.norm(cut, axis=1), 1.0, atol=1e-5)

    def test_it_cannot_widen(self, vectors) -> None:  # type: ignore[no-untyped-def]
        assert truncate(vectors, DIM * 2).shape == vectors.shape


class TestTheGrid:
    def test_the_reference_cell_retains_everything(self, world) -> None:  # type: ignore[no-untyped-def]
        """Measured through the same path as every other cell rather than
        assumed. If it ever came back below 1.0 the grid would be measuring its
        own arithmetic, and the number would say so."""
        grid = _grid(world)
        reference = next(
            cell for cell in grid.cells if cell.dim == grid.full_dim and cell.precision == "float32"
        )

        assert reference.retained == pytest.approx(1.0)
        assert reference.storage == pytest.approx(1.0)

    def test_storage_is_dimension_times_precision(self, world) -> None:  # type: ignore[no-untyped-def]
        grid = _grid(world)
        cell = next(c for c in grid.cells if c.dim == DIM // 4 and c.precision == "int8")

        assert cell.storage == pytest.approx(0.25 * PRECISIONS["int8"] / 4.0)

    def test_a_cheaper_cell_never_retains_more_than_it_costs_to_believe(
        self,
        world,  # type: ignore[no-untyped-def]
    ) -> None:
        """Retention falls, or at worst holds, as the representation gets
        cheaper along a row. Not a law of nature — a coarser code can win a tie
        by luck — so the assertion is on the trend across the whole row rather
        than on every adjacent pair."""
        grid = _grid(world)
        row = {c.precision: c.retained for c in grid.cells if c.dim == grid.full_dim}

        assert row["float32"] >= row["binary"]
        assert row["float32"] >= row["int8"] - 1e-6

    def test_rescoring_recovers_what_the_cheap_code_only_had_to_shortlist(
        self,
        world,  # type: ignore[no-untyped-def]
    ) -> None:
        """The cascade's shape on a different axis. A binary code has to put the
        answer somewhere in the top N, not at rank 1 — a weaker requirement, and
        the full-precision vectors that do the reordering are the ones the index
        already holds, so it costs no embedding at all."""
        grid = _grid(world)
        binary = next(c for c in grid.cells if c.dim == grid.full_dim and c.precision == "binary")

        assert binary.retained_rescored >= binary.retained

    def test_every_cell_carries_an_interval(self, world) -> None:  # type: ignore[no-untyped-def]
        grid = _grid(world)

        for cell in grid.cells:
            low, high = cell.interval
            assert low <= high

    def test_a_dimension_wider_than_the_index_is_skipped_and_named(
        self,
        world,  # type: ignore[no-untyped-def]
    ) -> None:
        """Truncation cannot widen a vector, and zero-padding one is a different
        index rather than a cheaper one."""
        grid = _grid(world, dims=[DIM * 4, DIM, DIM // 2])

        assert max(cell.dim for cell in grid.cells) == DIM
        assert any(str(DIM * 4) in warning for warning in grid.warnings)

    def test_it_serialises(self, world) -> None:  # type: ignore[no-untyped-def]
        payload = _grid(world, floor=0.9).to_dict()

        assert payload["full_dim"] == DIM
        assert payload["floor"] == 0.9
        assert len(payload["cells"]) == 3 * 4
        assert "simulated" in payload["simulation_note"]


class TestTheFloor:
    def test_it_names_the_cheapest_cell_that_clears_it(self, world) -> None:  # type: ignore[no-untyped-def]
        """A Pareto choice rather than a break-even: quality and cost are two
        axes and which matters more is not the tool's decision."""
        grid = _grid(world, floor=0.5)
        chosen = grid.cheapest_above(0.5)

        assert chosen is not None
        assert chosen.retained >= 0.5
        cheaper = [c for c in grid.cells if c.storage < chosen.storage]
        assert all(c.retained < 0.5 for c in cheaper)

    def test_an_unreachable_floor_is_said_rather_than_rounded_to(
        self,
        world,  # type: ignore[no-untyped-def]
    ) -> None:
        grid = _grid(world, floor=1.5)

        assert grid.cheapest_above(1.5) is None
        assert any("No cell" in warning for warning in grid.warnings)

    def test_the_rescored_frontier_is_available_separately(self, world) -> None:  # type: ignore[no-untyped-def]
        """The two arrangements have different frontiers, and a reader who
        intends to rescore should not be shown the one that does not."""
        grid = _grid(world, floor=0.9)
        plain = grid.cheapest_above(0.9)
        rescored = grid.cheapest_above(0.9, rescored=True)

        assert plain is not None
        assert rescored is not None
        assert rescored.storage <= plain.storage
