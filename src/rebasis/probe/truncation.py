"""What a cheaper representation of the *same* space costs.

The most common index transformation in the field is not a model change. It is a
cut in dimension and precision — truncate a Matryoshka-trained model to 512, or
store int8 instead of float32 — and the question it raises is the one ``probe``
already answers: **what do I lose, on my corpus rather than on a benchmark
average?**

Three things separate this from every other measurement in this package, and the
third is the reason it is worth having at all.

**No adapter, and no squeeze.** ADR 10 says retention is bounded by how much
structure the old space holds, and that gain and retention anti-correlate at
-0.958 — bridging fails precisely where it is needed. Nothing of the sort
applies here. The model does not change and the space does not change; only a
cheaper representation of the same space is used. What is left is arithmetic and
a measurement.

**A whole grid costs what one probe costs.** The model runs once. Truncating and
quantizing the vectors it produced is free, so sixteen cells are sixteen
searches over arrays that are already in memory — not sixteen embedding passes.

**The quantization axis is a simulation, and says so.** rebasis produces float32;
what a store does with it is the store's business, and the backends do not all
do the same thing — `sqlite-vec`'s `int8`, pgvector's `halfvec` and Qdrant's
`datatype` are three different narrowings. So a cell on this axis measures *what
that arithmetic costs*, which is a lower bound on what a particular store's
codec costs, and the report labels it rather than implying a round trip nobody
performed.

**Writing the result back is out of scope, deliberately.** Going from
``vector(1024)`` to ``vector(256)`` means recreating the column, which is DDL.
`migrate` changes vectors, not schemas — that is the line that keeps this tool
from becoming a vector database. The measurement says what a change is worth;
performing it stays the user's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from rebasis.core.base import l2_normalize
from rebasis.probe.metrics import bootstrap_ratio_ci, ndcg_per_query, top_k_search
from rebasis.types import as_float32

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rebasis.probe.groundtruth import GroundTruth
    from rebasis.types import FloatArray

__all__ = [
    "PRECISIONS",
    "RESCORE_SHARE_LIMIT",
    "GridCell",
    "TruncationGrid",
    "measure_grid",
    "quantize",
]

#: Bytes one component occupies, per precision. The storage axis of the grid.
#:
#: ``binary`` is one bit, so an eighth of a byte. Written as a float rather than
#: rounded up, because the point of the row is that it is 32 times cheaper than
#: float32 and rounding to 1 would say 4.
PRECISIONS: dict[str, float] = {
    "float32": 4.0,
    "float16": 2.0,
    "int8": 1.0,
    "binary": 0.125,
}

#: Candidate depth for the rescored variant of every cell.
#:
#: The pattern this measures — generate candidates with a cheap representation,
#: reorder them with the full-precision vectors — is the cascade on a different
#: axis, and it is the reason the binary row is worth measuring at all. The
#: depth matches `rebasis.probe.runner.CASCADE_N` so the two arrangements are
#: reported at the same one.
RESCORE_AT = 200

#: Share of the corpus above which a rescore depth stops meaning anything.
#:
#: Rescoring the top N recovers everything the cheap representation put anywhere
#: in the top N, so on a corpus of 2,000 a depth of 200 is a tenth of it and the
#: rescored column reads 1.000 for every cell — true, and a statement about the
#: sample rather than about the arrangement. Measured the first time this ran on
#: a 2,500-document fixture and every rescored cell came back exactly 1.000.
RESCORE_SHARE_LIMIT = 0.05

#: The largest int8 magnitude a symmetric scalar quantizer maps onto.
#:
#: 127 rather than 128: the range has to be symmetric or the quantizer moves the
#: mean, and a shifted mean is a rotation of the space rather than a coarsening
#: of it.
_INT8_MAX = 127.0


def quantize(vectors: FloatArray, precision: str) -> FloatArray:
    """Round ``vectors`` to a narrower representation and back to float32.

    Returned as float32 because everything downstream — the search, the metric,
    the interval — is defined on float32, and because that is what the *store*
    would hand back after decoding its own codes. What is being measured is the
    information the narrowing destroyed, not the dtype it was held in.

    Four narrowings, and the last two are not the same kind of thing as the
    first two:

    ``float32`` is the identity, and is in the table so the grid has a
    reference cell whose retention is 1.000 by construction rather than by
    accident.

    ``float16`` is a dtype round trip. Half precision carries about three
    decimal digits, so a unit-norm component near 1e-4 loses most of itself and
    one near 0.5 loses almost nothing.

    ``int8`` is a **symmetric per-vector scalar quantizer**: scale by the
    largest magnitude in the vector, round to the nearest of 255 levels, scale
    back. Per vector rather than per corpus because that is what the backends
    that do this do, and because a per-corpus scale would let one outlier
    document coarsen every other one.

    ``binary`` keeps the sign of each component and nothing else. Scored as
    +-1, which orders documents identically to Hamming distance over the packed
    bits — the inner product of two sign vectors is ``d - 2 x hamming`` — so the
    ranking is the one a binary index would produce while the arithmetic stays
    the one everything else here uses.
    """
    if precision == "float32":
        return as_float32(vectors)
    if precision == "float16":
        return as_float32(vectors.astype(np.float16))
    if precision == "int8":
        scale = np.abs(vectors).max(axis=1, keepdims=True)
        # A zero vector has no scale and no information; dividing by zero would
        # turn it into NaN and take every similarity with it.
        scale = np.where(scale > 0, scale, 1.0)
        codes = np.rint(vectors / scale * _INT8_MAX)
        return as_float32(codes / _INT8_MAX * scale)
    if precision == "binary":
        # `np.sign` returns 0 for a component that is exactly 0, which is a
        # third state a one-bit code does not have. Zero is folded upward, the
        # way a `> 0` threshold does.
        return as_float32(np.where(vectors >= 0, 1.0, -1.0))
    message = f"unknown precision {precision!r}; expected one of {', '.join(PRECISIONS)}"
    raise ValueError(message)


def truncate(vectors: FloatArray, dim: int) -> FloatArray:
    """Keep the first ``dim`` components and renormalise.

    Renormalising is not optional and not cosmetic. Cutting a unit vector's tail
    leaves a vector shorter than one, by an amount that differs per document —
    so an un-renormalised truncation ranks documents partly by how much of their
    norm survived, which is a property of the document rather than of the query.
    """
    if dim >= vectors.shape[1]:
        return as_float32(vectors)
    return l2_normalize(as_float32(vectors[:, :dim]), copy=True)


@dataclass(slots=True)
class GridCell:
    """One (dimension, precision) pair, measured."""

    dim: int
    precision: str
    #: Fraction of the full-precision, full-dimension nDCG@k this cell retains.
    retained: float
    #: The same, when the cell produces candidates that the full-precision
    #: vectors then reorder. The cascade's shape on a different axis.
    retained_rescored: float
    #: Paired bootstrap interval on ``retained``.
    interval: tuple[float, float]
    #: Fraction of the reference's storage this cell occupies.
    storage: float
    #: nDCG@k itself, for a reader who wants the absolute number.
    ndcg: float

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "dim": self.dim,
            "precision": self.precision,
            "retained": round(self.retained, 4),
            "retained_rescored": round(self.retained_rescored, 4),
            "interval": [round(v, 4) for v in self.interval],
            "storage": round(self.storage, 5),
            "ndcg": round(self.ndcg, 4),
        }


#: What the quantization axis is and is not.
SIMULATION_NOTE = (
    "The precision columns are simulated: rebasis produces float32 and each "
    "store narrows it in its own way, so a cell measures what the arithmetic "
    "costs rather than what your backend's codec costs. The dimension rows are "
    "not simulated — truncating a vector is the whole operation."
)


@dataclass(slots=True)
class TruncationGrid:
    """Every cell, plus what the cheapest acceptable one is."""

    cells: list[GridCell]
    full_dim: int
    k: int
    n_queries: int
    reference_ndcg: float
    rescore_at: int = RESCORE_AT
    floor: float | None = None
    simulation_note: str = SIMULATION_NOTE
    warnings: list[str] = field(default_factory=list)

    def cheapest_above(self, floor: float, *, rescored: bool = False) -> GridCell | None:
        """The cheapest cell that clears a quality floor.

        A Pareto choice rather than a break-even, and that is the honest shape:
        quality and cost are two axes and which one matters more is the user's
        call, not the tool's. What the tool can do is take the floor as given
        and name the cheapest cell above it.

        Ties on storage are broken by retention, so a reader who set a floor
        that several cells clear at the same price gets the best of them.
        """
        retention: Any = (lambda c: c.retained_rescored) if rescored else (lambda c: c.retained)
        clearing = [cell for cell in self.cells if retention(cell) >= floor]
        if not clearing:
            return None
        return min(clearing, key=lambda c: (c.storage, -retention(c)))

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form, for the report and for a script."""
        chosen = None if self.floor is None else self.cheapest_above(self.floor)
        return {
            "cells": [cell.to_dict() for cell in self.cells],
            "full_dim": self.full_dim,
            "k": self.k,
            "n_queries": self.n_queries,
            "reference_ndcg": round(self.reference_ndcg, 4),
            "rescore_at": self.rescore_at,
            "floor": self.floor,
            "cheapest_above_floor": None if chosen is None else chosen.to_dict(),
            "simulation_note": self.simulation_note,
            "warnings": list(self.warnings),
        }


def measure_grid(  # noqa: PLR0913 - one argument per axis of the grid
    *,
    doc_vectors: FloatArray,
    query_vectors: FloatArray,
    ground_truth: GroundTruth,
    dims: Sequence[int],
    precisions: Sequence[str],
    k: int = 10,
    rescore_at: int = RESCORE_AT,
    floor: float | None = None,
) -> TruncationGrid:
    """Measure every cell against the index's own full-precision state.

    ``doc_vectors`` and ``query_vectors`` are the index's own, at full width and
    full precision. Both are cut and rounded together in every cell: cutting only
    the documents leaves the inner product comparing coordinates that no longer
    correspond, which is not a cheaper representation but a wrong one.

    The reference is the top-left cell — full width, float32 — measured through
    the same code path as every other cell rather than assumed to be 1.000. If
    the two ever disagree, the grid is measuring its own arithmetic and the
    difference says so.
    """
    full_dim = int(doc_vectors.shape[1])
    reference_indices = _search(doc_vectors, query_vectors, ground_truth, depth=max(k, rescore_at))
    reference_per_query = ndcg_per_query(reference_indices[:, :k], ground_truth.relevant_sparse, k)
    reference = float(reference_per_query.mean()) if reference_per_query.size else 0.0

    warnings: list[str] = []
    too_wide = sorted({dim for dim in dims if dim > full_dim})
    if too_wide:
        warnings.append(
            f"{', '.join(str(d) for d in too_wide)} exceed this index's {full_dim} "
            f"dimensions and were skipped. Truncation cannot widen a vector, and "
            f"zero-padding one is a different index rather than a cheaper one."
        )

    cells: list[GridCell] = []
    for dim in sorted({min(dim, full_dim) for dim in dims}, reverse=True):
        documents = truncate(doc_vectors, dim)
        queries = truncate(query_vectors, dim)
        cells.extend(
            _cell(
                documents=documents,
                queries=queries,
                full_documents=doc_vectors,
                full_queries=query_vectors,
                ground_truth=ground_truth,
                dim=dim,
                precision=precision,
                k=k,
                rescore_at=rescore_at,
                full_dim=full_dim,
                reference=reference,
                reference_per_query=reference_per_query,
            )
            for precision in precisions
        )

    n_documents = int(doc_vectors.shape[0])
    if rescore_at > n_documents * RESCORE_SHARE_LIMIT:
        warnings.append(
            f"The rescore depth of {rescore_at} is {rescore_at / n_documents:.0%} of the "
            f"{n_documents:,} documents measured, so the rescored column says little: "
            f"reordering a tenth of a corpus recovers almost anything. On a real "
            f"index it is a small fraction, and the number means what it says there. "
            f"Increase --sample to make this column comparable."
        )

    if floor is not None and not any(cell.retained >= floor for cell in cells):
        warnings.append(
            f"No cell in this grid retains {floor:.0%} of what the index does today. "
            f"The best is {max(cell.retained for cell in cells):.3f}; either the floor "
            f"is above what a cheaper representation of this corpus can deliver, or "
            f"the grid needs a row between the ones it was given."
        )

    return TruncationGrid(
        cells=cells,
        full_dim=full_dim,
        k=k,
        n_queries=int(reference_per_query.size),
        reference_ndcg=reference,
        rescore_at=rescore_at,
        floor=floor,
        warnings=warnings,
    )


def _cell(  # noqa: PLR0913 - one argument per input the cell needs
    *,
    documents: FloatArray,
    queries: FloatArray,
    full_documents: FloatArray,
    full_queries: FloatArray,
    ground_truth: GroundTruth,
    dim: int,
    precision: str,
    k: int,
    rescore_at: int,
    full_dim: int,
    reference: float,
    reference_per_query: FloatArray,
) -> GridCell:
    """One cell, and the same cell fed through a full-precision rescore."""
    coded_documents = quantize(documents, precision)
    coded_queries = quantize(queries, precision)
    indices = _search(coded_documents, coded_queries, ground_truth, depth=max(k, rescore_at))

    per_query = ndcg_per_query(indices[:, :k], ground_truth.relevant_sparse, k)
    retained = _ratio(float(per_query.mean()) if per_query.size else 0.0, reference)

    rescored = _rescore(indices[:, :rescore_at], full_documents, full_queries)
    rescored_ndcg = ndcg_per_query(rescored[:, :k], ground_truth.relevant_sparse, k)
    retained_rescored = _ratio(
        float(rescored_ndcg.mean()) if rescored_ndcg.size else 0.0, reference
    )

    return GridCell(
        dim=dim,
        precision=precision,
        retained=retained,
        retained_rescored=retained_rescored,
        interval=bootstrap_ratio_ci(per_query, reference_per_query),
        storage=(dim / full_dim) * (PRECISIONS[precision] / PRECISIONS["float32"]),
        ndcg=float(per_query.mean()) if per_query.size else 0.0,
    )


def _ratio(value: float, reference: float) -> float:
    """A retention, or ``nan`` where the reference retrieved nothing.

    ``nan`` rather than 0 or 1: a corpus whose full-precision index answers no
    query has no retention to lose, and either constant would read as a
    measurement of the cheaper representation.
    """
    return value / reference if reference > 0 else float("nan")


def _search(
    documents: FloatArray, queries: FloatArray, ground_truth: GroundTruth, *, depth: int
) -> np.ndarray:
    """Top-``depth`` neighbours, with the ground truth's own self-mask applied."""
    indices, _ = top_k_search(
        queries, documents, k=min(depth, documents.shape[0]), self_mask=ground_truth.self_mask
    )
    return indices


def _rescore(
    candidates: np.ndarray, full_documents: FloatArray, full_queries: FloatArray
) -> np.ndarray:
    """Reorder each candidate set by the full-precision vectors' own similarity.

    The pattern that traces back to the Binary Passage Retriever: generate
    candidates with the cheap representation, rank them with the expensive one.
    It is the cascade's shape on a different axis, and unlike the cascade it
    costs no embedding at all — the full-precision vectors are the ones the
    index already holds.

    Per query rather than as one matrix, because the candidate sets differ
    between queries and there is no shared document axis to multiply against.
    """
    out = np.empty_like(candidates)
    for row in range(candidates.shape[0]):
        chosen = candidates[row]
        scores = full_documents[chosen] @ full_queries[row]
        out[row] = chosen[np.argsort(-scores)]
    return out
