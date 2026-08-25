"""How much of one space's geometry survives in the other — before any fit.

ADR 10 measured that retention is bounded by the source and rejected predicting
it from the model pair, because the evidence for that was a correlation over
fifteen runs. This is a different object. It is not a prediction and it does not
compete with one: it is a **bound**, from Maystre, Ortega Gonzalez, Park, Dolga,
Berariu, Zhao and Ciosek, *When Embedding Models Meet: Procrustes Bounds and
Applications* (arXiv:2510.13406), whose motivating scenario is this tool's:
the query model is upgraded and the document embeddings cannot be recomputed.

Their Corollary 1, in the paper's notation: if two models' pairwise inner
products agree to within ``δ`` in the mean-square sense,

    E[(xᵢᵀxⱼ − yᵢᵀyⱼ)²] ≤ δ²   ⟹   E[‖x̄ᵢ − yᵢ‖²] ≤ √(2D)·δ

where ``x̄`` is ``x`` under the best orthogonal map and ``D`` the dimension. The
bound is data-independent and independent of ``N``, and in the regime that
matters it is tighter than the previously known ones.

**What that buys.** ``δ`` is one Gram-matrix difference over a sample: no fit, no
candidate search, no held-out evaluation. It is available in the seconds before
`probe` starts fitting, and it says something the fit cannot contradict — the
alignment error *cannot* be worse than this, whatever adapter is chosen.

**What it does not buy, and this is the part to keep straight.** A small ``δ``
does not promise good retrieval. The bound runs one way: geometry preserved
implies alignment possible. The converse does not hold, and a low bound sitting
next to a bad ARR is not a contradiction — it means the alignment was available
and something else (a prefix, an encoding mismatch, a corpus the fit sample did
not cover) lost it. Reported as a bound, described as a bound, and never
substituted for the measurement.

The bound is on a squared distance between unit vectors, so it converts to
something a reader can hold: ``‖x̄ − y‖² = 2 − 2⟨x̄, y⟩`` gives an expected
cosine of at least ``1 − bound/2``. Above ``δ√(2D) = 2`` that floor drops below
zero and the bound says nothing at all — which is reported as *uninformative*
rather than as a floor of zero, because those are different statements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from rebasis.types import FloatArray

__all__ = ["GeometryBound", "geometry_bound"]

#: Rows used for the Gram matrix. The object is ``N × N``, so this is the one
#: number that decides whether the check is free or is its own problem: 2,000
#: rows is 16 MB at float32, and the estimate of a mean over four million
#: pairs is not improved by a fourth decimal place.
DEFAULT_SAMPLE = 2000

#: Above this the bound permits any orientation at all, since two unit vectors
#: are never more than 2 apart in squared distance.
VACUOUS = 2.0


@dataclass(frozen=True, slots=True)
class GeometryBound:
    """Maystre et al.'s bound, evaluated on one pair of spaces."""

    #: Root-mean-square difference between the two spaces' inner products.
    delta: float
    #: ``√(2D)·δ`` — the ceiling on ``E[‖x̄ᵢ − yᵢ‖²]`` under the best orthogonal map.
    bound: float
    #: Working dimension the bound was computed at.
    dim: int
    #: Rows compared.
    n_pairs: int

    @property
    def informative(self) -> bool:
        """Whether the bound says anything.

        Two unit vectors are at most 2 apart in squared distance, so a ceiling
        at or above 2 permits every possible outcome. Saying "the bound is
        2.4" invites a reader to compare it with a smaller one as if both were
        measurements of the same thing; saying it is uninformative does not.
        """
        return self.bound < VACUOUS

    @property
    def cosine_floor(self) -> float | None:
        """Lowest expected cosine the bound allows, or ``None`` when vacuous.

        ``‖x̄ − y‖² = 2 − 2⟨x̄, y⟩`` for unit vectors, so a ceiling on the left
        is a floor on the right. This is the form worth printing: a reader has
        an intuition for a cosine and none for a squared distance.
        """
        return 1.0 - self.bound / 2.0 if self.informative else None

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form, for the report and the audit record."""
        return {
            "geometry_delta": round(self.delta, 4),
            "alignment_bound": round(self.bound, 4),
            "cosine_floor": (None if self.cosine_floor is None else round(self.cosine_floor, 4)),
            "dim": self.dim,
            "n_pairs": self.n_pairs,
        }

    def explain(self) -> str:
        """One line for the report."""
        if not self.informative:
            return (
                f"Geometry preservation δ = {self.delta:.4f}. At {self.dim} dimensions the "
                f"bound this implies is {self.bound:.2f}, which is above 2 and therefore "
                f"says nothing: two unit vectors are never more than 2 apart, so the "
                f"inequality constrains nothing here. The measured ARR below is the answer."
            )
        return (
            f"Geometry preservation δ = {self.delta:.4f}. The two models' pairwise "
            f"similarities agree this closely, which bounds the error of the best "
            f"orthogonal alignment at {self.bound:.3f} — an expected cosine of at least "
            f"{self.cosine_floor:.3f} between an aligned vector and its target. "
            f"A bound, not a forecast: it says an alignment exists, not that "
            f"retrieval will use it."
        )


def geometry_bound(
    source: FloatArray,
    target: FloatArray,
    *,
    sample: int = DEFAULT_SAMPLE,
    seed: int = 0,
) -> GeometryBound:
    """Compare two spaces' inner products and bound the alignment error.

    Both sides must be ℓ2-normalised and row-aligned: row ``i`` of each is the
    same document under a different model. That alignment is the whole content
    of the measurement — comparing two Gram matrices of unrelated documents
    would produce a number with no meaning.

    When the dimensions differ, the bound is evaluated at the larger of the two.
    That follows the paper, which pads the smaller embedding with zeros so its
    original geometry is preserved, and it is also the conservative choice:
    ``√(2D)`` grows with ``D``, so the reported ceiling is the looser one.

    Args:
        source: Documents under the new model, ``(n, d_new)``.
        target: The same documents under the old model, ``(n, d_old)``.
        sample: Rows to compare. The Gram matrix is ``sample × sample``.
        seed: Which rows, when there are more than ``sample`` of them.

    Returns:
        The bound. ``delta`` is ``nan`` when there are too few rows for a
        pairwise statistic to mean anything.
    """
    n = min(source.shape[0], target.shape[0])
    minimum_rows = 2
    if n < minimum_rows:
        return GeometryBound(delta=float("nan"), bound=float("nan"), dim=0, n_pairs=n)

    if n > sample:
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(n, size=sample, replace=False))
        source, target = source[rows], target[rows]

    gram_source = np.asarray(source, dtype=np.float32) @ np.asarray(source, dtype=np.float32).T
    gram_target = np.asarray(target, dtype=np.float32) @ np.asarray(target, dtype=np.float32).T

    # The diagonal is 1 on both sides for normalised rows and contributes a
    # guaranteed zero to the mean, which would flatter δ by 1/n. Removed.
    difference = gram_source - gram_target
    off_diagonal = ~np.eye(difference.shape[0], dtype=bool)
    delta = float(np.sqrt(np.mean(np.square(difference[off_diagonal], dtype=np.float64))))

    dim = max(source.shape[1], target.shape[1])
    return GeometryBound(
        delta=delta,
        bound=float(np.sqrt(2.0 * dim) * delta),
        dim=dim,
        n_pairs=int(source.shape[0]),
    )
