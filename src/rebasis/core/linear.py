"""Affine adapters: full-rank ridge and its low-rank truncation.

``LinearAdapter`` is orthogonal Procrustes without the orthogonality constraint:
``g(x) = xW + b``. It supports a change of dimensionality natively, since ``W``
may be rectangular — so unlike Procrustes it needs no zero-padding trick when
``d_new ≠ d_old``.

``LowRankAffineAdapter`` truncates ``W`` to rank *r*. Memory falls from ``d²`` to
``r(d₁+d₂)``, and the truncation acts as regularisation.

**Measured caveat on rank (M0):** a rank-64 low-rank affine was projected to
reach 0.97–0.98. At d=384 the measurement gave **0.458** — below even plain
Procrustes. Rank 64 keeps 17% of the dimensions there, and at that ratio
truncation is information loss, not regularisation. The default is therefore
**proportional to the dimension** rather than a fixed 64.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from rebasis.core.base import BaseAdapter, check_fit_inputs
from rebasis.errors import FitFailed
from rebasis.types import AdapterKind, FloatArray, as_float32

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self

__all__ = ["LinearAdapter", "LowRankAffineAdapter", "default_rank"]

#: Fraction of the input dimension used as the default rank.
#:
#: A fixed rank of 64 was the original plan. M0 measured that at d=384 it keeps
#: only 17% of the dimensions and collapses quality (ARR 0.458 against 0.834 for
#: centred Procrustes). A proportional rank keeps the memory saving while staying
#: above the collapse threshold.
DEFAULT_RANK_FRACTION = 0.25
MIN_RANK = 32


def default_rank(dim: int) -> int:
    """Default rank for a given dimension (see :data:`DEFAULT_RANK_FRACTION`)."""
    return min(dim, max(MIN_RANK, round(dim * DEFAULT_RANK_FRACTION)))


class LinearAdapter(BaseAdapter):
    """``g(x) = xW + b``, solved by ridge regression.

    ``alpha`` is a small ridge term scaled by the Gram matrix trace, so it is
    dimensionless and behaves the same at any embedding scale. It keeps the
    solution finite when ``XᵀX`` is singular — fewer pairs than dimensions, or
    collinear features.
    """

    kind: ClassVar[AdapterKind] = "linear"

    def __init__(
        self, weight: FloatArray, bias: FloatArray, *, input_dim: int, output_dim: int
    ) -> None:
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        self.weight = weight
        self.bias = bias

    @classmethod
    def fit(cls, src: FloatArray, dst: FloatArray, *, alpha: float = 1e-3) -> Self:
        """Solve the ridge system."""
        check_fit_inputs(src, dst)
        x = as_float32(src)
        y = as_float32(dst)

        # Centering separates the translation from the regression: W carries only
        # the linear map, b the offset, and the ridge penalty never touches b.
        x_mean = x.mean(axis=0)
        y_mean = y.mean(axis=0)

        # Solved in float64: the normal equation squares the condition number,
        # and in float32 that produces measurable error. The result returns to
        # float32 at the boundary, so the float32 contract holds outside this
        # function.
        centred_x = (x - x_mean).astype(np.float64)
        gram = centred_x.T @ centred_x
        if alpha > 0:
            gram[np.diag_indices_from(gram)] += alpha * np.trace(gram) / gram.shape[0]
        try:
            solution = np.linalg.solve(gram, centred_x.T @ (y - y_mean).astype(np.float64))
        except np.linalg.LinAlgError as exc:
            raise FitFailed(
                "The ridge system is singular and could not be solved.",
                hint=(
                    "Increase --pairs, or raise alpha. This usually means the "
                    "vectors are collinear or there are fewer pairs than dimensions."
                ),
                context={"count": int(src.shape[0]), "dim": int(x.shape[1])},
                cause=exc,
            ) from exc

        weight = as_float32(solution)
        bias = as_float32(y_mean - x_mean @ weight)
        return cls(weight, bias, input_dim=x.shape[1], output_dim=y.shape[1])

    def apply(self, x: FloatArray) -> FloatArray:
        """One matrix multiply plus a bias. Hot path."""
        return as_float32(as_float32(x) @ self.weight + self.bias)

    def state_dict(self) -> dict[str, FloatArray]:
        """Weight matrix and bias."""
        return {"weight": self.weight, "bias": self.bias}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, FloatArray], config: Mapping[str, Any]) -> Self:
        """Reconstruct from stored weights."""
        return cls(
            as_float32(state["weight"]),
            as_float32(state["bias"]),
            input_dim=int(config["input_dim"]),
            output_dim=int(config["output_dim"]),
        )


class LowRankAffineAdapter(BaseAdapter):
    """``g(x) = x(UV) + b`` at rank *r*.

    The truncated SVD of the full-rank ridge solution. The singular values are
    folded into ``U`` so that applying costs two matrix multiplies rather than
    three.
    """

    kind: ClassVar[AdapterKind] = "low_rank_affine"

    def __init__(  # noqa: PLR0913 - two factors, a bias, two dimensions and a rank
        self,
        u: FloatArray,
        v: FloatArray,
        bias: FloatArray,
        *,
        input_dim: int,
        output_dim: int,
        rank: int,
    ) -> None:
        super().__init__(input_dim=input_dim, output_dim=output_dim, config={"rank": rank})
        self.u = u
        self.v = v
        self.bias = bias
        self.rank = rank

    @classmethod
    def fit(
        cls,
        src: FloatArray,
        dst: FloatArray,
        *,
        rank: int | None = None,
        alpha: float = 1e-3,
    ) -> Self:
        """Fit the full-rank solution, then truncate its SVD.

        ``rank`` defaults to :func:`default_rank` — proportional to the
        dimension, for the reason documented at the top of this module.
        """
        full = LinearAdapter.fit(src, dst, alpha=alpha)
        chosen = default_rank(full.input_dim) if rank is None else rank
        effective = min(chosen, *full.weight.shape)

        u_full, singular, vt_full = np.linalg.svd(
            full.weight.astype(np.float64), full_matrices=False
        )
        u = as_float32(u_full[:, :effective] * singular[:effective])
        v = as_float32(vt_full[:effective])
        return cls(
            u,
            v,
            full.bias,
            input_dim=full.input_dim,
            output_dim=full.output_dim,
            rank=effective,
        )

    def apply(self, x: FloatArray) -> FloatArray:
        """Two thin matrix multiplies plus a bias. Hot path."""
        return as_float32((as_float32(x) @ self.u) @ self.v + self.bias)

    def state_dict(self) -> dict[str, FloatArray]:
        """The two low-rank factors and the bias."""
        return {"u": self.u, "v": self.v, "bias": self.bias}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, FloatArray], config: Mapping[str, Any]) -> Self:
        """Reconstruct from stored weights."""
        return cls(
            as_float32(state["u"]),
            as_float32(state["v"]),
            as_float32(state["bias"]),
            input_dim=int(config["input_dim"]),
            output_dim=int(config["output_dim"]),
            rank=int(config["rank"]),
        )
