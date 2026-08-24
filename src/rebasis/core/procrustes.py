"""Orthogonal Procrustes adapters.

Two variants, and the difference between them is the single largest measured
effect in M0.

**Plain Procrustes** solves ``min ‖AR − B‖`` subject to ``RᵀR = I`` — a rotation
**through the origin**. Schönemann's closed form, one line of scipy; no
iteration, no seed, determinism for free.

**Centred Procrustes** removes the mean from both sides first:
``g(x) = (x − μ_src)R + μ_dst``. Embedding spaces are not isotropic — they carry
a large shared mean component — so when two spaces' means point in different
directions, a rotation pinned to the origin cannot close the gap however it
turns. The cross-lingual alignment literature (MUSE, vecmap) treats centering as
a standard step; our own preprocessing rule covers only ℓ2 normalisation before
the fit and renormalisation after apply.

**Measured (M0, 4 corpora × 3 model pairs):** centering raises ARR by +0.166 on
average at T0 and **+0.260 at T1**, with a best case of +0.75. It hurt in 1 of
12 cases at T0 and 2 of 12 at T1, always by ≤ 0.018 — within measurement noise.
Details in section 4 of ``docs/m0-findings.md``.

Because it is nearly free, never meaningfully harmful and often decisive,
**centring is the default** and plain Procrustes is retained for comparison.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import scipy.linalg

from rebasis.core.base import BaseAdapter, check_fit_inputs, pad_or_truncate
from rebasis.errors import FitFailed
from rebasis.types import AdapterKind, FloatArray, as_float32

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self

__all__ = ["CenteredProcrustesAdapter", "ProcrustesAdapter"]


class ProcrustesAdapter(BaseAdapter):
    """``g(x) = xR`` with ``RᵀR = I``. Closed-form SVD.

    Orthogonality is free regularisation: the transform preserves angles and
    norms, which makes overfitting on a small pair set essentially impossible.
    That property is why Procrustes stays useful at a few thousand pairs where a
    full-rank affine map would start memorising.
    """

    kind: ClassVar[AdapterKind] = "procrustes"

    def __init__(self, rotation: FloatArray, *, input_dim: int, output_dim: int) -> None:
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        self.rotation = rotation
        self._work_dim = rotation.shape[0]
        self._passthrough = input_dim == self._work_dim == output_dim

    @classmethod
    def fit(cls, src: FloatArray, dst: FloatArray) -> Self:
        """Solve the orthogonal Procrustes problem."""
        check_fit_inputs(src, dst)
        input_dim, output_dim = src.shape[1], dst.shape[1]
        work = max(input_dim, output_dim)
        a = pad_or_truncate(as_float32(src), work)
        b = pad_or_truncate(as_float32(dst), work)
        try:
            rotation, _scale = scipy.linalg.orthogonal_procrustes(a, b)
        except (ValueError, scipy.linalg.LinAlgError) as exc:
            raise FitFailed(
                "The orthogonal Procrustes solution did not converge.",
                hint="Usually degenerate input — all-identical or all-zero vectors.",
                context={"count": int(src.shape[0]), "dim": input_dim},
                cause=exc,
            ) from exc
        return cls(as_float32(rotation), input_dim=input_dim, output_dim=output_dim)

    def apply(self, x: FloatArray) -> FloatArray:
        """Rotate into the target space. Hot path — no validation."""
        if self._passthrough:
            return as_float32(as_float32(x) @ self.rotation)
        z = pad_or_truncate(as_float32(x), self._work_dim) @ self.rotation
        return as_float32(np.ascontiguousarray(z[..., : self.output_dim]))

    def state_dict(self) -> dict[str, FloatArray]:
        """The rotation matrix."""
        return {"rotation": self.rotation}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, FloatArray], config: Mapping[str, Any]) -> Self:
        """Reconstruct from stored weights."""
        return cls(
            as_float32(state["rotation"]),
            input_dim=int(config["input_dim"]),
            output_dim=int(config["output_dim"]),
        )


class CenteredProcrustesAdapter(BaseAdapter):
    """``g(x) = (x − μ_src)R + μ_dst`` with ``RᵀR = I``. **The default**.

    Still a rigid rotation in the centred space, so Procrustes' resistance to
    overfitting survives intact. The extra cost is two vectors — 2d parameters,
    3 KB at d=384 — against a measured average ARR gain of +0.26.

    ``with_scale`` additionally fits a single global scale factor. It is off by
    default: on ℓ2-normalised vectors the scale is close to 1 and the extra
    parameter buys nothing, but it matters if a caller ever fits on unnormalised
    input.
    """

    kind: ClassVar[AdapterKind] = "procrustes_centered"

    def __init__(  # noqa: PLR0913 - rotation, two means, two dimensions and a scale
        self,
        rotation: FloatArray,
        mean_src: FloatArray,
        mean_dst: FloatArray,
        *,
        input_dim: int,
        output_dim: int,
        scale: float = 1.0,
    ) -> None:
        super().__init__(input_dim=input_dim, output_dim=output_dim, config={"scale": scale})
        self.rotation = rotation
        self.mean_src = mean_src
        self.mean_dst = mean_dst
        self.scale = scale
        self._work_dim = rotation.shape[0]
        self._passthrough = input_dim == self._work_dim == output_dim and scale == 1.0
        # `(x − μs)R + μd` is `xR + (μd − μs R)`: the same map with one fewer
        # full-length array operation per query. Folded once here rather than
        # recomputed on every call. Measured equal to the unfolded form to
        # 7e-08 across a thousand queries, which leaves the ranking identical.
        self._bias = as_float32(mean_dst - self.scale * (mean_src @ self.rotation))

    @classmethod
    def fit(cls, src: FloatArray, dst: FloatArray, *, with_scale: bool = False) -> Self:
        """Solve the centred orthogonal Procrustes problem."""
        check_fit_inputs(src, dst)
        input_dim, output_dim = src.shape[1], dst.shape[1]
        work = max(input_dim, output_dim)
        a = pad_or_truncate(as_float32(src), work)
        b = pad_or_truncate(as_float32(dst), work)

        mean_a = a.mean(axis=0)
        mean_b = b.mean(axis=0)
        centred_a = a - mean_a
        try:
            rotation, singular_sum = scipy.linalg.orthogonal_procrustes(centred_a, b - mean_b)
        except (ValueError, scipy.linalg.LinAlgError) as exc:
            raise FitFailed(
                "The centred orthogonal Procrustes solution did not converge.",
                hint="Usually degenerate input — all-identical or all-zero vectors.",
                context={"count": int(src.shape[0]), "dim": input_dim},
                cause=exc,
            ) from exc

        scale = 1.0
        if with_scale:
            # scipy returns the sum of singular values; divided by ‖A‖_F² it is
            # the optimal global scale for ‖sAR − B‖.
            denominator = float(np.sum(centred_a**2))
            scale = float(singular_sum / denominator) if denominator > 0 else 1.0

        return cls(
            as_float32(rotation),
            as_float32(mean_a),
            as_float32(mean_b),
            input_dim=input_dim,
            output_dim=output_dim,
            scale=scale,
        )

    def apply(self, x: FloatArray) -> FloatArray:
        """Rotate, then add the folded offset. Hot path."""
        if self._passthrough:
            return as_float32(as_float32(x) @ self.rotation + self._bias)
        z = (pad_or_truncate(as_float32(x), self._work_dim) - self.mean_src) @ self.rotation
        if self.scale != 1.0:
            z *= self.scale
        z += self.mean_dst
        return as_float32(np.ascontiguousarray(z[..., : self.output_dim]))

    def state_dict(self) -> dict[str, FloatArray]:
        """Rotation plus the two mean vectors.

        The folded bias is not stored: it is derived from these three, and a
        stored copy is one more thing that can disagree with them.
        """
        return {
            "rotation": self.rotation,
            "mean_src": self.mean_src,
            "mean_dst": self.mean_dst,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, FloatArray], config: Mapping[str, Any]) -> Self:
        """Reconstruct from stored weights."""
        return cls(
            as_float32(state["rotation"]),
            as_float32(state["mean_src"]),
            as_float32(state["mean_dst"]),
            input_dim=int(config["input_dim"]),
            output_dim=int(config["output_dim"]),
            scale=float(config.get("scale", 1.0)),
        )
