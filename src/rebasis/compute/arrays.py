"""Array primitives shared by every layer.

These live in ``compute`` rather than in ``core`` because of the layer contract:
it puts ``core`` above ``compute``, so a helper both of them need has to sit at
or below ``compute``. ``probe`` and ``serve`` use them too.

The float32 contract is enforced at this boundary and nowhere else: input
arrives in whatever dtype the caller had, is converted once, and is never
converted again inside the core.
"""

from __future__ import annotations

import numpy as np

from rebasis.types import FloatArray, as_float32

__all__ = ["l2_normalize", "pad_or_truncate"]


def l2_normalize(x: FloatArray, *, copy: bool = True) -> FloatArray:
    """ℓ2-normalise rows.

    On normalised vectors cosine similarity *is* the inner product, which removes
    a norm computation from every search.

    Zero-norm rows are divided by a floor rather than by zero: a zero vector
    stays zero instead of becoming NaN. One NaN propagates through every
    downstream metric and turns a data problem into an unexplained quality
    problem.

    A single vector takes a shorter route. ``np.linalg.norm`` re-derives the axis
    handling and the dtype on every call, which at d=768 costs 4.7 µs against
    1.4 µs for the dot product it ends up doing -- mostly dispatch, not
    arithmetic. The whole hot-path budget for one query is 15 µs, so that is a
    third of it: measured on the reference host, 8.6 µs down to 3.1 µs at d=768,
    with the batch path untouched.

    Args:
        x: Vectors, one per row.
        copy: When ``False`` the input is normalised in place. The hot path uses
            that to avoid an allocation per query.
    """
    x = as_float32(x)
    if copy:
        x = x.copy()
    row = _single_row(x)
    if row is not None:
        x /= max(float(np.dot(row, row)) ** 0.5, 1e-12)
        return x
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    x /= norms
    return x


#: A matrix of vectors. One below that is a bare vector, one above is a shape
#: this module does not claim to handle on the fast path.
_MATRIX_NDIM = 2


def _single_row(x: FloatArray) -> FloatArray | None:
    """The one vector in ``x``, or ``None`` if there is more than one.

    Both shapes the hot path uses -- ``(d,)`` from a caller who has one vector
    and ``(1, d)`` from an encoder that always returns a batch -- reduce to the
    same scalar division.
    """
    if x.ndim == 1:
        return x
    if x.ndim == _MATRIX_NDIM and x.shape[0] == 1:
        return as_float32(x[0])
    return None


def pad_or_truncate(x: FloatArray, dim: int) -> FloatArray:
    """Match a matrix to ``dim`` columns by zero-padding or truncating.

    Orthogonal Procrustes needs a square matrix. When the two models differ in
    dimensionality the smaller side is zero-padded, the rotation is solved in the
    larger space, and the output is truncated to the target dimension.
    """
    x = as_float32(x)
    if x.shape[-1] == dim:
        return x
    if x.shape[-1] > dim:
        return as_float32(np.ascontiguousarray(x[..., :dim]))
    out = np.zeros((*x.shape[:-1], dim), dtype=np.float32)
    out[..., : x.shape[-1]] = x
    return out
