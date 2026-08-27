"""Adapter base class and the shared fitting contract.

The contract is the same everywhere, and the synthetic validation tests hold
every adapter to it::

    adapter = SomeAdapter.fit(src, dst)
    adapter.apply(src) ≈ dst

Under the default ``query_to_old`` direction, ``src = f_new(d)`` and
``dst = f_old(d)``: the adapter maps the new model's space onto the old index's
space, so the index is never touched.

**``apply`` is the hot path.** The budget for a single query is 15 µs, and M0
measured that at this size *every* numpy operation costs ~2.5–3.5 µs regardless
of the arithmetic involved — the cost is per-call overhead, not FLOPs.
Consequently ``apply`` performs no validation, logs nothing, copies no
dictionaries and allocates as little as it can. Validation happens once, in
``Bridge.load()``.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from rebasis.compute.arrays import l2_normalize, pad_or_truncate
from rebasis.errors import FitFailed
from rebasis.types import (  # noqa: TC001 - AdapterKind is a runtime ClassVar annotation
    AdapterKind,
    FloatArray,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self

# l2_normalize and pad_or_truncate are re-exported from compute.arrays: they
# are primitives every layer needs, and the layer contract puts compute below core.
__all__ = ["BaseAdapter", "check_fit_inputs", "l2_normalize", "pad_or_truncate"]

#: Fitting always operates on (n_pairs, dim) matrices.
_MATRIX_NDIM = 2


def check_fit_inputs(src: FloatArray, dst: FloatArray, *, minimum: int = 32) -> None:
    """Validate a fitting pair set.

    Raises:
        FitFailed: On mismatched lengths, too few pairs, or non-finite values.

    Non-finite values are checked because a single NaN silently propagates
    through the closed-form solutions and produces an adapter that is entirely
    NaN — which then fails as "quality is bad" rather than "the input was
    broken", and that is a much longer path to the real cause.
    """
    if src.ndim != _MATRIX_NDIM or dst.ndim != _MATRIX_NDIM:
        raise FitFailed(
            f"Fitting needs 2-D arrays, got shapes {src.shape} and {dst.shape}.",
            hint="Pass matrices of shape (n_pairs, dim).",
            context={"count": int(src.shape[0]) if src.ndim else 0},
        )
    if src.shape[0] != dst.shape[0]:
        raise FitFailed(
            f"Source and target have different pair counts: {src.shape[0]} and {dst.shape[0]}.",
            hint="Both sides must describe the SAME documents in the same order.",
            context={"count": int(src.shape[0])},
        )
    if src.shape[0] < minimum:
        raise FitFailed(
            f"Only {src.shape[0]} matched pairs; at least {minimum} are needed.",
            hint=(
                "Increase --pairs. Measurement suggests quality saturates around "
                "4000 pairs, and below a few hundred the adapter is unstable."
            ),
            context={"count": int(src.shape[0])},
        )
    if not (np.isfinite(src).all() and np.isfinite(dst).all()):
        raise FitFailed(
            "Input vectors contain NaN or infinity.",
            hint=(
                "A single non-finite value propagates through the closed form and "
                "produces an all-NaN adapter. Check the embedding backend."
            ),
            context={"count": int(src.shape[0])},
        )


class BaseAdapter(abc.ABC):
    """Common behaviour for every adapter.

    Subclasses provide ``apply`` (the hot path), ``state_dict``/``from_state_dict``
    (``.rbs`` serialisation) and a ``fit`` classmethod.
    """

    kind: ClassVar[AdapterKind]

    #: Scalar configuration that is not weights — rank, hidden width and so on.
    #: Stored as JSON in the ``.rbs`` manifest rather than as a tensor.
    config: dict[str, Any]

    def __init__(
        self, *, input_dim: int, output_dim: int, config: dict[str, Any] | None = None
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = dict(config or {})

    @abc.abstractmethod
    def apply(self, x: FloatArray) -> FloatArray:
        """Map vectors into the target space. Hot path — no validation.

        **Must return a new array, never ``x`` itself.** Callers normalise the
        result in place to save an allocation on a path budgeted at 15 µs, so an
        implementation that aliases its input silently rewrites the caller's
        query vector. Every adapter that multiplies gets this for free;
        :class:`~rebasis.core.identity.IdentityAdapter` did not, and did rewrite
        it until the contract was written down here.
        """

    @abc.abstractmethod
    def state_dict(self) -> dict[str, FloatArray]:
        """Weights to serialise, as contiguous float32 arrays."""

    @classmethod
    @abc.abstractmethod
    def from_state_dict(cls, state: Mapping[str, FloatArray], config: Mapping[str, Any]) -> Self:
        """Reconstruct an adapter from stored weights."""

    @property
    def type_name(self) -> str:
        """Name written to the ``.rbs`` manifest and to reports.

        Separate from :attr:`kind` because a composite adapter reports a
        composite name — `procrustes_centered+dsm` — while `kind` stays a plain
        class-level constant that the registry can look up.
        """
        return str(self.kind)

    def n_params(self) -> int:
        """Total parameter count."""
        return int(sum(v.size for v in self.state_dict().values()))

    @property
    def memory_bytes(self) -> int:
        """Weight size in bytes, float32."""
        return self.n_params() * 4

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(input_dim={self.input_dim}, "
            f"output_dim={self.output_dim}, params={self.n_params()})"
        )
