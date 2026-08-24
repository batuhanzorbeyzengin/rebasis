"""Diagonal scaling (DSM), applied on top of a base adapter.

``g'(x) = S · g(x)`` with ``S`` diagonal: *d* parameters, roughly 1.5 KB at
d=384, and under a microsecond to apply. The reference work reports +0.005–0.015
ARR from it.

Each dimension is a separate least-squares problem::

    s_j = ⟨ŷ_j, y_j⟩ / ⟨ŷ_j, ŷ_j⟩
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from rebasis.core.base import BaseAdapter
from rebasis.types import AdapterKind, FloatArray, as_float32

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self

__all__ = ["ScaledAdapter"]

#: Below this the base adapter maps a dimension to effectively zero, so there is
#: no ratio to fit and scaling it would amplify numerical noise.
_ZERO_VARIANCE = 1e-12


class ScaledAdapter(BaseAdapter):
    """A base adapter followed by per-dimension scaling.

    Composition rather than inheritance: any adapter can be wrapped, and the
    wrapper does not need to know which one it has.
    """

    #: Not meaningful on its own — the composite name comes from
    #: :attr:`type_name`, which folds in the base adapter's name.
    kind: ClassVar[AdapterKind] = "identity"

    def __init__(self, base: BaseAdapter, scale: FloatArray) -> None:
        super().__init__(
            input_dim=base.input_dim,
            output_dim=base.output_dim,
            config={"base_kind": base.kind, **base.config},
        )
        self.base = base
        self.scale = scale

    @property
    def type_name(self) -> str:
        """Composite name, so reports say `procrustes_centered+dsm`, not `dsm`."""
        return f"{self.base.type_name}+dsm"

    @classmethod
    def fit(cls, base: BaseAdapter, src: FloatArray, dst: FloatArray) -> Self:
        """Fit the diagonal on the residual of ``base``."""
        predicted = base.apply(src)
        target = as_float32(dst)
        numerator = np.einsum("ij,ij->j", predicted, target)
        denominator = np.einsum("ij,ij->j", predicted, predicted)
        # A dimension the base adapter maps to zero has no information to scale;
        # leaving it at 1.0 is the neutral choice.
        scale = as_float32(
            np.where(
                denominator > _ZERO_VARIANCE,
                numerator / np.maximum(denominator, _ZERO_VARIANCE),
                1.0,
            )
        )
        return cls(base, scale)

    def apply(self, x: FloatArray) -> FloatArray:
        """Base adapter followed by an element-wise scale. Hot path."""
        return as_float32(self.base.apply(x) * self.scale)

    def state_dict(self) -> dict[str, FloatArray]:
        """The base adapter's weights, prefixed, plus the diagonal."""
        return {
            **{f"base.{k}": v for k, v in self.base.state_dict().items()},
            "scale": self.scale,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, FloatArray], config: Mapping[str, Any]) -> Self:
        """Reconstruct from stored weights."""
        from rebasis.core.registry import build_adapter

        base_state = {k.removeprefix("base."): v for k, v in state.items() if k.startswith("base.")}
        base = build_adapter(str(config["base_kind"]), base_state, config)
        return cls(base, as_float32(state["scale"]))
