"""The no-op adapter.

Feeds the new model's vector straight into the old index. Retained because it is
the honest baseline: the reference work reports ARR ≈ 0.65 for it, and M0
measured **0.274** across four corpora — worse than the reference, and far below
every fitted adapter.

Keeping it in the ``auto`` comparison is what lets a report say "the adapter buys
you 0.56 ARR over doing nothing" instead of quoting an absolute number the user
has nothing to compare against.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from rebasis.core.base import BaseAdapter, pad_or_truncate
from rebasis.types import AdapterKind, FloatArray, as_float32

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self

__all__ = ["IdentityAdapter"]


class IdentityAdapter(BaseAdapter):
    """Passes vectors through, padding or truncating if the dimensions differ."""

    kind: ClassVar[AdapterKind] = "identity"

    @classmethod
    def fit(cls, src: FloatArray, dst: FloatArray) -> Self:
        """Record the dimensions. There is nothing to learn."""
        return cls(input_dim=int(src.shape[1]), output_dim=int(dst.shape[1]))

    def apply(self, x: FloatArray) -> FloatArray:
        """Return the input, matched to the output dimension."""
        return pad_or_truncate(as_float32(x), self.output_dim)

    def state_dict(self) -> dict[str, FloatArray]:
        """No weights."""
        return {}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, FloatArray], config: Mapping[str, Any]) -> Self:
        """Reconstruct from dimensions alone."""
        del state
        return cls(input_dim=int(config["input_dim"]), output_dim=int(config["output_dim"]))
