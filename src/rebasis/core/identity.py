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
        """Return the input, matched to the output dimension — as a new array.

        **The copy is the correctness, not an oversight.** Every other adapter
        allocates because it multiplies; this one has nothing to multiply by, so
        without the copy it hands back the caller's own array. ``Bridge`` then
        normalises the result in place — deliberately, to save an allocation on a
        path budgeted at 15 µs — and the caller's query vector is modified under
        them.

        Measured before it was fixed: ``bridge.to_index_space(q)`` with this
        adapter left ``q`` normalised, so a caller reusing ``q`` for a second
        index, a rerank or a log line was working with a different vector from
        the one they encoded. Nothing raised. The two paths that make it
        reachable are ``fit --method identity`` and loading a ``.rbs`` that
        records that type; ``auto`` never selects it, which is why this survived.

        The cost is one allocation on an adapter nobody should be serving with
        anyway, and :meth:`BaseAdapter.apply` now states the contract the rest of
        them already kept.
        """
        return pad_or_truncate(as_float32(x), self.output_dim).copy()

    def state_dict(self) -> dict[str, FloatArray]:
        """No weights."""
        return {}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, FloatArray], config: Mapping[str, Any]) -> Self:
        """Reconstruct from dimensions alone."""
        del state
        return cls(input_dim=int(config["input_dim"]), output_dim=int(config["output_dim"]))
