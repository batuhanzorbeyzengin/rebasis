"""The numpy compute backend.

Always available, and the **reference implementation**: in a device-parity
disagreement this is the one that is right. It exists so that
``pip install rebasis`` with no extras is a complete, working tool — not a
degraded one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rebasis.compute.arrays import l2_normalize
from rebasis.compute.device import Device
from rebasis.compute.search import top_k_search
from rebasis.types import FloatArray, as_float32

if TYPE_CHECKING:
    import numpy as np

__all__ = ["NumpyBackend"]


class NumpyBackend:
    """Numerical operations on the CPU with numpy and scipy."""

    def __init__(self, device: Device | None = None) -> None:
        self.device = device or Device("cpu")

    def matmul_topk(  # noqa: PLR0913 - search inputs plus the two masking hooks
        self,
        queries: FloatArray,
        docs: FloatArray,
        k: int,
        *,
        chunk_size: int = 1024,
        doc_bias: FloatArray | None = None,
        self_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, FloatArray]:
        """Exact top-k by chunked matmul plus ``argpartition``.

        Delegates to :func:`rebasis.compute.search.top_k_search` rather than
        reimplementing it: two copies of this loop would be two places for the
        memory invariant to be broken independently.
        """
        return top_k_search(
            queries, docs, k, chunk_size=chunk_size, doc_bias=doc_bias, self_mask=self_mask
        )

    def normalize(self, x: FloatArray) -> FloatArray:
        """ℓ2-normalise rows."""
        return l2_normalize(x)

    def apply_batch(self, weights: dict[str, FloatArray], x: FloatArray) -> FloatArray:
        """Apply a plain linear map. Richer adapters own their `apply`."""
        result = as_float32(x) @ weights["weight"]
        if "bias" in weights:
            result = result + weights["bias"]
        return as_float32(result)

    def __repr__(self) -> str:
        return f"NumpyBackend(device={self.device})"
