"""The torch compute backend.

Runs on ``cuda``, ``mps`` or ``cpu``. Optional: without the ``[torch]`` extra the
:class:`~rebasis.compute.numpy_backend.NumpyBackend` handles everything and no
functionality is lost.

**Conversion happens at the boundary.** Arrays come in as numpy and go out as
numpy; no torch tensor escapes this module. That is what keeps ``core`` free of
torch, as the layer contract requires, and makes the two backends
interchangeable in the device-parity tests.

Three MPS traps are handled here rather than discovered later:

1. **Contiguity.** Several MPS operations silently return zeros or NaN when the
   underlying tensor is not contiguous. ``.contiguous()`` at every entry point
   costs nothing and prevents a class of bug that presents as "the same float32
   code updates weights on CPU but not on MPS".
2. **``PYTORCH_ENABLE_MPS_FALLBACK``.** Convenient-looking, but relying on it
   inside a loop creates severe synchronisation stalls. Instead
   :func:`~rebasis.compute.device.capabilities` probes what is supported, and an
   unsupported job moves to CPU **whole** rather than bouncing per operation.
3. **float64.** MPS does not support it at all. The float32 contract, adopted
   for memory reasons, means the wall is never reached.

**OOM never kills a job.** On ``torch.cuda.OutOfMemoryError`` the batch halves
and the work retries; after three failures it falls back to CPU and continues.
Dying mid-migration because a batch was too large is not acceptable when the
whole design is built around resumability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rebasis.compute.device import Device
from rebasis.errors import DeviceOutOfMemory, MissingDependency
from rebasis.observability import Events, get_logger
from rebasis.types import FloatArray, as_float32

if TYPE_CHECKING:
    import numpy as np

__all__ = ["MAX_OOM_RETRIES", "TorchBackend"]

log = get_logger(__name__)

#: How many times a batch is halved before the work moves to CPU.
MAX_OOM_RETRIES = 3


class TorchBackend:
    """Numerical operations on a torch device."""

    def __init__(self, device: Device | None = None, *, memory_fraction: float = 0.8) -> None:
        try:
            import torch
        except ImportError as exc:
            raise MissingDependency(
                "The torch backend needs torch.",
                hint=(
                    'Install it with `pip install "rebasis[torch]"`. For GPU, use '
                    "PyTorch's own index URL — rebasis does not choose a wheel "
                    "variant for you."
                ),
                context={"backend": "torch"},
                cause=exc,
            ) from exc

        self.device = device or Device("cpu")
        self._torch = torch
        self._device = self.device.torch_device()

        if self.device.kind == "cuda" and memory_fraction < 1.0:
            # An explicit ceiling, so rebasis does not crowd out whatever else
            # the user is running on the same card.
            torch.cuda.set_per_process_memory_fraction(memory_fraction, self.device.index or 0)

    def _to_tensor(self, x: FloatArray) -> Any:
        """Move an array onto the device, contiguous — MPS trap 1."""
        return self._torch.from_numpy(as_float32(x)).to(self._device).contiguous()

    def matmul_topk(  # noqa: PLR0913 - search inputs plus the two masking hooks
        self,
        queries: FloatArray,
        docs: FloatArray,
        k: int,
        *,
        chunk_size: int = 2048,
        doc_bias: FloatArray | None = None,
        self_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, FloatArray]:
        """Exact top-k on the device.

        M0 measured this at **22-58× faster than numpy including transfer**, at
        every size from 2,000 documents upward — so the expected "borderline,
        use a 50k threshold" does not hold and there is no threshold to apply.

        Chunked for the same reason as the numpy path: the full score matrix is
        never materialised, on the device or off it.
        """
        import numpy as np

        torch = self._torch
        query_tensor = self._to_tensor(queries)
        doc_tensor = self._to_tensor(docs)
        bias_tensor = self._to_tensor(doc_bias) if doc_bias is not None else None

        n_queries = query_tensor.shape[0]
        # One fewer candidate is reachable when each query hides a document, so
        # the same cap as the numpy path — otherwise `topk` is asked for more
        # than exists on a corpus barely larger than k.
        k = max(1, min(k, doc_tensor.shape[0] - (1 if self_mask is not None else 0)))
        indices = np.empty((n_queries, k), dtype=np.int32)
        scores = np.empty((n_queries, k), dtype=np.float32)

        current_chunk = chunk_size
        start = 0
        attempt = 0
        while start < n_queries:
            stop = min(start + current_chunk, n_queries)
            try:
                block = query_tensor[start:stop] @ doc_tensor.T
                if bias_tensor is not None:
                    block = block + bias_tensor
                if self_mask is not None:
                    columns = torch.as_tensor(
                        self_mask[start:stop], device=block.device, dtype=torch.long
                    )
                    valid = columns >= 0
                    rows = torch.arange(stop - start, device=block.device)
                    block[rows[valid], columns[valid]] = float("-inf")
                top_scores, top_indices = torch.topk(block, k=k, dim=1)
            except torch.cuda.OutOfMemoryError as exc:
                attempt += 1
                if attempt > MAX_OOM_RETRIES:
                    raise DeviceOutOfMemory(
                        f"Ran out of device memory even at a batch of {current_chunk}.",
                        hint=(
                            "Lower --device-memory-fraction, or run this step on "
                            "the CPU with --device cpu."
                        ),
                        context={"device": str(self.device), "attempt": attempt},
                        cause=exc,
                    ) from exc
                current_chunk = max(1, current_chunk // 2)
                # empty_cache only after shrinking: calling it every batch keeps
                # resetting the allocator and costs more than it saves.
                torch.cuda.empty_cache()
                log.warning(
                    Events.COMPUTE_OOM_RECOVERED,
                    device=str(self.device),
                    attempt=attempt,
                    count=current_chunk,
                )
                continue

            indices[start:stop] = top_indices.cpu().numpy()
            scores[start:stop] = top_scores.cpu().numpy()
            start = stop

        return indices, scores

    def normalize(self, x: FloatArray) -> FloatArray:
        """ℓ2-normalise rows on the device."""
        tensor = self._to_tensor(x)
        normalised = tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return as_float32(normalised.cpu().numpy())

    def apply_batch(self, weights: dict[str, FloatArray], x: FloatArray) -> FloatArray:
        """Apply a linear map on the device.

        Worth using for a large backfill; **not** for a single query, where
        transfer alone exceeds the hot-path budget.
        """
        tensor = self._to_tensor(x)
        weight = self._to_tensor(weights["weight"])
        result = tensor @ weight
        if "bias" in weights:
            result = result + self._to_tensor(weights["bias"])
        return as_float32(result.cpu().numpy())

    def __repr__(self) -> str:
        return f"TorchBackend(device={self.device})"
