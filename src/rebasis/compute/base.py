"""Compute backend protocol.

Two backends, one protocol:

* :class:`~rebasis.compute.numpy_backend.NumpyBackend` — always available, and
  the **reference implementation**. When two backends disagree, this one is
  right.
* ``TorchBackend`` — cuda / mps / cpu, only when the ``[torch]`` extra is
  installed. Arrives in M2.

**Conversion happens at the boundary.** Tensors move onto a device inside a
backend and come back as numpy. ``core``, ``probe`` and ``migrate`` never see a
torch tensor; everything speaks ``FloatArray``. That keeps both backends
swappable and testable, and preserves ``core``'s independence under the layer
contract.

Which work belongs on which device is decided per operation, not once per run,
and M0 measured those decisions rather than assuming them:

* **Embedding generation** — GPU by a factor of 25-40×. Always use it when present.
* **Ground-truth kNN** — GPU by 22-58× *including transfer*, at every size from
  2,000 documents upward. This was expected to be borderline above a 50k
  threshold; the measurement found no threshold at all.
* **Orthogonal Procrustes fit** — 0.15–0.29 s on CPU. Transfer would cost more
  than the computation. Always CPU.
* **The hot path** — categorically CPU. A single query's budget is tens of
  microseconds and is already dominated by per-call overhead; a device transfer
  cannot fit inside it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

    from rebasis.compute.device import Device
    from rebasis.types import FloatArray

__all__ = ["ComputeBackend", "should_use_accelerator"]


@runtime_checkable
class ComputeBackend(Protocol):
    """Numerical operations that may run on an accelerator."""

    device: Device

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
        """Exact top-k by inner product. Never materialises the full matrix.

        ``self_mask`` names, per query, one document index to exclude. The T0
        tier needs it — a query drawn from inside the index retrieves itself
        otherwise, and every ARR is inflated by the same free hit. It belongs in
        the protocol rather than only in the numpy path, because a backend that
        cannot mask is a backend the measurement tiers cannot use.
        """
        ...

    def normalize(self, x: FloatArray) -> FloatArray:
        """ℓ2-normalise rows."""
        ...

    def apply_batch(self, weights: dict[str, FloatArray], x: FloatArray) -> FloatArray:
        """Apply adapter weights to a batch."""
        ...


#: Work that is worth moving to an accelerator, from M0 measurement.
#:
#: A value of ``None`` means "always, when a device is available"; a number is
#: the input size above which it pays. Nothing is listed on assumption — an
#: operation that has not been measured stays on the CPU.
ACCELERATOR_THRESHOLDS: dict[str, int | None] = {
    # 25-40× measured. The dominant cost in the whole pipeline.
    "embedding": None,
    # 22× at 2,000 documents including transfer, rising to 58×. A 50k threshold
    # was assumed; the measurement found none. Faiss's "a few thousand is
    # faster on CPU" applies to ANN search with small query batches, not to a
    # batched exhaustive kNN, which is one large dense matmul.
    "knn": None,
    # 3.3× at d=384, 5.9× at d=768.
    "mlp_fit": 5000,
    # Closed form, 0.15–0.29 s on CPU: transfer would cost more than the work.
    "procrustes_fit": None,
    # Per-call overhead already exceeds the budget; a transfer cannot fit.
    "hot_path": None,
}

#: Operations that stay on the CPU whatever hardware is present.
CPU_ONLY_OPERATIONS = frozenset({"procrustes_fit", "hot_path", "store_io", "manifest"})


def should_use_accelerator(operation: str, size: int, *, device_available: bool) -> bool:
    """Decide whether one operation should run on an accelerator.

    Args:
        operation: A key from :data:`ACCELERATOR_THRESHOLDS`.
        size: Input size, in whatever unit that operation's threshold uses.
        device_available: Whether an accelerator exists at all.

    An unmeasured operation returns ``False``: nothing moves to an accelerator on
    the assumption that it is faster. M0 vindicated that rule — the kNN threshold
    was assumed to be 50,000 and turned out not to exist, while nothing was lost
    by having kept it on the CPU until it was measured.
    """
    if not device_available or operation in CPU_ONLY_OPERATIONS:
        return False
    if operation not in ACCELERATOR_THRESHOLDS:
        return False
    threshold = ACCELERATOR_THRESHOLDS[operation]
    return True if threshold is None else size >= threshold
