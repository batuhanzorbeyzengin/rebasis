"""Exact top-k search.

Lives in ``compute`` rather than beside the metrics that call it because the
``O(batch × d)`` memory invariant is a compute-layer property, and because both
the metrics and ``NumpyBackend`` need it. Two copies of this loop would be two
places for the invariant to be broken independently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rebasis.types import FloatArray, as_float32

if TYPE_CHECKING:
    from rebasis.compute.device import Device

__all__ = ["top_k_search"]


def top_k_search(  # noqa: PLR0913 - search inputs plus the two masking hooks
    queries: FloatArray,
    docs: FloatArray,
    k: int,
    *,
    chunk_size: int = 1024,
    self_mask: np.ndarray | None = None,
    doc_bias: FloatArray | None = None,
    device: Device | None = None,
) -> tuple[np.ndarray, FloatArray]:
    """Exact top-k by chunked matmul plus ``argpartition``.

    The full score matrix is **never materialised**: at 10k×10k it is 400 MB in
    float32, while 1024-row chunks cost 40 MB. This is the concrete form of the
    architectural invariant that peak memory is ``O(batch × d)``.

    ``argpartition`` rather than ``argsort``: O(n) instead of O(n log n), worth
    several times at this size.

    Args:
        queries: Query vectors, ℓ2-normalised.
        docs: Document vectors, ℓ2-normalised.
        k: How many neighbours to return.
        chunk_size: Query rows processed per pass. Bounds peak memory.
        self_mask: Per-query document index to exclude, or -1. In the T0 tier a
            query chunk sits inside the index, and letting it retrieve itself
            inflates ARR.
        doc_bias: Per-document constant added to the score. The CSLS hubness
            penalty is exactly this.
        device: Where to run it. ``None`` falls back to the ambient device set
            by :func:`rebasis.compute.using_device`, which `probe` enters once
            for the whole run; a CPU device keeps the numpy loop below, which is
            the reference implementation. Anything else hands the work to that
            device's backend — measured at 22-58× on the ground-truth kNN, the
            second largest cost in a `probe` after embedding.
    """
    from rebasis.compute.device import active_device

    device = device if device is not None else active_device()
    if device is not None and not device.is_cpu:
        from rebasis.compute import get_backend

        return get_backend(device).matmul_topk(
            queries,
            docs,
            k,
            chunk_size=chunk_size,
            doc_bias=doc_bias,
            self_mask=self_mask,
        )

    queries = as_float32(queries)
    docs = as_float32(docs)
    n_queries, n_docs = queries.shape[0], docs.shape[0]
    k = max(1, min(k, n_docs - (1 if self_mask is not None else 0)))

    out_indices = np.empty((n_queries, k), dtype=np.int32)
    out_scores = np.empty((n_queries, k), dtype=np.float32)

    for start in range(0, n_queries, chunk_size):
        stop = min(start + chunk_size, n_queries)
        scores = queries[start:stop] @ docs.T

        if doc_bias is not None:
            scores = scores + doc_bias

        if self_mask is not None:
            rows = np.arange(stop - start)
            columns = self_mask[start:stop]
            valid = columns >= 0
            scores[rows[valid], columns[valid]] = -np.inf

        partition = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        partition_scores = np.take_along_axis(scores, partition, axis=1)
        order = np.argsort(-partition_scores, axis=1)
        out_indices[start:stop] = np.take_along_axis(partition, order, axis=1)
        out_scores[start:stop] = np.take_along_axis(partition_scores, order, axis=1)

    return out_indices, out_scores
