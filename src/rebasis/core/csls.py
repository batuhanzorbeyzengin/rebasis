"""CSLS hubness correction.

MUSE's observation: vectors mapped from one space into another develop
**hubness** — a few target vectors become the nearest neighbour of
disproportionately many queries and wreck retrieval. CSLS penalises documents
that are "close to everyone"::

    CSLS(q, d) = 2·cos(q, d) − r_T(d) − r_S(q)

For ranking within one query ``r_S(q)`` is constant and drops out. Ranking by the
remaining ``2·cos − r_T(d)`` is **identical** to ranking by ``cos − r_T(d)/2``,
so the whole correction reduces to a per-document bias of ``−r_T(d)/2``.

*The factor of two matters.* Using ``−r_T(d)`` doubles the penalty and makes CSLS
look worse than it is — a plausible reason the reference work's numbers might
disagree with anyone reimplementing it.

**Measured (M0, 107 configurations):** CSLS is **not** a free improvement. On
weak adapters (ARR < 0.5) it adds **+0.103**; on strong ones (ARR ≥ 0.8) it
*costs* **−0.045**. Spearman correlation between its gain and adapter quality is
**−0.704**. Hubness forms when the mapping is poor; when it is good there is no
crowding to correct and the penalty distorts real signal.

So CSLS is offered as a **variant that ``auto`` evaluates and selects**, never as
an always-on correction. Cost: computed once per document, zero per query — the
hot path is untouched.
"""

from __future__ import annotations

import numpy as np

from rebasis.types import FloatArray, as_float32

__all__ = ["CSLS_NEIGHBOURS", "csls_bias", "hubness_penalty"]

#: Neighbourhood size for the penalty. MUSE uses 10; the value is not sensitive.
CSLS_NEIGHBOURS = 10


def hubness_penalty(
    docs: FloatArray,
    query_sample: FloatArray,
    *,
    k: int = CSLS_NEIGHBOURS,
    chunk_size: int = 1024,
) -> FloatArray:
    """``r_T(d)``: mean similarity of each document to its *k* nearest queries.

    ``query_sample`` should be source-distribution vectors already mapped into
    the target space. That sample is available at fit time without any query
    log, which is what makes CSLS usable in the T0 tier at all.

    Computed in chunks so the ``(n_docs, n_queries)`` similarity matrix is never
    materialised: peak memory stays proportional to the chunk, not to the corpus.
    """
    docs = as_float32(docs)
    query_sample = as_float32(query_sample)
    n_docs = docs.shape[0]
    k = min(k, query_sample.shape[0])

    out = np.empty(n_docs, dtype=np.float32)
    for start in range(0, n_docs, chunk_size):
        stop = min(start + chunk_size, n_docs)
        sims = docs[start:stop] @ query_sample.T
        top = np.partition(sims, -k, axis=1)[:, -k:]
        out[start:stop] = top.mean(axis=1)
    return out


def csls_bias(
    docs: FloatArray,
    query_sample: FloatArray,
    *,
    k: int = CSLS_NEIGHBOURS,
    chunk_size: int = 1024,
) -> FloatArray:
    """Per-document ranking bias implementing CSLS.

    Returns ``−r_T(d)/2``, which added to a cosine score reproduces the CSLS
    ordering exactly. See the module docstring for why the halving is not
    optional.
    """
    penalty = hubness_penalty(docs, query_sample, k=k, chunk_size=chunk_size)
    return as_float32(-penalty / 2.0)
