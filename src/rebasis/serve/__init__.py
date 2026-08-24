"""Serving-time API.

The layer contract lets ``serve`` use ``core``, ``store``, ``embed``,
``manifest`` and ``observability`` — and **forbids importing torch**. That is not
an optimisation but a correctness constraint: a single query is budgeted at 15 µs
and a host→device→host transfer alone exceeds it.
"""

from __future__ import annotations

from rebasis.serve.bridge import Bridge
from rebasis.serve.hybrid import calibrated_merge, reciprocal_rank_fusion
from rebasis.serve.integrations import wrap_retriever

__all__ = ["Bridge", "calibrated_merge", "reciprocal_rank_fusion", "wrap_retriever"]
