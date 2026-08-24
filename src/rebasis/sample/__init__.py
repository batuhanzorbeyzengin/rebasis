"""Corpus sampling.

The layer contract lets ``sample`` depend on ``store`` and ``observability``.

The default is **stratified k-means**. Uniform random sampling under-represents
small clusters, and small clusters are exactly where heterogeneous drift hides:
one global adapter once reached 0.85 ARR on a corpus where two domain-specific
ones reached 0.94. Clustering the sample also makes the ``tail_arr`` metric
available for free, which is the only early warning of that situation.
"""

from __future__ import annotations

from rebasis.sample.strategies import (
    DEFAULT_FIT_PAIRS,
    MIN_SAMPLE,
    SampleResult,
    draw_sample,
    random_sample,
    split_disjoint,
    stratified_sample,
    suggested_cluster_count,
)

__all__ = [
    "DEFAULT_FIT_PAIRS",
    "MIN_SAMPLE",
    "SampleResult",
    "draw_sample",
    "random_sample",
    "split_disjoint",
    "stratified_sample",
    "suggested_cluster_count",
]
