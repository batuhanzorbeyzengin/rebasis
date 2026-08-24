"""Diagnosis: measure the drift, then recommend.

The layer contract lets ``probe`` use ``core``, ``sample``, ``store``,
``embed``, ``compute``, ``audit`` and ``observability``.

``probe`` is **read-only**. It never writes to the user's index, and a contract
test enforces that with a store wrapper that fails on any write.
"""

from __future__ import annotations

from rebasis.probe.decision import (
    BORDERLINE_BAND,
    DECISION_BANDS,
    DecisionResult,
    decide,
)
from rebasis.probe.groundtruth import GroundTruth, build_tier0, build_tier1
from rebasis.probe.metrics import (
    METRIC_VERSION,
    bootstrap_ci,
    bootstrap_ratio_ci,
    mrr_at_k,
    overlap_at_k,
    recall_at_k,
    score_shift_ks,
    spearman_topk,
    top_k_search,
)
from rebasis.probe.runner import CandidateMetrics, ProbeResult, run_probe
from rebasis.probe.session import (
    CorpusSample,
    QueryLog,
    draw_corpus_sample,
    probe_store,
    record_decision,
)

__all__ = [
    "BORDERLINE_BAND",
    "DECISION_BANDS",
    "METRIC_VERSION",
    "CandidateMetrics",
    "CorpusSample",
    "DecisionResult",
    "GroundTruth",
    "ProbeResult",
    "QueryLog",
    "bootstrap_ci",
    "bootstrap_ratio_ci",
    "build_tier0",
    "build_tier1",
    "decide",
    "draw_corpus_sample",
    "mrr_at_k",
    "overlap_at_k",
    "probe_store",
    "recall_at_k",
    "record_decision",
    "run_probe",
    "score_shift_ks",
    "spearman_topk",
    "top_k_search",
]
