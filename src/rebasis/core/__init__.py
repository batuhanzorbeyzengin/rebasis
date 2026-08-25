"""Adapter mathematics.

The layer contract puts ``core`` above ``compute``: it depends on ``compute``
and ``observability`` only. It knows nothing about stores, embedders or the CLI
— which is what keeps it testable in isolation and reusable as a library.

The adapter direction is ``query_to_old`` by default: the new model's query
vector is mapped into the old index's space, so the index is never rewritten.
That default is also why the bridge phase needs no shadow copy — there is
nothing to roll back.
"""

from __future__ import annotations

from rebasis.core.base import BaseAdapter, l2_normalize, pad_or_truncate
from rebasis.core.calibration import ScoreCalibrator
from rebasis.core.csls import CSLS_NEIGHBOURS, csls_bias, hubness_penalty
from rebasis.core.geometry import GeometryBound, geometry_bound
from rebasis.core.identity import IdentityAdapter
from rebasis.core.linear import LinearAdapter, LowRankAffineAdapter, default_rank
from rebasis.core.procrustes import CenteredProcrustesAdapter, ProcrustesAdapter
from rebasis.core.registry import ADAPTER_CLASSES, build_adapter, resolve_adapter_class
from rebasis.core.scaling import ScaledAdapter
from rebasis.core.selection import (
    CANDIDATE_METHODS,
    CPU_ONLY_METHODS,
    AdapterCandidate,
    fit_candidates,
    select_best,
)
from rebasis.core.serialization import (
    CURRENT_SCHEMA,
    AdapterManifest,
    load_adapter,
    save_adapter,
)

__all__ = [
    "ADAPTER_CLASSES",
    "CANDIDATE_METHODS",
    "CPU_ONLY_METHODS",
    "CSLS_NEIGHBOURS",
    "CURRENT_SCHEMA",
    "AdapterCandidate",
    "AdapterManifest",
    "BaseAdapter",
    "CenteredProcrustesAdapter",
    "GeometryBound",
    "IdentityAdapter",
    "LinearAdapter",
    "LowRankAffineAdapter",
    "ProcrustesAdapter",
    "ScaledAdapter",
    "ScoreCalibrator",
    "build_adapter",
    "csls_bias",
    "default_rank",
    "fit_candidates",
    "geometry_bound",
    "hubness_penalty",
    "l2_normalize",
    "load_adapter",
    "pad_or_truncate",
    "resolve_adapter_class",
    "save_adapter",
    "select_best",
]
