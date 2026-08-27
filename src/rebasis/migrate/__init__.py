"""Gradual migration.

The layer contract lets ``migrate`` use ``core``, ``store``, ``embed``,
``manifest``, ``audit`` and ``observability``.

This is the **only** part of rebasis that writes to a user's index, and it only
ever upserts — it never deletes. Everything about it is built around being
interruptible: every state is durable, so a job survives a closed laptop lid, a
full disk or an out-of-memory batch and resumes where it stopped.
"""

from __future__ import annotations

from rebasis.migrate.engine import MigrationEngine, MigrationResult
from rebasis.migrate.health import HealthComparison, IndexHealth, measure_index_health
from rebasis.migrate.power import PowerState, ResourceMonitor, power_state
from rebasis.migrate.queue import (
    JobQueue,
    QueueStats,
    clear_pause_request,
    pause_requested,
    request_pause,
    set_job_state,
)
from rebasis.migrate.refit import (
    MIN_IMPROVEMENT,
    RefitDecision,
    RefitPolicy,
    consider_refit,
)
from rebasis.migrate.spaces import MixedSpace, mixed_spaces, mixed_spaces_for
from rebasis.migrate.states import ItemState, JobState, can_transition

__all__ = [
    "MIN_IMPROVEMENT",
    "HealthComparison",
    "IndexHealth",
    "ItemState",
    "JobQueue",
    "JobState",
    "MigrationEngine",
    "MigrationResult",
    "MixedSpace",
    "PowerState",
    "QueueStats",
    "RefitDecision",
    "RefitPolicy",
    "ResourceMonitor",
    "can_transition",
    "clear_pause_request",
    "consider_refit",
    "measure_index_health",
    "mixed_spaces",
    "mixed_spaces_for",
    "pause_requested",
    "power_state",
    "request_pause",
    "set_job_state",
]
