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
from rebasis.migrate.power import PowerState, ResourceMonitor, power_state
from rebasis.migrate.queue import JobQueue, QueueStats, set_job_state
from rebasis.migrate.refit import (
    MIN_IMPROVEMENT,
    RefitDecision,
    RefitPolicy,
    consider_refit,
)
from rebasis.migrate.states import ItemState, JobState, can_transition

__all__ = [
    "MIN_IMPROVEMENT",
    "ItemState",
    "JobQueue",
    "JobState",
    "MigrationEngine",
    "MigrationResult",
    "PowerState",
    "QueueStats",
    "RefitDecision",
    "RefitPolicy",
    "ResourceMonitor",
    "can_transition",
    "consider_refit",
    "power_state",
    "set_job_state",
]
