"""The migration state machine.

Explicit states, and explicit transitions between them. The queue is
hand-written rather than taken from a package precisely so this could be
modelled directly rather than bent into someone else's abstraction.

The property that matters is **resumability**: every state is durable, so a job
killed at any point resumes from where it stopped rather than starting over. A
migration can take hours, and "start again" is not an acceptable answer to a
laptop lid closing.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["ITEM_TRANSITIONS", "JOB_TRANSITIONS", "ItemState", "JobState", "can_transition"]


class JobState(StrEnum):
    """Where a migration job is."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ItemState(StrEnum):
    """Where a single record is."""

    PENDING = "pending"
    SHADOWED = "shadowed"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


#: Allowed job transitions.
#:
#: ``PAUSED → RUNNING`` is the one that carries weight: memory pressure, a low
#: battery or a filling disk all pause rather than fail, and the user resumes
#: when the cause is gone.
JOB_TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.PENDING: frozenset({JobState.RUNNING, JobState.FAILED}),
    JobState.RUNNING: frozenset({JobState.PAUSED, JobState.COMPLETED, JobState.FAILED}),
    JobState.PAUSED: frozenset({JobState.RUNNING, JobState.FAILED}),
    # A completed job can still be rolled back — that is the whole point of the
    # shadow copy. "Finished" is not the same as "irreversible".
    JobState.COMPLETED: frozenset({JobState.ROLLED_BACK}),
    JobState.FAILED: frozenset({JobState.RUNNING, JobState.ROLLED_BACK}),
    JobState.ROLLED_BACK: frozenset(),
}

#: Allowed item transitions.
#:
#: ``PENDING → SHADOWED → DONE`` is deliberate ordering: the original vector is
#: safely stored *before* it is overwritten. A crash between the two leaves a
#: shadow with no matching write, which is harmless; the reverse would lose data.
ITEM_TRANSITIONS: dict[ItemState, frozenset[ItemState]] = {
    ItemState.PENDING: frozenset({ItemState.SHADOWED, ItemState.FAILED, ItemState.SKIPPED}),
    ItemState.SHADOWED: frozenset({ItemState.DONE, ItemState.FAILED}),
    ItemState.DONE: frozenset({ItemState.PENDING}),  # rollback returns it
    ItemState.FAILED: frozenset({ItemState.PENDING, ItemState.SKIPPED}),
    ItemState.SKIPPED: frozenset({ItemState.PENDING}),
}


def can_transition(current: JobState | ItemState, target: JobState | ItemState) -> bool:
    """Whether a transition is allowed."""
    if isinstance(current, JobState) and isinstance(target, JobState):
        return target in JOB_TRANSITIONS[current]
    if isinstance(current, ItemState) and isinstance(target, ItemState):
        return target in ITEM_TRANSITIONS[current]
    return False
