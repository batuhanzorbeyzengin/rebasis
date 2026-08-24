"""The audit trail.

Layer contract: ``audit`` may use ``manifest``, ``storage``, ``observability``
and ``types``.

**``audit`` is only ever called from above.** ``core`` writes no audit records:
``probe`` makes the decision, so ``probe`` writes the record. Keeping the
mathematics free of side effects is what keeps it simple to test and reusable.

One sentence covers the three channels: *you may throw away a log, you may never
enable telemetry, and you may not throw away an audit record.*
"""

from __future__ import annotations

from rebasis.audit.chain import ChainVerification, verify_chain
from rebasis.audit.reader import AuditReader
from rebasis.audit.record import GENESIS_HASH, AuditRecord, EnvironmentSnapshot
from rebasis.audit.replay import (
    CROSS_DEVICE_ARR_TOLERANCE,
    ReplayComparison,
    ReplayVerdict,
    compare_replay,
    replayable_inputs,
)
from rebasis.audit.writer import AuditWriter, is_audited

__all__ = [
    "CROSS_DEVICE_ARR_TOLERANCE",
    "GENESIS_HASH",
    "AuditReader",
    "AuditRecord",
    "AuditWriter",
    "ChainVerification",
    "EnvironmentSnapshot",
    "ReplayComparison",
    "ReplayVerdict",
    "compare_replay",
    "is_audited",
    "replayable_inputs",
    "verify_chain",
]
