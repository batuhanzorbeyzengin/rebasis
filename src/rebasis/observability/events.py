"""Stable event catalogue.

Free-text log messages cannot be searched or audited, so rebasis uses **stable
event names**: the message text may change, the event name may not. When a user
asks "why is my migration slow", the answer must be findable with
``grep migrate.batch.completed``.

Naming: ``<module>.<object>.<verb-past-tense>``, dot-separated, lowercase.

Every event carries a :class:`EventSpec` describing the level it is emitted at,
the fields it carries, and whether it produces an audit record. That metadata is
the single source for ``docs/reference/events.md``, which is generated rather
than hand-written — a hand-written catalogue drifts from the code within one
release.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "DROPPED_NO_TEXT",
    "DROPPED_UNJUDGED",
    "EVENT_CATALOG",
    "EventSpec",
    "Events",
    "is_known_event",
]


class Events(StrEnum):
    """Every event name rebasis may emit.

    Calling ``log.info("some string")`` instead of ``log.info(Events.X)`` is
    caught by the ``add_event_namespace`` processor and fails tests.
    """

    # runtime / observability
    RUNTIME_DETECTED = "runtime.detected"
    REDACTION_TRIGGERED = "observability.redaction.triggered"
    UNSAFE_LOGGING_ENABLED = "config.unsafe_logging_enabled"

    # probe
    PROBE_RUN_STARTED = "probe.run.started"
    PROBE_SAMPLE_DRAWN = "probe.sample.drawn"
    PROBE_GROUNDTRUTH_COMPUTED = "probe.groundtruth.computed"
    PROBE_METRICS_COMPUTED = "probe.metrics.computed"
    PROBE_DECISION_MADE = "probe.decision.made"
    PROBE_GEOMETRY_MEASURED = "probe.geometry.measured"
    PROBE_RUN_COMPLETED = "probe.run.completed"

    # fit
    FIT_ADAPTER_FITTED = "fit.adapter.fitted"
    FIT_ADAPTER_SELECTED = "fit.adapter.selected"
    FIT_CALIBRATOR_FITTED = "fit.calibrator.fitted"

    # adapter lifecycle
    ADAPTER_LOADED = "adapter.loaded"
    ADAPTER_REJECTED = "adapter.rejected"
    ADAPTER_APPLIED = "adapter.applied"
    ADAPTER_UPGRADED = "adapter.upgraded"

    # embed
    EMBED_PROFILE_RESOLVED = "embed.profile.resolved"
    EMBED_BATCH_COMPLETED = "embed.batch.completed"
    EMBED_BACKEND_UNAVAILABLE = "embed.backend.unavailable"

    # store
    STORE_OPENED = "store.opened"
    STORE_CAPABILITY_MISSING = "store.capability.missing"
    STORE_SEARCH_COMPLETED = "store.search.completed"
    STORE_WRITE_PERFORMED = "store.write.performed"

    # migrate
    MIGRATE_JOB_STARTED = "migrate.job.started"
    MIGRATE_BATCH_COMPLETED = "migrate.batch.completed"
    MIGRATE_BATCH_THROTTLED = "migrate.batch.throttled"
    MIGRATE_ITEM_FAILED = "migrate.item.failed"
    MIGRATE_JOB_PAUSED = "migrate.job.paused"
    MIGRATE_JOB_RESUMED = "migrate.job.resumed"
    MIGRATE_JOB_COMPLETED = "migrate.job.completed"
    MIGRATE_INDEX_MIXED = "migrate.index.mixed"
    MIGRATE_INDEX_MEASURED = "migrate.index.measured"
    MIGRATE_DURABILITY_VERIFIED = "migrate.durability.verified"
    MIGRATE_ADAPTER_REFITTED = "migrate.adapter.refitted"

    # compute
    COMPUTE_DEVICE_SELECTED = "compute.device.selected"
    COMPUTE_OOM_RECOVERED = "compute.oom.recovered"

    # storage
    STORAGE_WRITE_COMPLETED = "storage.write.completed"
    STORAGE_INTEGRITY_VERIFIED = "storage.integrity.verified"
    STORAGE_SHADOW_WRITTEN = "storage.shadow.written"
    STORAGE_ROLLBACK_COMPLETED = "storage.rollback.completed"
    STORAGE_GC_COMPLETED = "storage.gc.completed"

    # audit
    AUDIT_RECORD_WRITTEN = "audit.record.written"
    AUDIT_CHAIN_VERIFIED = "audit.chain.verified"

    # retry — a silent retry is the most common cause of undiagnosable
    # slowness, so every attempt is announced
    RETRY_ATTEMPTED = "retry.attempted"


@dataclass(frozen=True, slots=True)
class EventSpec:
    """Metadata for one event: level, fields carried, audit status.

    ``audited`` marks the small subset of events that also produce an audit
    record. Log and audit trail are different things: a log may be lost,
    sampled and reformatted; an audit record may not.
    """

    level: int
    fields: tuple[str, ...]
    audited: bool
    description: str


_D = logging.DEBUG
_I = logging.INFO

#: Fixed vocabulary for the ``dropped`` field.
#:
#: Why an enum and not a sentence: the redaction allowlist admits a field only
#: when it can never carry corpus content, and a free-form ``reason`` cannot
#: promise that. A closed set of codes can, and it groups on a dashboard.
DROPPED_NO_TEXT = "no_text"
DROPPED_UNJUDGED = "no_judged_document_in_sample"
_W = logging.WARNING
_E = logging.ERROR

EVENT_CATALOG: dict[Events, EventSpec] = {
    Events.RUNTIME_DETECTED: EventSpec(
        _I,
        ("environment", "log_level", "log_format", "blas_threads", "device"),
        audited=False,
        description="Detected runtime environment and resolved log configuration. Emitted "
        "as the first event of every run so that 'why is it logging at this "
        "level' is always answerable.",
    ),
    Events.REDACTION_TRIGGERED: EventSpec(
        _W,
        ("field_count",),
        audited=False,
        description="A field outside the allowlist reached the renderer and was redacted. "
        "In production code paths this counter must stay at zero — a non-zero "
        "value is a bug signal, not a safety net working as intended.",
    ),
    Events.UNSAFE_LOGGING_ENABLED: EventSpec(
        _W, (), audited=True, description="Redaction was disabled via --unsafe-log-content."
    ),
    Events.PROBE_RUN_STARTED: EventSpec(
        _I,
        ("run_id", "store_backend", "embed_backend", "dim"),
        audited=False,
        description="Probe run started.",
    ),
    Events.PROBE_SAMPLE_DRAWN: EventSpec(
        _I,
        ("count", "strategy", "seed", "dim", "dropped"),
        audited=False,
        description="Sample drawn from the corpus.",
    ),
    Events.PROBE_GROUNDTRUTH_COMPUTED: EventSpec(
        _I,
        ("count", "tier", "duration_ms", "dropped"),
        audited=False,
        description="Ground truth established.",
    ),
    Events.PROBE_METRICS_COMPUTED: EventSpec(
        _I,
        ("arr_r10", "arr_mrr", "naive_overlap", "score_shift", "tail_arr"),
        audited=False,
        description="Retrieval metrics computed.",
    ),
    Events.PROBE_GEOMETRY_MEASURED: EventSpec(
        _I,
        ("count", "dim", "geometry_delta", "alignment_bound"),
        audited=False,
        description="How closely the two models' pairwise similarities agree, and the "
        "bound on orthogonal alignment error that implies (Maystre et al., "
        "arXiv:2510.13406). Computed before any adapter is fitted. A bound, not a "
        "prediction: it says an alignment exists, not that retrieval will use it.",
    ),
    Events.PROBE_DECISION_MADE: EventSpec(
        _I,
        ("decision", "arr_r10", "borderline", "count", "seed", "device"),
        audited=True,
        description="The tool's actual output. Audited because this decision must remain "
        "defensible six months later.",
    ),
    Events.PROBE_RUN_COMPLETED: EventSpec(
        _I,
        ("run_id", "duration_ms", "peak_rss_bytes"),
        audited=False,
        description="Probe run finished.",
    ),
    Events.FIT_ADAPTER_FITTED: EventSpec(
        _I,
        ("adapter_type", "count", "dim", "duration_ms"),
        audited=True,
        description="An adapter was fitted.",
    ),
    Events.FIT_ADAPTER_SELECTED: EventSpec(
        _I, ("adapter_type", "arr_r10"), audited=True, description="auto selected the best adapter."
    ),
    Events.FIT_CALIBRATOR_FITTED: EventSpec(
        _I,
        ("count", "score_shift"),
        audited=False,
        description="Isotonic score calibrator fitted.",
    ),
    Events.ADAPTER_LOADED: EventSpec(
        _I,
        ("adapter_type", "direction", "dim"),
        audited=False,
        description="Adapter loaded from disk.",
    ),
    Events.ADAPTER_REJECTED: EventSpec(
        _E,
        ("error_code", "adapter_type"),
        audited=True,
        description="Fingerprint mismatch. Audited because it is security-relevant: it means "
        "someone tried to apply an adapter to an index it was not built for.",
    ),
    Events.ADAPTER_APPLIED: EventSpec(
        _D, ("adapter_type", "count"), audited=False, description="Adapter applied to a batch."
    ),
    Events.ADAPTER_UPGRADED: EventSpec(
        _I,
        ("adapter_type",),
        audited=True,
        description="An .rbs file was migrated to a newer schema.",
    ),
    Events.EMBED_PROFILE_RESOLVED: EventSpec(
        _I,
        ("embed_backend", "dim", "symmetric"),
        audited=False,
        description="Encoding profile resolved.",
    ),
    Events.EMBED_BATCH_COMPLETED: EventSpec(
        _D,
        ("embed_backend", "model_id", "batch_index", "count", "dim", "duration_ms"),
        audited=False,
        description="One embedding batch finished. DEBUG and sampled: per-record logging in "
        "a 500k migration buries the real signal.",
    ),
    Events.EMBED_BACKEND_UNAVAILABLE: EventSpec(
        _W,
        ("embed_backend", "error_code"),
        audited=False,
        description="Embedding backend unreachable.",
    ),
    Events.STORE_OPENED: EventSpec(
        _I, ("store_backend", "count", "dim"), audited=False, description="Vector store opened."
    ),
    Events.STORE_CAPABILITY_MISSING: EventSpec(
        _W,
        ("store_backend", "error_code"),
        audited=False,
        description="Store lacks a required capability.",
    ),
    Events.STORE_SEARCH_COMPLETED: EventSpec(
        _D,
        ("store_backend", "count", "duration_ms"),
        audited=False,
        description="Search completed.",
    ),
    Events.STORE_WRITE_PERFORMED: EventSpec(
        _I, ("store_backend", "count"), audited=True, description="User data was modified."
    ),
    Events.MIGRATE_JOB_STARTED: EventSpec(
        _I,
        ("job_id", "count", "adapter_type", "state"),
        audited=True,
        description="Migration job started.",
    ),
    Events.MIGRATE_BATCH_COMPLETED: EventSpec(
        _I,
        ("job_id", "batch_index", "count", "duration_ms", "state"),
        audited=False,
        description="One migration batch finished. This is the per-batch summary that "
        "replaces per-record logging.",
    ),
    Events.MIGRATE_BATCH_THROTTLED: EventSpec(
        _W,
        ("job_id", "batch_index", "peak_rss_bytes"),
        audited=False,
        description="Batch size halved because memory approached the ceiling.",
    ),
    Events.MIGRATE_ITEM_FAILED: EventSpec(
        _W,
        ("job_id", "record_id", "error_code"),
        audited=False,
        description="A single record failed.",
    ),
    Events.MIGRATE_JOB_PAUSED: EventSpec(
        _W,
        ("job_id", "state", "error_code"),
        audited=True,
        description="Migration paused; resumable.",
    ),
    Events.MIGRATE_JOB_RESUMED: EventSpec(
        _I, ("job_id", "state"), audited=True, description="Migration resumed."
    ),
    Events.MIGRATE_JOB_COMPLETED: EventSpec(
        _I, ("job_id", "count", "duration_ms"), audited=True, description="Migration completed."
    ),
    Events.MIGRATE_INDEX_MIXED: EventSpec(
        _W,
        ("job_id", "count", "state", "store_backend"),
        audited=False,
        description="The job stopped with records left, so the index now holds vectors "
        "from two embedding spaces and no single query is correct against all of it. "
        "`count` is how many records still carry the old model's geometry.",
    ),
    Events.MIGRATE_INDEX_MEASURED: EventSpec(
        _I,
        ("store_backend", "count", "ann_recall", "duration_ms"),
        audited=False,
        description="How much of the exact answer the store's own search returns, "
        "measured against streamed exact kNN over the same collection. Emitted "
        "either side of a migration: rewriting a vector does not rewrite the graph "
        "that was built around it.",
    ),
    Events.MIGRATE_DURABILITY_VERIFIED: EventSpec(
        _I,
        ("job_id", "count"),
        audited=True,
        description="A fresh connection confirmed the writes are still there.",
    ),
    Events.MIGRATE_ADAPTER_REFITTED: EventSpec(
        _I,
        ("job_id", "adapter_type", "arr_r10"),
        audited=True,
        description="Adapter refitted mid-migration.",
    ),
    Events.COMPUTE_DEVICE_SELECTED: EventSpec(
        _I,
        ("device", "backend"),
        audited=False,
        description="Compute device chosen for a sub-task.",
    ),
    Events.COMPUTE_OOM_RECOVERED: EventSpec(
        _W,
        ("device", "attempt", "count"),
        audited=False,
        description="Device OOM recovered by halving the batch. OOM must never kill a job.",
    ),
    Events.STORAGE_WRITE_COMPLETED: EventSpec(
        _D, ("duration_ms",), audited=False, description="Atomic write finished."
    ),
    Events.STORAGE_INTEGRITY_VERIFIED: EventSpec(
        _I, ("count",), audited=False, description="Integrity hashes verified."
    ),
    Events.STORAGE_SHADOW_WRITTEN: EventSpec(
        _I,
        ("job_id", "count"),
        audited=False,
        description="Shadow copy written for rollback.",
    ),
    Events.STORAGE_ROLLBACK_COMPLETED: EventSpec(
        _I, ("job_id", "count"), audited=True, description="Rollback restored the original vectors."
    ),
    Events.STORAGE_GC_COMPLETED: EventSpec(
        _I, ("count",), audited=True, description="Garbage collection removed artefacts."
    ),
    Events.AUDIT_RECORD_WRITTEN: EventSpec(
        _D,
        ("audit_seq", "action"),
        audited=False,
        description="An audit record was appended.",
    ),
    Events.AUDIT_CHAIN_VERIFIED: EventSpec(
        _I, ("count",), audited=False, description="Audit hash chain verified end to end."
    ),
    Events.RETRY_ATTEMPTED: EventSpec(
        _W,
        ("attempt", "duration_ms", "error_code"),
        audited=False,
        description="A transient operation is being retried.",
    ),
}


def is_known_event(name: str) -> bool:
    """Whether ``name`` is a registered event (used by the log processor)."""
    return name in _KNOWN


_KNOWN: frozenset[str] = frozenset(str(e) for e in Events)
