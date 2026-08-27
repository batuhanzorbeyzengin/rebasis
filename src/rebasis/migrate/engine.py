"""The migration engine.

Per batch, in this order:

1. Read the current vectors from the store.
2. **Write them to the shadow copy first**, before anything is overwritten.
3. Map them with the adapter, or re-embed with the new model.
4. Upsert.
5. **Read a sample back and compare.**
6. Check memory and power; throttle or pause.

Two of those steps carry most of the weight.

**Shadow before write.** A crash between steps 2 and 4 leaves a shadow entry with
no corresponding write, which is harmless. The reverse ordering would lose the
original vector permanently.

**Sampled read-back.** After every batch, 1% of the written records — at least
five — are read back and compared against what was sent. A store that silently
fails to write is one of the most common sources of quiet data loss, and nothing
else in the pipeline would notice it.

That check reads through the handle that did the writing, which is exactly the
handle a caching client will answer from its own memory. So a completed job runs
one more check on a **fresh connection**, against a reservoir sample kept from
the whole run. The two checks fail on different things: one catches a store that
never took the write, the other a store that took it and did not keep it.

The engine never dies where it can pause. Memory pressure, a low battery and a
**termination signal** all pause a checkpointed job; resuming is the user's call
once the cause is gone. A filling disk is refused before the job starts rather
than paused during it, because a disk that is already too small does not get
better by waiting — see `storage/budget.py`.

`SIGTERM` is the one worth naming, because it is how a Kubernetes `Job`, an
Airflow task and an Argo step all end. The CLI installs the handler and
`migrate/signals.py` explains why it is the CLI's job rather than this module's;
here it is one more reason in `_reason_to_stop`, indistinguishable from a
`rebasis pause` arriving from another terminal.
"""

from __future__ import annotations

import contextlib
import datetime
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from rebasis.core.base import l2_normalize
from rebasis.errors import MigrationInterrupted, StoreWriteFailed
from rebasis.migrate.power import ResourceMonitor, power_state
from rebasis.migrate.queue import (
    JobQueue,
    clear_pause_request,
    pause_requested,
    set_job_state,
)
from rebasis.migrate.refit import RefitPolicy, consider_refit
from rebasis.migrate.signals import stop_requested, stop_signal_name
from rebasis.migrate.states import ItemState, JobState
from rebasis.observability import (
    Events,
    Spans,
    get_logger,
    instrument,
    should_span_batch,
    span,
)
from rebasis.observability.semconv import DB_SYSTEM_NAME, REBASIS_MIGRATE_STATE
from rebasis.storage.shadow import ShadowStore
from rebasis.types import FloatArray  # noqa: TC001 - runtime annotation in a method signature

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from rebasis.audit import AuditWriter
    from rebasis.core.base import BaseAdapter
    from rebasis.manifest import ManifestDB
    from rebasis.migrate.queue import QueueStats
    from rebasis.store.base import VectorStore
    from rebasis.types import Embedder, EncodingProfile

__all__ = ["VERIFY_FRACTION", "MigrationEngine", "MigrationResult"]


def _ignore_progress(_count: int) -> None:
    """The default ``on_batch``: a migration with nobody watching."""


log = get_logger(__name__)

#: Fraction of each written batch read back and compared.
VERIFY_FRACTION = 0.01

#: Floor on that sample. 1% of a 200-record batch is two, which is not a check.
VERIFY_MINIMUM = 5

#: Tolerance for the read-back comparison. Not exact equality: some stores
#: round-trip through float32 in a way that loses the last bit, and failing a
#: whole batch on that would be a false alarm.
VERIFY_ATOL = 1e-4

#: Records kept aside for the end-of-job durability check, spread across the
#: whole run by reservoir sampling. Fixed, so the check costs the same on a
#: thousand records as on ten million and the ``O(batch × d)`` memory invariant
#: holds.
DURABILITY_SAMPLE = 64


@dataclass(slots=True)
class MigrationResult:
    """What a migration run did."""

    job_id: str
    state: JobState
    processed: int
    failed: int
    duration_seconds: float
    resources: dict[str, Any] = field(default_factory=dict)
    pause_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form, for the audit record."""
        return {
            "job_id": self.job_id,
            "state": str(self.state),
            "count": self.processed,
            "duration_ms": round(self.duration_seconds * 1000, 1),
            **self.resources,
        }


class MigrationEngine:
    """Runs one migration job."""

    def __init__(  # noqa: PLR0913 - each collaborator is a distinct dependency
        self,
        *,
        db: ManifestDB,
        store: VectorStore,
        adapter: BaseAdapter,
        shadow_root: Path,
        job_id: str | None = None,
        keep_original: bool = True,
        batch_size: int = 256,
        max_memory_bytes: int | None = None,
        power_aware: bool = True,
        audit: AuditWriter | None = None,
        store_uri: str = "",
        adapter_path: str = "",
        shadow_precision: str = "float32",
        refit: RefitPolicy | None = None,
        embedder: Embedder | None = None,
        adapter_root: Path | None = None,
        profiles: tuple[EncodingProfile, EncodingProfile] | None = None,
    ) -> None:
        self.db = db
        self.store = store
        self.store_uri = store_uri
        self.adapter = adapter
        # Recorded for the same reason as store_uri: `--resume` should need the
        # job id and nothing else. The column existed from schema 1 and was
        # written as "" the whole time, so no job before this one has it.
        self.adapter_path = adapter_path
        self.job_id = job_id or f"job-{uuid.uuid4().hex[:12]}"
        self.keep_original = keep_original
        self.power_aware = power_aware
        self.audit = audit
        self.queue = JobQueue(db, self.job_id)
        # Counted so batch spans can be sampled the way batch logs are.
        self._batches = 0
        self.shadow = ShadowStore(shadow_root, self.job_id, precision=shadow_precision)
        # Reservoir for the end-of-job durability check: ids, the vectors they
        # should hold, and how many candidates have gone past so far.
        self._kept: list[tuple[str, FloatArray]] = []
        self._seen = 0
        self._rng = np.random.default_rng(0)
        self.monitor = ResourceMonitor(
            max_memory_bytes=max_memory_bytes,
            dim=adapter.output_dim,
            initial_batch=batch_size,
        )
        # Refitting needs four things the rest of a migration does not: a policy
        # saying when, an embedder to make real pairs with, somewhere to write
        # the adapter it adopts, and the two profiles that adapter records. Any
        # of them missing means no refit; `_refit_is_possible` says which.
        self.refit = refit or RefitPolicy()
        self.embedder = embedder
        self.adapter_root = adapter_root
        self.profiles = profiles
        self._refits = 0

    def prepare(
        self,
        record_ids: list[str],
        *,
        priorities: dict[str, float] | None = None,
        total: int | None = None,
    ) -> int:
        """Register the job and fill its queue.

        Idempotent: calling it again after an interruption re-registers nothing
        and re-enqueues nothing, because the queue *is* the checkpoint.

        Args:
            record_ids: Ids to enqueue. May be one chunk of a larger corpus —
                the caller is expected to stream, so a five-million-record
                index never has all its ids in memory at once.
            priorities: Higher values are migrated first.
            total: The full corpus size, when ``record_ids`` is only a chunk of
                it. Recorded on the job so progress is reported against the
                whole migration rather than the first chunk.
        """
        now = _now()
        with self.db.transaction() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO jobs (
                    job_id, created_utc, updated_utc, state, store_backend, store_uri,
                    adapter_path, adapter_type, total_records, batch_size, keep_original
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.job_id,
                    now,
                    now,
                    JobState.PENDING,
                    self.store.capabilities.name,
                    self.store_uri,
                    self.adapter_path,
                    self.adapter.type_name,
                    total if total is not None else len(record_ids),
                    self.monitor.batch_size,
                    int(self.keep_original),
                ),
            )
        enqueued = self.queue.enqueue(record_ids, priorities)

        if self.audit:
            self.audit.write(
                Events.MIGRATE_JOB_STARTED,
                inputs={
                    "count": len(record_ids),
                    "adapter_type": self.adapter.type_name,
                    "keep_original": self.keep_original,
                    "batch_size": self.monitor.batch_size,
                    # What a rollback of this job will be worth, recorded at the
                    # moment the job is registered rather than left in terminal
                    # scrollback. `migrate` has no `--json`, so the audit trail
                    # is where a machine reads this — `rebasis audit show`
                    # prints the inputs as JSON and `audit export` emits JSONL.
                    # Three values and they are not two: `true` and `false` are
                    # findings about the store, `null` is the absence of one.
                    "store_quantized": self.store.capabilities.quantized,
                    # What a rollback of this job restores *to*. float16 halves
                    # the shadow and makes the restore close rather than exact,
                    # and which of the two was chosen is not recoverable from
                    # the index afterwards — only from here and from the shadow
                    # manifest beside it.
                    "shadow_precision": self.shadow.precision,
                },
                outputs={"job_id": self.job_id, "state": str(JobState.PENDING)},
                subject=self.job_id,
            )
        return enqueued

    # ── running ───────────────────────────────────────────────────────

    def run(
        self,
        *,
        limit: int | None = None,
        on_batch: Callable[[int], None] | None = None,
    ) -> MigrationResult:
        """Process the queue until it is empty, paused or the limit is reached.

        Args:
            limit: Stop after this many records. Used by ``--limit`` and by the
                end-to-end test that kills a job and resumes it.
            on_batch: Called with the number of records finished, once per
                batch. A callback rather than a progress bar because the engine
                sits below the CLI in the layer contract and has no business
                knowing what a terminal is. Exceptions raised by it are not
                caught: a broken display must not abandon a migration midway.
        """
        with span(Spans.MIGRATE, {"job_id": self.job_id}) as active:
            result = self._run(limit=limit, on_batch=on_batch)
            # Set at the end rather than the start, because the state is the
            # answer rather than the question: a trace filtered to
            # `rebasis.migrate.state = paused` is the one an operator wants, and
            # the reason lands next to it. No guard on `active`: with telemetry
            # off the tracer yields a no-op span that answers `set_attribute`,
            # which is what keeps this call site unconditional.
            active.set_attribute(REBASIS_MIGRATE_STATE, str(result.state))
            if result.pause_reason:
                active.set_attribute("rebasis.migrate.pause_reason", result.pause_reason)
            return result

    def _run(
        self,
        *,
        limit: int | None = None,
        on_batch: Callable[[int], None] | None = None,
    ) -> MigrationResult:
        """The run loop, inside the root span."""
        # Resolved once rather than tested every batch: the loop is the hot part
        # of a migration and a branch per batch buys nothing.
        notify = on_batch if on_batch is not None else _ignore_progress
        started = time.perf_counter()
        set_job_state(self.db, self.job_id, JobState.RUNNING)
        # Before the first batch, so a request left over from the run this one
        # is resuming does not stop it again immediately. Clearing it here
        # rather than in `rebasis resume` keeps one writer on the column: the
        # engine is the only thing that knows a run has actually begun.
        clear_pause_request(self.db, self.job_id)
        log.info(
            Events.MIGRATE_JOB_STARTED,
            job_id=self.job_id,
            count=self.queue.stats().total,
            adapter_type=self.adapter.type_name,
            state=str(JobState.RUNNING),
        )

        processed = 0
        failed = 0
        batch_index = 0
        pause_reason = ""
        last_refit_at = 0
        self._settle_refit()

        while True:
            if limit is not None and processed >= limit:
                break

            pause_reason = self._reason_to_stop()
            if pause_reason:
                break

            size = self.monitor.batch_size
            if limit is not None:
                size = min(size, limit - processed)
            ids = self.queue.next_batch(size)
            if not ids:
                break

            batch_started = time.perf_counter()
            try:
                succeeded, batch_failed = self._process_batch(ids)
            except MigrationInterrupted as exc:
                pause_reason = exc.message
                break

            processed += succeeded
            failed += batch_failed
            batch_index += 1
            notify(succeeded)

            log.info(
                Events.MIGRATE_BATCH_COMPLETED,
                job_id=self.job_id,
                batch_index=batch_index,
                count=succeeded,
                duration_ms=round((time.perf_counter() - batch_started) * 1000, 1),
                state=str(JobState.RUNNING),
            )

            # After the batch, never inside one. A refit swaps the adapter, and
            # swapping it half-way through a batch would leave that batch mapped
            # two ways with nothing recording where the seam is. On a boundary
            # the seam is the batch index, which the log already carries.
            last_refit_at = self._refit_if_due(processed, last_refit_at)

            pause_reason = self._pressure(batch_index)
            if pause_reason:
                break

        duration = time.perf_counter() - started
        state = self._finish(pause_reason, duration, processed)
        if state is JobState.COMPLETED:
            self.verify_durability()
        return MigrationResult(
            job_id=self.job_id,
            state=state,
            processed=processed,
            failed=failed,
            duration_seconds=duration,
            resources=self.monitor.summary(),
            pause_reason=pause_reason,
        )

    def _process_batch(self, ids: list[str]) -> tuple[int, int]:
        """Shadow, map, write, verify — in that order."""
        self._batches += 1
        if not should_span_batch(self._batches - 1):
            return self._process_batch_inner(ids)
        with span(Spans.MIGRATE_BATCH, {"count": len(ids), "batch": self._batches}):
            return self._process_batch_inner(ids)

    def _process_batch_inner(self, ids: list[str]) -> tuple[int, int]:
        """The batch itself. Split out so the span wrapper stays sampled."""
        records = list(self.store.iter_records(ids, with_vectors=True, with_text=False))
        present = [(r.id, r.vector) for r in records if r.vector is not None]
        if not present:
            self.queue.mark(ids, ItemState.SKIPPED)
            return 0, 0

        present_ids = [record_id for record_id, _ in present]
        originals = np.vstack([vector for _, vector in present])

        # Step 2, before anything is overwritten. A crash here costs nothing;
        # a crash after the write, without this, costs the original vectors.
        if self.keep_original:
            self.shadow.write(present_ids, originals)
            self.queue.mark(present_ids, ItemState.SHADOWED)

        mapped = l2_normalize(self.adapter.apply(originals), copy=False)

        try:
            # `db.system.name` is one of the few OTel names in this area that is
            # *stable* rather than development-status, so a collector already
            # knows what to do with it. There is no vector-database convention to
            # conform to — it is an open, unassigned issue upstream — and this
            # project does not invent one, so the backend's own declared name is
            # what goes in the standard field and nothing goes in an invented one.
            with span(
                Spans.STORE_UPSERT,
                {"count": len(present_ids), DB_SYSTEM_NAME: self.store.capabilities.name},
            ):
                self.store.upsert_vectors(present_ids, mapped)
        except StoreWriteFailed as exc:
            self.queue.mark(present_ids, ItemState.FAILED, error_code=exc.code)
            instrument("rebasis.migrate.items").add(len(present_ids), {"state": "failed"})
            log.warning(
                Events.MIGRATE_ITEM_FAILED,
                job_id=self.job_id,
                record_id=present_ids[0],
                error_code=exc.code,
            )
            return 0, len(present_ids)

        self._verify_sample(present_ids, mapped)
        self._keep_for_durability(present_ids, mapped)
        self.queue.mark(present_ids, ItemState.DONE)
        instrument("rebasis.migrate.items").add(len(present_ids), {"state": "done"})
        instrument("rebasis.migrate.progress").set(
            self.queue.stats().completed_fraction, {"job_id": self.job_id}
        )

        if self.audit:
            self.audit.write(
                Events.STORE_WRITE_PERFORMED,
                inputs={"job_id": self.job_id, "count": len(present_ids)},
                outputs={"count": len(present_ids)},
                subject=self.job_id,
            )
        return len(present_ids), 0

    def _verify_sample(self, ids: list[str], written: FloatArray) -> None:
        """Read a sample back and compare.

        A store that silently fails to write is a leading cause of quiet data
        loss, and nothing downstream would notice. On a mismatch the job stops
        rather than continuing to write into a store that is not storing.

        Raises:
            MigrationInterrupted: On a mismatch.
        """
        sample_size = max(VERIFY_MINIMUM, int(len(ids) * VERIFY_FRACTION))
        sample_size = min(sample_size, len(ids))
        positions = np.linspace(0, len(ids) - 1, sample_size, dtype=int)
        sampled_ids = [ids[i] for i in positions]

        read_back = {
            record.id: record.vector
            for record in self.store.iter_records(sampled_ids, with_vectors=True, with_text=False)
        }

        for position in positions:
            record_id = ids[position]
            actual = read_back.get(record_id)
            if actual is None or not np.allclose(actual, written[position], atol=VERIFY_ATOL):
                self.queue.mark([record_id], ItemState.FAILED, error_code="RB-E6002")
                raise MigrationInterrupted(
                    f"Record {record_id!r} does not match what was written to it.",
                    hint=(
                        "The store accepted the write but did not store it. This "
                        "is the failure mode that read-back verification exists "
                        "to catch. The job is stopped so nothing further is lost; "
                        "the shadow copy still holds the originals."
                    ),
                    context={"job_id": self.job_id, "record_id": record_id},
                )

    def _keep_for_durability(self, ids: list[str], written: FloatArray) -> None:
        """Reservoir-sample this batch into the end-of-job check.

        Reservoir rather than "the last batch": a store that stops persisting
        partway through would still look healthy in its final batch, which is
        the one still warm in every cache.
        """
        for offset, record_id in enumerate(ids):
            self._seen += 1
            if len(self._kept) < DURABILITY_SAMPLE:
                self._kept.append((record_id, written[offset].copy()))
                continue
            slot = int(self._rng.integers(self._seen))
            if slot < DURABILITY_SAMPLE:
                self._kept[slot] = (record_id, written[offset].copy())

    def durability_sample_ids(self) -> list[str]:
        """The records the end-of-job check will re-read on a fresh connection.

        Spread across the whole run rather than drawn from the final batch, so
        it says something about the job and not about the store's cache.
        """
        return [record_id for record_id, _ in self._kept]

    def verify_durability(self) -> int:
        """Reopen the store and check the sample is still there.

        The per-batch check cannot see this failure. Chroma caches a client per
        path within a process; a SQLite transaction can go unflushed; a remote
        store can acknowledge a write it later drops. In each case the handle
        that wrote is the last one that will tell you the truth, so this opens a
        new one.

        The engine's own handle is released first and reopened afterwards.
        Qdrant's local mode takes an exclusive lock on its storage folder, so
        "open a second connection" is not something every backend allows — and
        the backends that refuse are not the ones to skip the check on. The
        engine stays usable either way: rolling back straight after a completed
        run is a normal thing to want.

        A no-op when the engine was handed a store object rather than a URI —
        there is nothing to reopen — and when nothing was written.

        Returns:
            How many records were re-checked.

        Raises:
            WritesDidNotSurvive: If a record is missing or does not match.
        """
        if not self.store_uri or not self._kept:
            return 0

        from rebasis.errors import WritesDidNotSurvive
        from rebasis.store import open_store

        expected = dict(self._kept)
        _close(self.store)
        try:
            store = open_store(self.store_uri)
            try:
                found = {
                    record.id: record.vector
                    for record in store.iter_records(
                        list(expected), with_vectors=True, with_text=False
                    )
                }
            finally:
                _close(store)
        finally:
            self.store = open_store(self.store_uri)

        for record_id, vector in expected.items():
            actual = found.get(record_id)
            if actual is None or not np.allclose(actual, vector, atol=VERIFY_ATOL):
                raise WritesDidNotSurvive(
                    f"Record {record_id!r} does not hold what the migration wrote to it.",
                    hint=(
                        "Every batch verified at write time, so the store accepted "
                        "these writes and a fresh connection cannot see them. The "
                        "shadow copy still holds the originals: "
                        f"`rebasis rollback {self.job_id}` restores them."
                    ),
                    context={"job_id": self.job_id, "record_id": record_id},
                )

        log.info(
            Events.MIGRATE_DURABILITY_VERIFIED,
            job_id=self.job_id,
            count=len(expected),
        )
        if self.audit:
            self.audit.write(
                Events.MIGRATE_DURABILITY_VERIFIED,
                inputs={"job_id": self.job_id},
                outputs={"job_id": self.job_id, "count": len(expected)},
                subject=self.job_id,
            )
        return len(expected)

    def _pressure(self, batch_index: int) -> str:
        """Why the monitor says to stop after this batch, or ``""``.

        The monitor answers two questions in one call — stop, and throttle — and
        only the first ends the run. A throttle is logged and the loop carries
        on with a smaller batch.
        """
        should_pause, reason = self.monitor.check()
        if should_pause:
            return reason
        if reason:
            log.warning(
                Events.MIGRATE_BATCH_THROTTLED,
                job_id=self.job_id,
                batch_index=batch_index,
                peak_rss_bytes=self.monitor.peak_rss,
            )
        return ""

    # ── continuous refit ──────────────────────────────────────────────

    def _refit_if_due(self, processed: int, last_refit_at: int) -> int:
        """Attempt a refit if enough has been migrated, and say when it last ran."""
        if not self.refit.due(processed, last_refit_at):
            return last_refit_at
        self._attempt_refit()
        return processed

    def _refit_is_possible(self) -> str:
        """Why a refit cannot be attempted, or ``""`` when it can.

        Checked once, before the first attempt, so a job configured for a refit
        it can never perform says so at the start rather than at the first
        checkpoint an hour in.
        """
        if not self.refit.enabled:
            return "refitting is off"
        if self.embedder is None:
            return "no embedder was given, and every real pair costs a document re-embedded"
        if self.adapter_root is None or self.profiles is None:
            return "there is nowhere to write an adopted adapter, so a resume would lose it"
        if not self.store.capabilities.can_read_text:
            return f"{self.store.capabilities.name} does not return document text"
        return ""

    def _settle_refit(self) -> None:
        """Turn refitting off, once and loudly, when it cannot be done.

        At the start of the run rather than at the first checkpoint: a job
        configured to refit and unable to should not discover that an hour in,
        by which point the alternative — restart with an embedder — costs
        everything already migrated.
        """
        if not self.refit.enabled:
            return
        blocked = self._refit_is_possible()
        if not blocked:
            return
        log.warning(
            Events.MIGRATE_ADAPTER_REFITTED,
            job_id=self.job_id,
            adapter_type=self.adapter.type_name,
            error_code="refit_unavailable",
        )
        self._audit_refit(adopted=False, reason=blocked, scores=(0.0, 0.0), pairs=0)
        self.refit.enabled = False

    def _accumulate_pairs(self) -> tuple[FloatArray, FloatArray, int]:
        """Draw pairs from what is left to migrate, and re-embed them.

        Source vectors come from the store rather than from the shadow, because
        these records have not been migrated yet: what the index holds for them
        *is* the old model's vector. Target vectors are computed here — there is
        no other way to get one, since a migrated record carries the adapter's
        own image rather than the new model's output.

        Returns the pairs and how many records were sampled to get them, so a
        store that returns text for only some of them can be reported rather
        than silently producing a small fit.
        """
        if self.embedder is None:  # pragma: no cover - `_refit_is_possible` guards it
            msg = "a refit was attempted with no embedder"
            raise RuntimeError(msg)
        ids = self.queue.sample_pending(self.refit.sample_size, self._rng)
        if not ids:
            return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32), 0

        usable = [
            (record.vector, record.text)
            for record in self.store.iter_records(ids, with_vectors=True, with_text=True)
            if record.vector is not None and record.text
        ]
        if not usable:
            return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32), len(ids)

        source = l2_normalize(np.vstack([vector for vector, _ in usable]))
        target = l2_normalize(self.embedder.encode([text for _, text in usable], kind="document"))
        return source, target, len(ids)

    def _attempt_refit(self) -> None:
        """Refit on the accumulated pairs and adopt the result if it wins.

        Never fatal. A refit is an optimisation on top of a migration, and a
        migration that stopped because an optional improvement could not be
        computed would be worse than one that carried on with the adapter it
        already had.
        """
        self._refits += 1
        try:
            source, target, sampled = self._accumulate_pairs()
        except Exception as exc:  # noqa: BLE001 - an optional step must not end the job
            log.warning(
                Events.MIGRATE_ADAPTER_REFITTED,
                job_id=self.job_id,
                adapter_type=self.adapter.type_name,
                error_code=type(exc).__name__,
            )
            return

        if source.shape[0] < self.refit.min_pairs:
            log.info(
                Events.MIGRATE_ADAPTER_REFITTED,
                job_id=self.job_id,
                adapter_type=self.adapter.type_name,
                count=int(source.shape[0]),
            )
            self._audit_refit(
                adopted=False,
                reason=(
                    f"{source.shape[0]} usable pairs from {sampled} sampled records, "
                    f"below the {self.refit.min_pairs} a refit needs"
                ),
                scores=(0.0, 0.0),
                pairs=int(source.shape[0]),
            )
            return

        decision = consider_refit(
            self.adapter, src=source, dst=target, policy=self.refit, job_id=self.job_id
        )
        if not decision.adopted or decision.adapter is None:
            self._audit_refit(
                adopted=False,
                reason=decision.reason,
                scores=(decision.current_score, decision.candidate_score),
                pairs=decision.n_pairs,
            )
            return

        self.adapter = decision.adapter
        self._persist_adapter(decision.adapter)
        self._audit_refit(
            adopted=True,
            reason=decision.reason,
            scores=(decision.current_score, decision.candidate_score),
            pairs=decision.n_pairs,
        )

    def _persist_adapter(self, adapter: BaseAdapter) -> None:
        """Write the adopted adapter and point the job at it.

        Without this a `--resume` would reload the file `migrate` was started
        with and silently give back whatever the refit gained — which on a
        corpus that drifted is the largest single number this feature produces.
        The job row is what `--resume` reads, so updating it is what makes the
        adoption survive the process.
        """
        if self.adapter_root is None or self.profiles is None:  # pragma: no cover - guarded
            msg = "a refit was adopted with nowhere to write it"
            raise RuntimeError(msg)
        from rebasis.core.serialization import save_adapter

        old_profile, new_profile = self.profiles
        path = self.adapter_root / f"{self.job_id}-refit-{self._refits}.rbs"
        save_adapter(
            adapter,
            path,
            direction="old_to_new",
            old_profile=old_profile,
            new_profile=new_profile,
        )
        self.adapter_path = str(path)
        with self.db.transaction() as connection:
            connection.execute(
                "UPDATE jobs SET adapter_path = ?, adapter_type = ?, updated_utc = ? "
                "WHERE job_id = ?",
                (self.adapter_path, adapter.type_name, _now(), self.job_id),
            )

    def _audit_refit(
        self, *, adopted: bool, reason: str, scores: tuple[float, float], pairs: int
    ) -> None:
        """Record the attempt, adopted or not.

        Both outcomes, because "the refit was considered and declined" is the
        answer to "why did this job not improve", and an audit trail that only
        recorded successes could not give it.
        """
        if self.audit is None:
            return
        current, candidate = scores
        self.audit.write(
            Events.MIGRATE_ADAPTER_REFITTED,
            inputs={
                "job_id": self.job_id,
                "attempt": self._refits,
                "count": pairs,
                "min_improvement": self.refit.min_improvement,
            },
            outputs={
                "job_id": self.job_id,
                "adopted": adopted,
                "reason": reason,
                "current_score": round(current, 4),
                "candidate_score": round(candidate, 4),
                "adapter_path": self.adapter_path if adopted else "",
            },
            subject=self.job_id,
        )

    def _reason_to_stop(self) -> str:
        """Why this run should stop before starting another batch, or ``""``.

        Both questions are asked *before* the batch rather than after it, so a
        request made while one was in flight is honoured at the next boundary
        instead of one batch later.

        The pause request is a primary-key lookup against a local SQLite file,
        and the batch it guards spends its time in the store and the embedding
        model — it does not register against that.
        """
        if stop_requested():
            # Ahead of the manifest lookup on purpose: a supervisor that has sent
            # SIGTERM is counting down to SIGKILL, and the cheapest possible
            # check is the one that should decide.
            return f"{stop_signal_name() or 'a signal'} asked this process to stop"
        if pause_requested(self.db, self.job_id):
            return "a pause was requested"
        power = power_state(power_aware=self.power_aware)
        return power.reason if power.should_pause else ""

    def _finish(self, pause_reason: str, duration: float, processed: int) -> JobState:
        stats = self.queue.stats()
        # However this run ended, no request is outstanding any more: either it
        # was honoured, or the job stopped for some other reason and a request
        # aimed at a run that is over would only pause the next one. That keeps
        # `pause_requested` meaning exactly one thing — *asked, and still
        # running* — which is what makes it worth showing in `status`.
        clear_pause_request(self.db, self.job_id)
        if pause_reason:
            state = JobState.PAUSED
            log.warning(
                Events.MIGRATE_JOB_PAUSED,
                job_id=self.job_id,
                state=str(state),
                error_code=None,
            )
        elif stats.remaining == 0:
            state = JobState.COMPLETED
            log.info(
                Events.MIGRATE_JOB_COMPLETED,
                job_id=self.job_id,
                count=stats.done,
                duration_ms=round(duration * 1000, 1),
            )
        else:
            state = JobState.PAUSED

        set_job_state(self.db, self.job_id, state)
        self._warn_if_mixed(stats)

        if self.audit:
            event = (
                Events.MIGRATE_JOB_COMPLETED
                if state is JobState.COMPLETED
                else Events.MIGRATE_JOB_PAUSED
            )
            self.audit.write(
                event,
                inputs={"job_id": self.job_id},
                outputs={
                    "job_id": self.job_id,
                    "state": str(state),
                    "count": processed,
                    "duration_ms": round(duration * 1000, 1),
                    **self.monitor.summary(),
                },
                subject=self.job_id,
            )
        return state

    def _warn_if_mixed(self, stats: QueueStats) -> None:
        """Say so when the run leaves the index holding two embedding spaces.

        Some records now carry the new model's geometry and some still carry the
        old one, and there is no query that is correct against both: a bridged
        query mis-scores the migrated half, an unbridged one mis-scores the
        rest. Nothing raises, nothing is missing, and the ranking is wrong — so
        the only thing standing between a user and quietly bad results is being
        told.

        Failed records count as un-migrated. They are still in the old space,
        which is what makes the index mixed; leaving them out would report a
        clean index that is not one.
        """
        unmigrated = stats.pending + stats.shadowed + stats.failed
        if stats.done == 0 or unmigrated == 0:
            return
        log.warning(
            Events.MIGRATE_INDEX_MIXED,
            job_id=self.job_id,
            count=unmigrated,
            state=str(JobState.PAUSED),
            store_backend=self.store.capabilities.name,
        )

    def rollback(self, *, batch_size: int = 1024) -> int:
        """Restore the original vectors from the shadow copy.

        Bit-identical when the shadow was written at float32, which is the
        default precisely so that rollback is bit-identical.

        Bit-identical *to what was read*, which is the part that stops being a
        rebasis guarantee on a store that quantizes. The shadow holds what
        ``iter_records`` returned, and a store that keeps compressed codes
        returns a value decoded from them; so on such a store this restores the
        state the migration replaced, not the vectors the embedding model
        produced. ``StoreCapabilities.quantized`` is which of the two it is, and
        `migrate` says so before it writes anything.

        Raises:
            ShadowMissing: When there is no shadow — the job ran with
                ``--no-keep-original``, or ``gc`` removed it.
        """
        restored = 0
        for ids, vectors in self.shadow.iter_batches(batch_size):
            self.store.upsert_vectors(ids, vectors)
            self.queue.mark(ids, ItemState.PENDING)
            restored += len(ids)

        set_job_state(self.db, self.job_id, JobState.ROLLED_BACK)
        log.info(Events.STORAGE_ROLLBACK_COMPLETED, job_id=self.job_id, count=restored)

        if self.audit:
            self.audit.write(
                Events.STORAGE_ROLLBACK_COMPLETED,
                inputs={"job_id": self.job_id},
                outputs={"count": restored, "state": str(JobState.ROLLED_BACK)},
                subject=self.job_id,
            )
        return restored


def _close(store: VectorStore) -> None:
    """Release a store handle, if this backend has one to release.

    Not every backend does — an in-memory store has nothing to close — so this
    asks rather than assuming, and never turns a close into the reason a
    completed migration reports failure.
    """
    closer = getattr(store, "close", None)
    if callable(closer):
        with contextlib.suppress(Exception):
            closer()


def _now() -> str:
    return datetime.datetime.now(tz=datetime.UTC).isoformat()
