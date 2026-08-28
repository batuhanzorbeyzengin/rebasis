"""Error hierarchy and stable error codes.

In an open-source tool the error message is the most-read piece of
documentation. The goal is that **every error carries a code, a cause and a
next step**.

Three rules govern this module:

1. **Codes are stable.** ``RB-E3002`` means the same thing forever. Users google
   them, and the ``rebasis.errors`` metric is labelled by ``error_code``, so
   dashboards group by them. A code is never removed or repurposed — only marked
   deprecated, because user scripts depend on it.
2. **Third-party exceptions never cross a module boundary.** Each backend
   catches its own library's exception, converts it to a :class:`RebasisError`
   subclass, and keeps the original as ``__cause__``. Contract tests enforce
   this.
3. **``transient`` decides retry eligibility.** Only transient errors are
   retried, with exponential backoff and jitter, at most three times. Permanent
   errors are never retried — a silently retried permanent failure is the most
   common cause of undiagnosable slowness.

This module sits at the bottom of the layer contract and depends on nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "AdapterError",
    "AdapterSchemaVersion",
    "AtomicWriteFailed",
    "BackendMismatch",
    "BackendUnavailable",
    "BadThreshold",
    "CapabilityMissing",
    "CollectionNotFound",
    "ComputeError",
    "ConfigError",
    "DeviceOutOfMemory",
    "DeviceUnavailable",
    "EmbedError",
    "EmbeddingDimensionMismatch",
    "FitFailed",
    "GroundTruthUnavailable",
    "IncompatibleAdapter",
    "InsufficientDiskSpace",
    "InsufficientSamples",
    "IntegrityCheckFailed",
    "InvalidStoreURI",
    "ItemFailed",
    "LeakageDetected",
    "LockHeld",
    "MalformedQueryLog",
    "ManifestLocked",
    "MigrateError",
    "MigrationInterrupted",
    "MissingDependency",
    "ProbeError",
    "ProfileMismatch",
    "RebasisError",
    "SerializationError",
    "ShadowMissing",
    "StorageError",
    "StoreDimensionMismatch",
    "StoreError",
    "StoreUnsupported",
    "StoreWriteFailed",
    "UnknownModelProfile",
    "UnsupportedDtype",
    "UserAbort",
    "WritesDidNotSurvive",
]

# Exit codes are a contract for script users: changing one is a breaking change.
EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_USAGE = 2
EXIT_DOMAIN = 3
EXIT_ABORTED = 130


class RebasisError(Exception):
    """Base class for every error rebasis raises.

    Args:
        message: What happened, in one sentence, in the user's terms.
        hint: What to do next. Omit only when there is genuinely nothing
            actionable — an absent hint is a defect, not a style choice.
        context: Structured fields for logs and the error panel. Must contain
            **no document text, vectors, query text, file paths or
            credentials**; ids, counts and dimensions only.
        cause: The originating third-party exception, preserved as ``__cause__``.
    """

    code: ClassVar[str] = "RB-E0000"
    transient: ClassVar[bool] = False
    exit_code: ClassVar[int] = EXIT_DOMAIN

    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        context: Mapping[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.context: dict[str, Any] = dict(context or {})
        if cause is not None:
            self.__cause__ = cause

    def render(self) -> str:
        """Human-readable rendering: code, message, context, next step."""
        lines = [f"[{self.code}] {self.message}"]
        if self.context:
            details = " ".join(f"{k}={v!r}" for k, v in sorted(self.context.items()))
            lines.append(f"  context: {details}")
        if self.hint:
            lines.append(f"  hint: {self.hint}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.message


# ── RB-E1xxx — configuration ────────────────────────────────────────────
# Usage and configuration problems exit with 2, not 3: they mean the user
# invoked the tool wrongly, not that the domain operation failed.


class ConfigError(RebasisError):
    """Invalid configuration or invocation."""

    code = "RB-E1000"
    exit_code = EXIT_USAGE


class InvalidStoreURI(ConfigError):
    """The store URI could not be parsed or names an unknown backend."""

    code = "RB-E1001"


class MissingDependency(ConfigError):
    """An optional dependency is required for the requested operation."""

    code = "RB-E1002"


class BadThreshold(ConfigError):
    """A decision threshold is outside its valid range."""

    code = "RB-E1003"


class MalformedQueryLog(ConfigError):
    """A query log file could not be read as JSONL."""

    code = "RB-E1004"


# ── RB-E2xxx — embedding ────────────────────────────────────────────────


class EmbedError(RebasisError):
    """Embedding backend failure."""

    code = "RB-E2000"


class BackendUnavailable(EmbedError):
    """The embedding backend is temporarily unreachable."""

    code = "RB-E2001"
    transient = True


class ProfileMismatch(EmbedError):
    """The encoding profile does not match the one the index was built with."""

    code = "RB-E2002"


class UnknownModelProfile(EmbedError):
    """No encoding profile is known for this model and none was supplied."""

    code = "RB-E2003"


class EmbeddingDimensionMismatch(EmbedError):
    """Vector dimensionality does not match what the collection expects."""

    code = "RB-E2004"


# ── RB-E3xxx — vector store ─────────────────────────────────────────────


class StoreError(RebasisError):
    """Vector store failure."""

    code = "RB-E3000"


class StoreUnsupported(StoreError):
    """No backend is registered for this store type."""

    code = "RB-E3001"


class CapabilityMissing(StoreError):
    """The store does not support an operation this command requires."""

    code = "RB-E3002"


class CollectionNotFound(StoreError):
    """The named collection does not exist in the store."""

    code = "RB-E3003"


class StoreWriteFailed(StoreError):
    """A write to the store failed and may succeed on retry."""

    code = "RB-E3004"
    transient = True


class StoreDimensionMismatch(StoreError):
    """The vectors to write do not match the collection's dimension."""

    code = "RB-E3005"


# ── RB-E4xxx — adapter ──────────────────────────────────────────────────


class AdapterError(RebasisError):
    """Adapter fitting, loading or serialisation failure."""

    code = "RB-E4000"


class FitFailed(AdapterError):
    """The adapter could not be fitted (singular system, degenerate input)."""

    code = "RB-E4001"


class IncompatibleAdapter(AdapterError):
    """Adapter fingerprint does not match this index or these models."""

    code = "RB-E4002"


class AdapterSchemaVersion(AdapterError):
    """The ``.rbs`` schema version cannot be read by this release.

    The compatibility promise is explicit: a release producing schema *N* must
    read *N* and *N-1*. Anything older raises this error and points at
    ``rebasis adapter upgrade``. Silent failure or best-effort loading is
    forbidden.
    """

    code = "RB-E4003"


class SerializationError(AdapterError):
    """The adapter file is corrupt — integrity hash mismatch."""

    code = "RB-E4004"


# ── RB-E5xxx — probe ────────────────────────────────────────────────────


class ProbeError(RebasisError):
    """Diagnosis failure."""

    code = "RB-E5000"


class InsufficientSamples(ProbeError):
    """Too few samples to produce a trustworthy measurement."""

    code = "RB-E5001"


class GroundTruthUnavailable(ProbeError):
    """Ground truth could not be established from held-out chunks, a query log or synthesis."""

    code = "RB-E5002"


class LeakageDetected(ProbeError):
    """The fit set and the query set overlap.

    Raised at runtime, not only in tests. If this leaks through even once, every
    ARR number produced becomes meaningless — so it is a hard failure rather
    than a warning.
    """

    code = "RB-E5003"


# ── RB-E6xxx — migration ────────────────────────────────────────────────


class MigrateError(RebasisError):
    """Migration engine failure."""

    code = "RB-E6000"


class MigrationInterrupted(MigrateError):
    """The migration stopped before completion; resume from the checkpoint."""

    code = "RB-E6001"


class ItemFailed(MigrateError):
    """A single record failed to migrate."""

    code = "RB-E6002"


class ManifestLocked(MigrateError):
    """Another rebasis process holds the manifest lock."""

    code = "RB-E6003"


class InsufficientDiskSpace(MigrateError):
    """Not enough free space to complete the migration safely.

    Raised *before* the job starts. A migration that fills the disk halfway
    through is not an error to be handled — it is a design flaw to be prevented.
    """

    code = "RB-E6004"


class WritesDidNotSurvive(MigrateError):
    """The store accepted every write and a fresh connection cannot see them.

    Distinct from :class:`ItemFailed`, which is one record refused at write
    time. This is the whole job passing its per-batch checks and then not being
    there — a client that cached the writes, a transaction never committed, a
    file never flushed. It is detectable only by reopening the store, which is
    why it is checked once at the end rather than per batch.
    """

    code = "RB-E6005"


# ── RB-E7xxx — storage ──────────────────────────────────────────────────


class StorageError(RebasisError):
    """Durability or filesystem failure."""

    code = "RB-E7000"


class AtomicWriteFailed(StorageError):
    """An atomic write could not be completed; the target is untouched."""

    code = "RB-E7001"


class IntegrityCheckFailed(StorageError):
    """A stored artefact failed its integrity hash check."""

    code = "RB-E7002"


class ShadowMissing(StorageError):
    """The shadow copy needed to roll back this job is absent."""

    code = "RB-E7003"


class LockHeld(StorageError):
    """A file lock is held by another process."""

    code = "RB-E7004"


# ── RB-E8xxx — compute ──────────────────────────────────────────────────


class ComputeError(RebasisError):
    """Compute backend or device failure."""

    code = "RB-E8000"


class DeviceUnavailable(ComputeError):
    """The requested device is not available on this machine."""

    code = "RB-E8001"


class DeviceOutOfMemory(ComputeError):
    """The device ran out of memory; a smaller batch may succeed.

    Transient by design: OOM must never kill a job. The batch size is halved,
    the attempt is retried, and after three failures the work falls back to CPU
    and continues.
    """

    code = "RB-E8002"
    transient = True


class UnsupportedDtype(ComputeError):
    """The device does not support the requested dtype."""

    code = "RB-E8003"


class BackendMismatch(ComputeError):
    """Two compute backends disagree beyond the tolerance the device-parity tests allow."""

    code = "RB-E8004"


# ── RB-E9xxx — user ─────────────────────────────────────────────────────


class UserAbort(RebasisError):
    """The user interrupted the operation (Ctrl-C)."""

    code = "RB-E9000"
    exit_code = EXIT_ABORTED
