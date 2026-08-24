"""Audit record schema.

**A log and an audit trail are different things, and conflating them is a common
mistake.**

* A **log** is for debugging. It may be lost, it may be sampled, its format may
  change between releases.
* An **audit record** is for *reconstructing a decision*. It is lossless, never
  sampled, schema-versioned, and never deleted.

rebasis's output is a **decision** — "no reindex needed", or "you will lose 29%
of your recall". That decision has to remain defensible six months later.
Auditability here is not decoration; it is the product.

**Redaction applies to audit records too.** ``inputs_json`` contains no document
text: chunk ids, hashes and counts only. Being able to reconstruct a decision
does not require the content, only the parameters — so auditability costs nothing
in privacy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["GENESIS_HASH", "AuditRecord", "canonical_json", "record_hash"]

#: ``prev_hash`` of the first record. A fixed value rather than an empty string,
#: so a chain that starts correctly is distinguishable from one whose first row
#: lost its link.
GENESIS_HASH = "0" * 64


def canonical_json(payload: Any) -> str:
    """Deterministic JSON, so the same content always hashes identically.

    Sorted keys and no incidental whitespace: without both, a dictionary built in
    a different order would produce a different hash for identical data and break
    the chain for no reason.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """One entry in the audit trail."""

    ts_utc: str
    run_id: str
    actor: str
    action: str
    inputs: Mapping[str, Any]
    outputs: Mapping[str, Any]
    rebasis_version: str
    env: Mapping[str, Any]
    subject: str | None = None
    seq: int | None = None
    prev_hash: str = GENESIS_HASH
    hash: str = ""

    def payload(self) -> dict[str, Any]:
        """The fields the hash covers.

        ``seq`` is excluded on purpose: it is assigned by SQLite on insert, so
        including it would make the hash unknowable before the write.
        """
        return {
            "ts_utc": self.ts_utc,
            "run_id": self.run_id,
            "actor": self.actor,
            "action": self.action,
            "subject": self.subject,
            "inputs": dict(self.inputs),
            "outputs": dict(self.outputs),
            "rebasis_version": self.rebasis_version,
            "env": dict(self.env),
            "prev_hash": self.prev_hash,
        }

    def compute_hash(self) -> str:
        """sha256 over the canonical payload."""
        return hashlib.sha256(canonical_json(self.payload()).encode("utf-8")).hexdigest()

    def sealed(self, prev_hash: str) -> AuditRecord:
        """Return a copy linked to ``prev_hash`` and hashed."""
        from dataclasses import replace

        linked = replace(self, prev_hash=prev_hash)
        return replace(linked, hash=linked.compute_hash())

    def verify(self) -> bool:
        """Whether the stored hash matches the content."""
        return bool(self.hash) and self.hash == self.compute_hash()


def record_hash(record: AuditRecord) -> str:
    """Convenience wrapper over :meth:`AuditRecord.compute_hash`."""
    return record.compute_hash()


@dataclass(slots=True)
class EnvironmentSnapshot:
    """What the environment looked like when a decision was taken.

    Without this, a replay that produces different numbers cannot distinguish a
    regression from a change of hardware. Results may legitimately differ between
    CPU and GPU even with identical seeds, so the comparison has to know which it
    is looking at.
    """

    python: str = ""
    platform: str = ""
    numpy: str = ""
    scipy: str = ""
    torch: str | None = None
    device: str = "cpu"
    tf32_enabled: bool | None = None
    deterministic_mode: bool = False
    blas_threads: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def capture(cls) -> EnvironmentSnapshot:
        """Snapshot the current environment."""
        import platform as platform_module
        import sys

        import numpy as np
        import scipy

        from rebasis.compute import blas_info, numerical_fingerprint, resolve_device

        fingerprint = numerical_fingerprint()
        _, threads = blas_info()
        return cls(
            python=sys.version.split()[0],
            platform=platform_module.platform(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            torch=fingerprint.get("torch_version"),
            device=str(resolve_device("auto")),
            tf32_enabled=fingerprint.get("tf32_matmul"),
            deterministic_mode=bool(fingerprint.get("deterministic_mode", False)),
            blas_threads=threads,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        from dataclasses import asdict

        return asdict(self)
