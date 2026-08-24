"""The audit hash chain.

Each record carries the hash of the one before it, so deleting or altering a
record is detectable.

**What this is, precisely.** It is **tamper-evident, not tamper-proof.** It is a
local file, and its owner can regenerate the whole chain if they choose. The
purpose is to catch **accidental corruption and silent data loss**, not to
withstand a determined local attacker. Claiming "tamper-proof" would be false,
and a security property claimed but not delivered is worse than one never
claimed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rebasis.audit.record import GENESIS_HASH, AuditRecord

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["ChainVerification", "verify_chain"]


@dataclass(slots=True)
class ChainVerification:
    """The outcome of verifying a chain."""

    valid: bool
    records_checked: int
    broken_at: int | None = None
    reason: str = ""
    gaps: list[int] = field(default_factory=list)

    def describe(self) -> str:
        """A one-line summary for the CLI."""
        if self.valid:
            suffix = f" ({len(self.gaps)} recorded gaps)" if self.gaps else ""
            return f"{self.records_checked} records verified{suffix}"
        return f"broken at seq {self.broken_at}: {self.reason}"


def verify_chain(
    records: Sequence[AuditRecord], *, gap_seqs: Sequence[int] = ()
) -> ChainVerification:
    """Verify a chain end to end.

    Two checks per record, and both are needed:

    1. **Content hash.** Catches a record that was edited in place.
    2. **Link.** Catches a record that was deleted — the following record's
       ``prev_hash`` no longer matches its predecessor.

    Args:
        records: Records ordered by ``seq``.
        gap_seqs: Sequence numbers where a deliberate gap was recorded by
            ``audit prune``. Pruning is allowed but writes its own record
            describing the gap, so the chain stays verifiable across it.
    """
    if not records:
        return ChainVerification(valid=True, records_checked=0)

    known_gaps = set(gap_seqs)
    expected_prev = GENESIS_HASH

    for index, record in enumerate(records):
        if not record.verify():
            return ChainVerification(
                valid=False,
                records_checked=index,
                broken_at=record.seq,
                reason="content does not match its recorded hash — the record was altered",
            )

        if record.prev_hash != expected_prev and record.seq not in known_gaps:
            return ChainVerification(
                valid=False,
                records_checked=index,
                broken_at=record.seq,
                reason=(
                    "the link to the previous record is broken — a record was "
                    "deleted, or the chain was written by two processes at once"
                ),
            )
        expected_prev = record.hash

    return ChainVerification(
        valid=True,
        records_checked=len(records),
        gaps=sorted(known_gaps),
    )
