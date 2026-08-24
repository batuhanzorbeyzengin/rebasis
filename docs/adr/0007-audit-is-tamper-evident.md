# 7. The audit trail is tamper-evident, not tamper-proof

**Status:** Accepted · **Date:** 2026-08 · **Evidence:** `tests/unit/test_audit_chain.py`

## Decision

Audit records are chained by hash: each carries the hash of its predecessor.
The property this gives is **tamper-evidence** — an alteration is detectable —
and not tamper-proofness. The distinction is documented everywhere the chain is
mentioned, including in the output of `rebasis audit verify`.

## Context

A hash chain looks like a stronger guarantee than it is. Anyone who can write
the file can also recompute every subsequent hash, and rebasis stores the chain
in a local SQLite file the user owns. There is no signing key, no append-only
medium, no external anchor.

Describing that as "tamper-proof" would be a security claim the implementation
does not support, and the people most likely to believe it are the ones with a
compliance requirement.

## What it does buy

- **Accidental corruption is caught.** A truncated write, a failed disk, a
  half-restored backup all break the chain.
- **Partial edits are caught.** Editing one record without recomputing the rest
  is what a person editing a database by hand actually does.
- **The boundary is visible.** `verify` reports the first sequence number where
  the chain breaks, which is where to look.

## What it does not

- It does not stop the file's owner from rewriting the whole chain.
- It does not prove *when* a record was written. The timestamp is self-reported.
- It does not survive deletion of the file.

## Consequences

- The docstrings, the CLI output and the docs all say "tamper-evident, not
  tamper-proof". Repetition is deliberate: whichever surface a reader meets
  first has to carry the qualification.
- If tamper-proofness is ever needed, the honest routes are signing records with
  a key rebasis does not hold, or anchoring the chain head externally. Both are
  real work and neither is implied by what exists.

## Alternative

**Say nothing and let the chain imply the stronger property.** Rejected. A
security property that is inferred rather than stated is one nobody tested, and
the first person to rely on it will be the one who needed it most.
