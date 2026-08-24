"""Integrity verification.

A corrupt artefact must never load silently. An adapter whose weights have been
damaged does not crash — it just produces slightly worse results, which is the
hardest kind of failure to notice and the easiest to blame on the model.

Two levels, because the cost differs by orders of magnitude:

* **Fast path (default):** header parse plus shape, dtype and file size. Runs in
  microseconds, catches truncation and gross corruption.
* **Full path (``--verify``, and always before ``adapter upgrade``):** every
  tensor hash recomputed. Milliseconds for a few megabytes.

Shadow vector files are hashed **per block** rather than as a whole, so a single
damaged block does not invalidate an entire file and the affected record range
can be identified.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from rebasis.errors import IntegrityCheckFailed

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    import numpy as np

__all__ = [
    "BLOCK_SIZE",
    "hash_blocks",
    "hash_bytes",
    "hash_file",
    "verify_blocks",
    "verify_bytes",
]

#: Block size for shadow files. 64 MB balances hash count against the
#: granularity of "which records were affected".
BLOCK_SIZE = 64 * 1024 * 1024


def hash_bytes(data: bytes) -> str:
    """sha256 of a byte string."""
    return hashlib.sha256(data).hexdigest()


def hash_array(array: np.ndarray) -> str:
    """sha256 of an array's raw buffer.

    Hashes the contiguous buffer, so shape and dtype must be checked separately —
    two arrays with different shapes can share a buffer.
    """
    import numpy as np

    return hash_bytes(np.ascontiguousarray(array).tobytes())


def hash_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """sha256 of a file, read in chunks so large files never load in one piece."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_blocks(path: Path, *, block_size: int = BLOCK_SIZE) -> list[str]:
    """Per-block hashes of a file.

    Block granularity is what lets a damaged shadow file report *which records*
    are affected instead of being written off wholesale.
    """
    hashes: list[str] = []
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            hashes.append(hash_bytes(block))
    return hashes


def verify_bytes(data: bytes, expected: str, *, what: str = "artefact") -> None:
    """Check a byte string against its expected hash.

    Raises:
        IntegrityCheckFailed: On mismatch. There is no silent acceptance: a
            damaged artefact must fail loudly.
    """
    actual = hash_bytes(data)
    if actual != expected:
        raise IntegrityCheckFailed(
            f"Integrity check failed for {what}: content does not match its recorded hash.",
            hint=(
                "The file is corrupt. Restore it from a backup (`.bak.1` next to "
                "it) or refit the adapter."
            ),
            context={"count": len(data)},
        )


def verify_blocks(path: Path, expected: list[str], *, block_size: int = BLOCK_SIZE) -> list[int]:
    """Verify per-block hashes and return the indices of damaged blocks.

    Returns rather than raises: the caller needs to know *which* blocks failed in
    order to report the affected record range.
    """
    actual = hash_blocks(path, block_size=block_size)
    if len(actual) != len(expected):
        raise IntegrityCheckFailed(
            f"{path.name} has {len(actual)} blocks but {len(expected)} were recorded.",
            hint="The file was truncated or extended. Restore it from a backup.",
            context={"count": len(actual)},
        )
    return [i for i, (a, e) in enumerate(zip(actual, expected, strict=True)) if a != e]


def hash_state_dict(state: Mapping[str, np.ndarray]) -> dict[str, str]:
    """Per-tensor hashes for an adapter's weights, as recorded in its manifest."""
    return {name: hash_array(array) for name, array in sorted(state.items())}


def verify_state_dict(state: Mapping[str, np.ndarray], expected: Mapping[str, str]) -> None:
    """Verify every tensor in an adapter's weights.

    Raises:
        IntegrityCheckFailed: On the first mismatch, or when the tensor set
            itself differs from what was recorded.
    """
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    if missing or extra:
        raise IntegrityCheckFailed(
            "The adapter's tensor set does not match its manifest.",
            hint="The file is corrupt or was written by an incompatible version.",
            context={"count": len(state)},
        )
    for name, array in sorted(state.items()):
        actual = hash_array(array)
        if actual != expected[name]:
            raise IntegrityCheckFailed(
                f"Integrity check failed for tensor {name!r}.",
                hint=(
                    "The adapter file is corrupt. Restore it from a backup or "
                    "refit it; `rebasis fit` is cheap relative to a reindex."
                ),
                context={"count": int(array.size)},
            )


def iter_blocks(path: Path, *, block_size: int = BLOCK_SIZE) -> Iterator[bytes]:
    """Iterate a file in blocks without loading it whole."""
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            yield block
