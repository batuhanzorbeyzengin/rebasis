"""Shadow copies, and the rollback they enable.

**The problem.** An in-place upsert overwrites the old vector. If a migration is
interrupted, or the user simply does not like the result, there is no way back —
short of running the old model over the corpus again, which is precisely the cost
the tool exists to avoid.

**The decision: ``--keep-original`` defaults to ON.** Before a record is
migrated, its previous vector is appended to a block-hashed file under
``.rebasis/shadow/<job_id>/``. ``rebasis rollback <job_id>`` reads it back and
restores the index.

**What "restored" is worth, exactly.** The shadow holds the original bytes: at
the default float32 precision it is bit-identical to what was read, and that
half of the guarantee is rebasis'. The other half is not. Writing those bytes
back goes through the store's own upsert, and a store that normalises on write
returns them one unit in the last place away — measured at ~3e-08 for Chroma in
``hnsw:space=cosine``, with rebasis nowhere in the test. So: bit-identical at
the shadow, and identical to within the store's own float32 round trip once it
lands. A store that does not normalise (Chroma at ``l2``) round-trips exactly.

The cost is ``N × d × 4`` bytes, temporarily — 1.87 GB for 487k records at 1024
dimensions. That is shown in the pre-check before anything starts, and it is the
right default: a couple of gigabytes of temporary disk is far cheaper than
irreversibly damaging an index somebody built over months. A user who wants the
speed can ask for it explicitly.

Shadow files are **never deleted automatically**. ``rebasis gc`` asks for
confirmation, and the deletion itself writes an audit record.

**Note the asymmetry that makes bridging free.** The bridge phase needs no shadow
at all, because the index is never touched — an unremarked benefit of making
``query_to_old`` the default direction.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rebasis.errors import ShadowMissing
from rebasis.observability import Events, get_logger
from rebasis.storage.atomic import atomic_write_json
from rebasis.storage.integrity import hash_bytes
from rebasis.types import FloatArray, as_float32

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

__all__ = ["ShadowManifest", "ShadowStore"]

log = get_logger(__name__)

_MANIFEST = "shadow.json"
_VECTORS = "vectors.bin"
_IDS = "ids.json"


@dataclass(frozen=True, slots=True)
class ShadowSegment:
    """One appended batch: how many records it holds and its hash.

    Segments rather than fixed-size blocks, because a migration appends in
    batches whose size adapts to memory pressure. Hashing what was actually
    written keeps verification exact and still bounds the damage a corrupt region
    can do to the records inside that segment.
    """

    count: int
    hash: str


@dataclass(slots=True)
class ShadowManifest:
    """What a shadow file contains and how to verify it."""

    job_id: str
    record_count: int
    dim: int
    precision: str
    segments: list[ShadowSegment]
    created_utc: str

    @property
    def bytes_per_vector(self) -> int:
        """Storage per vector, given the precision."""
        return self.dim * (2 if self.precision == "float16" else 4)


class ShadowStore:
    """Append-only storage of the vectors a migration is about to overwrite."""

    def __init__(self, root: Path | str, job_id: str, *, precision: str = "float32") -> None:
        self.job_id = job_id
        self.directory = Path(root) / job_id
        self.precision = precision
        self._dtype = np.float16 if precision == "float16" else np.float32

    # ── writing ───────────────────────────────────────────────────────

    def append(self, ids: Sequence[str], vectors: FloatArray) -> ShadowManifest:
        """Append a batch of original vectors.

        **Append-only, and that is load-bearing.** A migration shadows one batch
        at a time, so rewriting the file per batch would leave only the last
        batch and silently destroy the ability to roll back the rest — which is
        the entire purpose of the shadow.

        The durability pattern is write-ahead: the data file is appended to
        directly (a crash mid-append leaves trailing bytes, which is harmless),
        and the manifest is then written **atomically**. The manifest is the
        authority on how many records are valid, so trailing garbage from an
        interrupted append is simply never read.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        matrix = as_float32(vectors)
        if matrix.shape[0] != len(ids):
            msg = f"{len(ids)} ids but {matrix.shape[0]} vectors"
            raise ValueError(msg)

        existing = self._read_manifest()
        if existing and existing.dim != matrix.shape[1]:
            msg = (
                f"shadow holds {existing.dim}-dimensional vectors but was given "
                f"{matrix.shape[1]}-dimensional ones"
            )
            raise ValueError(msg)

        stored = matrix.astype(self._dtype, copy=False)
        payload = np.ascontiguousarray(stored).tobytes()

        # Truncate to the recorded length first: a previous crash may have left
        # a partial append, and building on top of that would corrupt every
        # later offset.
        vectors_path = self.directory / _VECTORS
        valid_bytes = existing.record_count * existing.dim * self._itemsize if existing else 0
        with vectors_path.open("r+b" if vectors_path.exists() else "wb") as handle:
            handle.truncate(valid_bytes)
            handle.seek(valid_bytes)
            handle.write(payload)
            handle.flush()
            import os as _os

            _os.fsync(handle.fileno())

        segments = list(existing.segments) if existing else []
        segments.append(ShadowSegment(count=int(matrix.shape[0]), hash=hash_bytes(payload)))

        manifest = ShadowManifest(
            job_id=self.job_id,
            record_count=(existing.record_count if existing else 0) + int(matrix.shape[0]),
            dim=int(matrix.shape[1]),
            precision=self.precision,
            segments=segments,
            created_utc=(
                existing.created_utc
                if existing
                else datetime.datetime.now(tz=datetime.UTC).isoformat()
            ),
        )
        # Atomic, and last: until this lands, the appended bytes are invisible.
        atomic_write_json(self.directory / _MANIFEST, _manifest_to_json(manifest))

        all_ids = (self._read_ids() if existing else []) + [str(i) for i in ids]
        atomic_write_json(self.directory / _IDS, all_ids)

        log.info(Events.STORAGE_SHADOW_WRITTEN, job_id=self.job_id, count=len(ids))
        return manifest

    #: Retained for callers that write a shadow in one go.
    write = append

    # ── reading ───────────────────────────────────────────────────────

    def _read_manifest(self) -> ShadowManifest | None:
        """The manifest, or ``None`` when no shadow exists yet."""
        path = self.directory / _MANIFEST
        if not path.exists():
            return None
        return _manifest_from_json(json.loads(path.read_text()))

    def _read_ids(self) -> list[str]:
        path = self.directory / _IDS
        return list(json.loads(path.read_text())) if path.exists() else []

    @property
    def _itemsize(self) -> int:
        return 2 if self.precision == "float16" else 4

    def manifest(self) -> ShadowManifest:
        """Read the shadow manifest.

        Raises:
            ShadowMissing: When the shadow for this job does not exist — which is
                what a rollback of a job run with ``--no-keep-original`` hits.
        """
        path = self.directory / _MANIFEST
        if not path.exists():
            raise ShadowMissing(
                f"No shadow copy exists for job {self.job_id}.",
                hint=(
                    "Either the job ran with --no-keep-original, or the shadow was "
                    "removed by `rebasis gc`. Without it the original vectors "
                    "cannot be restored."
                ),
                context={"job_id": self.job_id},
            )
        return _manifest_from_json(json.loads(path.read_text()))

    def ids(self) -> list[str]:
        """The record ids this shadow covers, in file order."""
        return list(json.loads((self.directory / _IDS).read_text()))

    def read_vectors(self, *, verify: bool = True) -> FloatArray:
        """Read the stored vectors back.

        Memory-mapped: rollback must not need the whole file resident, for the
        same reason writing it does not.
        """
        manifest = self.manifest()
        path = self.directory / _VECTORS
        stored = np.memmap(
            path,
            dtype=self._dtype,
            mode="r",
            shape=(manifest.record_count, manifest.dim),
        )
        if verify:
            self._verify_segments(manifest, stored)
        return as_float32(np.asarray(stored))

    def _verify_segments(self, manifest: ShadowManifest, stored: np.memmap) -> None:
        """Check segment hashes and name the affected records."""
        damaged: list[tuple[int, int]] = []
        offset = 0
        for segment in manifest.segments:
            block = np.ascontiguousarray(stored[offset : offset + segment.count])
            if hash_bytes(block.tobytes()) != segment.hash:
                damaged.append((offset, offset + segment.count - 1))
            offset += segment.count

        if damaged:
            raise ShadowMissing(
                f"{len(damaged)} shadow segments are corrupt, affecting records "
                f"{damaged[0][0]}-{damaged[-1][1]}.",
                hint=(
                    "Segment hashing means the damage is bounded and identified: "
                    "records outside those ranges are still restorable."
                ),
                context={"job_id": self.job_id, "count": len(damaged)},
            )

    def iter_batches(self, batch_size: int = 1024) -> Iterator[tuple[list[str], FloatArray]]:
        """Stream the shadow back in batches, for rollback."""
        manifest = self.manifest()
        ids = self.ids()
        stored = np.memmap(
            self.directory / _VECTORS,
            dtype=self._dtype,
            mode="r",
            shape=(manifest.record_count, manifest.dim),
        )
        for start in range(0, manifest.record_count, batch_size):
            stop = min(start + batch_size, manifest.record_count)
            yield ids[start:stop], as_float32(np.asarray(stored[start:stop]))

    # ── size ──────────────────────────────────────────────────────────

    def size_bytes(self) -> int:
        """Bytes on disk, or 0 when there is no shadow."""
        path = self.directory / _VECTORS
        return path.stat().st_size if path.exists() else 0

    @staticmethod
    def estimate_bytes(record_count: int, dim: int, *, precision: str = "float32") -> int:
        """What a shadow would cost, for the pre-check."""
        return record_count * dim * (2 if precision == "float16" else 4)


def _manifest_to_json(manifest: ShadowManifest) -> dict[str, Any]:
    return {
        "job_id": manifest.job_id,
        "record_count": manifest.record_count,
        "dim": manifest.dim,
        "precision": manifest.precision,
        "segments": [{"count": s.count, "hash": s.hash} for s in manifest.segments],
        "created_utc": manifest.created_utc,
    }


def _manifest_from_json(payload: dict[str, Any]) -> ShadowManifest:
    return ShadowManifest(
        job_id=str(payload["job_id"]),
        record_count=int(payload["record_count"]),
        dim=int(payload["dim"]),
        precision=str(payload["precision"]),
        segments=[
            ShadowSegment(count=int(s["count"]), hash=str(s["hash"])) for s in payload["segments"]
        ],
        created_utc=str(payload["created_utc"]),
    )
