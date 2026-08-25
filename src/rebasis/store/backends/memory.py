"""In-memory store.

The reference implementation. It exists for three reasons: contract tests need a
backend with no external dependency, `probe` needs somewhere to hold a sample,
and every other backend is checked against this one's behaviour.

Being the reference, it is deliberately strict — it raises where a real store
would raise, rather than being permissive because it can afford to be.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rebasis.errors import CollectionNotFound, EmbeddingDimensionMismatch, StoreUnsupported
from rebasis.types import FloatArray, Hit, Record, StoreCapabilities, as_float32

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from rebasis.store.uri import StoreURI

__all__ = ["MemoryStore"]


class MemoryStore:
    """A vector store held entirely in memory."""

    def __init__(
        self,
        ids: Sequence[str],
        vectors: FloatArray,
        texts: Sequence[str] | None = None,
        metadata: Sequence[Mapping[str, Any]] | None = None,
        *,
        path: Path | str | None = None,
    ) -> None:
        vectors = as_float32(vectors)
        if len(ids) != vectors.shape[0]:
            msg = f"{len(ids)} ids but {vectors.shape[0]} vectors"
            raise ValueError(msg)
        self._ids = list(ids)
        self._vectors = np.ascontiguousarray(vectors)
        self._texts = list(texts) if texts is not None else None
        self._metadata = list(metadata) if metadata is not None else None
        self._index = {record_id: i for i, record_id in enumerate(self._ids)}
        # Set when the store was opened from a file. An upsert then writes back
        # through: a store that accepts a write and forgets it on exit is the
        # commonest source of silent data loss.
        self._path = Path(path) if path is not None else None

    @property
    def capabilities(self) -> StoreCapabilities:
        """Everything is supported; this is the reference backend."""
        return StoreCapabilities(
            can_read_vectors=True,
            can_read_text=self._texts is not None,
            can_upsert_vectors=True,
            can_filter=False,
            dimension_locked=True,
            supports_in_place_update=True,
            name="memory",
        )

    def count(self) -> int:
        """Number of records held."""
        return len(self._ids)

    def dimension(self) -> int:
        """Vector dimensionality."""
        return int(self._vectors.shape[1])

    def iter_records(
        self,
        ids: Sequence[str] | None = None,
        *,
        with_vectors: bool = True,
        with_text: bool = True,
        batch_size: int = 1000,
    ) -> Iterator[Record]:
        """Stream records lazily.

        A generator even though everything is already in memory: the contract
        test asserts laziness for every backend, and a reference implementation
        that cheats teaches the wrong lesson.
        """
        del batch_size  # meaningful only for backends that page
        selected = range(len(self._ids)) if ids is None else self._positions(ids)
        for position in selected:
            yield Record(
                id=self._ids[position],
                vector=self._vectors[position] if with_vectors else None,
                text=(self._texts[position] if with_text and self._texts is not None else None),
                metadata=self._metadata[position] if self._metadata is not None else {},
            )

    def _positions(self, ids: Sequence[str]) -> list[int]:
        missing = [i for i in ids if i not in self._index]
        if missing:
            raise CollectionNotFound(
                f"{len(missing)} requested ids are not in this collection.",
                hint=f"First missing id: {missing[0]!r}.",
                context={"count": len(missing)},
            )
        return [self._index[i] for i in ids]

    def search(self, vector: FloatArray, k: int, where: dict[str, Any] | None = None) -> list[Hit]:
        """Exact nearest neighbours by inner product.

        Vectors are assumed ℓ2-normalised, where cosine and inner product agree;
        computing norms again on every search would be wasted work.
        """
        del where  # filtering is not a declared capability
        query = as_float32(vector).reshape(-1)
        if query.shape[0] != self.dimension():
            raise EmbeddingDimensionMismatch(
                f"The collection is {self.dimension()}-dimensional but the query "
                f"is {query.shape[0]}-dimensional.",
                hint=(
                    "The adapter's output dimension does not match the index. Run `rebasis doctor`."
                ),
                context={"dim": self.dimension()},
            )

        scores = self._vectors @ query
        k = min(k, scores.shape[0])
        top = np.argpartition(-scores, kth=k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            Hit(id=self._ids[int(i)], score=float(scores[int(i)]), rank=rank)
            for rank, i in enumerate(top)
        ]

    def upsert_vectors(self, ids: Sequence[str], vectors: FloatArray) -> None:
        """Replace vectors in place."""
        vectors = as_float32(vectors)
        if vectors.shape[1] != self.dimension():
            raise EmbeddingDimensionMismatch(
                f"The collection is {self.dimension()}-dimensional but the vectors "
                f"are {vectors.shape[1]}-dimensional.",
                hint="Refit the adapter against this index's dimension.",
                context={"dim": self.dimension()},
            )
        for position, vector in zip(self._positions(ids), vectors, strict=True):
            self._vectors[position] = vector
        if self._path is not None:
            self.save(self._path)

    def rebuild_index(self) -> None:
        """A full matrix multiply has no structure to rebuild."""
        from rebasis.store.base import require_capability

        require_capability(self, "can_rebuild_index", operation="rebuilding the index")

    @classmethod
    def from_uri(cls, uri: StoreURI, **kwargs: Any) -> MemoryStore:
        """Open an ``.npz`` export, or an empty store.

        ``memory:///path/to/vectors.npz`` · ``memory://``

        The file holds ``ids``, ``vectors`` and optionally ``texts``. That makes
        the reference backend reachable from the CLI, which matters for two
        reasons: it is how the end-to-end tests exercise the real command path,
        and it is the escape hatch for someone whose store has no rebasis
        backend — export, probe, decide.

        Raises:
            CollectionNotFound: When the path does not exist.
            StoreUnsupported: When the file is not a readable ``.npz`` with the
                expected arrays.
        """
        del kwargs
        if not uri.path:
            return cls([], np.empty((0, 0), dtype=np.float32))

        path = Path(uri.path)
        if not path.exists():
            raise CollectionNotFound(
                f"No file at {path}.",
                hint="`memory://` with no path opens an empty store.",
                context={"store_backend": "memory", "path": str(path)},
            )

        try:
            with np.load(path, allow_pickle=False) as payload:
                ids = [str(i) for i in payload["ids"]]
                vectors = as_float32(payload["vectors"])
                texts = [str(t) for t in payload["texts"]] if "texts" in payload else None
        except (OSError, ValueError, KeyError) as exc:
            raise StoreUnsupported(
                f"{path.name} is not a readable rebasis export.",
                hint="Expected an .npz holding `ids`, `vectors` and optionally `texts`.",
                context={"store_backend": "memory", "path": str(path)},
                cause=exc,
            ) from exc

        return cls(ids, vectors, texts, path=path)

    def save(self, path: Path | str) -> Path:
        """Write this store as an ``.npz`` that :meth:`from_uri` can reopen.

        Atomically: an interrupted write leaves the previous file intact rather
        than a truncated one. This is the user's index.
        """
        from rebasis.storage.atomic import atomic_write_stream

        destination = Path(path)
        arrays: dict[str, Any] = {"ids": np.array(self._ids), "vectors": self._vectors}
        if self._texts is not None:
            arrays["texts"] = np.array(self._texts)
        with atomic_write_stream(destination) as handle:
            np.savez(handle, **arrays)
        return destination

    @classmethod
    def from_records(cls, records: Sequence[Record]) -> MemoryStore:
        """Build a store from records — how `probe` holds its sample."""
        ids = [r.id for r in records]
        vectors = np.vstack([r.vector for r in records if r.vector is not None])
        texts = [r.text or "" for r in records]
        return cls(ids, vectors, texts)
