"""Chroma backend.

Priority one, because Chroma's behaviour *is* the problem rebasis exists for: a
collection is locked to a dimensionality by its first batch of embeddings, so a
model with a different output size cannot be written to it at all.

The default ``query_to_old`` direction sidesteps that completely. The adapter
maps the new model's vector down to the index's dimensionality, so the
collection never sees a shape it did not expect and never has to be rebuilt.
Chroma's hardest constraint becomes a non-issue rather than an obstacle.

Every ``chromadb`` exception is converted at this boundary — a third-party
exception never crosses it. The caller sees a stable ``RB-Exxxx`` code with a
next step, never a client-library traceback, and the original stays as
``__cause__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from rebasis.errors import (
    CollectionNotFound,
    EmbeddingDimensionMismatch,
    MissingDependency,
    StoreError,
    StoreWriteFailed,
)
from rebasis.types import FloatArray, Hit, Record, StoreCapabilities, as_float32

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from rebasis.store.uri import StoreURI

__all__ = ["ChromaStore"]

#: Chroma pages its reads; this bounds peak memory per page.
DEFAULT_BATCH = 1000


class ChromaStore:
    """A Chroma collection."""

    def __init__(self, collection: Any, *, path: str = "") -> None:
        self._collection = collection
        self._path = path
        self._dimension: int | None = None

    @classmethod
    def from_uri(cls, uri: StoreURI, **kwargs: Any) -> ChromaStore:
        """Open the collection a URI points at.

        ``chroma:///path/to/db#collection_name``
        """
        try:
            import chromadb
        except ImportError as exc:
            raise MissingDependency(
                "The Chroma backend needs chromadb.",
                hint='Install it with `pip install "rebasis[chroma]"`.',
                context={"store_backend": "chroma"},
                cause=exc,
            ) from exc

        if not uri.collection:
            raise CollectionNotFound(
                "The Chroma URI does not name a collection.",
                hint="Append it after a #, for example `chroma:///path/to/db#my_collection`.",
                context={"store_backend": "chroma"},
            )

        client = chromadb.PersistentClient(path=uri.path or ".", **kwargs)
        try:
            collection = client.get_collection(name=uri.collection)
        except Exception as exc:
            names = _collection_names(client)
            raise CollectionNotFound(
                f"Chroma has no collection named {uri.collection!r}.",
                hint=f"Available: {', '.join(names) if names else '(none)'}.",
                context={"store_backend": "chroma"},
                cause=exc,
            ) from exc

        return cls(collection, path=uri.path)

    @property
    def capabilities(self) -> StoreCapabilities:
        """Chroma supports everything rebasis needs, subject to the dimension lock."""
        return StoreCapabilities(
            can_read_vectors=True,
            can_read_text=True,
            can_upsert_vectors=True,
            can_filter=True,
            # The collection is fixed to the dimensionality of its first write.
            # `query_to_old` makes that irrelevant.
            dimension_locked=True,
            supports_in_place_update=True,
            # A hard False, not a shrug, and checked rather than assumed
            # against both ends of the supported range — chromadb 0.5.23 and
            # 1.5.9. There is one storage encoding for a dense vector and it is
            # float32: `ScalarEncoding` has exactly two members, FLOAT32 and
            # INT32 (`chromadb/types.py`), the write boundary is
            # `np.array(vector, dtype=np.float32).tobytes()` and the read
            # boundary `np.frombuffer(vector, dtype=np.float32)`
            # (`chromadb/ingest/__init__.py`), and the index underneath is
            # hnswlib, which holds raw floats. No collection setting reaches
            # it: the whole configurable surface is `space`, `ef_construction`,
            # `ef_search`, `max_neighbors`, threading and batching.
            #
            # 1.x does carry a quantization concept — a 4-bit RaBitQ bound to
            # the SPANN index — and it is unreachable from here twice over: it
            # is a per-tenant server-side flag, and the bundled 1.5.9 binary
            # refuses to build a SPANN index locally at all. This backend opens
            # a `PersistentClient` and nothing else, so a Chroma collection
            # rebasis can see is a float32 one.
            quantized=False,
            name="chroma",
        )

    def count(self) -> int:
        """Number of records in the collection."""
        try:
            return int(self._collection.count())
        except Exception as exc:
            raise StoreError(
                "Could not read the collection size from Chroma.",
                hint="Check that the database path is readable and not locked.",
                context={"store_backend": "chroma"},
                cause=exc,
            ) from exc

    def dimension(self) -> int:
        """Vector dimensionality, read once from a single record."""
        if self._dimension is None:
            peek = self._collection.peek(limit=1)
            embeddings = peek.get("embeddings")
            if embeddings is None or len(embeddings) == 0:
                raise StoreError(
                    "The collection is empty, so its dimensionality cannot be determined.",
                    hint="rebasis needs an existing index to measure against.",
                    context={"store_backend": "chroma"},
                )
            self._dimension = int(np.asarray(embeddings[0]).shape[0])
        return self._dimension

    def iter_records(
        self,
        ids: Sequence[str] | None = None,
        *,
        with_vectors: bool = True,
        with_text: bool = True,
        batch_size: int = DEFAULT_BATCH,
    ) -> Iterator[Record]:
        """Stream records page by page.

        A generator, and paged: `.get()` with no limit would pull the whole
        collection into memory and break the ``O(batch × d)`` invariant on
        exactly the corpora where it matters.
        """
        include = []
        if with_vectors:
            include.append("embeddings")
        if with_text:
            include.append("documents")
        include.append("metadatas")

        if ids is not None:
            yield from self._page(list(ids), include)
            return

        total = self.count()
        for offset in range(0, total, batch_size):
            page = self._collection.get(limit=batch_size, offset=offset, include=include)
            yield from _records_from_page(page)

    def _page(self, ids: list[str], include: list[str]) -> Iterator[Record]:
        for start in range(0, len(ids), DEFAULT_BATCH):
            chunk = ids[start : start + DEFAULT_BATCH]
            page = self._collection.get(ids=chunk, include=include)
            returned = set(page.get("ids") or [])
            missing = [i for i in chunk if i not in returned]
            if missing:
                raise CollectionNotFound(
                    f"{len(missing)} requested ids are not in this collection.",
                    hint=f"First missing id: {missing[0]!r}.",
                    context={"store_backend": "chroma", "count": len(missing)},
                )
            yield from _records_from_page(page)

    def search(self, vector: FloatArray, k: int, where: dict[str, Any] | None = None) -> list[Hit]:
        """Nearest neighbours."""
        query = as_float32(vector).reshape(-1)
        if query.shape[0] != self.dimension():
            raise EmbeddingDimensionMismatch(
                f"The collection is {self.dimension()}-dimensional but the query "
                f"is {query.shape[0]}-dimensional.",
                hint=(
                    "The adapter's output dimension does not match the index. Run `rebasis doctor`."
                ),
                context={"store_backend": "chroma", "dim": self.dimension()},
            )
        try:
            result = self._collection.query(
                query_embeddings=[query.tolist()], n_results=k, where=where
            )
        except Exception as exc:
            raise StoreError(
                "The Chroma query failed.",
                hint="Check the filter expression, if you passed one.",
                context={"store_backend": "chroma"},
                cause=exc,
            ) from exc

        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            # Chroma returns a distance; rebasis speaks similarity everywhere.
            Hit(id=str(doc_id), score=float(1.0 - distance), rank=rank)
            for rank, (doc_id, distance) in enumerate(zip(ids, distances, strict=True))
        ]

    def upsert_vectors(self, ids: Sequence[str], vectors: FloatArray) -> None:
        """Replace vectors in place. The only write path rebasis has."""
        matrix = as_float32(vectors)
        if matrix.shape[1] != self.dimension():
            raise EmbeddingDimensionMismatch(
                f"The collection is {self.dimension()}-dimensional but the vectors "
                f"are {matrix.shape[1]}-dimensional.",
                hint=(
                    "Chroma locks a collection to the dimensionality of its "
                    "first write. Bridging needs no dimension change at all — "
                    "fit an adapter and put it in front of your queries instead "
                    "of rewriting the index."
                ),
                context={"store_backend": "chroma", "dim": self.dimension()},
            )
        try:
            self._collection.update(ids=list(ids), embeddings=matrix.tolist())
        except Exception as exc:
            raise StoreWriteFailed(
                f"Chroma rejected an update of {len(ids)} records.",
                hint="Transient failures are retried automatically; this one was not.",
                context={"store_backend": "chroma", "count": len(ids)},
                cause=exc,
            ) from exc

    def rebuild_index(self) -> None:
        """Chroma exposes no way to rebuild an existing collection's index.

        `.modify()` reaches the query-time settings — `ef_search`, thread count,
        batch size — and not `ef_construction` or `max_neighbors`, which are
        construction-only. There is no supported call that reconstructs the
        graph in place, so a collection that lost recall to a migration has to
        be rebuilt outside rebasis: dump and re-add, or a maintenance tool such
        as `chromadb-ops`.
        """
        from rebasis.store.base import require_capability

        require_capability(self, "can_rebuild_index", operation="rebuilding the index")


def _collection_names(client: Any) -> list[str]:
    try:
        return [c.name for c in client.list_collections()]
    except Exception:  # noqa: BLE001 - only used to enrich an error message
        return []


def _records_from_page(page: dict[str, Any]) -> Iterator[Record]:
    ids = page.get("ids") or []
    embeddings = page.get("embeddings")
    documents = page.get("documents")
    metadatas = page.get("metadatas")
    for index, doc_id in enumerate(ids):
        yield Record(
            id=str(doc_id),
            vector=(
                as_float32(embeddings[index])
                if embeddings is not None and len(embeddings) > index
                else None
            ),
            text=(documents[index] if documents is not None and len(documents) > index else None),
            metadata=(
                metadatas[index]
                if metadatas is not None and len(metadatas) > index and metadatas[index]
                else {}
            ),
        )
