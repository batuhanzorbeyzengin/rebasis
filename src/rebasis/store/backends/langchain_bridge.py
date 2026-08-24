"""LangChain vector store bridge.

A high-leverage adapter: one file buys access to dozens of stores through
LangChain's ``VectorStore`` interface, without rebasis writing a backend for each.

**The catch, stated plainly.** That interface does not expose
``iter_records(with_vectors=True)`` or an in-place vector update for every store
behind it. Many implementations can search and add, but cannot hand back the
vectors they hold. So this bridge probes what the wrapped object can actually do
and **reports it honestly**.

The consequence is worth spelling out: ``probe`` and the bridge phase generally
work here, and ``migrate`` often does not. That is a genuinely useful position —
diagnosis and bridging are the parts that need no writes. What would not be
acceptable is claiming the capability and failing halfway through someone's
migration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rebasis.errors import CapabilityMissing, StoreError
from rebasis.types import FloatArray, Hit, Record, StoreCapabilities, as_float32

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

__all__ = ["LangChainStoreAdapter"]


class LangChainStoreAdapter:
    """Wraps a LangChain ``VectorStore``."""

    def __init__(self, vector_store: Any, *, dimension: int | None = None) -> None:
        self._store = vector_store
        self._dimension = dimension
        self._name = f"langchain:{type(vector_store).__name__}"

    @property
    def capabilities(self) -> StoreCapabilities:
        """Probe the wrapped store rather than assume.

        Everything here is discovered by inspection: a capability is declared
        only when the method that implements it is actually present.
        """
        underlying = getattr(self._store, "_collection", None) or getattr(
            self._store, "_client", None
        )
        return StoreCapabilities(
            # LangChain has no portable "read the vectors back" call. Where the
            # native handle is reachable it can be done; otherwise it cannot.
            can_read_vectors=underlying is not None,
            can_read_text=hasattr(self._store, "similarity_search"),
            can_upsert_vectors=hasattr(self._store, "add_embeddings"),
            can_filter=True,
            dimension_locked=False,
            supports_in_place_update=False,
            name=self._name,
        )

    def count(self) -> int:
        """Number of records, where the wrapped store can report it."""
        for attribute in ("_collection", "_client"):
            handle = getattr(self._store, attribute, None)
            if handle is not None and hasattr(handle, "count"):
                return int(handle.count())
        raise CapabilityMissing(
            f"{self._name} does not expose a record count.",
            hint=(
                "LangChain's VectorStore interface has no portable count. Use the "
                "native rebasis backend for this store if one exists."
            ),
            context={"store_backend": self._name},
        )

    def dimension(self) -> int:
        """Vector dimensionality, from the constructor or by embedding a probe."""
        if self._dimension is not None:
            return self._dimension
        embeddings = getattr(self._store, "embeddings", None) or getattr(
            self._store, "embedding_function", None
        )
        if embeddings is not None and hasattr(embeddings, "embed_query"):
            self._dimension = len(embeddings.embed_query("dimension probe"))
            return self._dimension
        raise CapabilityMissing(
            f"{self._name} does not expose its dimensionality.",
            hint="Pass dimension= when constructing the adapter.",
            context={"store_backend": self._name},
        )

    def iter_records(
        self,
        ids: Sequence[str] | None = None,
        *,
        with_vectors: bool = True,
        with_text: bool = True,
        batch_size: int = 1000,
    ) -> Iterator[Record]:
        """Stream records through the native handle, when there is one.

        Raises:
            CapabilityMissing: When vectors are requested and the wrapped store
                cannot return them. Failing here, before any work begins, is far
                better than discovering it mid-migration.
        """
        del batch_size
        if with_vectors and not self.capabilities.can_read_vectors:
            raise CapabilityMissing(
                f"{self._name} cannot return stored vectors.",
                hint=(
                    "LangChain's interface does not expose them portably. `probe` "
                    "needs them, so use a native rebasis backend for this store."
                ),
                context={"store_backend": self._name},
            )

        collection = getattr(self._store, "_collection", None)
        if collection is None:
            raise CapabilityMissing(
                f"{self._name} does not expose an iterable collection.",
                hint="Use a native rebasis backend for this store.",
                context={"store_backend": self._name},
            )

        include = ["embeddings"] if with_vectors else []
        if with_text:
            include.append("documents")
        page = collection.get(ids=list(ids) if ids else None, include=include)

        for index, record_id in enumerate(page.get("ids") or []):
            embeddings = page.get("embeddings")
            documents = page.get("documents")
            yield Record(
                id=str(record_id),
                vector=(
                    as_float32(embeddings[index])
                    if with_vectors and embeddings is not None
                    else None
                ),
                text=documents[index] if with_text and documents is not None else None,
            )

    def search(self, vector: FloatArray, k: int, where: dict[str, Any] | None = None) -> list[Hit]:
        """Nearest neighbours by vector."""
        query = as_float32(vector).reshape(-1).tolist()
        method = getattr(self._store, "similarity_search_by_vector_with_relevance_scores", None)
        if method is None:
            method = getattr(self._store, "similarity_search_by_vector", None)
        if method is None:
            raise CapabilityMissing(
                f"{self._name} cannot search by vector.",
                hint="rebasis searches by vector, not by text; this store does not support it.",
                context={"store_backend": self._name},
            )

        try:
            results = method(query, k=k, filter=where) if where else method(query, k=k)
        except Exception as exc:
            raise StoreError(
                f"The {self._name} query failed.",
                hint="Check the filter expression, if you passed one.",
                context={"store_backend": self._name},
                cause=exc,
            ) from exc

        hits: list[Hit] = []
        for rank, item in enumerate(results):
            document, score = item if isinstance(item, tuple) else (item, 0.0)
            identifier = (
                getattr(document, "id", None) or (document.metadata or {}).get("id") or str(rank)
            )
            hits.append(Hit(id=str(identifier), score=float(score), rank=rank))
        return hits

    def upsert_vectors(self, ids: Sequence[str], vectors: FloatArray) -> None:
        """Not supported through this interface.

        LangChain has ``add_embeddings`` but no portable in-place *replace*, and
        adding would duplicate records rather than update them — which is worse
        than refusing.
        """
        del ids, vectors
        raise CapabilityMissing(
            f"{self._name} cannot replace vectors in place.",
            hint=(
                "LangChain exposes add, not update, so `migrate` cannot run "
                "through this bridge. `probe` and the bridge phase still work, "
                "and neither writes to the index."
            ),
            context={"store_backend": self._name},
        )
