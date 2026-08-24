"""LlamaIndex vector store bridge.

The counterpart to the LangChain bridge, and it carries the same honest caveat.

LlamaIndex's ``VectorStore`` protocol is built around ``query(VectorStoreQuery)``
and ``add(nodes)``. There is no portable way to enumerate stored vectors and no
portable in-place update, so this bridge declares what the wrapped object can
genuinely do and refuses the rest up front.

``stores_text`` is a real signal here rather than a guess: LlamaIndex stores
declare it, and it says whether the node text lives in the store or only in a
separate docstore.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rebasis.errors import CapabilityMissing, StoreError
from rebasis.types import FloatArray, Hit, Record, StoreCapabilities, as_float32

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

__all__ = ["LlamaIndexStoreAdapter"]


class LlamaIndexStoreAdapter:
    """Wraps a LlamaIndex ``VectorStore``."""

    def __init__(self, vector_store: Any, *, dimension: int | None = None) -> None:
        self._store = vector_store
        self._dimension = dimension
        self._name = f"llamaindex:{type(vector_store).__name__}"

    @property
    def capabilities(self) -> StoreCapabilities:
        """Probe the wrapped store rather than assume."""
        client = getattr(self._store, "client", None)
        return StoreCapabilities(
            # Only reachable where the store exposes its native client.
            can_read_vectors=client is not None,
            can_read_text=bool(getattr(self._store, "stores_text", False)),
            can_upsert_vectors=False,
            can_filter=True,
            dimension_locked=False,
            supports_in_place_update=False,
            name=self._name,
        )

    def count(self) -> int:
        """Number of records, where the native client can report it."""
        client = getattr(self._store, "client", None)
        if client is not None and hasattr(client, "count"):
            return int(client.count())
        raise CapabilityMissing(
            f"{self._name} does not expose a record count.",
            hint=(
                "LlamaIndex's VectorStore protocol has no portable count. Use the "
                "native rebasis backend for this store if one exists."
            ),
            context={"store_backend": self._name},
        )

    def dimension(self) -> int:
        """Vector dimensionality, from the constructor."""
        if self._dimension is None:
            raise CapabilityMissing(
                f"{self._name} does not expose its dimensionality.",
                hint="Pass dimension= when constructing the adapter.",
                context={"store_backend": self._name},
            )
        return self._dimension

    def iter_records(
        self,
        ids: Sequence[str] | None = None,
        *,
        with_vectors: bool = True,
        with_text: bool = True,
        batch_size: int = 1000,
    ) -> Iterator[Record]:
        """Not supported through this interface.

        Raises:
            CapabilityMissing: Always when vectors are requested. LlamaIndex has
                no portable enumeration, and `probe` needs the stored vectors —
                so this fails before any work starts rather than partway through.
        """
        del ids, with_text, batch_size
        if with_vectors:
            raise CapabilityMissing(
                f"{self._name} cannot enumerate stored vectors.",
                hint=(
                    "LlamaIndex's protocol has no portable way to read them back. "
                    "Use a native rebasis backend for the underlying store — "
                    "chroma and lancedb both have one."
                ),
                context={"store_backend": self._name},
            )
        return iter(())

    def search(self, vector: FloatArray, k: int, where: dict[str, Any] | None = None) -> list[Hit]:
        """Nearest neighbours through ``VectorStoreQuery``."""
        try:
            from llama_index.core.vector_stores.types import VectorStoreQuery
        except ImportError as exc:
            raise CapabilityMissing(
                "This bridge needs llama-index-core.",
                hint="Install it with `pip install llama-index-core`.",
                context={"store_backend": self._name},
                cause=exc,
            ) from exc

        query = VectorStoreQuery(
            query_embedding=as_float32(vector).reshape(-1).tolist(),
            similarity_top_k=k,
            filters=where,
        )
        try:
            result = self._store.query(query)
        except Exception as exc:
            raise StoreError(
                f"The {self._name} query failed.",
                hint="Check the filter expression, if you passed one.",
                context={"store_backend": self._name},
                cause=exc,
            ) from exc

        ids = result.ids or []
        similarities = result.similarities or [0.0] * len(ids)
        return [
            Hit(id=str(doc_id), score=float(score), rank=rank)
            for rank, (doc_id, score) in enumerate(zip(ids, similarities, strict=False))
        ]

    def upsert_vectors(self, ids: Sequence[str], vectors: FloatArray) -> None:
        """Not supported through this interface."""
        del ids, vectors
        raise CapabilityMissing(
            f"{self._name} cannot replace vectors in place.",
            hint=(
                "LlamaIndex exposes add and delete, not update, so `migrate` "
                "cannot run through this bridge. `probe` and the bridge phase "
                "still work, and neither writes to the index."
            ),
            context={"store_backend": self._name},
        )
