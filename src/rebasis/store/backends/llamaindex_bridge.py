"""LlamaIndex vector store bridge.

The counterpart to the LangChain bridge, and it carries the same honest caveat.

LlamaIndex's store interface is built around ``query(VectorStoreQuery)`` and
``add(nodes)``. There is no portable way to enumerate stored vectors and no
portable in-place update, so this bridge declares what the wrapped object can
genuinely do and refuses the rest up front.

**What that leaves is search, and only search.** Every read-shaped capability is
declared ``False`` — not because the wrapped store cannot hold vectors or text,
but because this bridge has no way to get them back out. A capability says what
the *bridge* can deliver, never what the store contains, and the difference is
the whole point: ``probe`` checks ``can_read_vectors`` before it starts, so a
``True`` here moves the failure from second zero into the middle of a run.

``stores_text`` is a real signal about the wrapped store — LlamaIndex stores
declare it, and it says whether the node text lives in the store or only in a
separate docstore. It says nothing about whether this bridge can read that text,
which it cannot, so it is not what ``can_read_text`` is derived from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rebasis.errors import CapabilityMissing, MissingDependency, StoreError
from rebasis.types import FloatArray, Hit, Record, StoreCapabilities, as_float32

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

__all__ = ["LlamaIndexStoreAdapter"]


class LlamaIndexStoreAdapter:
    """Wraps a LlamaIndex vector store."""

    def __init__(self, vector_store: Any, *, dimension: int | None = None) -> None:
        self._store = vector_store
        self._dimension = dimension
        self._name = f"llamaindex:{type(vector_store).__name__}"

    @property
    def capabilities(self) -> StoreCapabilities:
        """What this bridge implements — which is less than the store holds.

        Every field is a constant, deliberately. Inferring them from the wrapped
        object was worse than useless here: ``can_read_vectors`` was read off the
        presence of a ``client`` attribute, which ``BasePydanticVectorStore``
        declares abstract and every store built on it therefore has, while
        :meth:`iter_records` refuses unconditionally. The declaration promised a
        read that no line of this file performs.
        """
        return StoreCapabilities(
            # `iter_records` refuses unconditionally, so anything but False here
            # would be a promise this very file breaks two methods later.
            can_read_vectors=False,
            can_read_text=False,
            can_upsert_vectors=False,
            # rebasis passes filters as a mapping. `VectorStoreQuery.filters`
            # takes a `MetadataFilters` model, and LlamaIndex reads `.filters`
            # and `.condition` off it — attributes a mapping does not have. The
            # dataclass accepts the mapping without complaint and the store
            # fails on it later, which is precisely the shape of failure this
            # declaration exists to prevent.
            can_filter=False,
            dimension_locked=False,
            supports_in_place_update=False,
            name=self._name,
        )

    def count(self) -> int:
        """Number of records, where the native client can report it.

        Raises:
            CapabilityMissing: When there is no client, or it cannot count.
            StoreError: When the client has a ``count`` that means something
                else and refuses these arguments.
        """
        client = self._client()
        if client is not None and hasattr(client, "count"):
            try:
                return int(client.count())
            except Exception as exc:
                raise StoreError(
                    f"{self._name} has a native client whose count() this bridge could not use.",
                    hint=(
                        "The wrapped store's client counts something other than "
                        "records, or needs arguments. Pass the collection to a "
                        "native rebasis backend instead."
                    ),
                    context={"store_backend": self._name},
                    cause=exc,
                ) from exc
        raise CapabilityMissing(
            f"{self._name} does not expose a record count.",
            hint=(
                "LlamaIndex's vector store interface has no portable count. Use "
                "the native rebasis backend for this store if one exists."
            ),
            context={"store_backend": self._name},
        )

    def dimension(self) -> int:
        """Vector dimensionality, from the constructor.

        There is nothing to probe: the interface exposes no embedding model and
        no stored vector, so the only truthful source is the caller.

        Raises:
            CapabilityMissing: When the adapter was built without ``dimension=``.
        """
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
        """Not supported through this interface, for any projection.

        Refused unconditionally rather than only when vectors are asked for. An
        earlier version returned an empty iterator for the ids-and-text case,
        which is the worst of the three available answers: a caller cannot tell
        an empty collection from one this bridge cannot enumerate, and a
        migration reading zero records from a full store looks like success.

        Raises:
            CapabilityMissing: Always. Raised by the call itself, not by the
                first ``next()`` — a generator would defer it to the middle of
                the caller's loop, which is where this bridge exists not to fail.
        """
        del ids, with_vectors, with_text, batch_size
        raise CapabilityMissing(
            f"{self._name} cannot enumerate stored records.",
            hint=(
                "LlamaIndex's interface has no portable way to read them back. "
                "Use a native rebasis backend for the underlying store — chroma "
                "and lancedb both have one."
            ),
            context={"store_backend": self._name},
        )

    def search(self, vector: FloatArray, k: int, where: dict[str, Any] | None = None) -> list[Hit]:
        """Nearest neighbours through ``VectorStoreQuery``.

        The one operation this bridge performs.

        Raises:
            CapabilityMissing: When a filter is passed, which this bridge cannot
                translate.
            MissingDependency: When llama-index-core is not installed.
            StoreError: When the wrapped store's query fails, or answers in a
                shape whose hits cannot be identified.
        """
        if where is not None:
            raise CapabilityMissing(
                f"{self._name} cannot apply a filter.",
                hint=(
                    "LlamaIndex reads `.filters` and `.condition` off a "
                    "MetadataFilters model and rebasis passes a mapping. Search "
                    "without a filter, or use a native rebasis backend."
                ),
                context={"store_backend": self._name},
            )

        try:
            from llama_index.core.vector_stores.types import VectorStoreQuery
        except ImportError as exc:
            raise MissingDependency(
                "This bridge needs llama-index-core.",
                hint='Install it with `pip install "rebasis[llamaindex]"`.',
                context={"store_backend": self._name},
                cause=exc,
            ) from exc

        try:
            # A plain dataclass at 0.14, so this cannot fail today. It is
            # guarded because this is the one line that would break if the
            # query type gained validation or renamed a field, and a bridge
            # discovering that through somebody else's traceback is the failure
            # this whole module is written to avoid.
            query = VectorStoreQuery(
                query_embedding=as_float32(vector).reshape(-1).tolist(),
                similarity_top_k=k,
            )
        except Exception as exc:
            raise StoreError(
                "rebasis could not build a query the installed llama-index-core accepts.",
                hint=(
                    "The VectorStoreQuery interface has changed under this "
                    "bridge. Upgrade rebasis, or pin llama-index-core."
                ),
                context={"store_backend": self._name},
                cause=exc,
            ) from exc

        try:
            result = self._store.query(query)
        except Exception as exc:
            raise StoreError(
                f"The {self._name} query failed.",
                hint="Check that the collection holds vectors of this dimensionality.",
                context={"store_backend": self._name},
                cause=exc,
            ) from exc

        return self._hits(result)

    def _hits(self, result: Any) -> list[Hit]:
        """Map a ``VectorStoreQueryResult`` onto rebasis hits.

        All three of its fields are optional, and stores differ in which they
        fill: some answer with ids, some with nodes, most with both. Ids are
        taken from whichever is present, because rebasis matches a hit to a
        record by id and a hit whose id is invented matches nothing.
        """
        ids = list(getattr(result, "ids", None) or [])
        nodes = list(getattr(result, "nodes", None) or [])
        if not ids and nodes:
            ids = [_node_id(node) for node in nodes]
            if not all(ids):
                raise StoreError(
                    f"{self._name} answered with nodes that carry no id.",
                    hint=(
                        "rebasis matches a hit to a record by id, so it cannot "
                        "use these results. Use a native rebasis backend."
                    ),
                    context={"store_backend": self._name, "count": len(nodes)},
                )

        # Zipped by index rather than by `zip`: a store that returns fewer
        # similarities than ids would otherwise drop the unscored hits
        # silently, and a missing score is not a missing result.
        similarities = list(getattr(result, "similarities", None) or [])
        return [
            Hit(
                id=str(doc_id),
                score=float(similarities[rank]) if rank < len(similarities) else 0.0,
                rank=rank,
            )
            for rank, doc_id in enumerate(ids)
        ]

    def upsert_vectors(self, ids: Sequence[str], vectors: FloatArray) -> None:
        """Not supported through this interface.

        Raises:
            CapabilityMissing: Always.
        """
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

    def rebuild_index(self) -> None:
        """Not supported through this interface.

        The wrapped store may well have a search structure worth rebuilding.
        This bridge cannot reach it, and the protocol requires the refusal to
        arrive as :class:`CapabilityMissing` rather than as the ``AttributeError``
        a missing method would raise.

        Raises:
            CapabilityMissing: Always.
        """
        from rebasis.store.base import require_capability

        require_capability(self, "can_rebuild_index", operation="rebuilding the index")

    def _client(self) -> Any:
        """The wrapped store's native client, or None where there is none.

        ``client`` is an abstract property on ``BasePydanticVectorStore``, so
        reading it runs the store's own code — which may connect, and may fail.
        A bridge asking what a store can do must not be the thing that breaks.
        """
        try:
            return getattr(self._store, "client", None)
        except Exception:  # noqa: BLE001 - a failing probe means "no client", not a crash
            return None


def _node_id(node: Any) -> str:
    """A node's id, from either name LlamaIndex has used for it.

    ``BaseNode.node_id`` is the accessor; ``id_`` is the field behind it.
    """
    identifier = getattr(node, "node_id", None) or getattr(node, "id_", None)
    return str(identifier) if identifier else ""
