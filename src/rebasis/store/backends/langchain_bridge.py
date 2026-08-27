"""LangChain vector store bridge.

A high-leverage adapter: one file buys access to dozens of stores through
LangChain's ``VectorStore`` interface, without rebasis writing a backend for each.

**The catch, stated plainly.** That interface does not expose
``iter_records(with_vectors=True)`` or an in-place vector update for every store
behind it. Many implementations can search and add, but cannot hand back the
vectors they hold. So this bridge probes what the wrapped object can actually do
and **reports it honestly**.

The consequence is worth spelling out: ``probe`` and the bridge phase generally
work here, and ``migrate`` does not. That is a genuinely useful position —
diagnosis and bridging are the parts that need no writes. What would not be
acceptable is claiming the capability and failing halfway through someone's
migration.

**What "probe the object" can and cannot establish.** Almost every name on
``langchain_core.vectorstores.VectorStore`` is *concrete*, and several of the
concrete ones exist only to raise ``NotImplementedError``. ``similarity_search``
is abstract, so every store has it; ``similarity_search_by_vector`` is present on
every store too and refuses on most. Presence therefore proves very little, and
an inference drawn from ``hasattr`` alone is close to worthless — which is how
this bridge came to declare three capabilities it did not have. Every capability
below is now tied to the handle the corresponding method actually reads through,
and where presence genuinely cannot settle the question the bridge asks the
caller instead of guessing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from rebasis.errors import CapabilityMissing, CollectionNotFound, StoreError
from rebasis.types import FloatArray, Hit, Record, StoreCapabilities, as_float32

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

__all__ = ["LangChainStoreAdapter"]

#: How to read the float a by-vector search returns, when the caller knows.
#:
#: LangChain has no portable answer. ``langchain_chroma.Chroma`` returns a raw
#: distance from ``similarity_search_by_vector_with_relevance_scores`` — "lower
#: score represents more similarity", in its own docstring, despite the name —
#: while other integrations return a relevance score where higher is better.
#: Nothing on the object says which, so the bridge does not guess; see
#: :class:`LangChainStoreAdapter`.
ScoreKind = Literal["similarity", "distance"]

#: Methods that answer a query by vector, in the order they are tried.
#:
#: ``similarity_search_by_vector`` comes last because it is the only one of the
#: three that langchain-core defines: concrete, with a body of ``raise
#: NotImplementedError``. It is present on every LangChain store ever written
#: and its presence says nothing at all, so it is the fallback rather than the
#: first choice.
#:
#: The two scored names are integration-defined and nearly disjoint — of the 88
#: stores in ``langchain_community``, 29 define the first and 6 define the
#: second, and the two most-used stores land on opposite sides: FAISS and
#: PGVector have ``similarity_search_with_score_by_vector``, Chroma has
#: ``similarity_search_by_vector_with_relevance_scores``. Neither converges on a
#: signature past ``(embedding, k)``, which is why nothing beyond those two
#: arguments is passed unless the caller asked for it.
_SEARCH_METHODS = (
    "similarity_search_with_score_by_vector",
    "similarity_search_by_vector_with_relevance_scores",
    "similarity_search_by_vector",
)

#: Where an embedding model may be found on a wrapped store, widest first.
#:
#: ``embeddings`` is the base class's own property — concrete, and returning
#: ``None`` unless the integration overrides it, which is a supported state
#: rather than an error. The rest are the attribute names integrations actually
#: use, and there are five of them: across ``langchain_community``'s 88 stores,
#: ``_embedding`` appears 22 times, ``embedding_function`` 18, ``embedding`` 14,
#: ``_embedding_function`` 7 (this is Chroma's) and ``_embeddings`` 5. LangChain's
#: own retriever checks only ``embedding``, which covers a sixth of them.
_EMBEDDER_ATTRIBUTES = (
    "embeddings",
    "_embedding",
    "embedding_function",
    "embedding",
    "_embedding_function",
    "_embeddings",
)

#: Read once from the wrapped store; the answer never changes for its lifetime.
_PROBE_TEXT = "dimension probe"


class LangChainStoreAdapter:
    """Wraps a LangChain ``VectorStore``.

    Args:
        vector_store: The LangChain store to wrap. Duck-typed — the framework
            itself is never imported, so this costs nothing to anyone who does
            not already have it.
        dimension: The collection's dimensionality, when the caller knows it.
            Supplying it skips the probe described in :meth:`dimension`, which
            is a billed API call against a hosted embedding model.
        score_kind: How to read the float a by-vector search returns.
            ``"similarity"`` means higher is closer, ``"distance"`` means lower
            is closer and the bridge converts. Left unset, every hit is scored
            ``0.0`` and only the ranking is reported, because LangChain
            guarantees nothing about the direction of that number and a score
            that is silently inverted is worse than one that is visibly absent.
    """

    def __init__(
        self,
        vector_store: Any,
        *,
        dimension: int | None = None,
        score_kind: ScoreKind | None = None,
    ) -> None:
        self._store = vector_store
        self._dimension = dimension
        self._score_kind = score_kind
        self._name = f"langchain:{type(vector_store).__name__}"

    @property
    def capabilities(self) -> StoreCapabilities:
        """Probe the wrapped store rather than assume.

        Each answer is tied to the handle the method that implements it reads
        through, which is the only inference that cannot drift apart from the
        code below it. Three earlier ones did:

        * ``can_read_text`` was ``hasattr(store, "similarity_search")``. That
          method is abstract on the base class, so every LangChain store has it,
          and it searches *by* text rather than returning stored text — the text
          this bridge yields comes from the collection handle, like the vectors.
          The declaration was therefore ``True`` for every store, including the
          majority that cannot produce a single record.
        * ``can_read_vectors`` accepted a ``_client`` in place of a
          ``_collection``, while :meth:`iter_records` reads only through the
          collection and refuses without one.
        * ``can_upsert_vectors`` was ``hasattr(store, "add_embeddings")``, a
          method langchain-core does not define at all and whose signature
          differs between the integrations that do — and which
          :meth:`upsert_vectors` refuses regardless, because adding is not
          updating.

        A fourth, ``can_filter``, was not inferred from anything at all: it was
        the constant ``True``. It is now the constant ``False``, which is the
        answer inspection can actually support.
        """
        readable = self._readable()
        return StoreCapabilities(
            # One condition for both, because one call yields both: the
            # collection handle's `get`.
            can_read_vectors=readable,
            can_read_text=readable,
            # Never. `add_embeddings` appends; rebasis needs replacement, and
            # appending would duplicate every record it touched.
            can_upsert_vectors=False,
            # Not establishable, so not claimed. LangChain has no filter
            # contract: Chroma takes a mapping, langchain-core's own
            # InMemoryVectorStore takes a `Callable[[Document], bool]` and
            # raises on a mapping, Milvus takes no filter argument at all.
            # `search` still forwards a `where` the caller passes — they may
            # know their store — and converts whatever it refuses.
            can_filter=False,
            dimension_locked=False,
            supports_in_place_update=False,
            name=self._name,
        )

    def count(self) -> int:
        """Number of records, where the wrapped store can report it.

        Read from the same collection handle :meth:`iter_records` streams
        through, and from nothing else. An earlier version fell back to
        ``_client.count()``, which answers a different question — a client
        counts collections, or needs to be told which collection to count — so
        the number it returned did not describe the records this bridge
        iterates.

        Raises:
            CapabilityMissing: When the store exposes no countable collection.
            StoreError: When the handle has a ``count`` that refuses this call.
        """
        collection = self._collection()
        counter = getattr(collection, "count", None) if collection is not None else None
        if callable(counter):
            try:
                return int(counter())
            except Exception as exc:
                raise StoreError(
                    f"{self._name} exposes a collection whose count() this bridge could not use.",
                    hint=(
                        "The wrapped handle counts something other than records, "
                        "or needs arguments. Use the native rebasis backend for "
                        "this store if one exists."
                    ),
                    context={"store_backend": self._name},
                    cause=exc,
                ) from exc
        raise CapabilityMissing(
            f"{self._name} does not expose a record count.",
            hint=(
                "LangChain's VectorStore interface has no portable count. Use the "
                "native rebasis backend for this store if one exists."
            ),
            context={"store_backend": self._name},
        )

    def dimension(self) -> int:
        """Vector dimensionality, from the constructor or by embedding a probe.

        The probe is a real call to the wrapped store's embedding model, which
        against a hosted API is a billed request. It runs at most once per
        adapter: the result is cached, and every later call returns it. Pass
        ``dimension=`` to the constructor to skip it entirely.

        A failed probe is deliberately not cached. It is usually the embedding
        service being briefly unreachable, and remembering a transient failure
        for the lifetime of the process would turn a retryable error into a
        permanent one.

        Raises:
            CapabilityMissing: When neither a dimension nor an embedding model
                is available.
            StoreError: When the probe reaches the model and the model fails, or
                answers with nothing.
        """
        if self._dimension is not None:
            return self._dimension

        embedder = self._embedder()
        if embedder is None:
            raise CapabilityMissing(
                f"{self._name} does not expose its dimensionality.",
                hint="Pass dimension= when constructing the adapter.",
                context={"store_backend": self._name},
            )

        try:
            probe = embedder.embed_query(_PROBE_TEXT)
        except Exception as exc:
            raise StoreError(
                f"{self._name} could not embed a probe to establish its dimensionality.",
                hint="Pass dimension= when constructing the adapter to skip the probe.",
                context={"store_backend": self._name},
                cause=exc,
            ) from exc

        measured = len(probe)
        if measured == 0:
            raise StoreError(
                f"{self._name} embedded a probe and returned an empty vector.",
                hint="Pass dimension= when constructing the adapter to skip the probe.",
                context={"store_backend": self._name},
            )
        self._dimension = measured
        return self._dimension

    def iter_records(
        self,
        ids: Sequence[str] | None = None,
        *,
        with_vectors: bool = True,
        with_text: bool = True,
        batch_size: int = 1000,
    ) -> Iterator[Record]:
        """Stream records through the native handle, when there is one.

        Not itself a generator, which is the point: everything that can be
        established before the first record is established by the call rather
        than by the first ``next()``. A generator function defers its own body,
        so the ``CapabilityMissing`` below used to arrive from inside the
        caller's loop — the exact failure mode this bridge's honesty about
        capabilities exists to prevent. The returned object is still a
        generator, and still lazy.

        Raises:
            CapabilityMissing: When the wrapped store cannot enumerate its
                records. Failing here, before any work begins, is far better
                than discovering it mid-migration.
        """
        collection = self._collection()
        if not self._is_readable(collection):
            raise CapabilityMissing(
                f"{self._name} does not expose an iterable collection.",
                hint=(
                    "LangChain's interface does not expose stored records "
                    "portably, and `probe` needs them. Use a native rebasis "
                    "backend for this store — chroma, lancedb, qdrant, "
                    "sqlite-vec and faiss all have one."
                ),
                context={"store_backend": self._name},
            )
        include: list[str] = []
        if with_vectors:
            include.append("embeddings")
        if with_text:
            include.append("documents")
        # A non-positive window would page forever without ever advancing.
        window = max(1, batch_size)

        if ids is None:
            return self._stream_pages(collection, include, batch_size=window)
        return self._stream_ids(collection, [str(i) for i in ids], include, batch_size=window)

    def _stream_ids(
        self, collection: Any, ids: list[str], include: list[str], *, batch_size: int
    ) -> Iterator[Record]:
        """Fetch named records, a batch at a time, and insist on all of them."""
        for start in range(0, len(ids), batch_size):
            chunk = ids[start : start + batch_size]
            page = self._page(collection, include, ids=chunk)
            returned = {str(i) for i in (page.get("ids") or [])}
            missing = [i for i in chunk if i not in returned]
            if missing:
                # Chroma's `get` answers an unknown id by omitting it. A
                # migration that skipped records rather than failing on them
                # would leave a partly-rewritten index and no sign of it.
                raise CollectionNotFound(
                    f"{len(missing)} requested ids are not in this collection.",
                    hint=f"First missing id: {missing[0]!r}.",
                    context={"store_backend": self._name, "count": len(missing)},
                )
            yield from _records_from_page(page, include)

    def _stream_pages(
        self, collection: Any, include: list[str], *, batch_size: int
    ) -> Iterator[Record]:
        """Page through the whole collection, holding one page at a time.

        Paged rather than fetched whole. An earlier version called ``get()``
        with no window at all and yielded from the result, which satisfies every
        test that checks the *type* of what ``iter_records`` returns and none of
        the reason those tests exist: peak memory was ``O(N × d)``, and the
        corpora where that matters are the ones nobody has in development.
        """
        previous_first: str | None = None
        offset = 0
        while True:
            page = self._page(collection, include, limit=batch_size, offset=offset)
            page_ids = [str(i) for i in (page.get("ids") or [])]
            if not page_ids:
                return
            if page_ids[0] == previous_first:
                # The handle accepted `limit`/`offset` and ignored them, so the
                # next iteration would return this page again, forever. One
                # duplicated page is enough to know.
                raise StoreError(
                    f"{self._name} returned the same page twice while streaming.",
                    hint=(
                        "The wrapped collection ignores the limit/offset window, "
                        "so rebasis cannot read it without loading it whole. Use "
                        "a native rebasis backend for this store."
                    ),
                    context={"store_backend": self._name, "count": len(page_ids)},
                )
            yield from _records_from_page(page, include)
            if len(page_ids) < batch_size:
                return
            previous_first = page_ids[0]
            offset += len(page_ids)

    def _page(self, collection: Any, include: list[str], **window: Any) -> dict[str, Any]:
        """One ``get`` against the wrapped handle, with its errors converted."""
        try:
            page = collection.get(include=include, **window)
        except Exception as exc:
            raise StoreError(
                f"{self._name} refused a read of its collection.",
                hint=(
                    "The wrapped handle does not take the arguments Chroma's "
                    "`get` takes. Use a native rebasis backend for this store."
                ),
                context={"store_backend": self._name},
                cause=exc,
            ) from exc
        if not isinstance(page, dict):
            raise StoreError(
                f"{self._name} answered a collection read in an unreadable shape.",
                hint="rebasis expects Chroma's `get` mapping of ids, embeddings and documents.",
                context={"store_backend": self._name},
            )
        return page

    def search(self, vector: FloatArray, k: int, where: dict[str, Any] | None = None) -> list[Hit]:
        """Nearest neighbours by vector.

        Each candidate method is tried until one answers. A ``NotImplementedError``
        moves on to the next rather than escaping, because that is what the base
        class raises from the by-vector method every store inherits — reporting
        it as a failed query would name the wrong problem, and an earlier version
        did exactly that, converting it to a "the query failed" error whose hint
        pointed at a filter the caller had not passed.

        A ``where`` is forwarded as LangChain's ``filter=`` keyword even though
        ``can_filter`` is ``False``. The two are not in conflict: the capability
        says rebasis may not *rely* on filtering here, while forwarding lets a
        caller who knows their own store use it. What the bridge guarantees is
        that a store which cannot take the filter says so — several take no
        ``**kwargs`` at all and raise ``TypeError`` — rather than quietly
        returning unfiltered results.

        Raises:
            CapabilityMissing: When no method on the store searches by vector.
            StoreError: When one does and the query fails, or answers with
                documents that carry no id.
        """
        query = as_float32(vector).reshape(-1).tolist()
        for name in _SEARCH_METHODS:
            method = getattr(self._store, name, None)
            if not callable(method):
                continue
            results = self._query(method, query, k=k, where=where)
            if results is not None:
                return self._hits(results)

        raise CapabilityMissing(
            f"{self._name} cannot search by vector.",
            hint="rebasis searches by vector, not by text; this store does not support it.",
            context={"store_backend": self._name},
        )

    def _query(
        self, method: Any, query: list[float], *, k: int, where: dict[str, Any] | None
    ) -> Any:
        """Call one candidate search method, or report that it is a stub.

        ``None`` means "this name is present and does nothing", which is the
        state langchain-core's base class puts every store in for
        ``similarity_search_by_vector``. A store that genuinely has no matches
        answers with an empty list, so the two are distinguishable.

        Raises:
            StoreError: When the method exists, runs, and fails.
        """
        try:
            # `filter=` is passed only when there is one. Several integrations
            # take no `**kwargs` at all (both PGVectors, Cassandra), so handing
            # them `filter=None` is a TypeError rather than a no-op.
            results = method(query, k=k, filter=where) if where is not None else method(query, k=k)
        except NotImplementedError:
            return None
        except Exception as exc:
            raise StoreError(
                f"The {self._name} query failed.",
                hint="Check the filter expression, if you passed one.",
                context={"store_backend": self._name},
                cause=exc,
            ) from exc
        return results

    def _hits(self, results: Any) -> list[Hit]:
        """Map LangChain's answer onto rebasis hits.

        The answer is a list of documents, or of tuples whose first element is
        the document and whose second is a number. Longer tuples occur — some
        integrations append the raw embedding — so the shape is read by position
        rather than unpacked.
        """
        hits: list[Hit] = []
        for rank, item in enumerate(results):
            if isinstance(item, tuple):
                document, raw = (item[0], item[1] if len(item) > 1 else None)
            else:
                document, raw = item, None
            hits.append(Hit(id=self._document_id(document), score=self._score(raw), rank=rank))
        return hits

    def _document_id(self, document: Any) -> str:
        """A document's id, from the field or from its metadata.

        Raises:
            StoreError: When it has neither. An earlier version fell back to the
                document's position in the result list, which produced a Hit
                whose id matched no record in the collection — and rebasis
                matches hits to records by id, so every measurement taken
                through this bridge would have been silently meaningless rather
                than visibly broken.
        """
        identifier = getattr(document, "id", None)
        if not identifier:
            metadata = getattr(document, "metadata", None) or {}
            identifier = metadata.get("id")
        if not identifier:
            raise StoreError(
                f"{self._name} returned documents that carry no id.",
                hint=(
                    "rebasis matches a hit to a record by id. Store one in the "
                    "document metadata under `id`, or use a native rebasis "
                    "backend for this store."
                ),
                context={"store_backend": self._name},
            )
        return str(identifier)

    def _score(self, raw: Any) -> float:
        """Read the number a by-vector search returned, or decline to.

        ``0.0`` unless the caller said which direction the number runs in. The
        two scored methods this bridge calls are defined by integrations rather
        than by langchain-core, and they disagree: Chroma's
        ``similarity_search_by_vector_with_relevance_scores`` returns a distance
        where lower is closer, under a name that says the opposite. Passing that
        through as a similarity inverts the ranking of every score rebasis
        derives from it, silently. Reporting no score at all is worse for the
        caller who has a real relevance score and better for everyone else,
        which is why ``score_kind=`` exists.
        """
        if raw is None or self._score_kind is None:
            return 0.0
        if self._score_kind == "distance":
            return 1.0 - float(raw)
        return float(raw)

    def upsert_vectors(self, ids: Sequence[str], vectors: FloatArray) -> None:
        """Not supported through this interface.

        LangChain has ``add_embeddings`` on some integrations but no portable
        in-place *replace*, and adding would duplicate records rather than
        update them — which is worse than refusing.

        Raises:
            CapabilityMissing: Always.
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

    def rebuild_index(self) -> None:
        """Not supported through this interface.

        The store behind the bridge may well have a search structure worth
        rebuilding — a migration can cost real recall while every vector stays
        correct, which is what `docs/index-health.md` measures. LangChain
        exposes no way to reach it, and the protocol requires the refusal to
        arrive as :class:`CapabilityMissing` rather than as the ``AttributeError``
        a missing method would raise.

        Raises:
            CapabilityMissing: Always.
        """
        from rebasis.store.base import require_capability

        require_capability(self, "can_rebuild_index", operation="rebuilding the index")

    def _readable(self) -> bool:
        """Whether records can be streamed out of the wrapped store at all."""
        return self._is_readable(self._collection())

    @staticmethod
    def _is_readable(collection: Any) -> bool:
        """Whether a handle is one this bridge knows how to page through."""
        return collection is not None and callable(getattr(collection, "get", None))

    def _collection(self) -> Any:
        """The wrapped store's native collection handle, or None.

        Read defensively because ``_collection`` is not always an attribute:
        ``langchain_chroma.Chroma`` makes it a property that raises ``ValueError``
        when the collection has not been initialised. A ``getattr`` default
        catches only ``AttributeError``, so that exception used to escape the
        ``capabilities`` property — a third-party error crossing the boundary
        from the one call whose entire job is to answer a question safely.
        """
        try:
            return getattr(self._store, "_collection", None)
        except Exception:  # noqa: BLE001 - a handle that refuses to be read is one we do not have
            return None

    def _embedder(self) -> Any:
        """The wrapped store's embedding model, or None where there is none.

        Read defensively for the same reason as :meth:`_collection`, and
        accepted only when it has ``embed_query``. LangChain's own retriever
        guards the same lookup with ``isinstance(..., Embeddings)``, for a real
        reason — ``langchain_community``'s FAISS keeps a public
        ``embedding_function`` that may hold a bare callable — but a bridge that
        never imports the framework has no class to check against. Asking for
        the one method it is about to call is the narrower test anyway: it
        excludes the bare callable *and* an ``Embeddings`` subclass that has not
        implemented it.
        """
        for name in _EMBEDDER_ATTRIBUTES:
            try:
                candidate = getattr(self._store, name, None)
            # BLE001/S112: a property that refuses is one we do not have, and
            # this loop's whole job is to keep looking. Nothing is logged
            # because nothing has gone wrong — six attribute names are probed
            # and at most one of them is expected to answer.
            except Exception:  # noqa: BLE001, S112
                continue
            if candidate is not None and callable(getattr(candidate, "embed_query", None)):
                return candidate
        return None


def _records_from_page(page: dict[str, Any], include: list[str]) -> Iterator[Record]:
    """Turn one page of Chroma's ``get`` mapping into records.

    ``include`` is both the projection that was asked for and the projection
    that is read back, so the two cannot drift apart. Lengths are checked per
    index rather than assumed equal: what was asked for is not always what came
    back, and an ``IndexError`` from inside a stream is a third-party-shaped
    failure in the middle of somebody's job.
    """
    embeddings = page.get("embeddings") if "embeddings" in include else None
    documents = page.get("documents") if "documents" in include else None
    for index, record_id in enumerate(page.get("ids") or []):
        yield Record(
            id=str(record_id),
            vector=(
                as_float32(embeddings[index])
                if embeddings is not None and len(embeddings) > index
                else None
            ),
            text=(documents[index] if documents is not None and len(documents) > index else None),
        )
