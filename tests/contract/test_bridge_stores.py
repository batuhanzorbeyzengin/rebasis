"""Bridge store contract.

`test_vector_store.py` runs every *registered* backend through one suite. The two
bridges are not registered — they wrap an object the caller already has rather
than a URI rebasis can open — so they were the only backends in the project with
no tests at all, which `README.md` and `ROADMAP.md` both said out loud. This file
holds them to the same standard, and the standard's own words apply here with
more force than they do to a native backend:

    laziness, because a materialising ``iter_records`` breaks the memory
    invariant only on corpora large enough that nobody notices in development,
    and truthful capabilities, because a store that claims more than it can do
    fails halfway through a migration instead of at second zero.

More force, because a bridge's capabilities are **discovered by inspection of a
foreign object** rather than known. Every ``hasattr`` in those two modules is a
claim about somebody else's class, and a dependency bump is exactly the event
that makes such a claim false without making it noisy.

## What the fakes are grounded in

The fakes here mirror the *libraries*, not the bridges. A fake built from a
reading of the bridge passes by construction and keeps passing after the real
interface has moved, which is the failure this file exists to prevent. Each one
was written against the upstream source at a named version:

* **langchain-core** — `libs/core/langchain_core/vectorstores/base.py`,
  `vectorstores/in_memory.py`, `documents/base.py` and
  `embeddings/embeddings.py`, read at tags ``langchain-core==1.6.0`` and
  ``langchain-core==0.3.86``. Every signature, return type and
  ``NotImplementedError`` below is identical at both. Worth recording:
  `pyproject.toml` declares the extra as ``langchain-core>=0.3`` while the
  version that resolves today, and the one installed where this suite is run,
  is a **1.x major** — so this bridge already duck-types across a major version
  boundary of its host framework.
* **langchain-chroma** — `langchain_chroma/vectorstores.py` at ``1.1.0``, for the
  shape of the one integration the LangChain bridge is really written against.
* **llama-index-core** — `llama-index-core/llama_index/core/vector_stores/types.py`
  and `.../simple.py` at tag ``v0.14.24``, which is also the installed version.

The facts that shaped the fakes, each of which contradicted an inference one of
the bridges was making:

1. ``VectorStore`` has exactly two abstract members, ``similarity_search`` and
   ``from_texts``. ``similarity_search_by_vector`` is **concrete on the base
   class with a body of ``raise NotImplementedError``** — so it is present on
   every LangChain store ever written, and ``hasattr`` cannot tell a store that
   searches by vector from one that cannot.
2. ``add_embeddings`` and both scored by-vector methods are **not on the base
   class at all**. Of ``langchain_community``'s 88 stores, 29 define
   ``similarity_search_with_score_by_vector`` and 6 define
   ``similarity_search_by_vector_with_relevance_scores``; FAISS and PGVector have
   the first, Chroma has the second, and their signatures do not converge past
   ``(embedding, k)``.
3. The float those methods return has **no portable direction**. Chroma's
   ``..._with_relevance_scores`` returns a raw distance — "lower score represents
   more similarity", in its own docstring, under a name saying the opposite —
   while Supabase and Elasticsearch return higher-is-better numbers from the
   same family, and FAISS, Rockset and LanceDB flip direction on a constructor
   argument. Exactly one LangChain method has a documented and enforced
   directional contract, ``similarity_search_with_relevance_scores``, and it
   takes a query *string*, so a bridge that searches by vector cannot use it.
4. ``BasePydanticVectorStore.client`` is an **abstract property**, so every
   LlamaIndex store built on it has one — including ``SimpleVectorStore``, whose
   implementation is ``return`` and whose ``stores_text`` is ``False``.

## Two layers, and why both

The duck-typed fakes are the mandatory layer: they run everywhere, including the
core install CI uses, and they are the only way to pin the cases where capability
inference has to answer "no" — a real store gives you one shape, and the shapes
that matter are the ones it does not have.

The `pytest.importorskip` layer at the end drives the genuine
``InMemoryVectorStore`` and ``SimpleVectorStore``, both of which live in the
*core* packages and need no vendor, no server and no network. That layer is what
would actually notice the interface moving under the bridge. It skips cleanly
where the extras are absent, which is most places.
"""

from __future__ import annotations

import importlib.util
import itertools
import types
from typing import Any

import numpy as np
import pytest

from rebasis.core import l2_normalize
from rebasis.errors import (
    CapabilityMissing,
    CollectionNotFound,
    MissingDependency,
    RebasisError,
    StoreError,
)
from rebasis.store import LangChainStoreAdapter, LlamaIndexStoreAdapter, VectorStore

DIM = 8
N = 250

#: Smaller than N, so that anything paging actually pages more than once.
PAGE = 64


# ── LangChain fakes ───────────────────────────────────────────────────
#
# Written against langchain-core 1.6.0 and 0.3.86, which agree on all of it.


class FakeDocument:
    """``langchain_core.documents.Document``.

    Three fields matter to the bridge: ``id``, ``page_content`` and
    ``metadata``. ``id`` arrived in langchain-core 0.2.11 and is still
    ``str | None`` at 1.6.0 — the source calls it "optional at the moment" — so a
    document with no id is the ordinary case rather than an edge one, and the
    bridge is not allowed to assume one is there.
    """

    def __init__(
        self,
        page_content: str,
        *,
        document_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.page_content = page_content
        self.id = document_id
        self.metadata = metadata if metadata is not None else {}


class FakeEmbeddings:
    """``langchain_core.embeddings.Embeddings``.

    Both methods are abstract upstream, and note the parameter names differ
    between them — ``texts`` plural, ``text`` singular. Counts its calls, because
    the dimension probe is a billed request against a hosted model and "how many
    times" is the whole question.
    """

    def __init__(self, dimension: int = DIM) -> None:
        self.dimension = dimension
        self.query_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        del text
        self.query_calls += 1
        return [0.0] * self.dimension


class PositionalEmbeddings(FakeEmbeddings):
    """An embedder whose output depends on the text, so a search can rank.

    A constant vector makes every document equidistant, and a test asserting a
    ranking over ties passes by luck rather than by correctness.
    """

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        vector = [0.0] * self.dimension
        vector[len(text) % self.dimension] = 1.0
        return vector


class FakeChromaCollection:
    """The native handle a LangChain Chroma store keeps behind ``_collection``.

    Mirrors ``chromadb``'s ``Collection.get``: the ``ids`` / ``limit`` /
    ``offset`` / ``include`` window, a mapping whose unrequested projections come
    back as ``None``, embeddings as a numpy array rather than a list, and — the
    behaviour that matters most — **an unknown id answered by omitting it**
    rather than by raising. Silently dropping a requested record is how a
    migration loses rows without failing.
    """

    def __init__(self, ids: list[str], vectors: Any, texts: list[str]) -> None:
        self.ids = ids
        self.vectors = vectors
        self.texts = texts
        self._index = {record_id: i for i, record_id in enumerate(ids)}
        #: Every window this handle was asked for, in order.
        self.windows: list[tuple[int | None, int | None]] = []
        #: The largest number of records handed out in one call.
        self.largest_page = 0

    def count(self) -> int:
        return len(self.ids)

    def get(  # noqa: PLR0913, PLR0917 - chromadb's own signature; matching it is the point
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        where_document: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        del where, where_document
        self.windows.append((limit, offset))
        wanted = set(include or ())
        positions = self._positions(ids, limit=limit, offset=offset)
        self.largest_page = max(self.largest_page, len(positions))
        return {
            "ids": [self.ids[p] for p in positions],
            "embeddings": (
                np.asarray([self.vectors[p] for p in positions]) if "embeddings" in wanted else None
            ),
            "documents": [self.texts[p] for p in positions] if "documents" in wanted else None,
        }

    def _positions(
        self, ids: list[str] | None, *, limit: int | None, offset: int | None
    ) -> list[int]:
        if ids is not None:
            return [self._index[i] for i in ids if i in self._index]
        start = offset or 0
        stop = len(self.ids) if limit is None else start + limit
        return list(range(len(self.ids)))[start:stop]


class WindowBlindCollection(FakeChromaCollection):
    """A handle that accepts ``limit`` and ``offset`` and ignores them.

    Not hypothetical: a wrapper forwarding ``**kwargs`` to a client that takes no
    window swallows both silently. Paging against it would return page one
    forever, which is worse than either failing or materialising.
    """

    def _positions(
        self, ids: list[str] | None, *, limit: int | None, offset: int | None
    ) -> list[int]:
        del limit, offset
        return super()._positions(ids, limit=None, offset=None)


class LangChainStore:
    """What ``langchain_core.vectorstores.VectorStore`` actually guarantees.

    Only what the base class gives every subclass, and nothing more: the abstract
    ``similarity_search``, and the concrete ``similarity_search_by_vector`` whose
    body upstream is literally ``raise NotImplementedError``. No ``_collection``,
    no ``_client``, and an ``embeddings`` property that returns ``None`` and logs
    at DEBUG — a supported state rather than an error.

    This is the *floor*. Any capability the bridge declares for an instance of
    this class is one it invented.
    """

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[FakeDocument]:
        del query, k, kwargs
        return []

    def similarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[FakeDocument]:
        del embedding, k, kwargs
        raise NotImplementedError

    @property
    def embeddings(self) -> Any:
        return None


class ChromaLikeStore(LangChainStore):
    """``langchain_chroma.Chroma`` at 1.1.0, in the shape the bridge reads.

    ``_collection`` is a **property**, not a plain attribute, and it raises
    ``ValueError`` when the collection has not been initialised. A fake that set
    ``self._collection = ...`` would miss the one thing about this class most
    likely to break a bridge that reaches for private handles.
    """

    def __init__(self, collection: FakeChromaCollection | None, embedder: Any = None) -> None:
        self._chroma_collection = collection
        self._client = object()
        self._embedding_function = embedder

    @property
    def _collection(self) -> FakeChromaCollection:
        if self._chroma_collection is None:
            msg = "Chroma collection not initialized. Use `reset_collection` to re-create it."
            raise ValueError(msg)
        return self._chroma_collection

    @property
    def embeddings(self) -> Any:
        return self._embedding_function

    def similarity_search_by_vector_with_relevance_scores(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,  # noqa: A002 - LangChain's own parameter name
        **kwargs: Any,
    ) -> list[tuple[FakeDocument, float]]:
        """Chroma's own, and the float is a **distance**.

        Its docstring upstream reads "Lower score represents more similarity",
        under a method name saying the opposite. This one fact decides whether a
        bridge may pass a score through untouched.
        """
        del kwargs
        if filter is not None and "unsupported" in filter:
            msg = "unsupported filter"
            raise ValueError(msg)
        collection = self._collection
        distances = np.linalg.norm(collection.vectors - np.asarray(embedding), axis=1)
        order = np.argsort(distances)[:k]
        return [
            (
                FakeDocument(collection.texts[p], document_id=collection.ids[p]),
                float(distances[p]),
            )
            for p in order
        ]


class ClientOnlyStore(LangChainStore):
    """A store with a ``_client`` and no ``_collection``.

    Common — ``_client`` appears on eleven ``langchain_community`` stores. It
    pins the inference that used to accept a client in place of a collection and
    then refuse to iterate through it.
    """

    def __init__(self) -> None:
        self._client = object()


class FalsyCollectionStore(LangChainStore):
    """A store whose ``_collection`` is present and **falsy**.

    An empty collection defining ``__len__`` is falsy, and the declaration used
    to read ``getattr(store, "_collection", None) or getattr(store, "_client",
    None)`` — so an empty collection fell through to the client, or to nothing at
    all. A store holding no records is still a store that can be read.
    """

    def __init__(self, collection: FakeChromaCollection) -> None:
        self._collection = _Falsy(collection)


class _Falsy:
    """A collection handle that is ``False`` in a boolean context."""

    def __init__(self, inner: FakeChromaCollection) -> None:
        self._inner = inner

    def __bool__(self) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class AddEmbeddingsStore(LangChainStore):
    """A store defining ``add_embeddings``.

    ``langchain_community``'s FAISS and PGVector both do, with **different
    signatures** — FAISS takes ``(text, embedding)`` pairs, PGVector takes texts
    and embeddings as parallel lists — and neither replaces a vector in place.
    The presence of the name is what ``can_upsert_vectors`` used to be read from.
    """

    def __init__(self) -> None:
        self.added: list[tuple[str, list[float]]] = []

    def add_embeddings(
        self,
        text_embeddings: Any,
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        del metadatas, kwargs
        self.added.extend(text_embeddings)
        return ids or []


class PublicEmbeddingFunctionStore(LangChainStore):
    """A store keeping its model on a public ``embedding_function``.

    ``langchain_community``'s FAISS does this, and its own ``embeddings``
    property returns the value only when it is an ``Embeddings`` — so the
    attribute may hold a bare callable with no ``embed_query`` at all.
    """

    def __init__(self, embedder: Any) -> None:
        self.embedding_function = embedder


class PrivateEmbeddingStore(LangChainStore):
    """A store keeping its model on a private ``_embedding``.

    The most common name of the five, at 22 of ``langchain_community``'s 88
    stores, and one LangChain's own retriever does not look at.
    """

    def __init__(self, embedder: Any) -> None:
        self._embedding = embedder


class UnscoredStore(LangChainStore):
    """A store searching by vector and returning documents with no scores.

    ``similarity_search_by_vector`` returning ``list[Document]`` is the base
    class's own signature, and most integrations implement only that.
    """

    def __init__(self, ids: list[str], vectors: Any, texts: list[str]) -> None:
        self.ids = ids
        self.vectors = vectors
        self.texts = texts

    def similarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[FakeDocument]:
        del kwargs
        distances = np.linalg.norm(self.vectors - np.asarray(embedding), axis=1)
        order = np.argsort(distances)[:k]
        return [FakeDocument(self.texts[p], document_id=self.ids[p]) for p in order]


class StrictSignatureStore(LangChainStore):
    """A scored store taking no ``**kwargs``, and returning ``None`` for a score.

    Two real behaviours in one class, because they come from the same family of
    stores. ``langchain_community``'s PGVector, ``langchain_postgres``'s PGVector
    and Cassandra all declare their scored by-vector method with a closed
    signature, so a bridge forwarding anything extra gets a ``TypeError`` out of
    a third-party frame. And both PGVectors declare a return of
    ``list[tuple[Document, float]]`` while returning ``None`` in the float slot
    whenever their embedding function is unset — so the annotation is a lie the
    bridge has to survive.
    """

    def similarity_search_with_score_by_vector(
        self, embedding: list[float], k: int = 4
    ) -> list[tuple[FakeDocument, float | None]]:
        del embedding
        return [(FakeDocument("text", document_id=f"doc-{i}"), None) for i in range(k)]


class AnonymousDocumentStore(LangChainStore):
    """A store whose documents carry no id, in the field or in the metadata.

    The normal case before langchain-core 0.2.11 added ``Document.id``, and still
    the case for any store whose metadata does not happen to hold one.
    """

    def similarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[FakeDocument]:
        del embedding, kwargs
        return [FakeDocument(f"text {i}") for i in range(k)]


class MetadataIdStore(LangChainStore):
    """A store carrying its ids in document metadata, as older ones do."""

    def similarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[FakeDocument]:
        del embedding, kwargs
        return [FakeDocument("text", metadata={"id": f"doc-{i}"}) for i in range(k)]


class FlakyEmbedder:
    """An embedding model that fails once and then works.

    A hosted model being briefly unreachable is the ordinary case, not the
    exotic one.
    """

    def __init__(self) -> None:
        self.calls = 0

    def embed_query(self, text: str) -> list[float]:
        del text
        self.calls += 1
        if self.calls == 1:
            msg = "service unavailable"
            raise RuntimeError(msg)
        return [0.0] * DIM


# ── LlamaIndex fakes ──────────────────────────────────────────────────
#
# Written against llama-index-core v0.14.24.


class LlamaIndexStore:
    """``BasePydanticVectorStore``, in the shape the bridge reads.

    ``stores_text`` is a required field upstream and ``client`` is an **abstract
    property** — so every store built on that base has a ``client`` attribute,
    whatever it returns. Core's own ``SimpleVectorStore`` returns ``None`` from
    it and declares ``stores_text = False``.
    """

    is_embedding_query = True

    def __init__(self, *, stores_text: bool = True, client: Any = None) -> None:
        self.stores_text = stores_text
        self._client = client

    @property
    def client(self) -> Any:
        return self._client

    def query(self, query: Any, **kwargs: Any) -> Any:
        del query, kwargs
        return FakeQueryResult()


class CountingClient:
    """A native client that can report a record count, as Chroma's can."""

    def __init__(self, total: int) -> None:
        self._total = total

    def count(self) -> int:
        return self._total


class ArgumentHungryClient:
    """A native client whose ``count`` means something else.

    Qdrant's takes a collection name. Called with none it raises ``TypeError``
    from someone else's library, which must not reach the caller as itself.
    """

    def count(self, collection_name: str) -> int:
        del collection_name
        return 0


class RaisingClientStore(LlamaIndexStore):
    """A store whose ``client`` property fails when read.

    A lazily-connecting store does this when its service is unreachable. Asking
    a store what it can do must not be the thing that breaks.
    """

    @property
    def client(self) -> Any:
        msg = "not connected"
        raise RuntimeError(msg)


class FakeNode:
    """``BaseNode``: ``node_id`` is the accessor, ``id_`` the field behind it."""

    def __init__(self, node_id: str) -> None:
        self.id_ = node_id

    @property
    def node_id(self) -> str:
        return self.id_


class FakeQueryResult:
    """``VectorStoreQueryResult``: a dataclass whose three fields are all optional.

    Which of them a store fills differs — ``SimpleVectorStore`` returns ids and
    similarities and no nodes at all, others return nodes — so everything
    defaults to ``None`` here exactly as it does upstream.
    """

    def __init__(
        self,
        nodes: list[FakeNode] | None = None,
        similarities: list[float] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        self.nodes = nodes
        self.similarities = similarities
        self.ids = ids


# ── fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def corpus(rng: np.random.Generator) -> tuple[list[str], Any, list[str]]:
    """Ids, unit vectors and texts — the shape the store contract uses."""
    vectors = l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))
    return (
        [f"doc-{i}" for i in range(N)],
        vectors,
        [f"text of document {i}" for i in range(N)],
    )


@pytest.fixture
def collection(corpus: tuple[list[str], Any, list[str]]) -> FakeChromaCollection:
    """A populated Chroma-shaped handle."""
    return FakeChromaCollection(*corpus)


@pytest.fixture
def readable(collection: FakeChromaCollection) -> LangChainStoreAdapter:
    """A bridge over a store that can genuinely be read."""
    return LangChainStoreAdapter(ChromaLikeStore(collection), dimension=DIM)


# ── LangChain: capabilities ───────────────────────────────────────────


@pytest.mark.contract
@pytest.mark.parametrize(
    ("build", "expected"),
    [
        pytest.param(ChromaLikeStore, True, id="collection"),
        pytest.param(lambda _: ChromaLikeStore(None), False, id="collection-uninitialised"),
        pytest.param(lambda _: ClientOnlyStore(), False, id="client-only"),
        pytest.param(lambda _: LangChainStore(), False, id="bare"),
        pytest.param(FalsyCollectionStore, True, id="collection-falsy"),
        pytest.param(lambda _: AddEmbeddingsStore(), False, id="add-embeddings"),
    ],
)
def test_langchain_reads_are_declared_from_the_handle_that_performs_them(
    collection: FakeChromaCollection, build: Any, expected: bool
) -> None:
    """One handle answers both read capabilities, because one call yields both.

    Vectors and text come out of the same ``_collection.get``, so they cannot
    have different answers — and neither can be inferred from a
    ``similarity_search`` every LangChain store inherits, nor from a ``_client``
    this bridge never reads through.
    """
    capabilities = LangChainStoreAdapter(build(collection)).capabilities
    assert capabilities.can_read_vectors is expected
    assert capabilities.can_read_text is expected
    assert capabilities.name.startswith("langchain:")


@pytest.mark.contract
def test_langchain_never_declares_an_upsert_it_always_refuses(
    collection: FakeChromaCollection,
) -> None:
    """``add_embeddings`` appends; rebasis needs replacement.

    Declaring the capability from the presence of that method promised a write
    the next method down refuses unconditionally — the "fails halfway through a
    migration" shape exactly, and the easiest of all of them to see once asked.
    """
    for store in (AddEmbeddingsStore(), ChromaLikeStore(collection), LangChainStore()):
        adapter = LangChainStoreAdapter(store)
        assert adapter.capabilities.can_upsert_vectors is False
        with pytest.raises(CapabilityMissing) as raised:
            adapter.upsert_vectors(["doc-0"], np.ones((1, DIM), dtype=np.float32))
        assert raised.value.hint


@pytest.mark.contract
def test_langchain_does_not_claim_a_filter_contract_that_does_not_exist(
    readable: LangChainStoreAdapter,
) -> None:
    """``can_filter`` was the constant ``True`` and was never inferred at all.

    There is nothing to infer it from. Chroma takes a mapping, langchain-core's
    own ``InMemoryVectorStore`` takes a ``Callable[[Document], bool]`` and raises
    on a mapping, Milvus takes no filter argument. ``False`` is the answer
    inspection supports; ``search`` still forwards a filter the caller passes.
    """
    assert readable.capabilities.can_filter is False


@pytest.mark.contract
def test_langchain_capabilities_survive_a_handle_that_refuses_to_be_read() -> None:
    """A ``_collection`` property that raises must not raise *here*.

    ``langchain_chroma.Chroma`` makes ``_collection`` a property raising
    ``ValueError`` when the collection is not initialised, and ``getattr`` with a
    default swallows only ``AttributeError``. Asking a store what it can do is
    the one call that has to be answerable in every state.
    """
    adapter = LangChainStoreAdapter(ChromaLikeStore(None))
    assert adapter.capabilities.can_read_vectors is False
    with pytest.raises(CapabilityMissing):
        adapter.iter_records()


@pytest.mark.contract
def test_langchain_declares_a_name_a_reader_can_act_on(readable: LangChainStoreAdapter) -> None:
    """The declaration names the wrapped class, not merely "langchain"."""
    assert readable.capabilities.name == "langchain:ChromaLikeStore"


# ── LangChain: iteration ──────────────────────────────────────────────


@pytest.mark.contract
def test_langchain_iter_records_is_lazy(readable: LangChainStoreAdapter) -> None:
    """Streaming is the default, and it is not optional.

    The same assertion the store contract makes of every native backend, made
    the same way.
    """
    result = readable.iter_records()
    assert isinstance(result, types.GeneratorType), (
        "LangChainStoreAdapter.iter_records must be lazy, not materialise the collection"
    )


@pytest.mark.contract
def test_langchain_iter_records_refuses_before_the_first_next() -> None:
    """The refusal belongs to the call, not to the caller's loop.

    A generator function defers its whole body, so a capability check written
    inside one arrives from the middle of the consumer's iteration — precisely
    where a bridge that reports capabilities honestly promises not to fail.
    ``pytest.raises`` around the *call* is the assertion; wrapping ``next()``
    would have passed either way.
    """
    adapter = LangChainStoreAdapter(ClientOnlyStore())
    with pytest.raises(CapabilityMissing) as raised:
        adapter.iter_records()
    assert raised.value.hint


@pytest.mark.contract
def test_langchain_iter_records_pages_rather_than_loading_the_collection(
    collection: FakeChromaCollection,
) -> None:
    """Lazy in type is not lazy in memory.

    ``iter_records`` returned a generator and, on its first ``next()``, called
    ``get()`` with no window at all — every test checking the *type* passed, and
    peak memory was still ``O(N × d)``. This asserts the property those tests
    stand for: the handle is asked for windows, and never hands out more than one
    window's worth at a time.
    """
    adapter = LangChainStoreAdapter(ChromaLikeStore(collection), dimension=DIM)
    records = list(adapter.iter_records(batch_size=PAGE))

    assert len(records) == N
    assert collection.largest_page <= PAGE
    assert len(collection.windows) >= N // PAGE
    assert all(limit == PAGE for limit, _ in collection.windows)


@pytest.mark.contract
def test_langchain_iter_records_is_complete_and_unique(readable: LangChainStoreAdapter) -> None:
    """Every record exactly once — no gaps, no duplicates.

    Read with a window small enough to page several times, because the paging
    window is where a bridge loses or repeats rows.
    """
    ids = [
        r.id for r in readable.iter_records(with_vectors=False, with_text=False, batch_size=PAGE)
    ]
    assert len(ids) == N
    assert len(set(ids)) == N


@pytest.mark.contract
def test_langchain_iter_records_honours_its_projection_flags(
    readable: LangChainStoreAdapter,
) -> None:
    """Asking for no vectors must actually skip them."""
    record = next(readable.iter_records(with_vectors=False, with_text=False))
    assert record.vector is None
    assert record.text is None

    record = next(readable.iter_records(with_vectors=True, with_text=True))
    assert record.vector is not None
    assert record.vector.shape == (DIM,)
    assert record.text is not None


@pytest.mark.contract
def test_langchain_unknown_ids_raise_rather_than_being_skipped(
    readable: LangChainStoreAdapter,
) -> None:
    """Chroma answers an unknown id by omitting it.

    Passed through, that turns a migration's missing record into a shorter
    result nobody counts. The native chroma backend checks for it; the bridge
    reads through the same call and did not.
    """
    with pytest.raises(CollectionNotFound) as raised:
        list(readable.iter_records(["doc-0", "no-such-document"]))
    assert "no-such-document" in str(raised.value.hint)


@pytest.mark.contract
def test_langchain_refuses_a_handle_that_ignores_the_paging_window(
    corpus: tuple[list[str], Any, list[str]],
) -> None:
    """A window that is accepted and ignored would page forever.

    The second page comes back identical to the first, and one repetition is
    enough to know. Failing here beats both alternatives: looping without end,
    and silently reading the collection whole after being asked not to.
    """
    adapter = LangChainStoreAdapter(ChromaLikeStore(WindowBlindCollection(*corpus)))
    with pytest.raises(StoreError) as raised:
        list(adapter.iter_records(batch_size=PAGE))
    assert raised.value.hint


# ── LangChain: count and dimension ────────────────────────────────────


@pytest.mark.contract
def test_langchain_count_agrees_with_what_iteration_yields(
    readable: LangChainStoreAdapter,
) -> None:
    """The count and the stream must describe the same collection.

    Read from the same handle for that reason. The client fallback standing
    behind it answered a different question — a client counts collections, or
    needs telling which one — so it could return a number true of something and
    false of this.
    """
    assert readable.count() == N
    assert readable.count() == len(list(readable.iter_records(with_vectors=False, with_text=False)))


@pytest.mark.contract
@pytest.mark.parametrize(
    "store",
    [
        pytest.param(LangChainStore(), id="bare"),
        pytest.param(ClientOnlyStore(), id="client-only"),
    ],
)
def test_langchain_count_refuses_where_it_cannot_be_answered(store: Any) -> None:
    """No count is a refusal with a next step, never a guess."""
    with pytest.raises(CapabilityMissing) as raised:
        LangChainStoreAdapter(store).count()
    assert raised.value.hint


@pytest.mark.contract
def test_langchain_dimension_probes_the_model_at_most_once(
    collection: FakeChromaCollection,
) -> None:
    """The probe is a billed call, so it happens once or not at all.

    Against a hosted embedding API ``embed_query("dimension probe")`` is a paid
    request. Everything downstream asks a store its dimensionality freely, so
    "once per adapter" is the difference between a rounding error and a line on
    an invoice.
    """
    embedder = FakeEmbeddings()
    adapter = LangChainStoreAdapter(ChromaLikeStore(collection, embedder))

    assert adapter.dimension() == DIM
    assert adapter.dimension() == DIM
    assert adapter.dimension() == DIM
    assert embedder.query_calls == 1


@pytest.mark.contract
def test_langchain_dimension_given_at_construction_is_never_probed(
    collection: FakeChromaCollection,
) -> None:
    """``dimension=`` is how a caller spends nothing at all."""
    embedder = FakeEmbeddings()
    adapter = LangChainStoreAdapter(ChromaLikeStore(collection, embedder), dimension=DIM)

    assert adapter.dimension() == DIM
    assert embedder.query_calls == 0


@pytest.mark.contract
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(PublicEmbeddingFunctionStore, id="embedding_function"),
        pytest.param(PrivateEmbeddingStore, id="_embedding"),
    ],
)
def test_langchain_dimension_finds_the_model_wherever_it_is_kept(build: Any) -> None:
    """``embeddings`` is one name for the model; there are five others.

    Across ``langchain_community``'s 88 stores the model lives under
    ``_embedding`` 22 times, ``embedding_function`` 18, ``embedding`` 14,
    ``_embedding_function`` 7 and ``_embeddings`` 5. Looking at two of those
    names found the model on a minority of stores and refused on the rest.
    """
    assert LangChainStoreAdapter(build(FakeEmbeddings(dimension=16))).dimension() == 16


@pytest.mark.contract
def test_langchain_dimension_ignores_a_model_that_cannot_answer() -> None:
    """A bare callable under ``embedding_function`` is not an embedder.

    ``langchain_community``'s FAISS allows exactly that, and its own
    ``embeddings`` property declines to return it for the same reason.
    """
    adapter = LangChainStoreAdapter(PublicEmbeddingFunctionStore(lambda _text: [0.0] * DIM))
    with pytest.raises(CapabilityMissing) as raised:
        adapter.dimension()
    assert "dimension=" in str(raised.value.hint)


@pytest.mark.contract
def test_langchain_dimension_converts_a_failing_probe_and_does_not_cache_it() -> None:
    """A model that is briefly unreachable must stay retryable.

    Two properties, because they are one decision: the third-party exception is
    converted at the boundary, and the failure is *not* remembered — caching it
    would turn one bad minute into a permanently broken adapter.
    """
    embedder = FlakyEmbedder()
    adapter = LangChainStoreAdapter(PublicEmbeddingFunctionStore(embedder))

    with pytest.raises(StoreError):
        adapter.dimension()
    assert adapter.dimension() == DIM
    assert embedder.calls == 2


# ── LangChain: search ─────────────────────────────────────────────────


@pytest.mark.contract
def test_langchain_search_is_ranked_and_finds_an_exact_match_first(
    corpus: tuple[list[str], Any, list[str]],
) -> None:
    """Exactly k hits, ranked, with the query's own record at rank 0."""
    ids, vectors, texts = corpus
    adapter = LangChainStoreAdapter(UnscoredStore(ids, vectors, texts))

    hits = adapter.search(vectors[7], k=5)
    assert [h.rank for h in hits] == [0, 1, 2, 3, 4]
    assert hits[0].id == ids[7]


@pytest.mark.contract
def test_langchain_search_declines_to_report_a_score_it_cannot_read(
    collection: FakeChromaCollection,
) -> None:
    """A score whose direction is unknown is reported as no score.

    Chroma's ``similarity_search_by_vector_with_relevance_scores`` returns a
    **distance** — lower is closer — under a name saying the opposite, while
    Supabase and Elasticsearch return higher-is-better numbers from the same
    family and FAISS flips direction on a constructor argument. Nothing on the
    object says which. Passing the number through inverted every ranking derived
    from it, silently, for the most widely used LangChain store there is; ``0.0``
    is visibly absent rather than invisibly wrong.
    """
    hits = LangChainStoreAdapter(ChromaLikeStore(collection)).search(collection.vectors[3], k=5)

    assert hits[0].id == collection.ids[3]
    assert [h.score for h in hits] == [0.0] * 5


@pytest.mark.contract
def test_langchain_search_uses_the_score_when_told_which_way_it_runs(
    collection: FakeChromaCollection,
) -> None:
    """``score_kind=`` is the caller supplying what the object cannot.

    The same escape hatch as ``dimension=``, for the same reason: whoever wraps
    their own store knows what its numbers mean and the bridge does not. Told it
    is a distance, the bridge converts, and the order of the scores then agrees
    with the order of the hits.
    """
    adapter = LangChainStoreAdapter(ChromaLikeStore(collection), score_kind="distance")

    hits = adapter.search(collection.vectors[3], k=5)
    assert hits[0].id == collection.ids[3]
    assert hits[0].score == pytest.approx(1.0)
    assert all(a.score >= b.score for a, b in itertools.pairwise(hits))


@pytest.mark.contract
def test_langchain_search_survives_a_score_that_is_missing(
    corpus: tuple[list[str], Any, list[str]],
) -> None:
    """Both PGVectors return ``None`` where their annotation says ``float``.

    ``result.distance if self.embedding_function is not None else None``, in
    both of them. Asking for the score is not optional for the bridge — the
    tuple is the shape it reads — so ``float(None)`` would be a ``TypeError``
    from the middle of a result mapping, and asking for the score kind does not
    make one appear.
    """
    _, vectors, _ = corpus
    adapter = LangChainStoreAdapter(StrictSignatureStore(), score_kind="distance")

    hits = adapter.search(vectors[0], k=3)
    assert [h.id for h in hits] == ["doc-0", "doc-1", "doc-2"]
    assert [h.score for h in hits] == [0.0, 0.0, 0.0]


@pytest.mark.contract
def test_langchain_search_converts_a_closed_signature_refusing_a_filter(
    corpus: tuple[list[str], Any, list[str]],
) -> None:
    """A store that takes no filter must say so, not silently return everything.

    Community PGVector, langchain_postgres PGVector and Cassandra declare their
    scored by-vector method with no ``**kwargs`` at all. Forwarding a filter to
    one of those is a ``TypeError`` in a third-party frame, and the boundary
    converts it.
    """
    _, vectors, _ = corpus
    adapter = LangChainStoreAdapter(StrictSignatureStore())

    with pytest.raises(StoreError) as raised:
        adapter.search(vectors[0], k=3, where={"source": "a"})
    assert isinstance(raised.value.__cause__, TypeError)


@pytest.mark.contract
def test_langchain_search_never_invents_an_id(corpus: tuple[list[str], Any, list[str]]) -> None:
    """A hit rebasis cannot match to a record is not a usable hit.

    The fallback used to be the document's position in the result list, which
    produced ids like ``"0"`` matching nothing at all. Every measurement taken
    through such a hit is meaningless, and nothing anywhere would have said so.
    """
    _, vectors, _ = corpus
    adapter = LangChainStoreAdapter(AnonymousDocumentStore())

    with pytest.raises(StoreError) as raised:
        adapter.search(vectors[0], k=3)
    assert raised.value.hint


@pytest.mark.contract
def test_langchain_search_reads_an_id_out_of_the_metadata(
    corpus: tuple[list[str], Any, list[str]],
) -> None:
    """Where the document has no ``id``, the metadata is the documented place."""
    _, vectors, _ = corpus
    hits = LangChainStoreAdapter(MetadataIdStore()).search(vectors[0], k=2)
    assert [h.id for h in hits] == ["doc-0", "doc-1"]


@pytest.mark.contract
def test_langchain_search_by_vector_absence_is_reported_as_absence() -> None:
    """The base class's stub must not read as a failed query.

    ``similarity_search_by_vector`` is concrete on ``VectorStore`` with a body of
    ``raise NotImplementedError``, so it is present on every store and refuses on
    most. Caught as a query failure it produced "the query failed" with a hint
    about a filter the caller never passed; the truthful answer is that this
    store cannot search by vector at all.
    """
    with pytest.raises(CapabilityMissing) as raised:
        LangChainStoreAdapter(LangChainStore()).search(np.zeros(DIM, dtype=np.float32), k=3)
    assert "by vector" in str(raised.value)


@pytest.mark.contract
def test_langchain_search_converts_a_third_party_failure(
    collection: FakeChromaCollection,
) -> None:
    """No third-party exception crosses the module boundary.

    The user must get RB-Exxxx with a next step, not whatever the client library
    happened to raise — and the original stays reachable as ``__cause__``.
    """
    adapter = LangChainStoreAdapter(ChromaLikeStore(collection))
    with pytest.raises(RebasisError) as raised:
        adapter.search(collection.vectors[0], k=3, where={"unsupported": True})
    assert isinstance(raised.value.__cause__, ValueError)


# ── LlamaIndex ────────────────────────────────────────────────────────


@pytest.mark.contract
@pytest.mark.parametrize(
    "store",
    [
        pytest.param(LlamaIndexStore(stores_text=True, client=object()), id="text-and-client"),
        pytest.param(LlamaIndexStore(stores_text=False, client=None), id="neither"),
        pytest.param(RaisingClientStore(), id="client-raises"),
    ],
)
def test_llamaindex_declares_only_what_the_bridge_implements(store: Any) -> None:
    """Every read capability is False, whatever the wrapped store holds.

    ``client`` is an abstract property on ``BasePydanticVectorStore``, so its
    presence proves nothing — and ``stores_text`` says the text is in the store,
    not that this bridge can fetch it. It can fetch neither, so both answers are
    False and the store's own declarations are irrelevant to that.
    """
    capabilities = LlamaIndexStoreAdapter(store).capabilities
    assert capabilities.can_read_vectors is False
    assert capabilities.can_read_text is False
    assert capabilities.can_upsert_vectors is False
    assert capabilities.can_filter is False
    assert capabilities.name.startswith("llamaindex:")


@pytest.mark.contract
@pytest.mark.parametrize(
    ("with_vectors", "with_text"), [(True, True), (False, True), (True, False), (False, False)]
)
def test_llamaindex_iter_records_refuses_every_projection(
    with_vectors: bool, with_text: bool
) -> None:
    """Ids alone are as unreachable as vectors are.

    The ids-only case used to return an empty iterator, the worst of the three
    available answers: a caller cannot tell an empty collection from one that
    cannot be enumerated, and a migration reading zero records out of a full
    store looks exactly like success.
    """
    adapter = LlamaIndexStoreAdapter(LlamaIndexStore(), dimension=DIM)
    with pytest.raises(CapabilityMissing) as raised:
        adapter.iter_records(with_vectors=with_vectors, with_text=with_text)
    assert raised.value.hint


@pytest.mark.contract
def test_llamaindex_count_comes_from_the_native_client() -> None:
    """Where the client can count, that is the honest answer."""
    assert LlamaIndexStoreAdapter(LlamaIndexStore(client=CountingClient(N))).count() == N


@pytest.mark.contract
@pytest.mark.parametrize(
    ("store", "expected"),
    [
        pytest.param(LlamaIndexStore(client=None), CapabilityMissing, id="no-client"),
        pytest.param(RaisingClientStore(), CapabilityMissing, id="client-raises"),
        pytest.param(
            LlamaIndexStore(client=ArgumentHungryClient()),
            StoreError,
            id="count-means-something-else",
        ),
    ],
)
def test_llamaindex_count_refuses_rather_than_guessing(store: Any, expected: type) -> None:
    """A client whose ``count`` answers a different question is not a count.

    Qdrant's takes a collection name. Called with none it raises ``TypeError``,
    converted at the boundary like any other third-party failure rather than
    escaping as itself.
    """
    with pytest.raises(expected) as raised:
        LlamaIndexStoreAdapter(store).count()
    assert raised.value.hint


@pytest.mark.contract
def test_llamaindex_dimension_comes_only_from_the_caller() -> None:
    """There is nothing to probe, so there is nothing to guess."""
    assert LlamaIndexStoreAdapter(LlamaIndexStore(), dimension=DIM).dimension() == DIM
    with pytest.raises(CapabilityMissing) as raised:
        LlamaIndexStoreAdapter(LlamaIndexStore()).dimension()
    assert "dimension=" in str(raised.value.hint)


@pytest.mark.contract
def test_llamaindex_refuses_a_filter_it_cannot_translate() -> None:
    """rebasis passes a mapping; LlamaIndex reads attributes off a model.

    ``VectorStoreQuery`` is a plain dataclass, so it accepts the mapping without
    complaint and the store fails on ``.filters``/``.condition`` much later. A
    filter quietly accepted and then dropped or fatal is what a truthful
    capability declaration exists to prevent, so ``can_filter`` is False and the
    call says so before anything is sent.
    """
    adapter = LlamaIndexStoreAdapter(LlamaIndexStore(), dimension=DIM)
    with pytest.raises(CapabilityMissing) as raised:
        adapter.search(np.zeros(DIM, dtype=np.float32), k=3, where={"source": "a"})
    assert raised.value.hint


@pytest.mark.contract
def test_llamaindex_search_without_the_package_names_the_dependency() -> None:
    """A missing package is a usage problem, not a missing store capability.

    It was reported as ``CapabilityMissing``, which exits 3 and tells the user
    their store cannot do something — when in fact their store is fine and one
    ``pip install`` fixes it. ``MissingDependency`` exits 2 and says so, as every
    native backend already does for its own client library.
    """
    if importlib.util.find_spec("llama_index") is not None:
        pytest.skip("llama-index-core is installed; this pins the absent case")

    adapter = LlamaIndexStoreAdapter(LlamaIndexStore(), dimension=DIM)
    with pytest.raises(MissingDependency) as raised:
        adapter.search(np.zeros(DIM, dtype=np.float32), k=3)
    assert "llama-index-core" in str(raised.value)
    assert "rebasis[llamaindex]" in str(raised.value.hint)


@pytest.mark.contract
@pytest.mark.parametrize(
    ("result", "expected_ids", "expected_scores"),
    [
        pytest.param(
            FakeQueryResult(ids=["a", "b"], similarities=[0.9, 0.5]),
            ["a", "b"],
            [0.9, 0.5],
            id="ids-and-similarities",
        ),
        pytest.param(
            FakeQueryResult(nodes=[FakeNode("a"), FakeNode("b")], similarities=[0.9, 0.5]),
            ["a", "b"],
            [0.9, 0.5],
            id="nodes-only",
        ),
        pytest.param(
            FakeQueryResult(ids=["a", "b", "c"], similarities=[0.9]),
            ["a", "b", "c"],
            [0.9, 0.0, 0.0],
            id="fewer-scores-than-ids",
        ),
        pytest.param(FakeQueryResult(ids=["a"]), ["a"], [0.0], id="no-similarities"),
        pytest.param(FakeQueryResult(), [], [], id="empty"),
    ],
)
def test_llamaindex_maps_every_result_shape_without_dropping_a_hit(
    result: FakeQueryResult, expected_ids: list[str], expected_scores: list[float]
) -> None:
    """All three fields of a ``VectorStoreQueryResult`` are optional.

    Stores differ in which they fill — core's own ``SimpleVectorStore`` returns
    ids and similarities and leaves ``nodes`` empty — so ids come from whichever
    is present. Zipping ids against similarities dropped the unscored tail when a
    store returned fewer of the latter, and a missing score is not a missing
    result.

    Called directly rather than through ``search`` because ``search`` needs
    llama-index-core imported and this mapping does not. The whole path is
    covered by the real-library layer at the end of this file.
    """
    adapter = LlamaIndexStoreAdapter(LlamaIndexStore(), dimension=DIM)
    hits = adapter._hits(result)

    assert [h.id for h in hits] == expected_ids
    assert [h.score for h in hits] == expected_scores
    assert [h.rank for h in hits] == list(range(len(expected_ids)))


@pytest.mark.contract
def test_llamaindex_refuses_results_whose_hits_cannot_be_identified() -> None:
    """Nodes with no id give nothing to match a record against."""
    adapter = LlamaIndexStoreAdapter(LlamaIndexStore(), dimension=DIM)
    with pytest.raises(StoreError) as raised:
        adapter._hits(FakeQueryResult(nodes=[FakeNode("")], similarities=[0.9]))
    assert raised.value.hint


# ── both bridges ──────────────────────────────────────────────────────


@pytest.mark.contract
@pytest.mark.parametrize(
    "adapter",
    [
        pytest.param(LangChainStoreAdapter(LangChainStore()), id="langchain"),
        pytest.param(LlamaIndexStoreAdapter(LlamaIndexStore()), id="llamaindex"),
    ],
)
def test_bridges_satisfy_the_store_protocol(adapter: Any) -> None:
    """A bridge is a backend, so it answers every call a backend answers.

    Both were missing ``rebuild_index`` entirely, which made them fail this check
    and made the call raise ``AttributeError`` — a Python error about a missing
    method rather than a rebasis error about a missing capability.
    """
    assert isinstance(adapter, VectorStore)


@pytest.mark.contract
@pytest.mark.parametrize(
    "adapter",
    [
        pytest.param(LangChainStoreAdapter(LangChainStore()), id="langchain"),
        pytest.param(LlamaIndexStoreAdapter(LlamaIndexStore()), id="llamaindex"),
    ],
)
def test_bridges_refuse_to_rebuild_an_index_they_cannot_reach(adapter: Any) -> None:
    """Either the backend can rebuild its index or it says so.

    Neither can. What matters is that the refusal arrives as the error the
    protocol documents, with a next step, rather than as an ``AttributeError``
    from the middle of `migrate`'s recovery path — the moment a user is least
    able to absorb one.
    """
    assert adapter.capabilities.can_rebuild_index is False
    with pytest.raises(CapabilityMissing) as raised:
        adapter.rebuild_index()
    assert raised.value.hint


# ── the real libraries, where they are installed ──────────────────────
#
# Everything above runs anywhere, including the core install CI's default job
# uses. Everything below needs the optional extras, and is the layer that would
# actually notice the interface moving under the bridge.


@pytest.mark.contract
def test_real_langchain_in_memory_store_is_declared_unreadable() -> None:
    """langchain-core's own reference store cannot be read back, and says so.

    ``InMemoryVectorStore`` holds every vector in a dict and exposes neither
    ``_collection`` nor ``_client``, so the bridge's answer is "no" — from the
    real class, at whatever version is installed, rather than from a fake built
    to produce that answer. This is what a dependency bump would change quietly:
    a release growing either attribute would flip the declaration without anyone
    deciding to.
    """
    pytest.importorskip("langchain_core", reason="the langchain extra is not installed")
    from langchain_core.vectorstores import InMemoryVectorStore

    adapter = LangChainStoreAdapter(InMemoryVectorStore(embedding=FakeEmbeddings()))
    capabilities = adapter.capabilities

    assert capabilities.can_read_vectors is False
    assert capabilities.can_read_text is False
    assert capabilities.can_upsert_vectors is False
    with pytest.raises(CapabilityMissing):
        adapter.iter_records()
    with pytest.raises(CapabilityMissing):
        adapter.count()


@pytest.mark.contract
def test_real_langchain_in_memory_store_answers_the_dimension_probe() -> None:
    """The probe path, against the real ``embeddings`` property.

    ``InMemoryVectorStore`` overrides the base's property to return the model it
    was constructed with, so this exercises the branch that costs money in
    production — and confirms it costs exactly one call.
    """
    pytest.importorskip("langchain_core", reason="the langchain extra is not installed")
    from langchain_core.vectorstores import InMemoryVectorStore

    embedder = FakeEmbeddings(dimension=16)
    adapter = LangChainStoreAdapter(InMemoryVectorStore(embedding=embedder))

    assert adapter.dimension() == 16
    assert adapter.dimension() == 16
    assert embedder.query_calls == 1


@pytest.mark.contract
def test_real_langchain_in_memory_store_searches_by_vector() -> None:
    """The one operation that does work, driven through the real class.

    ``InMemoryVectorStore`` implements ``similarity_search_with_score_by_vector``
    for real rather than inheriting the base class's refusal, and it carries a
    ``Document.id`` — so hits come back identified, which is what rebasis needs
    and what the bridge is no longer allowed to fake. The scores are real cosine
    similarities and are still reported as ``0.0``, because the bridge is not
    told that and cannot ask.
    """
    pytest.importorskip("langchain_core", reason="the langchain extra is not installed")
    from langchain_core.vectorstores import InMemoryVectorStore

    embedder = PositionalEmbeddings()
    store = InMemoryVectorStore(embedding=embedder)
    store.add_texts(["alpha", "beta", "gamma"], ids=["a", "b", "c"])

    query = np.asarray(embedder.embed_query("beta"), dtype=np.float32)
    hits = LangChainStoreAdapter(store, dimension=DIM).search(query, k=3)

    assert [h.rank for h in hits] == [0, 1, 2]
    assert hits[0].id == "b"
    assert [h.score for h in hits] == [0.0, 0.0, 0.0]


@pytest.mark.contract
def test_real_llamaindex_simple_store_declares_nothing_it_cannot_do() -> None:
    """core's own reference store, through the real class.

    ``SimpleVectorStore`` declares ``stores_text = False`` and its ``client``
    property is a bare ``return``. Both were inputs to the declarations this
    bridge used to make, and both pointed the wrong way — the old code read "no
    vectors" here for a reason that had nothing to do with what the bridge does,
    and "yes vectors" for any store whose client happens not to be None.
    """
    pytest.importorskip("llama_index.core", reason="the llamaindex extra is not installed")
    from llama_index.core.vector_stores import SimpleVectorStore

    adapter = LlamaIndexStoreAdapter(SimpleVectorStore(), dimension=DIM)
    capabilities = adapter.capabilities

    assert capabilities.can_read_vectors is False
    assert capabilities.can_read_text is False
    assert capabilities.can_filter is False
    with pytest.raises(CapabilityMissing):
        adapter.iter_records()
    with pytest.raises(CapabilityMissing):
        adapter.count()


@pytest.mark.contract
def test_real_llamaindex_simple_store_searches_and_returns_identified_hits(
    corpus: tuple[list[str], Any, list[str]],
) -> None:
    """A real ``VectorStoreQuery`` in, a real ``VectorStoreQueryResult`` out.

    The layer that would catch the interface moving: the query is built by the
    bridge from the real dataclass — whose ``similarity_top_k`` defaults to 1, so
    a bridge that forgot to pass ``k`` would return one hit and look plausible —
    and the result is read by the real field names. ``SimpleVectorStore`` fills
    ids and similarities and leaves ``nodes`` empty, and its similarity is plain
    cosine, so the scores here are real and descending.
    """
    pytest.importorskip("llama_index.core", reason="the llamaindex extra is not installed")
    from llama_index.core.schema import TextNode
    from llama_index.core.vector_stores import SimpleVectorStore

    ids, vectors, texts = corpus
    store = SimpleVectorStore()
    store.add(
        [
            TextNode(id_=i, text=t, embedding=v.tolist())
            for i, v, t in zip(ids[:20], vectors[:20], texts[:20], strict=True)
        ]
    )

    hits = LlamaIndexStoreAdapter(store, dimension=DIM).search(vectors[3], k=5)

    assert [h.rank for h in hits] == [0, 1, 2, 3, 4]
    assert hits[0].id == ids[3]
    assert hits[0].score == pytest.approx(1.0)
    assert all(a.score >= b.score for a, b in itertools.pairwise(hits))
