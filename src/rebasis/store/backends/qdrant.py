"""Qdrant backend.

Qdrant's local mode needs no server — ``QdrantClient(path=...)`` runs against an
embedded database — so it belongs alongside the local-first backends rather than
being treated as an operations task. The same class serves a real cluster; only
the URI differs.

Two Qdrant specifics shape this file.

**Reads use ``scroll``, not ``search``.** ``scroll`` is Qdrant's cursor: it
returns a page and an offset to continue from, which is exactly the streaming
contract and keeps peak memory at ``O(batch × d)``.

**Points carry a UUID or an integer id, and the user's own id usually lives in
the payload.** So the backend resolves them: if the collection's points have a
payload field that looks like a document id, that is what rebasis reports, and
it is mapped back to point ids when writing.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Self

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

__all__ = ["QdrantStore"]

DEFAULT_BATCH = 256

#: Payload keys tried in order when the URI does not name one.
_ID_KEYS = ("id", "doc_id", "document_id", "chunk_id", "_id")
_TEXT_KEYS = ("text", "content", "document", "page_content", "chunk")


class QdrantStore:
    """A Qdrant collection, local or remote."""

    def __init__(
        self,
        client: Any,
        collection: str,
        *,
        id_key: str | None = None,
        text_key: str | None = None,
        vector_name: str | None = None,
    ) -> None:
        self._client = client
        self._collection = collection
        self._id_key = id_key
        self._text_key = text_key
        self._vector_name = vector_name
        self._dimension: int | None = None

    @classmethod
    def from_uri(cls, uri: StoreURI, **kwargs: Any) -> QdrantStore:
        """Open the collection a URI points at.

        ``qdrant:///path/to/local/db#documents`` — embedded, no server
        ``qdrant://localhost:6333#documents`` — a running instance

        Options name the payload layout when it is not conventional::

            ?id_key=doc_id&text_key=body&vector_name=dense
        """
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise MissingDependency(
                "The Qdrant backend needs qdrant-client.",
                hint='Install it with `pip install "rebasis[qdrant]"`.',
                context={"store_backend": "qdrant"},
                cause=exc,
            ) from exc

        if not uri.collection:
            raise CollectionNotFound(
                "The Qdrant URI does not name a collection.",
                hint="Append it after a #, e.g. `qdrant://localhost:6333#documents`.",
                context={"store_backend": "qdrant"},
            )

        client = _client(QdrantClient, uri, **kwargs)
        names = _collection_names(client)
        if uri.collection not in names:
            raise CollectionNotFound(
                f"Qdrant has no collection named {uri.collection!r}.",
                hint=f"Available: {', '.join(names) if names else '(none)'}.",
                context={"store_backend": "qdrant"},
            )

        options = uri.options or {}
        store = cls(
            client,
            uri.collection,
            id_key=options.get("id_key"),
            text_key=options.get("text_key"),
            vector_name=options.get("vector_name"),
        )
        store._resolve_payload_keys()
        return store

    @property
    def capabilities(self) -> StoreCapabilities:
        """Qdrant supports everything rebasis needs, including a reindex."""
        return StoreCapabilities(
            can_read_vectors=True,
            can_read_text=self._text_key is not None,
            can_upsert_vectors=True,
            can_filter=True,
            dimension_locked=True,
            supports_in_place_update=True,
            # The only backend of the five that can. Qdrant documents the move
            # and documents that it costs no downtime.
            can_rebuild_index=True,
            name="qdrant",
        )

    def count(self) -> int:
        """Number of points in the collection."""
        try:
            return int(self._client.count(self._collection, exact=True).count)
        except Exception as exc:
            raise StoreError(
                "Could not read the point count from Qdrant.",
                hint="Check that the collection exists and the client can reach it.",
                context={"store_backend": "qdrant"},
                cause=exc,
            ) from exc

    def dimension(self) -> int:
        """Vector dimensionality, from the collection's own configuration.

        Read from the config rather than a sample point: Qdrant knows it
        exactly, and an empty collection still answers.
        """
        if self._dimension is None:
            try:
                params = self._client.get_collection(self._collection).config.params
            except Exception as exc:
                raise StoreError(
                    "Could not read the Qdrant collection configuration.",
                    hint="Check that the collection exists.",
                    context={"store_backend": "qdrant"},
                    cause=exc,
                ) from exc
            self._dimension = _dimension_from(params, self._vector_name)
        return self._dimension

    def iter_records(
        self,
        ids: Sequence[str] | None = None,
        *,
        with_vectors: bool = True,
        with_text: bool = True,
        batch_size: int = DEFAULT_BATCH,
    ) -> Iterator[Record]:
        """Stream points with Qdrant's scroll cursor — lazy by construction."""
        if ids is not None:
            yield from self._retrieve(list(ids), with_vectors=with_vectors, with_text=with_text)
            return

        offset: Any = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=with_vectors,
            )
            for point in points:
                yield self._to_record(point, with_text=with_text)
            if offset is None:
                return

    def search(self, vector: FloatArray, k: int, where: dict[str, Any] | None = None) -> list[Hit]:
        """Nearest neighbours."""
        query = as_float32(vector).reshape(-1)
        if query.shape[0] != self.dimension():
            raise EmbeddingDimensionMismatch(
                f"The collection is {self.dimension()}-dimensional but the query is "
                f"{query.shape[0]}-dimensional.",
                hint="The adapter's output dimension does not match the index.",
                context={"store_backend": "qdrant", "dim": self.dimension()},
            )
        try:
            points = self._client.query_points(
                collection_name=self._collection,
                query=query.tolist(),
                limit=k,
                query_filter=_filter(where),
                with_payload=True,
            ).points
        except Exception as exc:
            raise StoreError(
                "The Qdrant query failed.",
                hint="Check the filter, if you passed one, and the collection's vector name.",
                context={"store_backend": "qdrant"},
                cause=exc,
            ) from exc

        return [
            Hit(id=self._identity(point), score=float(point.score), rank=rank)
            for rank, point in enumerate(points)
        ]

    def upsert_vectors(self, ids: Sequence[str], vectors: FloatArray) -> None:
        """Replace vectors in place. The only write path rebasis has.

        ``update_vectors`` rather than ``upsert``: it replaces the vector and
        leaves the payload untouched. A full ``upsert`` would need the payload
        resent, and anything not resent would be silently dropped — "rebasis
        does not take ownership of user data", in one API call.
        """
        matrix = as_float32(vectors)
        if matrix.shape[1] != self.dimension():
            raise EmbeddingDimensionMismatch(
                f"The collection is {self.dimension()}-dimensional but the vectors "
                f"are {matrix.shape[1]}-dimensional.",
                hint="Refit the adapter against this collection's dimension.",
                context={"store_backend": "qdrant", "dim": self.dimension()},
            )

        from qdrant_client import models

        point_ids = self._point_ids([str(i) for i in ids])
        try:
            self._client.update_vectors(
                collection_name=self._collection,
                points=[
                    models.PointVectors(
                        id=point_ids[str(record_id)],
                        vector=(
                            vector.tolist()
                            if self._vector_name is None
                            else {self._vector_name: vector.tolist()}
                        ),
                    )
                    for record_id, vector in zip(ids, matrix, strict=True)
                ],
            )
        except Exception as exc:
            raise StoreWriteFailed(
                f"Qdrant rejected an update of {len(ids)} points.",
                hint="Transient failures are retried automatically; this one was not.",
                context={"store_backend": "qdrant", "count": len(ids)},
                cause=exc,
            ) from exc

    def rebuild_index(self) -> None:
        """Make Qdrant rebuild the HNSW graph from the vectors now in it.

        The documented move, and Qdrant's own words for it: *"Changing `m` or
        `ef_construct` automatically triggers a full background HNSW rebuild"*,
        and *"queries continue to be served by the old index until the new index
        is complete, so there is no downtime"*
        (`qdrant.tech/documentation/faq/qdrant-fundamentals/`). That property is
        what makes this worth exposing at all — a repair that took the
        collection offline would not be one a user could take on trust.

        `ef_construct` is bumped by one and **left there**, which is the shape
        Qdrant's own guidance describes; reverting it immediately is advised
        against, and would in any case queue a second full rebuild to undo a
        change of no consequence. One point of `ef_construct` is inside the
        noise of the parameter's effect, and the alternative is a repair that
        does not happen.

        Returns as soon as the rebuild is *scheduled*. Qdrant does the work in
        the background and keeps answering queries throughout, so blocking here
        would hold a lock over an operation the database is explicitly designed
        to run without one. `rebasis migrate`'s own health check re-measures
        after the fact.

        Raises:
            StoreError: When Qdrant rejects the configuration change.
        """
        from qdrant_client import models

        try:
            current = self._client.get_collection(self._collection).config.hnsw_config
            ef_construct = int(getattr(current, "ef_construct", None) or 100)
            self._client.update_collection(
                collection_name=self._collection,
                hnsw_config=models.HnswConfigDiff(ef_construct=ef_construct + 1),
            )
        except Exception as exc:
            raise StoreError(
                "Qdrant refused to reconfigure the collection's HNSW index.",
                hint=(
                    "Rebuilding is triggered by changing `ef_construct`, which "
                    "needs permission to update the collection."
                ),
                context={"store_backend": "qdrant"},
                cause=exc,
            ) from exc

    def close(self) -> None:
        """Release the client.

        Qdrant's local mode takes an exclusive lock on its storage folder: a
        second client against the same path raises rather than waiting. Anything
        that opens a store transiently — `rollback` reopening what `migrate`
        wrote, a test reading back through a fresh connection — has to give the
        handle back first.

        Safe to call more than once, and safe on a network client where there is
        nothing to release.
        """
        client, self._client = self._client, None
        if client is not None:
            with contextlib.suppress(Exception):
                client.close()

    def __enter__(self) -> Self:
        """Support ``with open_store(...) as store:``."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Release the client on the way out."""
        self.close()

    # ── internals ─────────────────────────────────────────────────────

    def _resolve_payload_keys(self) -> None:
        """Learn the payload layout from one point.

        Guessing ``"text"`` and failing on a collection that used ``content``
        is an unnecessary way to lose a user; so is demanding the URI say so
        when it can simply be looked at.
        """
        if self._id_key is not None and self._text_key is not None:
            return
        try:
            points, _ = self._client.scroll(
                collection_name=self._collection, limit=1, with_payload=True, with_vectors=False
            )
        except Exception:  # noqa: BLE001 - a best-effort probe, never fatal
            return
        if not points:
            return
        payload = getattr(points[0], "payload", None) or {}
        if self._id_key is None:
            self._id_key = next((k for k in _ID_KEYS if k in payload), None)
        if self._text_key is None:
            self._text_key = next((k for k in _TEXT_KEYS if k in payload), None)

    def _retrieve(self, ids: list[str], *, with_vectors: bool, with_text: bool) -> Iterator[Record]:
        """Fetch specific records, by payload id or by point id."""
        point_ids = self._point_ids(ids)
        missing = [i for i in ids if i not in point_ids]
        if missing:
            raise CollectionNotFound(
                f"{len(missing)} requested ids are not in this collection.",
                hint=f"First missing id: {missing[0]!r}.",
                context={"store_backend": "qdrant", "count": len(missing)},
            )
        points = self._client.retrieve(
            collection_name=self._collection,
            ids=[point_ids[i] for i in ids],
            with_payload=True,
            with_vectors=with_vectors,
        )
        for point in points:
            yield self._to_record(point, with_text=with_text)

    def _point_ids(self, ids: list[str]) -> dict[str, Any]:
        """Map the ids rebasis reports back to Qdrant's own point ids."""
        if self._id_key is None:
            return {i: _native(i) for i in ids}

        from qdrant_client import models

        found: dict[str, Any] = {}
        for start in range(0, len(ids), DEFAULT_BATCH):
            chunk = ids[start : start + DEFAULT_BATCH]
            points, _ = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key=self._id_key, match=models.MatchAny(any=chunk))]
                ),
                limit=len(chunk),
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                found[str((point.payload or {}).get(self._id_key))] = point.id
        return found

    def _to_record(self, point: Any, *, with_text: bool) -> Record:
        payload = getattr(point, "payload", None) or {}
        return Record(
            id=self._identity(point),
            vector=_vector_of(point, self._vector_name),
            text=(str(payload.get(self._text_key)) if with_text and self._text_key else None),
            metadata=payload,
        )

    def _identity(self, point: Any) -> str:
        payload = getattr(point, "payload", None) or {}
        if self._id_key and payload.get(self._id_key) is not None:
            return str(payload[self._id_key])
        return str(point.id)


def _client(client_class: Any, uri: StoreURI, **kwargs: Any) -> Any:
    """Local mode when the URI carries a path, otherwise a network client."""
    if uri.host:
        url = f"http://{uri.host}"
        if uri.port:
            url = f"{url}:{uri.port}"
        return client_class(url=url, api_key=uri.password, **kwargs)
    return client_class(path=uri.path or ":memory:", **kwargs)


def _collection_names(client: Any) -> list[str]:
    try:
        return [c.name for c in client.get_collections().collections]
    except Exception:  # noqa: BLE001 - only used to enrich an error message
        return []


def _dimension_from(params: Any, vector_name: str | None) -> int:
    """Read the size out of Qdrant's vector config, named or unnamed."""
    vectors = getattr(params, "vectors", None)
    if vectors is None:
        raise StoreError(
            "The Qdrant collection declares no vector configuration.",
            hint="rebasis needs an existing index to measure against.",
            context={"store_backend": "qdrant"},
        )
    if isinstance(vectors, dict):
        if vector_name is not None and vector_name in vectors:
            return int(vectors[vector_name].size)
        if len(vectors) == 1:
            return int(next(iter(vectors.values())).size)
        raise StoreError(
            f"This collection has several named vectors: {', '.join(sorted(vectors))}.",
            hint="Name the one to use with ?vector_name=<name> in the URI.",
            context={"store_backend": "qdrant"},
        )
    return int(vectors.size)


def _vector_of(point: Any, vector_name: str | None) -> FloatArray | None:
    """Pull the vector out of a point, named or unnamed."""
    vector = getattr(point, "vector", None)
    if vector is None:
        return None
    if isinstance(vector, dict):
        chosen = vector.get(vector_name) if vector_name else next(iter(vector.values()), None)
        return as_float32(chosen) if chosen is not None else None
    return as_float32(vector)


def _filter(where: dict[str, Any] | None) -> Any:
    """Translate a plain mapping into a Qdrant filter."""
    if not where:
        return None

    from qdrant_client import models

    return models.Filter(
        must=[
            models.FieldCondition(key=k, match=models.MatchValue(value=v)) for k, v in where.items()
        ]
    )


def _native(value: str) -> Any:
    """Qdrant point ids are integers or UUIDs; keep an integer an integer."""
    try:
        return int(value)
    except ValueError:
        return value
