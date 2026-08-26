"""Two-stage retrieval — the bridge used as a recall stage.

`Bridge` maps a new model's query into the index's space and lets the index
produce the **final ranking**. `docs/bridge-band.md` measures what that is worth
— about one run in five — and
[ADR 10](../adr/0010-retention-is-bounded-by-the-source.md) says it cannot be
fitted higher, because a single global map into the old space cannot carry more
than the old space holds.

Both still hold. What this module changes is the assumption underneath them, not
the quality of the map::

    q_new  = f_new(query)                     the caller supplies this
    q_old  = bridge.to_index_space(q_new)
    cand   = store.search(q_old, k=N)         the bridge, as a recall stage
    v_new  = embedder.encode(text of cand)    re-embed — cached
    result = top-k by cos(q_new, v_new)       ranked in the NEW space

The last line is the new model scoring its own vectors, which is exactly the
ranking a full reindex would produce over those same documents. So the only
thing the bridge can lose is a relevant document that never reached the
candidate set, and what bounds the arrangement is its **recall@N** rather than
its nDCG@10. Putting a relevant document somewhere in the top 200 is a weaker
requirement than ranking it in the top 10, and measurably so: retention 0.893 at
recall@200 against 0.717 at nDCG@10.

`docs/cascade-band.md` measured that over 48 runs — sixteen corpora, three model
pairs. Single-stage bridging beat keeping the current model in **1**; this
arrangement beat it in **36**. Restricted to the runs where a full reindex is
genuinely an upgrade, the score is 36 of 37 against 1 of 37, and on the upper
rung it lands within two percent of rebuilding the index in every one of sixteen
runs. None of that contradicts ADR 10: retention at nDCG@10 is exactly where it
always was. A different quantity is doing the bounding.

**Step four is the entire cost, and it is on the hot path.**
[ADR 11](../adr/0011-the-hot-path-budget-is-per-dimension.md)'s budget — tens of
microseconds for the mapping — does not describe this at all. At an A10G's
measured rate for bge-base, 100 documents is about 0.2 s; on the CPU a laptop
has, it is seconds. Three consequences, all deliberate:

* **The cache is part of the design, not an option.** There is no constructor
  that omits it. It is also what makes this a lazy migration rather than a
  permanent tax: the documents people actually retrieve get embedded once and
  stay embedded, and every query after the first pays for its own misses only.
* **The cost is reported.** :class:`CascadeStats` splits a query into its three
  stages and names the hit rate and the number of documents embedded. A feature
  whose main risk is what it costs has to say what it costs, on the user's own
  hardware and their own traffic — the same reason ``probe`` extrapolates a
  reindex from a rate it measured rather than from a table of averages.
* **torch is still absent**, for the reason `serve/bridge.py` gives. The
  embedder may load it in its own process; nothing here references it.

`docs/cascade-band.md` closes by saying the cache did not exist, that its
behaviour under a real query distribution had not been measured, and that until
both changed this was a measurement rebasis did not serve. This module is the
cache and the instrument. It is not a new claim: ``probe`` still reports what a
two-stage arrangement would retain **without** recommending it, because the
missing measurement is a property of a query log rather than of a corpus, and
the only place it can be taken is a running system. That is what
:attr:`Cascade.stats` is for.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from rebasis.core.base import l2_normalize
from rebasis.store.base import require_capability
from rebasis.types import Hit, as_float32

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from rebasis.serve.bridge import Bridge
    from rebasis.storage.embedding_cache import EmbeddingCache
    from rebasis.store.base import VectorStore
    from rebasis.types import Embedder, FloatArray

    def _shared_cache_is_a_vector_cache(cache: EmbeddingCache) -> VectorCache:
        """Keep the offer made in :class:`VectorCache`'s docstring checked.

        :class:`rebasis.storage.EmbeddingCache` is documented below as a
        drop-in :class:`VectorCache`. Nothing here imports it at runtime and
        nothing calls this; it exists so that a change on either side is a type
        error rather than a surprise for whoever took the documentation at its
        word.
        """
        return cache


__all__ = [
    "CANDIDATES",
    "Cascade",
    "CascadeStats",
    "DiskVectorCache",
    "MemoryVectorCache",
    "VectorCache",
    "default_cache_dir",
]

#: How many documents the bridge is asked to recall for one query.
#:
#: `docs/cascade-band.md` measured N=100 and N=200 and found them a point or two
#: apart at nDCG@10 — the first evidence that the curve flattens early, though
#: *where* it flattens has not been measured. 100 is therefore the smaller claim
#: and half the documents to re-embed on a cold cache, which is the number that
#: decides whether the first query after a deploy takes 0.2 s or 0.4 s. It is
#: also the depth ``probe`` reports the arrangement at, so a user who read that
#: number gets the arrangement it described.
CANDIDATES = 100

#: Vectors the default in-memory cache holds before it evicts.
#:
#: At d=768 a float32 vector is 3 KB, so this is a ceiling of roughly 150 MB —
#: the size of a working set, not of a corpus, which is the whole premise: a
#: query log concentrates on the documents people actually ask about. A process
#: that knows its working set is larger should say so; one that wants the cache
#: to survive a restart wants :class:`DiskVectorCache`.
MEMORY_CACHE_ENTRIES = 50_000

#: Ceiling on one call into the embedder.
#:
#: The candidate set is the batch — at the default depth that is 100 documents,
#: one forward pass on any accelerator and a manageable one on a CPU. Splitting
#: it would pay the per-call overhead several times over for the one latency
#: this arrangement can least afford, and unlike a migration there is no next
#: batch to amortise it against. The ceiling exists so that an unusually deep
#: candidate set does not become an unusually large forward pass.
MAX_EMBED_BATCH = 512

#: Subdirectory of the state directory's cache that holds re-embedded vectors.
CASCADE_DIR = "cascade"

#: Hex characters of the key digest used as a shard directory — 256 of them.
#:
#: A single directory holding a hundred thousand small files is slow to list on
#: every filesystem and unpleasant on some. Two characters keeps lookups flat
#: and `ls` usable without inventing a second level of bookkeeping.
_SHARD = 2

_FLOAT32_BYTES = 4


def default_cache_dir() -> Path:
    """Resolve where a persistent cascade cache lives.

    ``.rebasis/cache/cascade/`` — under the directory ``rebasis gc`` already
    collects on a 30-day retention and that ``REBASIS_CACHE_DIR`` already names
    ("sample and embedding cache"). Inventing a location would mean a second
    thing for a user to find, a second thing to clean up, and a cache that
    outlives the state directory it belongs to.
    """
    from rebasis.config import settings
    from rebasis.manifest import CACHE_DIR, default_state_dir

    configured = settings().cache_dir
    root = Path(configured) if configured else default_state_dir() / CACHE_DIR
    return root / CASCADE_DIR


class VectorCache(Protocol):
    """Where re-embedded document vectors are kept between queries.

    Both methods take a whole batch. A query misses on a candidate set, not on
    one document, and a cache asked one id at a time can amortise nothing —
    neither a hundred file reads nor a round trip to whatever a user puts behind
    this protocol.

    Keys are opaque: :class:`Cascade` builds them, and an implementation must
    treat them as bytes rather than parse them.

    Three implementations ship. :class:`MemoryVectorCache` is the default and
    :class:`DiskVectorCache` survives a restart;
    :class:`rebasis.storage.EmbeddingCache` — the store ``probe`` uses to
    remember what it has already embedded — satisfies this protocol as it
    stands, and is the one to reach for when the working set is large enough
    that a directory of small files starts to cost something. It keeps a single
    SQLite file and writes a candidate set in one transaction, where
    :class:`DiskVectorCache` writes one fsynced file per document.
    """

    def get(self, keys: Sequence[str]) -> dict[str, FloatArray]:
        """Return the vectors held for ``keys``. Absent keys are simply absent."""
        ...

    def put(self, vectors: Mapping[str, FloatArray]) -> None:
        """Store vectors under their keys."""
        ...


class MemoryVectorCache:
    """A bounded LRU held in this process — the default.

    The default because it is the one that is always correct: it needs no
    directory, no permissions and no cleanup, and its cost is bounded by
    ``capacity`` rather than by how long the process has been running. What it
    cannot do is survive a restart, so the first query after every deploy pays
    the full re-embedding cost again. That is what :class:`DiskVectorCache` is
    for, and it is a deliberate default rather than an oversight: writing into a
    user's project directory is not something a library should start doing
    unasked.
    """

    __slots__ = ("_capacity", "_entries", "_lock")

    def __init__(self, capacity: int = MEMORY_CACHE_ENTRIES) -> None:
        self._entries: OrderedDict[str, FloatArray] = OrderedDict()
        self._capacity = max(1, capacity)
        # A serving process calls this from whichever thread the request landed
        # on. `move_to_end` and `popitem` are each safe under the GIL; the
        # read-then-evict sequence around them is not, and the failure it
        # produces is a cache that quietly holds more than its ceiling.
        self._lock = threading.Lock()

    def get(self, keys: Sequence[str]) -> dict[str, FloatArray]:
        """Return the vectors held for ``keys``, refreshing their recency."""
        found: dict[str, FloatArray] = {}
        with self._lock:
            for key in keys:
                vector = self._entries.get(key)
                if vector is not None:
                    self._entries.move_to_end(key)
                    found[key] = vector
        return found

    def put(self, vectors: Mapping[str, FloatArray]) -> None:
        """Store vectors, evicting the least recently used to stay in bounds."""
        with self._lock:
            for key, vector in vectors.items():
                self._entries[key] = vector
                self._entries.move_to_end(key)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)


class DiskVectorCache:
    """A cache under ``.rebasis/cache/``, which outlives the process.

    One file per vector, named by the digest of its key and written through
    :func:`rebasis.storage.atomic.atomic_write_bytes` — so an entry is either
    complete or absent, and a crash mid-write leaves neither a truncated vector
    nor a damaged neighbour. The file holds the raw float32 bytes and nothing
    else: the name already carries the identity, because the key it hashes
    contains the model's profile fingerprint.

    **It has no eviction policy of its own, on purpose.** ``rebasis gc`` already
    has one for this directory — a cache file untouched for 30 days is a
    candidate, and the whole category is listed and freed without confirmation.
    Reading a file updates its atime and not its mtime, so `gc` can collect an
    entry that queries are still hitting; the cost of that is re-embedding one
    document once a month, against one write syscall per cache *hit* to prevent
    it. Turning every read into a write to defend a cache is the wrong trade.

    **Why this exists alongside** :class:`rebasis.storage.EmbeddingCache`, which
    solves a near-identical problem: the two are the same idea at opposite
    scales, and merging them would have made one of them worse. Here a query
    touches a hundred documents and entries accumulate lazily over a query log,
    so a small file per vector needs no schema and no connection and lets `gc`
    expire one document at a time. ``probe`` embeds ten thousand documents in a
    single pass, where a file per vector is ten thousand files for `gc` to
    ``stat`` and ten thousand fsyncs to pay. They also key different things —
    record ids here, texts there — and expire on different clocks. Either can be
    handed to :class:`Cascade`; a process whose working set has grown past what
    a directory of small files handles comfortably should hand it the other one.

    Args:
        directory: Where to keep the files. Defaults to
            :func:`default_cache_dir`, which honours ``REBASIS_CACHE_DIR`` and
            ``REBASIS_STATE_DIR``.
    """

    __slots__ = ("_root", "write_failures")

    def __init__(self, directory: Path | str | None = None) -> None:
        self._root = Path(directory) if directory is not None else default_cache_dir()
        # Writes this cache could not complete — a full disk, a read-only mount,
        # a directory somebody removed underneath it. Counted rather than
        # raised, and exposed rather than hidden: see `put`.
        self.write_failures = 0

    @property
    def directory(self) -> Path:
        """Where this cache keeps its files."""
        return self._root

    def get(self, keys: Sequence[str]) -> dict[str, FloatArray]:
        """Read the vectors held for ``keys``. An unreadable file is a miss."""
        found: dict[str, FloatArray] = {}
        for key in keys:
            vector = _read_vector(self._path_for(key))
            if vector is not None:
                found[key] = vector
        return found

    def put(self, vectors: Mapping[str, FloatArray]) -> None:
        """Write each vector to its own file, atomically.

        A failed write increments :attr:`write_failures` and is not raised. A
        cache exists to make queries cheaper, and one that can take a query down
        is worse than no cache at all — the search has already succeeded by the
        time this is called, and the only thing lost is that the next query pays
        for the same documents again.

        The directory is not fsynced. A cache entry a power cut loses is a cache
        miss, and paying a directory fsync per write to prevent one would cost
        more than re-embedding the document it protects.
        """
        from rebasis.errors import RebasisError
        from rebasis.storage.atomic import atomic_write_bytes

        for key, vector in vectors.items():
            try:
                atomic_write_bytes(
                    self._path_for(key), as_float32(vector).tobytes(), fsync_dir=False
                )
            except (OSError, RebasisError):
                self.write_failures += 1

    def _path_for(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self._root / digest[:_SHARD] / digest


def _read_vector(path: Path) -> FloatArray | None:
    """Read one cached vector, treating anything unreadable as a miss."""
    try:
        blob = path.read_bytes()
    except OSError:
        return None
    if not blob or len(blob) % _FLOAT32_BYTES:
        return None
    # `frombuffer` views the bytes object, which is immutable, so the array it
    # hands back is read-only. One copy here keeps that out of the caller's way.
    return np.frombuffer(blob, dtype=np.float32).copy()


@dataclass(slots=True)
class CascadeStats:
    """What this arrangement has cost so far.

    Cumulative over the life of a :class:`Cascade`, because the number that
    matters — the hit rate — is a property of a stream of queries rather than of
    one. :meth:`reset` starts a fresh window.

    The counters are updated without a lock. Under concurrency an increment can
    be lost, which is acceptable for what this is: an instrument, read to decide
    whether the arrangement is affordable, not a ledger anything depends on.
    """

    queries: int = 0
    #: Candidates the bridge returned, summed over every query.
    candidates: int = 0
    cache_hits: int = 0
    #: Documents actually sent to the embedder — the cost that ADR 11's budget
    #: does not cover, and the one a warm cache drives to zero.
    documents_embedded: int = 0
    #: Candidates left standing in the bridge's order because they could not be
    #: re-embedded. Almost always: the store holds no text for them. Also covers
    #: a record the store stopped returning between the search and the read.
    kept_bridged: int = 0
    #: Mapping the query into the index's space — step 2.
    bridge_seconds: float = 0.0
    #: The store's own query — step 3.
    search_seconds: float = 0.0
    #: Reading text, the cache, the embedder and the reordering — steps 4 and 5.
    rerank_seconds: float = 0.0
    #: The embedder alone, out of :attr:`rerank_seconds`. Reported separately
    #: because it is the term that makes the difference between a cold query and
    #: a warm one, and the only one that scales with the candidate depth.
    embed_seconds: float = 0.0

    @property
    def cache_misses(self) -> int:
        """Candidates the cache did not hold."""
        return self.candidates - self.cache_hits

    @property
    def hit_rate(self) -> float:
        """Fraction of candidates the cache answered.

        ``nan`` before the first query: a cache that has been asked nothing has
        no hit rate, and reporting 0.0 would read as a cache that is not working.
        """
        return self.cache_hits / self.candidates if self.candidates else float("nan")

    @property
    def seconds(self) -> float:
        """Total time in the three stages. ``embed_seconds`` is inside the third."""
        return self.bridge_seconds + self.search_seconds + self.rerank_seconds

    @property
    def per_query_seconds(self) -> float:
        """Mean latency of one query — the number a serving budget is set from."""
        return self.seconds / self.queries if self.queries else float("nan")

    def reset(self) -> None:
        """Start a fresh measurement window."""
        self.queries = 0
        self.candidates = 0
        self.cache_hits = 0
        self.documents_embedded = 0
        self.kept_bridged = 0
        self.bridge_seconds = 0.0
        self.search_seconds = 0.0
        self.rerank_seconds = 0.0
        self.embed_seconds = 0.0

    def to_dict(self) -> dict[str, float]:
        """Serialisable form, for a report or a metrics exporter."""
        return {
            "queries": self.queries,
            "candidates": self.candidates,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.hit_rate, 4),
            "documents_embedded": self.documents_embedded,
            "kept_bridged": self.kept_bridged,
            "bridge_ms": round(self.bridge_seconds * 1000, 3),
            "search_ms": round(self.search_seconds * 1000, 2),
            "rerank_ms": round(self.rerank_seconds * 1000, 2),
            "embed_ms": round(self.embed_seconds * 1000, 2),
            "per_query_ms": round(self.per_query_seconds * 1000, 2),
        }


class Cascade:
    """The bridge recalls; the new model ranks.

    Args:
        store: The index, as it already is. Read only — nothing here writes to
            it, which is what keeps the arrangement free of risk: stopping using
            it *is* the rollback.
        bridge: The adapter, which turns the new model's query into a query the
            index can answer.
        embedder: The model being adopted. It re-embeds candidate documents, so
            it must be the same model whose vectors the caller passes to
            :meth:`search` — a mismatch would score a query against documents
            from a different space and raise nothing.
        candidates: Depth of the candidate set. See :data:`CANDIDATES`.
        cache: Where re-embedded vectors are kept. Defaults to
            :class:`MemoryVectorCache`; :class:`DiskVectorCache` survives a
            restart.

    Raises:
        CapabilityMissing: When the store cannot return document text. A store
            that cannot is one where the cache can never be filled, so every
            query would silently return the bridged order forever. Refusing at
            construction beats failing on the first query in production.

    Example:
        ```python
        cascade = Cascade(store, bridge, new_model, cache=DiskVectorCache())
        hits = cascade.search(new_model.encode(["how do I deploy?"], kind="query")[0])
        print(cascade.stats.to_dict())
        ```
    """

    __slots__ = (
        "_bridge",
        "_cache",
        "_candidates",
        "_embedder",
        "_fingerprint",
        "_stats",
        "_store",
    )

    def __init__(
        self,
        store: VectorStore,
        bridge: Bridge,
        embedder: Embedder,
        *,
        candidates: int = CANDIDATES,
        cache: VectorCache | None = None,
    ) -> None:
        require_capability(store, "can_read_text", operation="two-stage serving")
        self._store = store
        self._bridge = bridge
        self._embedder = embedder
        self._candidates = max(1, candidates)
        self._cache = MemoryVectorCache() if cache is None else cache
        # Resolved once. Every cache key carries it, so a vector produced by one
        # model is unreachable under another — which matters more here than
        # anywhere else in rebasis: a stale vector from the previous model would
        # not raise, it would rank, and the result would look like the upgrade
        # working badly rather than like the wrong vectors being used.
        self._fingerprint = embedder.profile.fingerprint()
        self._stats = CascadeStats()

    @property
    def stats(self) -> CascadeStats:
        """The live measurement. See :class:`CascadeStats`."""
        return self._stats

    @property
    def candidates(self) -> int:
        """Depth of the candidate set — what bounds this arrangement's recall."""
        return self._candidates

    def search(self, vector: FloatArray, k: int = 10, **kwargs: Any) -> list[Hit]:
        """Retrieve ``k`` documents, ranked in the new model's space.

        Args:
            vector: The query under the **new** model, ``(d_new,)`` or
                ``(1, d_new)``. The old-space query is derived from it here;
                asking the caller for both would be asking them to hold the
                detail this exists to hide.
            k: How many results to return.
            **kwargs: Passed through to the store's ``search`` — a metadata
                filter belongs on the candidate search, where it can still
                narrow anything.

        Returns:
            Up to ``k`` hits, best first. The score of a hit the new model
            scored is a cosine in the new space; the score of one it could not
            (see :attr:`CascadeStats.kept_bridged`) is the store's own score in
            the old space. The two are not on one scale — M0 measured a median
            KS distance of 0.924 between them — which is why the ordering here
            is positional rather than by score, and why a pipeline that filters
            on a fixed threshold should read
            :attr:`CascadeStats.kept_bridged` before trusting one.
        """
        query = l2_normalize(as_float32(vector).reshape(-1))

        started = time.perf_counter()
        bridged_query = self._bridge.to_index_space(query)
        bridged_at = time.perf_counter()
        # At least `k`: a caller asking for more results than the configured
        # candidate depth wants results, not a lecture about the depth.
        found = self._store.search(bridged_query, k=max(k, self._candidates), **kwargs)
        searched_at = time.perf_counter()
        ranked = self._rerank(query, found, k)
        finished = time.perf_counter()

        stats = self._stats
        stats.queries += 1
        stats.candidates += len(found)
        stats.bridge_seconds += bridged_at - started
        stats.search_seconds += searched_at - bridged_at
        stats.rerank_seconds += finished - searched_at
        return ranked

    def describe(self) -> dict[str, Any]:
        """A compact summary for logs and reports. Never called per query."""
        return {
            # `candidate_depth`, not `candidates`: the stats carry a running
            # total under that name, and one of the two would have silently
            # replaced the other.
            "candidate_depth": self._candidates,
            "cache": type(self._cache).__name__,
            "new_model": self._embedder.profile.model_id,
            "adapter_type": self._bridge.adapter_type,
            **self._stats.to_dict(),
        }

    def __repr__(self) -> str:
        return (
            f"Cascade({self._embedder.profile.model_id} reranks "
            f"{self._candidates} candidates, cache={type(self._cache).__name__})"
        )

    # ── internals ─────────────────────────────────────────────────────

    def _key(self, record_id: str) -> str:
        """The cache key for one record under the current model.

        Fingerprint first, and fixed length: a record id that happens to contain
        a colon cannot be made to collide with another model's key.
        """
        return f"{self._fingerprint}:{record_id}"

    def _check_width(self, encoded: FloatArray) -> None:
        """Refuse a batch whose width is not the one the profile declares.

        The cache is keyed on the encoding profile's fingerprint, and that
        fingerprint covers every field of the profile including ``dim``. So a
        cache namespace is supposed to hold one width — and everything
        downstream relies on it: ``_rerank`` stacks whatever the cache returns
        into one matrix and multiplies it by the query.

        What breaks the assumption is an embedder that does not honour its own
        profile — a hand-set ``--new-dim`` that does not match the model, or a
        model id that started resolving to different weights. Left alone, that
        surfaces as ``all the input array dimensions ... must match exactly``
        from inside numpy, several frames from anything the user chose. Checked
        here, at the one place vectors enter the cache, it names both numbers.

        Raises:
            EmbeddingDimensionMismatch: When the encoder's output is not the
                declared width.
        """
        declared = self._embedder.profile.dim
        actual = int(encoded.shape[1]) if encoded.ndim > 1 else 0
        if not declared or actual == declared:
            return
        from rebasis.errors import EmbeddingDimensionMismatch

        raise EmbeddingDimensionMismatch(
            f"{self._embedder.profile.model_id} returned {actual}-dimensional vectors "
            f"but its profile declares {declared}.",
            hint=(
                "The cache is keyed on the profile, so a mismatch would put two "
                "widths under one key. Check any --new-dim override against the "
                "model, and clear the cache if the model itself changed."
            ),
            context={"dim": actual, "model_id": self._embedder.profile.model_id},
        )

    def _rerank(self, query: FloatArray, found: Sequence[Hit], k: int) -> list[Hit]:
        """Score the candidates in the new space and place the rest around them."""
        if not found:
            return []

        keys = [self._key(hit.id) for hit in found]
        known = self._cache.get(keys)
        self._stats.cache_hits += len(known)
        misses = [hit.id for hit, key in zip(found, keys, strict=True) if key not in known]
        if misses:
            known.update(self._embed(misses))

        scorable = [position for position, key in enumerate(keys) if key in known]
        stranded = [position for position, key in enumerate(keys) if key not in known]
        self._stats.kept_bridged += len(stranded)

        ranked: deque[tuple[str, float]] = deque()
        if scorable:
            matrix = np.vstack([known[keys[position]] for position in scorable])
            # Both sides are unit vectors, so this inner product is the cosine
            # the arrangement is defined by — regardless of whether the model's
            # own profile says it normalises.
            similarity = matrix @ query
            # Stable, so candidates the new model cannot separate stay in the
            # order the bridge put them rather than in argsort's.
            order = np.argsort(-similarity, kind="stable")
            ranked.extend((found[scorable[int(j)]].id, float(similarity[int(j)])) for j in order)

        # A candidate that could not be re-embedded keeps the rank the bridge
        # gave it, and the reranked documents flow around it. Dropping it
        # instead would remove a document from someone's results for a reason
        # that has nothing to do with relevance — `probe` may drop a sampled
        # record with no text (`DROPPED_NO_TEXT`) because a sample is allowed to
        # be smaller than it asked for, and a result set is not.
        held = {position: (found[position].id, found[position].score) for position in stranded}
        placed: list[tuple[str, float]] = []
        for position in range(min(k, len(found))):
            entry = held.get(position)
            if entry is None and ranked:
                entry = ranked.popleft()
            if entry is not None:
                placed.append(entry)

        return [Hit(id=doc, score=score, rank=rank) for rank, (doc, score) in enumerate(placed)]

    def _embed(self, ids: Sequence[str]) -> dict[str, FloatArray]:
        """Re-embed the candidates the cache did not hold, and cache them.

        Only the misses reach the store: on a warm cache this arrangement never
        reads document text at all, which is most of what makes it affordable.
        The text is embedded exactly as the store holds it, because the claim
        being made is that these are the vectors a full reindex would have
        written.
        """
        texts = {
            record.id: record.text
            for record in self._store.iter_records(ids, with_vectors=False, with_text=True)
            if record.text and record.text.strip()
        }
        # The store's order is not promised to be the order it was asked in.
        ordered = [record_id for record_id in ids if record_id in texts]
        if not ordered:
            return {}

        started = time.perf_counter()
        encoded = self._embedder.encode(
            [texts[record_id] for record_id in ordered],
            kind="document",
            batch_size=min(len(ordered), MAX_EMBED_BATCH),
            # A progress bar in a serving path would be output nobody asked for,
            # on a stream that may not be a terminal.
            progress=False,
        )
        self._stats.embed_seconds += time.perf_counter() - started
        self._stats.documents_embedded += len(ordered)
        self._check_width(encoded)

        vectors = l2_normalize(encoded)
        fresh = {
            self._key(record_id): vector for record_id, vector in zip(ordered, vectors, strict=True)
        }
        self._cache.put(fresh)
        return fresh
