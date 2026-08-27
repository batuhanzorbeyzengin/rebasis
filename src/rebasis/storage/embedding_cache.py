"""Embeddings, kept between runs.

``.rebasis/cache/`` has been declared by ``storage/layout.py``, named by
``REBASIS_CACHE_DIR`` and collected by ``rebasis gc`` on a 30-day retention for
as long as the state directory has existed. Nothing ever wrote to it, so every
``rebasis probe`` embedded its whole sample from scratch: the same ten thousand
documents, with the same model, at the same cost, every time the sample size
moved, a query log arrived, or a second candidate model was tried. That is the
most visible everyday cost in the tool, and this module is what writes to the
directory that was already there.

**The shape: one SQLite file per model profile.** Three shapes were weighed.

*One file per vector*, which is what
:class:`rebasis.serve.cascade.DiskVectorCache` does. Right for serving — a query
touches a hundred documents and entries accumulate lazily — and wrong here. One
``probe`` run at the default sample size would leave ten thousand files per
model, each three kilobytes of payload in a four-kilobyte block, and
:func:`~rebasis.storage.gc.plan_gc` walks that directory with ``rglob`` and
calls ``stat`` on every file it finds. It also pays an ``fsync`` per vector,
which buys an atomicity guarantee a cache does not need: a half-written cache
entry is a cache miss.

*One array per batch*, keyed by a digest of the ordered text list. One read and
one write, and no partial hits at all — which is the objection, because
``--sample`` is a flag people move. Going from 10,000 documents to 20,000 would
invalidate the 10,000 vectors that are still exactly right, and the second run
with slightly different arguments is precisely the case this exists to serve.

*One SQLite file per model profile*, which is what this is. A partial hit is a
seek per key inside one open file rather than a file open per key — the
comparison that matters is against ten thousand ``open`` calls, not against one
round trip — a write is one transaction rather than n renames, concurrency is
WAL's problem rather than a lock file's, and ``sqlite3`` is in the standard
library, so none of it costs a dependency. The file is per profile rather than
one file for everything because **``gc``'s unit of retention is a file**: a
candidate model somebody evaluated once and abandoned should be able to age out
on its own, and in a single shared file the model they still use would keep
refreshing the mtime and hold every abandoned one alive with it.

**A stale vector does not raise — it ranks.** ``serve/cascade.py`` states the
principle and it is sharper here, because ``probe``'s output is a recommendation
rather than a ranking: a vector left by a different model, a different prefix or
a different pooling would not fail, it would be *measured*, and what came out
would be a plausible answer to a question nobody asked. So the key carries the
profile fingerprint — which pins the model id, the dimension, both prefixes, the
normalisation flag, the matryoshka dimension and the pooling, and moves whenever
a field is added to the profile at all
(:meth:`rebasis.types.EncodingProfile.fingerprint`) — plus the ``kind``, whether
the stored vectors have been ℓ2-normalised, the dtype, a version for the key
scheme itself, and a digest of the exact text. Nothing outside the key is
trusted: the file name is a convenience for a human deciding what to delete, and
two profiles that somehow shared a file still could not read each other's rows.

**Failure is a miss, never an exception.** A full disk, a read-only mount, a
directory somebody removed, a file another process is mid-write in, a row whose
blob has been truncated — every one of them means "embed it again". A cache that
can take a probe down after nine thousand documents have been embedded is worse
than no cache, and one that can change an answer is worse than that. Failures
are counted on :class:`CacheStats` rather than raised, which is the discipline
:meth:`rebasis.serve.cascade.DiskVectorCache.put` established.

**Durability is deliberately weaker than the manifest's.** ``manifest/db.py``
takes WAL + ``synchronous=FULL`` and explains why: an audit record may not be
lost. A cache entry may. This takes WAL + ``synchronous=NORMAL``, which in WAL
mode cannot corrupt the file on a power cut and can only lose the last commits —
and a lost cache entry is a cache miss. The pragmas are restated here rather than
imported because ``storage`` sits below ``manifest`` in the layer contract, the
same reason ``storage/layout.py`` holds the directory names that
``manifest.paths`` re-exports.

This is also why nothing here goes through ``storage/atomic.py``. That module
exists for one failure: ``open(path, "w")`` truncates the old content before the
new content is written, so ENOSPC destroys the file. SQLite has no such moment —
a transaction that runs out of space rolls back and leaves the database as it
was — so the rule that sends every other write in rebasis through an atomic
rename is answering a question this file does not raise.

**Concurrency.** Two ``rebasis probe`` runs against one state directory is an
ordinary thing for someone to do. WAL lets them read while a third writes, and
``busy_timeout`` makes a writer wait rather than fail; a wait that expires is a
counted write failure and nothing more, because two processes racing to write
the same key are writing the same bytes — the key is derived from the text. No
state lock is taken: ``storage/locks.py`` lists ``probe`` among the commands
that hold none, and a cache is not a reason to change that.

**What is on disk.** The vectors, and digests of the text — never the text
itself. That is not a privacy guarantee: text can be recovered from an
embedding, so a vector is as sensitive as the document it came from, which is
why ``record_decision`` keeps both out of the audit trail. It is a statement of
scope. The cache sits beside the index whose vectors it duplicates, under a
directory ``gc`` already collects, and it exposes nothing the index did not.
"""

from __future__ import annotations

import hashlib
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from rebasis.storage.layout import CACHE_DIR, STATE_DIR_NAME
from rebasis.types import as_float32

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Mapping, Sequence
    from typing import Self

    from rebasis.types import Embedder, EncodingProfile, FloatArray, TextKind

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "EMBEDDINGS_DIR",
    "KEY_VERSION",
    "STORED_NORMALIZED",
    "CacheStats",
    "CachedEmbedder",
    "EmbeddingCache",
    "cache_enabled",
    "cache_file_for",
    "default_embedding_cache_dir",
    "embedding_key",
    "open_cached_embedder",
]

#: Subdirectory of the cache directory these files live in.
#:
#: Beside ``cascade/`` rather than mixed into it: the two caches key different
#: things (a text here, a record id there) and expire on different clocks, and a
#: user looking at ``du`` output should be able to tell which is which.
EMBEDDINGS_DIR = "embeddings"

#: Schema of one cache file, recorded in ``PRAGMA user_version``.
#:
#: A file written by a newer release is refused rather than read: reading rows
#: under older semantics is the stale-vector failure in a different costume.
CACHE_SCHEMA_VERSION = 1

#: Version of the key scheme itself.
#:
#: Separate from the schema because the two change for different reasons. A new
#: column is a schema change and the old rows survive it; a change to what the
#: key *covers* must make every old row unreachable, and bumping this is how.
KEY_VERSION = 1

#: Whether the vectors :class:`CachedEmbedder` stores have been ℓ2-normalised.
#:
#: They have not. It memoises :meth:`rebasis.types.Embedder.encode` exactly, and
#: what a caller does afterwards is the caller's business —
#: ``probe/session.py`` normalises the assembled matrix rather than each chunk.
#: The flag is in the key rather than assumed, because a normalised vector and a
#: raw one are indistinguishable once written and reading the wrong one would
#: not raise.
STORED_NORMALIZED = False

_FLOAT32_BYTES = 4

#: How long a writer waits for another process before giving up, in ms. The same
#: five seconds ``manifest/db.py`` uses, and for the same reason: waiting is
#: cheaper than failing when the contention is another rebasis run.
_BUSY_TIMEOUT_MS = 5_000

#: Characters of the profile fingerprint carried in a file name. Identity lives
#: in the key inside the file; this only has to be unambiguous on a listing.
_NAME_DIGEST = 16

#: Ceiling on the model-id part of a file name.
_NAME_MAX = 40

_PRAGMAS = (
    "PRAGMA journal_mode = WAL",
    "PRAGMA synchronous = NORMAL",
    f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vectors (
    key    TEXT    PRIMARY KEY,
    dim    INTEGER NOT NULL,
    vector BLOB    NOT NULL
)
"""

#: Both statements are fixed text with bound parameters. A batched
#: ``WHERE key IN (...)`` would have to interpolate one placeholder per key into
#: the statement, and the saving — one round trip against a seek per key inside
#: an already-open file — is not worth a dynamically built SQL string in a
#: module whose whole job is to be boring.
_SELECT_ONE = "SELECT dim, vector FROM vectors WHERE key = ?"
_INSERT_ONE = "INSERT OR REPLACE INTO vectors (key, dim, vector) VALUES (?, ?, ?)"


def default_embedding_cache_dir(state_dir: Path | str | None = None) -> Path:
    """Resolve where the embedding cache lives.

    ``.rebasis/cache/embeddings/`` — under the directory ``rebasis gc`` already
    collects on a 30-day retention and that ``REBASIS_CACHE_DIR`` already names
    ("sample and embedding cache"). Inventing a location would mean a second
    thing for a user to find, a second thing to clean up, and a cache that
    outlives the state directory it belongs to.

    The state directory's default is restated rather than taken from
    :func:`rebasis.manifest.paths.default_state_dir`: ``storage`` sits below
    ``manifest`` in the layer contract, which is the same reason
    ``storage/layout.py`` owns the names that ``manifest.paths`` re-exports.

    Args:
        state_dir: The state directory to put the cache under, for a caller that
            already has one — ``rebasis probe --state-dir`` does, and the cache
            belongs beside the audit trail rather than in whatever directory the
            command happened to be run from.

    ``REBASIS_CACHE_DIR`` still outranks an explicit ``state_dir``, which looks
    like an exception to ``config.py``'s "an explicit argument outranks the
    environment" and is not one. The two name different things: ``--state-dir``
    says where rebasis keeps its state, ``REBASIS_CACHE_DIR`` says where the
    cache goes, and the more specific answer is the one that was asked.
    """
    from rebasis.config import settings

    resolved = settings()
    if resolved.cache_dir:
        return Path(resolved.cache_dir) / EMBEDDINGS_DIR
    if state_dir is not None:
        return Path(state_dir) / CACHE_DIR / EMBEDDINGS_DIR
    root = Path(resolved.state_dir) if resolved.state_dir else Path.cwd() / STATE_DIR_NAME
    return root / CACHE_DIR / EMBEDDINGS_DIR


def cache_enabled() -> bool:
    """Whether the user has left embedding caching on — ``REBASIS_EMBED_CACHE``.

    On by default, unlike :class:`rebasis.serve.cascade.DiskVectorCache`, and the
    difference is who is asking. ``Cascade`` is constructed by a user inside
    their own serving process, where nothing implies a state directory and
    starting to write into one unasked would be a library taking a liberty.
    ``probe`` reaches this through a CLI that has already created ``.rebasis/``
    for the audit trail, in a directory the user pointed it at — the directory
    exists, its retention policy exists, and the collector for it exists.

    The switch is read here rather than at each call site so that "do not cache
    my embeddings" holds wherever the directory happened to be configured.
    """
    from rebasis.config import settings

    return settings().embed_cache


def embedding_key(
    fingerprint: str,
    text: str,
    *,
    kind: str,
    normalized: bool = STORED_NORMALIZED,
    dtype: str = "float32",
) -> str:
    """Build the cache key for one text under one encoding profile.

    Everything that can change the vector is in here, because none of it can be
    recovered from the bytes on disk and a vector produced under the wrong one
    would be scored rather than rejected. See the module docstring.

    Args:
        fingerprint: :meth:`rebasis.types.EncodingProfile.fingerprint` of the
            model that produced — or is about to produce — the vector.
        text: The exact text handed to the model, prefix excluded: the prefix is
            a property of the profile and is already inside the fingerprint.
        kind: ``query`` or ``document``. Kept in the key even for a symmetric
            profile, where the two encodings are identical: "symmetric" is a
            claim about how a model is described, and the cost of being wrong
            about it is a wrong answer rather than a duplicated row.
        normalized: Whether the stored vector has been ℓ2-normalised.
        dtype: How the vector is stored.

    Returns:
        A 64-character hex digest, safe to use as a file name or a primary key.
    """
    material = "\x1f".join(
        (
            str(KEY_VERSION),
            fingerprint,
            kind,
            str(int(normalized)),
            dtype,
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def cache_file_for(profile: EncodingProfile, directory: Path | str | None = None) -> Path:
    """Path of the cache file that holds one profile's vectors.

    Named ``<model>-<fingerprint prefix>.sqlite``. The model id is there so that
    someone reading a directory listing can tell which of five 30 MB files
    belongs to the candidate they abandoned; the fingerprint is what separates
    two profiles of the same model. Neither is load-bearing — every row carries
    the full fingerprint, so even two profiles that collided into one file could
    not read each other's vectors.
    """
    root = Path(directory) if directory is not None else default_embedding_cache_dir()
    return root / f"{_slug(profile.model_id)}-{profile.fingerprint()[:_NAME_DIGEST]}.sqlite"


def open_cached_embedder(
    embedder: Embedder, *, directory: Path | str | None = None
) -> CachedEmbedder | None:
    """Wrap ``embedder`` so it reuses vectors it has already computed.

    Args:
        embedder: The model to memoise. Untouched — this reads its ``profile``
            and calls ``encode``, and does nothing else to it.
        directory: Where the cache files live. Defaults to
            :func:`default_embedding_cache_dir`.

    Returns:
        The wrapper, or ``None`` when ``REBASIS_EMBED_CACHE`` is off — in which
        case the caller keeps using the embedder it already has and nothing is
        written anywhere. An unwritable directory does **not** produce ``None``:
        it produces a wrapper whose every lookup misses, which is the same
        outcome one layer down and keeps the failure counted rather than
        branched on.
    """
    if not cache_enabled():
        return None
    return CachedEmbedder(embedder, EmbeddingCache(cache_file_for(embedder.profile, directory)))


@dataclass(slots=True)
class CacheStats:
    """What the cache did, counted rather than logged.

    Counted because the alternative is a log line per lookup in a loop that runs
    forty times per probe, and because the numbers worth having — the hit rate,
    and whether anything failed to write — are properties of a whole run rather
    than of one batch. The counters are not locked: an increment can be lost
    under concurrency, which is acceptable for an instrument that nothing
    depends on, and is the same trade :class:`rebasis.serve.cascade.CascadeStats`
    makes.
    """

    #: Texts answered from disk.
    hits: int = 0
    #: Texts the cache did not hold, including every text when it is unusable.
    misses: int = 0
    #: Texts actually sent to the model — the cost a warm cache drives to zero,
    #: and the denominator ``probe`` extrapolates a reindex from.
    encoded: int = 0
    #: Time inside the model alone, excluding the lookups around it.
    encode_seconds: float = 0.0
    #: Reads that failed outright. A damaged row is a miss, not a read failure;
    #: this counts the queries that could not run.
    read_failures: int = 0
    #: Vectors this cache could not store — a full disk, a read-only mount, a
    #: writer that waited out ``busy_timeout``. Counted rather than raised.
    write_failures: int = 0
    #: Cached vectors discarded because their width disagreed with what the
    #: model produced in the same call. Should be zero; see
    #: :meth:`CachedEmbedder.encode` for the one thing that moves it.
    width_mismatches: int = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of texts the cache answered.

        ``nan`` before the first lookup: a cache that has been asked nothing has
        no hit rate, and 0.0 would read as one that is not working.
        """
        asked = self.hits + self.misses
        return self.hits / asked if asked else float("nan")

    def to_dict(self) -> dict[str, float]:
        """Serialisable form, for a report or a diagnostic."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "encoded": self.encoded,
            "encode_seconds": round(self.encode_seconds, 3),
            "read_failures": self.read_failures,
            "write_failures": self.write_failures,
            "width_mismatches": self.width_mismatches,
        }


class EmbeddingCache:
    """A key-to-vector memo in one SQLite file.

    The keys are opaque: :func:`embedding_key` builds them and this treats them
    as bytes. That is deliberate, and it is what lets the same class serve two
    callers with different notions of identity — it satisfies
    :class:`rebasis.serve.cascade.VectorCache` as it stands, so a serving
    process whose working set is larger than a directory of small files can hold
    comfortably can pass one of these to :class:`~rebasis.serve.cascade.Cascade`
    instead of a ``DiskVectorCache``.

    Both methods take a whole batch — a caller misses on a sample, not on one
    text — which is what lets a write become one transaction, and what keeps the
    signature identical to the protocol ``serve.cascade`` already defines.

    Args:
        path: The SQLite file. Its parent is created on first use.
        stats: Where to count. Passed in when several caches report into one
            set of counters — ``probe`` opens one per model and wants a single
            number for the run.
    """

    __slots__ = ("_connection", "_lock", "_path", "_unusable", "stats")

    def __init__(self, path: Path | str, *, stats: CacheStats | None = None) -> None:
        self._path = Path(path)
        self._connection: sqlite3.Connection | None = None
        # Set once the file has proved unusable, so a run against a read-only
        # mount does not retry `mkdir` and `connect` on every batch. There is no
        # recovery to wait for: the reasons this flag gets set do not change
        # halfway through a probe.
        self._unusable = False
        # A serving process calls this from whichever thread the request landed
        # on, and one sqlite3 connection is not safe to share between threads by
        # itself. One lock rather than a connection per thread: a lookup takes
        # microseconds, and per-thread connections leak one file handle per
        # thread that ever touched the cache.
        self._lock = threading.RLock()
        self.stats = CacheStats() if stats is None else stats

    @property
    def path(self) -> Path:
        """Where this cache keeps its file."""
        return self._path

    @property
    def usable(self) -> bool:
        """Whether the file has been opened and has not failed.

        ``True`` before the first lookup, when nothing has been tried yet — this
        answers "has it failed", not "will it work".
        """
        return not self._unusable

    def get(self, keys: Sequence[str]) -> dict[str, FloatArray]:
        """Read the vectors held for ``keys``. Anything unreadable is a miss.

        Absent keys are simply absent, and so are damaged ones: a row whose blob
        length disagrees with its recorded dimension has been truncated, and the
        honest thing to do with it is embed the text again.
        """
        import sqlite3

        unique = list(dict.fromkeys(keys))
        found: dict[str, FloatArray] = {}
        if not unique:
            return found

        with self._lock:
            connection = self._open()
            if connection is not None:
                try:
                    self._select(connection, unique, found)
                except (sqlite3.Error, OSError):
                    self.stats.read_failures += 1

        self.stats.hits += len(found)
        self.stats.misses += len(unique) - len(found)
        return found

    def put(self, vectors: Mapping[str, FloatArray]) -> None:
        """Store vectors under their keys, in one transaction.

        A failed write increments :attr:`CacheStats.write_failures` and is not
        raised. The work this protects has already been done by the time it is
        called — the vectors exist, the caller has them — and the only thing a
        failure costs is that the next run computes them again. A cache that can
        take a run down is worse than no cache at all.

        Neither the file nor its directory is fsynced beyond what
        ``synchronous=NORMAL`` does. A cache entry a power cut loses is a cache
        miss, and paying for durability to prevent one would cost more than
        re-embedding the document it protects.
        """
        import sqlite3

        rows = [(key, *_to_blob(vector)) for key, vector in vectors.items()]
        if not rows:
            return

        with self._lock:
            connection = self._open()
            if connection is None:
                self.stats.write_failures += len(rows)
                return
            try:
                # BEGIN IMMEDIATE rather than the default deferred begin, for
                # the reason `manifest/db.py` gives: it takes the write lock up
                # front, so contention with another probe surfaces here instead
                # of at commit time with the work already done.
                connection.execute("BEGIN IMMEDIATE")
                connection.executemany(_INSERT_ONE, rows)
                connection.execute("COMMIT")
            except (sqlite3.Error, OSError):
                self.stats.write_failures += len(rows)
                with suppress(sqlite3.Error):
                    connection.execute("ROLLBACK")

    def close(self) -> None:
        """Close the file. Safe to call more than once.

        Worth calling: a clean close checkpoints the write-ahead log and removes
        the ``-wal`` and ``-shm`` sidecars, which leaves one file per model
        where ``gc`` and a user's ``du`` both expect one.
        """
        import sqlite3

        with self._lock:
            if self._connection is not None:
                with suppress(sqlite3.Error):
                    self._connection.close()
                self._connection = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"EmbeddingCache({self._path.name}, {self.stats.hits} hits)"

    # ── internals ─────────────────────────────────────────────────────

    def _select(
        self, connection: sqlite3.Connection, keys: list[str], into: dict[str, FloatArray]
    ) -> None:
        """Read one batch of keys.

        One statement per key, prepared once and cached by ``sqlite3``, so this
        is a B-tree seek rather than a syscall. See :data:`_SELECT_ONE`.
        """
        for key in keys:
            row = connection.execute(_SELECT_ONE, (key,)).fetchone()
            if row is None:
                continue
            vector = _from_blob(row[1], row[0])
            if vector is not None:
                into[key] = vector

    def _open(self) -> sqlite3.Connection | None:
        """Open the file, or report that it cannot be used. Never raises."""
        import sqlite3

        if self._connection is not None:
            return self._connection
        if self._unusable:
            return None

        connection: sqlite3.Connection | None = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                self._path,
                isolation_level=None,
                # The lock above serialises access; sqlite3's own thread check
                # would refuse a serving process's second thread outright.
                check_same_thread=False,
                timeout=_BUSY_TIMEOUT_MS / 1000,
            )
            for pragma in _PRAGMAS:
                connection.execute(pragma)
            if not self._prepare(connection):
                connection.close()
                self._unusable = True
                return None
        except (sqlite3.Error, OSError, ValueError):
            if connection is not None:
                with suppress(sqlite3.Error):
                    connection.close()
            self._unusable = True
            return None

        self._connection = connection
        return connection

    def _prepare(self, connection: sqlite3.Connection) -> bool:
        """Create or check the schema. ``False`` when the file must be left alone.

        A file written by a newer release is refused rather than migrated
        backwards. Reading rows whose meaning has changed is the stale-vector
        failure again, and "the cache did nothing today" is a cost the user can
        afford in a way that a silently wrong measurement is not.
        """
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version > CACHE_SCHEMA_VERSION:
            return False
        connection.execute(_SCHEMA)
        if version != CACHE_SCHEMA_VERSION:
            connection.execute(f"PRAGMA user_version = {CACHE_SCHEMA_VERSION}")
        return True


class CachedEmbedder:
    """An embedder that answers from the cache before it asks the model.

    A read-through memo of :meth:`rebasis.types.Embedder.encode` and nothing
    else: it applies no prefix, normalises nothing, and knows nothing about the
    model beyond its profile. What it owns is the part that is easy to get wrong
    — a partial hit sends only the missing texts to the model, and the array
    that comes back has **one row per text the caller passed, in the caller's
    order**, never a short one and never a reordered one.

    It satisfies :class:`rebasis.types.Embedder`, so it substitutes for the
    embedder it wraps with no other change at the call site.

    Args:
        embedder: The model to memoise.
        cache: Where its vectors live. One file per profile, so this is the
            cache for *this* embedder and not a shared one — see
            :func:`open_cached_embedder`.

    Example:
        ```python
        cached = open_cached_embedder(model, directory=default_embedding_cache_dir())
        vectors = (cached or model).encode(texts, kind="document")
        ```
    """

    __slots__ = ("_cache", "_embedder", "_fingerprint", "profile", "stats")

    def __init__(self, embedder: Embedder, cache: EmbeddingCache) -> None:
        self._embedder = embedder
        self._cache = cache
        # The wrapped model's profile, unchanged: this caches vectors, it does
        # not describe a different model.
        self.profile = embedder.profile
        # Resolved once. Every key carries it, so a vector produced under one
        # profile is unreachable under another — which matters more here than
        # almost anywhere else, because what consumes these vectors is a
        # measurement that turns into a recommendation.
        self._fingerprint = embedder.profile.fingerprint()
        self.stats = cache.stats

    @property
    def cache(self) -> EmbeddingCache:
        """The cache behind this embedder."""
        return self._cache

    def encode(
        self,
        texts: Sequence[str],
        *,
        kind: TextKind = "document",
        batch_size: int = 32,
        progress: bool = True,
    ) -> FloatArray:
        """Encode ``texts``, sending only the ones the cache does not hold.

        Duplicate texts collapse to one key, so a corpus that repeats a
        boilerplate header embeds it once rather than once per occurrence, and
        both positions still get a row.
        """
        if not texts:
            return self._embedder.encode(texts, kind=kind, batch_size=batch_size, progress=progress)

        keys = [embedding_key(self._fingerprint, text, kind=kind) for text in texts]
        text_of: dict[str, str] = {}
        for key, text in zip(keys, texts, strict=True):
            text_of.setdefault(key, text)

        held = self._cache.get(list(text_of))
        missing = [key for key in text_of if key not in held]
        produced: dict[str, FloatArray] = (
            self._run(text_of, missing, kind=kind, batch_size=batch_size, progress=progress)
            if missing
            else {}
        )

        # One width per call, or these rows cannot be stacked into an answer.
        # The model wins when it produced anything, because it is the only party
        # here that knows what the model does now; when nothing missed there is
        # no such authority and the rows written under this fingerprint are
        # taken at their word.
        #
        # That is a narrower guard than it looks, and deliberately so. A cache
        # cannot detect that the weights behind a fingerprint changed — the
        # fingerprint hashes how a model is *described*, not what it contains —
        # and nothing here pretends otherwise. What this does is stop a
        # disagreement that *is* visible from becoming a matrix of two widths:
        # padding or truncating would be a silently wrong answer, raising would
        # take a run down over a cache, so the disagreeing rows are re-encoded
        # and overwritten.
        width = _first_width(produced, missing) if missing else _first_width(held, list(text_of))
        stale = [key for key, vector in held.items() if int(vector.shape[-1]) != width]
        if stale:
            self.stats.width_mismatches += len(stale)
            for key in stale:
                del held[key]
            produced.update(
                self._run(text_of, stale, kind=kind, batch_size=batch_size, progress=progress)
            )

        if produced:
            self._cache.put(produced)
        held.update(produced)
        return as_float32(np.vstack([held[key] for key in keys]))

    def close(self) -> None:
        """Close the cache behind this embedder."""
        self._cache.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"CachedEmbedder({self.profile.model_id}, {self._cache.path.name})"

    # ── internals ─────────────────────────────────────────────────────

    def _run(
        self,
        text_of: Mapping[str, str],
        keys: Sequence[str],
        *,
        kind: TextKind,
        batch_size: int,
        progress: bool,
    ) -> dict[str, FloatArray]:
        """Encode the texts behind ``keys``, timing the model and not the cache.

        The time is kept separately because ``probe`` extrapolates the cost of a
        full reindex from it. Charging a lookup to the model would inflate that
        estimate by however long the cache took to answer.
        """
        started = time.perf_counter()
        block = as_float32(
            self._embedder.encode(
                [text_of[key] for key in keys],
                kind=kind,
                batch_size=batch_size,
                progress=progress,
            )
        )
        self.stats.encode_seconds += time.perf_counter() - started
        self.stats.encoded += len(keys)
        if block.ndim == 1:
            block = block.reshape(1, -1)
        # Strict: a backend that returned a different number of rows than it was
        # given has misaligned every text with someone else's vector, and that
        # is not a cache failure to swallow.
        return dict(zip(keys, block, strict=True))


def _slug(model_id: str) -> str:
    """A file-name-safe form of a model id. Cosmetic — identity is the digest."""
    kept = "".join(
        c if (c.isascii() and (c.isalnum() or c in "._-")) else "-" for c in model_id[:_NAME_MAX]
    )
    return kept.strip("-.") or "model"


def _to_blob(vector: FloatArray) -> tuple[int, bytes]:
    """Flatten one vector to its dimension and its raw float32 bytes."""
    flat = np.ascontiguousarray(as_float32(vector).reshape(-1))
    return int(flat.size), flat.tobytes()


def _from_blob(blob: object, dim: object) -> FloatArray | None:
    """Decode one stored vector, treating anything inconsistent as a miss.

    The recorded dimension is checked against the blob's length rather than
    trusted: that is what makes a truncated row — the shape a half-finished
    write leaves — a miss instead of a vector with the tail missing.
    """
    if not isinstance(blob, bytes) or not isinstance(dim, int) or dim <= 0:
        return None
    if len(blob) != dim * _FLOAT32_BYTES:
        return None
    # `frombuffer` views the bytes object, which is immutable, so the array it
    # hands back is read-only. One copy here keeps that out of the caller's way.
    return np.frombuffer(blob, dtype=np.float32).copy()


def _first_width(vectors: Mapping[str, FloatArray], order: Sequence[str]) -> int:
    """Width of the first vector in ``order`` that ``vectors`` holds.

    Ordered by the caller's own key order rather than by whatever the database
    returned, so that which width wins is reproducible.
    """
    for key in order:
        vector = vectors.get(key)
        if vector is not None:
            return int(vector.shape[-1])
    return 0
