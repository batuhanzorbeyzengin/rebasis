"""Embeddings remembered between runs.

The cache exists to remove the most visible everyday cost in the tool: every
``probe`` re-embedded its whole sample from scratch. That makes speed the
*point* — but speed is not what these tests are about, because a cache is only
allowed to make a run cheaper and is never allowed to make it different.

So the file asserts two things, in that order of importance:

* **It cannot change an answer.** A vector produced under one encoding profile
  is unreachable under another, ``query`` and ``document`` never share a row,
  and a damaged row is a miss. None of those failures raise if they are got
  wrong — a stale vector would be *measured*, and the recommendation that came
  out would be a plausible answer to a question nobody asked.
* **It cannot take a run down.** An unwritable directory, a corrupt file, a
  database from a future release: every one of them means "embed it again".

The embedder here counts every text it is asked for and returns a deterministic
vector, so "did the cache return what the model produced" is checkable bit for
bit rather than approximately, and "did it avoid the work" is a number rather
than a timing.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from rebasis.core import l2_normalize
from rebasis.probe.session import probe_store
from rebasis.storage.embedding_cache import (
    CACHE_SCHEMA_VERSION,
    CachedEmbedder,
    EmbeddingCache,
    cache_file_for,
    default_embedding_cache_dir,
    embedding_key,
    open_cached_embedder,
)
from rebasis.storage.gc import DAY, plan_gc
from rebasis.store import MemoryStore
from rebasis.types import EncodingProfile, as_float32

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

pytestmark = pytest.mark.unit

DIM = 16

#: The candidate model. Symmetric, so a probe encodes the corpus once.
NEW = EncodingProfile(model_id="acme/new-model", dim=DIM)

#: The same model, described differently. This is the realistic mistake: a user
#: who ran once without `--query-prefix` and once with it. The vectors differ;
#: nothing about the run says so.
PREFIXED = EncodingProfile(model_id="acme/new-model", dim=DIM, query_prefix="query: ")

TEXTS = ["alpha document", "beta document", "gamma document"]


class CountingEmbedder:
    """An embedder that records every text it was asked to encode.

    Deterministic in the text, the ``kind`` and the model id, so two profiles
    that differ only in a prefix still produce different vectors — which is what
    makes the fingerprint tests below assertions about correctness rather than
    about counters.

    Args:
        profile: What this pretends to be.
        table: Fixed vectors for known texts, when a test needs the corpus and
            the model to relate to each other. Anything absent is derived.
    """

    def __init__(self, profile: EncodingProfile, table: dict[str, Any] | None = None) -> None:
        self.profile = profile
        self.calls: list[list[str]] = []
        self._table = table or {}

    def encode(
        self,
        texts: Sequence[str],
        *,
        kind: str = "document",
        batch_size: int = 32,
        progress: bool = True,
    ) -> Any:
        del batch_size, progress
        self.calls.append(list(texts))
        if not texts:
            return np.empty((0, self.profile.dim), dtype=np.float32)
        return as_float32(np.vstack([self._vector(text, kind) for text in texts]))

    @property
    def seen(self) -> list[str]:
        """Every text handed to the model, over every call."""
        return [text for call in self.calls for text in call]

    def _vector(self, text: str, kind: str) -> Any:
        known = self._table.get(text)
        if known is not None:
            return known
        material = f"{self.profile.model_id}\x1f{self.profile.query_prefix}\x1f{kind}\x1f{text}"
        seed = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")
        return np.random.default_rng(seed).standard_normal(self.profile.dim).astype(np.float32)


def cached(profile: EncodingProfile, directory: Path) -> CachedEmbedder:
    """A counting embedder for ``profile``, wrapped in a cache under ``directory``."""
    return CachedEmbedder(
        CountingEmbedder(profile), EmbeddingCache(cache_file_for(profile, directory))
    )


def counter(wrapper: CachedEmbedder) -> CountingEmbedder:
    """The :class:`CountingEmbedder` a wrapper is holding."""
    model = wrapper._embedder
    assert isinstance(model, CountingEmbedder)
    return model


class TestAWarmCacheReturnsTheSameVectors:
    """The premise. If it returns anything else, nothing below matters."""

    def test_the_second_run_embeds_nothing(self, tmp_path: Path) -> None:
        first = cached(NEW, tmp_path)
        first.encode(TEXTS, kind="document")
        first.close()

        second = cached(NEW, tmp_path)
        second.encode(TEXTS, kind="document")

        assert counter(second).seen == []
        assert second.stats.hits == len(TEXTS)
        assert second.stats.encoded == 0

    def test_the_vectors_are_identical_bit_for_bit(self, tmp_path: Path) -> None:
        """Not "close": these feed a decision, and a cached run and a cold run
        that disagree in the last bit are two different measurements."""
        first = cached(NEW, tmp_path)
        cold = first.encode(TEXTS, kind="document")
        first.close()

        warm = cached(NEW, tmp_path).encode(TEXTS, kind="document")

        assert np.array_equal(cold, warm)
        assert warm.dtype == np.float32

    def test_it_survives_the_object_that_wrote_it(self, tmp_path: Path) -> None:
        """A cache that lives only as long as the process is a cache that never
        helps: the cost this removes is paid between runs, not inside one."""
        writer = cached(NEW, tmp_path)
        writer.encode(TEXTS, kind="document")
        writer.close()
        assert cache_file_for(NEW, tmp_path).exists()

        reader = cached(NEW, tmp_path)
        reader.encode(TEXTS, kind="document")

        assert reader.stats.hit_rate == 1.0


class TestAPartialHit:
    """The case that decides whether this is useful at all.

    ``--sample`` is a flag people move, and the second run with slightly
    different arguments is exactly what a cache has to serve. A shape that
    invalidates everything when one text changes would be no better than none.
    """

    def test_only_the_missing_texts_reach_the_model(self, tmp_path: Path) -> None:
        first = cached(NEW, tmp_path)
        first.encode(TEXTS[:2], kind="document")
        first.close()

        second = cached(NEW, tmp_path)
        second.encode([*TEXTS, "delta document"], kind="document")

        assert counter(second).seen == ["gamma document", "delta document"]
        assert second.stats.encoded == 2
        assert second.stats.hits == 2

    def test_the_answer_is_in_the_callers_order(self, tmp_path: Path) -> None:
        """Never a short array and never a reordered one. A silent misalignment
        here would pair every document with someone else's vector and produce a
        plausible-looking number that means nothing."""
        first = cached(NEW, tmp_path)
        first.encode(["beta document"], kind="document")
        first.close()

        second = cached(NEW, tmp_path)
        mixed = second.encode(TEXTS, kind="document")
        reference = CountingEmbedder(NEW).encode(TEXTS, kind="document")

        assert mixed.shape == (len(TEXTS), DIM)
        assert np.array_equal(mixed, reference)

    def test_a_repeated_text_is_embedded_once(self, tmp_path: Path) -> None:
        """Two positions, one key, and both positions still get a row."""
        wrapper = cached(NEW, tmp_path)

        vectors = wrapper.encode(["alpha document"] * 3, kind="document")

        assert counter(wrapper).seen == ["alpha document"]
        assert vectors.shape == (3, DIM)
        assert np.array_equal(vectors[0], vectors[2])


class TestTheKeyPinsWhatProducedTheVector:
    """The sharpest tests here, because this failure does not raise.

    A vector left by another model, another prefix or another pooling would be
    scored rather than rejected, and what came out would look like the candidate
    model performing badly rather than like the wrong vectors being used.
    """

    def test_a_changed_prefix_cannot_read_the_earlier_vectors(self, tmp_path: Path) -> None:
        """Same model id, one flag different — and every text is embedded again.

        The count is the cheap half of this. The half that matters is the second
        assertion: what comes back is what the *prefixed* profile produces, not
        what was already sitting on disk under the same model id.
        """
        plain = cached(NEW, tmp_path)
        plain.encode(TEXTS, kind="document")
        plain.close()

        prefixed = cached(PREFIXED, tmp_path)
        vectors = prefixed.encode(TEXTS, kind="document")

        assert prefixed.stats.hits == 0
        assert counter(prefixed).seen == TEXTS
        assert np.array_equal(vectors, CountingEmbedder(PREFIXED).encode(TEXTS, kind="document"))
        assert not np.array_equal(vectors, CountingEmbedder(NEW).encode(TEXTS, kind="document"))

    def test_two_profiles_do_not_even_share_a_file(self, tmp_path: Path) -> None:
        """Defence in depth, and the reason `gc` can expire one of them.

        The row key alone would keep the two apart. Separate files mean a
        candidate model somebody evaluated once and abandoned can age out
        without the model they still use holding it alive.
        """
        assert cache_file_for(NEW, tmp_path) != cache_file_for(PREFIXED, tmp_path)
        # And the name says which model it is, so "which of these five 30 MB
        # files can I delete" is answerable from a directory listing.
        assert cache_file_for(NEW, tmp_path).name.startswith("acme-new-model-")

    def test_query_and_document_do_not_share_a_vector(self, tmp_path: Path) -> None:
        """For an asymmetric model the two encodings are different vectors, and
        the difference is what the tool is for."""
        wrapper = cached(NEW, tmp_path)
        as_documents = wrapper.encode(TEXTS, kind="document")

        as_queries = wrapper.encode(TEXTS, kind="query")

        assert counter(wrapper).seen == TEXTS * 2
        assert not np.array_equal(as_documents, as_queries)

    def test_the_key_moves_with_every_ingredient(self) -> None:
        """Enumerated rather than implied: each of these changes the vector, so
        each of them has to change the key."""
        base = embedding_key(NEW.fingerprint(), "alpha", kind="document")
        variants = {
            base,
            embedding_key(PREFIXED.fingerprint(), "alpha", kind="document"),
            embedding_key(NEW.fingerprint(), "alpha", kind="query"),
            embedding_key(NEW.fingerprint(), "alpha ", kind="document"),
            embedding_key(NEW.fingerprint(), "alpha", kind="document", normalized=True),
            embedding_key(NEW.fingerprint(), "alpha", kind="document", dtype="float16"),
        }

        assert len(variants) == 6

    def test_the_same_ingredients_give_the_same_key(self) -> None:
        """Or nothing would ever hit."""
        assert embedding_key(NEW.fingerprint(), "alpha", kind="document") == embedding_key(
            NEW.fingerprint(), "alpha", kind="document"
        )


class TestFailureIsAMissAndNeverAnException:
    """A cache that can take a probe down after nine thousand documents is
    worse than no cache, and one that can change an answer is worse than that."""

    def test_an_unwritable_directory_still_returns_the_right_vectors(self, tmp_path: Path) -> None:
        blocked = tmp_path / "a-file"
        blocked.write_text("not a directory")
        wrapper = cached(NEW, blocked / "cache")

        vectors = wrapper.encode(TEXTS, kind="document")

        assert np.array_equal(vectors, CountingEmbedder(NEW).encode(TEXTS, kind="document"))
        assert wrapper.stats.misses == len(TEXTS)
        assert wrapper.stats.write_failures == len(TEXTS)
        assert not wrapper.cache.usable

    def test_an_unwritable_directory_is_given_up_on(self, tmp_path: Path) -> None:
        """There is no recovery to wait for — the reasons a directory cannot be
        created do not change halfway through a run — and a probe reads its
        sample in forty chunks rather than one."""
        blocked = tmp_path / "a-file"
        blocked.write_text("not a directory")
        wrapper = cached(NEW, blocked / "cache")

        wrapper.encode(TEXTS, kind="document")
        assert not wrapper.cache.usable

        wrapper.encode(TEXTS, kind="document")

        assert wrapper.stats.hits == 0
        assert counter(wrapper).seen == TEXTS * 2
        assert wrapper.stats.write_failures == 2 * len(TEXTS)

    def test_a_file_that_is_not_a_database_is_a_miss(self, tmp_path: Path) -> None:
        path = cache_file_for(NEW, tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"this is not a SQLite file, it is a note")

        wrapper = CachedEmbedder(CountingEmbedder(NEW), EmbeddingCache(path))
        vectors = wrapper.encode(TEXTS, kind="document")

        assert np.array_equal(vectors, CountingEmbedder(NEW).encode(TEXTS, kind="document"))
        assert wrapper.stats.hits == 0

    def test_a_truncated_row_is_a_miss(self, tmp_path: Path) -> None:
        """The shape a half-finished write leaves. The recorded dimension is
        checked against the blob rather than trusted, so what comes back is a
        re-embedded vector and never one with its tail missing."""
        writer = cached(NEW, tmp_path)
        expected = writer.encode(TEXTS, kind="document")
        writer.close()
        _sql(cache_file_for(NEW, tmp_path), "UPDATE vectors SET vector = substr(vector, 1, 8)")

        reader = cached(NEW, tmp_path)
        vectors = reader.encode(TEXTS, kind="document")

        assert reader.stats.hits == 0
        assert counter(reader).seen == TEXTS
        assert np.array_equal(vectors, expected)

    def test_a_database_from_a_newer_release_is_left_alone(self, tmp_path: Path) -> None:
        """Reading rows whose meaning may have changed is the stale-vector
        failure again. "The cache did nothing today" is a cost a user can
        afford; a silently wrong measurement is not."""
        writer = cached(NEW, tmp_path)
        writer.encode(TEXTS, kind="document")
        writer.close()
        path = cache_file_for(NEW, tmp_path)
        _sql(path, f"PRAGMA user_version = {CACHE_SCHEMA_VERSION + 1}")

        reader = cached(NEW, tmp_path)
        reader.encode(TEXTS, kind="document")

        assert reader.stats.hits == 0
        assert reader.stats.write_failures == len(TEXTS)
        assert _rows(path) == len(TEXTS), "the newer file was written to anyway"

    def test_a_cached_vector_of_another_width_never_reaches_the_answer(
        self, tmp_path: Path
    ) -> None:
        """A narrower guard than it looks, and deliberately so.

        A cache cannot tell that the weights behind a fingerprint changed — the
        fingerprint hashes how a model is *described* — and this does not
        pretend to. What it guarantees is that when the disagreement is visible,
        because something in the same call had to be embedded, the rows that
        disagree are re-encoded rather than stacked into a matrix of two widths
        or padded into a silently wrong answer.
        """
        writer = cached(NEW, tmp_path)
        writer.encode(TEXTS[:2], kind="document")
        writer.close()
        _sql(
            cache_file_for(NEW, tmp_path),
            "UPDATE vectors SET dim = 4, vector = substr(vector, 1, 16)",
        )

        reader = cached(NEW, tmp_path)
        vectors = reader.encode(TEXTS, kind="document")

        assert vectors.shape == (len(TEXTS), DIM)
        assert np.array_equal(vectors, CountingEmbedder(NEW).encode(TEXTS, kind="document"))
        assert reader.stats.width_mismatches == 2


class TestWhereItLivesAndWhoCleansItUp:
    """`.rebasis/cache/`, not a location of its own — the directory that already
    has a retention policy and a collector."""

    def test_it_writes_where_gc_already_looks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = tmp_path / "state"
        monkeypatch.setenv("REBASIS_STATE_DIR", str(state))
        assert default_embedding_cache_dir() == state / "cache" / "embeddings"

        wrapper = cached(NEW, default_embedding_cache_dir())
        wrapper.encode(TEXTS, kind="document")
        # A clean close checkpoints the write-ahead log, so `gc` and `du` both
        # see one file per model rather than three.
        wrapper.close()

        collected = [
            candidate
            for candidate in plan_gc(state, now=time.time() + 40 * DAY).candidates
            if candidate.category == "cache"
        ]
        assert [c.path.name for c in collected] == [cache_file_for(NEW).name]

    def test_a_named_state_directory_takes_it_along(self, tmp_path: Path) -> None:
        """`rebasis probe --state-dir` puts the audit trail somewhere; the cache
        belongs beside it, not in whatever directory the command was run from."""
        assert (
            default_embedding_cache_dir(tmp_path / "named")
            == tmp_path / "named" / "cache" / "embeddings"
        )

    def test_the_cache_directory_variable_still_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`--state-dir` says where the state goes; `REBASIS_CACHE_DIR` says
        where the cache goes, and the more specific answer is the one asked."""
        monkeypatch.setenv("REBASIS_STATE_DIR", str(tmp_path / "state"))
        monkeypatch.setenv("REBASIS_CACHE_DIR", str(tmp_path / "elsewhere"))

        assert default_embedding_cache_dir() == tmp_path / "elsewhere" / "embeddings"
        assert default_embedding_cache_dir(tmp_path / "named") == (
            tmp_path / "elsewhere" / "embeddings"
        )

    def test_the_user_can_turn_it_off(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reason to reach for this is distrust — "is this number real, or
        did it come from a cache?" — so it has to be possible without deleting a
        directory whose layout the user should not have to know."""
        monkeypatch.setenv("REBASIS_EMBED_CACHE", "0")

        assert open_cached_embedder(CountingEmbedder(NEW), directory=tmp_path) is None
        assert not list(tmp_path.iterdir())

    def test_it_is_on_when_nothing_is_said(self, tmp_path: Path) -> None:
        assert open_cached_embedder(CountingEmbedder(NEW), directory=tmp_path) is not None


class TestTheServingProtocol:
    """`EmbeddingCache` is offered to `serve.cascade` as a `VectorCache`.

    Its keys are opaque there — `Cascade` builds them from a record id — so the
    round trip has to work on any string, not only on a key this module built.
    """

    def test_it_round_trips_opaque_keys(self, tmp_path: Path) -> None:
        vector = l2_normalize(np.arange(DIM, dtype=np.float32) + 1.0)

        with EmbeddingCache(tmp_path / "cascade.sqlite") as cache:
            cache.put({"fingerprint:doc-0007": vector})
            found = cache.get(["fingerprint:doc-0007", "fingerprint:doc-0008"])

        assert list(found) == ["fingerprint:doc-0007"]
        assert np.array_equal(found["fingerprint:doc-0007"], vector)

    def test_an_empty_request_asks_the_file_nothing(self, tmp_path: Path) -> None:
        with EmbeddingCache(tmp_path / "cascade.sqlite") as cache:
            assert cache.get([]) == {}
            cache.put({})

        assert not (tmp_path / "cascade.sqlite").exists()

    def test_a_hit_rate_over_no_lookups_says_nothing(self, tmp_path: Path) -> None:
        """`nan` rather than 0.0, which would read as a cache that is not
        working rather than one that has not been asked anything."""
        import math

        cache = EmbeddingCache(tmp_path / "cascade.sqlite")

        assert math.isnan(cache.stats.hit_rate)
        assert "hit_rate" in cache.stats.to_dict()


class TestTheProbeDoesNotPayTwice:
    """The whole point, and the one number the cache could have corrupted."""

    def test_the_second_probe_embeds_nothing_and_decides_the_same(self, tmp_path: Path) -> None:
        world = _world()

        cold, _ = probe_store(world["store"], world["cold"], cache_dir=tmp_path, **_PROBE)
        warm, _ = probe_store(world["store"], world["warm"], cache_dir=tmp_path, **_PROBE)

        assert world["cold"].seen, "the first probe embedded nothing at all"
        assert world["warm"].seen == [], "the second probe re-embedded the corpus"
        assert warm.best.arr == pytest.approx(cold.best.arr)
        assert warm.decision.decision == cold.decision.decision

    def test_a_warm_run_reports_no_reindex_cost_rather_than_none_at_all(
        self, tmp_path: Path
    ) -> None:
        """A cache must not be able to make a reported number wrong. The rate is
        extrapolated from the documents this run actually embedded, so a run
        that embedded none has no rate — and "a full reindex takes no time" is
        the one answer that must not come out of it."""
        world = _world()

        cold, _ = probe_store(world["store"], world["cold"], cache_dir=tmp_path, **_PROBE)
        warm, _ = probe_store(world["store"], world["warm"], cache_dir=tmp_path, **_PROBE)

        # The whole collection, not the sample — `estimate_reindex_cost` says so
        # in as many words, and a reindex estimate over the sample would be the
        # wrong number by a factor of the sampling rate.
        assert cold.reindex_cost["n_documents"] == 320
        assert "seconds_per_document" in cold.reindex_cost
        assert warm.reindex_cost == {}

    def test_without_a_directory_nothing_is_written(self, tmp_path: Path) -> None:
        """`probe_store` is importable, and a library that starts writing into
        someone's project directory unasked has taken a liberty."""
        world = _world()

        probe_store(world["store"], world["cold"], **_PROBE)

        assert not list(tmp_path.iterdir())


#: Small, cheap and deterministic: these tests are about the cache, not about
#: retrieval quality, and a probe large enough to say anything about the latter
#: has no business in a unit file.
# `size - heldout` must clear `rebasis.sample.strategies.MIN_SAMPLE`, which is
# 200: below it the confidence interval is wider than the decision bands, so
# `draw_sample` refuses rather than returning a coin flip dressed as a
# recommendation. 260 - 40 = 220 clears it with room, and the store below holds
# more than the sample so that sampling is doing something.
_PROBE: dict[str, Any] = {
    "size": 260,
    "heldout": 40,
    "strategy": "random",
    "methods": ["procrustes"],
    "with_csls": False,
    "seed": 5,
}


def _world() -> dict[str, Any]:
    """A store in the old space and two identical views of the new model.

    Two embedder objects rather than one, so that "the second probe embedded
    nothing" is a fact about the cache rather than about a counter that was
    never reset.
    """
    rng = np.random.default_rng(11)
    n = 320
    centers = (rng.standard_normal((8, DIM)) * 3.0).astype(np.float32)
    old = l2_normalize(
        centers[rng.integers(0, 8, size=n)] + rng.standard_normal((n, DIM)).astype(np.float32) * 1.2
    )
    rotation = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype(np.float32)
    new = l2_normalize(old @ rotation.T + rng.standard_normal((n, DIM)).astype(np.float32) * 0.05)

    ids = [f"doc-{i:04d}" for i in range(n)]
    texts = [f"document number {i}" for i in range(n)]
    table = dict(zip(texts, new, strict=True))
    return {
        "store": MemoryStore(ids, old.copy(), texts),
        "cold": CountingEmbedder(NEW, table),
        "warm": CountingEmbedder(NEW, table),
    }


def _sql(path: Path, statement: str) -> None:
    """Run one statement against a cache file, from outside."""
    connection = sqlite3.connect(path)
    try:
        connection.execute(statement)
        connection.commit()
    finally:
        connection.close()


def _rows(path: Path) -> int:
    """How many vectors a cache file holds."""
    connection = sqlite3.connect(path)
    try:
        return int(connection.execute("SELECT count(*) FROM vectors").fetchone()[0])
    finally:
        connection.close()
