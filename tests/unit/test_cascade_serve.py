"""Two-stage serving: the bridge recalls, the new model ranks.

`docs/cascade-band.md` measured this arrangement on sixteen real corpora. These
tests are not a second measurement of it — a synthetic corpus cannot say
anything about retrieval quality that BEIR has not already said better. They
assert the properties the *code* has to have for that measurement to describe
what a user gets:

* the final ranking is produced by the new model, in the new model's space;
* what the bridge failed to recall is unrecoverable, and stays that way;
* the cache is what makes the arrangement affordable, and a vector produced by
  one model is unreachable under another;
* a document the new model cannot score is kept, not dropped;
* the cost is reported honestly enough to decide whether to pay it.

The corpus is built so those properties are visible. The old space is a rotation
of the new one that has lost detail — enough noise to scramble the ordering
inside a neighbourhood, not enough to move a document out of one. That is the
regime the measurement describes: recall survives, ranking does not. Every
document's text embeds back to its own new-model vector, so a document queried
by its own vector scores exactly 1.0 and *must* come first — which makes "the
new model produced this ranking" a fact about the output rather than a claim
about it.
"""

from __future__ import annotations

import math
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from rebasis.core import ProcrustesAdapter, l2_normalize
from rebasis.core.serialization import AdapterManifest
from rebasis.embed import PrecomputedEmbedder
from rebasis.errors import CapabilityMissing
from rebasis.serve.bridge import Bridge
from rebasis.serve.cascade import (
    Cascade,
    CascadeStats,
    DiskVectorCache,
    MemoryVectorCache,
    default_cache_dir,
)
from rebasis.storage.gc import DAY, plan_gc
from rebasis.store import MemoryStore
from rebasis.types import EncodingProfile

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

DIM = 32
N = 300

#: How many documents are queried by their own vector in a hit-rate check.
PROBES = 40

#: Scale of the detail the old space lost.
#:
#: Tuned to the regime the measurement is about, and measured across it: at 0.20
#: a document's own bridged query recalls it into the top 100 in 40 probes out
#: of 40 and ranks it first in 6. At 0.09 the bridge already ranks it first 31
#: times, so the control below stops being a control; at 0.50 the candidate set
#: starts losing the answer and the rerank has nothing to work with.
OLD_SPACE_NOISE = 0.20

NEW = EncodingProfile(model_id="new/model", dim=DIM)
OTHER = EncodingProfile(model_id="other/model", dim=DIM)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(17)


@pytest.fixture
def world(rng: np.random.Generator) -> dict[str, Any]:
    """A store in the old space, a bridge into it, and the new model.

    Clustered rather than uniform, for the reason `test_index_health.py` gives:
    on uniform vectors every neighbour is nearly equidistant and any comparison
    between two rankings becomes a test of tie-breaking.
    """
    centers = (rng.standard_normal((12, DIM)) * 3.0).astype(np.float32)
    assignment = rng.integers(0, 12, size=N)
    new = l2_normalize(centers[assignment] + rng.standard_normal((N, DIM)).astype(np.float32) * 1.1)

    rotation = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype(np.float32)
    noise = rng.standard_normal((N, DIM)).astype(np.float32) * OLD_SPACE_NOISE
    old = l2_normalize(new @ rotation.T + noise)

    ids = [f"doc-{i:04d}" for i in range(N)]
    texts = [f"document number {i}" for i in range(N)]
    adapter = ProcrustesAdapter.fit(new, old)

    return {
        "ids": ids,
        "texts": texts,
        "new": new,
        "store": MemoryStore(ids, old.copy(), texts),
        "bridge": Bridge(adapter, _manifest(adapter)),
        "embedder": PrecomputedEmbedder(NEW, dict(zip(texts, new, strict=True))),
    }


def _manifest(adapter: ProcrustesAdapter) -> AdapterManifest:
    return AdapterManifest(
        schema=1,
        adapter_type=adapter.type_name,
        direction="query_to_old",
        input_dim=adapter.input_dim,
        output_dim=adapter.output_dim,
        old_profile_fingerprint="old",
        new_profile_fingerprint="new",
        old_model_id="old-model",
        new_model_id="new-model",
        rebasis_version="test",
        config={},
        tensor_hashes={},
        symmetric=True,
        created_utc="2026-01-01T00:00:00+00:00",
    )


def cascade(world: dict[str, Any], **kwargs: Any) -> Cascade:
    return Cascade(world["store"], world["bridge"], world["embedder"], **kwargs)


def first_hit_rate(world: dict[str, Any], search: Any, *, k: int = 1) -> float:
    """How often a document's own query puts that document first.

    Every document is its own best answer under the new model — its vector
    scores exactly 1.0 against itself — so a correct two-stage arrangement finds
    it whenever the bridge recalled it. That makes one number cover both stages.
    """
    found = 0
    for i in range(PROBES):
        hits = search(world["new"][i], k)
        found += bool(hits) and hits[0].id == world["ids"][i]
    return found / PROBES


def bridged_only(world: dict[str, Any]) -> Any:
    """The control: the bridge producing the final ranking, as `Bridge` does."""

    def search(vector: np.ndarray, k: int) -> Any:
        return world["store"].search(world["bridge"].to_index_space(vector), k=k)

    return search


class TestTheNewModelProducesTheRanking:
    """Step 5 is the new model scoring its own vectors, or the finding is void."""

    def test_the_target_comes_first(self, world: dict[str, Any]) -> None:
        """A document scores 1.0 against its own query and nothing scores more,
        so reaching the candidate set is the only thing that can be missing."""
        assert first_hit_rate(world, cascade(world).search) > 0.95

    def test_the_bridge_alone_does_not(self, world: dict[str, Any]) -> None:
        """The control, and the reason the test above means anything.

        Single-stage bridging — what a user following the `Bridge` documentation
        does today — ranks in the old space, and the old space here has lost the
        detail that decides a top-1. Without this assertion the test above would
        pass just as happily on a corpus that was never hard. Measured at 0.150
        here, against 1.000 for the two-stage arrangement.
        """
        assert first_hit_rate(world, bridged_only(world)) < 0.5

    def test_the_candidate_set_is_the_ceiling(self, world: dict[str, Any]) -> None:
        """What bounds this arrangement is recall@N and nothing else.

        A relevant document that never reaches the candidate set cannot be
        recovered by any amount of reranking, and asking for a candidate set of
        two makes that concrete rather than theoretical.
        """
        deep = first_hit_rate(world, cascade(world).search)
        shallow = first_hit_rate(world, cascade(world, candidates=2).search)

        assert shallow < deep

    def test_it_returns_no_more_than_the_index_holds(self, world: dict[str, Any]) -> None:
        small = MemoryStore(world["ids"][:3], world["store"]._vectors[:3], world["texts"][:3])
        hits = Cascade(small, world["bridge"], world["embedder"]).search(world["new"][0], k=10)

        assert len(hits) == 3
        assert [hit.rank for hit in hits] == [0, 1, 2]


class TestTheCache:
    """The arrangement is unusable without one — `docs/cascade-band.md`."""

    def test_the_second_query_embeds_nothing(self, world: dict[str, Any]) -> None:
        """The whole premise: this is a lazy migration, not a permanent tax."""
        search = cascade(world)
        search.search(world["new"][0])
        cold = search.stats.documents_embedded
        assert cold == search.stats.candidates > 0, "a cold query embedded the candidate set"

        search.search(world["new"][0])

        assert search.stats.documents_embedded == cold, "a warm query re-embedded"
        assert search.stats.hit_rate == pytest.approx(0.5)

    def test_a_different_model_cannot_read_another_s_vectors(
        self, world: dict[str, Any], rng: np.random.Generator
    ) -> None:
        """The failure this key exists to prevent does not raise — it ranks.

        A vector left by the previous model would be scored, returned, and read
        as the upgrade working badly rather than as the wrong vectors being
        used. So the model's profile fingerprint is part of every key, and two
        models sharing one cache share nothing else.
        """
        shared = MemoryVectorCache()
        other = PrecomputedEmbedder(
            OTHER,
            dict(
                zip(
                    world["texts"],
                    l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32)),
                    strict=True,
                )
            ),
        )

        first = cascade(world, cache=shared)
        first.search(world["new"][0])
        second = Cascade(world["store"], world["bridge"], other, cache=shared)
        second.search(world["new"][0])

        assert second.stats.cache_hits == 0, "one model read another model's vectors"
        assert second.stats.documents_embedded == first.stats.documents_embedded
        assert len(shared) == 2 * first.stats.candidates

    def test_the_memory_cache_evicts_the_least_recently_used(self) -> None:
        vector = np.ones(DIM, dtype=np.float32)
        cache = MemoryVectorCache(capacity=2)
        cache.put({"a": vector, "b": vector})
        cache.get(["a"])

        cache.put({"c": vector})

        assert set(cache.get(["a", "b", "c"])) == {"a", "c"}

    def test_the_disk_cache_outlives_the_object(
        self, world: dict[str, Any], tmp_path: Path
    ) -> None:
        """Otherwise the first query after every deploy pays the full cost."""
        cascade(world, cache=DiskVectorCache(tmp_path / "vectors")).search(world["new"][0])

        restarted = cascade(world, cache=DiskVectorCache(tmp_path / "vectors"))
        hits = restarted.search(world["new"][0])

        assert restarted.stats.documents_embedded == 0
        assert restarted.stats.hit_rate == pytest.approx(1.0)
        assert hits[0].id == world["ids"][0]

    def test_the_disk_cache_round_trips_the_vector(self, tmp_path: Path) -> None:
        """Bit for bit: a cached vector is the one the model produced, or the
        ranking it feeds is not the ranking a reindex would have produced."""
        cache = DiskVectorCache(tmp_path)
        vector = l2_normalize(np.arange(DIM, dtype=np.float32) + 1.0)

        cache.put({"key": vector})

        assert np.array_equal(cache.get(["key"])["key"], vector)

    def test_the_disk_cache_writes_where_gc_already_looks(
        self, world: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`.rebasis/cache/`, not a location of its own.

        That directory already has a retention policy and a collector. A cache
        somewhere else would be a second thing to find and a second thing nobody
        cleans up.
        """
        state = tmp_path / "state"
        monkeypatch.setenv("REBASIS_STATE_DIR", str(state))
        assert default_cache_dir() == state / "cache" / "cascade"

        cascade(world, cache=DiskVectorCache()).search(world["new"][0])

        collected = [
            candidate
            for candidate in plan_gc(state, now=time.time() + 40 * DAY).candidates
            if candidate.category == "cache"
        ]
        assert len(collected) == 100

    def test_a_cache_that_cannot_write_does_not_fail_the_query(
        self, world: dict[str, Any], tmp_path: Path
    ) -> None:
        """A cache exists to make queries cheaper. One that can take a query
        down is worse than no cache at all — the search has already succeeded by
        the time the write happens."""
        blocked = tmp_path / "a-file"
        blocked.write_text("not a directory")
        cache = DiskVectorCache(blocked / "cache")

        hits = cascade(world, cache=cache).search(world["new"][0])

        assert len(hits) == 10
        assert hits[0].id == world["ids"][0]
        assert cache.write_failures == 100


class TestWhatCannotBeReEmbedded:
    """A record with no text has no new-model vector, and never will."""

    def test_it_keeps_the_rank_the_bridge_gave_it(self, world: dict[str, Any]) -> None:
        """Dropping it would remove a document from someone's results for a
        reason that has nothing to do with relevance. `probe` may drop a sampled
        record with no text, because a sample is allowed to come back smaller
        than it asked for; a result set is not."""
        query = world["new"][0]
        position = 3
        stranded = bridged_only(world)(query, 10)[position].id
        texts = [
            "" if record_id == stranded else text
            for record_id, text in zip(world["ids"], world["texts"], strict=True)
        ]
        world["store"] = MemoryStore(world["ids"], world["store"]._vectors, texts)

        hits = cascade(world).search(query, k=10)

        assert hits[position].id == stranded
        assert [hit.rank for hit in hits] == list(range(10))

    def test_it_is_counted_rather_than_hidden(self, world: dict[str, Any]) -> None:
        """A large count means the store is the problem, and no amount of cache
        warming will fix it. That is only actionable if it is visible."""
        texts = ["" if i % 2 else text for i, text in enumerate(world["texts"])]
        world["store"] = MemoryStore(world["ids"], world["store"]._vectors, texts)

        search = cascade(world)
        search.search(world["new"][0])

        assert search.stats.kept_bridged > 0
        assert search.stats.documents_embedded > 0

    def test_a_store_that_holds_no_text_is_refused_at_construction(
        self, world: dict[str, Any]
    ) -> None:
        """Its cache could never be filled, so every query would silently return
        the bridged order forever. Failing here beats failing in production on
        the first query."""
        textless = MemoryStore(world["ids"], world["store"]._vectors)

        with pytest.raises(CapabilityMissing, match="can_read_text"):
            Cascade(textless, world["bridge"], world["embedder"])


class TestWhatItReports:
    """A feature whose main risk is its cost has to report that cost."""

    def test_the_stages_account_for_the_whole_query(self, world: dict[str, Any]) -> None:
        search = cascade(world)
        search.search(world["new"][0])
        stats = search.stats

        assert stats.queries == 1
        assert stats.bridge_seconds > 0
        assert stats.search_seconds > 0
        assert stats.embed_seconds > 0
        # The embedder is inside the rerank stage, not beside it.
        assert stats.embed_seconds <= stats.rerank_seconds
        assert stats.seconds == pytest.approx(
            stats.bridge_seconds + stats.search_seconds + stats.rerank_seconds
        )
        assert stats.per_query_seconds == pytest.approx(stats.seconds)

    def test_every_miss_is_either_embedded_or_kept(self, world: dict[str, Any]) -> None:
        """The invariant that makes the counters readable as one story."""
        texts = ["" if i % 3 == 0 else text for i, text in enumerate(world["texts"])]
        world["store"] = MemoryStore(world["ids"], world["store"]._vectors, texts)

        search = cascade(world)
        for i in range(5):
            search.search(world["new"][i])
        stats = search.stats

        assert stats.documents_embedded + stats.kept_bridged == stats.cache_misses
        assert stats.cache_hits + stats.cache_misses == stats.candidates

    def test_a_hit_rate_over_no_queries_says_nothing(self) -> None:
        """`nan` rather than 0.0, which would read as a cache that is not
        working rather than one that has not been asked anything."""
        assert math.isnan(CascadeStats().hit_rate)
        assert math.isnan(CascadeStats().per_query_seconds)

    def test_reset_starts_a_fresh_window(self, world: dict[str, Any]) -> None:
        search = cascade(world)
        search.search(world["new"][0])
        search.stats.reset()

        assert search.stats.queries == 0
        assert math.isnan(search.stats.hit_rate)

    def test_the_summary_serialises(self, world: dict[str, Any]) -> None:
        search = cascade(world, candidates=20)
        search.search(world["new"][0])
        search.search(world["new"][1])
        described = search.describe()

        for key in ("candidate_depth", "cache", "new_model", "cache_hit_rate", "embed_ms"):
            assert key in described
        assert described["cache"] == "MemoryVectorCache"
        # The configured depth and the running total are two different numbers
        # and must not share a name — one would silently replace the other.
        assert described["candidate_depth"] == 20
        assert described["candidates"] == 40


class TestTheHotPath:
    def test_it_does_not_import_torch(self) -> None:
        """`serve` may not, and the rerank does not change that: the embedder
        may load torch in its own process, but nothing in this module reaches
        it. A subprocess, because another test here may already have imported
        it — which would make this pass for the wrong reason."""
        code = (
            "import sys; import rebasis.serve.cascade; "
            "assert 'torch' not in sys.modules, 'two-stage serving imported torch'"
        )
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
