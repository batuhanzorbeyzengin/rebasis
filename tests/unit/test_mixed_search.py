"""Searching an index a migration left holding two spaces.

`rebasis.migrate.spaces` makes a mixed index impossible to miss; this is the
part that makes it survivable. The property under test is not "the results are
good" — with half the corpus in each space no arrangement is as good as either
end — but that **each half is scored by the query that is right about it**.

The corpus is built so that failure is visible. Every document has one obviously
correct answer, the old and new spaces are related by a rotation the adapter
recovers exactly, and the two spaces are far enough apart that querying the
wrong half retrieves nothing. So a searcher that sends one query at both halves
scores near zero, and one that splits them correctly scores near one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from rebasis.core import ProcrustesAdapter, l2_normalize
from rebasis.core.serialization import AdapterManifest
from rebasis.manifest import ManifestDB, manifest_path
from rebasis.migrate import MigrationEngine
from rebasis.serve import Bridge, MixedSpaceSearch
from rebasis.serve.mixed import MAX_OVER_FETCH
from rebasis.store import MemoryStore

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

DIM = 32
N = 400


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(31)


@pytest.fixture
def world(tmp_path: Path, rng: np.random.Generator):  # type: ignore[no-untyped-def]
    """A store, an adapter that maps between its two spaces, and a job.

    ``old`` is what the index holds; ``new`` is the same documents under the
    model being adopted. The relationship is a rotation, so the adapter recovers
    it exactly — which is what makes a wrong-half query unambiguously wrong
    rather than merely worse.
    """
    centers = (rng.standard_normal((16, DIM)) * 3.0).astype(np.float32)
    assignment = rng.integers(0, 16, size=N)
    old = l2_normalize(centers[assignment] + rng.standard_normal((N, DIM)).astype(np.float32) * 1.1)
    rotation = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype(np.float32)
    new = l2_normalize(old @ rotation.T)

    ids = [f"doc-{i:04d}" for i in range(N)]
    store = MemoryStore(ids, old.copy(), [f"text {i}" for i in range(N)])

    # `query_to_old`: the adapter takes a new-model vector back to the index.
    adapter = ProcrustesAdapter.fit(new, old)
    bridge = Bridge(adapter, _manifest(adapter))

    engine = MigrationEngine(
        db=ManifestDB(manifest_path(tmp_path / "state")),
        store=store,
        # What `migrate` applies to the stored vectors to move them forward.
        adapter=ProcrustesAdapter.fit(old, new),
        shadow_root=tmp_path / "shadow",
        batch_size=25,
        power_aware=False,
    )
    engine.prepare(ids)
    return {
        "store": store,
        "bridge": bridge,
        "engine": engine,
        "ids": ids,
        "old": old,
        "new": new,
        "state": tmp_path / "state",
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


def searcher(world, **kwargs):  # type: ignore[no-untyped-def]
    return MixedSpaceSearch(
        world["store"],
        world["bridge"],
        job_id=world["engine"].job_id,
        state_dir=world["state"],
        **kwargs,
    )


def hit_rate(world, search, *, k: int = 5, queries: int = 60) -> float:  # type: ignore[no-untyped-def]
    """How often a document's own query retrieves it.

    Every document is its own best answer under either model, so a correct
    searcher finds it whichever half it is in. That makes one number cover both
    halves — and makes a searcher that only handles one of them visibly fail.
    """
    found = 0
    for i in range(queries):
        hits = search.search(world["new"][i], k=k)
        found += any(hit.id == world["ids"][i] for hit in hits)
    return found / queries


class TestItScoresEachHalfWithTheRightQuery:
    def test_a_half_migrated_index_still_answers(self, world) -> None:  # type: ignore[no-untyped-def]
        world["engine"].run(limit=N // 2)

        with searcher(world) as search:
            assert hit_rate(world, search) > 0.9

    @pytest.mark.parametrize("done", [0.1, 0.5, 0.9])
    def test_it_holds_across_the_whole_migration(self, world, done: float) -> None:  # type: ignore[no-untyped-def]
        """The mixture is worst in the middle and the ends are not free either:
        at 10% done the new-space search is mostly returning records that belong
        to the other half, and at 90% the bridged search is."""
        world["engine"].run(limit=int(N * done))

        with searcher(world) as search:
            assert hit_rate(world, search) > 0.9

    def test_a_single_query_against_the_whole_index_does_not(self, world) -> None:  # type: ignore[no-untyped-def]
        """The control, and the reason this class exists.

        Sending only the bridged query — which is what a user following the
        `Bridge` documentation does today — silently loses the half that has
        already moved. This asserts that the naive approach really is broken on
        this corpus, so the test above is measuring a fix rather than a corpus
        that was never hard.
        """
        world["engine"].run(limit=N // 2)
        store, bridge = world["store"], world["bridge"]

        found = 0
        for i in range(60):
            hits = store.search(bridge.to_index_space(world["new"][i]), k=5)
            found += any(hit.id == world["ids"][i] for hit in hits)

        # Roughly the un-migrated half, and nothing from the other one.
        assert found / 60 < 0.65


class TestTheEnds:
    def test_before_anything_moved_it_matches_the_plain_bridge(self, world) -> None:  # type: ignore[no-untyped-def]
        """Nothing is in the new space, so every result should come through the
        bridge — the mixed searcher must not degrade an index that is not mixed."""
        with searcher(world) as search:
            assert search.progress() == 0.0
            assert hit_rate(world, search) > 0.9

    def test_after_everything_moved_it_matches_the_new_model(self, world) -> None:  # type: ignore[no-untyped-def]
        world["engine"].run()

        with searcher(world) as search:
            assert search.progress() == pytest.approx(1.0)
            assert hit_rate(world, search) > 0.9


class TestWhatItReports:
    def test_progress_tracks_the_queue(self, world) -> None:  # type: ignore[no-untyped-def]
        world["engine"].run(limit=100)

        with searcher(world) as search:
            assert search.progress() == pytest.approx(0.25)

    def test_over_fetch_is_reported(self, world) -> None:  # type: ignore[no-untyped-def]
        """The running cost of a mixed index. Hidden, it would be a latency
        regression nobody could attribute; reported, it is an argument for
        finishing the migration."""
        world["engine"].run(limit=N // 2)

        with searcher(world) as search:
            search.search(world["new"][0], k=10)
            # Both halves hold about half the corpus, so each side looks about
            # twice as deep as it returns and the two together retrieve about
            # four hits for every one served.
            assert 3.5 < search.over_fetch < 4.5

    def test_over_fetch_is_bounded_on_a_lopsided_migration(self, world) -> None:  # type: ignore[no-untyped-def]
        """At 2.5% done the new-space side would need 40x `k` to find `k` of its
        own. The ceiling turns that into a short result rather than a slow one —
        twice `MAX_OVER_FETCH`, because it bounds each side and both are
        searched."""
        world["engine"].run(limit=10)

        with searcher(world) as search:
            search.search(world["new"][0], k=10)
            assert search.over_fetch <= 2 * MAX_OVER_FETCH


class TestItDoesNotWrite:
    def test_the_index_is_untouched(self, world) -> None:  # type: ignore[no-untyped-def]
        """A read path that wrote to the index would be the one thing worse than
        the problem it solves."""
        world["engine"].run(limit=N // 2)
        before = np.vstack([r.vector for r in world["store"].iter_records()])

        with searcher(world) as search:
            for i in range(20):
                search.search(world["new"][i], k=5)

        after = np.vstack([r.vector for r in world["store"].iter_records()])
        np.testing.assert_array_equal(before, after)
