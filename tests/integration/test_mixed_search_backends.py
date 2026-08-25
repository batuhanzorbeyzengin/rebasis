"""Searching a half-migrated index through a real client library.

The unit suite proves the split-and-merge on a store whose `search` is a matrix
multiply. What it cannot prove is that the logic survives a client library.
`MixedSpaceSearch` decides which half of the index a hit belongs to by matching
the id the backend handed back against the ids the migration queue recorded, and
it sizes each side's request from how far the migration has got. So it rests on
two things no unit test touches: that a backend's ids **round-trip** through
`search`, and that a backend asked for `n` neighbours really returns `n`. Both
fail silently. Ids that do not round-trip put every hit on the un-migrated side
and lose the half that has already moved; a `k` that quietly comes back short
truncates the merge while `over_fetch` still reports a deep look-up.

Every backend also scores differently — Chroma and sqlite-vec return distances,
LanceDB returns one on a third scale, FAISS an inner product, Qdrant a
similarity. With no calibrator the merge is rank fusion, which is precisely why
those scales do not have to agree; the thing that must survive is the *order*,
and that is asserted here rather than assumed.

The corpus is clustered rather than uniform, and the two spaces are one exact
rotation apart, for the reason the unit suite gives: every document is then its
own unambiguous best answer under either model, so a single hit rate covers both
halves and a searcher that serves only one of them fails visibly. Uniform
vectors in 32 dimensions leave the top few results near-ties decided by float
noise, which would make this a test of tie-breaking instead.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from rebasis.core import ProcrustesAdapter, l2_normalize
from rebasis.core.serialization import AdapterManifest
from rebasis.manifest import ManifestDB, manifest_path
from rebasis.migrate import MigrationEngine
from rebasis.serve import Bridge, MixedSpaceSearch
from rebasis.serve.mixed import MAX_OVER_FETCH
from rebasis.store import open_store

pytestmark = [pytest.mark.integration]

DIM = 32
N = 300

#: Queries drawn from each half. Enough that a half served by the wrong query
#: cannot pass by luck, and small enough that ten cases per backend stay inside
#: the integration budget.
QUERIES = 25

#: A `k` whose over-fetch ceiling (``k * MAX_OVER_FETCH``) exceeds the corpus, so
#: that one side genuinely asks a backend for more neighbours than it holds.
DEEP_K = 40

#: Written out rather than imported from the conftest that builds them:
#: `@pytest.fixture(params=...)` runs at collection, and a conftest is not
#: importable by name. Every one of these declares it can write, so none is
#: skipped — a backend that could not would have nothing to half-migrate.
BACKENDS = ("chroma", "faiss", "lancedb", "qdrant", "sqlite-vec")


def closing(store: object) -> None:
    """Release a backend's handle if it holds one.

    Qdrant's local mode takes an exclusive lock on its storage folder and a
    second client raises rather than waiting, so a handle left open makes the
    next read fail. Most backends have nothing to release.
    """
    close = getattr(store, "close", None)
    if callable(close):
        with contextlib.suppress(Exception):
            close()


@pytest.fixture(params=BACKENDS, ids=lambda n: n)
def world(request, tmp_path, rng, make_store):  # type: ignore[no-untyped-def]
    """A real store holding the old space, and the adapter between the two.

    ``old`` is what the index was built with; ``new`` is the same documents
    under the model being adopted. The relationship is a rotation, so Procrustes
    recovers it exactly — which is what makes a wrong-half query unambiguously
    wrong rather than merely worse, and therefore what makes a backend that
    quietly serves one half fail loudly here.
    """
    centers = (rng.standard_normal((16, DIM)) * 3.0).astype(np.float32)
    assignment = rng.integers(0, 16, size=N)
    old = l2_normalize(centers[assignment] + rng.standard_normal((N, DIM)).astype(np.float32) * 1.1)
    rotation = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype(np.float32)
    new = l2_normalize(old @ rotation.T)

    ids = [f"doc-{i:04d}" for i in range(N)]
    texts = [f"text of document {i}" for i in range(N)]
    uri = make_store(request.param, tmp_path, ids, old, texts)

    # `query_to_old`: the adapter takes a new-model query back to the index.
    adapter = ProcrustesAdapter.fit(new, old)
    return {
        "backend": request.param,
        "uri": uri,
        "ids": ids,
        "old": old,
        "new": new,
        "bridge": Bridge(adapter, _manifest(adapter)),
        "state": tmp_path / "state",
        "shadow": tmp_path / "shadow",
        "job_id": "",
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


def migrate(world, *, limit: int) -> None:  # type: ignore[no-untyped-def]
    """Move ``limit`` of the records into the new model's space, and stop there.

    ``power_aware=False`` because a pause on battery would leave a different
    number of records migrated on every run, and every assertion below is about
    a known split. ``store_uri`` is passed so the engine's own durability check
    runs: a backend that accepted the writes and forgot them would otherwise
    reach the searcher looking like an un-migrated index.
    """
    store = open_store(world["uri"])
    engine = MigrationEngine(
        db=ManifestDB(manifest_path(world["state"])),
        store=store,
        # What `migrate` applies to the stored vectors to move them forward.
        adapter=ProcrustesAdapter.fit(world["old"], world["new"]),
        shadow_root=world["shadow"],
        batch_size=64,
        power_aware=False,
        store_uri=world["uri"],
    )
    engine.prepare(list(world["ids"]))
    engine.run(limit=limit)
    world["job_id"] = engine.job_id
    closing(engine.store)
    engine.db.close()


def halves(world) -> tuple[list[int], list[int]]:  # type: ignore[no-untyped-def]
    """Which documents actually moved, read from the index rather than the queue.

    `MixedSpaceSearch` takes the split from the manifest. Taking the *expected*
    split from there too would have both agree about a migration neither of them
    checked happened — a job that marked records done without the backend
    keeping the writes would pass. The index is the ground truth: every record
    holds either its old vector or its new one, and the two are a rotation apart.
    """
    store = open_store(world["uri"])
    try:
        stored = {record.id: record.vector for record in store.iter_records(with_text=False)}
    finally:
        closing(store)

    is_new = [
        float(stored[record_id] @ world["new"][i]) > float(stored[record_id] @ world["old"][i])
        for i, record_id in enumerate(world["ids"])
    ]
    return (
        [i for i, moved in enumerate(is_new) if moved],
        [i for i, moved in enumerate(is_new) if not moved],
    )


@contextlib.contextmanager
def searching(world, store=None):  # type: ignore[no-untyped-def]
    """A searcher over a connection opened after the migration wrote.

    Fresh rather than reused: Chroma caches its client per path, so a handle
    opened before the migration answers from the state it already had, and the
    whole index would look un-migrated.
    """
    own = store is None
    handle = open_store(world["uri"]) if own else store
    try:
        with MixedSpaceSearch(
            handle,
            world["bridge"],
            job_id=world["job_id"],
            state_dir=world["state"],
        ) as search:
            yield search
    finally:
        if own:
            closing(handle)


def hit_rate(world, search, indices, *, k: int = 5) -> float:  # type: ignore[no-untyped-def]
    """How often a document's own query retrieves it.

    Every document is its own best answer under either model, so a correct
    searcher finds it whichever half it is in. Handed one half's indices at a
    time, this says which half a backend is failing — an id convention that does
    not round-trip loses the migrated half specifically, and would be invisible
    in a number averaged over both.
    """
    found = 0
    for i in indices:
        hits = search.search(world["new"][i], k=k)
        found += any(hit.id == world["ids"][i] for hit in hits)
    return found / len(indices)


class _Recorder:
    """A store that answers normally and remembers what it was asked for.

    `over_fetch` counts the hits that came back, so it already reflects a
    backend that served less than it was asked for. What it cannot show is
    *which* side was short, or whether the shortfall was the ceiling doing its
    job or the client library quietly capping a request. Watching the calls
    separates the two.
    """

    def __init__(self, store: object) -> None:
        self._store = store
        self.calls: list[tuple[int, int]] = []

    def search(self, vector, k, **kwargs):  # type: ignore[no-untyped-def]
        """Answer the query, then record the depth asked for against the depth given."""
        hits = self._store.search(vector, k=k, **kwargs)  # type: ignore[attr-defined]
        self.calls.append((k, len(hits)))
        return hits

    def __getattr__(self, name: str) -> object:
        return getattr(self._store, name)


def _vectors(store: object) -> dict[str, np.ndarray]:
    return {r.id: r.vector for r in store.iter_records(with_text=False)}  # type: ignore[attr-defined]


class TestEachHalfIsScoredByTheQueryThatIsRightAboutIt:
    def test_a_migrated_document_is_still_found_by_its_own_query(self, world) -> None:  # type: ignore[no-untyped-def]
        """The half a client library can lose without saying so.

        A migrated record is reachable only through the raw new-model query
        *and* through its id being recognised in the queue. If the backend
        returns ids in a shape the manifest never saw — a rowid, a point number,
        a FAISS label — every hit lands on the un-migrated side and this half
        disappears while the searcher still returns a full, plausible result.
        """
        migrate(world, limit=N // 2)
        moved, _ = halves(world)

        with searching(world) as search:
            rate = hit_rate(world, search, moved[:QUERIES])

        assert rate > 0.9, f"{world['backend']} lost the migrated half: {rate:.2f}"

    def test_an_untouched_document_is_still_found_by_its_own_query(self, world) -> None:  # type: ignore[no-untyped-def]
        """The other half, and the one a mixed index must not regress.

        These records are exactly where they were before the migration started,
        so a bridged query is right about them and always was. Losing them means
        the over-fetch depth came back short, or the filter discarded records it
        had no queue entry for rather than records it had a *done* entry for.
        """
        migrate(world, limit=N // 2)
        _, stayed = halves(world)

        with searching(world) as search:
            rate = hit_rate(world, search, stayed[:QUERIES])

        assert rate > 0.9, f"{world['backend']} lost the un-migrated half: {rate:.2f}"

    def test_a_single_bridged_query_finds_only_the_half_it_is_right_about(self, world) -> None:  # type: ignore[no-untyped-def]
        """The control, and the reason the two tests above mean anything.

        Sending only the bridged query is what a user following the `Bridge`
        documentation does today. Asserting it fails *on this backend's own
        metric* is what rules out the alternative explanation for a good hit
        rate: that the two spaces are not far enough apart to tell, which would
        differ between a cosine backend and an L2 one.
        """
        migrate(world, limit=N // 2)
        moved, stayed = halves(world)
        bridge = world["bridge"]

        store = open_store(world["uri"])
        try:
            found = 0
            probes = moved[:QUERIES] + stayed[:QUERIES]
            for i in probes:
                hits = store.search(bridge.to_index_space(world["new"][i]), k=5)
                found += any(hit.id == world["ids"][i] for hit in hits)
        finally:
            closing(store)

        # Roughly the un-migrated half, and nothing from the other one.
        assert found / len(probes) < 0.65, f"{world['backend']}: the two spaces are not separable"

    @pytest.mark.parametrize("done", [0.1, 0.9])
    def test_it_holds_at_the_lopsided_ends_of_a_migration(self, world, done: float) -> None:  # type: ignore[no-untyped-def]
        """The ends are where the over-fetch depth stops being a formality.

        At 10% done the new-space side has to look far past `k` to find `k`
        records of its own, and at 90% the bridged side does. That is the regime
        where a backend that clamps or truncates a large request turns into
        missing results, so both halves are checked at both ends.
        """
        migrate(world, limit=int(N * done))
        moved, stayed = halves(world)

        with searching(world) as search:
            moved_rate = hit_rate(world, search, moved[:QUERIES])
            stayed_rate = hit_rate(world, search, stayed[:QUERIES])

        assert moved_rate > 0.9, f"{world['backend']} at {done:.0%}: migrated {moved_rate:.2f}"
        assert stayed_rate > 0.9, f"{world['backend']} at {done:.0%}: untouched {stayed_rate:.2f}"


class TestTheDepthItAsksFor:
    def test_the_backend_returns_the_depth_it_was_asked_for(self, world) -> None:  # type: ignore[no-untyped-def]
        """Over-fetching only works if the fetch happened.

        The whole design trades query depth for not writing a space marker into
        anybody's payload, and it pays that price on the assumption that asking
        for 2·k returns 2·k. A backend that silently caps a request would make
        the searcher discard half of what it planned for and hand back a short
        result — visible in `over_fetch`, which counts what arrived, but not
        attributable to a cap rather than to the ceiling without watching the
        calls.
        """
        migrate(world, limit=N // 2)

        store = open_store(world["uri"])
        recorder = _Recorder(store)
        try:
            with searching(world, recorder) as search:
                search.search(world["new"][0], k=10)
        finally:
            closing(store)

        # One query per space, and no more: a third would mean the searcher is
        # paying for a round trip the merge never reads.
        assert len(recorder.calls) == 2
        for requested, returned in recorder.calls:
            assert returned == requested, (
                f"{world['backend']} returned {returned} of the {requested} asked for"
            )

    def test_asking_deeper_than_the_index_holds_comes_back_short_not_broken(self, world) -> None:  # type: ignore[no-untyped-def]
        """A small collection early in a migration asks for more than exists.

        At 10% done with `k` this size the ceiling still puts the new-space
        request above the record count, and the backends disagree about what
        that means: FAISS pads the result with -1 labels, Chroma and LanceDB
        clamp, sqlite-vec takes `k` as a bound in SQL. An exception here would
        make a mixed index unqueryable on exactly the collections small enough
        to migrate in one sitting.
        """
        migrate(world, limit=N // 10)

        store = open_store(world["uri"])
        recorder = _Recorder(store)
        try:
            with searching(world, recorder) as search:
                hits = search.search(world["new"][0], k=DEEP_K)
        finally:
            closing(store)

        assert max(requested for requested, _ in recorder.calls) > N, (
            "the corpus outgrew DEEP_K and this no longer over-asks the index"
        )
        for requested, returned in recorder.calls:
            assert returned == min(requested, N), (
                f"{world['backend']} returned {returned} of the {requested} asked for"
            )
        assert 0 < len(hits) <= DEEP_K

    def test_over_fetch_is_reported(self, world) -> None:  # type: ignore[no-untyped-def]
        """The running cost of a mixed index, measured against a real backend.

        Hidden, it would be a latency regression nobody could attribute;
        reported, it is an argument for finishing the migration. It is checked
        here as well as in the unit suite because the figure is only honest if
        the backend served the depth it describes — the assertion above is what
        makes this one worth reading.
        """
        migrate(world, limit=N // 2)

        with searching(world) as search:
            search.search(world["new"][0], k=10)
            # Both halves hold about half the corpus, so each side looks about
            # twice as deep as it returns and the two together retrieve about
            # four hits for every one served.
            assert 3.5 < search.over_fetch < 4.5

    def test_over_fetch_is_bounded_on_a_lopsided_migration(self, world) -> None:  # type: ignore[no-untyped-def]
        """At 3% done the new-space side would need 30x `k` to find `k` of its
        own, and every one of those neighbours costs the backend real work. The
        ceiling turns that into a short result rather than a slow one — twice
        `MAX_OVER_FETCH`, because it bounds each side and both are searched."""
        migrate(world, limit=10)

        with searching(world) as search:
            search.search(world["new"][0], k=10)
            assert search.over_fetch <= 2 * MAX_OVER_FETCH


class TestItDoesNotWrite:
    def test_the_index_is_untouched(self, world) -> None:  # type: ignore[no-untyped-def]
        """A read path that wrote to the index would be the one thing worse than
        the problem it solves.

        The control is a plain `search`, not a second read. On Qdrant's local
        mode the first cosine query normalises the stored vectors in place, so a
        collection read after any search differs from the same collection read
        before one. That is the backend's behaviour and not the searcher's, so
        the question worth asking is whether two queries per call move anything
        a single bare query would not.
        """
        migrate(world, limit=N // 2)

        store = open_store(world["uri"])
        try:
            before = _vectors(store)
            store.search(before[world["ids"][0]], k=5)
            control = _vectors(store)
            with searching(world, store) as search:
                for i in range(15):
                    search.search(world["new"][i], k=5)
            after = _vectors(store)
        finally:
            closing(store)

        assert before.keys() == after.keys()
        for record_id, vector in control.items():
            np.testing.assert_array_equal(after[record_id], vector)
