"""The calibrated half of a mixed-index search, which nothing exercised.

`MixedSpaceSearch` merges its two halves with `serve.calibrated_merge`, and that
function has two branches. Without a calibrator it falls back to reciprocal rank
fusion; **with** one it maps old-space scores onto the new-space distribution and
merges by score. Every test the class had ran the first branch — the fixture
built a `Bridge` with no calibrator — so the branch that runs whenever somebody
loads a real `.rbs` was the untested one.

That is not a hypothetical gap. Two defects were found in `calibrated_merge` by
reading it: it sorted on `(-score, id)`, and an isotonic calibrator produces far
fewer distinct levels than it has inputs, so ties were the common case and every
one of them went to whichever document id sorted first. At the endpoints of a
migration — where the index holds one space and there is a single right answer —
the merge reproduced the store's own ranking on 4% to 16% of queries against
rank fusion's 100%.

So these are the properties that had nothing holding them:

* the calibrated path **runs** on a half-migrated index and returns sensible
  results, rather than raising or emptying out;
* at both endpoints it reduces to what the store returned, which is the
  regression the tie-break fix was for;
* the merge orders by calibrated score rather than by id.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from rebasis.core import ProcrustesAdapter, l2_normalize
from rebasis.core.calibration import ScoreCalibrator
from rebasis.core.serialization import AdapterManifest
from rebasis.manifest import ManifestDB, manifest_path
from rebasis.migrate import MigrationEngine
from rebasis.serve import Bridge, MixedSpaceSearch
from rebasis.store import MemoryStore

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

DIM = 32
N = 300
K = 5


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


@pytest.fixture
def world(tmp_path: Path):  # type: ignore[no-untyped-def]
    """A store mid-migration, and a bridge that carries a real calibrator.

    The calibrator is fitted the way `probe` fits one — on two score
    distributions rather than on paired results — using the bridged and oracle
    similarities of the same held-out queries. Fitting it from the data rather
    than constructing one by hand is what makes the isotonic step function have
    the shape the real thing has: far fewer levels than inputs, which is where
    the ties come from.
    """
    rng = np.random.default_rng(19)
    centers = (rng.standard_normal((12, DIM)) * 3.0).astype(np.float32)
    assignment = rng.integers(0, 12, size=N)
    old = l2_normalize(centers[assignment] + rng.standard_normal((N, DIM)).astype(np.float32) * 1.1)
    rotation = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype(np.float32)
    new = l2_normalize(old @ rotation.T)

    ids = [f"doc-{i:04d}" for i in range(N)]
    store = MemoryStore(ids, old.copy(), [f"text {i}" for i in range(N)])
    to_old = ProcrustesAdapter.fit(new, old)

    held = rng.permutation(N)[:80]
    bridged = l2_normalize(to_old.apply(new[held]), copy=False)
    calibrator = ScoreCalibrator.fit(
        np.einsum("ij,ij->i", bridged, old[held]),
        np.einsum("ij,ij->i", new[held], new[held]),
    )

    engine = MigrationEngine(
        db=ManifestDB(manifest_path(tmp_path / "state")),
        store=store,
        adapter=ProcrustesAdapter.fit(old, new),
        shadow_root=tmp_path / "shadow",
        batch_size=25,
        power_aware=False,
    )
    engine.prepare(ids)
    return {
        "store": store,
        "bridge": Bridge(to_old, _manifest(to_old), calibrator=calibrator),
        "plain": Bridge(to_old, _manifest(to_old)),
        "engine": engine,
        "ids": ids,
        "new": new,
        "state": tmp_path / "state",
    }


def searcher(world, *, calibrated: bool = True):  # type: ignore[no-untyped-def]
    return MixedSpaceSearch(
        world["store"],
        world["bridge"] if calibrated else world["plain"],
        job_id=world["engine"].job_id,
        state_dir=world["state"],
    )


def hit_rate(world, search, *, queries: int = 60) -> float:  # type: ignore[no-untyped-def]
    """How often a document's own query retrieves it, across both halves."""
    picks = np.linspace(0, N - 1, queries, dtype=int)
    found = 0
    for position in picks:
        hits = search.search(world["new"][position], k=K)
        found += int(world["ids"][position] in {hit.id for hit in hits})
    return found / len(picks)


class TestTheCalibratorIsActuallyInPlay:
    def test_the_bridge_carries_one(self, world) -> None:  # type: ignore[no-untyped-def]
        """Guards the fixture. Every earlier test of this class built a bridge
        without a calibrator, which is how the branch went untested — a fixture
        that silently lost it again would do the same."""
        assert world["bridge"].calibrator is not None
        assert world["plain"].calibrator is None

    def test_it_collapses_the_scores_onto_few_levels(self, world) -> None:  # type: ignore[no-untyped-def]
        """The property the tie-break bug rested on: isotonic regression maps
        many inputs onto few outputs, so ties are the common case rather than an
        edge one."""
        calibrator = world["bridge"].calibrator
        scores = np.linspace(0.0, 1.0, 40, dtype=np.float32)

        levels = len(set(np.round(calibrator.transform(scores), 6).tolist()))

        assert levels < len(scores)


class TestItServesAHalfMigratedIndex:
    def test_the_calibrated_path_finds_documents_in_both_halves(self, world) -> None:  # type: ignore[no-untyped-def]
        world["engine"].run(limit=N // 2)

        assert hit_rate(world, searcher(world)) > 0.8

    def test_it_is_no_worse_than_rank_fusion_here(self, world) -> None:  # type: ignore[no-untyped-def]
        """Not a claim that calibration wins — `docs/mixed-space-fusion.md`
        measures that on real corpora and the answer is nuanced. What this holds
        is that turning it on does not break the arrangement, which is what an
        untested branch is most likely to have done."""
        world["engine"].run(limit=N // 2)

        calibrated = hit_rate(world, searcher(world))
        fused = hit_rate(world, searcher(world, calibrated=False))

        assert calibrated >= fused - 0.1, (calibrated, fused)

    def test_it_returns_k_hits(self, world) -> None:  # type: ignore[no-untyped-def]
        world["engine"].run(limit=N // 2)

        assert len(searcher(world).search(world["new"][0], k=K)) == K

    def test_the_hits_are_ordered_by_score(self, world) -> None:  # type: ignore[no-untyped-def]
        """The merge sorts on the calibrated score; a result that came back in
        the order the two sides were concatenated would still look plausible."""
        world["engine"].run(limit=N // 2)

        hits = searcher(world).search(world["new"][7], k=K)

        assert [hit.score for hit in hits] == sorted((h.score for h in hits), reverse=True)
        assert [hit.rank for hit in hits] == list(range(len(hits)))


class TestTheEndpointsReduceToTheStore:
    """The regression the tie-break fix was for.

    At 0% and 100% migrated the index holds one space and there is a single
    right answer — what the store returned. Before `rank` joined the sort key
    this merge reproduced it on 4% to 16% of queries.
    """

    def test_before_the_migration_starts(self, world) -> None:  # type: ignore[no-untyped-def]
        search = searcher(world)
        query = world["new"][3]
        expected = [
            hit.id for hit in world["store"].search(world["bridge"].to_index_space(query), k=K)
        ]

        assert [hit.id for hit in search.search(query, k=K)] == expected

    def test_after_it_finishes(self, world) -> None:  # type: ignore[no-untyped-def]
        world["engine"].run()
        search = searcher(world)
        query = world["new"][3]
        expected = [hit.id for hit in world["store"].search(query, k=K)]

        assert [hit.id for hit in search.search(query, k=K)] == expected

    def test_it_holds_across_many_queries(self, world) -> None:  # type: ignore[no-untyped-def]
        """One query agreeing could be luck; the measured failure rate was
        84% to 96% of queries disagreeing."""
        world["engine"].run()
        search = searcher(world)

        agreed = 0
        picks = np.linspace(0, N - 1, 40, dtype=int)
        for position in picks:
            query = world["new"][position]
            expected = [hit.id for hit in world["store"].search(query, k=K)]
            agreed += int([hit.id for hit in search.search(query, k=K)] == expected)

        assert agreed == len(picks)
