"""A rejected batch is retried, then halved, rather than failed whole.

Two mechanisms, and they answer different failures.

**Retry** is for a store that refused a write it would take a moment later — a
node rebalancing, a connection reset. `StoreWriteFailed` declares itself
transient, which is what makes it eligible, and `retry_transient` was written for
exactly this and called from nowhere until now.

**Splitting** is for the failure retrying cannot fix: one record the store will
never take. Before this, a rejected batch was marked `FAILED` whole, so a single
oversized payload cost its two hundred and fifty-five neighbours a place in the
failed list and a second pass on the next `resume`. Nothing was lost — the queue
is the checkpoint — but the operator had 256 records to look at instead of one.

The last two tests keep the cost honest. Splitting is bounded, because a store
that is simply unreachable fails every half and splitting all the way down costs
511 writes to learn what the first one already said. And retrying happens once
per batch rather than once per split: measured, retrying inside the split took
23 seconds to isolate one record from a batch of sixteen, nearly all of it
backing off from a refusal the first three attempts had already settled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from rebasis.core import ProcrustesAdapter, l2_normalize
from rebasis.errors import StoreWriteFailed
from rebasis.manifest import ManifestDB, manifest_path
from rebasis.migrate import JobState, MigrationEngine
from rebasis.migrate.engine import BISECT_MAX_DEPTH
from rebasis.observability.retry import MAX_ATTEMPTS
from rebasis.store import MemoryStore

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

pytestmark = pytest.mark.unit

DIM = 8
N = 32
BATCH = 16


class Rejecting(MemoryStore):
    """A store that refuses any write containing one of ``poison``.

    Models the real shape of the problem rather than a flaky connection: the
    record is not one the store will ever take, so retrying does nothing and only
    separating it from its neighbours helps.
    """

    def __init__(self, *args: Any, poison: Iterable[str], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.poison = set(poison)
        self.attempts: list[list[str]] = []

    def upsert_vectors(self, ids: list[str], vectors: Any) -> None:
        self.attempts.append(list(ids))
        if self.poison & set(ids):
            raise StoreWriteFailed(
                "This store refuses that record.",
                hint="Nothing to do; the record is the problem.",
                context={"store_backend": "rejecting"},
            )
        super().upsert_vectors(ids, vectors)


class FlakyOnce(MemoryStore):
    """Refuses the first write and takes every one after it."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.attempts = 0

    def upsert_vectors(self, ids: list[str], vectors: Any) -> None:
        self.attempts += 1
        if self.attempts == 1:
            raise StoreWriteFailed(
                "Not right now.",
                hint="Try again.",
                context={"store_backend": "flaky"},
            )
        super().upsert_vectors(ids, vectors)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(17)


def build(
    tmp_path: Path, store: MemoryStore, ids: list[str], rng: np.random.Generator
) -> MigrationEngine:
    vectors = l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))
    rotation = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype(np.float32)
    adapter = ProcrustesAdapter.fit(vectors, l2_normalize(vectors @ rotation.T))
    engine = MigrationEngine(
        db=ManifestDB(manifest_path(tmp_path / "state")),
        store=store,
        adapter=adapter,
        shadow_root=tmp_path / "shadow",
        batch_size=BATCH,
        power_aware=False,
    )
    engine.prepare(ids)
    return engine


def corpus(rng: np.random.Generator) -> tuple[list[str], Any, list[str]]:
    ids = [f"doc-{i:04d}" for i in range(N)]
    vectors = l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))
    return ids, vectors, [f"text {i}" for i in range(N)]


def failed_ids(engine: MigrationEngine) -> list[str]:
    return [record_id for record_id, _code, _attempts in engine.queue.failed_records()]


class TestRetry:
    def test_a_transient_refusal_does_not_fail_the_batch(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """`retry_transient` was written for this and called from nowhere."""
        ids, vectors, texts = corpus(rng)
        store = FlakyOnce(ids, vectors, texts)
        engine = build(tmp_path, store, ids, rng)

        result = engine.run()

        assert result.state is JobState.COMPLETED
        assert result.failed == 0
        assert engine.queue.stats().remaining == 0
        # One refusal, so at least one attempt more than there were batches.
        assert store.attempts > N // BATCH


class TestSplitting:
    def test_one_bad_record_does_not_fail_its_neighbours(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """The whole point: the records that were fine get written."""
        ids, vectors, texts = corpus(rng)
        store = Rejecting(ids, vectors, texts, poison=["doc-0003"])
        engine = build(tmp_path, store, ids, rng)

        result = engine.run()

        assert result.failed == 1, "only the poisoned record should fail"
        assert result.processed == N - 1
        assert failed_ids(engine) == ["doc-0003"]

    def test_the_failing_record_is_the_one_reported(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        ids, vectors, texts = corpus(rng)
        store = Rejecting(ids, vectors, texts, poison=["doc-0021"])
        engine = build(tmp_path, store, ids, rng)

        engine.run()

        assert failed_ids(engine) == ["doc-0021"]

    def test_two_bad_records_in_one_batch_are_both_isolated(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """Bisection is not one split; it recurses into whichever half failed."""
        ids, vectors, texts = corpus(rng)
        store = Rejecting(ids, vectors, texts, poison=["doc-0001", "doc-0014"])
        engine = build(tmp_path, store, ids, rng)

        result = engine.run()

        assert result.failed == 2
        assert result.processed == N - 2

    def test_what_was_written_is_what_is_verified(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """The read-back must check the surviving rows, not the batch as sent.

        The batch handed to the store and the batch that landed are no longer
        the same list once a split has happened, and comparing the wrong one
        would report a durability failure that did not occur.
        """
        ids, vectors, texts = corpus(rng)
        store = Rejecting(ids, vectors, texts, poison=["doc-0007"])
        engine = build(tmp_path, store, ids, rng)

        result = engine.run()

        assert result.state is JobState.COMPLETED
        assert result.failed == 1


class TestTheSplittingIsBounded:
    def test_an_unreachable_store_is_not_bisected_to_the_last_record(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """Splitting all the way down costs 511 writes to learn what the first said.

        With every record poisoned, no split can help. The bound is what stops
        the engine paying for that discovery on every batch of a long run.
        """
        ids, vectors, texts = corpus(rng)
        store = Rejecting(ids, vectors, texts, poison=ids)
        engine = build(tmp_path, store, ids, rng)

        result = engine.run()

        assert result.processed == 0
        assert result.failed == N

        first_batch = [call for call in store.attempts if set(call) <= set(ids[:BATCH])]
        # A 16-record batch fully split would reach single records; bounded, the
        # smallest group attempted is 16 / 2**BISECT_MAX_DEPTH or one record,
        # whichever is larger.
        floor = max(1, BATCH // 2**BISECT_MAX_DEPTH)
        assert min(len(call) for call in first_batch) >= floor

    def test_the_retry_happens_once_per_batch_not_once_per_split(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """Backing off at every node costs seconds to learn nothing new.

        Counted rather than timed: a timing assertion on a shared runner is
        noise. With one poisoned record in a batch of sixteen, only the first
        write of that batch may be attempted more than once — every attempt after
        the split is a single call, because the batch's own retries already
        established that waiting does not help.
        """
        ids, vectors, texts = corpus(rng)
        store = Rejecting(ids, vectors, texts, poison=["doc-0003"])
        engine = build(tmp_path, store, ids, rng)

        engine.run()

        first = [call for call in store.attempts if len(call) == BATCH and call[0] == ids[0]]
        assert len(first) <= MAX_ATTEMPTS, "the full batch is retried"

        splits = [call for call in store.attempts if len(call) < BATCH]
        assert len(splits) == len({tuple(call) for call in splits}), (
            "a sub-batch was written more than once, so a split is being retried"
        )
