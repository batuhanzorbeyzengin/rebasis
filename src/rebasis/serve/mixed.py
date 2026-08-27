"""Querying an index that is halfway between two models.

[`migrate`](../guides/migration.md) rewrites an index record by record, and every
supported way of stopping it short — ``--limit``, ``--priority access``, a pause
on memory or battery — leaves the collection holding **two embedding spaces at
once**. `rebasis.migrate.spaces` makes that impossible to miss. This module is
the other half: making it survivable.

The problem, stated exactly. After a partial migration:

* ``bridge.to_index_space(q)`` is correct against the records that have **not**
  moved, and meaningless against the ones that have;
* raw ``f_new(q)`` is the reverse.

There is no single query that is right about all of it, so this sends **both**
and keeps only the half each one is right about:

```
hits_old = search(bridge(q_new)) → keep only the un-migrated records
hits_new = search(q_new)         → keep only the migrated records
result   = calibrated_merge(hits_old, hits_new, k)
```

The merge is the code that has been sitting in `serve/hybrid.py` since the
design and had nothing to call it. With the isotonic calibrator from the `.rbs`
the two score distributions become comparable; without one it falls back to
reciprocal rank fusion, which throws the scores away and is *correct* — M0
measured a median KS distance of 0.924 between the two spaces, so comparing raw
scores would let one side win for reasons unrelated to relevance.

**Which records have moved is read from the manifest, not from the store.** The
alternative — writing a `rebasis_space` field onto every migrated record — would
let the backend's own filter do the work, and it would mean rebasis writing a
field into somebody's payload that was not there before. The whole store
contract is one write path that only ever replaces vectors, and buying a filter
by widening it is not a trade this makes. The manifest already knows; the queue
*is* the record of what moved.

The cost of that choice is over-fetching: each side asks for more than `k` and
discards what belongs to the other, so the depth is scaled by how far the
migration has actually got. That is measured per query rather than assumed, and
:attr:`MixedSpaceSearch.over_fetch` reports it.

This is for the window between starting a migration and finishing it. When the
job completes there is one space again and the plain `Bridge` — or no bridge at
all — is the right thing to use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from rebasis.serve.hybrid import calibrated_merge

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from rebasis.serve.bridge import Bridge
    from rebasis.store.base import VectorStore
    from rebasis.types import FloatArray, Hit

__all__ = ["MixedSpaceSearch"]

#: Ceiling on how far **one side** is asked to over-fetch.
#:
#: Without one, a migration at 1% would ask for 100x `k` from the new-space side
#: to find `k` migrated records — a query cost that scales with how *little*
#: progress has been made, which is the wrong way round. At the ceiling the
#: result is short rather than slow, and :attr:`MixedSpaceSearch.over_fetch`
#: says so.
#:
#: Per side, so the worst case a query can reach is twice this: both halves at
#: the ceiling is the state a migration passes through in the middle, where
#: neither side can be skipped.
MAX_OVER_FETCH = 8

#: Ids looked up in one manifest round trip.
_LOOKUP_CHUNK = 900

#: How long a progress reading is reused before the manifest is asked again.
#:
#: Counting the queue is ``O(job size)``: measured at 44 µs over 300 rows, 12 ms
#: over 100,000 and **251 ms over two million**. Doing that per query would put
#: a cost that scales with the corpus on a path budgeted in microseconds — the
#: same mistake :meth:`MixedSpaceSearch._moved` is written to avoid, one method
#: further down the file.
#:
#: What the reading is used for is choosing a search depth, and a migration
#: moves a few hundred records a second, so five seconds of staleness moves the
#: depth by a fraction of one result. :meth:`progress` itself never uses the
#: cache — a caller who asks is asking to be told now.
PROGRESS_TTL_SECONDS = 5.0


def _check_one_width(store: VectorStore, bridge: Bridge) -> None:
    """Refuse a mixed index whose two halves are not the same width.

    This arrangement sends the **raw** new-model query at the store, alongside
    the bridged one, and keeps the half each is right about. That only makes
    sense if a raw query fits the index at all — so the adapter has to map a
    width onto itself, ``input_dim == output_dim``, and both have to equal the
    index's.

    It is not an arbitrary restriction, and refusing here is not conservatism.
    A partial migration that changed the width would leave two vector widths in
    one collection: every store that declares ``dimension_locked`` rejects the
    second one outright, and a store that does not would hold a collection no
    single query can search. There is no version of a half-migrated index with
    two widths that this class could serve — the arrangement is impossible
    before it is unsupported.

    Checked at construction rather than at the first query, because the first
    query is a serving path and a caller who installed this at start-up should
    find out then.

    Raises:
        EmbeddingDimensionMismatch: When the three widths are not one width.
    """
    try:
        index = store.dimension()
    except Exception:  # noqa: BLE001 - an empty or unreadable index is not this check's business
        return
    if bridge.input_dim == bridge.output_dim == index:
        return

    from rebasis.errors import EmbeddingDimensionMismatch

    raise EmbeddingDimensionMismatch(
        f"A half-migrated index has to hold one width. The adapter maps "
        f"{bridge.input_dim} to {bridge.output_dim} and the index is {index}.",
        hint=(
            "This arrangement sends the unmapped new-model query at the index "
            "beside the bridged one, so the new model's width has to be the "
            "index's. Where the widths differ there is no half-migrated index "
            "to serve: the store rejects the second width, or holds a "
            "collection no single query can search. Finish the migration, or "
            "roll it back, and serve one space."
        ),
        context={"dim": index},
    )


class MixedSpaceSearch:
    """Search an index that a migration has left holding two spaces.

    Args:
        store: The collection being migrated. Read only — this never writes.
        bridge: The adapter, for the un-migrated half.
        job_id: Which migration split the index. Its queue is what says
            which records have moved.
        state_dir: Where that job's manifest lives; defaults to the same
            project-local ``.rebasis/`` everything else uses.

    Example:
        ```python
        search = MixedSpaceSearch(store, bridge, job_id="job-8f2a1c4e0b73")
        hits = search.search(new_model.encode(["how do I deploy?"])[0], k=10)
        ```
    """

    __slots__ = (
        "_bridge",
        "_db",
        "_job_id",
        "_last_over_fetch",
        "_progress",
        "_progress_at",
        "_store",
    )

    def __init__(
        self,
        store: VectorStore,
        bridge: Bridge,
        *,
        job_id: str,
        state_dir: Path | str | None = None,
    ) -> None:
        from rebasis.manifest import ManifestDB, default_state_dir, manifest_path

        self._store = store
        self._bridge = bridge
        self._job_id = job_id
        _check_one_width(store, bridge)
        directory = default_state_dir() if state_dir is None else _as_path(state_dir)
        self._db = ManifestDB(manifest_path(directory))
        self._last_over_fetch = 1.0
        self._progress = self.progress()
        self._progress_at = _now()

    @property
    def over_fetch(self) -> float:
        """Hits the last query actually retrieved, over the ``k`` it returned.

        A **measurement**, not the plan: it counts what the store handed back,
        which is what the depth cost. The two differ whenever a side is asked
        for more than the index holds — asking for 400 of a 300-record
        collection costs 300, not 400 — and reporting the request would
        overstate the bill in exactly the case a user is most likely to hit.

        It rises as the migration approaches either end: at 5% done most of what
        the new-space search returns belongs to the other half and is discarded.
        Reported rather than hidden because it is the running cost of a mixed
        index, and the cheapest way to lower it is to finish the migration.

        Bounded by twice :data:`MAX_OVER_FETCH` — that ceiling is per side, and
        both sides are searched everywhere except at the two ends, where one is
        skipped and this falls to about 1.

        Written by :meth:`search` and read here, so it describes *a* recent
        query rather than a specific one when several threads share an instance.
        """
        return self._last_over_fetch

    def progress(self) -> float:
        """Fraction of the job's records now in the new model's space.

        Reads the queue, which is ``O(job size)`` — 12 ms over 100,000 rows and
        251 ms over two million. :meth:`search` therefore uses a cached reading
        (:data:`PROGRESS_TTL_SECONDS`); this asks the manifest every time,
        because a caller who calls it is asking for the current answer.
        """
        rows = self._db.query(
            "SELECT state, COUNT(*) AS n FROM job_items WHERE job_id = ? GROUP BY state",
            (self._job_id,),
        )
        counts = {str(row["state"]): int(row["n"]) for row in rows}
        total = sum(counts.values())
        return counts.get("done", 0) / total if total else 0.0

    def search(self, vector: FloatArray, k: int = 10, **kwargs: Any) -> list[Hit]:
        """Retrieve from both halves of the index and merge them.

        Args:
            vector: The query under the **new** model. The old-space query is
                derived from it by the bridge; asking the caller for both would
                be asking them to hold a detail this exists to hide.
            k: How many results to return.
            **kwargs: Passed through to the store's ``search``.

        Returns:
            Up to ``k`` hits. Fewer only when the index genuinely holds fewer
            matching records than asked for, or when the over-fetch ceiling was
            reached on a very lopsided migration — :attr:`over_fetch` reports
            which.
        """
        done = self._cached_progress()

        # At either end one of the two searches is guaranteed to return nothing
        # usable — every hit would be discarded as belonging to the other half —
        # and a mixed searcher left installed across a whole migration sits at
        # `done == 0` before it starts and `done == 1` after it ends. Paying two
        # store round trips for a provably empty result set, in the two states
        # it spends the most time in, is not a trade worth making.
        migrated = self._store.search(vector, k=_depth(k, share=done), **kwargs) if done > 0 else []
        bridged = (
            self._store.search(
                self._bridge.to_index_space(vector), k=_depth(k, share=1.0 - done), **kwargs
            )
            if done < 1
            else []
        )

        moved = self._moved({hit.id for hit in migrated} | {hit.id for hit in bridged})
        new_side = _renumber(hit for hit in migrated if hit.id in moved)
        old_side = _renumber(hit for hit in bridged if hit.id not in moved)

        # What was retrieved, over what is returned — the actual cost, not the
        # depth that was requested.
        self._last_over_fetch = (len(migrated) + len(bridged)) / k if k else 1.0
        # The calibrator maps old-space scores onto the new-space distribution,
        # which is what makes the two comparable at all. It came from the same
        # `.rbs` as the adapter, fitted on held-out scores.
        return calibrated_merge(old_side, new_side, k=k, calibrator=self._bridge.calibrator)

    def close(self) -> None:
        """Release the manifest handle."""
        self._db.close()

    def __enter__(self) -> Self:
        """Support ``with MixedSpaceSearch(...) as search:``."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Release the manifest handle on the way out."""
        self.close()

    # ── internals ─────────────────────────────────────────────────────

    def _cached_progress(self) -> float:
        """The progress reading :meth:`search` sizes its requests from.

        Refreshed on a timer rather than per query. What it decides is a search
        depth, and a migration moves a few hundred records a second, so a
        reading five seconds old moves that depth by a fraction of one result —
        against a manifest scan that costs a quarter of a second on a large job.
        """
        now = _now()
        if now - self._progress_at >= PROGRESS_TTL_SECONDS:
            self._progress = self.progress()
            self._progress_at = now
        return self._progress

    def _moved(self, ids: set[str]) -> set[str]:
        """Which of these records the migration has already rewritten.

        Asked about the ids the two searches actually returned, rather than
        loading the job's whole done-list: on a five-million-record migration
        that list is hundreds of megabytes of Python strings, and the question
        being asked is only ever about a few hundred of them.
        """
        if not ids:
            return set()
        ordered = sorted(ids)
        moved: set[str] = set()
        for start in range(0, len(ordered), _LOOKUP_CHUNK):
            chunk = ordered[start : start + _LOOKUP_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            rows = self._db.query(
                f"SELECT record_id FROM job_items "  # noqa: S608 - placeholders only
                f"WHERE job_id = ? AND state = 'done' AND record_id IN ({placeholders})",
                (self._job_id, *chunk),
            )
            moved.update(str(row["record_id"]) for row in rows)
        return moved


def _depth(k: int, *, share: float) -> int:
    """How deep one side has to look to expect ``k`` of its own records.

    A side holding ``share`` of the corpus returns roughly ``share`` of what it
    is asked for, so it is asked for ``k / share`` — bounded, because as
    ``share`` approaches zero that number does not.
    """
    if share <= 0:
        return k
    return min(int(k / share) + 1, k * MAX_OVER_FETCH)


def _renumber(hits: Iterable[Hit]) -> list[Hit]:
    """Re-rank a filtered result from zero.

    RRF reads ``hit.rank``, and after discarding the other half's records the
    surviving ranks have gaps in them. Left alone, a side whose first three hits
    belonged to the other half would have its true best result scored as if it
    had come fourth.
    """
    from rebasis.types import Hit as _Hit

    return [_Hit(id=hit.id, score=hit.score, rank=rank) for rank, hit in enumerate(hits)]


def _now() -> float:
    """A monotonic clock, for the progress cache's timer."""
    import time

    return time.monotonic()


def _as_path(value: Path | str) -> Path:
    from pathlib import Path as _Path

    return _Path(value)
