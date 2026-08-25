"""Whether an index currently holds vectors from one space or two.

``migrate`` rewrites an index record by record, and every way of stopping it
short is a documented feature: ``--limit`` migrates a slice, ``--priority
access`` migrates the records people read first, memory pressure and a low
battery both pause a checkpointed job. Each of those leaves the collection
holding **two embedding spaces at once** — some vectors the old model wrote,
some the new one — and no query is correct against both:

* ``bridge.to_index_space(q)`` is meaningless against the records that already
  moved, because they are no longer in the space it maps into;
* raw ``f_new(q)`` is meaningless against the records that have not.

Whichever a caller sends, part of the corpus is scored against the wrong
geometry, and on a graph index the traversal itself runs over the mixture. There
is no error, no exception and nothing in the result to say so — the number of
records is right, the text is right, and the ranking is wrong.

That is the failure this module exists to make impossible to miss. It answers
one question — *is this index mixed right now, and by how much* — from the
manifest alone, without opening the store, so `status` can ask it while a
migration is in flight and `migrate` can say it on the way out.

The tool's own rule applies here more than anywhere: partial support beats none,
**silent** partial support does not. A user who runs ``--limit 5000`` and keeps
querying is getting quietly wrong answers today, and the only reason it has not
been reported is that quietly wrong answers do not get reported.

Saying so is the floor, not the ceiling.
:class:`~rebasis.serve.mixed.MixedSpaceSearch` is the other half — it sends both
queries and keeps only the half each is right about, using this module's own
answer to decide which is which.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rebasis.migrate.queue import JobQueue
from rebasis.migrate.states import JobState

if TYPE_CHECKING:
    from rebasis.manifest import ManifestDB

__all__ = ["MixedSpace", "mixed_spaces", "mixed_spaces_for"]

#: Job states in which a partially finished queue leaves the index mixed.
#:
#: ``rolled_back`` is excluded because the shadow copy put every vector back;
#: ``completed`` because nothing is left to move. ``pending`` stays in — a job
#: that was interrupted mid-batch can hold ``done`` records with its state never
#: having advanced past the row it was inserted with.
_UNFINISHED = frozenset({JobState.PENDING, JobState.RUNNING, JobState.PAUSED, JobState.FAILED})


@dataclass(frozen=True, slots=True)
class MixedSpace:
    """An index that currently holds vectors from two embedding spaces."""

    job_id: str
    state: str
    store_uri: str
    store_backend: str
    adapter_type: str
    #: Records rewritten with the new model's geometry.
    migrated: int
    #: Records still carrying the old model's geometry, counting the ones that
    #: failed. A failed record is as un-migrated as a pending one, and leaving
    #: it out of this figure would understate the mixture.
    unmigrated: int
    #: Whether the shadow copy can still undo this.
    reversible: bool

    @property
    def total(self) -> int:
        """Records the job was given."""
        return self.migrated + self.unmigrated

    @property
    def fraction(self) -> float:
        """Share of the job's records now in the new space."""
        return self.migrated / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, object]:
        """Serialisable form, for ``status --json``."""
        return {
            "job_id": self.job_id,
            "state": self.state,
            "store_uri": self.store_uri,
            "store_backend": self.store_backend,
            "migrated": self.migrated,
            "unmigrated": self.unmigrated,
            "fraction": round(self.fraction, 4),
            "reversible": self.reversible,
        }

    def explain(self) -> str:
        """One paragraph a user can act on.

        Plain text with no markup: it is printed by the CLI, written into a
        report and returned by the API, and only one of those three renders
        Rich tags.
        """
        return (
            f"{self.store_uri or self.store_backend or 'this index'} holds two embedding "
            f"spaces: {self.migrated:,} of {self.total:,} records ({self.fraction:.0%}) "
            f"have the new model's vectors and {self.unmigrated:,} still have the old "
            f"model's. Search results are wrong either way until job {self.job_id} "
            f"finishes — a bridged query mis-scores the migrated records, an unbridged "
            f"one mis-scores the rest."
        )

    def next_steps(self) -> tuple[str, ...]:
        """What can be done about it, in the order it should be considered.

        Two ways to *resolve* the mixture and one way to *live with* it. The
        third is listed last on purpose: searching a mixed index correctly costs
        two queries and an over-fetch on every request, which is a reasonable
        price for a migration window and a poor one for a steady state.
        """
        steps = [f"rebasis migrate --resume {self.job_id}   (finish it)"]
        if self.reversible:
            steps.append(f"rebasis rollback {self.job_id}   (put the index back)")
        else:
            steps.append(
                "This job kept no shadow copy, so finishing it is the only way back "
                "to a single space."
            )
        steps.append(
            f"MixedSpaceSearch(store, bridge, job_id={self.job_id!r})   "
            "(search it correctly meanwhile)"
        )
        return tuple(steps)


def mixed_spaces(db: ManifestDB) -> list[MixedSpace]:
    """Every index left holding two spaces by an unfinished job.

    Reads the manifest and nothing else — no store is opened, no lock is taken —
    so this is safe to call while a migration is running, which is exactly when
    the answer is wanted.
    """
    rows = db.query(
        "SELECT * FROM jobs WHERE state IN (?, ?, ?, ?) ORDER BY updated_utc DESC",
        tuple(str(state) for state in sorted(_UNFINISHED)),
    )

    from rebasis.manifest import JobRow

    found: list[MixedSpace] = []
    for raw in rows:
        job = JobRow.from_row(raw)
        stats = JobQueue(db, job.job_id).stats()
        # `skipped` is neither: those records had no vector to migrate, so they
        # are not in either space and counting them as unmigrated would report
        # a mixture that is not there.
        unmigrated = stats.pending + stats.shadowed + stats.failed
        if stats.done == 0 or unmigrated == 0:
            continue
        found.append(
            MixedSpace(
                job_id=job.job_id,
                state=job.state,
                store_uri=job.store_uri,
                store_backend=job.store_backend,
                adapter_type=job.adapter_type,
                migrated=stats.done,
                unmigrated=unmigrated,
                reversible=job.reversible,
            )
        )
    return found


def mixed_spaces_for(db: ManifestDB, store_uri: str) -> list[MixedSpace]:
    """The unfinished jobs that left *this* store mixed.

    Matched on the URI as recorded. Two spellings of the same collection are two
    stores as far as this is concerned, which errs toward reporting a mixture
    that is not there rather than staying quiet about one that is.
    """
    return [state for state in mixed_spaces(db) if state.store_uri == store_uri]
