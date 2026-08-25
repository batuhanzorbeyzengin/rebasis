"""Can the index still *find* what the migration wrote to it?

Everything else `migrate` checks answers a different question. The per-batch
read-back proves the store took the write. The end-of-job check on a fresh
connection proves it kept it. Neither proves the record can still be retrieved,
and on every graph-based backend that is a separate question with a separate
answer.

**Why it is separate.** HNSW picks a record's edges when the record is inserted,
from the geometry of the vectors around it at that moment. Rewriting the vector
does not rewrite the graph. Afterwards the edges describe a neighbourhood that
no longer exists, and traversal walks toward the wrong region — so a search can
miss a document that is sitting in the index, correct and verified, one hop
outside the path the graph sends the query down. The counts are right, the
payloads are right, nothing raises, and recall falls.

This is not a hypothetical. Qdrant's own incremental-HNSW work states the rule
plainly — a changed vector value invalidates the graph the same way a deletion
does (`qdrant/qdrant#6325`) — and a production report measured search quality at
34% until the collection was force-reindexed
(`qdrant/qdrant#7147`). The literature calls the general case unreachable points
and node isolation under update (arXiv:2407.07871, arXiv:2507.19802). rebasis
writes to five backends and asks none of them about it.

**How it is measured.** Take a sample of records, use their own stored vectors
as queries, and compare what the store's index returns against exact nearest
neighbours computed by streaming the corpus. Their overlap is the index's
recall against its own contents — a property of the index structure, not of the
embedding model, and directly comparable before and after a migration.

The exact side streams. Peak memory is ``O((sample + batch) × d)``, the same
invariant every other read path holds, so this runs on a five-million-record
collection at the cost of a scan rather than the cost of a matrix.

A record retrieves itself first in both rankings, which would put a free hit in
every comparison. Both sides ask for one extra neighbour and drop the query's
own id, so the figure is about the other k.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from rebasis.compute.search import top_k_search
from rebasis.observability import Events, get_logger

if TYPE_CHECKING:
    from rebasis.store.base import VectorStore
    from rebasis.types import FloatArray

__all__ = ["HealthComparison", "IndexHealth", "measure_index_health"]

log = get_logger(__name__)

#: Records used as probe queries. Enough that a mean over them resolves the
#: kind of drop that matters — a collapse from 1.00 to 0.34, not a wobble in the
#: third decimal — and few enough that the exact side stays one scan rather than
#: a second matrix.
DEFAULT_SAMPLE = 200

#: Cut-off. Ten because that is what a RAG pipeline retrieves, so a fall here is
#: a fall in the thing the user actually consumes.
DEFAULT_K = 10

#: Documents pulled from the store per page while the exact side streams.
SCAN_BATCH = 1024


@dataclass(frozen=True, slots=True)
class IndexHealth:
    """How much of the exact answer this index's own search returns."""

    #: Mean overlap between the store's search and exact kNN, over the probes.
    recall: float
    n_probes: int
    k: int
    #: Documents the exact side scanned. Reported because the number is what
    #: makes the recall meaningful: exact against 500 documents is a different
    #: claim from exact against 5,000,000.
    n_documents: int
    duration_seconds: float
    backend: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form, for the audit record and ``--json``."""
        return {
            "ann_recall": round(self.recall, 4),
            "n_probes": self.n_probes,
            "k": self.k,
            "n_documents": self.n_documents,
            "duration_ms": round(self.duration_seconds * 1000, 1),
            "store_backend": self.backend,
        }


@dataclass(frozen=True, slots=True)
class HealthComparison:
    """The same measurement either side of a migration."""

    before: IndexHealth
    after: IndexHealth

    @property
    def delta(self) -> float:
        """Change in recall. Negative means the index got worse at finding."""
        return self.after.recall - self.before.recall

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "ann_recall_before": round(self.before.recall, 4),
            "ann_recall_after": round(self.after.recall, 4),
            "ann_recall_delta": round(self.delta, 4),
            "k": self.after.k,
            "n_probes": self.after.n_probes,
        }

    def explain(self) -> str:
        """What happened, in a sentence, with no threshold applied.

        Deliberately reports rather than judges. What counts as a serious drop
        depends on the backend and on the index parameters, and this project
        does not publish a threshold it has not measured. The number and its
        direction are the finding; the reader decides.
        """
        if self.delta < 0:
            return (
                f"The index returns less of the exact answer than before: "
                f"recall@{self.after.k} against exact kNN fell from "
                f"{self.before.recall:.3f} to {self.after.recall:.3f} over "
                f"{self.after.n_probes} probes. The vectors are correct and verified — "
                f"this is the search structure, which was built against the geometry "
                f"the old vectors had. If this backend can rebuild its index, doing so "
                f"is what recovers it."
            )
        return (
            f"The index finds as much as before: recall@{self.after.k} against exact "
            f"kNN went from {self.before.recall:.3f} to {self.after.recall:.3f}."
        )


def measure_index_health(
    store: VectorStore,
    *,
    sample: int = DEFAULT_SAMPLE,
    k: int = DEFAULT_K,
    seed: int = 0,
    batch_size: int = SCAN_BATCH,
) -> IndexHealth:
    """Compare the store's own search against exact nearest neighbours.

    Two streaming passes over the collection: one to reservoir-sample the probe
    vectors, one to compute their exact neighbours. Neither materialises the
    corpus.

    Args:
        store: The collection to measure. Read only.
        sample: Records used as probes.
        k: Cut-off.
        seed: Recorded so the same probes are drawn either side of a migration —
            comparing two runs that sampled differently would measure the sample.
        batch_size: Records per page while scanning.

    Returns:
        The measurement. ``recall`` is ``nan`` when the collection is too small
        to have a meaningful neighbourhood, which is honest: a five-record
        collection returns everything and says nothing about the index.
    """
    started = time.perf_counter()
    backend = store.capabilities.name

    probe_ids, probes = _draw_probes(store, sample=sample, seed=seed, batch_size=batch_size)
    if probes.size == 0:
        return IndexHealth(
            recall=float("nan"),
            n_probes=0,
            k=k,
            n_documents=0,
            duration_seconds=time.perf_counter() - started,
            backend=backend,
        )

    exact, scanned = _exact_neighbours(store, probes, k=k + 1, batch_size=batch_size)

    overlaps = []
    for position, record_id in enumerate(probe_ids):
        # The probe is its own nearest neighbour in both rankings. Dropping it
        # from each keeps the comparison about the other k.
        truth = [doc for doc in exact[position] if doc != record_id][:k]
        if not truth:
            continue
        found = [hit.id for hit in store.search(probes[position], k=k + 1) if hit.id != record_id]
        overlaps.append(len(set(truth) & set(found[:k])) / len(truth))

    recall = float(np.mean(overlaps)) if overlaps else float("nan")
    duration = time.perf_counter() - started

    log.info(
        Events.MIGRATE_INDEX_MEASURED,
        store_backend=backend,
        count=len(overlaps),
        ann_recall=round(recall, 4),
        duration_ms=round(duration * 1000, 1),
    )
    return IndexHealth(
        recall=recall,
        n_probes=len(overlaps),
        k=k,
        n_documents=scanned,
        duration_seconds=duration,
        backend=backend,
    )


def _draw_probes(
    store: VectorStore, *, sample: int, seed: int, batch_size: int
) -> tuple[list[str], FloatArray]:
    """Reservoir-sample records to use as queries.

    Reservoir rather than "the first N": the first page of a collection is
    whatever the store's iteration order puts there, which on several backends
    is insertion order — and the oldest records are the ones whose graph
    neighbourhood has had the most time to be disturbed. Measuring only those
    would report the worst case as the average.
    """
    rng = np.random.default_rng(seed)
    kept_ids: list[str] = []
    kept: list[FloatArray] = []
    seen = 0

    for record in store.iter_records(with_vectors=True, with_text=False, batch_size=batch_size):
        if record.vector is None:
            continue
        seen += 1
        if len(kept) < sample:
            kept_ids.append(record.id)
            kept.append(np.asarray(record.vector, dtype=np.float32))
            continue
        slot = int(rng.integers(seen))
        if slot < sample:
            kept_ids[slot] = record.id
            kept[slot] = np.asarray(record.vector, dtype=np.float32)

    if not kept:
        return [], np.empty((0, 0), dtype=np.float32)
    return kept_ids, np.vstack(kept)


def _exact_neighbours(
    store: VectorStore, probes: FloatArray, *, k: int, batch_size: int
) -> tuple[list[list[str]], int]:
    """Exact top-k for each probe, by streaming the corpus.

    The running top-k is kept as ``(probes, k)`` scores and ids and merged with
    each page. That is the whole reason this is affordable: the score matrix for
    one page is ``probes × batch``, never ``probes × N``.
    """
    n_probes = probes.shape[0]
    best_scores = np.full((n_probes, k), -np.inf, dtype=np.float32)
    best_ids: list[list[str]] = [[""] * k for _ in range(n_probes)]
    scanned = 0

    page_ids: list[str] = []
    page_vectors: list[FloatArray] = []

    def flush() -> None:
        nonlocal page_ids, page_vectors
        if not page_vectors:
            return
        block = np.vstack(page_vectors)
        indices, scores = top_k_search(probes, block, k=min(k, block.shape[0]))
        _merge(best_scores, best_ids, scores, indices, page_ids, k=k)
        page_ids = []
        page_vectors = []

    for record in store.iter_records(with_vectors=True, with_text=False, batch_size=batch_size):
        if record.vector is None:
            continue
        page_ids.append(record.id)
        page_vectors.append(np.asarray(record.vector, dtype=np.float32))
        scanned += 1
        if len(page_vectors) >= batch_size:
            flush()
    flush()

    return best_ids, scanned


def _merge(  # noqa: PLR0913 - the running state and the page it folds in
    best_scores: FloatArray,
    best_ids: list[list[str]],
    scores: FloatArray,
    indices: np.ndarray,
    page_ids: list[str],
    *,
    k: int,
) -> None:
    """Fold one page's top-k into the running one, in place.

    ``kind="stable"`` so that equal scores keep the order they were seen in.
    Ties are common — a corpus with duplicated chunks produces them by the
    hundred — and an unstable sort would make the exact side depend on page
    boundaries, which is the one thing a *reference* answer may not do.
    """
    for row in range(best_scores.shape[0]):
        combined_scores = np.concatenate([best_scores[row], scores[row]])
        combined_ids = best_ids[row] + [page_ids[int(i)] for i in indices[row]]
        order = np.argsort(-combined_scores, kind="stable")[:k]
        best_scores[row] = combined_scores[order]
        best_ids[row] = [combined_ids[int(i)] for i in order]
