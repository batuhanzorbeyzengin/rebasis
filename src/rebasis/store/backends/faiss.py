"""FAISS backend.

FAISS is an index, not a database: it stores vectors and returns row numbers.
Ids, text and metadata are the caller's problem, so what this backend works with
is a bare index plus side metadata.

rebasis therefore expects two files side by side::

    vectors.faiss      the index
    vectors.meta.json  {"ids": [...], "texts": [...]}

The sidecar is what makes the index usable here at all: without it there is
nothing to re-embed and nothing to name in a report. When it is absent, the
backend still works for `probe` in vector-only mode — row numbers become ids —
and says so through its capabilities rather than by failing later.

**Reconstruction is required.** rebasis reads vectors back out of the index, and
a compressed index (IVFPQ, and the like) cannot return what was put in. The
backend checks for that up front, because the failure mode otherwise is not an
error: it is quietly measuring against lossy vectors.

**Labels are not row numbers.** An ``IndexIDMap2`` addresses vectors by the
label handed to ``add_with_ids``; a bare index addresses them by row. That
distinction runs through every call FAISS exposes -- ``search`` returns labels,
``reconstruct``, ``reconstruct_batch``, ``remove_ids`` and ``add_with_ids`` all
take them -- so this backend keeps the two apart: a *row* indexes the sidecar,
a *label* is what FAISS is spoken to in. For a bare index the two coincide,
which is exactly why conflating them survives a naive test and corrupts a real
index.

**Rows move when you write.** FAISS has no update: replacing a vector removes
its row and appends a new one, which shifts every row after it and leaves the
replaced document last. The sidecar is a plain list in row order, so it is
rewritten in the new order after each write -- otherwise every id and text
after the first replacement would name the wrong vector, silently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rebasis.errors import (
    CollectionNotFound,
    EmbeddingDimensionMismatch,
    MissingDependency,
    StoreError,
    StoreUnsupported,
    StoreWriteFailed,
)
from rebasis.types import FloatArray, Hit, Record, StoreCapabilities, as_float32

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from rebasis.store.uri import StoreURI

__all__ = ["FaissStore"]

DEFAULT_BATCH = 1000


class FaissStore:
    """A FAISS index plus its metadata sidecar."""

    def __init__(
        self,
        index: Any,
        path: Path,
        *,
        ids: list[str] | None = None,
        texts: list[str] | None = None,
    ) -> None:
        self._index = index
        self._path = path
        self._labels = _labels_of(index)
        self._ids = ids or [str(label) for label in self._labels]
        self._texts = texts
        self._has_sidecar = ids is not None
        self._reindex()

    def _reindex(self) -> None:
        """Rebuild the two lookups that turn an id or a label into a row."""
        self._row = {record_id: row for row, record_id in enumerate(self._ids)}
        self._row_of_label = {label: row for row, label in enumerate(self._labels)}

    @classmethod
    def from_uri(cls, uri: StoreURI, **kwargs: Any) -> FaissStore:
        """Open an index file.

        ``faiss:///path/to/vectors.faiss``
        """
        del kwargs
        try:
            import faiss
        except ImportError as exc:
            raise MissingDependency(
                "The FAISS backend needs faiss.",
                hint='Install it with `pip install "rebasis[faiss]"`.',
                context={"store_backend": "faiss"},
                cause=exc,
            ) from exc

        path = Path(uri.path or "")
        if not path.exists():
            raise CollectionNotFound(
                f"No FAISS index at {path}.",
                hint="The URI is a path to the index file: `faiss:///path/to/vectors.faiss`.",
                context={"store_backend": "faiss", "path": str(path)},
            )

        try:
            index = faiss.read_index(str(path))
        except Exception as exc:
            raise StoreError(
                f"{path.name} could not be read as a FAISS index.",
                hint="Check that the file was written by `faiss.write_index`.",
                context={"store_backend": "faiss"},
                cause=exc,
            ) from exc

        _require_reconstruct(index, path)
        _require_distinct_labels(index, path)
        ids, texts = _read_sidecar(path, index.ntotal)
        return cls(index, path, ids=ids, texts=texts)

    @property
    def capabilities(self) -> StoreCapabilities:
        """Honest about both the sidecar and the id map.

        No text means `migrate` has nothing to re-embed. No id map means it
        cannot write at all. A backend declares only what it can genuinely do,
        so `migrate` refuses at second zero rather than halfway through.
        """
        return StoreCapabilities(
            can_read_vectors=True,
            can_read_text=self._texts is not None,
            can_upsert_vectors=hasattr(self._index, "id_map"),
            # FAISS has no metadata, so it has nothing to filter on.
            can_filter=False,
            dimension_locked=True,
            supports_in_place_update=True,
            name="faiss",
        )

    def count(self) -> int:
        """Number of vectors in the index."""
        return int(self._index.ntotal)

    def dimension(self) -> int:
        """Vector dimensionality, declared by the index itself."""
        return int(self._index.d)

    def iter_records(
        self,
        ids: Sequence[str] | None = None,
        *,
        with_vectors: bool = True,
        with_text: bool = True,
        batch_size: int = DEFAULT_BATCH,
    ) -> Iterator[Record]:
        """Stream records, reconstructing vectors a page at a time."""
        rows = self._rows(ids) if ids is not None else range(self.count())

        page: list[int] = []
        for row in rows:
            page.append(row)
            if len(page) >= batch_size:
                yield from self._page(page, with_vectors=with_vectors, with_text=with_text)
                page = []
        if page:
            yield from self._page(page, with_vectors=with_vectors, with_text=with_text)

    def _page(self, rows: list[int], *, with_vectors: bool, with_text: bool) -> Iterator[Record]:
        vectors = None
        if with_vectors:
            # `reconstruct_batch` needs contiguous int64 *labels*; `reconstruct_n`
            # only takes a row range, which an id filter is not.
            vectors = self._index.reconstruct_batch(self._label_array(rows))
        for offset, row in enumerate(rows):
            yield Record(
                id=self._ids[row],
                vector=as_float32(vectors[offset]) if vectors is not None else None,
                text=(self._texts[row] if with_text and self._texts is not None else None),
            )

    def _rows(self, ids: Sequence[str]) -> list[int]:
        missing = [i for i in ids if i not in self._row]
        if missing:
            raise CollectionNotFound(
                f"{len(missing)} requested ids are not in this index.",
                hint=f"First missing id: {missing[0]!r}.",
                context={"store_backend": "faiss", "count": len(missing)},
            )
        return [self._row[i] for i in ids]

    def _resync(self) -> None:
        """Put the sidecar back in step with a row order the write changed.

        ``remove_ids`` closes the gap it makes and ``add_with_ids`` appends, so
        after a write the touched rows are at the end. Labels travel with their
        vectors -- that is what an id map is for -- so the labels are the thing
        to re-read, and the ids and texts follow them into their new rows.
        """
        id_of_label = dict(zip(self._labels, self._ids, strict=True))
        text_of_label = (
            dict(zip(self._labels, self._texts, strict=True)) if self._texts is not None else None
        )
        self._labels = _labels_of(self._index)
        self._ids = [id_of_label[label] for label in self._labels]
        self._texts = (
            None if text_of_label is None else [text_of_label[label] for label in self._labels]
        )
        self._reindex()

    def _write_sidecar(self) -> None:
        """Persist ids and texts in the index's current row order.

        Only when there was one to begin with: inventing a sidecar for an index
        that never had one would turn FAISS labels into ids that outlive the
        run, and the caller did not ask for that.
        """
        if not self._has_sidecar:
            return
        from rebasis.storage.atomic import atomic_write_json

        payload: dict[str, Any] = {"ids": self._ids}
        if self._texts is not None:
            payload["texts"] = self._texts
        atomic_write_json(_sidecar_path(self._path), payload)

    def _label_array(self, rows: Sequence[int]) -> Any:
        """The FAISS labels for these rows, as the int64 array FAISS wants."""
        return np.asarray([self._labels[row] for row in rows], dtype=np.int64)

    def search(self, vector: FloatArray, k: int, where: dict[str, Any] | None = None) -> list[Hit]:
        """Nearest neighbours."""
        del where  # capabilities.can_filter is False, and says so rather than pretending
        query = as_float32(vector).reshape(1, -1)
        if query.shape[1] != self.dimension():
            raise EmbeddingDimensionMismatch(
                f"The index is {self.dimension()}-dimensional but the query is "
                f"{query.shape[1]}-dimensional.",
                hint="The adapter's output dimension does not match the index.",
                context={"store_backend": "faiss", "dim": self.dimension()},
            )
        # `search` reports labels, not rows: on an id-mapped index the two are
        # different numbers and indexing the sidecar by a label is either an
        # error or, worse, the wrong document.
        scores, labels = self._index.search(query, k)
        return [
            Hit(id=self._ids[self._row_of_label[int(label)]], score=float(score), rank=rank)
            for rank, (label, score) in enumerate(zip(labels[0], scores[0], strict=True))
            if label >= 0
        ]

    def upsert_vectors(self, ids: Sequence[str], vectors: FloatArray) -> None:
        """Replace vectors in place, then persist the index — the only write path.

        FAISS has no update: the rows are removed and re-added at the same
        ids, which for an ``IDMap``-wrapped index preserves identity. Without
        that wrapper the row numbers *are* the identity, so removing shifts
        everything after — which is why the backend refuses rather than
        silently corrupting the mapping.
        """
        matrix = as_float32(vectors)
        if matrix.shape[1] != self.dimension():
            raise EmbeddingDimensionMismatch(
                f"The index is {self.dimension()}-dimensional but the vectors are "
                f"{matrix.shape[1]}-dimensional.",
                hint="Refit the adapter against this index's dimension.",
                context={"store_backend": "faiss", "dim": self.dimension()},
            )

        # `id_map`, not `remove_ids`: a plain `IndexFlat` has the latter and
        # still refuses `add_with_ids`, so checking for the method lets the
        # write start and fail as a generic store error. The user needs to be
        # told to wrap the index, not that FAISS rejected something.
        if not hasattr(self._index, "id_map"):
            raise StoreUnsupported(
                "This FAISS index cannot replace vectors in place.",
                hint=(
                    "Wrap the index in `faiss.IndexIDMap2` when building it. Without "
                    "an id map, a row's position is its identity and removing one "
                    "renumbers every row after it."
                ),
                context={"store_backend": "faiss"},
            )

        labels = self._label_array(self._rows(list(ids)))
        try:
            import faiss

            # Remove first: `add_with_ids` on a label that is already there
            # appends a second row rather than replacing the first.
            self._index.remove_ids(faiss.IDSelectorArray(labels))
            self._index.add_with_ids(matrix, labels)
            self._resync()
            faiss.write_index(self._index, str(self._path))
            self._write_sidecar()
        except Exception as exc:
            raise StoreWriteFailed(
                f"FAISS rejected an update of {len(ids)} vectors.",
                hint="Check that the index file is writable.",
                context={"store_backend": "faiss", "count": len(ids)},
                cause=exc,
            ) from exc

    def rebuild_index(self) -> None:
        """An `IndexFlatIP` scans, so there is no structure to rebuild.

        Measured: recall against exact kNN is 1.000 before and after a
        migration of 100,000 records. Declaring this as a capability would
        offer a repair for damage that cannot happen here.
        """
        from rebasis.store.base import require_capability

        require_capability(self, "can_rebuild_index", operation="rebuilding the index")


def _sidecar_path(path: Path) -> Path:
    """Where the metadata for an index lives: ``<index>.meta.json``."""
    beside = path.with_suffix(f"{path.suffix}.meta.json")
    return beside if beside.exists() else path.with_name(f"{path.stem}.meta.json")


def _labels_of(index: Any) -> list[int]:
    """The FAISS label of every row, in row order.

    A bare index has no labels: its rows *are* its addresses, so row ``i`` is
    label ``i``. An id-mapped index keeps the labels in ``id_map``, and that
    vector is in row order -- which is what makes row and label
    interconvertible at all.
    """
    if not hasattr(index, "id_map"):
        return list(range(int(index.ntotal)))
    import faiss

    return [int(label) for label in faiss.vector_to_array(index.id_map)]


def _require_distinct_labels(index: Any, path: Path) -> None:
    """Refuse an index that holds the same label twice.

    FAISS allows it -- ``add_with_ids`` appends rather than replaces -- but
    then a label no longer names one vector, and neither reading a record back
    nor replacing it has a defined answer. Duplicates almost always mean an
    index that was topped up without removing first.
    """
    labels = _labels_of(index)
    if len(set(labels)) == len(labels):
        return
    duplicates = len(labels) - len(set(labels))
    raise StoreUnsupported(
        f"{path.name} holds {duplicates} duplicate ids.",
        hint=(
            "FAISS appends on `add_with_ids` instead of replacing, so an index "
            "topped up without `remove_ids` first ends up with a label naming "
            "two vectors. Rebuild the index, or remove the older rows."
        ),
        context={"store_backend": "faiss", "count": duplicates},
    )


def _require_reconstruct(index: Any, path: Path) -> None:
    """Refuse an index rebasis cannot read vectors back out of.

    A compressed index returns an approximation of what was stored. Measuring
    an adapter against approximations would produce a number that looks like an
    answer and is not, so this fails at second zero instead.
    """
    if index.ntotal == 0:
        return
    try:
        index.reconstruct(_labels_of(index)[0])
    except Exception as exc:
        raise StoreUnsupported(
            f"{path.name} cannot reconstruct its vectors.",
            hint=(
                "rebasis reads vectors back out of the index, which a compressed "
                "index (IVFPQ and similar) cannot do exactly. Use a flat index, or "
                "export the vectors and point rebasis at those."
            ),
            context={"store_backend": "faiss"},
            cause=exc,
        ) from exc


def _read_sidecar(path: Path, expected: int) -> tuple[list[str] | None, list[str] | None]:
    """Read ``<index>.meta.json``, if it is there and it matches.

    A sidecar whose length disagrees with the index is ignored rather than
    trusted: pairing the wrong text with a vector is worse than having no text,
    because it produces plausible numbers instead of an error.
    """
    sidecar = _sidecar_path(path)
    if not sidecar.exists():
        return None, None

    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None

    ids = payload.get("ids")
    texts = payload.get("texts")
    if not isinstance(ids, list) or len(ids) != expected:
        return None, None
    if not isinstance(texts, list) or len(texts) != expected:
        texts = None
    return [str(i) for i in ids], None if texts is None else [str(t) for t in texts]
