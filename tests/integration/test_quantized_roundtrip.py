"""What a quantized store actually does to a vector, measured rather than argued.

`capabilities.quantized` is a claim about a round trip: that what comes back out
of this store is decoded from a code rather than being what went in. FAISS is
where that claim can be checked without a server — an ``IndexScalarQuantizer``
is one constructor call — so this is the measurement behind the sentence
`migrate` prints, and behind `docs/guides/migration.md`.

Two directions matter and they are the same arithmetic seen from opposite ends.
Reading is what fills the shadow copy, so it decides what `rollback` restores.
Writing is what the migration does, and rebasis re-reads a sample of every batch
and compares it against what it sent — so a codec coarser than that tolerance
does not merely lose precision, it stops the job. Both are measured here.

Deliberately small: one flat index as the control, one 8-bit scalar-quantized
index as the subject. How *much* a given codec costs a given corpus is a
different piece of work, and this is not it.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from rebasis.core import l2_normalize
from rebasis.migrate.engine import VERIFY_ATOL

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.integration]

DIM = 32
N = 512

#: Labels that are not row numbers — the same convention the rest of the FAISS
#: suite uses, because an id-mapped index addresses by label and a fixture where
#: the two coincide cannot tell a correct backend from a confused one.
LABELS = np.arange(1000, 1000 + N * 3, 3, dtype=np.int64)


@pytest.fixture(scope="module")
def faiss_module() -> Any:
    """Import faiss here rather than at module scope, and skip where it aborts.

    On macOS faiss-cpu and torch each link their own OpenMP runtime and a
    process holding both aborts (faiss-wheels#40, pytorch#149201). An abort at
    import time takes the whole run with it, so the check has to happen before
    the import — which means the import cannot be at the top of the file.
    """
    import importlib.util
    import sys

    if sys.platform == "darwin" and importlib.util.find_spec("torch") is not None:
        pytest.skip("faiss-cpu and torch abort when both are loaded on macOS")
    return pytest.importorskip("faiss", reason="the faiss extra is not installed")


@pytest.fixture
def vectors(rng: np.random.Generator) -> Any:
    return l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))


def _write(faiss_module: Any, inner: Any, tmp_path: Path, vectors: Any) -> str:
    """Wrap an index in the id map rebasis needs, add the vectors, add a sidecar."""
    index = faiss_module.IndexIDMap2(inner)
    index.add_with_ids(vectors, LABELS)
    path = tmp_path / "vectors.faiss"
    faiss_module.write_index(index, str(path))
    path.with_suffix(".faiss.meta.json").write_text(
        json.dumps({"ids": [f"doc-{i}" for i in range(N)]}), encoding="utf-8"
    )
    return f"faiss://{path}"


@pytest.fixture
def flat(faiss_module: Any, tmp_path: Path, vectors: Any) -> str:
    """The control: a flat index stores the float32 vector itself."""
    return _write(faiss_module, faiss_module.IndexFlatIP(DIM), tmp_path, vectors)


@pytest.fixture
def scalar_quantized(faiss_module: Any, tmp_path: Path, vectors: Any) -> str:
    """The subject: one byte per component instead of four.

    ``QT_8bit`` fits a per-dimension range at ``train`` and keeps it there —
    adding does not retrain — so the codec is fixed for the life of the index
    and every number below is deterministic given the seed.
    """
    inner = faiss_module.IndexScalarQuantizer(
        DIM, faiss_module.ScalarQuantizer.QT_8bit, faiss_module.METRIC_INNER_PRODUCT
    )
    inner.train(vectors)
    return _write(faiss_module, inner, tmp_path, vectors)


def _read(uri: str) -> tuple[Any, dict[str, Any]]:
    from rebasis.store import open_store

    store = open_store(uri)
    return store, {r.id: r.vector for r in store.iter_records()}


def _worst(read_back: dict[str, Any], expected: Any) -> float:
    """The largest single-component deviation over the whole collection."""
    return max(
        float(np.abs(read_back[f"doc-{i}"] - expected[i]).max()) for i in range(expected.shape[0])
    )


class TestTheCapabilityMatchesTheRoundTrip:
    """A declaration is only worth something if the store behaves that way."""

    def test_a_flat_index_declares_false_and_returns_what_it_was_given(
        self, flat: str, vectors: Any
    ) -> None:
        store, read_back = _read(flat)

        assert store.capabilities.quantized is False
        assert _worst(read_back, vectors) == 0.0

    def test_a_scalar_quantized_index_declares_true_and_does_not(
        self, scalar_quantized: str, vectors: Any
    ) -> None:
        """Declared from the stored code size, confirmed by the vectors.

        This is the case rebasis used to be silent about. The index reconstructs
        without raising, so the existing check that refuses an unreadable index
        lets it straight through — and every vector it hands back is a decode.
        """
        store, read_back = _read(scalar_quantized)
        worst = _worst(read_back, vectors)

        assert store.capabilities.quantized is True
        assert worst > 0.0, "an 8-bit codec that lost nothing would not be a codec"


class TestWhatTheGapCostsAMigration:
    """The two sentences `migrate` prints, each with the measurement under it."""

    def test_reading_is_lossy_which_is_what_the_shadow_copy_holds(
        self, scalar_quantized: str, vectors: Any
    ) -> None:
        """So `rollback` restores the decoded view, not what was embedded.

        The gap measured here existed before rebasis was involved: it is the
        difference between the vectors this collection was built from and the
        vectors it can return. Nothing rebasis does can close it, and the only
        honest thing to do is say which of the two a rollback restores.
        """
        _, read_back = _read(scalar_quantized)

        assert _worst(read_back, vectors) > VERIFY_ATOL

    def test_writing_is_lossy_by_more_than_the_read_back_tolerance(
        self, scalar_quantized: str, rng: np.random.Generator
    ) -> None:
        """Which is why a quantized store can stop a migration on its first batch.

        `migrate` re-reads a sample of every batch and compares it to what it
        sent, to ``VERIFY_ATOL``. That check exists to catch a store that
        accepts a write and does not keep it; a store that re-encodes on write
        trips it for a different reason, and the plan says so beforehand rather
        than letting it surface as a failed batch.
        """
        replacement = l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))
        store, _ = _read(scalar_quantized)

        store.upsert_vectors([f"doc-{i}" for i in range(N)], replacement)

        read_back = {r.id: r.vector for r in store.iter_records()}
        assert _worst(read_back, replacement) > VERIFY_ATOL
