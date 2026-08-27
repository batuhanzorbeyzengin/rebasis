"""Whether a store keeps what it is given, and what `migrate` does about it.

`rollback` is sold on a bit-identical shadow copy. The shadow is bit-identical
to *what the store returned*, and a store that keeps compressed codes returns a
value decoded from them — so on such a store "the original" is a narrower thing
than the sentence suggests. These tests pin the three parts of saying so: the
capability's tri-state, the fact that the shadow really does hold the store's
decoded view rather than the vectors that were embedded, and what the plan
prints for each of the three answers.

Everything here runs against fakes. A quantizer is four lines of numpy, and a
fake can be made exactly as coarse as a claim needs — which a real store cannot.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from rebasis.cli._common import console
from rebasis.cli.migrate import _note_quantized_store
from rebasis.core import IdentityAdapter
from rebasis.manifest import ManifestDB, manifest_path
from rebasis.migrate import JobState, MigrationEngine
from rebasis.store.backends.memory import MemoryStore
from rebasis.types import Record, StoreCapabilities, as_float32

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

DIM = 8
N = 16


class QuantizingStore:
    """A store that rounds every vector to a grid, the way a codec does.

    ``step`` is the whole point of the fake: it is the width of one code, so it
    sets both whether the round trip is lossy and by how much. Rounding rather
    than truncating, because that is what a scalar quantizer does and because a
    truncating fake would bias every deviation the same way and hide a sign
    error in anything reading it.
    """

    def __init__(self, ids: Sequence[str], vectors: Any, *, step: float) -> None:
        self._ids = list(ids)
        self._step = step
        self._vectors = self._encode(as_float32(vectors))

    def _encode(self, vectors: Any) -> Any:
        return as_float32(np.round(np.asarray(vectors) / self._step) * self._step)

    @property
    def capabilities(self) -> StoreCapabilities:
        return StoreCapabilities(
            can_read_vectors=True,
            can_read_text=False,
            can_upsert_vectors=True,
            can_filter=False,
            dimension_locked=True,
            supports_in_place_update=True,
            quantized=True,
            name="quantizing-fake",
        )

    def count(self) -> int:
        return len(self._ids)

    def dimension(self) -> int:
        return int(self._vectors.shape[1])

    def iter_records(
        self,
        ids: Sequence[str] | None = None,
        *,
        with_vectors: bool = True,
        with_text: bool = True,
        batch_size: int = 1000,
    ) -> Iterator[Record]:
        del with_text, batch_size
        wanted = self._ids if ids is None else [str(i) for i in ids]
        for record_id in wanted:
            position = self._ids.index(record_id)
            yield Record(
                id=record_id,
                vector=self._vectors[position] if with_vectors else None,
            )

    def upsert_vectors(self, ids: Sequence[str], vectors: Any) -> None:
        encoded = self._encode(vectors)
        for offset, record_id in enumerate(ids):
            self._vectors[self._ids.index(str(record_id))] = encoded[offset]


class DeclaringStore:
    """Nothing but a capability declaration, for the paths that read only that."""

    def __init__(self, *, quantized: bool | None) -> None:
        self._quantized = quantized

    @property
    def capabilities(self) -> StoreCapabilities:
        return StoreCapabilities(
            can_read_vectors=True,
            can_read_text=False,
            can_upsert_vectors=True,
            can_filter=False,
            dimension_locked=True,
            supports_in_place_update=True,
            quantized=self._quantized,
            name="declaring-fake",
        )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(7)


@pytest.fixture
def originals(rng: np.random.Generator) -> Any:
    from rebasis.core import l2_normalize

    return l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))


def ids() -> list[str]:
    return [f"doc-{i:03d}" for i in range(N)]


def build_engine(tmp_path: Path, store: Any, *, db: Any = None, **kwargs: Any) -> MigrationEngine:
    """A job over ``store`` with the identity adapter.

    The identity is the right adapter here for the same reason `rollback` uses
    it: these tests are about what the store does to a vector, and an adapter
    that changed it too would put a second transformation between the write and
    the read.
    """
    return MigrationEngine(
        db=db if db is not None else ManifestDB(manifest_path(tmp_path / "state")),
        store=store,
        adapter=IdentityAdapter(input_dim=DIM, output_dim=DIM),
        shadow_root=tmp_path / "shadow",
        batch_size=N,
        power_aware=False,
        **kwargs,
    )


def spoken(store: Any, *, dry_run: bool) -> str:
    """What the plan prints, with Rich's line wrapping flattened out.

    Wrapping is a property of the terminal width, so asserting on raw output
    would make these tests fail on a narrow one for no reason.
    """
    with console.capture() as captured:
        _note_quantized_store(store, dry_run=dry_run)
    return " ".join(captured.get().split())


class TestTheCapability:
    """Three states, and the default is the one that claims nothing."""

    def test_a_backend_that_did_not_look_claims_nothing(self) -> None:
        """``None`` is the default because ``False`` would be a guarantee.

        ``can_rebuild_index`` next door defaults to ``False`` safely: refusing
        to offer a repair costs nobody anything. ``quantized=False`` is the
        opposite — it promises that what you write is what you read back, which
        is exactly what `rollback` is sold on.
        """
        declared = StoreCapabilities(
            can_read_vectors=True,
            can_read_text=True,
            can_upsert_vectors=True,
            can_filter=False,
            dimension_locked=True,
            supports_in_place_update=True,
        )

        assert declared.quantized is None

    def test_the_reference_backend_looked(self, originals: Any) -> None:
        """A float32 numpy array with no encoder between write and read."""
        store = MemoryStore(ids(), originals)

        assert store.capabilities.quantized is False


class TestWhatTheShadowHolds:
    """The claim `migrate` makes, checked rather than asserted in prose."""

    def test_the_shadow_holds_the_store_view_not_the_embedding(
        self, tmp_path: Path, originals: Any
    ) -> None:
        """Bit-identical to what was *read*, which is not what was embedded.

        A step this fine keeps the run alive — the read-back check would stop a
        coarser one — so that the shadow can be inspected after a completed
        job. The point is not the size of the gap but which of the two vectors
        the shadow copies.
        """
        store = QuantizingStore(ids(), originals, step=1e-6)
        engine = build_engine(tmp_path, store)
        engine.prepare(ids())

        engine.run()

        # Keyed by id rather than compared row for row: the queue orders a
        # batch itself, and this is a claim about content, not about order.
        shadowed = dict(zip(engine.shadow.ids(), engine.shadow.read_vectors(), strict=True))
        untouched = QuantizingStore(ids(), originals, step=1e-6)
        assert set(shadowed) == set(ids())
        for position, record in enumerate(untouched.iter_records()):
            assert np.array_equal(shadowed[record.id], record.vector)
            assert not np.array_equal(shadowed[record.id], originals[position])

    def test_rollback_restores_what_the_collection_read_back(
        self, tmp_path: Path, originals: Any
    ) -> None:
        """So "restored" means the state the migration replaced, exactly.

        It does not mean the vectors the embedding model produced. That
        precision was spent when the collection was built, and no shadow copy
        taken afterwards can return it.
        """
        store = QuantizingStore(ids(), originals, step=1e-6)
        before = np.vstack([r.vector for r in store.iter_records()])
        engine = build_engine(tmp_path, store)
        engine.prepare(ids())
        engine.run()

        engine.rollback()

        after = np.vstack([r.vector for r in store.iter_records()])
        assert np.array_equal(after, before)
        assert not np.array_equal(after, originals)

    def test_a_codec_coarser_than_the_read_back_tolerance_stops_the_job(
        self, tmp_path: Path, originals: Any
    ) -> None:
        """And stops it with the shadow copy already on disk.

        `migrate` re-reads a sample of every batch and compares it to what it
        sent, to ``VERIFY_ATOL``. A store that re-encodes on write cannot return
        what it was given, so a codec coarser than that trips the check — which
        is why the plan says so before the confirmation rather than after the
        first batch. The step here is two orders of magnitude coarser, so this
        does not sit on the tolerance and does not move if the tolerance does.
        """
        store = QuantizingStore(ids(), originals, step=1e-2)
        engine = build_engine(tmp_path, store)
        engine.prepare(ids())

        result = engine.run()

        assert result.state is JobState.PAUSED
        assert "does not match what was written" in result.pause_reason
        assert engine.shadow.manifest().record_count == N


class TestWhatThePlanSays:
    """`migrate` states the difference; it never refuses over it."""

    def test_a_quantized_store_is_named_before_anything_is_written(self) -> None:
        said = spoken(DeclaringStore(quantized=True), dry_run=False)
        assert "stores its vectors quantized" in said

    def test_a_store_that_keeps_what_it_is_given_says_nothing(self) -> None:
        """There is no finding, so there is no line.

        A caveat printed on every migration is a caveat that stops being read,
        and the store that most needs the warning would be the one it stopped
        being read on.
        """
        assert spoken(DeclaringStore(quantized=False), dry_run=False) == ""
        assert spoken(DeclaringStore(quantized=False), dry_run=True) == ""

    def test_an_unknown_is_not_a_finding(self) -> None:
        """Silent on the normal path, stated under `--dry-run`.

        An unknown is the absence of an answer, not an answer — warning on it
        would fire on every third-party store behind a bridge. `--dry-run` is
        the one place the user has asked for everything the plan knows, so it
        is the one place the gap is named.
        """
        assert spoken(DeclaringStore(quantized=None), dry_run=False) == ""
        assert "could not be determined" in spoken(DeclaringStore(quantized=None), dry_run=True)

    def test_the_dry_run_says_what_the_short_form_leaves_out(self) -> None:
        """The write direction, and the tolerance that acts on it.

        The tolerance is read from ``VERIFY_ATOL`` rather than written out, in
        the test as in the message: a literal here would let the two drift apart
        and still pass.
        """
        from rebasis.migrate.engine import VERIFY_ATOL

        tolerance = f"{VERIFY_ATOL:.0e}"
        short = spoken(DeclaringStore(quantized=True), dry_run=False)
        full = spoken(DeclaringStore(quantized=True), dry_run=True)

        assert tolerance not in short
        assert tolerance in full
        assert "re-encoded on write" in full


class TestTheMachineReadableRecord:
    """`migrate` has no `--json`, so the audit trail is where a script reads this."""

    def _recorded(self, tmp_path: Path, store: Any) -> Any:
        from rebasis.audit import AuditWriter

        writer = AuditWriter(ManifestDB(manifest_path(tmp_path / "state")), run_id="run-quantized")
        build_engine(tmp_path, store, db=writer.db, audit=writer).prepare(ids())
        rows = writer.db.query("SELECT inputs_json FROM audit_records ORDER BY seq DESC LIMIT 1")
        writer.db.close()
        return json.loads(rows[0]["inputs_json"])["store_quantized"]

    def test_a_finding_is_recorded_as_a_boolean(self, tmp_path: Path, originals: Any) -> None:
        assert self._recorded(tmp_path, MemoryStore(ids(), originals)) is False

    def test_an_unknown_is_recorded_as_null(self, tmp_path: Path) -> None:
        """``null``, not ``false``. A script must be able to tell them apart."""
        assert self._recorded(tmp_path, DeclaringStore(quantized=None)) is None

    def test_a_quantized_store_is_recorded_as_true(self, tmp_path: Path) -> None:
        assert self._recorded(tmp_path, DeclaringStore(quantized=True)) is True
