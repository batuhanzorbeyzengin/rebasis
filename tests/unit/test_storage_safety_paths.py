"""The branches that only run when something has gone wrong.

Every module here is one where a bug costs data rather than accuracy, and every
path here is one that a normal run never takes: a corrupt shadow segment, a
temporary file left by a process that died, a lock held by someone else, a chain
that does not verify. These modules are held to full coverage for exactly that
reason — the code that only runs on a bad day is the code nobody notices is
wrong.
"""

from __future__ import annotations

import os
import time
from pathlib import Path  # noqa: TC003 - runtime annotations on fixtures

import numpy as np
import pytest

from rebasis.audit.chain import ChainVerification
from rebasis.errors import LockHeld, ShadowMissing
from rebasis.storage.atomic import cleanup_stale_temp_files
from rebasis.storage.locks import lock_info, state_lock
from rebasis.storage.shadow import ShadowStore

pytestmark = pytest.mark.unit

DIM = 8


class TestTheChainSaysWhatItFound:
    def test_a_clean_chain_reports_how_many_it_checked(self) -> None:
        assert ChainVerification(valid=True, records_checked=12).describe() == "12 records verified"

    def test_a_pruned_chain_says_so_rather_than_looking_shorter(self) -> None:
        """A gap left by `audit prune` is deliberate. Reporting the count alone
        would make a pruned trail indistinguishable from a truncated one."""
        result = ChainVerification(valid=True, records_checked=12, gaps=[4, 9])

        assert result.describe() == "12 records verified (2 recorded gaps)"

    def test_a_broken_chain_names_the_record_and_the_reason(self) -> None:
        """ "Broken" without a sequence number sends someone to read the whole
        trail by hand."""
        result = ChainVerification(
            valid=False, records_checked=7, broken_at=5, reason="prev_hash mismatch"
        )

        assert "seq 5" in result.describe()
        assert "prev_hash mismatch" in result.describe()


class TestSweepingUpAfterACrashedProcess:
    def test_an_old_temporary_file_is_removed(self, tmp_path: Path) -> None:
        stale = tmp_path / "manifest.json.abc.tmp"
        stale.write_text("half a write", encoding="utf-8")
        old = time.time() - 200_000
        os.utime(stale, (old, old))

        removed = cleanup_stale_temp_files(tmp_path)

        assert removed == [stale]
        assert not stale.exists()

    def test_a_fresh_one_is_left_alone(self, tmp_path: Path) -> None:
        """It may belong to a write happening right now in another process, and
        deleting it would corrupt a run that is going fine."""
        fresh = tmp_path / "manifest.json.def.tmp"
        fresh.write_text("in progress", encoding="utf-8")

        assert cleanup_stale_temp_files(tmp_path) == []
        assert fresh.exists()

    def test_a_directory_that_does_not_exist_is_not_an_error(self, tmp_path: Path) -> None:
        """Nothing to sweep is a normal outcome for a first run."""
        assert cleanup_stale_temp_files(tmp_path / "never-created") == []

    def test_only_temporary_files_are_touched(self, tmp_path: Path) -> None:
        keeper = tmp_path / "manifest.db"
        keeper.write_text("the actual state", encoding="utf-8")
        old = time.time() - 200_000
        os.utime(keeper, (old, old))

        cleanup_stale_temp_files(tmp_path)

        assert keeper.exists()


class TestTheShadowKnowsWhenItIsDamaged:
    @pytest.fixture
    def shadow(self, tmp_path: Path, rng: np.random.Generator) -> ShadowStore:
        store = ShadowStore(tmp_path / "shadow", "job-1")
        for batch in range(3):
            store.append(
                [f"doc-{batch}-{i}" for i in range(4)],
                rng.standard_normal((4, DIM)).astype(np.float32),
            )
        return store

    def test_an_intact_shadow_reads_back(self, shadow: ShadowStore) -> None:
        assert shadow.read_vectors().shape == (12, DIM)

    def test_a_corrupted_segment_is_caught_and_named(self, shadow: ShadowStore) -> None:
        """The point of hashing per segment rather than per file is that the
        error can say *which records* are affected — the rest are still
        recoverable, and a rollback that restores 8 of 12 beats one that refuses
        because a byte moved."""
        path = shadow.directory / "vectors.bin"
        data = bytearray(path.read_bytes())
        data[DIM * 4 + 3] ^= 0xFF  # inside the second batch
        path.write_bytes(bytes(data))

        with pytest.raises(ShadowMissing) as caught:
            shadow.read_vectors()

        assert "corrupt" in str(caught.value).lower()

    def test_verification_can_be_skipped_when_the_caller_knows_better(
        self, shadow: ShadowStore
    ) -> None:
        """Hashing the whole shadow on every read would double the cost of a
        rollback that has already verified once."""
        path = shadow.directory / "vectors.bin"
        data = bytearray(path.read_bytes())
        data[0] ^= 0xFF
        path.write_bytes(bytes(data))

        assert shadow.read_vectors(verify=False).shape == (12, DIM)


class TestTheStateLock:
    def test_it_records_who_holds_it_and_why(self, tmp_path: Path) -> None:
        with state_lock(tmp_path, operation="migrate"):
            info = lock_info(tmp_path)

        assert info is not None
        assert info["pid"] == os.getpid()
        assert info["operation"] == "migrate"
        assert info["alive"] is True

    def test_a_second_holder_is_refused_and_told_who_has_it(self, tmp_path: Path) -> None:
        """Two writers in one manifest is the failure this prevents, and "it is
        locked" without naming the holder is not actionable."""
        with (
            state_lock(tmp_path, operation="migrate"),
            pytest.raises(LockHeld) as caught,
            state_lock(tmp_path, operation="gc", timeout=0.01),
        ):
            pass

        assert str(os.getpid()) in (caught.value.hint or "")
        assert "migrate" in (caught.value.hint or "")

    def test_it_is_released_even_when_the_block_raises(self, tmp_path: Path) -> None:
        """A lock that survives an exception locks the user out of their own
        state directory until they find the file and delete it."""
        with pytest.raises(RuntimeError, match="boom"), state_lock(tmp_path, operation="migrate"):
            raise RuntimeError("boom")

        with state_lock(tmp_path, operation="gc"):
            pass

    def test_no_lock_means_no_info(self, tmp_path: Path) -> None:
        assert lock_info(tmp_path) is None
