"""Atomic write guarantees.

Storage guarantees that are not tested disappear at the first refactor. These
tests assert the property that matters: **on any failure the target file is left
untouched.**
"""

from __future__ import annotations

import json
import os
from pathlib import Path  # noqa: TC003 - pytest resolves fixture annotations at runtime

import pytest

from rebasis.errors import AtomicWriteFailed, InsufficientDiskSpace
from rebasis.storage import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_stream,
    atomic_write_text,
    require_free_space,
    rotate_backup,
)


@pytest.mark.unit
def test_writes_content(tmp_path: Path) -> None:
    """The basic case."""
    target = tmp_path / "artefact.bin"
    atomic_write_bytes(target, b"payload")
    assert target.read_bytes() == b"payload"


@pytest.mark.unit
def test_creates_parent_directories(tmp_path: Path) -> None:
    """A missing directory is created rather than raising."""
    target = tmp_path / "nested" / "deeper" / "file.txt"
    atomic_write_text(target, "hello")
    assert target.read_text() == "hello"


@pytest.mark.unit
def test_overwrite_replaces_atomically(tmp_path: Path) -> None:
    """The second write fully replaces the first, leaving no residue."""
    target = tmp_path / "artefact.bin"
    atomic_write_bytes(target, b"first-and-much-longer")
    atomic_write_bytes(target, b"second")
    assert target.read_bytes() == b"second"


@pytest.mark.unit
def test_failure_mid_write_leaves_target_untouched(tmp_path: Path) -> None:
    """The core guarantee.

    A crash between opening and swapping must not damage the existing file — the
    target is never opened for writing in the first place.
    """
    target = tmp_path / "artefact.bin"
    atomic_write_bytes(target, b"original content")

    def crash() -> None:
        with atomic_write_stream(target) as handle:
            handle.write(b"partial")
            msg = "crash mid-write"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="crash mid-write"):
        crash()

    assert target.read_bytes() == b"original content"


@pytest.mark.unit
def test_keyboard_interrupt_also_cleans_up(tmp_path: Path) -> None:
    """Ctrl-C must not leave a half-written temporary file behind either."""
    target = tmp_path / "artefact.bin"
    atomic_write_bytes(target, b"original")

    def interrupt() -> None:
        with atomic_write_stream(target) as handle:
            handle.write(b"partial")
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        interrupt()

    assert target.read_bytes() == b"original"
    assert not list(tmp_path.glob("*.tmp")), "a temporary file survived the interrupt"


@pytest.mark.unit
def test_no_temp_files_survive_success(tmp_path: Path) -> None:
    """A successful write leaves nothing but the target."""
    target = tmp_path / "artefact.bin"
    atomic_write_bytes(target, b"payload")
    assert [p.name for p in tmp_path.iterdir()] == ["artefact.bin"]


@pytest.mark.unit
def test_temp_file_is_created_in_the_target_directory(tmp_path: Path) -> None:
    """The temporary file must share a filesystem with the target.

    On a different filesystem `os.replace` fails outright instead of silently
    degrading to copy-and-delete — which is the behaviour we want.
    """
    target = tmp_path / "artefact.bin"
    seen: list[Path] = []
    with atomic_write_stream(target) as handle:
        handle.write(b"x")
        seen.extend(tmp_path.glob("*.tmp"))
    assert seen, "no temporary file was created in the target directory"


@pytest.mark.unit
def test_json_round_trip(tmp_path: Path) -> None:
    """JSON is written sorted and parses back."""
    target = tmp_path / "manifest.json"
    payload = {"schema": 1, "direction": "query_to_old", "dim": 384}
    atomic_write_json(target, payload)
    assert json.loads(target.read_text()) == payload


@pytest.mark.unit
def test_json_validates_before_swapping(tmp_path: Path) -> None:
    """Unserialisable content must not destroy an existing file."""
    target = tmp_path / "manifest.json"
    atomic_write_json(target, {"good": True})

    with pytest.raises(AtomicWriteFailed):
        atomic_write_json(target, {"bad": object()})

    assert json.loads(target.read_text()) == {"good": True}


@pytest.mark.unit
def test_rotate_backup_shifts_older_versions(tmp_path: Path) -> None:
    """One step back per rotation, oldest discarded."""
    target = tmp_path / "manifest.db"

    for version in ("v1", "v2", "v3", "v4"):
        target.write_text(version)
        rotate_backup(target, keep=3)

    assert (tmp_path / "manifest.db.bak.1").read_text() == "v4"
    assert (tmp_path / "manifest.db.bak.2").read_text() == "v3"
    assert (tmp_path / "manifest.db.bak.3").read_text() == "v2"
    assert not (tmp_path / "manifest.db.bak.4").exists()


@pytest.mark.unit
def test_rotate_backup_is_a_noop_for_a_missing_file(tmp_path: Path) -> None:
    """Nothing to back up is not an error."""
    assert rotate_backup(tmp_path / "absent.db") is None


@pytest.mark.unit
def test_space_precheck_rejects_impossible_requests(tmp_path: Path) -> None:
    """The job does not start when the space is not there.

    "The disk filled halfway through" is not an error to handle but a design
    flaw to prevent.
    """
    with pytest.raises(InsufficientDiskSpace) as excinfo:
        require_free_space(tmp_path, needed_bytes=1 << 60)
    assert excinfo.value.code == "RB-E6004"
    assert "gc --dry-run" in (excinfo.value.hint or "")


@pytest.mark.unit
def test_space_precheck_passes_for_a_small_request(tmp_path: Path) -> None:
    """A modest requirement on a normal filesystem passes."""
    require_free_space(tmp_path, needed_bytes=1024)


@pytest.mark.unit
def test_enospc_is_translated_and_leaves_target_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real ENOSPC becomes RB-E6004 and does not damage the target.

    Simulated rather than filled for real: creating a small filesystem needs
    privileges CI does not have, and the branch under test is the error
    translation, not the kernel's behaviour.

    `os.fsync` is patched rather than `os.write`: buffered writes reach the
    descriptor through the stream object, not through the `os.write` module
    attribute, and fsync is a genuine point at which a full disk surfaces.
    """
    target = tmp_path / "artefact.bin"
    atomic_write_bytes(target, b"original")

    def failing_fsync(fd: int) -> None:
        del fd
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "fsync", failing_fsync)

    def write_when_full() -> None:
        with atomic_write_stream(target) as handle:
            handle.write(b"x" * 4096)

    with pytest.raises(InsufficientDiskSpace) as excinfo:
        write_when_full()

    assert excinfo.value.code == "RB-E6004"
    monkeypatch.undo()
    assert target.read_bytes() == b"original", "the target was damaged by a failed write"
    assert not list(tmp_path.glob("*.tmp")), "a temporary file survived the failure"
