"""Atomic writes.

Every file write in rebasis goes through this module. Direct ``open(path, "w")``
elsewhere under ``src/`` is rejected by a test, because the naive approach has a
genuinely destructive failure mode:

**ENOSPC is the insidious one.** When you write over a file directly and the
disk fills, the old content has already been truncated and the new content
cannot be written. The user's data is simply gone. Atomic write removes that
mode entirely: the target file is never opened for writing at all.

Three steps, and the **third is the one that is usually skipped**:

1. ``fsync`` the temporary file — content reaches the disk
2. ``os.replace`` — the atomic swap (guaranteed by POSIX rename)
3. ``fsync`` the *directory* — the rename itself reaches the disk

Without step 3 the rename is atomic with respect to other processes but not with
respect to a crash: a power cut can leave a **zero-length file** where the target
used to be. That is a bug class found and patched in ``ldconfig``, not a
theoretical concern.
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rebasis.errors import AtomicWriteFailed, InsufficientDiskSpace

#: errno for "no space left on device". Named because a bare 28 in an except
#: clause is unreadable, and this is the branch guarding the most destructive
#: failure mode there is.
_ENOSPC = 28

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_stream",
    "atomic_write_text",
    "free_bytes",
    "require_free_space",
    "rotate_backup",
]

#: Backups kept per critical file. One step back costs one file size.
DEFAULT_BACKUP_COUNT = 3


def _fsync_directory(path: Path) -> None:
    """Fsync a directory so that a rename inside it is durable.

    Silently unsupported on some platforms and filesystems (notably Windows),
    where the failure is not actionable — the swap has already happened and the
    caller cannot do anything useful about the missing flush.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


@contextmanager
def atomic_write_stream(
    path: Path, *, fsync_dir: bool = True, private: bool = True
) -> Iterator[Any]:
    """Write a file atomically through a stream.

    Used for large files (shadow vectors) that must not be held in memory
    in one piece: they are written incrementally, then fsynced and swapped once.

    The temporary file is created **in the same directory** as the target. On a
    different filesystem ``os.replace`` fails outright rather than falling back
    to copy-and-delete the way ``shutil.move`` does — which is what we want. A
    loud failure beats a silent loss of atomicity.

    Args:
        path: Where the finished file goes.
        fsync_dir: Whether to fsync the directory after the swap.
        private: Keep ``mkstemp``'s owner-only permissions. The default for
            everything rebasis writes into its own state directory, and wrong
            for anything the user asked for by name: a report written with
            ``--report`` is theirs to read, serve or send, and inheriting 0600
            from a temporary file is an accident rather than a policy.
    """
    path = Path(path)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())  # 1. content to disk
        if not private:
            _apply_umask(tmp)
        os.replace(tmp, path)  # 2. atomic swap
        if fsync_dir:
            _fsync_directory(directory)  # 3. directory metadata to disk
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        if exc.errno == _ENOSPC:
            raise InsufficientDiskSpace(
                f"Ran out of disk space writing {path.name}.",
                hint=(
                    "The target file was left untouched. Free some space and retry; "
                    "`rebasis gc --dry-run` shows what can be removed."
                ),
                context={"count": 0},
                cause=exc,
            ) from exc
        raise AtomicWriteFailed(
            f"Could not write {path.name} atomically.",
            hint="The target file was left untouched.",
            cause=exc,
        ) from exc
    except BaseException:
        # Including KeyboardInterrupt: a half-written temporary file must not
        # survive a Ctrl-C either.
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_bytes(
    path: Path, data: bytes, *, fsync_dir: bool = True, private: bool = True
) -> None:
    """Write bytes atomically."""
    with atomic_write_stream(path, fsync_dir=fsync_dir, private=private) as handle:
        handle.write(data)


def atomic_write_text(
    path: Path, text: str, *, encoding: str = "utf-8", private: bool = True
) -> None:
    """Write text atomically."""
    atomic_write_bytes(path, text.encode(encoding), private=private)


def _apply_umask(tmp: Path) -> None:
    """Give a file the permissions an ordinary write would have given it.

    ``mkstemp`` creates at 0600, which is right for a state file and wrong for
    one the user named. There is no way to read the umask without setting it,
    so it is set and immediately restored.
    """
    umask = os.umask(0)
    os.umask(umask)
    with suppress(OSError):
        tmp.chmod(0o666 & ~umask)


def atomic_write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Write JSON atomically, validating before the swap.

    The temporary file is re-read and parsed before ``os.replace`` is called. If
    serialisation produced something unreadable, the swap never happens and **the
    existing file is left untouched** — a corrupt manifest is worse than a stale
    one.
    """
    try:
        # Serialisation itself must be inside the guard: an unserialisable value
        # raises here, not in the parse-back, and a raw TypeError must never
        # cross this boundary — callers only ever see a RebasisError.
        encoded = json.dumps(payload, indent=indent, sort_keys=True, ensure_ascii=False)
        json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise AtomicWriteFailed(
            f"Refusing to write {path.name}: the serialised payload does not parse back.",
            hint="This is a bug in rebasis. The existing file was left untouched.",
            cause=exc,
        ) from exc
    atomic_write_bytes(path, encoded.encode("utf-8"))


def rotate_backup(path: Path, *, keep: int = DEFAULT_BACKUP_COUNT) -> Path | None:
    """Move the current file to ``.bak.1``, shifting older backups down.

    For critical files (manifest, adapters) one step back costs a file's worth of
    disk. Returns the backup path, or ``None`` when there was nothing to back up.
    """
    if not path.exists():
        return None

    oldest = path.with_suffix(path.suffix + f".bak.{keep}")
    oldest.unlink(missing_ok=True)
    for index in range(keep - 1, 0, -1):
        src = path.with_suffix(path.suffix + f".bak.{index}")
        if src.exists():
            os.replace(src, path.with_suffix(path.suffix + f".bak.{index + 1}"))

    target = path.with_suffix(path.suffix + ".bak.1")
    os.replace(path, target)
    _fsync_directory(path.parent)
    return target


def free_bytes(path: Path) -> int:
    """Free space on the filesystem holding ``path``."""
    probe = path if path.exists() else path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return int(os.statvfs(probe).f_bavail * os.statvfs(probe).f_frsize)


def require_free_space(path: Path, needed_bytes: int, *, safety_margin: float = 0.20) -> None:
    """Fail before starting if there is not enough room.

    A migration that fills the disk halfway through is not an error to be
    handled but a design flaw to be prevented — so the check happens up front,
    with a margin, and the job simply does not start.

    Raises:
        InsufficientDiskSpace: When free space is below the requirement.
    """
    required = int(needed_bytes * (1 + safety_margin))
    available = free_bytes(path)
    if available < required:
        raise InsufficientDiskSpace(
            f"Need {required / 1024**3:.2f} GB free "
            f"(including a {safety_margin:.0%} margin) but only "
            f"{available / 1024**3:.2f} GB is available.",
            hint=(
                "Free some space and retry. `rebasis gc --dry-run` lists what can "
                "be removed, and `--no-keep-original` skips the shadow copy at "
                "the cost of losing rollback."
            ),
            context={"count": needed_bytes},
        )


def cleanup_stale_temp_files(directory: Path, *, older_than_seconds: float = 86_400) -> list[Path]:
    """Remove leftover ``.tmp`` files from crashed processes.

    Only files older than the threshold are touched: a fresh temporary file may
    belong to a write happening right now in another process.
    """
    import time

    if not directory.exists():
        return []
    cutoff = time.time() - older_than_seconds
    removed: list[Path] = []
    for candidate in directory.glob("*.tmp"):
        with suppress(OSError):
            if candidate.stat().st_mtime < cutoff:
                candidate.unlink()
                removed.append(candidate)
    return removed
