"""Process locking.

Two ``rebasis migrate`` processes on the same manifest would corrupt the job
state. A file lock in the state directory prevents that.

**Which commands take the lock, and why the split matters:**

* Exclusive: ``migrate``, ``adapter upgrade``, ``gc``, schema migration — every
  command that writes.
* No lock: ``probe``, ``fit``, ``audit``, ``status`` — read-only, or writing only
  to their own new files.

That split is what lets a user watch ``rebasis status`` while a migration runs,
which is exactly when they most want to.

**A stale lock is never broken automatically.** The holder's PID and start time
are recorded; if the process is gone, rebasis says so and suggests removing the
file. Breaking it automatically would mean guessing that a process is dead, and
being wrong about that is how two writers end up in the same manifest.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rebasis.errors import LockHeld

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["LOCK_NAME", "lock_info", "state_lock"]

LOCK_NAME = "rebasis.lock"

#: How long to wait for the lock before reporting it held. Short on purpose: a
#: user who runs two migrations wants to be told, not to wait.
DEFAULT_TIMEOUT = 5.0


def _lock_path(state_dir: Path | str) -> Path:
    return Path(state_dir) / LOCK_NAME


def _info_path(state_dir: Path | str) -> Path:
    return Path(state_dir) / f"{LOCK_NAME}.info"


def lock_info(state_dir: Path | str) -> dict[str, Any] | None:
    """Who holds the lock, if anyone.

    Returns the recorded PID and start time, plus whether that process still
    appears to be alive — which is what turns "the lock is held" into an
    actionable message.
    """
    path = _info_path(state_dir)
    if not path.exists():
        return None
    try:
        info: dict[str, Any] = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    info["alive"] = _process_alive(int(info.get("pid", -1)))
    return info


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil

        return bool(psutil.pid_exists(pid))
    except ImportError:
        try:
            os.kill(pid, 0)
        except (OSError, ProcessLookupError):
            return False
        return True


@contextmanager
def state_lock(
    state_dir: Path | str, *, operation: str, timeout: float = DEFAULT_TIMEOUT
) -> Iterator[None]:
    """Hold the exclusive state lock for the duration of a write operation.

    Args:
        state_dir: The ``.rebasis/`` directory.
        operation: What is being done, so a conflict message can name it.
        timeout: Seconds to wait before reporting the lock held.

    Raises:
        LockHeld: When another process holds it. The message says which PID, when
            it started, and whether it still appears to be running.
    """
    from filelock import FileLock, Timeout

    directory = Path(state_dir)
    directory.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(_lock_path(directory)), timeout=timeout)

    try:
        lock.acquire()
    except Timeout as exc:
        existing = lock_info(directory)
        if existing and not existing.get("alive", True):
            hint = (
                f"The holder (PID {existing.get('pid')}) is no longer running, so "
                f"this is a stale lock. Remove {_lock_path(directory)} and retry. "
                "rebasis does not break locks by itself: guessing that a process "
                "is dead and being wrong means two writers in one manifest."
            )
        elif existing:
            hint = (
                f"PID {existing.get('pid')} has been running {existing.get('operation')} "
                f"since {existing.get('started_utc')}. Wait for it, or stop it."
            )
        else:
            hint = "Another rebasis process is writing to this state directory."
        raise LockHeld(
            f"Could not take the state lock for {operation}.",
            hint=hint,
            context={"state": operation},
            cause=exc,
        ) from exc

    import datetime

    _info_path(directory).write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "operation": operation,
                "started_utc": datetime.datetime.now(tz=datetime.UTC).isoformat(),
                "started_monotonic": time.monotonic(),
            }
        )
    )
    try:
        yield
    finally:
        _info_path(directory).unlink(missing_ok=True)
        lock.release()
