"""Turning a termination signal into a request to stop at a batch boundary.

A long migration does not run on a laptop with somebody watching it. It runs as
a Kubernetes ``Job``, an Airflow task or an Argo step, and all three end a
process the same way: **SIGTERM, a grace period, then SIGKILL.** Kubernetes'
default grace period is thirty seconds.

Before this module the engine had no handler for either signal. A SIGTERM was
Python's default — immediate termination — so a migration was killed wherever it
happened to be, typically mid-batch, and what saved the data was the checkpoint
rather than anything the process did on the way out. The recovery was correct;
the shutdown was not.

The mechanism it plugs into already existed. ``rebasis pause`` records a request
that the engine reads at the top of every batch, so a job stops at a boundary
rather than in the middle of one. A signal is the same request arriving from a
different direction, and this module is the adapter between them.

**Why a flag and not a database write.** A signal handler runs between two
bytecodes of whatever was executing, which may be SQLite. Writing to the manifest
from a handler risks re-entering a connection mid-statement. The handler sets a
boolean; :func:`stop_requested` is what the engine reads, next to the
manifest-backed pause request it was already reading.

**Why the CLI installs it and the engine does not.** ``signal.signal`` is
process-wide and only works on the main thread. A library that installed a
handler on import would take SIGTERM away from the application embedding it,
which is not a library's to take. :class:`~rebasis.migrate.engine.MigrationEngine`
only ever *reads* the flag, so a caller driving the engine from their own code
keeps their own signal handling and simply never sets it.

**The second signal is not caught.** The first restores the default handler
before returning, so pressing Ctrl-C twice, or a supervisor escalating, stops
the process immediately the way it always did. A graceful stop that cannot be
interrupted is a hang with better manners.

**The grace period has to outlast a batch.** A batch that takes longer than
``terminationGracePeriodSeconds`` is still SIGKILLed part-way, and no handler
changes that — the guide says to raise the period above the batch duration, or
lower ``--batch`` below it.
"""

from __future__ import annotations

import signal
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import FrameType

__all__ = ["stop_on_terminate", "stop_requested", "stop_signal_name"]

#: Set by the handler, read by the engine. A plain flag rather than a
#: `threading.Event` because nothing waits on it — the engine polls it once per
#: batch — and assignment to a module-level name is atomic under the GIL and
#: under free-threading alike.
_stop = threading.Event()

#: Which signal asked, so the run can say so rather than reporting a bare
#: "stopped". Written before the flag is set, so a reader that sees the flag
#: sees the name too.
_signal_name = ""


def stop_requested() -> bool:
    """Whether a termination signal has asked this process to stop."""
    return _stop.is_set()


def stop_signal_name() -> str:
    """The signal that asked, or ``""`` if none has."""
    return _signal_name


@contextmanager
def stop_on_terminate() -> Iterator[None]:
    """Catch SIGTERM and SIGINT once, and turn them into a stop request.

    Restores whatever handlers were installed before, on the way out and on an
    exception alike, so a caller that wraps something else around this gets its
    own handling back.

    Outside the main thread — and on a platform without one of the signals —
    installing is skipped rather than raising. The engine then behaves exactly as
    it did before this module existed, which is the correct fallback: the
    checkpoint is still the thing that makes an abrupt kill survivable.
    """
    global _signal_name  # noqa: PLW0603 - module state is the point; see the docstring
    _stop.clear()
    _signal_name = ""

    def handler(signum: int, _frame: FrameType | None) -> None:
        global _signal_name  # noqa: PLW0603 - as above
        _signal_name = signal.Signals(signum).name
        # Default first, flag second. If the batch in flight outlasts the grace
        # period the supervisor escalates, and a second signal must not land on
        # a handler that only sets a flag that is already set.
        signal.signal(signum, signal.SIG_DFL)
        _stop.set()

    installed: list[tuple[int, object]] = []
    try:
        if threading.current_thread() is threading.main_thread():
            for name in ("SIGTERM", "SIGINT"):
                signum = getattr(signal, name, None)
                if signum is None:
                    continue
                installed.append((signum, signal.getsignal(signum)))
                signal.signal(signum, handler)
        yield
    finally:
        for signum, previous in installed:
            # `getsignal` returns None for a handler set outside Python. There is
            # nothing to restore in that case, and passing None would raise.
            if previous is not None:
                signal.signal(signum, previous)  # type: ignore[arg-type]
        _stop.clear()
        _signal_name = ""
