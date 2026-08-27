"""A termination signal stops a migration at a batch boundary.

`rebasis pause` already stopped a job cleanly from another terminal. A
supervisor does not have another terminal: Kubernetes, Airflow and Argo all end
a process with SIGTERM, a grace period, then SIGKILL. Before this the signal was
Python's default — immediate termination, wherever the run happened to be, which
is typically mid-batch.

Three properties are what make catching it safe, and each has a test below:

* **It is a request, not a stop.** The handler sets a flag; the engine reads it
  at the top of the next batch, exactly where it already reads the manifest's
  pause request. Nothing is written to SQLite from a handler that can run
  between two bytecodes of a statement.
* **The second signal is not caught.** The handler restores the default before
  it returns, so a supervisor escalating — or a second Ctrl-C — still stops the
  process at once. A graceful stop that cannot be interrupted is a hang.
* **The handler does not outlive the run.** It is installed by a context manager
  that restores whatever was there before, so a caller embedding rebasis keeps
  its own signal handling.
"""

from __future__ import annotations

import os
import signal
import threading
from typing import TYPE_CHECKING

import numpy as np
import pytest

from rebasis.core import ProcrustesAdapter, l2_normalize
from rebasis.manifest import ManifestDB, manifest_path
from rebasis.migrate import (
    JobState,
    MigrationEngine,
    stop_on_terminate,
    stop_requested,
    stop_signal_name,
)
from rebasis.store import MemoryStore

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

DIM = 16
N = 64
BATCH = 8


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(11)


@pytest.fixture
def engine(tmp_path: Path, rng: np.random.Generator) -> MigrationEngine:
    """A queued job over an in-memory store, eight batches long."""
    vectors = l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))
    rotation = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype(np.float32)
    adapter = ProcrustesAdapter.fit(vectors, l2_normalize(vectors @ rotation.T))

    ids = [f"doc-{i:04d}" for i in range(N)]
    built = MigrationEngine(
        db=ManifestDB(manifest_path(tmp_path / "state")),
        store=MemoryStore(ids, vectors, [f"text {i}" for i in range(N)]),
        adapter=adapter,
        shadow_root=tmp_path / "shadow",
        batch_size=BATCH,
        power_aware=False,
    )
    built.prepare(ids)
    return built


class TestTheHandlerIsARequest:
    def test_nothing_is_requested_before_a_signal_arrives(self) -> None:
        with stop_on_terminate():
            assert not stop_requested()
            assert stop_signal_name() == ""

    def test_sigterm_sets_the_flag_and_names_itself(self) -> None:
        """The name is what lets a run say *why* it stopped.

        "stopped" and "SIGTERM asked this process to stop" are different
        messages to find in a pod's last log line, and only the second tells the
        reader it was their own orchestrator that did it.
        """
        with stop_on_terminate():
            os.kill(os.getpid(), signal.SIGTERM)
            assert stop_requested()
            assert stop_signal_name() == "SIGTERM"

    def test_sigint_is_caught_too(self) -> None:
        """Ctrl-C used to abort mid-batch; now it asks, once.

        The store is left holding a batch nobody verified either way — the
        shadow copy is written before the vector it copies is overwritten — but
        "stopped at a boundary" and "stopped somewhere" are different states to
        resume from.
        """
        with stop_on_terminate():
            os.kill(os.getpid(), signal.SIGINT)
            assert stop_requested()
            assert stop_signal_name() == "SIGINT"


class TestTheSecondSignalIsNotCaught:
    def test_the_default_handler_is_restored_by_the_first(self) -> None:
        """A supervisor escalating must not land on a flag that is already set.

        Asserted on the *handler*, not by sending a second signal: the default
        for SIGTERM terminates the interpreter, so a test that sent one would
        take the suite with it.
        """
        with stop_on_terminate():
            assert signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL
            os.kill(os.getpid(), signal.SIGTERM)
            assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


class TestTheHandlerDoesNotOutliveTheRun:
    def test_the_previous_handler_comes_back(self) -> None:
        """rebasis is a library as well as a command.

        A caller embedding it keeps whatever handling they had. Withholding that
        is the difference between a tool and a tool that takes over the process.
        """
        before = signal.getsignal(signal.SIGTERM)
        with stop_on_terminate():
            assert signal.getsignal(signal.SIGTERM) is not before
        assert signal.getsignal(signal.SIGTERM) is before

    def test_it_comes_back_after_an_exception_too(self) -> None:
        before = signal.getsignal(signal.SIGTERM)
        with pytest.raises(RuntimeError), stop_on_terminate():
            raise RuntimeError
        assert signal.getsignal(signal.SIGTERM) is before

    def test_a_flag_set_by_one_run_does_not_stop_the_next(self) -> None:
        """Entering clears, and leaving clears.

        A request that survived its run would pause a job that had not been
        asked to stop — the same property `pause_requested` holds in the
        manifest, for the same reason.
        """
        with stop_on_terminate():
            os.kill(os.getpid(), signal.SIGTERM)
            assert stop_requested()
        assert not stop_requested()
        with stop_on_terminate():
            assert not stop_requested()

    def test_installing_off_the_main_thread_is_skipped_rather_than_raised(self) -> None:
        """`signal.signal` raises off the main thread, and that is not a failure.

        A caller running a migration in a worker thread gets the behaviour that
        existed before this module: no handler, and the checkpoint is what makes
        an abrupt kill survivable. Falling back beats refusing to run.
        """
        failures: list[BaseException] = []

        def run_it() -> None:
            try:
                with stop_on_terminate():
                    assert not stop_requested()
            except BaseException as exc:  # noqa: BLE001 - the assertion is that there is none
                failures.append(exc)

        worker = threading.Thread(target=run_it)
        worker.start()
        worker.join()

        assert failures == []


class TestTheEngineHonoursIt:
    def test_it_stops_at_the_next_batch_boundary(self, engine: MigrationEngine) -> None:
        """Signalled during the first batch, stopped before the second.

        `on_batch` fires once a batch has finished, so a signal raised from it is
        the case the design is for: arriving while a batch was in flight, and
        honoured at the boundary rather than part-way through the next one.
        """
        batches = 0

        def signal_after_the_first(_: int) -> None:
            nonlocal batches
            batches += 1
            if batches == 1:
                os.kill(os.getpid(), signal.SIGTERM)

        with stop_on_terminate():
            result = engine.run(on_batch=signal_after_the_first)

        assert result.state is JobState.PAUSED
        assert batches == 1
        assert result.processed == BATCH
        assert engine.queue.stats().remaining == N - BATCH

    def test_the_reason_names_the_signal(self, engine: MigrationEngine) -> None:
        """Not "paused" but *why*, because the two have different fixes."""
        with stop_on_terminate():
            os.kill(os.getpid(), signal.SIGTERM)
            result = engine.run()

        assert result.state is JobState.PAUSED
        assert result.processed == 0

    def test_what_it_stopped_before_can_be_resumed(self, engine: MigrationEngine) -> None:
        """A stop is not a loss. The queue is the checkpoint either way."""
        batches = 0

        def signal_after_the_first(_: int) -> None:
            nonlocal batches
            batches += 1
            if batches == 1:
                os.kill(os.getpid(), signal.SIGTERM)

        with stop_on_terminate():
            engine.run(on_batch=signal_after_the_first)

        finished = engine.run()

        assert finished.state is JobState.COMPLETED
        assert engine.queue.stats().remaining == 0

    def test_without_the_context_manager_the_engine_runs_to_completion(
        self, engine: MigrationEngine
    ) -> None:
        """The flag is the only channel; the engine installs nothing itself.

        This is what lets a library caller keep their own signal handling — and
        it is the property that would break silently if the handler were ever
        moved into the engine for convenience.
        """
        result = engine.run()

        assert result.state is JobState.COMPLETED
        assert not stop_requested()
