"""`rebasis doctor --calibrate`, and the fallback that makes a partial one safe.

`rebasis.compute.thresholds` records speedups measured on one GPU against one
CPU, and says so — a faster host narrows every one of them. `--calibrate`
measures the machine in front of it instead.

It cannot measure everything. `embed` dominates a probe and needs a model, and a
diagnostic that downloaded 400 MB to answer a question nobody asked would be a
bad citizen. So it records what it timed and nothing else, which puts the weight
on :func:`worth_accelerating` falling back **per key**: reading an absent
``embed`` as a zero would turn off the one operation that matters, on a machine
that had just been measured and found fast.

The timings themselves are hardware and are not asserted here. What is asserted
is everything around them — that a partial calibration is safe, that the file
round-trips, and that the one path in `doctor` that writes only writes when it
is asked to by name.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from rebasis.cli import app
from rebasis.compute.thresholds import (
    MEASURED_SPEEDUPS,
    WORTH_IT,
    Calibration,
    calibration_path,
    load_calibration,
    worth_accelerating,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

runner = CliRunner()
WIDE = {"COLUMNS": "220"}


class TestAPartialCalibrationIsSafe:
    def test_an_unmeasured_key_falls_back_to_the_reference(self) -> None:
        """The bug this rule exists to stop.

        `--calibrate` cannot time `embed`, so a real calibration is missing it.
        Replacing the whole table with a partial one would read that absence as
        "not worth accelerating" and move the dominant cost of a probe back onto
        the CPU — on a machine that had just been measured and found fast.
        """
        partial = Calibration(device="cuda:0", speedups={"knn": 31.4})

        assert worth_accelerating("embed", partial) is True
        assert worth_accelerating("embed", partial) == worth_accelerating("embed", None)

    def test_a_measured_key_wins_over_the_reference(self) -> None:
        """Otherwise the measurement would be decoration."""
        slow = Calibration(device="mps", speedups={"knn": 1.1})

        assert MEASURED_SPEEDUPS["knn"] >= WORTH_IT
        assert worth_accelerating("knn", slow) is False

    def test_an_operation_nobody_measured_stays_on_the_cpu(self) -> None:
        """The rule the whole module runs on: the burden is on an operation to
        show it benefits, not on the CPU to defend itself."""
        assert worth_accelerating("something_nobody_timed", None) is False
        assert worth_accelerating("something_nobody_timed", Calibration("cuda", {})) is False


class TestTheFileRoundTrips:
    def test_what_was_written_reads_back(self, tmp_path: Path) -> None:
        calibration = Calibration(
            device="cuda:0",
            speedups={"knn": 31.4, "fit_mlp": 7.5},
            measured_utc="2026-08-26T16:39:35+00:00",
            host="somewhere",
            notes={"knn": {"documents": 20000}},
        )
        calibration_path(tmp_path).write_text(json.dumps(calibration.to_dict()), encoding="utf-8")

        loaded = load_calibration(tmp_path)

        assert loaded is not None
        assert loaded.speedups == calibration.speedups
        assert loaded.notes == calibration.notes
        assert loaded.host == "somewhere"

    def test_the_shape_of_the_measurement_survives(self, tmp_path: Path) -> None:
        """A ratio without its configuration is not reproducible, and this file
        outlives the terminal it was printed in."""
        written = Calibration(device="cuda", speedups={"knn": 9.0}, notes={"knn": {"dim": 768}})
        calibration_path(tmp_path).write_text(json.dumps(written.to_dict()), encoding="utf-8")

        loaded = load_calibration(tmp_path)

        assert loaded is not None
        assert loaded.notes["knn"]["dim"] == 768

    def test_an_unreadable_file_is_absent_rather_than_fatal(self, tmp_path: Path) -> None:
        """It is a cache of a measurement, and the measurement can be taken
        again. Raising here would break `doctor` for someone whose problem is
        that their disk is corrupt."""
        calibration_path(tmp_path).write_text("{not json", encoding="utf-8")

        assert load_calibration(tmp_path) is None

    def test_no_file_is_no_calibration(self, tmp_path: Path) -> None:
        assert load_calibration(tmp_path) is None


class TestTheCommandWritesOnlyWhenAsked:
    def test_plain_doctor_writes_nothing(self, tmp_path: Path) -> None:
        """`doctor` is what somebody runs when they are already confused, and
        the last command that should be able to change anything."""
        result = runner.invoke(app, ["doctor", "--state-dir", str(tmp_path)], env=WIDE)

        assert result.exit_code == 0, result.output
        assert not calibration_path(tmp_path).exists()

    def test_json_carries_the_key_either_way(self, tmp_path: Path) -> None:
        """A script reading a bug report should not have to know which flags the
        reporter passed — the same rule `store` follows."""
        result = runner.invoke(app, ["doctor", "--json", "--state-dir", str(tmp_path)], env=WIDE)

        payload = json.loads(result.stdout)
        assert payload["calibration"] is None

    def test_with_no_accelerator_it_says_so_and_writes_nothing(self, tmp_path: Path) -> None:
        """A calibration of a CPU against itself is 1.0 by construction, and
        writing it would overwrite the reference table with an identity."""
        result = runner.invoke(
            app, ["doctor", "--calibrate", "--state-dir", str(tmp_path)], env=WIDE
        )

        assert result.exit_code == 0, result.output
        if "No accelerator" in result.output:
            assert not calibration_path(tmp_path).exists()
        else:  # pragma: no cover - only on a machine that has one
            assert calibration_path(tmp_path).exists()


class TestItRecordsWhatItMeasured:
    def test_a_measurement_that_raises_is_skipped_rather_than_fatal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`doctor` is run by people whose environment is already broken, so a
        timing that blows up has to be a missing row rather than a stack trace.
        """
        from rebasis.cli import doctor as doctor_module

        class FakeDevice:
            is_cpu = False

            def __str__(self) -> str:
                return "cuda:0"

        def explode(_: Any) -> float:
            msg = "no kernel for you"
            raise RuntimeError(msg)

        monkeypatch.setattr(doctor_module, "_time_knn", explode)
        monkeypatch.setattr(doctor_module, "_time_mlp", lambda _: 4.0)
        monkeypatch.setattr("rebasis.compute.resolve_device", lambda _: FakeDevice())
        monkeypatch.setattr("rebasis.compute.torch_available", lambda: True)

        calibration = doctor_module._run_calibration(tmp_path)

        assert calibration is not None
        assert "knn" not in calibration.speedups
        assert calibration.speedups["fit_mlp"] == 4.0

    def test_nothing_measurable_writes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rebasis.cli import doctor as doctor_module

        class FakeDevice:
            is_cpu = False

            def __str__(self) -> str:
                return "cuda:0"

        def explode(_: Any) -> float:
            msg = "nothing works here"
            raise RuntimeError(msg)

        monkeypatch.setattr(doctor_module, "_time_knn", explode)
        monkeypatch.setattr("rebasis.compute.resolve_device", lambda _: FakeDevice())
        monkeypatch.setattr("rebasis.compute.torch_available", lambda: False)

        assert doctor_module._run_calibration(tmp_path) is None
        assert not calibration_path(tmp_path).exists()

    def test_it_writes_into_the_state_directory_and_nowhere_else(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rebasis.cli import doctor as doctor_module

        class FakeDevice:
            is_cpu = False

            def __str__(self) -> str:
                return "cuda:0"

        monkeypatch.setattr(doctor_module, "_time_knn", lambda _: 12.0)
        monkeypatch.setattr("rebasis.compute.resolve_device", lambda _: FakeDevice())
        monkeypatch.setattr("rebasis.compute.torch_available", lambda: False)

        state = tmp_path / "state"
        doctor_module._run_calibration(state)

        written = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file())
        assert [str(p) for p in written] == ["state/calibration.json"]
