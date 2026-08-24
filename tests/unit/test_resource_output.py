"""Nothing a measurement dependency prints may reach stdout.

`rebasis probe --json` promises that stdout holds the decision and nothing else,
which is the whole reason a script can pipe it into `jq`. That promise is only
as good as the quietest dependency on the path.

Zeus, the optional energy backend, breaks it: probing for an AMD SMI library
prints the dlopen failure straight to stdout, so on a CI runner with the CPU
torch build the JSON arrived behind

    /opt/rocm/lib/libamd_smi.so: cannot open shared object file

Silencing its loggers was not enough, because that line never went through
logging. These tests install a stand-in that prints the same way, so the
guarantee is checked on every machine rather than only on one that has zeus.
"""

from __future__ import annotations

import sys
import types

import pytest

from rebasis.observability.resources import measure_resources

pytestmark = pytest.mark.unit


def _noisy_zeus(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fake `zeus.monitor` that narrates to stdout, as the real one does.

    The prints are the fixture, not a debugging leftover: what is under test is
    that they cannot escape.
    """
    print("importing zeus: /opt/rocm/lib/libamd_smi.so: cannot open")  # noqa: T201

    class FakeMonitor:
        def __init__(self) -> None:
            print("Unable to find libamd_smi.so")  # noqa: T201

        def begin_window(self, _name: str) -> None:
            print("[zeus] begin window")  # noqa: T201

        def end_window(self, _name: str) -> object:
            print("[zeus] end window")  # noqa: T201
            return types.SimpleNamespace(total_energy=0.0)

    module = types.ModuleType("zeus.monitor")
    module.ZeusMonitor = FakeMonitor  # type: ignore[attr-defined]
    package = types.ModuleType("zeus")
    monkeypatch.setitem(sys.modules, "zeus", package)
    monkeypatch.setitem(sys.modules, "zeus.monitor", module)


class TestStdoutStaysClean:
    def test_the_energy_backend_cannot_print_to_stdout(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The failure this reproduces put a dlopen error in front of the JSON."""
        _noisy_zeus(monkeypatch)
        capsys.readouterr()

        with measure_resources(device="cpu"):
            pass

        assert capsys.readouterr().out == ""

    def test_the_measurement_still_happens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Silencing the narration must not silence the numbers with it."""
        _noisy_zeus(monkeypatch)

        with measure_resources(device="cpu", blas_threads=4) as usage:
            pass

        assert usage.summary.device == "cpu"
        assert usage.summary.blas_threads == 4
        assert usage.summary.wall_seconds >= 0.0
        # Zero joules is Zeus saying it could not measure, not a measurement.
        assert usage.summary.energy_wh is None
