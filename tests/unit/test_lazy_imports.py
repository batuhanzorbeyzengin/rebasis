"""Optional dependencies must stay optional.

These tests check a runtime property that static analysis cannot express: that
importing rebasis does not drag in torch or OpenTelemetry. import-linter sees a
function-body ``import torch`` as a dependency and cannot tell laziness from
eagerness, so the check lives here where the real behaviour is observable.

Why it matters: ``pip install rebasis`` ships no torch at all. If an import
crept up to module level, the package would break for every user without torch —
and it would break at import time, before any error message could help them.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# Each entry: (module to import, dependency that must NOT be pulled in)
LAZY_CASES = [
    ("rebasis", "torch"),
    ("rebasis", "opentelemetry"),
    ("rebasis.types", "torch"),
    ("rebasis.errors", "torch"),
    ("rebasis.observability", "torch"),
    ("rebasis.observability", "opentelemetry"),
    ("rebasis.compute", "torch"),
    ("rebasis.compute.device", "torch"),
    # The serving hot path must not import torch.
    ("rebasis.serve", "torch"),
    ("rebasis.serve.bridge", "torch"),
    ("rebasis.core", "torch"),
]


@pytest.mark.unit
@pytest.mark.parametrize(("module", "forbidden"), LAZY_CASES)
def test_import_does_not_pull_optional_dependency(module: str, forbidden: str) -> None:
    """Importing ``module`` must leave ``forbidden`` out of ``sys.modules``.

    Run in a subprocess: within the test process the dependency may already have
    been imported by another test, which would make this pass for the wrong
    reason.
    """
    code = (
        f"import sys; import {module}; "
        f"assert {forbidden!r} not in sys.modules, "
        f"'{module} imported {forbidden} at module level'"
    )
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.unit
def test_device_detection_works_without_torch() -> None:
    """Device resolution must not fail when torch is absent.

    Detection is defensive by design: the worst outcome of a failed probe is
    falling back to CPU, never a crash.
    """
    from rebasis.compute import available_devices, resolve_device

    device = resolve_device("auto")
    assert device.kind in {"cpu", "cuda", "mps"}
    devices = available_devices()
    assert devices[0].kind == "cpu", "CPU must lead the list: it is the reference backend"
