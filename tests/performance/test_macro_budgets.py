"""Macro benchmarks against the stated fit, search and load budgets.

Marked ``slow``: excluded from the default run, executed nightly on the
project's own host, where the hardware is known and the numbers mean something.

**The gate is 120% of the budget, not 100%.** A budget is a design
target measured on one machine; a run at 105% is information, not a defect. What
120% catches is the thing worth catching — an algorithmic change that moved the
cost by an order of magnitude — while leaving room for a slower disk or a busy
host. Nightly it warns; before a release it blocks.

Wall clock never blocks a PR. These numbers exist as a time series; what gates
a PR is the `memory` layer, which asserts peak allocation rather than seconds
and therefore measures the same on a shared runner as on this host. An earlier
version of this sentence named an instruction-count benchmark as the gate. There
was none: the CodSpeed job that would have been it collected nothing — `pytest
--codspeed` gathers only tests using the `benchmark` fixture, and none did — and
it was removed. A docstring naming a gate that does not exist is worse than one
naming no gate at all, because it stops anyone from looking.

**Marked `perf` as well as `slow`, and that second marker is what makes the
sentence above true.** `slow` alone did not: CI runs `-m "not network and not
perf"`, so a `slow`-only file was on the merge path, and `ci.yml`'s own comment
gives the reason it should not have been — "the perf layer asserts timing, and a
shared runner cannot measure timing … noise wearing a red X". That reasoning is
about what a test measures, not about which marker it happens to carry, and it
applies here exactly. Observed: under a loaded host the MLP fit blew a 180-second
budget it clears in **17 seconds** when measured alone, a 10x swing with nothing
in the code changed. `slow` still keeps it out of the default developer run;
`perf` keeps it off the merge path; `scripts/remote.sh test` runs `-m "not
network"` and so still runs it on the host, which is where its numbers mean
something.
"""

from __future__ import annotations

import gc
import time
import tracemalloc
from dataclasses import dataclass

import numpy as np
import pytest

from rebasis.core import fit_candidates, l2_normalize
from rebasis.types import EncodingProfile

pytestmark = [pytest.mark.slow, pytest.mark.perf]

#: The gate sits at 120% of the stated budget.
GATE = 1.20

#: The reference dimension for every fit budget.
DIM = 768

#: The budget quotes 20,000 pairs. M0 measured the curve flattening near 4,000,
#: so this is well past saturation — which is the point: the budget has to hold
#: at the size it names, not at the size we would choose.
PAIRS = 20_000


@dataclass(frozen=True, slots=True)
class Budget:
    """One row of the budget table."""

    label: str
    seconds: float
    peak_mb: float | None = None

    def gate_seconds(self) -> float:
        return self.seconds * GATE

    def gate_bytes(self) -> float:
        return (self.peak_mb or 0) * GATE * 1024**2


BUDGETS = {
    "procrustes": Budget("Adapter fit - OP", seconds=20, peak_mb=500),
    "low_rank_affine": Budget("Adapter fit - LA", seconds=90, peak_mb=600),
    "residual_mlp": Budget("Adapter fit - MLP", seconds=180, peak_mb=800),
    "auto": Budget("auto (all three plus evaluation)", seconds=360, peak_mb=800),
    "knn": Budget("Ground truth kNN, 10k x 10k chunked", seconds=30, peak_mb=300),
    "load": Budget(".rbs load, MLP + DSM", seconds=0.05, peak_mb=20),
}


def measure(call) -> tuple[float, int]:  # type: ignore[no-untyped-def]
    """Wall time and peak traced allocation for one call."""
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    try:
        call()
        elapsed = time.perf_counter() - started
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return elapsed, peak


def check(budget: Budget, elapsed: float, peak: int) -> None:
    """Report the measurement, then gate on it."""
    print(  # noqa: T201 - the time series is the point of this test
        f"\n{budget.label}: {elapsed:.2f}s (budget {budget.seconds}s), "
        f"{peak / 1024**2:.0f} MB (budget {budget.peak_mb} MB)"
    )
    assert elapsed <= budget.gate_seconds(), (
        f"{budget.label} took {elapsed:.1f}s against a {budget.seconds}s budget "
        f"(gate {budget.gate_seconds():.1f}s)"
    )
    if budget.peak_mb is not None:
        assert peak <= budget.gate_bytes(), (
            f"{budget.label} peaked at {peak / 1024**2:.0f} MB against a {budget.peak_mb} MB budget"
        )


@pytest.fixture(scope="module")
def pairs() -> tuple[np.ndarray, np.ndarray]:
    """Matched pairs at the budget's stated size and dimension."""
    rng = np.random.default_rng(20260823)
    src = l2_normalize(rng.standard_normal((PAIRS, DIM)).astype(np.float32))
    rotation = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype(np.float32)
    return src, l2_normalize(src @ rotation.T)


@pytest.mark.parametrize("method", ["procrustes", "low_rank_affine", "residual_mlp"])
def test_fitting_one_adapter_stays_within_budget(
    pairs: tuple[np.ndarray, np.ndarray], method: str
) -> None:
    src, dst = pairs

    elapsed, peak = measure(lambda: fit_candidates(src, dst, normalize=False, methods=[method]))

    check(BUDGETS[method], elapsed, peak)


def test_auto_stays_within_budget(pairs: tuple[np.ndarray, np.ndarray]) -> None:
    """Shared preprocessing is what makes this less than the sum of the three."""
    src, dst = pairs

    elapsed, peak = measure(lambda: fit_candidates(src, dst, normalize=False))

    check(BUDGETS["auto"], elapsed, peak)


def test_ground_truth_knn_stays_within_budget() -> None:
    """10k x 10k, chunked. The single most expensive thing rebasis does."""
    from rebasis.compute import top_k_search

    rng = np.random.default_rng(20260823)
    vectors = l2_normalize(rng.standard_normal((10_000, DIM)).astype(np.float32))

    elapsed, peak = measure(lambda: top_k_search(vectors, vectors, k=10))

    check(BUDGETS["knn"], elapsed, peak)


def test_loading_an_adapter_stays_within_budget(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """`.rbs` loading is budgeted at 50 ms — it sits in front of every query."""
    from rebasis.core import load_adapter, save_adapter

    rng = np.random.default_rng(20260823)
    src = l2_normalize(rng.standard_normal((2_000, DIM)).astype(np.float32))
    dst = l2_normalize(rng.standard_normal((2_000, DIM)).astype(np.float32))
    profile = EncodingProfile(model_id="m", dim=DIM)

    candidates = fit_candidates(src, dst, normalize=False, methods=["residual_mlp"])
    path = save_adapter(
        candidates[0].adapter,
        tmp_path / "a.rbs",
        direction="query_to_old",
        old_profile=profile,
        new_profile=profile,
    )

    # First load warms the import of safetensors; the budget is about reading a
    # file, not about paying for a module import once per process.
    load_adapter(path)
    elapsed, peak = measure(lambda: load_adapter(path))

    check(BUDGETS["load"], elapsed, peak)
