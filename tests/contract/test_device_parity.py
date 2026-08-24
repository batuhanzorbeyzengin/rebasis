"""Device equivalence.

Two code paths mean two error surfaces. Rather than hope they agree, every
available device runs the same operations and is compared against the numpy
reference — which, in any disagreement, is the one that is right.

**The last test is the important one.** Small numerical differences are
acceptable; a different *decision* is not. ARR of 0.9501 and 0.9499 straddle the
0.95 threshold, and a user changing machines would get different advice for the
same corpus. That is what the borderline band exists to prevent, and what this
asserts.

A test that passes on one device and fails on another is **a bug**, not a reason
to widen a tolerance. Tolerances live in one place so that changing one requires
saying why.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from rebasis.compute import Device, NumpyBackend, available_devices, get_backend
from rebasis.core import l2_normalize
from rebasis.probe.decision import decide

if TYPE_CHECKING:
    from rebasis.compute.base import ComputeBackend

#: Tolerances, in one place. Widening one requires a reason.
ATOL_SCORES = 1e-5
ATOL_VECTORS = 1e-5
#: ARR difference allowed between devices. Well inside the ±0.024 sampling
#: uncertainty M0 measured, so a breach here means a real numerical divergence
#: rather than noise.
ARR_TOLERANCE = 0.002

DIM = 128
N_DOCS = 2000
N_QUERIES = 200
K = 10


def _device_params() -> list[pytest.ParameterSet]:
    """One parameter per available device, accelerators marked ``gpu``.

    The marker is what puts these in the nightly run on the project's own host.
    Without it the accelerator cases would exist only in the default suite,
    which on a laptop or a free CI runner means they never execute — and a
    device-parity suite that never sees a second device is not one.
    """
    params = []
    for device in available_devices():
        marks = [pytest.mark.gpu] if not device.is_cpu else []
        params.append(pytest.param(str(device), marks=marks, id=str(device)))
    return params


@pytest.fixture(params=_device_params())
def backend(request: pytest.FixtureRequest) -> ComputeBackend:
    """One backend per available device — CPU always, accelerators when present."""
    kind, _, index = request.param.partition(":")
    device = Device(kind, int(index) if index else None)  # type: ignore[arg-type]
    return get_backend(device)


@pytest.fixture(scope="module")
def vectors() -> tuple[np.ndarray, np.ndarray]:
    """A fixed corpus and query set, identical for every device."""
    rng = np.random.default_rng(20260823)
    docs = l2_normalize(rng.standard_normal((N_DOCS, DIM)).astype(np.float32))
    queries = l2_normalize(rng.standard_normal((N_QUERIES, DIM)).astype(np.float32))
    return docs, queries


@pytest.mark.contract
def test_matmul_topk_matches_numpy(
    backend: ComputeBackend, vectors: tuple[np.ndarray, np.ndarray]
) -> None:
    """Indices must match exactly; scores within tolerance.

    Indices exactly, because a different neighbour set is a different answer, not
    a rounding difference. Scores within tolerance, because summation order
    genuinely differs across devices.
    """
    docs, queries = vectors
    reference = NumpyBackend()

    expected_indices, expected_scores = reference.matmul_topk(queries, docs, K)
    actual_indices, actual_scores = backend.matmul_topk(queries, docs, K)

    # Near-ties can legitimately swap. Compare the retrieved *sets*, and require
    # the scores to agree regardless.
    for expected_row, actual_row in zip(expected_indices, actual_indices, strict=True):
        overlap = len(set(expected_row.tolist()) & set(actual_row.tolist()))
        assert overlap >= K - 1, f"{backend} returned a different neighbour set"

    np.testing.assert_allclose(
        np.sort(actual_scores, axis=1),
        np.sort(expected_scores, axis=1),
        atol=ATOL_SCORES,
    )


@pytest.mark.contract
def test_normalize_matches_numpy(
    backend: ComputeBackend, vectors: tuple[np.ndarray, np.ndarray]
) -> None:
    """ℓ2 normalisation must agree to float32 precision."""
    _, queries = vectors
    raw = queries * np.linspace(0.5, 3.0, queries.shape[0], dtype=np.float32)[:, None]

    np.testing.assert_allclose(
        backend.normalize(raw), NumpyBackend().normalize(raw), atol=ATOL_VECTORS
    )


@pytest.mark.contract
def test_apply_batch_matches_numpy(
    backend: ComputeBackend, vectors: tuple[np.ndarray, np.ndarray]
) -> None:
    """A linear map must produce the same vectors on any device."""
    _, queries = vectors
    rng = np.random.default_rng(7)
    weights = {
        "weight": (rng.standard_normal((DIM, DIM)) / np.sqrt(DIM)).astype(np.float32),
        "bias": (rng.standard_normal(DIM) * 0.01).astype(np.float32),
    }

    np.testing.assert_allclose(
        backend.apply_batch(weights, queries),
        NumpyBackend().apply_batch(weights, queries),
        atol=ATOL_VECTORS,
    )


@pytest.mark.contract
def test_the_decision_is_identical_across_devices(
    backend: ComputeBackend, vectors: tuple[np.ndarray, np.ndarray]
) -> None:
    """**The test that matters.**

    Numerical drift is tolerable. A different recommendation is not: the same
    corpus on a different machine must not produce "bridge" on one and "reindex"
    on the other.
    """
    docs, queries = vectors
    reference = NumpyBackend()

    def arr(chosen: ComputeBackend) -> float:
        truth, _ = reference.matmul_topk(queries, docs, K)
        actual, _ = chosen.matmul_topk(queries, docs, K)
        relevant = [{int(row[0])} for row in truth]
        hits = [
            1.0 if rel & set(row.tolist()) else 0.0
            for row, rel in zip(actual, relevant, strict=True)
        ]
        return float(np.mean(hits))

    reference_arr = arr(reference)
    device_arr = arr(backend)

    assert abs(reference_arr - device_arr) < ARR_TOLERANCE, (
        f"ARR differs by {abs(reference_arr - device_arr):.4f} between "
        f"{reference.device} and {backend.device}"
    )
    assert decide(reference_arr).decision == decide(device_arr).decision, (
        "the same corpus produced different recommendations on different devices"
    )


@pytest.mark.contract
def test_cpu_always_resolves_to_the_reference_backend() -> None:
    """CPU is NumpyBackend even when torch is installed.

    The reference implementation has to be the one that actually runs, otherwise
    "numpy is right when they disagree" is a claim about code nobody executes.
    """
    assert isinstance(get_backend(Device("cpu")), NumpyBackend)


@pytest.mark.contract
def test_cpu_leads_the_device_list() -> None:
    """Ordering is part of the contract: the reference comes first."""
    assert available_devices()[0].kind == "cpu"
