"""Compute abstraction: devices and backends.

The layer contract lets ``compute`` depend only on ``errors``, ``types`` and
``observability``. torch is optional and imported lazily: a plain
``pip install rebasis`` has no torch at all, so importing this package must never
pull it in.
"""

from __future__ import annotations

from rebasis.compute.arrays import l2_normalize, pad_or_truncate
from rebasis.compute.base import (
    ACCELERATOR_THRESHOLDS,
    CPU_ONLY_OPERATIONS,
    ComputeBackend,
    should_use_accelerator,
)
from rebasis.compute.determinism import (
    deterministic_mode_enabled,
    measurement_precision,
    numerical_fingerprint,
)
from rebasis.compute.device import (
    Device,
    DeviceCapabilities,
    active_device,
    available_devices,
    blas_info,
    blas_thread_count,
    capabilities,
    describe_devices,
    resolve_device,
    torch_available,
    user_set_thread_limits,
    using_device,
)
from rebasis.compute.numpy_backend import NumpyBackend
from rebasis.compute.search import top_k_search
from rebasis.compute.thresholds import (
    MEASURED_SPEEDUPS,
    Calibration,
    calibration_path,
    load_calibration,
    worth_accelerating,
)


def get_backend(device: Device | None = None) -> ComputeBackend:
    """Return the right backend for a device.

    CPU always resolves to :class:`NumpyBackend`, even when torch is
    installed: it is the reference implementation, and on CPU torch adds an
    import and a conversion for no gain.
    """
    resolved = device or Device("cpu")
    if resolved.is_cpu:
        return NumpyBackend(resolved)
    from rebasis.compute.torch_backend import TorchBackend

    return TorchBackend(resolved)


__all__ = [
    "ACCELERATOR_THRESHOLDS",
    "CPU_ONLY_OPERATIONS",
    "MEASURED_SPEEDUPS",
    "Calibration",
    "ComputeBackend",
    "Device",
    "DeviceCapabilities",
    "NumpyBackend",
    "active_device",
    "available_devices",
    "blas_info",
    "blas_thread_count",
    "calibration_path",
    "capabilities",
    "describe_devices",
    "deterministic_mode_enabled",
    "get_backend",
    "l2_normalize",
    "load_calibration",
    "measurement_precision",
    "numerical_fingerprint",
    "pad_or_truncate",
    "resolve_device",
    "should_use_accelerator",
    "top_k_search",
    "torch_available",
    "user_set_thread_limits",
    "using_device",
    "worth_accelerating",
]
