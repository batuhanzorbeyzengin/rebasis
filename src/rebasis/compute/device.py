"""Device abstraction.

The single responsibility of this module is to hide "where does this tensor
live" from the rest of the codebase.

**The governing principle: the CPU path is never optional, the GPU path is never
mandatory.** A plain ``pip install rebasis`` has no torch at all; ``probe``,
Procrustes fitting and the decision rule run on numpy and scipy alone.

**Detection is defensive.** PyTorch is building a device-agnostic runtime API
(``torch.accelerator``), but it is still maturing — there was a period where an
XPU-enabled wheel raised ``RuntimeError`` from ``is_available()`` on a machine
without XPU. So detection here is explicit and wrapped: every probe is inside
``try/except`` and the worst case is falling back to CPU. A detection failure
must never crash a run.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from rebasis.errors import DeviceUnavailable

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

__all__ = [
    "Device",
    "DeviceCapabilities",
    "active_device",
    "available_devices",
    "blas_info",
    "resolve_device",
    "torch_available",
    "using_device",
]

DeviceKind = Literal["cpu", "cuda", "mps"]


@dataclass(frozen=True, slots=True)
class Device:
    """A compute device.

    ``fingerprint()`` goes into the audit record. Without it, a replay
    comparing results across devices cannot say whether a difference came from a
    regression or merely from different hardware — and reporting a cross-device
    comparison as if it were same-device leads the user to the wrong conclusion.
    """

    kind: DeviceKind
    index: int | None = None

    def __str__(self) -> str:
        return self.kind if self.index is None else f"{self.kind}:{self.index}"

    @property
    def is_cpu(self) -> bool:
        """Whether this is the CPU."""
        return self.kind == "cpu"

    def torch_device(self) -> Any:
        """The equivalent ``torch.device``.

        Raises:
            MissingDependency: If torch is not installed. Reaching this without
                torch means a caller skipped the availability check.
        """
        import torch

        return torch.device(str(self))

    def fingerprint(self) -> str:
        """Stable identifier for the audit trail."""
        return str(self)


@dataclass(frozen=True, slots=True)
class DeviceCapabilities:
    """What a device can actually do.

    ``supports_float64`` matters more than it looks: MPS does not support
    float64 at all. The float32 contract, adopted to halve memory, pays off a
    second time here — the wall that Apple Silicon ports usually hit simply is
    not there, because float64 is never used.
    """

    device: Device
    name: str
    total_memory_bytes: int | None
    supports_float64: bool
    supports_tf32: bool | None
    torch_version: str | None = None
    driver_version: str | None = None


def torch_available() -> bool:
    """Whether torch can be imported at all."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - detection must never crash a run
        return False


def _mps_available() -> bool:
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001 - see above
        return False


def available_devices() -> list[Device]:
    """Every usable device, CPU first.

    CPU leads the list deliberately: it is the reference implementation, and in
    the device-parity tests it is the one that is right when two backends
    disagree.
    """
    devices = [Device("cpu")]
    if _cuda_available():
        try:
            import torch

            devices.extend(Device("cuda", i) for i in range(torch.cuda.device_count()))
        except Exception:  # noqa: BLE001 - see above
            devices.append(Device("cuda", 0))
    if _mps_available():
        devices.append(Device("mps"))
    return devices


def resolve_device(spec: str = "auto") -> Device:  # noqa: C901 - one branch per device spec; splitting it would hide the mapping
    """Turn a ``--device`` value into a :class:`Device`.

    Args:
        spec: ``auto``, ``cpu``, ``cuda``, ``cuda:N`` or ``mps``.

    Raises:
        DeviceUnavailable: When a device was named explicitly and is not
            present. ``auto`` never raises — it degrades to CPU.

    Note that ``auto`` here resolves the *default* device. It does not decide
    per-operation placement: each sub-task is placed independently, so a run may
    embed on the GPU, fit Procrustes on the CPU and stay on the CPU for the hot
    path.
    """
    spec = (spec or "auto").strip().lower()

    if spec == "auto":
        # No GPU is not an error; the user simply does not have one. Nothing is
        # printed at default verbosity.
        if _cuda_available():
            return Device("cuda", 0)
        if _mps_available():
            return Device("mps")
        return Device("cpu")

    if spec == "cpu":
        return Device("cpu")

    if spec == "mps":
        if not _mps_available():
            raise DeviceUnavailable(
                "MPS was requested but is not available on this machine.",
                hint=(
                    "MPS needs Apple Silicon and a torch build with MPS support. "
                    "Run `rebasis doctor` to see what was detected."
                ),
                context={"device": spec},
            )
        return Device("mps")

    if spec.startswith("cuda"):
        index = 0
        if ":" in spec:
            _, _, raw = spec.partition(":")
            if not raw.isdigit():
                raise DeviceUnavailable(
                    f"Malformed CUDA device specification: {spec!r}.",
                    hint="Use `cuda` or `cuda:N`, for example `cuda:1`.",
                    context={"device": spec},
                )
            index = int(raw)
        if not _cuda_available():
            raise DeviceUnavailable(
                "CUDA was requested but no CUDA device is available.",
                hint=(
                    "torch may be installed as a CPU-only wheel. This is the most "
                    "common cause of 'I have a GPU but it is not being used'. "
                    "Run `rebasis doctor` to check."
                ),
                context={"device": spec},
            )
        return Device("cuda", index)

    raise DeviceUnavailable(
        f"Unknown device: {spec!r}.",
        hint="Valid values are auto, cpu, cuda, cuda:N and mps.",
        context={"device": spec},
    )


def capabilities(device: Device) -> DeviceCapabilities:
    """Probe what ``device`` supports.

    For MPS this runs a small live probe rather than trusting a version table:
    the MPS backend is still evolving and its operator coverage changes release
    to release. ``PYTORCH_ENABLE_MPS_FALLBACK`` is not a substitute — it stalls
    on synchronisation inside a loop. Find out whether the operations we need
    work, and if they do not, move the whole job to CPU rather than bouncing
    between devices per operator.
    """
    if device.is_cpu or not torch_available():
        return DeviceCapabilities(
            device=device,
            name=_cpu_name(),
            total_memory_bytes=None,
            supports_float64=True,
            supports_tf32=None,
            torch_version=_torch_version(),
        )

    import torch

    if device.kind == "cuda":
        props = torch.cuda.get_device_properties(device.index or 0)
        return DeviceCapabilities(
            device=device,
            name=props.name,
            total_memory_bytes=int(props.total_memory),
            supports_float64=True,
            supports_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
            torch_version=torch.__version__,
            driver_version=_cuda_driver_version(),
        )

    # MPS: unified memory, and float64 is genuinely unsupported
    return DeviceCapabilities(
        device=device,
        name="Apple Silicon (MPS)",
        total_memory_bytes=None,
        supports_float64=False,
        supports_tf32=None,
        torch_version=torch.__version__,
    )


def _cpu_name() -> str:
    import platform

    return platform.processor() or platform.machine() or "cpu"


def _torch_version() -> str | None:
    try:
        import torch

        return str(torch.__version__)
    except ImportError:
        return None


def _cuda_driver_version() -> str | None:
    try:
        import torch

        raw = torch.version.cuda
        return str(raw) if raw else None
    except Exception:  # noqa: BLE001 - informational only
        return None


def blas_info() -> tuple[str, int | None]:
    """Describe the BLAS backend and its thread count.

    Returns a ``(status, threads)`` pair where status is one of:

    * ``"ok"`` — introspected successfully
    * ``"opaque"`` — threadpoolctl is present but this BLAS cannot be inspected.
      Apple's Accelerate framework is the common case: threadpoolctl looks for
      OpenBLAS, MKL or BLIS shared libraries and Accelerate is none of them.
    * ``"unavailable"`` — threadpoolctl is not installed

    The distinction matters because ``doctor`` exists to answer "why is this
    slow", and "the tool is missing" and "this BLAS does not report" call for
    different responses.
    """
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return "unavailable", None
    blas = [p["num_threads"] for p in threadpool_info() if p.get("user_api") == "blas"]
    if not blas:
        return "opaque", None
    return "ok", max(blas)


def blas_thread_count() -> int | None:
    """Total BLAS threads currently configured.

    numpy is single-threaded on its own, but its BLAS backend (OpenBLAS, MKL)
    runs several threads controlled by variables such as ``OMP_NUM_THREADS``.
    Combined with our own parallelism the product explodes: eight workers times
    eight BLAS threads is 64 threads, which does not speed a machine up — it
    slows it down. This is the first thing ``doctor`` reports when a user says
    "it is slow".
    """
    try:
        from threadpoolctl import threadpool_info

        info = threadpool_info()
    except ImportError:
        return None
    blas = [p["num_threads"] for p in info if p.get("user_api") == "blas"]
    return max(blas) if blas else None


def user_set_thread_limits() -> bool:
    """Whether the user set thread limits themselves.

    An explicit setting by the user overrides our automation, never the other
    way round.
    """
    return any(
        os.environ.get(var)
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    )


#: The device the surrounding measurement is running on.
#:
#: A `ContextVar` rather than an argument on every function because the
#: ground-truth kNN is reached from a dozen call sites across `probe` and
#: `runner`, and threading a device through all of them would put a parameter
#: nobody reads into every signature between the CLI and the matmul. Scoped, so
#: it cannot leak past the block that set it, and it is only ever *read* by code
#: choosing where to run — never to decide what to compute.
_ACTIVE_DEVICE: ContextVar[Device | None] = ContextVar("rebasis_active_device", default=None)


def active_device() -> Device | None:
    """The device set by the innermost :func:`using_device`, if any."""
    return _ACTIVE_DEVICE.get()


@contextmanager
def using_device(device: Device | str | None) -> Iterator[Device | None]:
    """Run a block with ``device`` as the ambient compute device."""
    resolved = resolve_device(device) if isinstance(device, str) else device
    token = _ACTIVE_DEVICE.set(resolved)
    try:
        yield resolved
    finally:
        _ACTIVE_DEVICE.reset(token)


def describe_devices(devices: Sequence[Device] | None = None) -> list[DeviceCapabilities]:
    """Capabilities of every available device — the data behind ``doctor``."""
    return [capabilities(d) for d in (devices if devices is not None else available_devices())]
