"""Determinism and numerical consistency.

PyTorch states plainly that results may not be reproducible between CPU and GPU
even with identical seeds. Rather than hide that, rebasis records what it ran on
and compares accordingly.

**TF32 is the silent one.** On Ampere and later, matmul may quietly use TF32 —
fewer mantissa bits for more throughput. That is not acceptable while computing
a decision metric, and M0 measured the cost: with TF32 enabled a 4096³ matmul
deviated from CPU by a maximum absolute error of **0.106**, against **0.001**
with it disabled. Roughly 100× the deviation for 30% more throughput
(23.4 → 30.5 TFLOP/s).

So: **TF32 off on measurement paths** (``probe``, ``eval``), left to the backend
on the embedding path during migration, where the model provider's choice
already governs. The distinction is written into the audit record.

``torch.backends.cudnn.allow_tf32`` defaults to **on** while the matmul flag
defaults to off. Embedding models do not take the convolution path, so it is
currently inert — but it belongs on the disable list rather than being left to
chance.
"""

from __future__ import annotations

import os
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "deterministic_mode_enabled",
    "measurement_precision",
    "numerical_fingerprint",
]


def deterministic_mode_enabled() -> bool:
    """Whether ``REBASIS_DETERMINISTIC=1`` is set.

    On by default in tests and during ``audit replay``; off in normal use,
    because it costs throughput.
    """
    return os.environ.get("REBASIS_DETERMINISTIC", "").strip().lower() in {"1", "true", "yes"}


@contextmanager
def measurement_precision(*, deterministic: bool | None = None) -> Iterator[None]:
    """Disable TF32 and, optionally, enable full determinism.

    Wraps every path whose output feeds a decision. Restores the previous
    settings on exit so that an embedding backend's own choices are left alone
    outside this block.

    A no-op when torch is absent, which keeps call sites unconditional.
    """
    try:
        import torch
    except ImportError:
        yield
        return

    strict = deterministic_mode_enabled() if deterministic is None else deterministic
    previous_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    previous_benchmark = torch.backends.cudnn.benchmark

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if strict:
        # cuBLAS needs this set before the handle is created to make reductions
        # reproducible; setting it late is silently ineffective.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        # Best-effort: some operations have no deterministic implementation, and
        # `warn_only` keeps those from aborting a run that is otherwise fine.
        with suppress(Exception):
            torch.use_deterministic_algorithms(mode=True, warn_only=True)

    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul
        torch.backends.cudnn.allow_tf32 = previous_cudnn
        torch.backends.cudnn.benchmark = previous_benchmark
        if strict:
            with suppress(Exception):
                torch.use_deterministic_algorithms(mode=False)


def numerical_fingerprint() -> dict[str, Any]:
    """What the audit record needs to interpret a comparison.

    Without this, a replay that produces different numbers cannot distinguish a
    regression from a change of hardware — and reporting a cross-device
    comparison as though it were same-device leads the reader to the wrong
    conclusion.
    """
    fingerprint: dict[str, Any] = {
        "dtype": "float32",
        "deterministic_mode": deterministic_mode_enabled(),
    }
    try:
        import torch
    except ImportError:
        fingerprint["torch_version"] = None
        return fingerprint

    fingerprint["torch_version"] = str(torch.__version__)
    fingerprint["tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
    fingerprint["tf32_cudnn"] = bool(torch.backends.cudnn.allow_tf32)
    if torch.cuda.is_available():
        fingerprint["device_name"] = torch.cuda.get_device_name(0)
        fingerprint["cuda_version"] = str(torch.version.cuda)
    return fingerprint
