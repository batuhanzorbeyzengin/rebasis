"""Residual MLP adapter. Requires torch.

``g(x) = xW + b + W₂·GELU(xW₁ + b₁) + b₂`` with a hidden width of 256.

"Residual" describes the design, not just the shape: the linear term ``W`` is
initialised from the closed-form ridge solution and the last MLP layer starts at
zero, so training **begins exactly at the linear solution** and learns only the
residual. It converges far faster than training from scratch and, in practice,
cannot end up below the linear solution in a bad local minimum.

Three things M0 measured about this adapter, all of which shape how it should be
used:

* **Quality is not distinguishable from centred Procrustes.** Mean T1 ARR 0.837
  versus 0.834 — far inside the ±0.042 measurement uncertainty.
* **It is the slowest on the hot path by a wide margin.** 91 µs for a single
  query at d=768 on a cloud vCPU, against 34 µs for Procrustes: six times the
  15 µs budget. The cost is per-call numpy overhead (five operations versus one),
  not arithmetic — at batch 256 it drops to ~11 µs per query.
* **It is the largest.** 3.76 MB at d=768. The projected figure was 1.6 MB, which
  counts only ``W₁ + W₂`` and omits the linear ``Wx`` term from the same formula.

So it is offered, and ``auto`` will pick it when it genuinely wins — but centred
Procrustes is the better default.

torch is imported lazily and only here: ``import rebasis`` must stay torch-free,
and the hot path converts the weights to numpy so that serving never touches
torch either.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import scipy.special

from rebasis.core.base import BaseAdapter, check_fit_inputs
from rebasis.core.linear import LinearAdapter
from rebasis.errors import FitFailed, MissingDependency
from rebasis.types import AdapterKind, FloatArray, as_float32

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self

__all__ = ["DEFAULT_HIDDEN", "ResidualMLPAdapter"]

DEFAULT_HIDDEN = 256
_SQRT_2 = float(np.sqrt(2.0))


class ResidualMLPAdapter(BaseAdapter):
    """A linear map plus a learned residual MLP."""

    kind: ClassVar[AdapterKind] = "residual_mlp"

    def __init__(
        self, state: Mapping[str, FloatArray], *, input_dim: int, output_dim: int, hidden: int
    ) -> None:
        super().__init__(input_dim=input_dim, output_dim=output_dim, config={"hidden": hidden})
        self._state = {k: as_float32(v) for k, v in state.items()}
        self.hidden = hidden

    @classmethod
    def fit(  # noqa: PLR0913 - each argument is a distinct training knob
        cls,
        src: FloatArray,
        dst: FloatArray,
        *,
        hidden: int = DEFAULT_HIDDEN,
        epochs: int = 60,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        val_fraction: float = 0.1,
        patience: int = 8,
        device: str = "cpu",
        seed: int = 0,
    ) -> Self:
        """Train the residual on top of the closed-form linear solution.

        Raises:
            MissingDependency: When torch is not installed.
            FitFailed: When training produces non-finite weights.
        """
        check_fit_inputs(src, dst)
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise MissingDependency(
                "The residual MLP adapter needs torch.",
                hint=(
                    'Install it with `pip install "rebasis[torch]"`, or use '
                    "--method procrustes_centered, which measured as equivalent "
                    "in quality at half the memory and a third of the latency."
                ),
                context={"adapter_type": "residual_mlp"},
                cause=exc,
            ) from exc

        torch.manual_seed(seed)
        dev = torch.device(device)
        base = LinearAdapter.fit(src, dst)
        d_in, d_out = base.input_dim, base.output_dim

        # MPS trap: several operations silently return zeros or NaN when
        # the underlying tensor is not contiguous. `.contiguous()` at every entry
        # point costs nothing and saves a week of debugging.
        x_all = torch.from_numpy(as_float32(src)).to(dev).contiguous()
        y_all = torch.from_numpy(as_float32(dst)).to(dev).contiguous()

        n = x_all.shape[0]
        n_val = max(1, int(n * val_fraction))
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(n, generator=generator).to(dev)
        val_idx, train_idx = permutation[:n_val], permutation[n_val:]
        x_train, y_train = x_all[train_idx].contiguous(), y_all[train_idx].contiguous()
        x_val, y_val = x_all[val_idx].contiguous(), y_all[val_idx].contiguous()

        linear = nn.Linear(d_in, d_out).to(dev)
        with torch.no_grad():
            # nn.Linear computes y = xWᵀ + b, so torch stores (d_out, d_in)
            # where ours is (d_in, d_out).
            linear.weight.copy_(torch.from_numpy(base.weight.T).to(dev))
            linear.bias.copy_(torch.from_numpy(base.bias).to(dev))

        mlp = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out)).to(dev)
        with torch.no_grad():
            # Zeroing the last layer makes g(x) start out exactly equal to the
            # linear solution, so training begins from a known-good point.
            mlp[-1].weight.zero_()
            mlp[-1].bias.zero_()

        optimiser = torch.optim.Adam(
            list(linear.parameters()) + list(mlp.parameters()), lr=learning_rate
        )
        loss_fn = nn.MSELoss()

        best_loss = float("inf")
        best: dict[str, Any] = {}
        stale = 0
        n_train = x_train.shape[0]

        for _epoch in range(epochs):
            linear.train()
            mlp.train()
            order = torch.randperm(n_train, generator=generator).to(dev)
            for start in range(0, n_train, batch_size):
                index = order[start : start + batch_size]
                optimiser.zero_grad(set_to_none=True)
                loss_fn(linear(x_train[index]) + mlp(x_train[index]), y_train[index]).backward()
                optimiser.step()

            linear.eval()
            mlp.eval()
            with torch.no_grad():
                validation = loss_fn(linear(x_val) + mlp(x_val), y_val).item()
            if validation < best_loss - 1e-7:
                best_loss = validation
                best = {
                    "W": linear.weight.detach().cpu().numpy().T.copy(),
                    "b": linear.bias.detach().cpu().numpy().copy(),
                    "W1": mlp[0].weight.detach().cpu().numpy().T.copy(),
                    "b1": mlp[0].bias.detach().cpu().numpy().copy(),
                    "W2": mlp[2].weight.detach().cpu().numpy().T.copy(),
                    "b2": mlp[2].bias.detach().cpu().numpy().copy(),
                }
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

        if not best or not all(np.isfinite(v).all() for v in best.values()):
            raise FitFailed(
                "Residual MLP training produced non-finite weights.",
                hint="Lower the learning rate, or use --method procrustes_centered.",
                context={"count": int(src.shape[0]), "adapter_type": "residual_mlp"},
            )

        return cls(
            {k: as_float32(np.ascontiguousarray(v)) for k, v in best.items()},
            input_dim=d_in,
            output_dim=d_out,
            hidden=hidden,
        )

    def apply(self, x: FloatArray) -> FloatArray:
        """Linear term plus the MLP residual. Numpy only — no torch."""
        state = self._state
        x = as_float32(x)
        hidden = x @ state["W1"] + state["b1"]
        # Exact erf-based GELU, matching torch's default. M0 measured the tanh
        # approximation as *slower* here (5.26 µs against 3.62 µs): fewer FLOPs
        # but more numpy calls, and at this size the call count is what costs.
        hidden = 0.5 * hidden * (1.0 + scipy.special.erf(hidden / _SQRT_2))
        return as_float32(x @ state["W"] + state["b"] + hidden @ state["W2"] + state["b2"])

    def state_dict(self) -> dict[str, FloatArray]:
        """All six weight tensors."""
        return dict(self._state)

    @classmethod
    def from_state_dict(cls, state: Mapping[str, FloatArray], config: Mapping[str, Any]) -> Self:
        """Reconstruct from stored weights."""
        return cls(
            state,
            input_dim=int(config["input_dim"]),
            output_dim=int(config["output_dim"]),
            hidden=int(config["hidden"]),
        )
