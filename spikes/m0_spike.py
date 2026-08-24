"""rebasis M0 spike — the first measurement milestone.

This file is the whole of phase M0: a single file, no package skeleton. It
answers three questions:

  1. **Is the T0 ground-truth proxy valid?** ("the backbone of the project")
     Without a real query log, ``probe`` uses documents as stand-ins for
     queries. How closely that proxy tracks the real query distribution is an
     unmeasured assumption. The only way to measure it is to run both tiers side
     by side on a corpus that has BOTH real queries AND relevance judgements.

  2. **Does CSLS improve ARR?**
     The reference work (K1) never tried hubness correction. If it helps, it is
     rebasis's first original contribution to the literature; if it does not,
     that is a result too.

  3. **Are the hot-path latency and device budgets realistic?**
     All of them are estimates. They are to be measured on real hardware
     (MacBook + g5.2xlarge) and updated; the GPU/CPU thresholds fall out of the
     same measurements.

Deliberate simplifications — these belong in M1 and would be noise in M0:
  · no error hierarchy; bare exceptions
  · no logging; output goes straight to stdout via rich
  · no audit trail
  · no store abstraction — everything is a numpy array in memory

Deliberately preserved — these affect the measurement itself:
  · the float32 contract
  · never materialising the full distance matrix; chunked matmul plus
    ``argpartition``
  · asymmetric model prefixes — get these wrong and every number is silently
    invalid
  · fit set and query set strictly disjoint
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import psutil
import scipy.linalg
import scipy.special
import scipy.stats
from rich.console import Console
from rich.table import Table

FloatArray = np.ndarray

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Measurement infrastructure
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Timing:
    """Duration and memory for one operation — the core of the resource summary."""

    label: str
    wall_seconds: float
    peak_rss_bytes: int
    n_items: int | None = None

    @property
    def per_item_ms(self) -> float | None:
        if not self.n_items:
            return None
        return self.wall_seconds * 1000.0 / self.n_items


_TIMINGS: list[Timing] = []


@contextmanager
def measure(label: str, n_items: int | None = None) -> Iterator[None]:
    """Measure wall time and peak RSS — the raw data for budget calibration."""
    proc = psutil.Process()
    # No gc.collect() before sampling: it would not improve the measurement,
    # only add noise.
    rss_before = proc.memory_info().rss
    t0 = time.perf_counter()
    try:
        yield
    finally:
        wall = time.perf_counter() - t0
        rss_after = proc.memory_info().rss
        timing = Timing(
            label=label,
            wall_seconds=wall,
            peak_rss_bytes=max(rss_before, rss_after),
            n_items=n_items,
        )
        _TIMINGS.append(timing)
        detail = f" · {timing.per_item_ms:.3f} ms/item" if timing.per_item_ms is not None else ""
        console.print(
            f"  [dim]⏱[/dim] {label}: [bold]{wall:.2f}s[/bold]"
            f" · RSS {rss_after / 1024**2:.0f} MB{detail}"
        )


def describe_host() -> dict[str, Any]:
    """Which hardware produced these numbers. Without this the numbers mean nothing."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda_available"] = torch.cuda.is_available()
        info["torch_mps_available"] = torch.backends.mps.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
            # TF32 must be off on measurement paths; record the state
            info["tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
    except ImportError:
        info["torch"] = None

    # BLAS thread inventory. The first place to look when someone says
    # "it is slow".
    try:
        from threadpoolctl import threadpool_info

        info["threadpools"] = [
            {"api": p["user_api"], "impl": p["internal_api"], "threads": p["num_threads"]}
            for p in threadpool_info()
        ]
    except ImportError:
        info["threadpools"] = None
    return info


# ─────────────────────────────────────────────────────────────────────────────
# Vector helpers — the float32 contract
# ─────────────────────────────────────────────────────────────────────────────


def as_f32(x: np.ndarray) -> FloatArray:
    """One conversion, at the boundary. dtype conversion inside the core is forbidden."""
    return np.asarray(x, dtype=np.float32)


def l2_normalize(x: FloatArray, *, copy: bool = True) -> FloatArray:
    """ℓ2 normalisation. On normalised vectors cosine equals inner product.

    Zero-norm rows are divided by 1.0, so a zero vector stays zero instead of
    producing NaN.
    """
    x = as_f32(x)
    if copy:
        x = x.copy()
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    x /= norms
    return x


def topk_search(
    queries: FloatArray,
    docs: FloatArray,
    k: int,
    *,
    chunk_size: int = 1024,
    self_mask: np.ndarray | None = None,
    doc_bias: FloatArray | None = None,
) -> tuple[np.ndarray, FloatArray]:
    """Exact top-k via chunked matmul plus ``argpartition``.

    The full score matrix is NEVER materialised: 10k×10k float32 is 400 MB, and
    the same work in 1024-row chunks costs 40 MB. This is the concrete form of
    the architectural invariant — ``O(batch × d)``, not ``O(N × d)``.

    ``argpartition`` rather than ``argsort``: O(n) instead of O(n log n), which
    is worth several times at 10k×10k.

    Args:
        self_mask: Document index to mask out per query (-1 means no mask). In
            T0 a query chunk sits in its own index; letting it retrieve itself
            inflates ARR artificially.
        doc_bias: Constant added to each document's score. The CSLS hubness
            penalty (``-r_T(d)``) is exactly this.
    """
    n_q = queries.shape[0]
    n_d = docs.shape[0]
    k = min(k, n_d - (1 if self_mask is not None else 0))

    out_idx = np.empty((n_q, k), dtype=np.int32)
    out_score = np.empty((n_q, k), dtype=np.float32)

    for start in range(0, n_q, chunk_size):
        stop = min(start + chunk_size, n_q)
        scores = queries[start:stop] @ docs.T  # (chunk, n_d)

        if doc_bias is not None:
            scores += doc_bias  # broadcast over (n_d,)

        if self_mask is not None:
            rows = np.arange(stop - start)
            cols = self_mask[start:stop]
            valid = cols >= 0
            scores[rows[valid], cols[valid]] = -np.inf

        part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_scores, axis=1)  # sorts k elements only
        out_idx[start:stop] = np.take_along_axis(part, order, axis=1)
        out_score[start:stop] = np.take_along_axis(part_scores, order, axis=1)

    return out_idx, out_score


# ─────────────────────────────────────────────────────────────────────────────
# Adapters
#
# Direction: g maps the new space into the old one (``query→old``, the default).
# Every fit call is ``fit(src, dst)`` with ``apply(src) ≈ dst``. Here
# ``src = f_new(d)`` and ``dst = f_old(d)``.
# ─────────────────────────────────────────────────────────────────────────────


class Adapter:
    """Adapter base. Becomes the Protocol in ``core/base.py`` in M1."""

    name: str = "base"

    def apply(self, x: FloatArray) -> FloatArray:  # pragma: no cover - base
        raise NotImplementedError

    def n_params(self) -> int:  # pragma: no cover - base
        raise NotImplementedError

    @property
    def memory_mb(self) -> float:
        """Weight size assuming float32."""
        return self.n_params() * 4 / 1024**2


def _pad_to(x: FloatArray, dim: int) -> FloatArray:
    """Dimension matching: zero-pad on the right.

    Orthogonal Procrustes needs a square matrix. When dimensions differ, the
    smaller side is zero-padded; the OP solution is free in the padded subspace
    and the output is truncated to the target dimension. This is the cheapest
    way to make OP work when ``d_new ≠ d_old``; M1 compares it against a PCA
    projection.
    """
    if x.shape[1] == dim:
        return x
    if x.shape[1] > dim:
        return np.ascontiguousarray(x[:, :dim])
    out = np.zeros((x.shape[0], dim), dtype=np.float32)
    out[:, : x.shape[1]] = x
    return out


class IdentityAdapter(Adapter):
    """No adaptation. Feeds the new model's vector straight into the old index.

    This is the ARR≈0.65 baseline. Only meaningful when the dimensions match;
    when they do not, it cannot run at all.
    """

    name = "identity"

    def __init__(self, dim_out: int) -> None:
        self.dim_out = dim_out

    def apply(self, x: FloatArray) -> FloatArray:
        return _pad_to(as_f32(x), self.dim_out)

    def n_params(self) -> int:
        return 0


class ProcrustesAdapter(Adapter):
    """Orthogonal Procrustes: g(x) = xR with RᵀR = I. Closed-form SVD (K11).

    scipy already provides the closed form in one line; optimising the
    Procrustes loss with SGD would be both slower and unnecessary.

    Orthogonality is free regularisation: the transform preserves angles and
    norms, which makes overfitting on a small sample all but impossible.
    """

    name = "procrustes"

    def __init__(self, R: FloatArray, dim_out: int) -> None:
        self.R = R
        self.dim_out = dim_out

    @classmethod
    def fit(cls, src: FloatArray, dst: FloatArray) -> ProcrustesAdapter:
        d_out = dst.shape[1]
        d_work = max(src.shape[1], dst.shape[1])
        a = _pad_to(as_f32(src), d_work)
        b = _pad_to(as_f32(dst), d_work)
        r, _scale = scipy.linalg.orthogonal_procrustes(a, b)
        return cls(as_f32(r), d_out)

    def apply(self, x: FloatArray) -> FloatArray:
        z = _pad_to(as_f32(x), self.R.shape[0]) @ self.R
        return np.ascontiguousarray(z[:, : self.dim_out])

    def n_params(self) -> int:
        return int(self.R.size)


class CenteredProcrustesAdapter(Adapter):
    """Centred Procrustes: g(x) = (x − μ_src)R + μ_dst, RᵀR = I.

    Plain OP carries a silent constraint: it searches for a rotation **through
    the origin**. Embedding spaces are not isotropic — they carry a large shared
    mean component (anisotropy). When two spaces have their means pointing in
    different directions, no rotation pinned to the origin can close that gap,
    and OP lands far below what it is capable of.

    The cross-lingual alignment literature (MUSE, vecmap) applies centering
    before Procrustes as a standard step. The reference work (K1) does not
    mention it, and rebasis's own pipeline applies only ℓ2 normalisation. This
    adapter exists so the difference can be measured against plain OP.

    Orthogonality is preserved — it is still a rigid rotation in the centred
    space, so OP's resistance to overfitting survives. Extra cost: two vectors
    (2d parameters).
    """

    name = "procrustes_centered"

    def __init__(
        self, R: FloatArray, mu_src: FloatArray, mu_dst: FloatArray, dim_out: int, scale: float
    ) -> None:
        self.R = R
        self.mu_src = mu_src
        self.mu_dst = mu_dst
        self.dim_out = dim_out
        self.scale = scale

    @classmethod
    def fit(
        cls, src: FloatArray, dst: FloatArray, *, with_scale: bool = False
    ) -> CenteredProcrustesAdapter:
        d_out = dst.shape[1]
        d_work = max(src.shape[1], dst.shape[1])
        a = _pad_to(as_f32(src), d_work)
        b = _pad_to(as_f32(dst), d_work)
        mu_a = a.mean(axis=0)
        mu_b = b.mean(axis=0)
        r, ssum = scipy.linalg.orthogonal_procrustes(a - mu_a, b - mu_b)
        # scipy returns ``ssum``, the sum of singular values; divided by ‖A‖_F²
        # it gives the optimal global scale for ‖sAR − B‖.
        scale = 1.0
        if with_scale:
            denom = float(np.sum((a - mu_a) ** 2))
            scale = float(ssum / denom) if denom > 0 else 1.0
        return cls(as_f32(r), as_f32(mu_a), as_f32(mu_b), d_out, scale)

    def apply(self, x: FloatArray) -> FloatArray:
        z = (_pad_to(as_f32(x), self.R.shape[0]) - self.mu_src) @ self.R
        if self.scale != 1.0:
            z = z * self.scale
        z = z + self.mu_dst
        return np.ascontiguousarray(z[:, : self.dim_out])

    def n_params(self) -> int:
        return int(self.R.size + self.mu_src.size + self.mu_dst.size)


class LinearAdapter(Adapter):
    """Full-rank affine: g(x) = xW + b, solved with ridge.

    OP without the orthogonality constraint. It supports dimension change
    natively (W may be rectangular), so unlike OP it needs no zero-padding trick
    when ``d_new ≠ d_old``.

    ``alpha`` is a small ridge term: it keeps the solution finite when X'X is
    singular (fewer pairs than dimensions, or collinear features). Passing 0
    falls back to the minimum-norm lstsq solution.
    """

    name = "linear"

    def __init__(self, W: FloatArray, b: FloatArray) -> None:
        self.W = W
        self.b = b

    @classmethod
    def fit(cls, src: FloatArray, dst: FloatArray, *, alpha: float = 1e-3) -> LinearAdapter:
        # Centering separates the translation from the regression: W carries only
        # the rotation, b the mean. It also keeps the ridge penalty off b.
        x = as_f32(src)
        y = as_f32(dst)
        x_mean = x.mean(axis=0)
        y_mean = y.mean(axis=0)
        xc = x - x_mean
        yc = y - y_mean

        # Solve in float64: the normal equation squares the condition number,
        # which produces measurable error in float32. The result returns to
        # float32 at the boundary.
        xc64 = xc.astype(np.float64)
        gram = xc64.T @ xc64
        if alpha > 0:
            gram[np.diag_indices_from(gram)] += alpha * np.trace(gram) / gram.shape[0]
        w64 = np.linalg.solve(gram, xc64.T @ yc.astype(np.float64))
        w = as_f32(w64)
        b = as_f32(y_mean - x_mean @ w)
        return cls(w, b)

    def apply(self, x: FloatArray) -> FloatArray:
        return as_f32(x) @ self.W + self.b

    def n_params(self) -> int:
        return int(self.W.size + self.b.size)


class LowRankAffineAdapter(Adapter):
    """Low-rank affine: g(x) = x(UVᵀ) + b at rank r (default r=64).

    The truncated SVD of the full-rank W. Two benefits: memory drops from d² to
    r(d₁+d₂) (2.4 MB → 0.4 MB at d=768, r=64), and the truncation acts as
    regularisation, potentially generalising better than the full-rank solution
    on a small sample.
    """

    name = "low_rank_affine"

    def __init__(self, U: FloatArray, V: FloatArray, b: FloatArray, rank: int) -> None:
        self.U = U  # (d_src, r)
        self.V = V  # (r, d_dst)
        self.b = b
        self.rank = rank

    @classmethod
    def fit(
        cls, src: FloatArray, dst: FloatArray, *, rank: int = 64, alpha: float = 1e-3
    ) -> LowRankAffineAdapter:
        full = LinearAdapter.fit(src, dst, alpha=alpha)
        r = min(rank, min(full.W.shape))
        u, s, vt = np.linalg.svd(full.W.astype(np.float64), full_matrices=False)
        # Fold the singular values into U to save one matrix multiply
        u_r = as_f32(u[:, :r] * s[:r])
        v_r = as_f32(vt[:r])
        return cls(u_r, v_r, full.b, r)

    def apply(self, x: FloatArray) -> FloatArray:
        return (as_f32(x) @ self.U) @ self.V + self.b

    def n_params(self) -> int:
        return int(self.U.size + self.V.size + self.b.size)


class ScaledAdapter(Adapter):
    """DSM — per-dimension scaling on top of a base adapter (``S·g(x)``).

    d parameters (~3 KB at d=768), under 1 µs. Worth +0.005–0.015 ARR in the
    reference work. Separate least squares per dimension:
    ``s_j = <ŷ_j, y_j> / <ŷ_j, ŷ_j>``.
    """

    name = "dsm"

    def __init__(self, base: Adapter, scale: FloatArray) -> None:
        self.base = base
        self.scale = scale
        self.name = f"{base.name}+dsm"

    @classmethod
    def fit(cls, base: Adapter, src: FloatArray, dst: FloatArray) -> ScaledAdapter:
        pred = base.apply(src)
        num = np.einsum("ij,ij->j", pred, as_f32(dst))
        den = np.einsum("ij,ij->j", pred, pred)
        scale = as_f32(np.where(den > 1e-12, num / np.maximum(den, 1e-12), 1.0))
        return cls(base, scale)

    def apply(self, x: FloatArray) -> FloatArray:
        return self.base.apply(x) * self.scale

    def n_params(self) -> int:
        return self.base.n_params() + int(self.scale.size)


class ResidualMLPAdapter(Adapter):
    """g(x) = xW + b + MLP(x) where MLP = W₂·GELU(xW₁+b₁)+b₂, hidden 256.

    "Residual" here is the design itself: the linear term W is initialised from
    the closed form (:class:`LinearAdapter`) and the network learns only the
    RESIDUAL. Compared with training from scratch this converges far faster and,
    in practice, removes the risk of ending up BELOW the linear solution in a bad
    local minimum.

    Without torch this adapter is disabled and OP and LA keep working.
    """

    name = "residual_mlp"

    def __init__(self, state: dict[str, FloatArray], hidden: int) -> None:
        self.state = state
        self.hidden = hidden

    @classmethod
    def fit(
        cls,
        src: FloatArray,
        dst: FloatArray,
        *,
        hidden: int = 256,
        epochs: int = 60,
        batch_size: int = 256,
        lr: float = 1e-3,
        val_fraction: float = 0.1,
        device: str = "cpu",
        seed: int = 0,
        patience: int = 8,
    ) -> ResidualMLPAdapter:
        import torch
        from torch import nn

        torch.manual_seed(seed)
        dev = torch.device(device)

        # Linear base from the closed form. The network learns the residual only.
        base = LinearAdapter.fit(src, dst)

        d_in = src.shape[1]
        d_out = dst.shape[1]

        # MPS trap: some operations silently return zeros or NaN when the
        # underlying tensors are not contiguous. Calling .contiguous() at every
        # entry point costs nothing and saves a week of debugging.
        x_all = torch.from_numpy(as_f32(src)).to(dev).contiguous()
        y_all = torch.from_numpy(as_f32(dst)).to(dev).contiguous()

        n = x_all.shape[0]
        n_val = max(1, int(n * val_fraction))
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g).to(dev)
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        x_tr, y_tr = x_all[tr_idx].contiguous(), y_all[tr_idx].contiguous()
        x_val, y_val = x_all[val_idx].contiguous(), y_all[val_idx].contiguous()

        lin = nn.Linear(d_in, d_out).to(dev)
        with torch.no_grad():
            # nn.Linear computes y = xWᵀ + b, so torch stores (d_out, d_in) while
            # ours is (d_in, d_out).
            lin.weight.copy_(torch.from_numpy(base.W.T).to(dev))
            lin.bias.copy_(torch.from_numpy(base.b).to(dev))

        mlp = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out)).to(dev)
        with torch.no_grad():
            # Zero the last layer so that g(x) starts out exactly equal to the
            # linear solution. Training therefore begins from a known-good point
            # and never has to climb back to it.
            mlp[-1].weight.zero_()
            mlp[-1].bias.zero_()

        params = list(lin.parameters()) + list(mlp.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        best_state: dict[str, torch.Tensor] = {}
        bad_epochs = 0
        n_tr = x_tr.shape[0]

        for _epoch in range(epochs):
            lin.train()
            mlp.train()
            order = torch.randperm(n_tr, generator=g).to(dev)
            for start in range(0, n_tr, batch_size):
                idx = order[start : start + batch_size]
                xb, yb = x_tr[idx], y_tr[idx]
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(lin(xb) + mlp(xb), yb)
                loss.backward()
                opt.step()

            lin.eval()
            mlp.eval()
            with torch.no_grad():
                val = loss_fn(lin(x_val) + mlp(x_val), y_val).item()
            if val < best_val - 1e-7:
                best_val = val
                best_state = {
                    k: v.detach().clone()
                    for k, v in list(lin.state_dict().items())
                    + [(f"mlp.{k}", v) for k, v in mlp.state_dict().items()]
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

        # The hot path does not import torch — convert the weights to numpy.
        st = {k: v.cpu().numpy().astype(np.float32) for k, v in best_state.items()}
        state = {
            "W": np.ascontiguousarray(st["weight"].T),
            "b": st["bias"],
            "W1": np.ascontiguousarray(st["mlp.0.weight"].T),
            "b1": st["mlp.0.bias"],
            "W2": np.ascontiguousarray(st["mlp.2.weight"].T),
            "b2": st["mlp.2.bias"],
        }
        return cls(state, hidden)

    def apply(self, x: FloatArray) -> FloatArray:
        s = self.state
        x = as_f32(x)
        h = x @ s["W1"] + s["b1"]
        # Exact erf-based GELU — identical to torch's default
        h = 0.5 * h * (1.0 + scipy.special.erf(h / np.sqrt(2.0, dtype=np.float32)))
        return x @ s["W"] + s["b"] + h @ s["W2"] + s["b2"]

    def n_params(self) -> int:
        return int(sum(v.size for v in self.state.values()))


# ─────────────────────────────────────────────────────────────────────────────
# CSLS — hubness correction (K21)
# ─────────────────────────────────────────────────────────────────────────────


def csls_document_penalty(
    docs: FloatArray, query_sample: FloatArray, *, k: int = 10, chunk_size: int = 1024
) -> FloatArray:
    """r_T(d): a document's mean similarity to its k nearest neighbours in the query distribution.

    MUSE's observation: vectors mapped from one space into another develop
    HUBNESS — a few target vectors become the neighbour of disproportionately
    many queries and wreck retrieval. CSLS corrects this by penalising documents
    that are "close to everyone":

        CSLS(q, d) = 2·cos(q, d) − r_T(d) − r_S(q)

    For ranking purposes ``r_S(q)`` is CONSTANT per query, so it cannot change
    the ordering of results for a single query and need not be computed. What
    remains is a per-document constant penalty — which means CSLS amounts to
    passing ``doc_bias = −r_T(d)`` to :func:`topk_search`. The factor of 2 also
    leaves ranking unchanged, but is kept so that score scales stay comparable
    with the reference.

    Cost: once per document, zero per query. The hot path is untouched.
    """
    n_d = docs.shape[0]
    k = min(k, query_sample.shape[0])
    out = np.empty(n_d, dtype=np.float32)
    for start in range(0, n_d, chunk_size):
        stop = min(start + chunk_size, n_d)
        sims = docs[start:stop] @ query_sample.T
        topk = np.partition(sims, -k, axis=1)[:, -k:]
        out[start:stop] = topk.mean(axis=1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


def recall_at_k(retrieved: np.ndarray, relevant: list[set[int]], k: int) -> float:
    """Mean Recall@k, skipping queries with no relevant documents so the denominator stays sound."""
    scores = []
    for row, rel in zip(retrieved[:, :k], relevant, strict=True):
        if not rel:
            continue
        scores.append(len(rel & set(row.tolist())) / len(rel))
    return float(np.mean(scores)) if scores else float("nan")


def recall_per_query(retrieved: np.ndarray, relevant: list[set[int]], k: int) -> FloatArray:
    """Per-query recall — the raw input for the bootstrap confidence interval."""
    vals = [
        len(rel & set(row.tolist())) / len(rel)
        for row, rel in zip(retrieved[:, :k], relevant, strict=True)
        if rel
    ]
    return as_f32(vals)


def bootstrap_ci(
    per_query: FloatArray, *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for the mean (percentile method).

    Why this is needed: ``probe`` outputs a DECISION, and the decision is made
    against thresholds. If the measurement's own uncertainty is the same size as
    the band width, which side of a threshold you land on is decided by sampling
    noise. The borderline band is assumed to be ±0.005; only a measurement like
    this can say whether that figure is realistic.
    """
    if per_query.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = per_query.size
    idx = rng.integers(0, n, size=(n_boot, n))
    means = per_query[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(per_query.mean()), float(lo), float(hi)


def overlap_at_k(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Top-k intersection ratio between two rankings. Symmetric, in [0, 1]."""
    return float(
        np.mean(
            [len(set(x[:k].tolist()) & set(y[:k].tolist())) / k for x, y in zip(a, b, strict=True)]
        )
    )


def mrr_at_k(retrieved: np.ndarray, relevant: list[set[int]], k: int) -> float:
    """Mean Reciprocal Rank@k — ranking quality, the basis of ``ARR_MRR``."""
    scores = []
    for row, rel in zip(retrieved[:, :k], relevant, strict=True):
        if not rel:
            continue
        rr = 0.0
        for rank, doc in enumerate(row.tolist(), start=1):
            if doc in rel:
                rr = 1.0 / rank
                break
        scores.append(rr)
    return float(np.mean(scores)) if scores else float("nan")


def spearman_topk(gt_idx: np.ndarray, pred_idx: np.ndarray, k: int) -> float:
    """Spearman correlation between the ground-truth top-k order and its order in the prediction.

    Items absent from the prediction are assigned a penalty rank of ``k + 1``;
    otherwise a correlation computed only over the items that were found makes
    bad adapters look good.
    """
    vals = []
    for gt_row, pr_row in zip(gt_idx[:, :k], pred_idx, strict=True):
        pos = {doc: i for i, doc in enumerate(pr_row[:k].tolist())}
        ranks_pred = [pos.get(doc, k) for doc in gt_row.tolist()]
        if len(set(ranks_pred)) < 2:
            continue
        rho = scipy.stats.spearmanr(list(range(len(gt_row))), ranks_pred).statistic
        if not np.isnan(rho):
            vals.append(float(rho))
    return float(np.mean(vals)) if vals else float("nan")


def score_shift_ks(a: FloatArray, b: FloatArray) -> float:
    """KS distance between two cosine score distributions (``score_shift``).

    Rarely discussed and critical: many RAG pipelines use a FIXED threshold such
    as ``similarity > 0.7``. An adapter can preserve ranking perfectly and still
    shift the absolute scores enough to silently empty those filters. Above 0.1
    this raises a warning independent of the decision band.
    """
    return float(scipy.stats.ks_2samp(a.ravel(), b.ravel()).statistic)


DECISION_BANDS: tuple[tuple[float, str], ...] = (
    (0.95, "bridge_sufficient"),
    (0.85, "bridge_and_migrate"),
    (0.70, "caution"),
    (0.0, "full_reindex"),
)


def decide(arr_at_10: float, *, borderline: float = 0.005) -> dict[str, Any]:
    """The decision rule plus its uncertainty band.

    "Small numerical differences are acceptable; a different decision is not."
    When ARR is within ±0.005 of a threshold the decision is flagged borderline
    and the user is advised to enlarge the sample. Without this, 0.9501 and
    0.9499 produce two different recommendations and a change of device changes
    the advice.
    """
    decision = "full_reindex"
    for threshold, label in DECISION_BANDS:
        if arr_at_10 >= threshold:
            decision = label
            break
    nearest = min((abs(arr_at_10 - t), t) for t, _ in DECISION_BANDS if t > 0)
    return {
        "arr_r10": arr_at_10,
        "decision": decision,
        "borderline": bool(nearest[0] <= borderline),
        "nearest_threshold": nearest[1],
        "distance_to_threshold": round(nearest[0], 5),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Encoding profiles — the most likely source of silent errors
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EncodingProfile:
    """How a model encodes queries versus documents.

    This class exists for one reason: ``nomic-embed-text``, E5 and BGE encode
    queries and documents DIFFERENTLY. Fit an adapter on document pairs, apply
    it to query vectors, and you get a train/serve mismatch that raises no error
    and only lowers ARR. Skipping this distinction in M0 would have made every
    number suspect.
    """

    model_id: str
    dim: int
    query_prefix: str | None = None
    document_prefix: str | None = None
    normalize: bool = True

    @property
    def symmetric(self) -> bool:
        return self.query_prefix == self.document_prefix

    def prefix_for(self, kind: Literal["query", "document"]) -> str:
        p = self.query_prefix if kind == "query" else self.document_prefix
        return p or ""

    def fingerprint(self) -> str:
        import hashlib

        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# Known profiles. These move to ``embed/data/profiles.yaml`` in M1; the values
# come from each model card's official usage instructions.
PROFILES: dict[str, EncodingProfile] = {
    # Symmetric: queries and documents encoded identically.
    "sentence-transformers/all-MiniLM-L6-v2": EncodingProfile(
        model_id="sentence-transformers/all-MiniLM-L6-v2", dim=384
    ),
    # ASYMMETRIC: BGE v1.5 wants an instruction prefix on the query for
    # retrieval, none on the document. The effect is pronounced on short queries.
    "BAAI/bge-small-en-v1.5": EncodingProfile(
        model_id="BAAI/bge-small-en-v1.5",
        dim=384,
        query_prefix="Represent this sentence for searching relevant passages: ",
        document_prefix=None,
    ),
    # ASYMMETRIC: the classic E5 "query: " / "passage: " split.
    "intfloat/e5-small-v2": EncodingProfile(
        model_id="intfloat/e5-small-v2",
        dim=384,
        query_prefix="query: ",
        document_prefix="passage: ",
    ),
}


class SentenceTransformerEmbedder:
    """sentence-transformers wrapper, profile-aware.

    The model is loaded lazily so that probe paths which only draw a sample can
    run without downloading anything.
    """

    def __init__(self, profile: EncodingProfile, *, device: str | None = None) -> None:
        self.profile = profile
        self.device = device
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.profile.model_id, device=self.device)
        return self._model

    def encode(
        self,
        texts: Sequence[str],
        *,
        kind: Literal["query", "document"],
        batch_size: int = 64,
        progress: bool = False,
    ) -> FloatArray:
        prefix = self.profile.prefix_for(kind)
        payload = [prefix + t for t in texts] if prefix else list(texts)
        vecs = self.model.encode(
            payload,
            batch_size=batch_size,
            show_progress_bar=progress,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we normalise ourselves
        )
        return as_f32(vecs)


# ─────────────────────────────────────────────────────────────────────────────
# Corpus
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Corpus:
    """Documents plus, where available, real queries and relevance judgements.

    When ``queries``/``qrels`` are populated the T1 tier can run — and
    measuring T0 against T1 is the whole point of M0.
    """

    name: str
    doc_ids: list[str]
    doc_texts: list[str]
    query_ids: list[str] = field(default_factory=list)
    query_texts: list[str] = field(default_factory=list)
    qrels: dict[str, set[str]] = field(default_factory=dict)

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    @property
    def has_real_queries(self) -> bool:
        return bool(self.query_ids and self.qrels)


def load_beir(dataset: str = "beir/scifact/test", *, max_docs: int | None = None) -> Corpus:
    """Load a BEIR corpus through ir_datasets.

    ``ir_datasets`` is a DELIBERATE choice: pulling ``BeIR/scifact`` through
    ``datasets`` yields 1109 queries (all splits) while the qrels cover only the
    test split — most queries would have no judgements and the recall denominator
    would break silently. ``ir_datasets`` returns the 300 queries of the split.
    """
    import ir_datasets

    ds = ir_datasets.load(dataset)

    doc_ids: list[str] = []
    doc_texts: list[str] = []
    for d in ds.docs_iter():
        title = getattr(d, "title", "") or ""
        text = getattr(d, "text", "") or ""
        doc_ids.append(d.doc_id)
        doc_texts.append(f"{title} {text}".strip())
        if max_docs and len(doc_ids) >= max_docs:
            break

    keep = set(doc_ids)
    qrels: dict[str, set[str]] = {}
    for qr in ds.qrels_iter():
        if qr.relevance > 0 and qr.doc_id in keep:
            qrels.setdefault(qr.query_id, set()).add(qr.doc_id)

    query_ids: list[str] = []
    query_texts: list[str] = []
    for q in ds.queries_iter():
        # A query with no judgements breaks recall: either the denominator is
        # zero or the query counts as a permanent failure. Both make the number
        # meaningless.
        if q.query_id in qrels:
            query_ids.append(q.query_id)
            query_texts.append(q.text)

    return Corpus(
        name=dataset,
        doc_ids=doc_ids,
        doc_texts=doc_texts,
        query_ids=query_ids,
        query_texts=query_texts,
        qrels=qrels,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Embedding cache — so the same measurement can be re-run cheaply
# ─────────────────────────────────────────────────────────────────────────────


def cached_encode(
    embedder: SentenceTransformerEmbedder,
    texts: Sequence[str],
    *,
    kind: Literal["query", "document"],
    cache_dir: Path,
    tag: str,
    batch_size: int = 64,
) -> FloatArray:
    """Cache embeddings on disk; never recompute the same (model, profile, kind, texts).

    The key includes the profile fingerprint, so a prefix change AUTOMATICALLY
    misses the cache. That matters: reintroducing the prefix bug through a stale
    cache would make it impossible to diagnose.
    """
    import hashlib

    h = hashlib.sha256()
    h.update(embedder.profile.fingerprint().encode())
    h.update(kind.encode())
    h.update(str(len(texts)).encode())
    for t in texts[:64]:
        h.update(t.encode("utf-8", "replace"))
    for t in texts[-64:]:
        h.update(t.encode("utf-8", "replace"))
    key = h.hexdigest()[:24]

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{tag}-{key}.npy"
    if path.exists():
        arr = np.load(path)
        if arr.shape[0] == len(texts):
            console.print(f"  [dim]cache[/dim] {tag} ({kind}) ← {path.name}")
            return as_f32(arr)

    with measure(f"encode:{tag}:{kind}", n_items=len(texts)):
        vecs = embedder.encode(texts, kind=kind, batch_size=batch_size, progress=False)
    np.save(path, vecs)
    return vecs


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1 — synthetic validation ("the project's most valuable test")
# ─────────────────────────────────────────────────────────────────────────────


def random_orthogonal(dim: int, rng: np.random.Generator) -> FloatArray:
    """Random orthogonal matrix under the Haar measure (QR with sign correction)."""
    a = rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(a)
    q *= np.sign(np.diag(r))
    return as_f32(q)


def experiment_synthetic(seed: int = 0) -> dict[str, Any]:
    """Do the adapters recover a known transform?

    Catches implementation errors in milliseconds without downloading a model. An
    adapter that fails here makes every number it later produces on a real corpus
    untrustworthy — which is why this is M0's FIRST step.
    """
    console.rule("[bold]Experiment 1 — synthetic validation")
    rng = np.random.default_rng(seed)
    results: dict[str, Any] = {}

    # 1. OP must recover a known rotation exactly
    for dim, n in ((64, 500), (384, 2000)):
        a = l2_normalize(as_f32(rng.standard_normal((n, dim))))
        r_true = random_orthogonal(dim, rng)
        b = as_f32(a @ r_true.T)
        adapter = ProcrustesAdapter.fit(src=b, dst=a)
        err = float(np.abs(adapter.apply(b) - a).max())
        ok = err < 1e-4
        results[f"procrustes_recovers_rotation_d{dim}"] = {"max_abs_err": err, "passed": ok}
        console.print(
            f"  {'[green]✓[/green]' if ok else '[red]✗[/red]'} "
            f"Procrustes d={dim}, n={n}: max|err| = {err:.2e}"
        )

    # 2. The linear adapter must recover a known affine, dimension change included
    for d_in, d_out in ((128, 128), (384, 256)):
        x = as_f32(rng.standard_normal((3000, d_in)))
        w_true = as_f32(rng.standard_normal((d_in, d_out)) / np.sqrt(d_in))
        b_true = as_f32(rng.standard_normal(d_out) * 0.1)
        y = x @ w_true + b_true
        adapter = LinearAdapter.fit(src=x, dst=y, alpha=0.0)
        err = float(np.abs(adapter.apply(x) - y).max())
        ok = err < 1e-3
        results[f"linear_recovers_affine_{d_in}to{d_out}"] = {"max_abs_err": err, "passed": ok}
        console.print(
            f"  {'[green]✓[/green]' if ok else '[red]✗[/red]'} "
            f"Linear {d_in}→{d_out}: max|err| = {err:.2e}"
        )

    # 3. The low-rank adapter must recover a genuinely low-rank transform
    d, r = 256, 32
    x = as_f32(rng.standard_normal((4000, d)))
    u_true = as_f32(rng.standard_normal((d, r)) / np.sqrt(d))
    v_true = as_f32(rng.standard_normal((r, d)) / np.sqrt(r))
    y = x @ (u_true @ v_true)
    adapter = LowRankAffineAdapter.fit(src=x, dst=y, rank=r, alpha=0.0)
    rel = float(np.abs(adapter.apply(x) - y).max() / (np.abs(y).max() + 1e-12))
    ok = rel < 1e-2
    results["low_rank_recovers_low_rank"] = {"max_rel_err": rel, "passed": ok}
    console.print(
        f"  {'[green]✓[/green]' if ok else '[red]✗[/red]'} LowRank d={d}, r={r}: rel err = {rel:.2e}"
    )

    # 4. apply must agree between batched and single-vector calls
    x = as_f32(rng.standard_normal((64, 128)))
    y = as_f32(rng.standard_normal((64, 128)))
    ad = LinearAdapter.fit(x, y)
    batch = ad.apply(x)
    single = np.vstack([ad.apply(x[i : i + 1]) for i in range(x.shape[0])])
    err = float(np.abs(batch - single).max())
    ok = err < 1e-5
    results["apply_batch_equals_single"] = {"max_abs_err": err, "passed": ok}
    console.print(
        f"  {'[green]✓[/green]' if ok else '[red]✗[/red]'} apply batch == single: max|err| = {err:.2e}"
    )

    # 5. topk_search must match a brute-force argsort exactly
    q = l2_normalize(as_f32(rng.standard_normal((97, 64))))
    docs = l2_normalize(as_f32(rng.standard_normal((503, 64))))
    idx, _ = topk_search(q, docs, k=10, chunk_size=16)
    ref = np.argsort(-(q @ docs.T), axis=1)[:, :10]
    ok = bool((idx == ref).all())
    results["topk_matches_bruteforce"] = {"passed": ok}
    console.print(
        f"  {'[green]✓[/green]' if ok else '[red]✗[/red]'} chunked topk == brute-force argsort"
    )

    results["all_passed"] = all(
        v["passed"] for v in results.values() if isinstance(v, dict) and "passed" in v
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2 — T0 proxy validity and CSLS
# ─────────────────────────────────────────────────────────────────────────────


def eval_against(
    q_vecs: FloatArray,
    doc_vecs: FloatArray,
    relevant: list[set[int]],
    k: int,
    *,
    gt_idx: np.ndarray | None = None,
    doc_bias: FloatArray | None = None,
    self_mask: np.ndarray | None = None,
) -> tuple[dict[str, float], np.ndarray, FloatArray]:
    """Evaluate one retrieval configuration."""
    idx, scores = topk_search(q_vecs, doc_vecs, k=k, self_mask=self_mask, doc_bias=doc_bias)
    out = {
        "recall_at_k": recall_at_k(idx, relevant, k),
        "mrr_at_k": mrr_at_k(idx, relevant, k),
    }
    if gt_idx is not None:
        out["overlap_at_k"] = overlap_at_k(idx, gt_idx, k)
        out["spearman_topk"] = spearman_topk(gt_idx, idx, k)
    return out, idx, scores


def build_adapters(
    src: FloatArray,
    dst: FloatArray,
    *,
    use_mlp: bool,
    mlp_device: str,
    seed: int,
) -> dict[str, Adapter]:
    """Fit every adapter parametrisation — the raw form of ``auto``.

    ``auto`` fits the adapters in sequence but shares a single preprocessing
    step: normalisation happens once in the caller, only fitting happens here.
    """
    adapters: dict[str, Adapter] = {}

    with measure("fit:procrustes", n_items=src.shape[0]):
        adapters["procrustes"] = ProcrustesAdapter.fit(src, dst)
    with measure("fit:procrustes_centered", n_items=src.shape[0]):
        adapters["procrustes_centered"] = CenteredProcrustesAdapter.fit(src, dst)
    with measure("fit:linear", n_items=src.shape[0]):
        adapters["linear"] = LinearAdapter.fit(src, dst)
    with measure("fit:low_rank_affine", n_items=src.shape[0]):
        adapters["low_rank_affine"] = LowRankAffineAdapter.fit(src, dst, rank=64)

    # DSM rides on top of a base adapter: d parameters, under 1 µs
    with measure("fit:procrustes_centered+dsm", n_items=src.shape[0]):
        adapters["procrustes_centered+dsm"] = ScaledAdapter.fit(
            adapters["procrustes_centered"], src, dst
        )

    if use_mlp:
        try:
            with measure(f"fit:residual_mlp[{mlp_device}]", n_items=src.shape[0]):
                adapters["residual_mlp"] = ResidualMLPAdapter.fit(
                    src, dst, device=mlp_device, seed=seed
                )
        except Exception as exc:  # no torch, or a device problem — OP/LA carry on
            console.print(f"  [yellow]![/yellow] residual_mlp skipped: {exc}")

    # ``identity``: no adaptation. With matching dimensions this is the ARR≈0.65
    # baseline.
    adapters["identity"] = IdentityAdapter(dst.shape[1])
    return adapters


def experiment_model_pair(
    corpus: Corpus,
    old_id: str,
    new_id: str,
    *,
    cache_dir: Path,
    device: str | None = None,
    mlp_device: str = "cpu",
    k: int = 10,
    fit_size: int | None = None,
    heldout_size: int = 1000,
    seed: int = 0,
    use_mlp: bool = True,
    csls_k: int = 10,
) -> dict[str, Any]:
    """Measure T0 and T1 SIDE BY SIDE for one model pair.

    M0's central question is answered here: does the T0 proxy that ``probe`` uses
    by default when no query log exists (documents standing in for queries)
    produce the same DECISION as T1, measured with real queries?

    The profile trap is measured too: on asymmetric models, how much does
    applying a document-fitted adapter to query vectors cost? Without that
    number the "two adapters" rule stays an intuition.
    """
    old_profile = PROFILES[old_id]
    new_profile = PROFILES[new_id]
    console.rule(f"[bold]{old_id}  →  {new_id}")
    console.print(
        f"  old: d={old_profile.dim} symmetric={old_profile.symmetric} · "
        f"new: d={new_profile.dim} symmetric={new_profile.symmetric}"
    )

    old_emb = SentenceTransformerEmbedder(old_profile, device=device)
    new_emb = SentenceTransformerEmbedder(new_profile, device=device)

    def slug(s: str) -> str:
        return s.replace("/", "_")

    # ── Document side: what lives in the index ────────────────────────────
    a_doc = l2_normalize(
        cached_encode(
            old_emb,
            corpus.doc_texts,
            kind="document",
            cache_dir=cache_dir,
            tag=f"{slug(old_id)}-docs",
        )
    )
    b_doc = l2_normalize(
        cached_encode(
            new_emb,
            corpus.doc_texts,
            kind="document",
            cache_dir=cache_dir,
            tag=f"{slug(new_id)}-docs",
        )
    )

    # ── Split: fit set and query set strictly disjoint ─────────────────────
    rng = np.random.default_rng(seed)
    perm = rng.permutation(corpus.n_docs)
    heldout_ids = perm[:heldout_size]
    pool = perm[heldout_size:]
    fit_ids = pool if fit_size is None else pool[:fit_size]
    leak = set(fit_ids.tolist()) & set(heldout_ids.tolist())
    if leak:
        msg = f"LEAKAGE: fit and query sets overlap on {len(leak)} items"
        raise RuntimeError(msg)
    console.print(f"  fit pairs: {len(fit_ids)} · T0 queries: {len(heldout_ids)} (disjoint ✓)")

    # ── Query-side encoding ───────────────────────────────────────────────
    # For a symmetric model the query encoding IS the document encoding —
    # recomputing it would waste time and bloat the cache for nothing.
    def query_encoded_docs(
        emb: SentenceTransformerEmbedder, doc_side: FloatArray, model_id: str
    ) -> FloatArray:
        if emb.profile.symmetric:
            return doc_side
        return l2_normalize(
            cached_encode(
                emb,
                corpus.doc_texts,
                kind="query",
                cache_dir=cache_dir,
                tag=f"{slug(model_id)}-docs-as-queries",
            )
        )

    a_doc_q = query_encoded_docs(old_emb, a_doc, old_id)
    b_doc_q = query_encoded_docs(new_emb, b_doc, new_id)

    results: dict[str, Any] = {
        "old_model": old_id,
        "new_model": new_id,
        "old_profile": asdict(old_profile),
        "new_profile": asdict(new_profile),
        "corpus": corpus.name,
        "n_docs": corpus.n_docs,
        "n_fit_pairs": len(fit_ids),
        "n_t0_queries": len(heldout_ids),
        "k": k,
        "seed": seed,
    }

    # ── Adapter fitting ───────────────────────────────────────────────────
    # g_doc: from document pairs. Correct for virtual backfill and symmetric models.
    console.print("\n  [bold]adapter fit — document pairs (g_doc)[/bold]")
    g_doc = build_adapters(
        b_doc[fit_ids], a_doc[fit_ids], use_mlp=use_mlp, mlp_device=mlp_device, seed=seed
    )

    # g_query: the SAME texts, encoded with the query prefix.
    asymmetric = not (old_profile.symmetric and new_profile.symmetric)
    if asymmetric:
        console.print("  [bold]adapter fit — query pairs (g_query)[/bold]")
        g_query = build_adapters(
            b_doc_q[fit_ids],
            a_doc_q[fit_ids],
            use_mlp=use_mlp,
            mlp_device=mlp_device,
            seed=seed,
        )
    else:
        console.print("  [dim]profiles are symmetric → single adapter (g_query = g_doc)[/dim]")
        g_query = g_doc
    results["asymmetric"] = asymmetric

    for name, ad in g_doc.items():
        results.setdefault("adapter_size", {})[name] = {
            "n_params": ad.n_params(),
            "memory_mb": round(ad.memory_mb, 3),
        }

    # ── T0 evaluation ─────────────────────────────────────────────────────
    results["t0"] = _eval_tier0(
        a_doc=a_doc,
        b_doc=b_doc,
        a_doc_q=a_doc_q,
        b_doc_q=b_doc_q,
        heldout_ids=heldout_ids,
        g_doc=g_doc,
        g_query=g_query,
        k=k,
        csls_k=csls_k,
        asymmetric=asymmetric,
    )

    # ── T1 evaluation ─────────────────────────────────────────────────────
    if corpus.has_real_queries:
        results["t1"] = _eval_tier1(
            corpus=corpus,
            old_emb=old_emb,
            new_emb=new_emb,
            a_doc=a_doc,
            b_doc=b_doc,
            b_doc_q=b_doc_q,
            g_query=g_query,
            g_doc=g_doc,
            cache_dir=cache_dir,
            k=k,
            csls_k=csls_k,
            asymmetric=asymmetric,
            slug_old=slug(old_id),
            slug_new=slug(new_id),
        )
    else:
        results["t1"] = None
        console.print("  [yellow]![/yellow] corpus has no real queries — T1 skipped")

    return results


def _csls_bias(docs: FloatArray, mapped_sample: FloatArray, *, k: int) -> FloatArray:
    """The per-document ranking bias implied by CSLS.

    CSLS(q,d) = 2·cos(q,d) − r_T(d) − r_S(q). The constant ``r_S(q)`` drops out;
    ranking by the remaining ``2·cos − r_T(d)`` is IDENTICAL to ranking by
    ``cos − r_T(d)/2``. The bias is therefore ``−r_T(d)/2``, not ``−r_T(d)``:
    forgetting the factor of 2 doubles the penalty and makes CSLS look worse than
    it is.
    """
    penalty = csls_document_penalty(docs, mapped_sample, k=k)
    return as_f32(-penalty / 2.0)


def _renorm_apply(adapter: Adapter, x: FloatArray) -> FloatArray:
    """Apply the adapter, then renormalise."""
    return l2_normalize(adapter.apply(x), copy=False)


def _eval_tier0(
    *,
    a_doc: FloatArray,
    b_doc: FloatArray,
    a_doc_q: FloatArray,
    b_doc_q: FloatArray,
    heldout_ids: np.ndarray,
    g_doc: dict[str, Adapter],
    g_query: dict[str, Adapter],
    k: int,
    csls_k: int,
    asymmetric: bool,
) -> dict[str, Any]:
    """T0: held-out chunks act as queries; ground truth is the new model's in-sample kNN.

    The ground truth is precisely "what would I get after a full reindex":
    documents encoded with the new model, query encoded with the new model,
    exhaustive kNN. The adapter's job is to reproduce that inside the OLD index.
    """
    console.print("\n  [bold]T0 — documents as query proxies[/bold]")
    q_new = b_doc_q[heldout_ids]
    q_old = a_doc_q[heldout_ids]
    # The query chunk sits inside the index; letting it find itself inflates ARR.
    self_mask = heldout_ids.astype(np.int64)

    with measure("groundtruth:knn", n_items=len(heldout_ids)):
        gt_idx, gt_scores = topk_search(q_new, b_doc, k=k, self_mask=self_mask)
    relevant = [set(row.tolist()) for row in gt_idx]

    # DECOMPOSITION — which of the two explains the T0/T1 gap?
    #
    #   (a) THE PROXY: standing documents in for queries (T0's honest caveat), or
    #   (b) THE METRIC: T0 asks for the whole kNN set to be reproduced, while T1
    #       asks only that the relevant document (≈1.1 per query in sparse qrels)
    #       lands in the top k.
    #
    # To separate them, T0 is re-measured with T1's metric shape: only the SINGLE
    # nearest neighbour of the ground truth counts as relevant — the structural
    # equivalent of qrel sparsity. If the relaxed T0 ≈ T1, the proxy is sound and
    # the problem is entirely one of threshold calibration.
    relevant_gt1 = [{int(row[0])} for row in gt_idx]
    relevant_gt3 = [set(row[:3].tolist()) for row in gt_idx]

    out: dict[str, Any] = {"n_queries": len(heldout_ids)}

    # Baseline 1: do nothing — query with the old model, search the old index.
    # This is ``naive_overlap``: the raw severity of the drift.
    m, idx_old, _ = eval_against(q_old, a_doc, relevant, k, gt_idx=gt_idx, self_mask=self_mask)
    out["old_model_only"] = m
    out["naive_overlap_at_k"] = m["recall_at_k"]
    out["old_model_only_gt1"] = recall_at_k(idx_old, relevant_gt1, k)

    # Baseline 2: push the raw new-model vector into the old index.
    # This is the ARR≈0.65 figure. Meaningful only when dimensions match.
    if q_new.shape[1] == a_doc.shape[1]:
        m, _, _ = eval_against(q_new, a_doc, relevant, k, gt_idx=gt_idx, self_mask=self_mask)
        out["unadapted"] = m

    per_adapter: dict[str, Any] = {}
    for name, adapter in g_query.items():
        mapped = _renorm_apply(adapter, q_new)
        m, idx, scores = eval_against(
            mapped, a_doc, relevant, k, gt_idx=gt_idx, self_mask=self_mask
        )
        entry: dict[str, Any] = {"plain": m}
        entry["arr"] = m["recall_at_k"]
        entry["arr_gt1"] = recall_at_k(idx, relevant_gt1, k)
        entry["arr_gt3"] = recall_at_k(idx, relevant_gt3, k)
        _m, _lo, _hi = bootstrap_ci(recall_per_query(idx, relevant_gt1, k))
        entry["arr_gt1_ci"] = [_lo, _hi]
        entry["arr_gt1_ci_halfwidth"] = (_hi - _lo) / 2.0
        # score_shift: ranking can be preserved while fixed-threshold filters break
        entry["score_shift_ks"] = score_shift_ks(scores, gt_scores)

        # CSLS ablation — same adapter, ranking correction only
        sample = mapped[: min(4000, mapped.shape[0])]
        bias = _csls_bias(a_doc, sample, k=csls_k)
        m_csls, _, _ = eval_against(
            mapped, a_doc, relevant, k, gt_idx=gt_idx, doc_bias=bias, self_mask=self_mask
        )
        entry["csls"] = m_csls
        entry["arr_csls"] = m_csls["recall_at_k"]
        entry["csls_delta"] = m_csls["recall_at_k"] - m["recall_at_k"]

        # The profile trap: applying a document-fitted adapter to a query vector
        if asymmetric and name in g_doc:
            mism = _renorm_apply(g_doc[name], q_new)
            m_mis, _, _ = eval_against(mism, a_doc, relevant, k, gt_idx=gt_idx, self_mask=self_mask)
            entry["profile_mismatch_arr"] = m_mis["recall_at_k"]
            entry["profile_mismatch_cost"] = m["recall_at_k"] - m_mis["recall_at_k"]

        entry["decision"] = decide(entry["arr"])
        per_adapter[name] = entry

    out["adapters"] = per_adapter
    best = max(
        (n for n in per_adapter if n != "identity"),
        key=lambda n: per_adapter[n]["arr"],
        default=None,
    )
    out["best_adapter"] = best
    out["best_arr"] = per_adapter[best]["arr"] if best else None

    out["calibration"] = _eval_calibration(
        q_new=q_new,
        a_doc=a_doc,
        gt_scores=gt_scores,
        g_query=g_query,
        k=k,
        self_mask=self_mask,
    )
    return out


def _eval_calibration(
    *,
    q_new: FloatArray,
    a_doc: FloatArray,
    gt_scores: FloatArray,
    g_query: dict[str, Adapter],
    k: int,
    self_mask: np.ndarray,
) -> dict[str, Any]:
    """Does isotonic calibration actually close the score shift?

    The measured problem: the distribution of adapted scores differs so much from
    the oracle's that every FIXED-threshold filter such as ``similarity > 0.7``
    breaks silently. Isotonic regression is the proposed fix: being monotone it
    cannot disturb ranking, it only moves the score scale onto the target
    distribution.

    The calibrator is fitted on a HELD-OUT subset and measured on a SEPARATE one.
    Fitting and measuring on the same scores would flatter the calibration.
    """
    from sklearn.isotonic import IsotonicRegression

    n = q_new.shape[0]
    n_cal = max(50, n // 3)
    if n <= n_cal + 50:
        return {"skipped": "not enough queries to both calibrate and evaluate"}

    out: dict[str, Any] = {"n_calibration": n_cal, "n_eval": n - n_cal}
    per_adapter: dict[str, Any] = {}

    for name, adapter in g_query.items():
        if name == "identity":
            continue
        mapped = _renorm_apply(adapter, q_new)
        _, scores = topk_search(mapped, a_doc, k=k, self_mask=self_mask)

        x_cal = scores[:n_cal].ravel()
        y_cal = gt_scores[:n_cal].ravel()
        x_ev = scores[n_cal:].ravel()
        y_ev = gt_scores[n_cal:].ravel()

        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(x_cal, y_cal)
        x_ev_cal = as_f32(iso.predict(x_ev))

        ks_before = float(scipy.stats.ks_2samp(x_ev, y_ev).statistic)
        ks_after = float(scipy.stats.ks_2samp(x_ev_cal, y_ev).statistic)

        # Monotonicity should preserve ranking — that is the claim, so measure it.
        rank_preserved = bool(np.all(np.diff(iso.predict(np.sort(x_ev)[:5000])) >= -1e-6))
        per_adapter[name] = {
            "score_shift_before": ks_before,
            "score_shift_after": ks_after,
            "improvement": ks_before - ks_after,
            "passes_threshold_after": ks_after <= 0.1,
            "rank_preserved": rank_preserved,
            "mean_abs_score_error_before": float(np.mean(np.abs(x_ev - y_ev))),
            "mean_abs_score_error_after": float(np.mean(np.abs(x_ev_cal - y_ev))),
        }

    out["adapters"] = per_adapter
    return out


def _eval_tier1(
    *,
    corpus: Corpus,
    old_emb: SentenceTransformerEmbedder,
    new_emb: SentenceTransformerEmbedder,
    a_doc: FloatArray,
    b_doc: FloatArray,
    b_doc_q: FloatArray,
    g_query: dict[str, Adapter],
    g_doc: dict[str, Adapter],
    cache_dir: Path,
    k: int,
    csls_k: int,
    asymmetric: bool,
    slug_old: str,
    slug_new: str,
) -> dict[str, Any]:
    """T1: a real query log — always preferred where available.

    Here the ground truth is human judgement (qrels), not a proxy. The oracle is
    no longer 1.0: even a full reindex does not achieve perfect recall. That is
    why ARR is a genuine RATIO at T1 — adapted recall over oracle recall.
    """
    console.print("\n  [bold]T1 — real queries with qrels[/bold]")
    doc_pos = {doc_id: i for i, doc_id in enumerate(corpus.doc_ids)}
    relevant = [
        {doc_pos[d] for d in corpus.qrels.get(qid, set()) if d in doc_pos}
        for qid in corpus.query_ids
    ]

    q_new = l2_normalize(
        cached_encode(
            new_emb,
            corpus.query_texts,
            kind="query",
            cache_dir=cache_dir,
            tag=f"{slug_new}-queries",
        )
    )
    q_old = l2_normalize(
        cached_encode(
            old_emb,
            corpus.query_texts,
            kind="query",
            cache_dir=cache_dir,
            tag=f"{slug_old}-queries",
        )
    )

    out: dict[str, Any] = {"n_queries": len(corpus.query_ids)}

    # Oracle: the full reindex. The ceiling the tool promises the user.
    m_oracle, oracle_idx, oracle_scores = eval_against(q_new, b_doc, relevant, k)
    out["oracle_new_model"] = m_oracle
    oracle_recall = m_oracle["recall_at_k"]
    _om, _olo, _ohi = bootstrap_ci(recall_per_query(oracle_idx, relevant, k))
    out["oracle_recall_ci"] = [_olo, _ohi]
    out["oracle_recall_ci_halfwidth"] = (_ohi - _olo) / 2.0

    # Do nothing
    m_old, _, _ = eval_against(q_old, a_doc, relevant, k, gt_idx=oracle_idx)
    out["old_model_only"] = m_old
    out["old_model_arr"] = m_old["recall_at_k"] / oracle_recall if oracle_recall else float("nan")

    # Raw new vector into the old index
    if q_new.shape[1] == a_doc.shape[1]:
        m_un, _, _ = eval_against(q_new, a_doc, relevant, k, gt_idx=oracle_idx)
        out["unadapted"] = m_un
        out["unadapted_arr"] = (
            m_un["recall_at_k"] / oracle_recall if oracle_recall else float("nan")
        )

    per_adapter: dict[str, Any] = {}
    for name, adapter in g_query.items():
        mapped = _renorm_apply(adapter, q_new)
        m, idx, scores = eval_against(mapped, a_doc, relevant, k, gt_idx=oracle_idx)
        arr = m["recall_at_k"] / oracle_recall if oracle_recall else float("nan")
        entry: dict[str, Any] = {
            "plain": m,
            "arr": arr,
            "score_shift_ks": score_shift_ks(scores, oracle_scores),
        }
        # ARR is a RATIO whose numerator and denominator come from the same
        # queries. A query-level paired bootstrap cancels the shared query
        # variance correctly; sampling the two independently would inflate the
        # interval.
        pq_ad = recall_per_query(idx, relevant, k)
        pq_or = recall_per_query(oracle_idx, relevant, k)
        if pq_ad.size and pq_ad.size == pq_or.size:
            _rng = np.random.default_rng(0)
            _bi = _rng.integers(0, pq_ad.size, size=(2000, pq_ad.size))
            _num = pq_ad[_bi].mean(axis=1)
            _den = pq_or[_bi].mean(axis=1)
            _ratios = np.divide(_num, _den, out=np.full_like(_num, np.nan), where=_den > 0)
            _lo, _hi = np.nanpercentile(_ratios, [2.5, 97.5])
            entry["arr_ci"] = [float(_lo), float(_hi)]
            entry["arr_ci_halfwidth"] = float((_hi - _lo) / 2.0)

        sample = _renorm_apply(adapter, b_doc_q[: min(4000, b_doc_q.shape[0])])
        bias = _csls_bias(a_doc, sample, k=csls_k)
        m_csls, _, _ = eval_against(mapped, a_doc, relevant, k, gt_idx=oracle_idx, doc_bias=bias)
        arr_csls = m_csls["recall_at_k"] / oracle_recall if oracle_recall else float("nan")
        entry["csls"] = m_csls
        entry["arr_csls"] = arr_csls
        entry["csls_delta"] = arr_csls - arr

        if asymmetric and name in g_doc:
            mism = _renorm_apply(g_doc[name], q_new)
            m_mis, _, _ = eval_against(mism, a_doc, relevant, k, gt_idx=oracle_idx)
            arr_mis = m_mis["recall_at_k"] / oracle_recall if oracle_recall else float("nan")
            entry["profile_mismatch_arr"] = arr_mis
            entry["profile_mismatch_cost"] = arr - arr_mis

        entry["decision"] = decide(arr)
        per_adapter[name] = entry

    out["adapters"] = per_adapter
    best = max(
        (n for n in per_adapter if n != "identity"),
        key=lambda n: per_adapter[n]["arr"],
        default=None,
    )
    out["best_adapter"] = best
    out["best_arr"] = per_adapter[best]["arr"] if best else None
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3 — hot-path latency
# ─────────────────────────────────────────────────────────────────────────────


def experiment_apply_latency(
    *, dim: int = 384, repeats: int = 2000, batch: int = 256, seed: int = 0
) -> dict[str, Any]:
    """The ``Bridge.to_index_space()`` budget: under 15 µs single, under 1.5 ms for 256.

    This path runs in the HOT path of a RAG request, where logging, dict copying
    and validation are all forbidden. The measurement is therefore bare:
    ``apply`` plus renormalisation, nothing else.

    Measured on CPU: for a single query the host→device→host transfer alone
    exceeds the budget, so a GPU would slow this down rather than speed it up.
    """
    console.rule(f"[bold]Experiment 3 — hot-path latency (CPU, d={dim})")
    rng = np.random.default_rng(seed)
    src = l2_normalize(as_f32(rng.standard_normal((4000, dim))))
    dst = l2_normalize(as_f32(rng.standard_normal((4000, dim))))

    adapters: dict[str, Adapter] = {
        "procrustes": ProcrustesAdapter.fit(src, dst),
        "linear": LinearAdapter.fit(src, dst),
        "low_rank_affine": LowRankAffineAdapter.fit(src, dst, rank=64),
    }
    adapters["procrustes_centered"] = CenteredProcrustesAdapter.fit(src, dst)
    adapters["procrustes_centered+dsm"] = ScaledAdapter.fit(
        adapters["procrustes_centered"], src, dst
    )
    try:
        adapters["residual_mlp"] = ResidualMLPAdapter.fit(src, dst, epochs=5, device="cpu")
    except Exception as exc:
        console.print(f"  [yellow]![/yellow] residual_mlp skipped: {exc}")

    q1 = l2_normalize(as_f32(rng.standard_normal((1, dim))))
    qb = l2_normalize(as_f32(rng.standard_normal((batch, dim))))

    out: dict[str, Any] = {"dim": dim, "batch": batch, "repeats": repeats, "adapters": {}}
    table = Table(title=f"apply latency (d={dim}, CPU)")
    table.add_column("adapter")
    table.add_column("memory", justify="right")
    table.add_column("single query", justify="right")
    table.add_column("budget 15 µs", justify="center")
    table.add_column(f"batch {batch}", justify="right")
    table.add_column("budget 1.5 ms", justify="center")

    for name, ad in adapters.items():
        for _ in range(50):  # warm-up — the first calls include BLAS init
            ad.apply(q1)

        samples = []
        for _ in range(7):
            t0 = time.perf_counter()
            for _ in range(repeats):
                l2_normalize(ad.apply(q1), copy=False)
            samples.append((time.perf_counter() - t0) / repeats * 1e6)
        single_us = statistics.median(samples)

        for _ in range(10):
            ad.apply(qb)
        samples_b = []
        for _ in range(7):
            t0 = time.perf_counter()
            for _ in range(50):
                l2_normalize(ad.apply(qb), copy=False)
            samples_b.append((time.perf_counter() - t0) / 50 * 1e3)
        batch_ms = statistics.median(samples_b)

        ok1 = single_us < 15.0
        ok2 = batch_ms < 1.5
        out["adapters"][name] = {
            "memory_mb": round(ad.memory_mb, 3),
            "single_query_us": round(single_us, 2),
            "single_within_budget": ok1,
            f"batch{batch}_ms": round(batch_ms, 3),
            "batch_within_budget": ok2,
        }
        table.add_row(
            name,
            f"{ad.memory_mb:.2f} MB",
            f"{single_us:.2f} µs",
            "[green]✓[/green]" if ok1 else "[red]✗[/red]",
            f"{batch_ms:.3f} ms",
            "[green]✓[/green]" if ok2 else "[red]✗[/red]",
        )
    console.print(table)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4 — sample-size curve (reference: 0.97+ at 5,000 pairs, saturating at 16,000)
# ─────────────────────────────────────────────────────────────────────────────


def experiment_sample_curve(
    corpus: Corpus,
    old_id: str,
    new_id: str,
    *,
    cache_dir: Path,
    sizes: Sequence[int],
    device: str | None = None,
    k: int = 10,
    heldout_size: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """How many matched pairs are needed? This is the ``--pairs`` advice the tool gives.

    The reference work used d=768 and different corpora; our numbers are not
    expected to match. What matters is the SHAPE of the curve — where it flattens.
    """
    console.rule("[bold]Experiment 4 — sample-size curve")
    curve: list[dict[str, Any]] = []
    for size in sizes:
        if size > corpus.n_docs - heldout_size:
            console.print(f"  [dim]{size} pairs do not fit this corpus, skipped[/dim]")
            continue
        res = experiment_model_pair(
            corpus,
            old_id,
            new_id,
            cache_dir=cache_dir,
            device=device,
            k=k,
            fit_size=size,
            heldout_size=heldout_size,
            seed=seed,
            use_mlp=False,  # OP/LA suffice for the curve; MLP adds minutes per point
        )
        point = {
            "n_pairs": size,
            "t0_best_arr": res["t0"]["best_arr"],
            "t0_procrustes_arr": res["t0"]["adapters"]["procrustes"]["arr"],
            "t0_linear_arr": res["t0"]["adapters"]["linear"]["arr"],
            "t1_best_arr": res["t1"]["best_arr"] if res["t1"] else None,
        }
        curve.append(point)
        console.print(
            f"  [bold]{size:>6}[/bold] pairs → T0 ARR {point['t0_best_arr']:.4f}"
            + (f" · T1 ARR {point['t1_best_arr']:.4f}" if point["t1_best_arr"] else "")
        )
    return {"old_model": old_id, "new_model": new_id, "curve": curve}


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


def print_pair_summary(res: dict[str, Any]) -> None:
    """M0's headline output: T0 and T1 side by side, decisions included."""
    t0, t1 = res["t0"], res["t1"]
    table = Table(
        title=f"{res['old_model'].split('/')[-1]} → {res['new_model'].split('/')[-1]}"
        f"  ·  {res['n_fit_pairs']} pairs  ·  k={res['k']}"
    )
    table.add_column("adapter")
    table.add_column("memory", justify="right")
    table.add_column("T0 ARR", justify="right")
    table.add_column("T0 +CSLS", justify="right")
    table.add_column("T1 ARR", justify="right")
    table.add_column("T1 +CSLS", justify="right")
    table.add_column("T0 decision")
    table.add_column("T1 decision")

    def fmt(v: float | None, *, delta: float | None = None) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "[dim]—[/dim]"
        s = f"{v:.4f}"
        if delta is not None and abs(delta) >= 5e-5:
            colour = "green" if delta > 0 else "red"
            s += f" [{colour}]{delta:+.4f}[/{colour}]"
        return s

    for name in t0["adapters"]:
        e0 = t0["adapters"][name]
        e1 = t1["adapters"].get(name) if t1 else None
        d0 = e0["decision"]["decision"]
        d1 = e1["decision"]["decision"] if e1 else None
        agree = (d1 is None) or (d0 == d1)
        size = res["adapter_size"].get(name, {}).get("memory_mb", 0.0)
        table.add_row(
            name,
            f"{size:.2f} MB",
            fmt(e0["arr"]),
            fmt(e0["arr_csls"], delta=e0["csls_delta"]),
            fmt(e1["arr"]) if e1 else "[dim]—[/dim]",
            fmt(e1["arr_csls"], delta=e1["csls_delta"]) if e1 else "[dim]—[/dim]",
            d0,
            (d1 if agree else f"[red]{d1}[/red]") if d1 else "[dim]—[/dim]",
        )
    console.print(table)

    console.print(
        f"  baseline — T0 old-model-only (drift severity): "
        f"[bold]{t0['naive_overlap_at_k']:.4f}[/bold]"
    )
    if "unadapted" in t0:
        console.print(
            f"  baseline — T0 unadapted (raw new vector into old index): "
            f"[bold]{t0['unadapted']['recall_at_k']:.4f}[/bold]"
        )
    if t1:
        console.print(
            f"  T1 oracle (full reindex) recall@{res['k']}: "
            f"[bold]{t1['oracle_new_model']['recall_at_k']:.4f}[/bold]"
            f"  ·  old-model-only ARR: [bold]{t1['old_model_arr']:.4f}[/bold]"
        )
    if res["asymmetric"]:
        costs = {
            n: e["profile_mismatch_cost"]
            for n, e in t0["adapters"].items()
            if "profile_mismatch_cost" in e
        }
        if costs:
            worst = max(costs.values())
            console.print(
                f"  [bold]profile mismatch[/bold] — T0 ARR cost of applying g_doc "
                f"to queries: at most [bold]{worst:+.4f}[/bold] "
                f"({', '.join(f'{n} {v:+.4f}' for n, v in costs.items())})"
            )


def summarize_t0_validity(pair_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Answer M0's question 1: does the T0 proxy predict T1?

    Two distinct questions, not to be conflated:

    * NUMERICAL closeness — how large is ``|ARR_T0 − ARR_T1|``?
    * DECISION agreement — do the two land in the same decision band?

    The second matters more than the first. The user never sees ARR's decimals;
    they see "bridge" or "reindex".
    """
    rows: list[dict[str, Any]] = []
    for res in pair_results:
        if not res.get("t1"):
            continue
        for name, e0 in res["t0"]["adapters"].items():
            e1 = res["t1"]["adapters"].get(name)
            if not e1:
                continue
            rows.append(
                {
                    "pair": f"{res['old_model'].split('/')[-1]}→{res['new_model'].split('/')[-1]}",
                    "corpus": res["corpus"],
                    "adapter": name,
                    "arr_t0": e0["arr"],
                    "arr_t0_gt1": e0.get("arr_gt1"),
                    "arr_t0_gt3": e0.get("arr_gt3"),
                    "arr_t1": e1["arr"],
                    "abs_error_gt1": (
                        abs(e0["arr_gt1"] - e1["arr"]) if e0.get("arr_gt1") is not None else None
                    ),
                    "decision_t0_gt1": (
                        decide(e0["arr_gt1"])["decision"] if e0.get("arr_gt1") is not None else None
                    ),
                    "abs_error": abs(e0["arr"] - e1["arr"]),
                    "signed_error": e0["arr"] - e1["arr"],
                    "decision_t0": e0["decision"]["decision"],
                    "decision_t1": e1["decision"]["decision"],
                    "decisions_agree": e0["decision"]["decision"] == e1["decision"]["decision"],
                }
            )

    if not rows:
        return {"n": 0, "note": "no model pair has T1 data"}

    t0_vals = [r["arr_t0"] for r in rows]
    t1_vals = [r["arr_t1"] for r in rows]
    summary: dict[str, Any] = {
        "n": len(rows),
        "mean_abs_error": float(np.mean([r["abs_error"] for r in rows])),
        "max_abs_error": float(np.max([r["abs_error"] for r in rows])),
        "mean_signed_error": float(np.mean([r["signed_error"] for r in rows])),
        "decision_agreement_rate": float(np.mean([r["decisions_agree"] for r in rows])),
        "rows": rows,
    }

    # Measurement uncertainty — how narrow are the decision bands (width 0.10)
    # relative to it?
    hw_t0 = [
        e0.get("arr_gt1_ci_halfwidth")
        for res in pair_results
        for e0 in res["t0"]["adapters"].values()
        if e0.get("arr_gt1_ci_halfwidth") is not None
    ]
    hw_t1 = [
        e1.get("arr_ci_halfwidth")
        for res in pair_results
        if res.get("t1")
        for e1 in res["t1"]["adapters"].values()
        if e1.get("arr_ci_halfwidth") is not None
    ]
    if hw_t0 or hw_t1:
        summary["measurement_uncertainty"] = {
            "t0_gt1_ci_halfwidth_median": float(np.median(hw_t0)) if hw_t0 else None,
            "t0_gt1_ci_halfwidth_max": float(np.max(hw_t0)) if hw_t0 else None,
            "t1_arr_ci_halfwidth_median": float(np.median(hw_t1)) if hw_t1 else None,
            "t1_arr_ci_halfwidth_max": float(np.max(hw_t1)) if hw_t1 else None,
            "decision_band_width": 0.10,
        }

    gt1 = [r for r in rows if r.get("arr_t0_gt1") is not None]
    if gt1:
        summary["gt1_variant"] = {
            "mean_abs_error": float(np.mean([r["abs_error_gt1"] for r in gt1])),
            "max_abs_error": float(np.max([r["abs_error_gt1"] for r in gt1])),
            "mean_signed_error": float(np.mean([r["arr_t0_gt1"] - r["arr_t1"] for r in gt1])),
            "decision_agreement_rate": float(
                np.mean([r["decision_t0_gt1"] == r["decision_t1"] for r in gt1])
            ),
        }
        xs = [r["arr_t0_gt1"] for r in gt1]
        ys = [r["arr_t1"] for r in gt1]
        if len(gt1) >= 3 and len(set(xs)) > 1 and len(set(ys)) > 1:
            summary["gt1_variant"]["pearson_r"] = float(scipy.stats.pearsonr(xs, ys).statistic)
            summary["gt1_variant"]["spearman_r"] = float(scipy.stats.spearmanr(xs, ys).statistic)

    if len(rows) >= 3 and len(set(t0_vals)) > 1 and len(set(t1_vals)) > 1:
        summary["pearson_r"] = float(scipy.stats.pearsonr(t0_vals, t1_vals).statistic)
        summary["spearman_r"] = float(scipy.stats.spearmanr(t0_vals, t1_vals).statistic)
    return summary


def summarize_csls(pair_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Answer M0's question 2: does CSLS improve ARR?"""
    deltas: dict[str, list[float]] = {"t0": [], "t1": []}
    per_adapter: dict[str, list[float]] = {}
    for res in pair_results:
        for tier in ("t0", "t1"):
            block = res.get(tier)
            if not block:
                continue
            for name, e in block["adapters"].items():
                if name == "identity":
                    continue
                d = e.get("csls_delta")
                if d is None or np.isnan(d):
                    continue
                deltas[tier].append(d)
                per_adapter.setdefault(name, []).append(d)

    out: dict[str, Any] = {}
    for tier, vals in deltas.items():
        if vals:
            out[tier] = {
                "n": len(vals),
                "mean_delta": float(np.mean(vals)),
                "median_delta": float(np.median(vals)),
                "max_delta": float(np.max(vals)),
                "min_delta": float(np.min(vals)),
                "improved_fraction": float(np.mean([v > 0 for v in vals])),
            }
    out["per_adapter_mean_delta"] = {n: float(np.mean(v)) for n, v in sorted(per_adapter.items())}

    # Relationship between the CSLS correction and adapter quality.
    # Hypothesis: hubness forms when the adapter maps poorly — a weak transform
    # crowds documents into a narrow region and a few become everyone's
    # neighbour. If true, the CSLS gain correlates NEGATIVELY with ARR: much for
    # a bad adapter, little for a good one. That distinguishes "always enable
    # CSLS" from "CSLS is a safety net".
    pts: list[tuple[float, float]] = []
    for res in pair_results:
        for tier in ("t0", "t1"):
            block = res.get(tier)
            if not block:
                continue
            for name, e in block["adapters"].items():
                d = e.get("csls_delta")
                if d is None or np.isnan(d) or np.isnan(e["arr"]):
                    continue
                pts.append((e["arr"], d))
    if len(pts) >= 4:
        xs = [a for a, _ in pts]
        ys = [b for _, b in pts]
        if len(set(xs)) > 1 and len(set(ys)) > 1:
            out["delta_vs_arr_pearson"] = float(scipy.stats.pearsonr(xs, ys).statistic)
            out["delta_vs_arr_spearman"] = float(scipy.stats.spearmanr(xs, ys).statistic)
        weak = [b for a, b in pts if a < 0.5]
        strong = [b for a, b in pts if a >= 0.8]
        out["mean_delta_weak_adapters"] = float(np.mean(weak)) if weak else None
        out["mean_delta_strong_adapters"] = float(np.mean(strong)) if strong else None
        out["n_weak"] = len(weak)
        out["n_strong"] = len(strong)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


DEFAULT_PAIRS: tuple[tuple[str, str], ...] = (
    # Symmetric → asymmetric, same dimension. The "trap" scenario:
    # if prefix handling breaks, ARR collapses; handled correctly it stays high.
    ("sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"),
    # Symmetric → asymmetric, different family and a different prefix scheme
    ("sentence-transformers/all-MiniLM-L6-v2", "intfloat/e5-small-v2"),
    # Asymmetric → asymmetric: both sides prefixed, both schemes different
    ("intfloat/e5-small-v2", "BAAI/bge-small-en-v1.5"),
)


def resolve_device(spec: str) -> str | None:
    """Defensive device detection: a detection failure never crashes, it falls back to CPU."""
    if spec == "cpu":
        return "cpu"
    try:
        import torch

        if spec == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if spec.startswith("cuda") and torch.cuda.is_available():
            return spec
        if spec == "mps" and torch.backends.mps.is_available():
            return "mps"
        console.print(f"  [yellow]![/yellow] device {spec!r} unavailable → cpu")
        return "cpu"
    except Exception as exc:
        console.print(f"  [yellow]![/yellow] device detection failed ({exc}) → cpu")
        return "cpu"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="m0_spike",
        description="rebasis M0 spike — T0 validity, CSLS contribution, budget calibration",
    )
    parser.add_argument(
        "experiment",
        choices=["synthetic", "pairs", "latency", "curve", "all"],
        help="which experiment to run",
    )
    parser.add_argument(
        "--dataset",
        default="beir/scifact/test",
        help="comma-separated; several corpora avoid generalising from one",
    )
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:N|mps")
    parser.add_argument(
        "--mlp-device",
        default=None,
        help="device for MLP fitting; defaults to --device (separate for the device breakdown)",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--fit-size", type=int, default=None, help="default: all remaining documents"
    )
    parser.add_argument("--heldout-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-mlp", action="store_true", help="verify the torch-free path")
    parser.add_argument("--csls-k", type=int, default=10)
    parser.add_argument(
        "--latency-dims",
        default="384,768",
        help="dimensions to measure latency at; the budget is defined for d=768",
    )
    parser.add_argument(
        "--curve-sizes", default="500,1000,2000,4000", help="sample-curve points, comma-separated"
    )
    parser.add_argument("--cache-dir", default=".rebasis-m0-cache")
    parser.add_argument("--out", default="reports", help="directory for the JSON report")
    parser.add_argument("--tag", default=None, help="tag appended to the report filename")
    args = parser.parse_args(argv)

    host = describe_host()
    console.rule("[bold cyan]rebasis M0 spike")
    console.print(
        f"  {host['platform']}  ·  {host['cpu_count_logical']} vCPU  ·  "
        f"{host['ram_total_gb']} GB RAM  ·  Python {host['python']}"
    )
    if host.get("torch"):
        acc = (
            f"cuda ({host.get('cuda_device')})"
            if host.get("torch_cuda_available")
            else ("mps" if host.get("torch_mps_available") else "none")
        )
        console.print(f"  torch {host['torch']}  ·  accelerator: {acc}")
        if host.get("torch_cuda_available"):
            # TF32 must be OFF on measurement paths. The measured deviation
            # (0.106 with TF32 on, 0.001 off) is the justification.
            import torch

            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            console.print("  [dim]TF32 disabled on measurement paths[/dim]")
            host["tf32_disabled_for_measurement"] = True

    device = resolve_device(args.device)
    mlp_device = args.mlp_device or (device or "cpu")
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "host": host,
        "args": vars(args),
        "device": device,
        "mlp_device": mlp_device,
    }
    want = args.experiment

    if want in ("synthetic", "all"):
        report["synthetic"] = experiment_synthetic(seed=args.seed)
        if not report["synthetic"]["all_passed"]:
            console.print(
                "\n[red bold]Synthetic validation FAILED.[/red bold] "
                "If the adapter maths is broken, no number measured on a real "
                "corpus is trustworthy — stopping."
            )
            _write_report(report, out_dir, args.tag)
            return 1

    if want in ("latency", "all"):
        report["latency_by_dim"] = {}
        for dim in [int(s) for s in args.latency_dims.split(",") if s.strip()]:
            report["latency_by_dim"][str(dim)] = experiment_apply_latency(dim=dim, seed=args.seed)
        # The budget is defined for d=768; the verdict comes from there.
        report["latency"] = report["latency_by_dim"].get(
            "768", next(iter(report["latency_by_dim"].values()))
        )

    dataset_names = [s.strip() for s in args.dataset.split(",") if s.strip()]
    corpora: list[Corpus] = []
    if want in ("pairs", "curve", "all"):
        for name in dataset_names:
            with measure(f"corpus:load:{name}"):
                c = load_beir(name, max_docs=args.max_docs)
            console.print(
                f"  corpus [bold]{c.name}[/bold]: {c.n_docs} documents · "
                f"{len(c.query_ids)} real queries · "
                f"{sum(len(v) for v in c.qrels.values())} qrels"
            )
            corpora.append(c)

    if want in ("pairs", "all") and corpora:
        pair_results = []
        for c in corpora:
            # The held-out query set comes out of the corpus; a fixed 1000 on a
            # small corpus would empty the fit set.
            heldout = min(args.heldout_size, max(200, c.n_docs // 5))
            if heldout != args.heldout_size:
                console.print(
                    f"  [dim]{c.name}: held-out queries {args.heldout_size} → {heldout} "
                    f"(corpus has {c.n_docs} documents)[/dim]"
                )
            for old_id, new_id in DEFAULT_PAIRS:
                res = experiment_model_pair(
                    c,
                    old_id,
                    new_id,
                    cache_dir=cache_dir,
                    device=device,
                    mlp_device=mlp_device,
                    k=args.k,
                    fit_size=args.fit_size,
                    heldout_size=heldout,
                    seed=args.seed,
                    use_mlp=not args.no_mlp,
                    csls_k=args.csls_k,
                )
                print_pair_summary(res)
                pair_results.append(res)
        report["pairs"] = pair_results
        report["t0_validity"] = summarize_t0_validity(pair_results)
        report["csls"] = summarize_csls(pair_results)
        _print_verdicts(report)

    if want in ("curve", "all") and corpora:
        corpus = corpora[0]
        sizes = [int(s) for s in args.curve_sizes.split(",") if s.strip()]
        report["sample_curve"] = experiment_sample_curve(
            corpus,
            *DEFAULT_PAIRS[0],
            cache_dir=cache_dir,
            sizes=sizes,
            device=device,
            k=args.k,
            heldout_size=args.heldout_size,
            seed=args.seed,
        )

    report["timings"] = [asdict(t) for t in _TIMINGS]
    path = _write_report(report, out_dir, args.tag)
    console.print(f"\n[green]✓[/green] report written: [bold]{path}[/bold]")
    return 0


def _print_verdicts(report: dict[str, Any]) -> None:
    """Collect M0's answers to its three questions in one place."""
    console.rule("[bold cyan]M0 verdicts")

    v = report.get("t0_validity", {})
    if v.get("n"):
        console.print("[bold]1. Is the T0 proxy valid?[/bold]")
        console.print(f"   configurations compared        : {v['n']}")
        console.print(f"   mean |ARR_T0 − ARR_T1|         : {v['mean_abs_error']:.4f}")
        console.print(f"   worst |ARR_T0 − ARR_T1|        : {v['max_abs_error']:.4f}")
        bias = v["mean_signed_error"]
        label = (
            "→ T0 OPTIMISTIC"
            if bias > 0.01
            else ("→ T0 PESSIMISTIC" if bias < -0.01 else "→ no systematic bias")
        )
        console.print(f"   signed error (T0 − T1)         : {bias:+.4f}  {label}")
        rate = v["decision_agreement_rate"]
        colour = "green" if rate >= 0.9 else ("yellow" if rate >= 0.7 else "red")
        console.print(
            f"   [bold]DECISION AGREEMENT[/bold]             : [{colour}]{rate:.1%}[/{colour}]"
        )
        if "pearson_r" in v:
            console.print(
                f"   Pearson r / Spearman ρ         : {v['pearson_r']:.3f} / {v['spearman_r']:.3f}"
            )
        g = v.get("gt1_variant")
        if g:
            console.print(
                "\n   [bold]decomposition[/bold] — measure T0 with T1's metric shape "
                "(ground truth = nearest neighbour only, matching qrel sparsity):"
            )
            console.print(f"   mean |ARR_T0ᵍᵗ¹ − ARR_T1|      : {g['mean_abs_error']:.4f}")
            console.print(f"   signed error                   : {g['mean_signed_error']:+.4f}")
            rate2 = g["decision_agreement_rate"]
            colour2 = "green" if rate2 >= 0.9 else ("yellow" if rate2 >= 0.7 else "red")
            console.print(
                f"   [bold]DECISION AGREEMENT (relaxed)[/bold]   : [{colour2}]{rate2:.1%}[/{colour2}]"
            )
            if "spearman_r" in g:
                console.print(
                    f"   Pearson r / Spearman ρ         : {g['pearson_r']:.3f} / {g['spearman_r']:.3f}"
                )
        u = v.get("measurement_uncertainty")
        if u:
            console.print(
                "\n   [bold]measurement uncertainty[/bold] (bootstrap 95% CI half-width):"
            )
            if u.get("t0_gt1_ci_halfwidth_median") is not None:
                console.print(
                    f"   T0 ARR   median ±{u['t0_gt1_ci_halfwidth_median']:.4f} · "
                    f"worst ±{u['t0_gt1_ci_halfwidth_max']:.4f}"
                )
            if u.get("t1_arr_ci_halfwidth_median") is not None:
                console.print(
                    f"   T1 ARR   median ±{u['t1_arr_ci_halfwidth_median']:.4f} · "
                    f"worst ±{u['t1_arr_ci_halfwidth_max']:.4f}"
                )
            console.print(
                f"   decision band width            : {u['decision_band_width']:.2f}"
                "   [dim](when uncertainty approaches the band, the threshold call is "
                "decided by sampling noise)[/dim]"
            )

    c = report.get("csls", {})
    if c.get("t0") or c.get("t1"):
        console.print("\n[bold]2. Does CSLS improve ARR?[/bold]")
        for tier in ("t0", "t1"):
            if tier not in c:
                continue
            b = c[tier]
            verdict = (
                "[green]helps[/green]"
                if b["mean_delta"] > 0.002
                else (
                    "[red]hurts[/red]" if b["mean_delta"] < -0.002 else "[yellow]neutral[/yellow]"
                )
            )
            console.print(
                f"   {tier.upper()}: mean Δ = {b['mean_delta']:+.4f} · "
                f"median {b['median_delta']:+.4f} · "
                f"range [{b['min_delta']:+.4f}, {b['max_delta']:+.4f}] · "
                f"improved {b['improved_fraction']:.0%} → {verdict}"
            )
        if c.get("mean_delta_weak_adapters") is not None:
            strong = (
                f"{c['mean_delta_strong_adapters']:+.4f} (n={c['n_strong']})"
                if c.get("mean_delta_strong_adapters") is not None
                else "—"
            )
            console.print(
                f"   weak adapters (ARR<0.5) Δ = {c['mean_delta_weak_adapters']:+.4f} "
                f"(n={c['n_weak']}) · strong (ARR≥0.8) Δ = {strong}"
            )
        if "delta_vs_arr_spearman" in c:
            note = (
                "→ CSLS is a SAFETY NET: much for a bad adapter, little for a good one"
                if c["delta_vs_arr_spearman"] < -0.3
                else ""
            )
            console.print(
                f"   Δ ↔ ARR correlation (Spearman) : {c['delta_vs_arr_spearman']:+.3f}  {note}"
            )

    cal_rows = [
        (name, e)
        for res in report.get("pairs", [])
        for name, e in (res["t0"].get("calibration", {}).get("adapters", {}) or {}).items()
    ]
    if cal_rows:
        before = [e["score_shift_before"] for _, e in cal_rows]
        after = [e["score_shift_after"] for _, e in cal_rows]
        passing = [e["passes_threshold_after"] for _, e in cal_rows]
        ranks = [e["rank_preserved"] for _, e in cal_rows]
        console.print("\n[bold]4. Does isotonic calibration close the score shift?[/bold]")
        console.print(
            f"   score_shift (KS) BEFORE calibration : median {np.median(before):.3f} "
            f"(threshold is 0.1 — {np.mean([b > 0.1 for b in before]):.0%} exceed it)"
        )
        console.print(
            f"   score_shift (KS) AFTER calibration  : median {np.median(after):.3f} "
            f"({np.mean(passing):.0%} below the threshold)"
        )
        console.print(f"   ranking preserved (monotonicity)    : {np.mean(ranks):.0%}")

    lat = report.get("latency", {})
    if lat:
        over = [n for n, e in lat["adapters"].items() if not e["single_within_budget"]]
        console.print("\n[bold]3. Does the hot-path budget hold?[/bold]")
        console.print(
            "   single query under 15 µs: "
            + ("[green]all passed[/green]" if not over else f"[red]over: {', '.join(over)}[/red]")
        )


def _write_report(report: dict[str, Any], out_dir: Path, tag: str | None) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"m0-{tag}-{stamp}.json" if tag else f"m0-{stamp}.json"
    path = out_dir / name
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return path


if __name__ == "__main__":
    sys.exit(main())
