"""Where is the GPU threshold for kNN?

An earlier measurement moved the vectors onto the GPU BEFORE starting the timer.
That is fair for the "embeddings were produced on the GPU and are still there"
case, but not for "the data is on the CPU and must be transferred". The earlier
'borderline' verdict for kNN was precisely about transfer cost, so both are
measured here.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m0_spike import as_f32, l2_normalize, topk_search

torch.backends.cuda.matmul.allow_tf32 = False

DIM, K, N_QUERIES = 384, 10, 10_000
SIZES = (2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000)


def main() -> None:
    rng = np.random.default_rng(0)
    print(
        f"{'n_docs':>9} {'CPU numpy':>11} {'GPU (resident)':>15} {'GPU (+transfer)':>16} "
        f"{'resident':>10} {'+transfer':>11} {'verdict':<14}"
    )
    print("-" * 96)

    for n_docs in SIZES:
        q = l2_normalize(as_f32(rng.standard_normal((N_QUERIES, DIM))))
        docs = l2_normalize(as_f32(rng.standard_normal((n_docs, DIM))))

        topk_search(q[:256], docs, k=K)  # warm-up
        t0 = time.perf_counter()
        topk_search(q, docs, k=K)
        cpu = time.perf_counter() - t0

        qt, dt_ = torch.from_numpy(q).cuda(), torch.from_numpy(docs).cuda()

        def gpu_knn(a: torch.Tensor, b: torch.Tensor) -> None:
            for start in range(0, a.shape[0], 2048):
                torch.topk(a[start : start + 2048] @ b.T, k=K, dim=1)

        gpu_knn(qt, dt_)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gpu_knn(qt, dt_)
        torch.cuda.synchronize()
        gpu_resident = time.perf_counter() - t0
        del qt, dt_
        torch.cuda.empty_cache()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        qt, dt_ = torch.from_numpy(q).cuda(), torch.from_numpy(docs).cuda()
        for start in range(0, qt.shape[0], 2048):
            _, idx = torch.topk(qt[start : start + 2048] @ dt_.T, k=K, dim=1)
            idx.cpu()  # results travel back to the host
        torch.cuda.synchronize()
        gpu_transfer = time.perf_counter() - t0
        del qt, dt_
        torch.cuda.empty_cache()

        r1, r2 = cpu / gpu_resident, cpu / gpu_transfer
        verdict = "move to GPU" if r2 > 2.0 else ("borderline" if r2 > 1.2 else "keep on CPU")
        print(
            f"{n_docs:>9} {cpu:>10.3f}s {gpu_resident:>14.3f}s {gpu_transfer:>15.3f}s "
            f"{r1:>9.1f}× {r2:>10.1f}× {verdict:<14}"
        )


if __name__ == "__main__":
    main()
