"""Device breakdown — the same work, two devices, one machine.

Measuring on separate machines and comparing would be misleading; the CPU and
CUDA numbers here come from the SAME host, the same corpus and the same run.
This is the empirical basis for deciding which work belongs on the GPU.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m0_spike import (
    PROFILES,
    LinearAdapter,
    ProcrustesAdapter,
    ResidualMLPAdapter,
    SentenceTransformerEmbedder,
    as_f32,
    l2_normalize,
    load_beir,
    topk_search,
)

# TF32 off on measurement paths
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

MAX_DOCS = 4000
rows: list[tuple[str, str, float, str]] = []


def record(job: str, device: str, seconds: float, extra: str = "") -> None:
    rows.append((job, device, seconds, extra))
    print(f"  {job:<38} {device:<6} {seconds:>8.2f}s  {extra}")


def main() -> None:
    print(f"host: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU'}")
    corpus = load_beir("beir/scidocs", max_docs=MAX_DOCS)
    texts = corpus.doc_texts
    print(f"corpus: {len(texts)} documents\n")

    print("── embedding generation (expected GPU gain: 'very high') ──")
    for model_id in (
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-small-en-v1.5",
    ):
        short = model_id.split("/")[-1]
        for device in ("cpu", "cuda"):
            if device == "cuda" and not torch.cuda.is_available():
                continue
            emb = SentenceTransformerEmbedder(PROFILES[model_id], device=device)
            emb.encode(texts[:64], kind="document", batch_size=64)  # warm-up
            t0 = time.perf_counter()
            emb.encode(texts, kind="document", batch_size=64)
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            record(f"embedding {short}", device, dt, f"{len(texts) / dt:.0f} docs/s")
            del emb

    rng = np.random.default_rng(0)

    print("\n── adapter fitting ──")
    for dim in (384, 768):
        n = 20000
        src = l2_normalize(as_f32(rng.standard_normal((n, dim))))
        dst = l2_normalize(as_f32(rng.standard_normal((n, dim))))

        # Orthogonal Procrustes fit — no GPU gain expected, always CPU
        t0 = time.perf_counter()
        ProcrustesAdapter.fit(src, dst)
        record(f"OP fit ({n} pairs, d={dim})", "cpu", time.perf_counter() - t0)

        t0 = time.perf_counter()
        LinearAdapter.fit(src, dst)
        record(f"Linear fit ({n} pairs, d={dim})", "cpu", time.perf_counter() - t0)

        for device in ("cpu", "cuda"):
            if device == "cuda" and not torch.cuda.is_available():
                continue
            t0 = time.perf_counter()
            ResidualMLPAdapter.fit(src, dst, device=device, epochs=30)
            if device == "cuda":
                torch.cuda.synchronize()
            record(f"MLP fit ({n} pairs, d={dim})", device, time.perf_counter() - t0)

    print("\n── ground truth kNN (expected 'borderline', threshold to be measured) ──")
    for n_docs in (10_000, 50_000, 200_000):
        n_q = 10_000
        q = l2_normalize(as_f32(rng.standard_normal((n_q, 384))))
        docs = l2_normalize(as_f32(rng.standard_normal((n_docs, 384))))

        t0 = time.perf_counter()
        topk_search(q, docs, k=10)
        record(f"kNN {n_q}x{n_docs} chunked numpy", "cpu", time.perf_counter() - t0)

        if torch.cuda.is_available():
            qt = torch.from_numpy(q).cuda()
            dt_ = torch.from_numpy(docs).cuda()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            # Same memory discipline: the full score matrix is never materialised
            for start in range(0, n_q, 2048):
                torch.topk(qt[start : start + 2048] @ dt_.T, k=10, dim=1)
            torch.cuda.synchronize()
            record(f"kNN {n_q}x{n_docs} chunked torch", "cuda", time.perf_counter() - t0)
            del qt, dt_
            torch.cuda.empty_cache()

    print("\n── summary: GPU/CPU speedup ──")
    by_job: dict[str, dict[str, float]] = {}
    for job, device, seconds, _ in rows:
        by_job.setdefault(job, {})[device] = seconds
    for job, devs in by_job.items():
        if "cpu" in devs and "cuda" in devs:
            ratio = devs["cpu"] / devs["cuda"]
            verdict = (
                "move to GPU" if ratio > 2.0 else ("borderline" if ratio > 1.2 else "keep on CPU")
            )
            print(f"  {job:<38} {ratio:>6.2f}×   {verdict}")


if __name__ == "__main__":
    main()
