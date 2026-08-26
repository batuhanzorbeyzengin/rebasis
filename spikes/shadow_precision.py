"""What a float16 shadow copy gives up, measured.

``ShadowStore`` has taken a ``precision`` argument since it was written and
nothing has ever passed ``float16``. The roadmap says why: it halves the shadow's
disk cost and gives up the bit-identical rollback guarantee, and **a half
guarantee may be more dangerous than no guarantee**. The plumbing was left in
place and the option withheld until that was settled.

This settles it, by measuring the only thing that decides it: **after a rollback
from a float16 shadow, does the index return what it returned before the
migration?** Not "how big is the numeric error" — an error nobody can retrieve a
different document because of is not a cost — but whether the ranking moves.

Three things per corpus and model:

**Does anything break outright.** float16 tops out at 65504 and flushes below
about 6e-8. A unit vector cannot overflow; an unnormalised one can, and this
ladder has both — SentenceTransformer models here are normalised and
model2vec's static models are not, so the question is measured rather than
assumed for each.

**Does the ranking move.** Top-k overlap between the index as it was and the
index a rollback would leave, over the corpus' own real queries, plus the share
of queries whose first hit changes. This is the guarantee in the form a user
holds it.

**Does quality move.** nDCG@10 against human judgements, both indexes, scored
with `ranx` — so the answer sits beside every other number this project
publishes rather than floating free.

    PYTHONPATH=src python spikes/shadow_precision.py \\
        --corpus heldout --corpus beir --out reports/shadow/rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rebasis.types import FloatArray

#: The band harness owns corpus loading, the ladders and the embedding cache.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

#: Cut-offs the overlap is reported at. 1 because a changed first hit is the
#: most visible thing a rollback could do, 10 because that is the depth every
#: other measurement here uses.
CUTOFFS = (1, 10)


def band() -> Any:
    """The band harness, imported late so `--help` costs nothing."""
    import bridge_band

    return bridge_band


def round_trip(vectors: FloatArray) -> FloatArray:
    """Exactly what a float16 shadow hands back.

    ``ShadowStore`` casts on write and casts back on read; nothing else in the
    path touches the values, so this is the whole of the loss.
    """
    return np.asarray(vectors.astype(np.float16), dtype=np.float32)


def numeric(original: FloatArray, restored: FloatArray) -> dict[str, Any]:
    """How far the values moved, and whether any of them left the format.

    ``flushed`` counts components that were non-zero and came back zero;
    ``overflowed`` counts ones that came back infinite. The second is not a
    precision loss but a destroyed vector, which is why it is counted
    separately rather than folded into a mean error.
    """
    finite = np.isfinite(restored)
    error = np.abs(original - np.where(finite, restored, 0.0))
    magnitude = np.abs(original)
    norms = np.linalg.norm(original, axis=1)
    return {
        "abs_error_max": float(error.max()),
        "abs_error_mean": float(error.mean()),
        # Relative to the vector norm rather than to each component: a component
        # near zero has an unbounded relative error and says nothing about
        # whether the vector still points where it did.
        "error_over_norm_max": float((np.linalg.norm(original - restored, axis=1) / norms).max()),
        "flushed_fraction": float(((magnitude > 0) & (np.abs(restored) == 0)).mean()),
        "overflowed": int((~finite).sum()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "unit_vectors": bool(np.allclose(norms, 1.0, atol=1e-3)),
    }


def ranking(
    original: FloatArray,
    restored: FloatArray,
    queries: FloatArray,
    self_mask: np.ndarray | None,
) -> dict[str, Any]:
    """How much of the ranking survives the round trip."""
    from rebasis.compute import top_k_search
    from rebasis.probe.metrics import overlap_at_k

    depth = max(CUTOFFS)
    before, _ = top_k_search(queries, original, k=depth, self_mask=self_mask)
    after, _ = top_k_search(queries, restored, k=depth, self_mask=self_mask)
    result: dict[str, Any] = {
        f"overlap@{k}": round(overlap_at_k(before, after, k), 6) for k in CUTOFFS
    }
    result["top1_changed"] = round(float((before[:, 0] != after[:, 0]).mean()), 6)
    result["identical_rankings"] = round(float((before == after).all(axis=1).mean()), 6)
    return result


def measure(
    corpus: Any,
    model_id: str,
    *,
    cache_dir: Path,
    device: str,
    encoder_cache: dict[str, Any],
) -> dict[str, Any]:
    """One corpus under one model: the index as it is, and after a round trip."""
    b = band()
    started = time.perf_counter()
    encoded = b.encode_corpus(
        model_id=model_id,
        corpus=corpus,
        cache_dir=cache_dir,
        device=device,
        encoder_cache=encoder_cache,
    )
    original = encoded.documents
    restored = round_trip(original)

    runs = {
        "before": b_search(original, encoded.queries, corpus),
        "after": b_search(restored, encoded.queries, corpus),
    }
    scored = b.score(corpus, runs, cutoffs=[10])

    return {
        "corpus": corpus.name,
        "model": model_id,
        "n_documents": len(corpus.doc_ids),
        "n_queries": len(corpus.query_ids),
        "dim": int(original.shape[1]),
        "bytes_saved_fraction": 0.5,
        "numeric": numeric(original, restored),
        "ranking": ranking(original, restored, encoded.queries, corpus.self_mask),
        "scores": scored.aggregate,
        "duration_seconds": round(time.perf_counter() - started, 1),
    }


def b_search(documents: FloatArray, queries: FloatArray, corpus: Any) -> Any:
    """A run in the shape `bridge_band.score` expects."""
    from rebasis.compute import top_k_search

    return top_k_search(queries, documents, k=100, self_mask=corpus.self_mask)


def already_done(out: Path) -> set[tuple[str, str]]:
    """Keys already in the output, so a re-run resumes instead of repeating."""
    if not out.exists():
        return set()
    return {
        (row["corpus"], row["model"])
        for row in (
            json.loads(line)
            for line in out.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the grid and append one row per corpus and model."""
    b = band()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", action="append", default=None)
    parser.add_argument("--ladder", default="default", choices=sorted(b.LADDERS))
    parser.add_argument("--out", type=Path, default=Path("reports/shadow/rows.jsonl"))
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / "band-cache")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    # Every model the ladder names, each measured once per corpus. A shadow
    # holds whatever the index holds, so the question is about a model's output
    # rather than about a migration between two of them.
    models = sorted({model for rung in b.LADDERS[args.ladder] for model in rung})
    datasets = b.resolve_corpora(args.corpus or ["beir"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(args.out)
    encoder_cache: dict[str, Any] = {}

    for dataset in datasets:
        corpus = b.load_corpus(dataset)
        print(f"\n{dataset}\n  {len(corpus.doc_ids):,} documents", flush=True)
        for model_id in models:
            if (corpus.name, model_id) in done:
                print(f"  {model_id}  (done)", flush=True)
                continue
            row = measure(
                corpus,
                model_id,
                cache_dir=args.cache_dir,
                device=args.device,
                encoder_cache=encoder_cache,
            )
            with args.out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            rank = row["ranking"]
            print(
                f"  {model_id}  unit={row['numeric']['unit_vectors']}  "
                f"overlap@10 {rank['overlap@10']:.6f}  "
                f"top1 changed {rank['top1_changed']:.6f}  "
                f"ndcg {row['scores']['before']['ndcg@10']:.4f} -> "
                f"{row['scores']['after']['ndcg@10']:.4f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
