"""What a completed migration is actually worth, on real corpora.

`rebasis migrate` rewrites the indexed document vectors and, until now, applied
the wrong map to do it — a `query_to_old` adapter, which leaves an index no query
can answer. `fit --direction old_to_new` produces the right one. This measures
what the right one buys, which is a different question from every number the
project has published: [`bridge-band.md`](../docs/bridge-band.md) measures what a
*bridged query* retrieves against an untouched index, and this measures what a
*raw query* retrieves against a rewritten one.

Four configurations, one index, real judgements::

    status quo     old query      -> old index         what you have
    bridged        adapter(new q) -> old index         what `Bridge` serves
    migrated       new query      -> forward(old docs) what `migrate` leaves
    reindexed      new query      -> new index         the ceiling

There is no fifth row bounding `migrated` from above, and the reason is worth
stating rather than leaving as an omission: for a *document* map the best
possible result is the documents' own new-model vectors, because no map of the
old ones can beat having actually re-embedded them. So the ceiling on `migrated`
**is** `reindexed`, as an identity rather than a measurement, and the quantity
worth reading is the gap between the two. Where that gap is large,
[ADR 10](../docs/adr/0010-retention-is-bounded-by-the-source.md) is the reading:
the old space does not carry the new model's neighbourhoods forward, and no
choice of adapter family will make it.

Scored with `ranx` against human judgements, exactly as the band harness does, so
the numbers sit beside the published ones rather than floating free. The adapters
come from the same `fit_candidates` path the CLI runs, applied through the same
`BaseAdapter.apply`, so what is measured is the tool rather than a
reimplementation of it.

    PYTHONPATH=src python spikes/migration_band.py \\
        --corpus beir --ladder default --out reports/migration/rows.jsonl
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
#: Imported rather than restated so that a corpus means the same thing here as it
#: does there — including the self-removal convention, which changed a published
#: number by 0.2 nDCG the one time it was missed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

FIT_PAIRS = 4000


def band() -> Any:
    """The band harness, imported late so `--help` costs nothing."""
    import bridge_band

    return bridge_band


#: The family both directions are fitted with here.
#:
#: One method rather than `auto`, deliberately. `auto` picks by *scoring* on a
#: held-out set, and the two directions are scored on different questions — a
#: bridged query against an untouched index, against a raw query against a
#: rewritten one — so an `auto` on each side would vary the family *and* the
#: direction at once and the comparison would answer neither. This is the
#: measured default and `auto`'s usual winner (ADR 10, 15 of 15).
DEFAULT_METHOD = "procrustes_centered"


def _fit(src: FloatArray, dst: FloatArray, method: str) -> Any:
    """One adapter of one family, through the same call the CLI makes."""
    from rebasis.core import fit_candidates

    candidates = fit_candidates(src, dst, normalize=False, methods=[method])
    if not candidates:
        msg = f"{method} could not be fitted on {src.shape[0]} pairs"
        raise RuntimeError(msg)
    return candidates[0]


def measure(
    corpus: Any,
    old_model: str,
    new_model: str,
    *,
    cutoffs: Sequence[int],
    cache_dir: Path,
    device: str,
    seed: int,
    method: str,
    encoder_cache: dict[str, Any],
) -> dict[str, Any]:
    """One corpus, one model pair: fit both directions and score four rows."""
    from rebasis.compute import top_k_search
    from rebasis.core import l2_normalize

    b = band()
    started = time.perf_counter()
    shared = {
        "corpus": corpus,
        "cache_dir": cache_dir,
        "device": device,
        "encoder_cache": encoder_cache,
    }
    old = b.encode_corpus(model_id=old_model, **shared)
    new = b.encode_corpus(model_id=new_model, **shared)

    n = len(corpus.doc_ids)
    rng = np.random.default_rng(seed)
    pool = rng.permutation(n)
    fit_rows = np.sort(pool[: min(FIT_PAIRS, n)])

    # Both directions from the same pairs, so the comparison isolates the
    # direction and not the sample.
    backward = _fit(new.documents[fit_rows], old.documents[fit_rows], method)
    forward = _fit(old.documents[fit_rows], new.documents[fit_rows], method)

    depth = max(cutoffs)
    mask = corpus.self_mask
    migrated = l2_normalize(forward.adapter.apply(old.documents), copy=False)
    bridged_q = l2_normalize(backward.adapter.apply(new.queries), copy=False)

    runs = {
        "status_quo": top_k_search(old.queries, old.documents, k=depth, self_mask=mask),
        "bridged": top_k_search(bridged_q, old.documents, k=depth, self_mask=mask),
        "migrated": top_k_search(new.queries, migrated, k=depth, self_mask=mask),
        "reindexed": top_k_search(new.queries, new.documents, k=depth, self_mask=mask),
    }
    scored = b.score(corpus, runs, cutoffs=cutoffs)

    return {
        "corpus": corpus.name,
        "old_model": old_model,
        "new_model": new_model,
        "old_dim": int(old.documents.shape[1]),
        "new_dim": int(new.documents.shape[1]),
        "n_documents": n,
        "n_queries": len(corpus.query_ids),
        "self_removal": mask is not None,
        "cutoffs": list(cutoffs),
        "backward_adapter": backward.method,
        "forward_adapter": forward.method,
        "n_fit_pairs": int(fit_rows.size),
        "scores": scored.aggregate,
        "seed": seed,
        "duration_seconds": round(time.perf_counter() - started, 1),
    }


def already_done(out: Path) -> set[tuple[str, str, str]]:
    """Keys already in the output, so a re-run resumes instead of repeating."""
    if not out.exists():
        return set()
    seen: set[tuple[str, str, str]] = set()
    for line in out.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        seen.add((row["corpus"], row["old_model"], row["new_model"]))
    return seen


def main(argv: Sequence[str] | None = None) -> int:
    """Run the grid and append one row per cell."""
    b = band()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", action="append", default=None)
    parser.add_argument("--ladder", default="default", choices=sorted(b.LADDERS))
    parser.add_argument("--k", default="10,100")
    parser.add_argument("--method", default=DEFAULT_METHOD, help="Adapter family")
    parser.add_argument("--out", type=Path, default=Path("reports/migration/rows.jsonl"))
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / "band-cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    cutoffs = [int(part) for part in args.k.split(",") if part.strip()]
    datasets = b.resolve_corpora(args.corpus or ["beir"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(args.out)
    # One encoder per model for the whole grid: loading a sentence-transformer
    # takes longer than encoding a small corpus with it.
    encoder_cache: dict[str, Any] = {}

    for dataset in datasets:
        corpus = b.load_corpus(dataset)
        print(f"\n{dataset}\n  {len(corpus.doc_ids):,} documents", flush=True)
        for old_model, new_model in b.LADDERS[args.ladder]:
            if (corpus.name, old_model, new_model) in done:
                print(f"  {old_model} -> {new_model}  (already done)", flush=True)
                continue
            print(f"  {old_model} -> {new_model}", flush=True)
            row = measure(
                corpus,
                old_model,
                new_model,
                cutoffs=cutoffs,
                cache_dir=args.cache_dir,
                device=args.device,
                seed=args.seed,
                method=args.method,
                encoder_cache=encoder_cache,
            )
            with args.out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            s = row["scores"]
            k = cutoffs[0]
            print(
                f"    status_quo {s['status_quo'][f'ndcg@{k}']:.4f}  "
                f"bridged {s['bridged'][f'ndcg@{k}']:.4f}  "
                f"migrated {s['migrated'][f'ndcg@{k}']:.4f}  "
                f"reindexed {s['reindexed'][f'ndcg@{k}']:.4f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
