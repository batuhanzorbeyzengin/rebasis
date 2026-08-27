"""Is an adapter chain worth what a direct fit costs?

The roadmap has carried this since the first release: *v1 -> v2 -> v3 without a
full refit at each step. Error accumulation across a chain has not been measured,
and refitting against the original is probably more accurate — which is worth
knowing rather than assuming.* This is the measurement.

**When a chain is even a choice.** Fitting ``v3 -> v1`` directly needs pairs, and
the pairs are always there: the old vectors are in the index and the new ones
come from one embedding pass over a sample. So a direct fit is never
*unavailable* — chaining buys the embedding pass back, and only if a ``v3 -> v2``
adapter already exists that somebody else paid for. That is the trade being
priced here, and it is worth stating because it means a chain has to be nearly
free of error to be worth taking.

**Both directions, because they are different questions.** A query chain is what
`rebasis.Bridge` would serve: ``B1(B2(q3))``, mapping a v3 query back through v2
into the index's v1 space. A document chain is what `migrate` would write:
``A3(A2(A1(d1)))``, carrying the indexed vectors forward. Composition order and
what the error lands on differ, so both are run.

Every arm is scored the same way `docs/migration-band.md` scores one: against
what a full reindex to the newest model returns, over the corpus' own real
queries. Direct and chained share that ground truth exactly, so the difference
between them is the chain and nothing else.

    PYTHONPATH=src python spikes/chained_adapters.py \\
        --corpus heldout --corpus beir --out reports/chain/rows.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rebasis.core.base import BaseAdapter
    from rebasis.types import FloatArray

#: The band harness owns corpus loading, the ladders and the embedding cache.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

#: The models, oldest first. Every span of two or more is a chain to price.
LADDER = (
    "minishlab/potion-base-8M",
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
)

FIT_PAIRS = 4000

#: One family in every link, so a difference between a chain and a direct fit is
#: the composition rather than the family. The measured default and `auto`'s
#: usual winner.
#:
#: ``--method`` exists to settle one question the grid raised rather than to
#: offer a knob. `procrustes_centered` subtracts a mean before it rotates, so a
#: two-link chain of it is *two* centrings — a strictly richer function than the
#: single centred rotation a direct fit produces, and a candidate explanation
#: for the one span where the chain came out ahead. Plain `procrustes` has no
#: centring step, so a chain of it is one rotation exactly like the direct fit,
#: which makes the pair of runs a test of that explanation.
METHOD = "procrustes_centered"


def band() -> Any:
    """The band harness, imported late so `--help` costs nothing."""
    import bridge_band

    return bridge_band


def _fit(src: FloatArray, dst: FloatArray, method: str) -> BaseAdapter:
    """One adapter, through the same call the CLI makes."""
    from rebasis.core import fit_candidates

    candidates = fit_candidates(src, dst, normalize=False, methods=[method])
    if not candidates:
        msg = f"{method} could not be fitted on {src.shape[0]} pairs"
        raise RuntimeError(msg)
    return candidates[0].adapter


def _apply_chain(adapters: Sequence[BaseAdapter], vectors: FloatArray) -> FloatArray:
    """Put vectors through every link in order, normalising between them.

    Normalising between links rather than only at the end is what the serving
    path does — every consumer of an adapter's output normalises after
    ``apply()`` — so a chain measured without it would be measuring something
    the tool does not do.
    """
    from rebasis.core import l2_normalize

    current = vectors
    for adapter in adapters:
        current = l2_normalize(adapter.apply(current), copy=False)
    return current


def _recall(
    queries: FloatArray,
    documents: FloatArray,
    truth: list[set[int]],
    self_mask: np.ndarray | None,
    k: int,
) -> float:
    """Recall of one configuration against the reindex it is standing in for."""
    from rebasis.compute import top_k_search
    from rebasis.probe.metrics import recall_at_k

    got, _ = top_k_search(queries, documents, k=k, self_mask=self_mask)
    return float(recall_at_k(got, truth, k))


def measure_span(
    corpus: Any,
    encodings: dict[str, Any],
    span: Sequence[str],
    *,
    k: int,
    seed: int,
    method: str,
) -> dict[str, Any]:
    """One span of the ladder: the direct fit against the chain over it.

    ``span`` is oldest-first, so ``span[0]`` is what the index holds and
    ``span[-1]`` is the model being upgraded to.
    """
    from rebasis.compute import top_k_search

    oldest, newest = span[0], span[-1]
    old, new = encodings[oldest], encodings[newest]
    n = len(corpus.doc_ids)
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.permutation(n)[: min(FIT_PAIRS, n)])
    mask = corpus.self_mask

    # What a full reindex to the newest model returns. Both arms are scored
    # against exactly this, so the difference between them is the chain.
    truth_indices, _ = top_k_search(new.queries, new.documents, k=k, self_mask=mask)
    truth = [set(row.tolist()) for row in truth_indices[:, :k]]

    # ── the query direction: what `Bridge` would serve ────────────────────
    # A link maps *from* the later model *into* the earlier one, and the chain
    # is applied newest-first, so a v3 query walks back to v1 one rung at a
    # time.
    query_links = [
        _fit(encodings[later].documents[rows], encodings[earlier].documents[rows], method)
        for earlier, later in itertools.pairwise(span)
    ]
    query_direct = _fit(new.documents[rows], old.documents[rows], method)

    # ── the document direction: what `migrate` would write ────────────────
    document_links = [
        _fit(encodings[earlier].documents[rows], encodings[later].documents[rows], method)
        for earlier, later in itertools.pairwise(span)
    ]
    document_direct = _fit(old.documents[rows], new.documents[rows], method)

    scores = {
        "query_direct": _recall(
            _apply_chain([query_direct], new.queries), old.documents, truth, mask, k
        ),
        "query_chained": _recall(
            _apply_chain(list(reversed(query_links)), new.queries), old.documents, truth, mask, k
        ),
        "document_direct": _recall(
            new.queries, _apply_chain([document_direct], old.documents), truth, mask, k
        ),
        "document_chained": _recall(
            new.queries, _apply_chain(document_links, old.documents), truth, mask, k
        ),
        # The ceiling both are standing in for, and 1.0 by construction. Kept in
        # the row so a reader can see that rather than take it on trust.
        "reindexed": _recall(new.queries, new.documents, truth, mask, k),
    }

    return {
        "corpus": corpus.name,
        "span": list(span),
        "links": len(span) - 1,
        "oldest": oldest,
        "newest": newest,
        "n_documents": n,
        "n_queries": len(corpus.query_ids),
        "n_fit_pairs": int(rows.size),
        "k": k,
        "method": method,
        "scores": {name: round(value, 4) for name, value in scores.items()},
        "query_cost": round(scores["query_chained"] - scores["query_direct"], 4),
        "document_cost": round(scores["document_chained"] - scores["document_direct"], 4),
        "seed": seed,
    }


def spans() -> list[tuple[str, ...]]:
    """Every contiguous run of two or more rungs, shortest first."""
    found: list[tuple[str, ...]] = []
    for length in range(2, len(LADDER) + 1):
        for start in range(len(LADDER) - length + 1):
            found.append(tuple(LADDER[start : start + length]))
    return found


def already_done(out: Path) -> set[tuple[str, str, str]]:
    """Keys already in the output, so a re-run resumes instead of repeating."""
    if not out.exists():
        return set()
    return {
        (row["corpus"], "->".join(row["span"]), row.get("method", METHOD))
        for row in (
            json.loads(line)
            for line in out.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the grid and append one row per corpus and span."""
    b = band()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", action="append", default=None)
    parser.add_argument("--method", default=METHOD, help="Adapter family for every link")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("reports/chain/rows.jsonl"))
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / "band-cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    datasets = b.resolve_corpora(args.corpus or ["beir"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(args.out)
    encoder_cache: dict[str, Any] = {}

    for dataset in datasets:
        corpus = b.load_corpus(dataset)
        print(f"\n{dataset}\n  {len(corpus.doc_ids):,} documents", flush=True)
        wanted = [
            span for span in spans() if (corpus.name, "->".join(span), args.method) not in done
        ]
        if not wanted:
            print("  (done)", flush=True)
            continue
        encodings = {
            model_id: b.encode_corpus(
                model_id=model_id,
                corpus=corpus,
                cache_dir=args.cache_dir,
                device=args.device,
                encoder_cache=encoder_cache,
            )
            for model_id in LADDER
        }
        for span in wanted:
            started = time.perf_counter()
            row = measure_span(
                corpus, encodings, span, k=args.k, seed=args.seed, method=args.method
            )
            row["duration_seconds"] = round(time.perf_counter() - started, 1)
            with args.out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            s = row["scores"]
            print(
                f"  {row['links']} link(s) {span[0].split('/')[-1]} -> "
                f"{span[-1].split('/')[-1]}: "
                f"query {s['query_direct']:.4f} -> {s['query_chained']:.4f} "
                f"({row['query_cost']:+.4f})   "
                f"document {s['document_direct']:.4f} -> {s['document_chained']:.4f} "
                f"({row['document_cost']:+.4f})",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
