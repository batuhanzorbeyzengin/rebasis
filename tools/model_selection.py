"""M1 — does `rebasis compare` order candidates the way a full reindex does?

`probe` was measured as a **threshold** and found weak: the count that said
otherwise was an identity, and what survived was a rank correlation
(`docs/bridge-band.md` section 9). `rebasis compare` makes the ranking claim
explicitly, so it has to be scored as a ranking — and against the baseline
anybody actually uses.

For each corpus the harness fixes one incumbent (the model in the index) and
ranks N candidates two ways:

**The estimate.** `compare_models` over a **sample** of the corpus, with the
corpus' own judged queries. Each candidate's ``upgrade_gain`` is oracle recall
over the incumbent's recall on the held-out split — the number a user sees
without embedding anything twice.

**The truth.** Each candidate's nDCG@10 over the **whole** corpus, its own
vectors against its own index, scored by ranx against the human judgements. That
is what a full reindex to that candidate would deliver.

Different sample, different metric, different cut-off. It is not the identity
`bridge_advantage` turned out to be, and `--summarise` checks that rather than
asserting it.

**The null is the one people use: the published leaderboard order.** MTEB is an
average over 56 tasks and it is nobody's corpus, so the same order is predicted
for every corpus here. If `compare` cannot beat that, reading the table is
enough and this command is unnecessary. Scores are taken from the model cards
themselves and are recorded in :data:`PUBLISHED_MTEB` with their sources.

    uv run --extra sentence-transformers --with ir-datasets --with ranx \\
        --with model2vec python tools/model_selection.py \\
        --corpora heldout --cache-dir ~/band-cache \\
        --out reports/band/selection.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bridge_band  # the harness this one reuses, found via the line above

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The model in the index, and the models being weighed against it.
#:
#: One incumbent and three candidates spanning the ladder, so the ordering
#: question has something to get wrong: the three are 5, 11 and 12 MTEB points
#: apart from each other, which is wide enough that a published table has an
#: opinion and narrow enough that a corpus can disagree with it.
INCUMBENT = "minishlab/potion-base-8M"
CANDIDATES = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
)

#: MTEB averages as the models' own cards report them, with the source.
#:
#: Read from the cards rather than from a leaderboard snapshot, because a
#: leaderboard moves and a card is versioned with the model. Two cards cover
#: four models: minishlab's compares itself against all-MiniLM-L6-v2, and BAAI's
#: table carries both BGE rows.
#:
#: These are averages over 56 tasks. That is exactly the objection this
#: measurement exists to test — an average over 56 tasks is nobody's corpus —
#: so they are the null and not the answer.
PUBLISHED_MTEB: dict[str, float] = {
    # huggingface.co/minishlab/potion-base-8M — "Avg (MTEB)"
    "minishlab/potion-base-8M": 51.08,
    # the same card, quoting the model it compares itself against
    "sentence-transformers/all-MiniLM-L6-v2": 55.93,
    # huggingface.co/BAAI/bge-base-en-v1.5 — "Average (56)"
    "BAAI/bge-small-en-v1.5": 62.17,
    "BAAI/bge-base-en-v1.5": 63.55,
}

#: Documents drawn for the estimate. Deliberately a small share of every corpus
#: here — the claim is that a sample answers the question, so a sample that was
#: most of the corpus would not be testing it.
SAMPLE = 4_000
HELDOUT = 1_000


def _query_log(corpus: Any) -> Any:
    """The corpus' judged queries, in the shape ``probe`` takes."""
    from rebasis.probe.session import QueryLog

    return QueryLog(
        queries=list(corpus.query_texts),
        qrels=[set(corpus.qrels.get(query_id, {})) for query_id in corpus.query_ids],
        metadata={"source": corpus.name},
    )


def _embedder(corpus: Any, encoded: Any) -> Any:
    """A `PrecomputedEmbedder` over one model's cached vectors.

    Both tables, because an asymmetric model encodes a query differently from a
    document and the probe re-encodes the sample as queries. The same
    construction `bridge_band.fit_bridge` uses, for the same reason: what is
    being measured is rebasis' pipeline, not a second one written here.
    """
    from rebasis.embed import PrecomputedEmbedder

    documents = dict(zip(corpus.doc_texts, encoded.documents, strict=True))
    queries: dict[str, Any] = dict(zip(corpus.query_texts, encoded.queries, strict=True))
    if encoded.documents_as_queries is not None:
        queries.update(zip(corpus.doc_texts, encoded.documents_as_queries, strict=True))
    return PrecomputedEmbedder(encoded.profile, documents, query_vectors=queries)


def truth(corpus: Any, encoded: Any, *, device: str) -> float:
    """What a full reindex to this candidate would deliver, on the whole corpus.

    nDCG@10 through ranx against the human judgements — a different metric at a
    different cut-off from the estimate, computed on a different population.
    That separation is what makes the correlation below a result rather than
    arithmetic.
    """
    from rebasis.compute import resolve_device, top_k_search, using_device

    with using_device(resolve_device(device)):
        indices, scores = top_k_search(
            encoded.queries, encoded.documents, k=10, self_mask=corpus.self_mask
        )
    scored = bridge_band.score(corpus, {"full_reindex": (indices, scores)}, cutoffs=(10,))
    return float(scored.aggregate["full_reindex"]["ndcg@10"])


def measure(  # noqa: PLR0913 - one argument per input to a run
    corpus_name: str,
    *,
    cache_dir: Path,
    device: str,
    seed: int,
    sample: int,
    heldout: int,
    encoder_cache: dict[str, Any],
) -> dict[str, Any]:
    """One corpus: every candidate's estimate, and every candidate's truth."""
    from rebasis.probe.comparison import compare_models
    from rebasis.store import MemoryStore

    started = time.perf_counter()
    corpus = bridge_band.load_corpus(corpus_name)
    if not corpus.query_ids:
        msg = f"{corpus_name} has no judged queries, so no upgrade can be measured"
        raise RuntimeError(msg)

    shared = {"corpus": corpus, "cache_dir": cache_dir, "device": device}
    incumbent = bridge_band.encode_corpus(model_id=INCUMBENT, encoder_cache=encoder_cache, **shared)
    encoded = {
        model: bridge_band.encode_corpus(model_id=model, encoder_cache=encoder_cache, **shared)
        for model in CANDIDATES
    }

    # The index, as the user has it: the incumbent's vectors and the text.
    store = MemoryStore(corpus.doc_ids, incumbent.documents, corpus.doc_texts)
    result = compare_models(
        store,
        {model: _embedder(corpus, encoded[model]) for model in CANDIDATES},
        old_embedder=_embedder(corpus, incumbent),
        query_log=_query_log(corpus),
        size=min(sample, len(corpus.doc_ids)),
        heldout=heldout,
        k=10,
        seed=seed,
        old_model=INCUMBENT,
        device=device,
    )

    estimates = {c.model: c.upgrade_gain for c in result.candidates}
    measured = {model: truth(corpus, encoded[model], device=device) for model in CANDIDATES}
    return {
        "corpus": corpus.name,
        "incumbent": INCUMBENT,
        "n_documents": len(corpus.doc_ids),
        "n_queries": len(corpus.query_ids),
        "sample": result.sample,
        "seed": seed,
        "candidates": [
            {
                "model": model,
                "upgrade_gain": estimates.get(model),
                "true_ndcg_at_10": measured[model],
                "published_mteb": PUBLISHED_MTEB[model],
            }
            for model in CANDIDATES
        ],
        "incumbent_ndcg_at_10": truth(corpus, incumbent, device=device),
        "duration_seconds": round(time.perf_counter() - started, 1),
    }


def _order(values: dict[str, float | None]) -> list[str]:
    """Models best first, dropping any the run could not score."""
    scored = {model: value for model, value in values.items() if value is not None}
    return sorted(scored, key=lambda model: scored[model], reverse=True)


def summarise(path: Path) -> str:
    """Rank correlation, top-1 accuracy, and the published order it must beat."""
    from scipy import stats

    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        return f"no rows in {path}"

    lines = [
        f"{len(rows)} corpora, {len(CANDIDATES)} candidates each, incumbent {INCUMBENT}",
        "",
        "| corpus | rho | tau | probe's top-1 | published top-1 | true best |",
        "|" + "---|" * 6,
    ]
    taus: list[float] = []
    rhos: list[float] = []
    probe_hits = 0
    published_hits = 0
    identity_gaps: list[float] = []

    for row in rows:
        estimate = {c["model"]: c["upgrade_gain"] for c in row["candidates"]}
        true = {c["model"]: c["true_ndcg_at_10"] for c in row["candidates"]}
        published = {c["model"]: c["published_mteb"] for c in row["candidates"]}
        if any(value is None for value in estimate.values()):
            lines.append(f"| {_corpus_label(row['corpus'])} | — | — | — | — | — |")
            continue

        estimate_values = np.array([estimate[m] for m in CANDIDATES], dtype=float)
        true_values = np.array([true[m] for m in CANDIDATES], dtype=float)
        rho = stats.spearmanr(estimate_values, true_values).statistic
        tau = stats.kendalltau(estimate_values, true_values).statistic
        if not np.isnan(rho):
            rhos.append(float(rho))
        if not np.isnan(tau):
            taus.append(float(tau))

        best_true = _order(true)[0]
        probe_best = _order(estimate)[0]
        published_best = _order(published)[0]
        probe_hits += probe_best == best_true
        published_hits += published_best == best_true

        # The identity check. `upgrade_gain` is the candidate's recall over the
        # incumbent's on a sample; the outcome is its nDCG@10 over the whole
        # corpus divided by the incumbent's. If those were the same quantity the
        # ratio below would be 1 on every candidate of every corpus.
        incumbent_true = float(row["incumbent_ndcg_at_10"]) or 1.0
        identity_gaps.extend(abs(estimate[m] - (true[m] / incumbent_true)) for m in CANDIDATES)

        lines.append(
            f"| {_corpus_label(row['corpus'])} "
            f"| {rho:+.3f} | {tau:+.3f} | {_short(probe_best)} | {_short(published_best)} "
            f"| {_short(best_true)} |"
        )

    total = len(rows)
    lines += [
        "",
        f"mean Spearman rho = {np.mean(rhos):+.3f} over {len(rhos)} corpora",
        f"mean Kendall tau  = {np.mean(taus):+.3f}",
        "",
        "top-1 accuracy — is the candidate each rule puts first genuinely first?",
        f"  rebasis compare      {probe_hits}/{total} = {probe_hits / total:.4f}",
        f"  published MTEB order {published_hits}/{total} = {published_hits / total:.4f}",
        "",
        "identity check — is the estimate the outcome written twice?",
        f"  max |upgrade_gain - (true nDCG ratio)| = {max(identity_gaps):.4f}",
        f"  mean {np.mean(identity_gaps):.4f} over {len(identity_gaps)} candidate/corpus pairs.",
        "  The estimate is recall on a sample against the incumbent; the outcome is",
        "  nDCG@10 over the whole corpus. Different metric, different population.",
    ]
    return "\n".join(lines)


def _corpus_label(name: str) -> str:
    """Short, stable name for a corpus, whichever loader it came from."""
    label = name.removeprefix("mmteb:mteb/").removeprefix("beir/")
    return label.removeprefix("cqadupstack/").removesuffix("/test")


def _short(model: str) -> str:
    """The model name a table column has room for."""
    return model.rsplit("/", 1)[-1].removesuffix("-en-v1.5").removesuffix("-v2")


def build_parser() -> argparse.ArgumentParser:
    """Command line, deliberately close to the other harnesses' own."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", action="append", default=[])
    parser.add_argument(
        "--corpora", action="append", default=[], choices=sorted(bridge_band.CORPORA)
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("~/band-cache").expanduser())
    parser.add_argument("--out", type=Path, default=Path("reports/band/selection.jsonl"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample", type=int, default=SAMPLE)
    parser.add_argument("--heldout", type=int, default=HELDOUT)
    parser.add_argument(
        "--summarise",
        type=Path,
        default=None,
        help="Read a finished .jsonl and print what it says; runs nothing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the grid and append one JSON row per corpus."""
    args = build_parser().parse_args(argv)
    if args.summarise is not None:
        print(summarise(args.summarise))
        return 0

    names = list(args.corpus)
    for group in args.corpora:
        names.extend(bridge_band.CORPORA[group])
    if not names:
        print("nothing to run: pass --corpus or --corpora", file=sys.stderr)
        return 2

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoder_cache: dict[str, Any] = {}
    failures = 0

    with args.out.open("a", encoding="utf-8") as handle:
        for name in names:
            try:
                row = measure(
                    name,
                    cache_dir=args.cache_dir,
                    device=args.device,
                    seed=args.seed,
                    sample=args.sample,
                    heldout=args.heldout,
                    encoder_cache=encoder_cache,
                )
            except Exception as error:  # noqa: BLE001 - one bad corpus must not end the grid
                failures += 1
                print(f"FAILED {name}: {error}", file=sys.stderr)
                continue
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            ordering = ", ".join(
                f"{_short(c['model'])}={c['upgrade_gain']:.3f}"
                if c["upgrade_gain"] is not None
                else f"{_short(c['model'])}=—"
                for c in row["candidates"]
            )
            print(f"{name}: {ordering}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
