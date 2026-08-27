"""Where an access log belongs in `probe`, and whether the interval survives it.

The roadmap: *the sampler supports weights; nothing passes them. Weighting the
sample by what people actually read would make ARR describe the queries that
matter, but it also makes the sample non-uniform in a way the confidence
interval does not currently model.*

Reading the code, that entry names one place to put the weights and there are
**two**, because `probe`'s sample does two jobs at once. It is the mini-index
every measurement runs against, and it is the pool the query proxies are split
out of. Weighting it changes both:

``weighted_sample``
    What passing weights to :func:`~rebasis.sample.draw_sample` does today. The
    mini-index fills with frequently-read documents, so the *distractors* change
    — and retrieval quality is a property of what else is in the index, not only
    of the query.

``weighted_queries``
    Sample uniformly, then draw the query proxies ∝ weight. The mini-index stays
    a fair miniature of the real one; only what is asked of it changes.

The second is what "describe the queries that matter" actually asks for, and the
first is what the roadmap's sentence would have produced. Which is right is a
measurement, not an argument, so both are run against the quantity a user cares
about: **ARR over the whole corpus, asked with the queries people actually
send.**

And then the part the entry was blocked on: does the bootstrap interval still
cover, at its nominal rate, once the draw is not uniform?

**Two effects have to be kept apart or the answer is meaningless.** A `probe`
sample is not only a sample — it is the *mini-index* every measurement runs
against, and a 3,500-document mini-index is an easier place to retrieve in than
the 23,000-document corpus it came from. So an estimate from any design sits
above the whole-corpus quantity, weighted or not, and reading that gap as the
cost of weighting would be reading a pre-existing property of sampling.

So each design is scored against **its own expectation at its own sample
size** — the mean over many replicates of that design — which is what its
interval is entitled to cover. The whole-corpus quantity is reported beside it,
labelled as the mini-index effect, so the confound is visible rather than
hidden.

The diagnostic that decides it is one ratio, and it needs no estimand: the
**median half-width of the interval** divided by the **spread of the estimates
across replicates**. The bootstrap resamples the queries of one run and never
resamples the run, so this asks whether the width it produces matches the
variability the procedure actually has.

**The target is 1.96, not 1.** A correctly calibrated 95% interval around a
roughly normal estimator is exactly ±1.96 standard deviations wide, so a ratio
near 1.96 means the interval is right, below it means optimistic, and above it
means conservative. Reading the ratio against 1 would call a correct interval
"twice too wide", which is how a calibration check turns into a false alarm.

Whether the interval is already off *before* any weighting is applied is what
the plain design is in the grid to answer.

**Where the weights come from.** A document some real query was judged relevant
to is a document somebody reads. That is a proxy and it is measured rather than
invented; what cannot be measured is *how much* more often, so the hot-to-cold
ratio is swept.

    PYTHONPATH=src python spikes/access_weighted.py \\
        --corpus beir --out reports/access/rows.jsonl
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

#: How much more often a judged document is read than an unjudged one. Swept
#: because no access log is available to measure it from, and a single value
#: would answer "does it matter at 10x" rather than "does it matter".
RATIOS = (10.0, 100.0)

#: One replicate is one `probe` run — a sample, a fit, and two searches.
#:
#: The headline number needs no target at all: the median interval half-width
#: against the spread of the estimates themselves says whether the interval
#: describes the procedure's variability, and neither side of that ratio is an
#: estimand anyone has to define. Coverage is reported beside it and is the
#: weaker of the two, because a design's expectation has to be estimated from
#: the same replicates — at this count its standard error is a tenth of the
#: spread, which makes coverage slightly optimistic and is stated rather than
#: corrected for.
REPLICATES = 120

SAMPLE = 4000
QUERIES = 500
K = 10
METHOD = "procrustes_centered"
N_BOOT = 1000


def band() -> Any:
    """The band harness, imported late so `--help` costs nothing."""
    import bridge_band

    return bridge_band


def access_weights(corpus: Any, ratio: float) -> FloatArray:
    """A weight per document: ``ratio`` for judged ones, 1 for the rest."""
    position = {doc_id: i for i, doc_id in enumerate(corpus.doc_ids)}
    weights = np.ones(len(corpus.doc_ids), dtype=np.float64)
    for relevant in corpus.qrels.values():
        for doc_id in relevant:
            index = position.get(doc_id)
            if index is not None:
                weights[index] = ratio
    return weights


def _fit(src: FloatArray, dst: FloatArray) -> Any:
    """One adapter, through the same call the CLI makes."""
    from rebasis.core import fit_candidates

    candidates = fit_candidates(src, dst, normalize=False, methods=[METHOD])
    if not candidates:
        msg = f"{METHOD} could not be fitted on {src.shape[0]} pairs"
        raise RuntimeError(msg)
    return candidates[0].adapter


def _per_query_recall(
    adapter: Any,
    *,
    old_index: FloatArray,
    new_index: FloatArray,
    new_queries: FloatArray,
    self_positions: np.ndarray,
    k: int,
) -> FloatArray:
    """Recall per query of a bridged query against what a reindex would return.

    ``self_positions`` excludes each query's own document from both the truth
    and the run: a proxy that retrieves itself is a guaranteed hit and would be
    counted twice over.
    """
    from rebasis.compute import top_k_search
    from rebasis.core import l2_normalize
    from rebasis.probe.metrics import recall_per_query

    truth, _ = top_k_search(new_queries, new_index, k=k, self_mask=self_positions)
    relevant = [set(row.tolist()) for row in truth[:, :k]]
    bridged = l2_normalize(adapter.apply(new_queries), copy=False)
    got, _ = top_k_search(bridged, old_index, k=k, self_mask=self_positions)
    return recall_per_query(got, relevant, k)


def target_arr(
    old: Any, new: Any, weights: FloatArray, *, adapter: Any, rng: np.random.Generator, k: int
) -> float:
    """ARR over the **whole** corpus, asked with access-weighted queries.

    This is the quantity a user with an access log wants an estimate of: the
    real index, and the questions people actually send it. Every design below is
    scored on how close it gets to this and on whether its interval covers it.
    """
    n = old.documents.shape[0]
    probability = weights / weights.sum()
    # Many more queries than any single replicate draws, so the target is a
    # property of the corpus rather than of one query set.
    picks = rng.choice(n, size=min(4 * QUERIES, n), replace=False, p=probability)
    return float(
        _per_query_recall(
            adapter,
            old_index=old.documents,
            new_index=new.documents,
            new_queries=new.documents[picks],
            self_positions=picks,
            k=k,
        ).mean()
    )


def _draw(design: str, n: int, weights: FloatArray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """One replicate's mini-index and its query proxies, for one design."""
    rng = np.random.default_rng(seed)
    uniform = np.ones(n, dtype=np.float64) / n
    weighted = weights / weights.sum()

    sample_p = weighted if design == "weighted_sample" else uniform
    sample = rng.choice(n, size=min(SAMPLE, n), replace=False, p=sample_p)

    if design == "weighted_queries":
        inside = weights[sample]
        query_p = inside / inside.sum()
    else:
        query_p = None
    chosen = rng.choice(sample.size, size=QUERIES, replace=False, p=query_p)
    queries = sample[chosen]
    index = np.setdiff1d(sample, queries)
    return index, queries


def replicate(
    design: str,
    old: Any,
    new: Any,
    weights: FloatArray,
    *,
    seed: int,
    k: int,
) -> tuple[float, tuple[float, float]]:
    """One `probe`-shaped run: its ARR and its bootstrap interval."""
    from rebasis.probe.metrics import bootstrap_ci

    n = old.documents.shape[0]
    index, queries = _draw(design, n, weights, seed)
    # Fitted on the mini-index's own rows, which is what `probe` does: the fit
    # set is the sample minus the held-out query proxies.
    adapter = _fit(new.documents[index], old.documents[index])
    everything = np.concatenate([index, queries])
    positions = {int(original): i for i, original in enumerate(everything)}
    per_query = _per_query_recall(
        adapter,
        old_index=old.documents[everything],
        new_index=new.documents[everything],
        new_queries=new.documents[queries],
        self_positions=np.array([positions[int(q)] for q in queries], dtype=np.int64),
        k=k,
    )
    return float(per_query.mean()), bootstrap_ci(per_query, n_boot=N_BOOT, seed=seed)


def measure(
    corpus: Any,
    old_model: str,
    new_model: str,
    *,
    ratio: float,
    k: int,
    cache_dir: Path,
    device: str,
    encoder_cache: dict[str, Any],
) -> dict[str, Any]:
    """One corpus, one model pair, one access ratio: three designs."""
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

    n = old.documents.shape[0]
    if n < SAMPLE + QUERIES:
        msg = f"{corpus.name} has {n} documents; a replicate needs {SAMPLE + QUERIES}"
        raise RuntimeError(msg)

    weights = access_weights(corpus, ratio)
    flat = np.ones(n, dtype=np.float64)
    rng = np.random.default_rng(0)
    # The whole-corpus quantities. Reported, not used as the coverage target:
    # every design estimates retention on a *mini-index*, which is an easier
    # place to retrieve in, so the gap between the two is the sampling effect
    # rather than anything weighting did. The adapter is fitted on the whole
    # corpus because these are the quantities being described, not estimates of
    # them.
    reference = _fit(new.documents, old.documents)
    whole_corpus = {
        "weighted": round(target_arr(old, new, weights, adapter=reference, rng=rng, k=k), 4),
        "uniform": round(target_arr(old, new, flat, adapter=reference, rng=rng, k=k), 4),
    }

    designs: dict[str, Any] = {}
    for design in ("uniform", "weighted_queries", "weighted_sample"):
        estimates, bounds = [], []
        for seed in range(REPLICATES):
            value, interval = replicate(design, old, new, weights, seed=seed, k=k)
            estimates.append(value)
            bounds.append(interval)
        array = np.array(estimates, dtype=np.float64)
        # A design's own expectation at its own sample size, which is what its
        # interval is entitled to cover.
        expectation = float(array.mean())
        covered = sum(1 for low, high in bounds if low <= expectation <= high)
        half_widths = np.array([(high - low) / 2 for low, high in bounds], dtype=np.float64)
        spread = float(array.std(ddof=1))
        designs[design] = {
            "mean": round(expectation, 4),
            "sd_across_replicates": round(spread, 4),
            "median_ci_half_width": round(float(np.median(half_widths)), 4),
            # The diagnostic in one number, and it is read against **1.96**:
            # a correctly calibrated 95% interval around a roughly normal
            # estimator is exactly that many standard deviations wide. Below it
            # the interval is optimistic, above it conservative. Read against 1
            # instead, a correct interval looks twice too wide.
            "half_width_over_sd": round(float(np.median(half_widths)) / spread, 3)
            if spread > 0
            else None,
            "coverage_of_own_expectation": round(covered / REPLICATES, 4),
            "mini_index_effect": round(
                expectation - whole_corpus["uniform" if design == "uniform" else "weighted"], 4
            ),
        }

    return {
        "corpus": corpus.name,
        "old_model": old_model,
        "new_model": new_model,
        "ratio": ratio,
        "n_documents": n,
        "n_hot": int((weights > 1).sum()),
        "sample": SAMPLE,
        "queries": QUERIES,
        "replicates": REPLICATES,
        "k": k,
        "whole_corpus": whole_corpus,
        "designs": designs,
        "duration_seconds": round(time.perf_counter() - started, 1),
    }


def already_done(out: Path) -> set[tuple[str, str, str, float]]:
    """Keys already in the output, so a re-run resumes instead of repeating."""
    if not out.exists():
        return set()
    return {
        (row["corpus"], row["old_model"], row["new_model"], row["ratio"])
        for row in (
            json.loads(line)
            for line in out.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the grid and append one row per cell."""
    b = band()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", action="append", default=None)
    parser.add_argument("--ladder", default="default", choices=sorted(b.LADDERS))
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--out", type=Path, default=Path("reports/access/rows.jsonl"))
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / "band-cache")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    datasets = b.resolve_corpora(args.corpus or ["beir"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(args.out)
    encoder_cache: dict[str, Any] = {}

    for dataset in datasets:
        corpus = b.load_corpus(dataset)
        print(f"\n{dataset}\n  {len(corpus.doc_ids):,} documents", flush=True)
        for old_model, new_model in b.LADDERS[args.ladder]:
            for ratio in RATIOS:
                if (corpus.name, old_model, new_model, ratio) in done:
                    print(f"  {old_model} -> {new_model} [{ratio:g}x]  (done)", flush=True)
                    continue
                print(f"  {old_model} -> {new_model} [{ratio:g}x]", flush=True)
                try:
                    row = measure(
                        corpus,
                        old_model,
                        new_model,
                        ratio=ratio,
                        k=args.k,
                        cache_dir=args.cache_dir,
                        device=args.device,
                        encoder_cache=encoder_cache,
                    )
                except RuntimeError as exc:
                    print(f"    skipped: {exc}", flush=True)
                    continue
                with args.out.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                for name, stats in row["designs"].items():
                    print(
                        f"    {name:18s} mean {stats['mean']:.4f}  "
                        f"sd {stats['sd_across_replicates']:.4f}  "
                        f"ci/2 {stats['median_ci_half_width']:.4f}  "
                        f"ratio {stats['half_width_over_sd']}  "
                        f"coverage {stats['coverage_of_own_expectation']:.2f}  "
                        f"mini-index {stats['mini_index_effect']:+.4f}",
                        flush=True,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
