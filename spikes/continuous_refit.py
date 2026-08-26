"""Does refitting the adapter part-way through a migration buy anything?

`migrate/refit.py` has been complete and unreachable since it was written: a
win-only adoption guard, a held-out comparison, unit tests, and no caller. The
roadmap's entry says what is missing is plumbing — an embedder on the engine and
a `--refit` flag. It is not only plumbing, and this measures the part that is
not.

**The premise the module was built on no longer holds.** Its docstring says
pairs become available "for free" during a migration, because records already
migrated carry new-model vectors. They do not: a migrated record carries
``A(old)``, the adapter's *image* of the old vector, so fitting on those pairs
fits ``A`` to reproduce ``A``. Any real pair needs the document re-embedded, and
once an embedder is running the question is no longer "which pairs are free" but
**"which pairs are worth paying for"**.

Two answers, and they point in opposite directions:

``migrated``
    Sample the records already migrated. This is what the roadmap describes, and
    what `refit.py`'s caveat about priority order is written against.

``remaining``
    Sample the records not yet migrated. Their old vectors are still in the
    store, so no shadow copy is needed — and, more to the point, **they are the
    records the refitted adapter will actually be applied to.** Under
    ``--priority access`` the migrated half is the frequently-read records and
    the remaining half is the long tail; a map tuned on the first and applied to
    the second is tuned on the wrong distribution.

The argument for ``remaining`` is only an argument. Five arms, measured::

    baseline                the original adapter, fitted on FIT_PAIRS
    refit:migrated          K pairs from the first half, alone
    refit:remaining         K pairs from the remainder, alone
    accumulated:migrated    FIT_PAIRS + K pairs from the first half
    accumulated:remaining   FIT_PAIRS + K pairs from the remainder

**The last two exist because the first comparison is a trap.** A refit fitted on
K pairs is being compared against an adapter fitted on FIT_PAIRS, so at K below
that budget the arms measure "is K worse than FIT_PAIRS" — which this project
already knows the answer to — and not "does refitting help". The engine can only
accumulate pairs it creates itself: an ``.rbs`` file carries a matrix, not the
pairs it was fitted on. The ``accumulated`` arms are the case where it *could*
have them, which is the most favourable reading the feature can be given, and
the one worth beating.

``K`` is swept, so what comes out is not a verdict but a crossover: **how many
documents a migration must re-embed before the map it refits is better than the
map it would replace.** That number is the cost, and `RefitPolicy` has to be set
against it.

Everything is scored the way `rebasis probe --fit-migration` scores a forward
map: rewrite the slice, send a **raw** new-model query at it, and count what a
full reindex of that slice would have returned. Real corpora, real queries.

Three migration orders, and the third is a different experiment wearing the
same harness:

``sequential``
    Corpus order. What ``--priority none`` does.

``access``
    Documents that appear in the corpus' own qrels first, then the rest. A
    proxy for ``--priority access`` that is measured rather than invented: a
    document some real query was judged relevant to is a document somebody
    reads.

``arrival``
    Only for a ``mix:`` corpus — several collections read as one index. Every
    document of the first member, then every document of the second, with the
    original adapter fitted on the **first member alone**. This is the corpus
    that *grew* during the migration, into a domain the adapter never saw.

**Why ``arrival`` exists, and why the other two are not enough.** Drift-Adapter
reports a continuous-adaptation result (arXiv:2509.23471 §5.6): refitting the
adapter hourly keeps ARR above 0.95 over 24 hours, where a fixed adapter
"trained only at T=0" degrades to around 0.83. Read carefully, that degradation
has a cause rebasis does not share. Their adapter maps *queries into the old
space*, and their index fills with items that are "now purely in the f_new
space" — so a query mapped backwards is wrong for a growing share of the index,
and refitting is how they chase it. rebasis serves exactly that index with
two-space search instead (`rebasis.serve.MixedSpaceSearch`), which is a
structural answer rather than a moving one.

What is left of their scenario, once that is taken out, is the corpus changing
*in kind*: an adapter fitted on the documents that existed then, applied to
documents about something else. That is what ``arrival`` measures, and it is
the one case where a refit has something to learn that more pairs of the same
corpus cannot teach it.

    PYTHONPATH=src python spikes/continuous_refit.py \\
        --corpus beir --ladder default --out reports/refit/rows.jsonl

    PYTHONPATH=src python spikes/continuous_refit.py \\
        --corpus mix:beir/cqadupstack/android+beir/cqadupstack/mathematica \\
        --order arrival --fit-scope first --out reports/refit/drift.jsonl
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

    from rebasis.core.base import BaseAdapter
    from rebasis.types import FloatArray

#: The band harness owns corpus loading, the ladders and the embedding cache.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

#: Pairs the original adapter is fitted on. `rebasis fit`'s default is 4000, and
#: it is swept because **the answer depends on it**: a refit is competing with
#: the adapter it would replace, so what it has to beat is that adapter's own
#: fit budget. 1000 is the under-budgeted case — a small corpus, a hurried fit,
#: or `--pairs` turned down — and it is the case the feature would exist for.
FIT_SWEEP = (4000, 1000)

#: Pairs a refit is fitted on, swept. `RefitPolicy.min_pairs` is the first of
#: them; the rest are there because the cost of a refit *is* this number, and a
#: single value would answer "does it win at 1000" rather than "what does
#: winning cost".
REFIT_SWEEP = (1000, 2000, 4000, 8000)

#: Documents that must stay out of every fit so there is something to score on.
#: The evaluation set is the remainder minus the largest budget any arm draws,
#: and below this it stops being a measurement of the collection.
EVALUATION_FLOOR = 1000

#: The family both fits use. One method rather than `auto`, so a difference
#: between arms is the pairs and not the family.
METHOD = "procrustes_centered"

#: Where the migration is interrupted to refit.
SPLIT = 0.5


def band() -> Any:
    """The band harness, imported late so `--help` costs nothing."""
    import bridge_band

    return bridge_band


def _fit(src: FloatArray, dst: FloatArray) -> BaseAdapter:
    """One adapter, through the same call the CLI makes."""
    from rebasis.core import fit_candidates

    candidates = fit_candidates(src, dst, normalize=False, methods=[METHOD])
    if not candidates:
        msg = f"{METHOD} could not be fitted on {src.shape[0]} pairs"
        raise RuntimeError(msg)
    return candidates[0].adapter


def _remap_self_mask(corpus: Any, evaluation: np.ndarray) -> np.ndarray | None:
    """The corpus' self-exclusion, expressed in the evaluated slice's positions.

    ArguAna's queries *are* documents, and letting one retrieve itself puts a
    guaranteed hit at rank 1 in both the ground truth and the run. The mask the
    harness carries is in whole-corpus positions; everything here is scored on a
    slice, so it has to be translated. A masked document outside the slice
    becomes ``-1``, which is what the harness already uses for "nothing to
    exclude for this query".
    """
    mask = corpus.self_mask
    if mask is None:
        return None
    position = {int(original): i for i, original in enumerate(evaluation)}
    return np.array([position.get(int(m), -1) for m in mask], dtype=np.int64)


def _retention(
    adapter: BaseAdapter,
    *,
    old_docs: FloatArray,
    new_docs: FloatArray,
    new_queries: FloatArray,
    self_mask: np.ndarray | None,
    k: int,
) -> float:
    """What a migration of this slice with this adapter would deliver.

    Rewrite the slice with the map, send raw new-model queries at it, and count
    what a full reindex of the same slice returns. A real reindex scores 1.0 by
    construction, so this reads directly as the fraction of one.
    """
    from rebasis.compute import top_k_search
    from rebasis.core import l2_normalize
    from rebasis.probe.metrics import recall_at_k

    truth, _ = top_k_search(new_queries, new_docs, k=k, self_mask=self_mask)
    relevant = [set(row.tolist()) for row in truth[:, :k]]

    migrated = l2_normalize(adapter.apply(old_docs), copy=False)
    got, _ = top_k_search(new_queries, migrated, k=k, self_mask=self_mask)
    return float(recall_at_k(got, relevant, k))


#: Prefix for a corpus that is several corpora read as one index. The same
#: construction `spikes/per_cluster.py` uses, and reused rather than restated so
#: that `mix:` means one thing across the spikes: the **corpus** is assembled,
#: the **drift** is not — it is whatever two real models do to real text.
MIX_PREFIX = "mix:"

#: Document proxies used as queries where a mixture has no query set of its own.
#: They are removed from the index before anything is scored, which is the
#: harness' `t0-knn` convention and makes self-exclusion structural rather than a
#: flag someone has to remember.
PROXY_QUERIES = 500


def mix_members(dataset: str) -> list[str]:
    """The corpora behind a name — one of them, unless it is a mixture."""
    if not dataset.startswith(MIX_PREFIX):
        return [dataset]
    return [part for part in dataset.removeprefix(MIX_PREFIX).split("+") if part]


def combine(b: Any, dataset: str, parts: list[Any]) -> Any:
    """Read several corpora as one index.

    Document ids are prefixed with the member's position so two collections
    cannot collide on a shared id. The query set is empty: a mixture is scored
    against the new model's own neighbours, with document proxies standing in
    for queries.
    """
    if len(parts) == 1:
        return parts[0]
    ids: list[str] = []
    texts: list[str] = []
    for position, part in enumerate(parts):
        ids.extend(f"{position}:{doc_id}" for doc_id in part.doc_ids)
        texts.extend(part.doc_texts)
    return b.Corpus(
        name=dataset, doc_ids=ids, doc_texts=texts, query_ids=[], query_texts=[], qrels={}
    )


def encode_parts(b: Any, parts: list[Any], model_id: str, **shared: Any) -> Any:
    """Encode every member and stack the results in member order.

    Each member goes through ``encode_corpus`` under its own name, so a mixture
    reads the same warm ``.npy`` files a plain run of that corpus reads and
    writes no cache entry of its own.
    """
    encoded = [b.encode_corpus(corpus=part, model_id=model_id, **shared) for part in parts]
    if len(encoded) == 1:
        return encoded[0]
    as_queries = (
        None
        if encoded[0].documents_as_queries is None
        else np.vstack([e.documents_as_queries for e in encoded])
    )
    return b.Encoded(
        profile=encoded[0].profile,
        documents=np.vstack([e.documents for e in encoded]),
        queries=np.empty((0, encoded[0].documents.shape[1]), dtype=np.float32),
        documents_as_queries=as_queries,
    )


def _arrival_order(parts: list[Any]) -> np.ndarray:
    """Every document of the first member, then every document of the next.

    The corpus as it grew: the adapter was fitted when only the first member
    existed, and the rest arrived while the migration was running.
    """
    offsets = np.cumsum([0, *[len(part.doc_ids) for part in parts]])
    return np.arange(int(offsets[-1]), dtype=np.int64)


def _access_order(corpus: Any, rng: np.random.Generator) -> np.ndarray:
    """Judged documents first, then the rest — a proxy for access order.

    Measured rather than invented: a document some real query was judged
    relevant to is a document somebody reads. Within each group the order is
    random, so the proxy says "read or not" and claims no ranking beyond that.
    """
    n = len(corpus.doc_ids)
    position = {doc_id: i for i, doc_id in enumerate(corpus.doc_ids)}
    judged: set[int] = set()
    for relevant in corpus.qrels.values():
        for doc_id in relevant:
            index = position.get(doc_id)
            if index is not None:
                judged.add(index)

    hot = np.array(sorted(judged), dtype=np.int64)
    cold = np.array([i for i in range(n) if i not in judged], dtype=np.int64)
    rng.shuffle(hot)
    rng.shuffle(cold)
    return np.concatenate([hot, cold])


def _queries_for(
    corpus: Any, new: Any, evaluation: np.ndarray, rng: np.random.Generator
) -> tuple[FloatArray, np.ndarray]:
    """The queries to score with, and the slice they are scored against.

    A collection with real queries uses them. A mixture has none — it is several
    collections read as one index — so document proxies stand in, encoded the way
    a query is encoded and **removed from the slice** before anything is scored.
    Removing them is what makes self-exclusion structural: a proxy cannot
    retrieve itself, cannot enter its own ground truth, and cannot be drawn as a
    fit pair, none of which depends on anyone remembering a flag.
    """
    if len(corpus.query_ids):
        return new.queries, evaluation
    proxies = rng.permutation(evaluation)[: min(PROXY_QUERIES, evaluation.size // 2)]
    as_queries = new.documents_as_queries
    source = new.documents if as_queries is None else as_queries
    return source[np.sort(proxies)], np.setdiff1d(evaluation, proxies)


def _sequence_for(
    corpus: Any, parts: list[Any], order: str, n: int, rng: np.random.Generator
) -> np.ndarray:
    """The order this migration processes records in."""
    if order == "arrival":
        return _arrival_order(parts)
    if order == "access":
        return _access_order(corpus, rng)
    # Corpus order, not a shuffle: `--priority none` migrates in insertion
    # order, and shuffling here would measure a third thing.
    return np.arange(n, dtype=np.int64)


def measure(
    corpus: Any,
    parts: list[Any],
    old_model: str,
    new_model: str,
    *,
    order: str,
    fit_pairs: int,
    fit_scope: str,
    k: int,
    cache_dir: Path,
    device: str,
    seed: int,
    encoder_cache: dict[str, Any],
) -> dict[str, Any]:
    """One corpus, one model pair, one order, one fit budget: every arm at every K."""
    b = band()
    started = time.perf_counter()
    shared = {"cache_dir": cache_dir, "device": device, "encoder_cache": encoder_cache}
    old = encode_parts(b, parts, old_model, **shared)
    new = encode_parts(b, parts, new_model, **shared)

    n = len(corpus.doc_ids)
    rng = np.random.default_rng(seed)
    sequence = _sequence_for(corpus, parts, order, n, rng)

    # `arrival` splits at the member boundary — that is where the corpus grew —
    # and everything else splits halfway.
    cut = len(parts[0].doc_ids) if order == "arrival" else int(n * SPLIT)
    first, second = sequence[:cut], sequence[cut:]
    # Clamped to what this collection can spare rather than skipping it. A small
    # corpus contributes a shorter curve, which is a real point about small
    # corpora; dropping it would leave the sweep describing large ones only.
    # `EVALUATION_FLOOR` keeps enough of the remainder unfitted to score on.
    headroom = min(len(first), len(second) - EVALUATION_FLOOR)
    sweep = [budget for budget in REFIT_SWEEP if budget <= headroom]
    if not sweep:
        msg = (
            f"{corpus.name} splits into {len(first)}/{len(second)}, which cannot "
            f"spare {min(REFIT_SWEEP)} pairs and still leave {EVALUATION_FLOOR} to score on"
        )
        raise RuntimeError(msg)
    budget = max(sweep)

    # `corpus`: the adapter was fitted on a sample of the whole index, which is
    # what `rebasis fit` does today. `first`: it was fitted before the rest of
    # the corpus existed, which is the only honest scope for `arrival`.
    pool = first if fit_scope == "first" else sequence
    fit_rows = np.sort(rng.permutation(pool)[: min(fit_pairs, pool.size)])
    baseline = _fit(old.documents[fit_rows], new.documents[fit_rows])

    # Nested draws: the K=1000 sample is the first 1000 of the K=8000 one, so a
    # larger budget is the smaller one plus more rather than a different sample.
    # Without that, two points on the sweep differ by which documents were drawn
    # as well as by how many, and the curve would carry both.
    pools = {"migrated": rng.permutation(first), "remaining": rng.permutation(second)}

    # Scored on the remainder minus every row any arm could be fitted on, so no
    # arm is measured on its own fit data. The exclusion is applied to all of
    # them rather than per arm: a different evaluation set per arm would make
    # the arms incomparable.
    evaluation = np.setdiff1d(second, pools["remaining"][:budget])
    queries, evaluation = _queries_for(corpus, new, evaluation, rng)
    if evaluation.size < EVALUATION_FLOOR or not queries.size:
        msg = f"{corpus.name} has {evaluation.size} documents and {queries.shape[0]} queries left"
        raise RuntimeError(msg)

    scoring = {
        "old_docs": old.documents[evaluation],
        "new_docs": new.documents[evaluation],
        "new_queries": queries,
        "self_mask": _remap_self_mask(corpus, evaluation),
        "k": k,
    }

    scores: dict[str, float] = {"baseline": _retention(baseline, **scoring)}
    for source, source_pool in pools.items():
        for budget_k in sweep:
            drawn = np.sort(source_pool[:budget_k])
            scores[f"refit:{source}:{budget_k}"] = _retention(
                _fit(old.documents[drawn], new.documents[drawn]), **scoring
            )
            # The most favourable reading the feature can be given: the engine
            # keeps what `fit` was fitted on and adds to it. It cannot today —
            # an `.rbs` carries a matrix, not its pairs — so this is the arm
            # that says whether making it possible would be worth doing.
            with_original = np.union1d(fit_rows, drawn)
            scores[f"accumulated:{source}:{budget_k}"] = _retention(
                _fit(old.documents[with_original], new.documents[with_original]), **scoring
            )

    return {
        "corpus": corpus.name,
        "old_model": old_model,
        "new_model": new_model,
        "order": order,
        "fit_scope": fit_scope,
        "n_documents": n,
        "n_members": len(parts),
        "n_queries": int(queries.shape[0]),
        "n_evaluated": int(evaluation.size),
        "n_judged": int(sum(len(v) for v in corpus.qrels.values())),
        "split_at": int(cut),
        "k": k,
        "method": METHOD,
        "fit_pairs": fit_pairs,
        "n_fit_pairs": int(fit_rows.size),
        "refit_sweep": sweep,
        "scores": {name: round(value, 4) for name, value in scores.items()},
        "gains": {
            name: round(value - scores["baseline"], 4)
            for name, value in scores.items()
            if name != "baseline"
        },
        "seed": seed,
        "duration_seconds": round(time.perf_counter() - started, 1),
    }


def already_done(out: Path) -> set[tuple[str, str, str, str, int, str]]:
    """Keys already in the output, so a re-run resumes instead of repeating."""
    if not out.exists():
        return set()
    seen: set[tuple[str, str, str, str, int, str]] = set()
    for line in out.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        seen.add(
            (
                row["corpus"],
                row["old_model"],
                row["new_model"],
                row["order"],
                row["fit_pairs"],
                row.get("fit_scope", "corpus"),
            )
        )
    return seen


def main(argv: Sequence[str] | None = None) -> int:
    """Run the grid and append one row per cell."""
    b = band()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", action="append", default=None)
    parser.add_argument("--ladder", default="default", choices=sorted(b.LADDERS))
    parser.add_argument(
        "--order", action="append", default=None, choices=["sequential", "access", "arrival"]
    )
    parser.add_argument(
        "--fit-scope",
        default="corpus",
        choices=["corpus", "first"],
        help="Draw the original adapter's pairs from the whole index, or only from "
        "the half that existed when it was fitted",
    )
    parser.add_argument(
        "--fit-pairs",
        action="append",
        type=int,
        default=None,
        help="Pairs the original adapter is fitted on; repeatable",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("reports/refit/rows.jsonl"))
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / "band-cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    orders = args.order or ["sequential", "access"]
    budgets = args.fit_pairs or list(FIT_SWEEP)
    datasets: list[str] = []
    for name in args.corpus or ["beir"]:
        datasets.extend([name] if name.startswith(MIX_PREFIX) else b.resolve_corpora([name]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(args.out)
    encoder_cache: dict[str, Any] = {}

    for dataset in datasets:
        parts = [b.load_corpus(member) for member in mix_members(dataset)]
        corpus = combine(b, dataset, parts)
        print(f"\n{dataset}\n  {len(corpus.doc_ids):,} documents", flush=True)
        for old_model, new_model in b.LADDERS[args.ladder]:
            for order in orders:
                if order == "arrival" and len(parts) < 2:
                    print(f"  [arrival] needs a {MIX_PREFIX} corpus; skipped", flush=True)
                    continue
                for fit_pairs in budgets:
                    key = (corpus.name, old_model, new_model, order, fit_pairs, args.fit_scope)
                    label = (
                        f"  {old_model} -> {new_model} "
                        f"[{order}, fit={fit_pairs} from {args.fit_scope}]"
                    )
                    if key in done:
                        print(f"{label}  (done)", flush=True)
                        continue
                    print(label, flush=True)
                    try:
                        row = measure(
                            corpus,
                            parts,
                            old_model,
                            new_model,
                            order=order,
                            fit_pairs=fit_pairs,
                            fit_scope=args.fit_scope,
                            k=args.k,
                            cache_dir=args.cache_dir,
                            device=args.device,
                            seed=args.seed,
                            encoder_cache=encoder_cache,
                        )
                    except RuntimeError as exc:
                        print(f"    skipped: {exc}", flush=True)
                        continue
                    with args.out.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row) + "\n")
                    gains = row["gains"]
                    best = max(gains, key=lambda name: gains[name])
                    print(
                        f"    baseline {row['scores']['baseline']:.4f}   "
                        f"best arm {best} {gains[best]:+.4f}   "
                        f"arms above baseline "
                        f"{sum(1 for v in gains.values() if v > 0)}/{len(gains)}",
                        flush=True,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
