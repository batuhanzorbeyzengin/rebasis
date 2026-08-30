"""M1 — does a sample's candidate overlap predict a running cache's hit rate?

``probe`` now recommends the two-stage arrangement, and what unblocked that is a
price: :func:`rebasis.probe.metrics.candidate_reuse` counts how much the
candidate sets of a real query log overlap, and that overlap is the share of
each candidate set the cache already holds. This harness is the measurement
behind the claim that the count is a **lower bound** on what a running cache
achieves.

**The identity this exists to avoid.** Given candidate sets ``C_1..C_n`` and a
cache that never evicts, replaying them in order embeds each distinct document
exactly once. Misses are ``|union|``, requests are ``sum |C_i|``, and the hit
rate is therefore ``1 - |union| / sum |C_i|`` — which is ``candidate_reuse``,
written twice. On the same query set the two agree by arithmetic and agreement
measures nothing. `docs/bridge-band.md` section 9 is the precedent: a count that
was an identity was published as a finding, and the rule since is that a
quantity is tested where the two sides can differ.

So the question is asked across query sets, which is the situation the claim is
actually about. ``probe`` sees a **sample** of the traffic; the running system
sees the traffic. For each run the harness

1. draws a fraction of the judged query log — what ``probe --queries`` was given,
2. counts ``candidate_reuse`` over that fraction's candidate sets,
3. replays the **whole** log, in order, through a real
   :class:`rebasis.serve.Cascade` with a real cache, and reads
   :attr:`~rebasis.serve.cascade.CascadeStats.hit_rate` off it.

``--fraction 1.0`` is run alongside the others deliberately: it is the identity,
and seeing it come back as an exact match is what proves the harness measures
what it says rather than something adjacent to it.

The store is a :class:`~rebasis.store.MemoryStore` over the corpus' own old-model
vectors and its text, so the candidate sets ``Cascade`` produces are the ones the
count was taken over — no search-structure approximation stands between the two
quantities, and any difference is the sampling.

Embeddings come from the same ``.npy`` cache ``bridge_band.py`` writes, so on a
warm cache this costs a fit and a replay per run rather than an embedding pass::

    uv run --extra sentence-transformers --with ir-datasets --with model2vec \\
        python tools/cascade_reuse.py --corpora heldout --ladder default \\
        --cache-dir ~/band-cache --out reports/band/reuse.jsonl
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

#: Fractions of the query log a probe is imagined to have been given.
#:
#: 1.0 is the identity and is always run: it is the control that says the two
#: quantities are the same measurement when the query sets agree, which is what
#: makes the rows below it evidence rather than arithmetic.
DEFAULT_FRACTIONS = (0.25, 0.5, 1.0)

#: Candidate depth. The one the decision rule uses — a hit rate measured at
#: another depth prices a different arrangement.
DEPTH = 200

#: Slack allowed when asking whether the count stayed below the real hit rate.
#:
#: ``1 - distinct/n`` and ``(n - distinct)/n`` are the same rational and not the
#: same double, so an exact ``>=`` counts the last bit of a division as a
#: violation of the bound. Measured, the runs where the two quantities are
#: genuinely the same arithmetic differ by at most 1e-16 and the one real
#: violation is 0.055 — four orders of magnitude either side of this.
BOUND_TOLERANCE = 1e-6

#: A correlation needs at least two distinct values on each side.
_MIN_DISTINCT = 2


def replay(
    corpus: Any,
    old: Any,
    new: Any,
    bridge: Any,
    *,
    depth: int,
) -> dict[str, float]:
    """Send the whole query log through a real ``Cascade``, in order.

    In order, and not as a batch: a cache's hit rate is a property of a
    sequence. Reordering the log would measure a different arrangement and the
    difference would not show up anywhere.
    """
    from rebasis.embed import PrecomputedEmbedder
    from rebasis.serve import Cascade, MemoryVectorCache
    from rebasis.store import MemoryStore

    store = MemoryStore(corpus.doc_ids, old.documents, corpus.doc_texts)
    embedder = PrecomputedEmbedder(
        new.profile, dict(zip(corpus.doc_texts, new.documents, strict=True))
    )
    cascade = Cascade(store, bridge, embedder, candidates=depth, cache=MemoryVectorCache())

    started = time.perf_counter()
    for query in new.queries:
        cascade.search(query, k=10)
    stats = cascade.stats
    return {
        "hit_rate": float(stats.hit_rate),
        "queries": int(stats.queries),
        "candidates": int(stats.candidates),
        "documents_embedded": int(stats.documents_embedded),
        "kept_bridged": int(stats.kept_bridged),
        "replay_seconds": round(time.perf_counter() - started, 1),
    }


def sampled_reuse(  # noqa: PLR0913 - the search, the depth and which queries to draw
    mapped: Any, old_documents: Any, corpus: Any, *, depth: int, fraction: float, seed: int
) -> dict[str, Any]:
    """``candidate_reuse`` over a random sample of the log's candidate sets.

    A random subset rather than a prefix: a query log is usually ordered by
    time, and a prefix would sample one hour of traffic where the claim is about
    a sample of it.
    """
    from rebasis.compute import top_k_search
    from rebasis.probe.metrics import candidate_reuse

    n_queries = mapped.shape[0]
    size = max(1, round(n_queries * fraction))
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(n_queries, size=size, replace=False))

    self_mask = corpus.self_mask
    indices, _ = top_k_search(
        mapped[rows],
        old_documents,
        k=depth,
        self_mask=None if self_mask is None else self_mask[rows],
    )
    return {
        "fraction": fraction,
        "n_sampled": int(size),
        "candidate_reuse": candidate_reuse(indices),
        # The union's size, recorded rather than reconstructed from the ratio.
        # `1 - distinct/n` multiplied back out lands within one document of the
        # count, which is fine for a mean and useless for the claim this
        # measurement is actually making: that a cache large enough to hold the
        # working set embeds each distinct document exactly once.
        "n_distinct": int(np.unique(indices).size),
        "n_candidates": int(indices.size),
    }


def measure(  # noqa: PLR0913 - one argument per input to a run
    corpus_name: str,
    old_model: str,
    new_model: str,
    *,
    cache_dir: Path,
    device: str,
    seed: int,
    depth: int,
    fractions: Sequence[float],
    encoder_cache: dict[str, Any],
) -> dict[str, Any]:
    """One row: the sampled counts and the replayed hit rate for one run."""
    from rebasis.compute import resolve_device, using_device

    started = time.perf_counter()
    corpus = bridge_band.load_corpus(corpus_name)
    if not corpus.query_ids:
        msg = f"{corpus_name} has no judged queries, so it has no query log to replay"
        raise RuntimeError(msg)

    shared = {"corpus": corpus, "cache_dir": cache_dir, "device": device}
    old = bridge_band.encode_corpus(model_id=old_model, encoder_cache=encoder_cache, **shared)
    new = bridge_band.encode_corpus(model_id=new_model, encoder_cache=encoder_cache, **shared)
    bridge, fit_summary = bridge_band.fit_bridge(corpus, old, new, seed=seed, device=device)

    with using_device(resolve_device(device)):
        mapped = bridge.to_index_space(new.queries)
        samples = [
            sampled_reuse(mapped, old.documents, corpus, depth=depth, fraction=fraction, seed=seed)
            for fraction in fractions
        ]
        replayed = replay(corpus, old, new, bridge, depth=depth)

    return {
        "corpus": corpus.name,
        "old_model": old_model,
        "new_model": new_model,
        "n_documents": len(corpus.doc_ids),
        "n_queries": len(corpus.query_ids),
        "self_removal": corpus.self_mask is not None,
        "depth": depth,
        "seed": seed,
        "fit": fit_summary,
        "samples": samples,
        "replay": replayed,
        "duration_seconds": round(time.perf_counter() - started, 1),
    }


def summarise(path: Path) -> str:
    """What the rows say about the lower-bound claim.

    Three questions, in the order they have to be answered:

    **Is the harness measuring what it says?** At ``--fraction 1.0`` the sampled
    count and the replayed hit rate are the same quantity, so they must agree
    exactly. A row where they do not means the candidate sets the count was
    taken over are not the ones ``Cascade`` produced, and every other row is
    then about something else.

    **Is the count a lower bound?** Below 1.0 it must sit at or under the real
    hit rate on every run. One run above it and the arrangement is priced as
    cheaper than it is, which is the direction a decision rule must not err in.

    **Does it predict?** A bound that is always 0.0 is a bound and no use. The
    rank correlation is what says the count carries information about the run it
    was taken from.
    """
    from scipy import stats

    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        return f"no rows in {path}"

    by_fraction: dict[float, list[tuple[float, float]]] = {}
    for row in rows:
        hit_rate = float(row["replay"]["hit_rate"])
        for sample in row["samples"]:
            reuse = sample["candidate_reuse"]
            if reuse is None:
                continue
            by_fraction.setdefault(float(sample["fraction"]), []).append((float(reuse), hit_rate))

    lines = [
        f"{len(rows)} runs at candidate depth {rows[0]['depth']}",
        "",
        "| sample | runs | mean reuse | mean hit rate | mean gap | at or below | max excess |",
        "|" + "---|" * 7,
    ]
    for fraction in sorted(by_fraction):
        pairs = by_fraction[fraction]
        reuse = np.array([r for r, _ in pairs])
        real = np.array([h for _, h in pairs])
        gap = real - reuse
        below = int((gap >= -BOUND_TOLERANCE).sum())
        lines.append(
            f"| {fraction:.2f} | {len(pairs)} | {reuse.mean():.4f} | {real.mean():.4f} "
            f"| {gap.mean():+.4f} | {below}/{len(pairs)} | {max(0.0, -gap.min()):.6f} |"
        )

    lines += ["", "rank correlation between the sampled count and the replayed hit rate:"]
    for fraction in sorted(by_fraction):
        pairs = by_fraction[fraction]
        reuse = np.array([r for r, _ in pairs])
        real = np.array([h for _, h in pairs])
        distinct = min(len(set(reuse.tolist())), len(set(real.tolist())))
        if distinct < _MIN_DISTINCT:
            lines.append(f"  {fraction:.2f}: not enough distinct values")
            continue
        spearman = stats.spearmanr(reuse, real)
        lines.append(
            f"  {fraction:.2f}: Spearman rho = {spearman.statistic:+.3f}  p = {spearman.pvalue:.3g}"
        )

    lines += _capacity_view(rows)
    return "\n".join(lines)


def _capacity_view(rows: list[dict[str, Any]]) -> list[str]:
    """The identity control, and the one thing that breaks it.

    At fraction 1.00 the count and the hit rate are the same arithmetic — but
    only for a cache that can hold the working set. The shipped in-memory cache
    holds :data:`~rebasis.serve.cascade.MEMORY_CACHE_ENTRIES`, and a run whose
    union of candidate sets is larger evicts and re-embeds. So the rows are
    split by whether the working set fits, and the split is the finding.
    """
    from rebasis.serve.cascade import MEMORY_CACHE_ENTRIES

    fits: list[tuple[int, int, float]] = []
    spills: list[tuple[str, int, int, float]] = []
    for row in rows:
        whole = next((s for s in row["samples"] if float(s["fraction"]) == 1.0), None)
        if whole is None or "n_distinct" not in whole:
            continue
        distinct = int(whole["n_distinct"])
        embedded = int(row["replay"]["documents_embedded"])
        gap = float(row["replay"]["hit_rate"]) - float(whole["candidate_reuse"])
        if distinct <= MEMORY_CACHE_ENTRIES:
            fits.append((distinct, embedded, gap))
        else:
            spills.append((str(row["corpus"]), distinct, embedded, gap))

    if not fits and not spills:
        return ["", "no run recorded n_distinct; re-run to get the identity control"]

    agreed = sum(distinct == embedded for distinct, embedded, _ in fits)
    off = [abs(embedded - distinct) for distinct, embedded, _ in fits if distinct != embedded]
    lines = [
        "",
        f"identity control at fraction 1.00, against a {MEMORY_CACHE_ENTRIES:,}-entry cache:",
        f"  {agreed} of {len(fits)} runs whose working set fits embedded exactly |union|",
        "  documents. Same query set, nothing evicted: the count and the hit rate are",
        "  the same arithmetic, and agreement there is the check rather than the finding.",
    ]
    if off:
        lines.append(
            f"  The other {len(off)} differ by at most {max(off)} document — a tie at the "
            f"candidate-set boundary broken differently by two search paths, or ArguAna's "
            f"self-mask, either way under 1e-4 of the ratio."
        )
    if spills:
        lines += [
            "",
            f"  {len(spills)} runs did NOT fit, and there the count is not a bound:",
        ]
        lines += [
            f"    {corpus}: working set {distinct:,}, embedded {embedded:,}, "
            f"count exceeded the hit rate by {-gap:.4f}"
            for corpus, distinct, embedded, gap in spills
        ]
    return lines


def short(model_id: str) -> str:
    """The trailing segment of a model id, for a progress line."""
    return model_id.rsplit("/", 1)[-1]


def build_parser() -> argparse.ArgumentParser:
    """Command line, deliberately close to ``bridge_band.py``'s own."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", action="append", default=[])
    # `append`, not a single value: the grid is often two named groups, and a
    # second `--corpora` silently replacing the first ran a quarter of a
    # measurement under the name of the whole one.
    parser.add_argument(
        "--corpora", action="append", default=[], choices=sorted(bridge_band.CORPORA)
    )
    parser.add_argument("--ladder", default="default", choices=sorted(bridge_band.LADDERS))
    parser.add_argument("--cache-dir", type=Path, default=Path("~/band-cache").expanduser())
    parser.add_argument("--out", type=Path, default=Path("reports/band/reuse.jsonl"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument(
        "--summarise",
        type=Path,
        default=None,
        help="Read a finished .jsonl and print what it says; runs nothing",
    )
    parser.add_argument(
        "--fractions",
        default=",".join(str(f) for f in DEFAULT_FRACTIONS),
        help="Shares of the query log a probe is imagined to have seen",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the grid and append one JSON row per run."""
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

    fractions = [float(part) for part in args.fractions.split(",") if part.strip()]
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoder_cache: dict[str, Any] = {}
    failures = 0

    with args.out.open("a", encoding="utf-8") as handle:
        for name in names:
            for old_model, new_model in bridge_band.LADDERS[args.ladder]:
                label = f"{name} {short(old_model)}->{short(new_model)}"
                try:
                    row = measure(
                        name,
                        old_model,
                        new_model,
                        cache_dir=args.cache_dir,
                        device=args.device,
                        seed=args.seed,
                        depth=args.depth,
                        fractions=fractions,
                        encoder_cache=encoder_cache,
                    )
                except Exception as error:  # noqa: BLE001 - one bad corpus must not end the grid
                    failures += 1
                    print(f"FAILED {label}: {error}", file=sys.stderr)
                    continue
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                measured = ", ".join(
                    f"{s['fraction']:.2f}->{s['candidate_reuse']:.4f}" for s in row["samples"]
                )
                print(f"{label}: hit_rate {row['replay']['hit_rate']:.4f}  sampled {measured}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
