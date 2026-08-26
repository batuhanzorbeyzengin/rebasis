"""Does truncating the new model's vector ever beat fitting an adapter?

``ROADMAP.md`` lists this as an open question under *Beyond 0.3*: "for models
trained with nested representations, the right answer may be 'truncate and
renormalise', with no adapter at all." It is open. Nothing in this repository
has ever measured it, and this is the harness that can.

## Truncate-and-renormalise is not a new adapter — it is `IdentityAdapter`

This was checked before anything was built, because if it were a new adapter the
answer would be a seventh candidate and it is not.

``IdentityAdapter.apply`` is ``pad_or_truncate(x, self.output_dim)``: when the
new model is wider than the old index it already truncates. What it does not do
is renormalise, and a truncated unit vector is not a unit vector — so the whole
question is whether renormalisation happens somewhere else. It does, on **every**
path that consumes an adapter's output:

* ``probe/runner.py`` — the held-out scoring path (``evaluate_candidate``), the
  CSLS sample, and the calibrator fit. All three wrap ``apply`` in
  ``l2_normalize(..., copy=False)``.
* ``serve/bridge.py`` — ``Bridge.to_index_space`` renormalises by default, and
  it is the single door ``serve/cascade.py``, ``serve/mixed.py`` and the
  LangChain retriever all go through.
* ``migrate/engine.py`` and ``migrate/refit.py`` — the vectors written back into
  the index are normalised before they are written.

The one exception is ``ScaledAdapter.fit``, which reads ``base.apply(src)``
unnormalised to fit the DSM diagonal on the residual. That is fit-internal
arithmetic; no vector from it is ever served or scored.

And even where renormalisation did not happen it would not matter to a ranking:
``top_k_search`` is a plain inner product, and a positive scalar on the query
cannot reorder ``q·d``. The single case where query scale *does* interact is
CSLS, whose per-document bias is additive — and there the query has already been
renormalised before the search.

So "truncate and renormalise" is ``identity`` under another name, and this spike
measures it under the name it already has.

## Why it has never been measured anyway

Three separate reasons, none of them a missing adapter:

* ``identity`` is not in ``CANDIDATE_METHODS``. ``auto`` never fits it; it is
  reachable only by naming it, which is what this spike does.
* ``run_probe``'s "do nothing" baseline is dimension-gated — ``baselines
  ["unadapted"]`` is computed only when the two dimensions agree. On exactly the
  rungs where truncation is the question, the probe reports no no-adapter number
  at all.
* M0's ``identity`` figure of 0.2741 was measured on three model pairs that are
  **all 384-to-384** (``spikes/m0_spike.py``: MiniLM→bge-small, MiniLM→e5-small,
  e5-small→bge-small). At equal dimensions ``pad_or_truncate`` is a no-op, so
  M0 measured no adaptation — it never truncated anything.

``tools/bridge_band.py`` does not fill the gap either, and says so: its
``naive_swap_padded`` configuration zero-pads *both* spaces out to the wider one
precisely because padding changes no inner product, and its ``_pad_to``
docstring rules truncation out as "a measurement of that choice". Which is the
measurement nobody has taken.

## Why the scope is wider than the roadmap entry

The roadmap scopes this to Matryoshka-trained models. Takeshita, Takeshita,
Ponzetto and Ruffinelli, *To MRL or not to MRL: Text Embeddings are Robust to
Truncation Without Matryoshka Learning, Except In Heavy Truncation Scenarios*
(arXiv:2605.16608), report that truncated embeddings of models trained *without*
MRL are competitive with, and often outperform, models trained with it, and that
MRL's advantage appears only under heavy truncation — which they put at a
reduction of at least 80%.

On ``tools/bridge_band.py``'s ladders the reduction a bridge would need is 33%
to 67%, all of it inside the range where that paper says MRL training is not
what decides the outcome. Every row this spike writes carries its own
``truncation`` ratio so that claim can be read against the numbers rather than
asserted over them.

## What is measured

``auto``'s own held-out comparison, unchanged, with ``identity`` added to the
candidate list: ``probe_store`` samples the index, fits every candidate on one
disjoint slice and scores them all on another. Every score here comes from the
shipped ``fit_candidates`` / ``select_best`` path — the only thing computed in
this file is the counterfactual, ``best_scoring``, which is the candidate that
would have won on ARR alone.

That counterfactual is the point of half the output. ``select_best`` treats
scores within 0.005 as equal and then prefers **fewer parameters**, and a
zero-parameter candidate wins every tie it reaches. So a run where
``truncate_in_tie_band`` is true and ``truncate_wins`` is false is a run where
adding ``identity`` to ``CANDIDATE_METHODS`` would have changed the answer, and
the count of those is what decides whether it should be added.

Corpora, model pairs, the corpus loader, the embedding cache and the fit all
come from ``tools/bridge_band.py`` by **import**, so these numbers sit beside the
existing band rather than floating free::

    ~/rebasis/.venv/bin/python spikes/truncate_candidate.py \\
        --corpus beir/scifact/test --ladder default \\
        --out reports/truncate/rows.jsonl

Numbers, not adjectives: whatever it prints is what goes in the docs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rebasis.core import CANDIDATE_METHODS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rebasis.types import FloatArray

#: The candidate list `auto` ships, plus the one this spike exists to measure.
#: `identity` last because `CANDIDATE_METHODS` is ordered cheapest-first for a
#: reason — a run interrupted early still has a usable result — and a candidate
#: that costs nothing to fit cannot make that ordering worse wherever it sits.
METHODS: tuple[str, ...] = (*CANDIDATE_METHODS, "identity")

#: Scores within this distance are treated as equal by `select_best`, which then
#: breaks the tie on parameter count. Mirrored here to compute the
#: counterfactual — what would have won on score alone — and nothing else. It is
#: a private literal in `rebasis.core.selection`; if it moves there, it has to
#: move here, and a row whose `tie_break_decided` disagrees with `selected` is
#: how that would show up.
TIE_TOLERANCE = 0.005

#: Held-out share of the fit budget. The shipped 4000/1000 split, and the same
#: fraction `tools/bridge_band.py` uses, so a row here is fitted on the same
#: amount of data as a row there.
HELDOUT_SHARE = 4


def band() -> Any:
    """``tools/bridge_band.py``, imported for its corpora, models and encodings.

    Imported rather than copied so that a corpus loaded here is the corpus the
    band was measured on and the embedding cache is the same cache. ``tools/`` is
    not a package; the repository root goes on the path so the implicit
    namespace package resolves.
    """
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tools import bridge_band

    return bridge_band


def base_name(name: str) -> str:
    """The adapter family behind a report name, with both suffixes removed.

    ``evaluate_candidate`` names a candidate ``identity+csls`` or
    ``procrustes_centered@query`` depending on what was measured, and a
    comparison against the bare family name would silently miss both.
    """
    return name.split("+", 1)[0].split("@", 1)[0]


def probe_one(
    corpus: Any,
    old: Any,
    new: Any,
    *,
    seed: int,
    device: str,
    fit_pairs: int,
) -> Any:
    """Run ``auto``'s own comparison over one corpus and one model pair.

    Built the way ``tools/bridge_band.py``'s ``fit_bridge`` builds it — a
    ``MemoryStore`` over the old model's document vectors and a
    ``PrecomputedEmbedder`` over the new model's — because what is being measured
    is the shipped selection path and a second arrangement here would be
    measuring something else.

    No ``.rbs`` round trip: this spike never serves a query, it reads the
    per-candidate scores off the probe result, and a candidate that is not
    selected is never serialised anyway.
    """
    from rebasis.embed import PrecomputedEmbedder
    from rebasis.probe.session import probe_store
    from rebasis.store import MemoryStore

    store = MemoryStore(corpus.doc_ids, old.documents, corpus.doc_texts)

    document_table = dict(zip(corpus.doc_texts, new.documents, strict=True))
    query_table: dict[str, FloatArray] = dict(zip(corpus.query_texts, new.queries, strict=True))
    # An asymmetric new model makes the probe encode the sampled documents a
    # second time the way a query is encoded, and a table holding only the query
    # strings would refuse it.
    if new.documents_as_queries is not None:
        query_table.update(zip(corpus.doc_texts, new.documents_as_queries, strict=True))

    embedder = PrecomputedEmbedder(new.profile, document_table, query_vectors=query_table)

    heldout = max(1, fit_pairs // HELDOUT_SHARE)
    result, _ = probe_store(
        store,
        embedder,
        size=fit_pairs + heldout,
        heldout=heldout,
        k=10,
        seed=seed,
        device=device,
        methods=METHODS,
    )
    return result


def summarise(result: Any) -> dict[str, Any]:
    """Turn one probe result into the row this spike is about.

    ``selected`` is ``select_best``'s own answer, read off the result rather than
    recomputed. ``best_scoring`` is the only thing derived here, and it exists so
    that the tie-break's effect is a number instead of an argument.
    """
    rows = [c.to_dict() for c in result.candidates]
    scored = [r for r in rows if r["arr_r10"] is not None]

    truncate_rows = [r for r in scored if base_name(r["adapter_type"]) == "identity"]
    truncate_arr = max((r["arr_r10"] for r in truncate_rows), default=None)

    best = max(scored, key=lambda r: r["arr_r10"])
    ranking = sorted(scored, key=lambda r: -r["arr_r10"])

    selected = result.best.to_dict()
    in_band = truncate_arr is not None and best["arr_r10"] - truncate_arr <= TIE_TOLERANCE
    return {
        "candidates": rows,
        # More than one identity row means the new model is asymmetric and the
        # probe fitted a second candidate list on query-encoded pairs. A
        # zero-parameter candidate learns nothing from either, so the two rows
        # are the same measurement reported twice — recorded because a candidate
        # list that produced it in `auto` would put both in a user's report.
        "n_truncate_candidates": len(truncate_rows),
        "truncate_arr": truncate_arr,
        "truncate_rank": (
            None
            if truncate_arr is None
            else 1 + next(i for i, r in enumerate(ranking) if r["arr_r10"] == truncate_arr)
        ),
        "selected": selected["adapter_type"],
        "selected_arr": selected["arr_r10"],
        "best_scoring": best["adapter_type"],
        "best_scoring_arr": best["arr_r10"],
        "gap_to_best": (None if truncate_arr is None else round(best["arr_r10"] - truncate_arr, 4)),
        "truncate_wins": base_name(selected["adapter_type"]) == "identity",
        # True when `select_best` would hand the run to truncation on the
        # parameter-count tie-break if `identity` were in `CANDIDATE_METHODS`.
        "truncate_in_tie_band": in_band,
        # True when the tie-break, not the score, chose the winner.
        "tie_break_decided": selected["adapter_type"] != best["adapter_type"],
        "old_model_only": result.baselines.get("old_model_only"),
        "unadapted": result.baselines.get("unadapted"),
        "geometry_bound": None if result.geometry is None else round(result.geometry.bound, 4),
        "n_fit_pairs": result.n_fit_pairs,
        "tier": result.ground_truth_tier,
    }


def run_cell(
    bb: Any,
    dataset: str,
    pair: tuple[str, str],
    *,
    seed: int,
    device: str,
    fit_pairs: int,
    limit_docs: int | None,
    cache_dir: Path,
    encoder_cache: dict[str, Any],
) -> dict[str, Any]:
    """One corpus, one rung: load, encode both sides, probe, summarise."""
    old_model, new_model = pair
    started = time.perf_counter()

    corpus = bb.load_corpus(dataset, limit=limit_docs)
    shared = {"corpus": corpus, "cache_dir": cache_dir, "device": device}
    old = bb.encode_corpus(model_id=old_model, encoder_cache=encoder_cache, **shared)
    new = bb.encode_corpus(model_id=new_model, encoder_cache=encoder_cache, **shared)

    old_dim = int(old.documents.shape[1])
    new_dim = int(new.documents.shape[1])

    result = probe_one(corpus, old, new, seed=seed, device=device, fit_pairs=fit_pairs)

    return {
        "corpus": dataset,
        "old_model": old_model,
        "new_model": new_model,
        "old_dim": old_dim,
        "new_dim": new_dim,
        # The reduction the adapter direction actually asks for: rebasis maps the
        # NEW model's query into the OLD index's space, so truncation is
        # new_dim -> old_dim. `None` in the other direction, where there is
        # nothing to truncate and `identity` zero-pads instead — which changes no
        # inner product and is a relabelling rather than a transform.
        "truncation": (None if new_dim <= old_dim else round(1 - old_dim / new_dim, 4)),
        "seed": seed,
        "duration_seconds": round(time.perf_counter() - started, 1),
        **summarise(result),
    }


def key_of(row: dict[str, Any]) -> tuple[str, str, str, int]:
    """What makes a row unique, so a resumed run repeats nothing."""
    return (row["corpus"], row["old_model"], row["new_model"], int(row["seed"]))


def already_done(out: Path) -> set[tuple[str, str, str, int]]:
    """Cells the output file already holds.

    The ladder is GPU-hours; an interrupted run that had to start again would
    mean nobody ever finishes one.
    """
    if not out.exists():
        return set()
    done: set[tuple[str, str, str, int]] = set()
    for line in out.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" not in row:
            done.add(key_of(row))
    return done


def rung_label(row: dict[str, Any]) -> str:
    """Which rung a row is, short enough for a table and still unambiguous.

    The dimensions alone are not enough once two ladders are read together:
    ``tools/bridge_band.py``'s ``default`` and ``wide`` both contain a 384-to-256
    rung, and they are different model pairs. The model names have to be in the
    label or half the table is unattributable.
    """
    short = [
        name.split("/")[-1].removesuffix("-en-v1.5")
        for name in (row["old_model"], row["new_model"])
    ]
    return f"{short[1]}({row['new_dim']}) -> {short[0]}({row['old_dim']})"


def report(rows: list[dict[str, Any]]) -> None:
    """The table a reader looks at first.

    Sorted by how much truncation lost, because the interesting rows are at both
    ends and a corpus-ordered table buries them.
    """
    usable = [r for r in rows if "error" not in r and r.get("truncate_arr") is not None]
    print()
    header = (
        f"{'corpus':22s} {'rung':46s} {'trunc':>6s} "
        f"{'truncate':>9s} {'best':>9s} {'gap':>8s}  winner"
    )
    print(header)
    print("-" * len(header))
    for row in sorted(usable, key=lambda r: -(r["gap_to_best"] or 0.0)):
        ratio = "-" if row["truncation"] is None else f"{row['truncation']:.0%}"
        flag = "  <-- tie band" if row["truncate_in_tie_band"] else ""
        print(
            f"{row['corpus'][:22]:22s} {rung_label(row)[:46]:46s} {ratio:>6s} "
            f"{row['truncate_arr']:9.4f} {row['best_scoring_arr']:9.4f} "
            f"{row['gap_to_best']:+8.4f}  {row['best_scoring']}{flag}"
        )

    if not usable:
        print("no usable rows")
        return

    wins = sum(1 for r in usable if r["truncate_wins"])
    tied = sum(1 for r in usable if r["truncate_in_tie_band"])
    truncating = [r for r in usable if r["truncation"] is not None]
    gaps = sorted(r["gap_to_best"] for r in usable)
    median = gaps[len(gaps) // 2]
    print()
    print(f"rows                        {len(usable)}")
    print(f"  of which truncate         {len(truncating)} (the rest zero-pad)")
    print(f"truncation won              {wins}")
    print(f"truncation inside the band  {tied}  (would win on the parameter tie-break)")
    print(f"gap to the best score       median {median:+.4f}  worst {gaps[-1]:+.4f}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run every requested corpus against every requested rung."""
    bb = band()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        action="append",
        default=None,
        help=f"Dataset or group name; repeatable. Groups: {', '.join(sorted(bb.CORPORA))}",
    )
    parser.add_argument(
        "--ladder",
        action="append",
        default=None,
        choices=sorted(bb.LADDERS),
        help="Repeatable. Every rung of every named ladder is run",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fit-pairs", type=int, default=bb.FIT_PAIRS)
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / "band-cache")
    parser.add_argument("--out", type=Path, default=Path("reports/truncate/rows.jsonl"))
    args = parser.parse_args(argv)

    datasets = bb.resolve_corpora(args.corpus or ["beir"])
    pairs: list[tuple[str, str]] = []
    for ladder in args.ladder or ["default"]:
        for pair in bb.LADDERS[ladder]:
            if pair not in pairs:
                pairs.append(pair)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(args.out)
    encoder_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    for dataset in datasets:
        for pair in pairs:
            if (dataset, pair[0], pair[1], args.seed) in done:
                print(f"skip {dataset} {pair[0]} -> {pair[1]} (already done)", flush=True)
                continue
            try:
                row = run_cell(
                    bb,
                    dataset,
                    pair,
                    seed=args.seed,
                    device=args.device,
                    fit_pairs=args.fit_pairs,
                    limit_docs=args.limit_docs,
                    cache_dir=args.cache_dir,
                    encoder_cache=encoder_cache,
                )
            except Exception as exc:
                row = {
                    "corpus": dataset,
                    "old_model": pair[0],
                    "new_model": pair[1],
                    "seed": args.seed,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
            # Appended as it goes rather than dumped at the end: this runs for
            # hours and a crash at the last rung must not cost the first.
            with args.out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

    report(rows)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
