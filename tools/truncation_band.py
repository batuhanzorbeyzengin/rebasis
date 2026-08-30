"""M1 — what truncation and quantization cost, on real corpora.

`rebasis probe --truncate --quantize` reports a grid: the same model, held more
cheaply, and what fraction of today's nDCG@10 survives. This harness runs that
grid over the ladder's corpora and against real relevance judgements, so the
numbers in `docs/truncation-band.md` come from something a reader can re-run.

Two questions, and the second is the one the tool exists for.

**Does it reproduce the published averages?** A multi-task evaluation of 22
models on patents reports five representative models retaining 94-98% of their
nDCG@10 at 512 dimensions and above 88% at 256. Those are averages over one
domain. Reproducing their shape is what says this harness measures the same
thing everybody else measures.

**How much does it vary per corpus?** If the variance is low, the published
averages are enough and this flag is unnecessary. If it is high, "measure on
your own corpus" is proved again. `docs/golden-findings.md` section 7's warning
applies word for word: scifact is scientific abstracts, and a band measured
there is not a band for an Obsidian vault. **Both outcomes get published.**

The grid is run against **human judgements**, which is what makes a retention
here quality rather than agreement. The queries are encoded with the same model
the documents are, and both sides are cut together in every cell.

Embeddings come from the same ``.npy`` cache `bridge_band.py` writes, so on a
warm cache a whole grid costs a few seconds per corpus — the model runs once and
truncating what it produced is free.

    uv run --extra sentence-transformers --with ir-datasets --with ranx \\
        --with model2vec python tools/truncation_band.py \\
        --corpora heldout --model BAAI/bge-base-en-v1.5 \\
        --cache-dir ~/band-cache --out reports/band/truncation.jsonl
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

#: Dimensions every model is measured at, before being capped at its own width.
DIMS = (1024, 768, 512, 256, 128, 64)

#: Precisions, in the order the grid prints them.
PRECISIONS = ("float32", "float16", "int8", "binary")

#: Depth the rescored variant of each cell generates candidates at.
RESCORE_AT = 200


def measure(  # noqa: PLR0913 - one argument per input to a run
    corpus_name: str,
    model_id: str,
    *,
    cache_dir: Path,
    device: str,
    seed: int,
    dims: Sequence[int],
    precisions: Sequence[str],
    encoder_cache: dict[str, Any],
) -> dict[str, Any]:
    """One corpus, one model: every cell of the grid against human judgements."""
    from rebasis.probe.groundtruth import build_tier1
    from rebasis.probe.truncation import measure_grid

    started = time.perf_counter()
    corpus = bridge_band.load_corpus(corpus_name)
    if not corpus.query_ids:
        message = f"{corpus_name} has no judged queries, so retention here has no reference"
        raise RuntimeError(message)

    encoded = bridge_band.encode_corpus(
        model_id=model_id,
        corpus=corpus,
        cache_dir=cache_dir,
        device=device,
        encoder_cache=encoder_cache,
    )

    position = {name: i for i, name in enumerate(corpus.doc_ids)}
    qrels = [
        {position[name] for name in corpus.qrels.get(query_id, {}) if name in position}
        for query_id in corpus.query_ids
    ]
    keep = [i for i, judged in enumerate(qrels) if judged]
    if not keep:
        message = f"{corpus_name}: no judged document is in the corpus"
        raise RuntimeError(message)

    queries = encoded.queries[keep]
    truth = build_tier1(encoded.documents, queries, [qrels[i] for i in keep], k=10)
    grid = measure_grid(
        doc_vectors=encoded.documents,
        query_vectors=queries,
        ground_truth=truth,
        dims=dims,
        precisions=precisions,
        k=10,
        rescore_at=RESCORE_AT,
    )
    return {
        "corpus": corpus.name,
        "model": model_id,
        "matryoshka": encoded.profile.matryoshka_dim is not None,
        "n_documents": len(corpus.doc_ids),
        "n_queries": len(keep),
        "seed": seed,
        **grid.to_dict(),
        "duration_seconds": round(time.perf_counter() - started, 1),
    }


def summarise(path: Path) -> str:
    """Per-cell means across corpora, and the spread that decides the thesis."""
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        return f"no rows in {path}"

    out: list[str] = []
    for model in sorted({row["model"] for row in rows}):
        subset = [row for row in rows if row["model"] == model]
        mrl = "Matryoshka-trained" if subset[0]["matryoshka"] else "not Matryoshka-trained"
        out += [
            f"## {model} ({mrl}) — {len(subset)} corpora, full width {subset[0]['full_dim']}",
            "",
        ]
        out.extend(_table(subset, "retained"))
        out += ["", "the same cells with a full-precision rescore of the top 200:", ""]
        out.extend(_table(subset, "retained_rescored"))
        out += ["", "spread across corpora — max minus min, per cell:", ""]
        out.extend(_table(subset, "retained", spread=True))
        out.append("")
    return "\n".join(out)


def _table(rows: list[dict[str, Any]], field: str, *, spread: bool = False) -> list[str]:
    """One row per dimension, one column per precision."""
    precisions = list(dict.fromkeys(cell["precision"] for cell in rows[0]["cells"]))
    values: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        for cell in row["cells"]:
            values.setdefault((cell["dim"], cell["precision"]), []).append(float(cell[field]))

    dims = sorted({dim for dim, _ in values}, reverse=True)
    lines = [
        "| dim | " + " | ".join(precisions) + " |",
        "|" + "---|" * (len(precisions) + 1),
    ]
    for dim in dims:
        cells = []
        for precision in precisions:
            series = values.get((dim, precision))
            if not series:
                cells.append("—")
                continue
            array = np.array(series, dtype=float)
            cells.append(f"{array.max() - array.min():.3f}" if spread else f"{array.mean():.3f}")
        lines.append(f"| {dim} | " + " | ".join(cells) + " |")
    return lines


def build_parser() -> argparse.ArgumentParser:
    """Command line, deliberately close to the other harnesses' own."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", action="append", default=[])
    parser.add_argument(
        "--corpora", action="append", default=[], choices=sorted(bridge_band.CORPORA)
    )
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--cache-dir", type=Path, default=Path("~/band-cache").expanduser())
    parser.add_argument("--out", type=Path, default=Path("reports/band/truncation.jsonl"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dims", default=",".join(str(d) for d in DIMS))
    parser.add_argument("--precisions", default=",".join(PRECISIONS))
    parser.add_argument(
        "--summarise",
        type=Path,
        default=None,
        help="Read a finished .jsonl and print what it says; runs nothing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the grid and append one JSON row per corpus and model."""
    args = build_parser().parse_args(argv)
    if args.summarise is not None:
        print(summarise(args.summarise))
        return 0

    names = list(args.corpus)
    for group in args.corpora:
        names.extend(bridge_band.CORPORA[group])
    if not names or not args.model:
        print("nothing to run: pass --corpus/--corpora and --model", file=sys.stderr)
        return 2

    dims = [int(part) for part in args.dims.split(",") if part.strip()]
    precisions = [part.strip() for part in args.precisions.split(",") if part.strip()]
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoder_cache: dict[str, Any] = {}
    failures = 0

    with args.out.open("a", encoding="utf-8") as handle:
        for model_id in args.model:
            for name in names:
                try:
                    row = measure(
                        name,
                        model_id,
                        cache_dir=args.cache_dir,
                        device=args.device,
                        seed=args.seed,
                        dims=dims,
                        precisions=precisions,
                        encoder_cache=encoder_cache,
                    )
                except Exception as error:  # noqa: BLE001 - one bad corpus must not end the grid
                    failures += 1
                    print(f"FAILED {name} {model_id}: {error}", file=sys.stderr)
                    continue
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                # The widest measured cell at or below half the model's width.
                # `full_dim // 2` is often not one of `--dims` — 768 halves to
                # 384 and the default list has 512 and 256 — and printing an em
                # dash for every corpus made the progress line useless.
                narrow = [
                    c
                    for c in row["cells"]
                    if c["precision"] == "float32" and c["dim"] <= row["full_dim"] // 2
                ]
                if narrow:
                    best = max(narrow, key=lambda c: c["dim"])
                    print(
                        f"{name} {model_id}: {best['dim']} of {row['full_dim']} "
                        f"retains {best['retained']:.3f}"
                    )
                else:
                    print(f"{name} {model_id}: measured {len(row['cells'])} cells")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
