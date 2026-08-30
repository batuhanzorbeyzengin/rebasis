"""What alignability looks like, on corpora somebody can check.

`rebasis expose` returns one number and offers no band. That refusal needs
defending, and it needs context: a user reading 0.62 off their own index has no
way to know whether that is unusual. This harness supplies the second and, in
doing so, the evidence for the first.

**Why there is no band.** `docs/_local`'s plan for this command proposed
reporting the *centroid-agreement* diagnostic and banding it low/medium/high,
having measured it ranking the outcome at Spearman +0.833. Two things changed
that. Centroid agreement is not computable without an oracle map fitted on
paired data — `spikes/unpaired_align.py::_reference_permutation` says so — and
the outcome it predicts is directly measurable by an index's owner, who holds
both the vectors and the text. So `expose` reports the outcome.

And an outcome cannot be banded the way a predictor can. A band on a predictor
is a measurement: you have the outcomes, you find the threshold that separates
them. A band on alignability would be a **policy**: somebody deciding that 0.4
is acceptable and 0.6 is not, with no labelled "this index was exfiltrated" to
calibrate against. That is a judgement about a reader's risk tolerance, and no
measurement this project can take produces it.

What this harness produces instead is a **range**: alignability across corpora,
reference models and seeds, so a reader can see where their own number sits
among numbers taken the same way. That is context, not a classifier, and the
difference is the whole point.

    uv run --extra sentence-transformers --with ir-datasets --with model2vec \\
        python tools/exposure_band.py --corpora heldout \\
        --cache-dir ~/band-cache --out reports/band/exposure.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bridge_band  # the harness this one reuses, found via the line above

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The model whose vectors stand in for the index's.
#:
#: Each of these is somebody's index in the world. The pair (indexed, reference)
#: is what alignability is a property of — not of the index alone — so the grid
#: crosses them rather than fixing one.
INDEXED = (
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
)

#: The public model an adversary would reach for. Local, small, and the most
#: downloaded sentence embedding model there is — which is precisely why it is
#: the realistic choice for somebody who has your vectors and not your model.
REFERENCES = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
)


def measure(  # noqa: PLR0913 - one argument per input to a run
    corpus_name: str,
    indexed_model: str,
    reference_model: str,
    *,
    cache_dir: Path,
    device: str,
    seed: int,
    sample: int,
    heldout: int,
    encoder_cache: dict[str, Any],
    seeds: int = 3,
    dim: int | None = None,
    precision: str = "float32",
) -> dict[str, Any]:
    """One (corpus, indexed model, reference model, seed) cell.

    Run through the **shipped** :func:`~rebasis.probe.exposure.measure_exposure`
    rather than a reimplementation, for the reason every harness here gives:
    what is being characterised is the number a user gets, and a second
    implementation would characterise itself.
    """
    from rebasis.embed import PrecomputedEmbedder
    from rebasis.probe.exposure import measure_exposure
    from rebasis.probe.truncation import quantize, truncate
    from rebasis.store import MemoryStore

    started = time.perf_counter()
    corpus = bridge_band.load_corpus(corpus_name)
    shared = {"corpus": corpus, "cache_dir": cache_dir, "device": device}
    indexed = bridge_band.encode_corpus(
        model_id=indexed_model, encoder_cache=encoder_cache, **shared
    )
    reference = bridge_band.encode_corpus(
        model_id=reference_model, encoder_cache=encoder_cache, **shared
    )

    # M3 of the exposure item, and the link to the truncation grid: a cheaper
    # index is a different index to align to, and whether it is *harder* to
    # align is the question. Applied to the indexed side only — the adversary's
    # own reference model is theirs to choose and is not the thing being made
    # cheaper.
    stored = indexed.documents
    if dim is not None:
        stored = truncate(stored, dim)
    if precision != "float32":
        stored = quantize(stored, precision)

    store = MemoryStore(corpus.doc_ids, stored, corpus.doc_texts)
    embedder = PrecomputedEmbedder(
        reference.profile, dict(zip(corpus.doc_texts, reference.documents, strict=True))
    )
    result = measure_exposure(
        store,
        embedder,
        size=min(sample, len(corpus.doc_ids)),
        heldout=heldout,
        seed=seed,
        seeds=seeds,
    )
    return {
        "corpus": corpus.name,
        "indexed_model": indexed_model,
        "reference_model": reference_model,
        "same_family": _family(indexed_model) == _family(reference_model),
        "same_model": indexed_model == reference_model,
        "stored_dim": int(stored.shape[1]),
        "stored_precision": precision,
        "full_dim": int(indexed.documents.shape[1]),
        **result.to_dict(),
        "duration_seconds": round(time.perf_counter() - started, 1),
    }


def _family(model_id: str) -> str:
    """The publisher, which is the coarsest thing "same family" can mean."""
    return model_id.split("/", 1)[0]


def summarise(path: Path) -> str:
    """The range, and the two things that move it."""
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        return f"no rows in {path}"

    values = np.array([float(row["alignability"]) for row in rows])
    pools = sorted({int(row["pool"]) for row in rows})
    lines = [
        f"{len(rows)} cells, each ranking a document against {pools} others",
        "",
        (
            f"alignability: min {values.min():.3f}, "
            f"median {np.median(values):.3f}, max {values.max():.3f}"
        ),
        "",
        "| indexed model | reference model | cells | min | median | max |",
        "|" + "---|" * 6,
    ]
    pairs = sorted({(row["indexed_model"], row["reference_model"]) for row in rows})
    for indexed, reference in pairs:
        subset = np.array(
            [
                float(row["alignability"])
                for row in rows
                if row["indexed_model"] == indexed and row["reference_model"] == reference
            ]
        )
        lines.append(
            f"| {_short(indexed)} | {_short(reference)} | {subset.size} "
            f"| {subset.min():.3f} | {np.median(subset):.3f} | {subset.max():.3f} |"
        )

    # The same-model cells are excluded from both groups, not folded into the
    # first. They are the positive control — there is nothing to align — and
    # counting them as "same family" put a median of 1.000 into a comparison
    # group and reported 0.955 for a driver whose real figure is a third of that.
    real = [row for row in rows if not row["same_model"]]
    held = sorted(
        {
            (int(row.get("stored_dim", 0)), str(row.get("stored_precision", "float32")))
            for row in real
        }
    )
    if len(held) > 1:
        lines += [
            "",
            "by how the index is *stored* — M3, and the link to the truncation grid:",
            "",
            "| stored as | cells | min | median | max |",
            "|" + "---|" * 5,
        ]
        for dim, precision in held:
            subset = np.array(
                [
                    float(row["alignability"])
                    for row in real
                    if int(row.get("stored_dim", 0)) == dim
                    and str(row.get("stored_precision", "float32")) == precision
                ]
            )
            lines.append(
                f"| {dim} / {precision} | {subset.size} | {subset.min():.3f} "
                f"| {np.median(subset):.3f} | {subset.max():.3f} |"
            )

    lines += [
        "",
        "by whether the two models come from the same publisher",
        "(the same-model cells are the control and are excluded from both):",
        "",
    ]
    for label, wanted in (("same family", True), ("different family", False)):
        subset = np.array(
            [float(row["alignability"]) for row in real if row["same_family"] is wanted]
        )
        if subset.size:
            lines.append(
                f"  {label:18s} {subset.size:3d} cells, median {np.median(subset):.3f}, "
                f"range {subset.min():.3f}-{subset.max():.3f}"
            )

    same = np.array([float(row["alignability"]) for row in rows if row["same_model"]])
    if same.size:
        lines += [
            "",
            f"  control: {same.size} cells where the two models are the *same* model,",
            f"  median {np.median(same):.3f}. Not a result — there is nothing to align,",
            "  and this is what the harness scores when the answer is free. It is here",
            "  to show the measurement can reach 1.000 at all.",
        ]
    lines += [
        "",
        "There is no band here and there will not be one. Alignability is the",
        "outcome rather than a predictor of one, and banding an outcome means",
        "choosing a policy threshold with no labelled harm to calibrate against.",
        "The range above is context for a reader's own number, which is a",
        "different and honest thing to offer.",
    ]
    return "\n".join(lines)


def _short(model_id: str) -> str:
    """The model name a table column has room for."""
    return model_id.rsplit("/", 1)[-1].removesuffix("-en-v1.5").removesuffix("-v2")


def build_parser() -> argparse.ArgumentParser:
    """Command line, deliberately close to the other harnesses' own."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", action="append", default=[])
    parser.add_argument(
        "--corpora", action="append", default=[], choices=sorted(bridge_band.CORPORA)
    )
    parser.add_argument("--indexed", action="append", default=[])
    parser.add_argument("--reference", action="append", default=[])
    parser.add_argument("--cache-dir", type=Path, default=Path("~/band-cache").expanduser())
    parser.add_argument("--out", type=Path, default=Path("reports/band/exposure.jsonl"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", default="0,1,2")
    parser.add_argument("--sample", type=int, default=20_000)
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help=(
            "Attempts per cell. The command's default is 3 and reports the best "
            "of them; 1 is a cheaper grid whose figures are therefore a floor "
            "on what the command would print"
        ),
    )
    parser.add_argument("--heldout", type=int, default=1_000)
    parser.add_argument(
        "--truncate",
        default=None,
        help="Comma-separated dimensions to hold the *indexed* vectors at",
    )
    parser.add_argument(
        "--quantize",
        default="float32",
        help="Comma-separated precisions to hold the indexed vectors at",
    )
    parser.add_argument("--summarise", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the grid and append one JSON row per cell."""
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

    indexed_models = args.indexed or list(INDEXED)
    references = args.reference or list(REFERENCES)
    seeds = [int(s) for s in str(args.seed).split(",") if s.strip()]
    dims: list[int | None] = (
        [int(part) for part in args.truncate.split(",") if part.strip()]
        if args.truncate
        else [None]
    )
    precisions = [part.strip() for part in args.quantize.split(",") if part.strip()]
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoder_cache: dict[str, Any] = {}
    failures = 0

    with args.out.open("a", encoding="utf-8") as handle:
        for name in names:
            for indexed_model in indexed_models:
                for reference_model in references:
                    for seed, dim, precision in product(seeds, dims, precisions):
                        try:
                            row = measure(
                                name,
                                indexed_model,
                                reference_model,
                                cache_dir=args.cache_dir,
                                device=args.device,
                                seed=seed,
                                sample=args.sample,
                                heldout=args.heldout,
                                encoder_cache=encoder_cache,
                                seeds=args.seeds,
                                dim=dim,
                                precision=precision,
                            )
                        except Exception as error:  # noqa: BLE001 - one bad cell must not end the grid
                            failures += 1
                            print(f"FAILED {name}: {error}", file=sys.stderr)
                            continue
                        handle.write(json.dumps(row) + "\n")
                        handle.flush()
                        held = (
                            "full"
                            if dim is None and precision == "float32"
                            else f"{row['stored_dim']}/{precision}"
                        )
                        print(
                            f"{name} {_short(indexed_model)}<-{_short(reference_model)} "
                            f"seed {seed} [{held}]: {row['alignability']:.3f}"
                        )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
