"""Build the golden fixtures.

Golden tests answer a question no synthetic test can: does the pipeline reach
the right *decision* on real embeddings of a real corpus? Random vectors with a
planted rotation prove the mathematics; they cannot prove that prefix handling
works, or that hard drift is recognised as hard.

**The fixture holds vectors, not models.** Each pair is embedded once, here, and
the vectors are stored. The tests then run the whole probe pipeline through
``PrecomputedEmbedder`` with no model download and no network — which is what
lets them run in CI, and what makes them fast enough to be run at all.

The fixture is not committed: it is ~10 MB per model. This script
regenerates it, and writes a manifest with a SHA-256 per file so a corrupted or
substituted fixture fails loudly rather than shifting a band.

    uv run --extra sentence-transformers --with ir-datasets \\
        python tools/make_golden.py --out tests/golden/data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

#: The four scenarios. Each names a *kind* of model change, because
#: what is being tested is whether the tool recognises the kind — not whether
#: two particular checkpoints happen to align.
SCENARIOS: dict[str, dict[str, str]] = {
    "same_family": {
        "old": "sentence-transformers/all-MiniLM-L6-v2",
        "new": "sentence-transformers/all-MiniLM-L12-v2",
        "note": "same family, consecutive version, same dimension",
    },
    "different_family": {
        "old": "sentence-transformers/all-MiniLM-L6-v2",
        "new": "BAAI/bge-small-en-v1.5",
        "note": "different family, same dimension, asymmetric query prefix",
    },
    "prefix_trap": {
        "old": "sentence-transformers/all-MiniLM-L6-v2",
        "new": "intfloat/e5-small-v2",
        "note": "symmetric to asymmetric: collapses if prefix handling breaks",
    },
    "hard_drift": {
        "old": "sentence-transformers/all-MiniLM-L6-v2",
        "new": "minishlab/potion-base-8M",
        "note": "static distilled embeddings — a genuinely different architecture",
    },
    # The case the decision rule was changed for: a large upgrade bridged
    # imperfectly, which lands at a low ARR the bands read as `full_reindex`
    # while the break-even says bridging wins. Nothing else here exercises it.
    "large_upgrade": {
        "old": "minishlab/potion-base-8M",
        "new": "BAAI/bge-base-en-v1.5",
        "note": "weak old model, strong new one: low ARR, break-even above 1",
    },
}

DATASET = "beir/scifact"


def load_corpus(dataset: str, limit: int) -> tuple[list[str], list[str]]:
    import ir_datasets

    ds = ir_datasets.load(dataset)
    ids, texts = [], []
    for doc in ds.docs_iter():
        text = f"{getattr(doc, 'title', '')} {getattr(doc, 'text', '')}".strip()
        if text:
            ids.append(doc.doc_id)
            texts.append(text)
        if len(ids) >= limit:
            break
    return ids, texts


def load_queries(dataset: str, keep: set[str]) -> tuple[list[str], list[list[str]]]:
    """Real queries with human judgements, restricted to the corpus slice."""
    import ir_datasets

    ds = ir_datasets.load(f"{dataset}/test")
    qrels: dict[str, set[str]] = {}
    for qrel in ds.qrels_iter():
        if qrel.relevance > 0 and qrel.doc_id in keep:
            qrels.setdefault(qrel.query_id, set()).add(qrel.doc_id)
    texts = {q.query_id: q.text for q in ds.queries_iter()}

    queries, relevant = [], []
    for qid, rel in qrels.items():
        if qid in texts:
            queries.append(texts[qid])
            relevant.append(sorted(rel))
    return queries, relevant


def embed(
    model_id: str, documents: list[str], queries: list[str], device: str
) -> dict[str, object]:
    """Encode with the model's own retrieval instructions.

    The prefixes come from rebasis' profile table rather than being guessed
    here, so the fixture exercises the same prefix handling the tool uses.
    """
    from rebasis.embed import profile_for

    profile = profile_for(model_id)
    if model_id.startswith("minishlab/"):
        from model2vec import StaticModel

        static = StaticModel.from_pretrained(model_id)

        def encode(texts: list[str]) -> np.ndarray:
            return np.asarray(static.encode(texts), dtype=np.float32)
    else:
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer(model_id, device=device)

        def encode(texts: list[str]) -> np.ndarray:
            return np.asarray(
                st.encode(
                    texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False
                ),
                dtype=np.float32,
            )

    doc_prefix, query_prefix = profile.document_prefix or "", profile.query_prefix or ""
    return {
        "documents": np.asarray(encode([doc_prefix + t for t in documents]), dtype=np.float32),
        "queries": np.asarray(encode([query_prefix + q for q in queries]), dtype=np.float32),
        "dim": profile.dim,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("tests/golden/data"))
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ids, texts = load_corpus(args.dataset, args.limit)
    queries, relevant = load_queries(args.dataset, set(ids))
    print(f"{args.dataset}: {len(ids)} documents, {len(queries)} judged queries", flush=True)

    cache: dict[str, dict[str, object]] = {}
    manifest: dict[str, object] = {
        "dataset": args.dataset,
        "n_documents": len(ids),
        "n_queries": len(queries),
        "scenarios": {},
        "files": {},
    }

    for name, spec in SCENARIOS.items():
        for role in ("old", "new"):
            model_id = spec[role]
            if model_id not in cache:
                print(f"  embedding with {model_id}", flush=True)
                cache[model_id] = embed(model_id, texts, queries, args.device)

        path = args.out / f"{name}.npz"
        old, new = cache[spec["old"]], cache[spec["new"]]
        np.savez_compressed(
            path,
            ids=np.array(ids),
            texts=np.array(texts),
            queries=np.array(queries),
            relevant=np.array([json.dumps(r) for r in relevant]),
            old_documents=old["documents"],
            new_documents=new["documents"],
            old_queries=old["queries"],
            new_queries=new["queries"],
        )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["scenarios"][name] = {  # type: ignore[index]
            **spec,
            "old_dim": int(old["documents"].shape[1]),  # type: ignore[union-attr]
            "new_dim": int(new["documents"].shape[1]),  # type: ignore[union-attr]
        }
        manifest["files"][f"{name}.npz"] = digest  # type: ignore[index]
        print(f"  wrote {path.name}  sha256={digest[:16]}...", flush=True)

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest written to {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
