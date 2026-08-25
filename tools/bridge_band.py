"""Reproduce the bridge-band measurement — four configurations, one index.

``docs/bridge-band.md`` reports 62 runs and nothing in the repository could
produce a 63rd. This is that harness. It exists for three reasons, in order of
how much they matter:

**A cited number nobody can re-derive is not evidence.** The decision rule moved
twice on these runs, and the ROADMAP's whole account of where the headroom is
rests on them. A reader who wants to check the anti-correlation, or a
contributor who changes an adapter and wants to know what it did to the band,
needs to be able to run it.

**The cut-off is a parameter, not a constant.** Everything in that document is
measured at k=10, because that is what a RAG pipeline hands to a model. But the
bridge does not have to be the thing that produces the final ranking — it can be
the thing that produces a *candidate set* which the new model then reorders in
its own space, and what bounds that arrangement is recall@N, not nDCG@10. So
``--k`` takes a list, and every configuration is scored at every one of them.

**The scorer is not ours.** Grading a tool with its own metric code tests
consistency, not correctness, so the scoring goes through
`ranx <https://github.com/AmenRa/ranx>`_ exactly as the original runs did.

Five configurations, all against the same index::

    status quo     old query  -> old index    what you have today
    naive swap     new query  -> old index    just change the model
    bridged        adapter()  -> old index    what rebasis promises
    cascade@N      bridged top-N, reranked by the new model in its own space
    full reindex   new query  -> new index    the ceiling

The fourth is the one this harness was extended for. If the bridge produces a
*candidate set* rather than the final ranking, the only thing it can lose is a
relevant document that failed to reach the top N — everything after that is the
new model ranking in its own space, which is what a full reindex would have
done. So the arrangement is bounded by the bridge's **recall@N**, and rebasis'
band was measured entirely at nDCG@10. Measured rather than assumed: reranking
is not free of risk, and published counter-examples exist where a reranker makes
a strong first stage worse.

The adapter comes from the same ``probe_store`` -> ``save_adapter`` path the
``rebasis fit`` CLI runs, and is applied through the documented ``Bridge`` API,
so what is measured is the tool a user would run rather than a reimplementation
of it.

Embeddings are cached per (corpus, model, kind) as ``.npy``, because a three-rung
ladder over one corpus reuses each model's vectors twice and the ladder is the
expensive part::

    uv run --extra sentence-transformers --with ir-datasets --with ranx \\
        --with model2vec --with datasets python tools/bridge_band.py \\
        --corpus beir/cqadupstack/android --ladder default \\
        --k 10,100,200 --cascade 100,200 --out reports/band/rows.jsonl

``datasets`` is only needed for the ``mmteb`` group, which reads MTEB's own
layout from Hugging Face; everything else goes through ir_datasets.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from rebasis.types import EncodingProfile, FloatArray

#: The three-rung ladder the held-out runs used. Each rung is one step up in
#: capability, which is the axis the squeeze in `docs/bridge-band.md` runs along:
#: a bigger upgrade means a weaker old model and less for any adapter to map.
LADDERS: dict[str, tuple[tuple[str, str], ...]] = {
    "default": (
        ("minishlab/potion-base-8M", "sentence-transformers/all-MiniLM-L6-v2"),
        ("sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"),
        ("BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5"),
    ),
    # The two rungs that skip a step. A larger jump than the ladder takes in one
    # go, which is where the band is widest and least measured.
    "wide": (
        ("minishlab/potion-base-8M", "BAAI/bge-small-en-v1.5"),
        ("minishlab/potion-base-8M", "BAAI/bge-base-en-v1.5"),
        ("sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-base-en-v1.5"),
    ),
}

#: Every cqadupstack forum plus fiqa — the eleven the held-out section used, and
#: the two (`tex`, `wordpress`) it did not.
CORPORA: dict[str, tuple[str, ...]] = {
    "cqadupstack": tuple(
        f"beir/cqadupstack/{forum}"
        for forum in (
            "android",
            "english",
            "gaming",
            "gis",
            "mathematica",
            "physics",
            "programmers",
            "stats",
            "tex",
            "unix",
            "webmasters",
            "wordpress",
        )
    ),
    "heldout": (
        *(
            f"beir/cqadupstack/{forum}"
            for forum in (
                "android",
                "english",
                "gaming",
                "gis",
                "mathematica",
                "physics",
                "programmers",
                "stats",
                "tex",
                "unix",
                "webmasters",
                "wordpress",
            )
        ),
        "beir/fiqa/test",
    ),
    "beir": ("beir/scifact/test", "beir/nfcorpus/test", "beir/arguana"),
    # The three tasks Maystre et al. report their cross-model retrieval grid on
    # (arXiv:2510.13406, Figure 4), so this ladder can be read against published
    # numbers on the same collections. Their hard-negative variants rather than
    # the full sets: that is what the paper evaluates, and the full HotpotQA and
    # FEVER are 10.6M documents between them — an embedding bill of roughly 140
    # GB and 22 GPU-hours to answer a question the 390k-document variants answer.
    "mmteb": (
        "mmteb:mteb/HotpotQA_test_top_250_only_w_correct-v2",
        "mmteb:mteb/FEVER_test_top_250_only_w_correct-v2",
        "beir/trec-covid",
    ),
}

#: Prefix marking a corpus that comes from a Hugging Face dataset in the MTEB
#: layout (``corpus``/``queries``/``default`` configs) rather than from
#: ir_datasets.
MMTEB_PREFIX = "mmteb:"

#: Corpora evaluated with self-removal: a query is itself a document in the
#: collection, and the standard evaluation excludes a query's own document from
#: its results. Getting this wrong moved ArguAna's number by 0.2 nDCG the first
#: time round (`docs/bridge-band.md`, section 7).
SELF_REMOVAL = frozenset({"beir/arguana"})

#: Fitting budget and held-out set, matching the `rebasis fit` defaults. The
#: budget saturates near 4000 pairs; the next 20000 bought +0.001.
FIT_PAIRS = 4000
FIT_HELDOUT = 1000


# ── corpus ────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Corpus:
    """One retrieval collection: documents, judged queries, and the judgements."""

    name: str
    doc_ids: list[str]
    doc_texts: list[str]
    query_ids: list[str]
    query_texts: list[str]
    #: ``qrels[query_id][doc_id] = grade``, restricted to documents that are
    #: actually in ``doc_ids``.
    qrels: dict[str, dict[str, int]]

    @property
    def self_mask(self) -> np.ndarray | None:
        """Per-query document position to exclude, or ``None``.

        ArguAna's queries *are* documents. Letting one retrieve itself puts a
        guaranteed irrelevant hit at rank 1 and moves every metric.
        """
        if self.name not in SELF_REMOVAL:
            return None
        position = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        return np.array([position.get(q, -1) for q in self.query_ids], dtype=np.int64)


def load_corpus(dataset: str, *, limit: int | None = None) -> Corpus:
    """Read a corpus and its judged queries.

    Two sources, one shape. ``mmteb:<hf-name>`` reads MTEB's own layout from
    Hugging Face — three configs named ``corpus``, ``queries`` and ``default``;
    anything else is an ir_datasets name.

    Queries with no judged document inside the corpus slice are dropped rather
    than scored zero: they would count a guaranteed miss equally against every
    configuration, which changes the absolute numbers without changing any
    comparison.
    """
    if dataset.startswith(MMTEB_PREFIX):
        return _load_mmteb(dataset, limit=limit)

    import ir_datasets

    dataset_object = ir_datasets.load(dataset)

    doc_ids: list[str] = []
    doc_texts: list[str] = []
    for doc in dataset_object.docs_iter():
        text = f"{getattr(doc, 'title', '')} {getattr(doc, 'text', '')}".strip()
        if not text:
            continue
        doc_ids.append(doc.doc_id)
        doc_texts.append(text)
        if limit is not None and len(doc_ids) >= limit:
            break

    present = set(doc_ids)
    qrels: dict[str, dict[str, int]] = {}
    for qrel in dataset_object.qrels_iter():
        if qrel.relevance > 0 and qrel.doc_id in present:
            qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = int(qrel.relevance)

    query_ids: list[str] = []
    query_texts: list[str] = []
    for query in dataset_object.queries_iter():
        if query.query_id in qrels:
            query_ids.append(query.query_id)
            query_texts.append(query.text)

    return Corpus(
        name=dataset,
        doc_ids=doc_ids,
        doc_texts=doc_texts,
        query_ids=query_ids,
        query_texts=query_texts,
        qrels={qid: qrels[qid] for qid in query_ids},
    )


# ── embedding, cached ─────────────────────────────────────────────────


def _load_mmteb(dataset: str, *, limit: int | None = None) -> Corpus:
    """Read one of MTEB's retrieval datasets from Hugging Face.

    The hard-negative variants are the reason this loader exists. MMTEB pairs
    each test query with its correct documents plus the top 250 negatives a
    strong retriever surfaced, which keeps the collection at a few hundred
    thousand documents while leaving the task hard — and it is what
    arXiv:2510.13406 evaluates, so measuring anything else here would not be
    comparable with the numbers it publishes.
    """
    from datasets import load_dataset

    name = dataset.removeprefix(MMTEB_PREFIX)
    corpus = load_dataset(name, "corpus", split="test")
    queries = load_dataset(name, "queries", split="test")
    judgements = load_dataset(name, "default", split="test")

    doc_ids: list[str] = []
    doc_texts: list[str] = []
    for record in corpus:
        text = f"{record.get('title') or ''} {record.get('text') or ''}".strip()
        if not text:
            continue
        doc_ids.append(str(record["_id"]))
        doc_texts.append(text)
        if limit is not None and len(doc_ids) >= limit:
            break

    present = set(doc_ids)
    qrels: dict[str, dict[str, int]] = {}
    for row in judgements:
        if int(row["score"]) > 0 and str(row["corpus-id"]) in present:
            qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])

    query_ids: list[str] = []
    query_texts: list[str] = []
    for row in queries:
        if str(row["_id"]) in qrels:
            query_ids.append(str(row["_id"]))
            query_texts.append(str(row["text"]))

    return Corpus(
        name=dataset,
        doc_ids=doc_ids,
        doc_texts=doc_texts,
        query_ids=query_ids,
        query_texts=query_texts,
        qrels={qid: qrels[qid] for qid in query_ids},
    )


def _slug(text: str) -> str:
    return text.replace("/", "_").replace(":", "_")


def _encoder(model_id: str, device: str) -> Any:
    """A callable that turns texts into unit vectors.

    model2vec's static models are not sentence-transformers models and have no
    device to place; everything else goes through SentenceTransformer, batching
    the way `make_golden.py` does so the two produce comparable vectors.
    """
    if model_id.startswith("minishlab/"):
        from model2vec import StaticModel

        static = StaticModel.from_pretrained(model_id)

        def encode_static(texts: list[str]) -> FloatArray:
            return np.asarray(static.encode(texts), dtype=np.float32)

        return encode_static

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, device=device)

    def encode(texts: list[str]) -> FloatArray:
        return np.asarray(
            model.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )

    return encode


def embed_cached(  # noqa: PLR0913 - every argument is part of the cache key or the work
    *,
    model_id: str,
    profile: EncodingProfile,
    texts: Sequence[str],
    kind: str,
    corpus_name: str,
    part: str,
    cache_dir: Path,
    device: str,
    encoder_cache: dict[str, Any],
) -> FloatArray:
    """Encode ``texts`` with the model's own prefix for ``kind``, once ever.

    The cache key carries the kind as well as the model: an asymmetric model
    encodes a document one way and a query another, and the ladder needs both
    encodings of the *documents* — the second is what lets `auto` measure the
    query-specific fit strategy against the shared one.
    """
    path = cache_dir / f"{_slug(corpus_name)}__{_slug(model_id)}__{part}_{kind}.npy"
    if path.exists():
        return np.load(path)

    if model_id not in encoder_cache:
        print(f"    loading {model_id}", flush=True)
        encoder_cache[model_id] = _encoder(model_id, device)

    prefix = profile.prefix_for(kind)  # type: ignore[arg-type]
    started = time.perf_counter()
    vectors = encoder_cache[model_id]([prefix + t for t in texts])
    vectors = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))

    path.parent.mkdir(parents=True, exist_ok=True)
    # Written through a handle rather than by name: `np.save` appends `.npy` to
    # a path that does not already end in it, which would leave the temporary
    # file under a third name and the rename pointing at nothing.
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, vectors)
    tmp.replace(path)
    print(
        f"    embedded {len(texts):,} {part} as {kind} with {model_id} "
        f"in {time.perf_counter() - started:.0f}s -> {path.name}",
        flush=True,
    )
    return vectors


@dataclass(slots=True)
class Encoded:
    """Everything one model produced for one corpus."""

    profile: EncodingProfile
    documents: FloatArray
    queries: FloatArray
    #: Documents encoded the way a *query* is encoded. Only meaningful — and
    #: only computed — for an asymmetric model.
    documents_as_queries: FloatArray | None


def encode_corpus(
    *,
    model_id: str,
    corpus: Corpus,
    cache_dir: Path,
    device: str,
    encoder_cache: dict[str, Any],
) -> Encoded:
    """Encode a corpus with one model, reusing whatever is already on disk."""
    from rebasis.embed import profile_for

    profile = profile_for(model_id)
    shared = {
        "model_id": model_id,
        "profile": profile,
        "corpus_name": corpus.name,
        "cache_dir": cache_dir,
        "device": device,
        "encoder_cache": encoder_cache,
    }
    documents = embed_cached(texts=corpus.doc_texts, kind="document", part="docs", **shared)
    queries = embed_cached(texts=corpus.query_texts, kind="query", part="queries", **shared)
    documents_as_queries = (
        None
        if profile.symmetric
        else embed_cached(texts=corpus.doc_texts, kind="query", part="docs", **shared)
    )
    return Encoded(
        profile=profile,
        documents=documents,
        queries=queries,
        documents_as_queries=documents_as_queries,
    )


# ── the run ───────────────────────────────────────────────────────────


@contextlib.contextmanager
def _temporary_directory() -> Iterator[Path]:
    path = Path(tempfile.mkdtemp(prefix="rebasis-band-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def fit_bridge(
    corpus: Corpus, old: Encoded, new: Encoded, *, seed: int, device: str
) -> tuple[Any, dict[str, Any]]:
    """Fit an adapter the way ``rebasis fit`` does, and load it as a ``Bridge``.

    Round-tripping through the ``.rbs`` file rather than using the in-memory
    adapter is deliberate: the serialised form is what a user actually holds,
    and a measurement of something else would be measuring something else.
    """
    from rebasis.core import save_adapter
    from rebasis.embed import PrecomputedEmbedder
    from rebasis.probe.session import probe_store
    from rebasis.serve import Bridge
    from rebasis.store import MemoryStore

    store = MemoryStore(corpus.doc_ids, old.documents, corpus.doc_texts)

    # An asymmetric model needs the document texts in the query table too: the
    # probe encodes the sample a second time the way a query is encoded, and a
    # table holding only the query strings would refuse it.
    document_table = dict(zip(corpus.doc_texts, new.documents, strict=True))
    query_table: dict[str, FloatArray] = dict(zip(corpus.query_texts, new.queries, strict=True))
    if new.documents_as_queries is not None:
        query_table.update(zip(corpus.doc_texts, new.documents_as_queries, strict=True))

    embedder = PrecomputedEmbedder(new.profile, document_table, query_vectors=query_table)

    result, _ = probe_store(
        store,
        embedder,
        size=FIT_PAIRS + FIT_HELDOUT,
        heldout=FIT_HELDOUT,
        k=10,
        seed=seed,
        device=device,
    )
    if result.adapter is None:
        msg = f"no adapter could be fitted for {corpus.name}"
        raise RuntimeError(msg)

    with _temporary_directory() as directory:
        path = save_adapter(
            result.adapter,
            directory / "adapter.rbs",
            direction="query_to_old",
            old_profile=old.profile,
            new_profile=new.profile,
            calibrator=result.calibrator,
            evaluation=result.to_dict(),
        )
        bridge = Bridge.load(path, verify=True)

    return bridge, {
        "adapter_type": result.best.name,
        "arr_r10": round(result.best.arr, 4),
        "n_fit_pairs": result.n_fit_pairs,
        "n_params": result.best.n_params,
    }


def _run_dict(
    corpus: Corpus, indices: np.ndarray, scores: np.ndarray
) -> dict[str, dict[str, float]]:
    """Turn a top-k result into the mapping ranx wants."""
    doc_ids = corpus.doc_ids
    return {
        query_id: {
            doc_ids[int(position)]: float(score)
            for position, score in zip(row, score_row, strict=True)
        }
        for query_id, row, score_row in zip(corpus.query_ids, indices, scores, strict=True)
    }


def score(
    corpus: Corpus,
    runs: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    cutoffs: Sequence[int],
) -> dict[str, dict[str, float]]:
    """Score every configuration with ranx, at every cut-off."""
    from ranx import Qrels, Run, evaluate

    qrels = Qrels(corpus.qrels)
    metrics = [f"{name}@{k}" for k in cutoffs for name in ("ndcg", "recall", "mrr")]

    scored: dict[str, dict[str, float]] = {}
    for label, (indices, values) in runs.items():
        measured = evaluate(qrels, Run(_run_dict(corpus, indices, values)), metrics)
        scored[label] = {metric: round(float(measured[metric]), 4) for metric in metrics}
    return scored


def measure(  # noqa: PLR0913 - one argument per input to a run
    corpus: Corpus,
    old_model: str,
    new_model: str,
    *,
    cache_dir: Path,
    cutoffs: Sequence[int],
    device: str,
    seed: int,
    encoder_cache: dict[str, Any],
    cascade: Sequence[int] = (),
) -> dict[str, Any]:
    """One row of the band: every configuration over one corpus and model pair."""
    from rebasis.compute import resolve_device, top_k_search, using_device

    started = time.perf_counter()
    shared = {"corpus": corpus, "cache_dir": cache_dir, "device": device}
    old = encode_corpus(model_id=old_model, encoder_cache=encoder_cache, **shared)
    new = encode_corpus(model_id=new_model, encoder_cache=encoder_cache, **shared)

    bridge, fit_summary = fit_bridge(corpus, old, new, seed=seed, device=device)
    mapped = bridge.to_index_space(new.queries)

    # The pre-fit signal, measured on the same corpus as everything else so the
    # bound and the retention it bounds can be read against each other across
    # the whole ladder. One Gram-matrix difference; no fit.
    from rebasis.core import geometry_bound

    geometry = geometry_bound(new.documents, old.documents, seed=seed)

    depth = max(cutoffs)
    self_mask = corpus.self_mask

    with using_device(resolve_device(device)):
        runs: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "status_quo": top_k_search(old.queries, old.documents, k=depth, self_mask=self_mask),
            "bridged": top_k_search(mapped, old.documents, k=depth, self_mask=self_mask),
            "full_reindex": top_k_search(new.queries, new.documents, k=depth, self_mask=self_mask),
        }
        # Only measurable when the dimensions agree: otherwise the new vector
        # cannot physically enter the old index at all, and the naive swap is
        # not a thing a user could do rather than a thing that scores badly.
        if new.queries.shape[1] == old.documents.shape[1]:
            runs["naive_swap"] = top_k_search(
                new.queries, old.documents, k=depth, self_mask=self_mask
            )

        # The two-stage arrangement, measured rather than estimated. The
        # candidate set comes out of the old index through the bridge; the
        # ranking within it is the new model scoring its own vectors, which is
        # exactly the ranking a full reindex would produce over those same
        # documents. Only the top 10 of the result is kept, so it is directly
        # comparable with every other row at k=10.
        bridged_indices, _ = runs["bridged"]
        for stage in cascade:
            if stage > depth:
                continue
            runs[f"cascade@{stage}"] = _rerank(
                new.queries, new.documents, bridged_indices[:, :stage], depth=depth
            )

    scored = score(corpus, runs, cutoffs=cutoffs)

    return {
        "corpus": corpus.name,
        "old_model": old_model,
        "new_model": new_model,
        "old_dim": int(old.documents.shape[1]),
        "new_dim": int(new.documents.shape[1]),
        "n_documents": len(corpus.doc_ids),
        "n_queries": len(corpus.query_ids),
        "self_removal": corpus.name in SELF_REMOVAL,
        "cutoffs": list(cutoffs),
        "cascade": list(cascade),
        "fit": fit_summary,
        "geometry": geometry.to_dict(),
        "scores": scored,
        "seed": seed,
        "duration_seconds": round(time.perf_counter() - started, 1),
    }


def _rerank(
    queries: FloatArray, documents: FloatArray, candidates: np.ndarray, *, depth: int
) -> tuple[np.ndarray, FloatArray]:
    """Reorder each query's candidate set by the new model's own similarity.

    Per query rather than as one matrix: the candidate sets differ between
    queries, so there is no shared document axis to multiply against. At a few
    thousand queries and a few hundred candidates this is a rounding error
    beside the embedding that produced them.

    The result is padded back out to ``depth`` with the candidates that did not
    make the cut, in their original order, so every configuration hands `ranx`
    a list of the same length and the metrics at every cut-off stay defined.
    """
    n_queries = queries.shape[0]
    out_indices = np.empty((n_queries, depth), dtype=np.int64)
    out_scores = np.empty((n_queries, depth), dtype=np.float32)

    for row in range(n_queries):
        chosen = candidates[row]
        scores = documents[chosen] @ queries[row]
        order = np.argsort(-scores)
        ranked = chosen[order]
        # Anything beyond the candidate set keeps its bridged order below the
        # reranked block, which is what a real cascade would serve if asked for
        # more than N.
        tail = [int(i) for i in candidates[row] if int(i) not in set(ranked.tolist())]
        filled = list(ranked.tolist()) + tail
        while len(filled) < depth:
            filled.append(int(filled[-1]) if filled else 0)
        out_indices[row] = filled[:depth]
        padded = list(scores[order].tolist())
        while len(padded) < depth:
            padded.append(float(padded[-1]) - 1.0 if padded else 0.0)
        out_scores[row] = padded[:depth]

    return out_indices, out_scores


# ── entry point ───────────────────────────────────────────────────────


def resolve_corpora(names: Sequence[str]) -> list[str]:
    """Expand group names into dataset names, keeping order and dropping repeats."""
    resolved: list[str] = []
    for name in names:
        for dataset in CORPORA.get(name, (name,)):
            if dataset not in resolved:
                resolved.append(dataset)
    return resolved


def already_done(out: Path) -> set[tuple[str, str, str]]:
    """Rows the output file already holds, so a re-run resumes rather than repeats.

    The harness is hours of GPU time over a ladder; an interrupted run that had
    to start again would mean nobody ever finishes one.
    """
    if not out.exists():
        return set()
    done: set[tuple[str, str, str]] = set()
    for line in out.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        with contextlib.suppress(json.JSONDecodeError, KeyError):
            row = json.loads(line)
            done.add((row["corpus"], row["old_model"], row["new_model"]))
    return done


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        action="append",
        default=None,
        help=f"ir_datasets name, or a group: {', '.join(sorted(CORPORA))}",
    )
    parser.add_argument("--ladder", default="default", choices=sorted(LADDERS))
    parser.add_argument("--k", default="10,100,200", help="Comma-separated cut-offs")
    parser.add_argument(
        "--cascade",
        default="100,200",
        help="Candidate-set sizes to measure a two-stage arrangement at; empty to skip",
    )
    parser.add_argument("--out", type=Path, default=Path("reports/band/rows.jsonl"))
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / "band-cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument(
        "--force", action="store_true", help="Re-measure rows the output already holds"
    )
    args = parser.parse_args(argv)

    cutoffs = [int(part) for part in args.k.split(",") if part.strip()]
    cascade = [int(part) for part in args.cascade.split(",") if part.strip()]
    datasets = resolve_corpora(args.corpus or ["heldout"])
    rungs = LADDERS[args.ladder]
    done = set() if args.force else already_done(args.out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    planned = [
        (dataset, old, new)
        for dataset in datasets
        for old, new in rungs
        if args.force or (dataset, old, new) not in done
    ]
    print(
        f"{len(planned)} runs to measure "
        f"({len(datasets)} corpora x {len(rungs)} rungs, {len(done)} already done)",
        flush=True,
    )

    encoder_cache: dict[str, Any] = {}
    for dataset in datasets:
        pending = [(o, n) for d, o, n in planned if d == dataset]
        if not pending:
            continue
        print(f"\n=== {dataset} ===", flush=True)
        corpus = load_corpus(dataset, limit=args.limit_docs)
        print(
            f"  {len(corpus.doc_ids):,} documents, {len(corpus.query_ids):,} judged queries",
            flush=True,
        )
        for old_model, new_model in pending:
            print(f"  -- {old_model} -> {new_model}", flush=True)
            row = measure(
                corpus,
                old_model,
                new_model,
                cache_dir=args.cache_dir,
                cutoffs=cutoffs,
                device=args.device,
                seed=args.seed,
                encoder_cache=encoder_cache,
                cascade=cascade,
            )
            with args.out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            headline = row["scores"]["bridged"]
            print(
                "     bridged "
                + "  ".join(f"ndcg@{k}={headline[f'ndcg@{k}']:.3f}" for k in cutoffs)
                + "  "
                + "  ".join(f"recall@{k}={headline[f'recall@{k}']:.3f}" for k in cutoffs),
                flush=True,
            )

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
