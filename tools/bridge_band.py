"""Reproduce the bridge-band measurement — four configurations, one index.

``docs/bridge-band.md`` reports 62 runs and nothing in the repository could
produce a 63rd. This is that harness. It exists for four reasons, in order of
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

**The protocol is a parameter too.** Published work on this same problem is
evaluated against a ground truth the *new model* defines — held-out documents
standing in for queries, scored against the new model's own exact kNN. That is
rebasis' T0 tier, and a number measured there is not comparable with a number
measured against human judgements however similar the two look. ``--protocol``
selects which one runs::

    t1-judged             real queries        human judgements
    t0-knn-real-queries   real queries        the new model's exact kNN
    t0-knn                held-out documents  the new model's exact kNN

Two things separate the ends of that list — who asks, and what counts as a right
answer — so a difference between them cannot be attributed to either. The middle
row changes only the second, which is the whole reason it exists. All three write
to the same file and every row names its protocol, so they can never be read as
one series.

Six configurations, all against the same index::

    status quo          old query  -> old index    what you have today
    naive swap          new query  -> old index    just change the model
    naive swap padded   the same, when the dimensions do not agree
    bridged             adapter()  -> old index    what rebasis promises
    cascade@N           bridged top-N, reranked by the new model in its own space
    full reindex        new query  -> new index    the ceiling

The fifth is the one this harness was extended for. If the bridge produces a
*candidate set* rather than the final ranking, the only thing it can lose is a
relevant document that failed to reach the top N — everything after that is the
new model ranking in its own space, which is what a full reindex would have
done. So the arrangement is bounded by the bridge's **recall@N**, and rebasis'
band was measured entirely at nDCG@10. Measured rather than assumed: reranking
is not free of risk, and published counter-examples exist where a reranker makes
a strong first stage worse.

The third exists because a naive swap is only a thing a user can *do* when the
two models share a dimension, and neither of the two published evaluations this
harness is read against has a model pair that does. Zero-padding the shorter
space is the only convention under which the configuration is defined at all,
and it preserves every inner product, so what it measures is the swap rather
than the padding.

**Padding is this harness' convention, not an operation anyone has.** Where the
new model is the wider one — which is most of the ladder — the padding falls on
the *indexed documents*, and widening an index means rewriting it, which is the
one thing the whole tool exists to avoid. ``IdentityAdapter`` is not a precedent
for it either: it pads only when the new model is narrower, and truncates in the
other direction. So this row is reported under its own name, as a *different
configuration* from the naive swap rather than a fallback for it, and it is a
figure for a configuration nobody can deploy.

The adapter comes from the same ``probe_store`` -> ``save_adapter`` path the
``rebasis fit`` CLI runs, and is applied through the documented ``Bridge`` API,
so what is measured is the tool a user would run rather than a reimplementation
of it.

Per-query scores are written beside the rows, one JSON file per run, because a
mean is not enough to test anything. ``docs/bridge-band.md`` counts how often the
break-even predicted the outcome and attaches no significance to the count;
`ranx` can run a paired randomisation test on exactly these arrays, and it needs
the arrays. Each file carries the query ids next to the scores so two runs'
arrays cannot be lined up by position and quietly compared.

Embeddings are cached per (corpus, model, kind) as ``.npy``, because a three-rung
ladder over one corpus reuses each model's vectors twice and the ladder is the
expensive part::

    uv run --extra sentence-transformers --with ir-datasets --with ranx \\
        --with model2vec --with datasets python tools/bridge_band.py \\
        --corpus beir/cqadupstack/android --ladder default \\
        --k 10,100,200 --cascade 100,200 --out reports/band/rows.jsonl

``datasets`` is only needed for the ``mmteb`` and ``drift-adapter`` groups, which
read Hugging Face directly; everything else goes through ir_datasets.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
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
    # The pair Vejendla evaluates every text result on (arXiv:2509.23471,
    # section 4). Carried so that the model pair can be held fixed as well as
    # the corpus when this harness is read against those numbers — otherwise a
    # difference has two candidate causes and the comparison establishes
    # nothing.
    "drift-adapter": (
        (
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
        ),
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
    # The three text collections arXiv:2509.23471 evaluates on. They are
    # classification datasets with no queries and no judgements, so they can
    # only ever be run under `t0-knn` — which is the point: they are the
    # paper's corpora, and running them establishes whether the corpus or the
    # protocol is what separates its band from rebasis'.
    "drift-adapter": (
        "hfdocs:fancyzhx/ag_news",
        "hfdocs:fancyzhx/dbpedia_14",
        "hfdocs:dair-ai/emotion:split",
    ),
}

#: Prefix marking a corpus that comes from a Hugging Face dataset in the MTEB
#: layout (``corpus``/``queries``/``default`` configs) rather than from
#: ir_datasets.
MMTEB_PREFIX = "mmteb:"

#: Prefix marking a plain Hugging Face dataset read for its text column alone —
#: ``hfdocs:<repo>[:<config>[:<split>]]``. It yields documents and nothing else,
#: which is all the ``t0-knn`` protocol needs and less than ``t1-judged``
#: requires; asking for the latter is an error rather than an empty run.
HFDOCS_PREFIX = "hfdocs:"

#: The three evaluation protocols, and the whole reason this harness gained a
#: flag. Two axes vary between them — where the queries come from, and what
#: counts as a right answer — and the third protocol exists precisely so that
#: the two can be told apart::
#:
#:     t1-judged             real queries        human judgements
#:     t0-knn-real-queries   real queries        the new model's exact kNN
#:     t0-knn                held-out documents  the new model's exact kNN
#:
#: The first is rebasis' own tier and every existing row. The last is the
#: protocol arXiv:2509.23471 evaluates on. The middle one holds the query
#: distribution fixed and changes only the ground truth, which is the only way
#: to say how much of the difference between the two ends is which. See
#: ``docs/vs-drift-adapter.md``.
PROTOCOLS = ("t1-judged", "t0-knn-real-queries", "t0-knn")

#: Corpora evaluated with self-removal: a query is itself a document in the
#: collection, and the standard evaluation excludes a query's own document from
#: its results. Getting this wrong moved ArguAna's number by 0.2 nDCG the first
#: time round (`docs/bridge-band.md`, section 7).
SELF_REMOVAL = frozenset({"beir/arguana"})

#: Fitting budget and held-out set, matching the `rebasis fit` defaults. The
#: budget saturates near 4000 pairs; the next 20000 bought +0.001.
FIT_PAIRS = 4000
FIT_HELDOUT = 1000

#: How many documents stand in for queries under ``t0-knn``, and how deep the
#: ground truth goes. The depth is 10 because that is what Recall@10 means; the
#: proxy count is 1000 rather than the 10,000 arXiv:2509.23471 uses because these
#: corpora are tens of thousands of documents, not a million, and holding out ten
#: thousand of them would be removing a tenth of the collection being searched.
T0_PROXIES = 1000
T0_TRUTH_K = 10

#: Never hold out more than this share of a collection as query proxies. A
#: proxy is removed from the index it is searching, so a large share changes the
#: collection rather than sampling it.
T0_MAX_PROXY_SHARE = 0.1


# ── corpus ────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Corpus:
    """One retrieval collection: documents, queries, and what counts as right."""

    name: str
    doc_ids: list[str]
    doc_texts: list[str]
    query_ids: list[str]
    query_texts: list[str]
    #: ``qrels[query_id][doc_id] = grade``, restricted to documents that are
    #: actually in ``doc_ids``.
    qrels: dict[str, dict[str, int]]
    #: Which protocol assembled this collection. Three of them write to one
    #: file, and a reader who cannot tell them apart would average a ratio
    #: against human relevance with a ratio against a model's own neighbours.
    protocol: str = "t1-judged"
    #: Where the judgements came from, in enough detail to check.
    truth: dict[str, Any] = field(default_factory=lambda: {"source": "human"})
    #: True when the query set was physically removed from ``doc_ids``. That is
    #: what makes self-exclusion structural under ``t0-knn`` rather than a flag
    #: someone has to remember to set.
    queries_removed: bool = False
    #: Set only where a collection holds several rows that are the same document
    #: — copies made by ``--replicate``. See :attr:`score_ids`.
    duplicate_of: list[str] = field(default_factory=list)
    #: The ``--limit-docs`` this collection was truncated to, if any. It exists
    #: to keep :func:`embed_cached` honest: the cache is keyed on a corpus name,
    #: and a truncated run shares that name with the full one while holding
    #: different vectors. See :attr:`cache_name`.
    limit: int | None = None

    @property
    def cache_name(self) -> str:
        """The name this collection's embeddings are cached under.

        The corpus name, plus the truncation when there is one. Without that
        suffix a ``--limit-docs`` run and a full run write to the same ``.npy``
        and the second to load gets the first's vectors — which raises rather
        than lying, because the length check in :func:`fit_bridge` catches it,
        but it leaves a poisoned file that every later run trips over. An
        untruncated run keeps the bare name, so a warm cache stays warm.
        """
        return self.name if self.limit is None else f"{self.name}@{self.limit}"

    @property
    def score_ids(self) -> list[str]:
        """The name each indexed row is *scored* under.

        Normally its own id. Where copies exist it is the underlying document,
        so that retrieving a different copy counts as finding the thing rather
        than as a miss. Scoring copies as distinct would make a duplicated
        corpus look harder than it is, which is the opposite of the reading
        being tested.
        """
        return self.duplicate_of or self.doc_ids

    @property
    def self_mask(self) -> np.ndarray | None:
        """Per-query document position to exclude, or ``None``.

        ArguAna's queries *are* documents. Letting one retrieve itself puts a
        guaranteed irrelevant hit at rank 1 and moves every metric — and under
        ``t0-knn-real-queries`` it would also put itself in its own ground
        truth, which is worse: the configuration would then be scored on a hit
        it is guaranteed to get.

        ``t0-knn`` needs no mask at all, and that is structural rather than an
        omission: its query proxies are removed from ``doc_ids`` before anything
        is encoded, so a proxy cannot be retrieved, cannot enter its own ground
        truth and cannot be drawn as a fit pair.
        """
        if self.queries_removed or self.name not in SELF_REMOVAL:
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
    if dataset.startswith(HFDOCS_PREFIX):
        return _load_hfdocs(dataset, limit=limit)

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
        limit=limit,
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
        limit=limit,
    )


#: Columns a plain Hugging Face dataset might hold its document text in, most
#: specific first. Guessing is acceptable here and nowhere else in this file:
#: the wrong column produces empty strings, which is loud.
_TEXT_COLUMNS = ("text", "content", "sentence", "document")


def _load_hfdocs(dataset: str, *, limit: int | None = None) -> Corpus:
    """Read a plain Hugging Face dataset for its documents and nothing else.

    ``hfdocs:<repo>[:<config>[:<split>]]``. The classification corpora
    arXiv:2509.23471 evaluates on — AG-News, DBpedia-14, Emotion — have no
    queries and no relevance judgements at all, which is precisely why that
    paper's ground truth has to be a model's own nearest neighbours. This
    loader returns a collection with an empty query set, and
    :func:`document_proxy_view` supplies the rest under ``t0-knn``. Under
    ``t1-judged`` the run fails rather than scoring an empty query set.
    """
    from datasets import load_dataset

    repository, _, tail = dataset.removeprefix(HFDOCS_PREFIX).partition(":")
    configuration, _, split = tail.partition(":")
    records = load_dataset(repository, configuration or None, split=split or "train")

    columns = set(records.column_names)
    body = next((name for name in _TEXT_COLUMNS if name in columns), None)
    if body is None:
        msg = f"{dataset}: no text column among {sorted(columns)}"
        raise RuntimeError(msg)

    doc_ids: list[str] = []
    doc_texts: list[str] = []
    for position, record in enumerate(records):
        title = str(record.get("title") or "") if "title" in columns else ""
        text = f"{title} {record.get(body) or ''}".strip()
        if not text:
            continue
        doc_ids.append(str(position))
        doc_texts.append(text)
        if limit is not None and len(doc_ids) >= limit:
            break

    return Corpus(
        name=dataset,
        doc_ids=doc_ids,
        doc_texts=doc_texts,
        query_ids=[],
        query_texts=[],
        qrels={},
        limit=limit,
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
        "corpus_name": corpus.cache_name,
        "cache_dir": cache_dir,
        "device": device,
        "encoder_cache": encoder_cache,
    }
    documents = embed_cached(texts=corpus.doc_texts, kind="document", part="docs", **shared)
    # A corpus loaded for `t0-knn` alone has no query set: its queries are
    # documents, chosen later. Encoding an empty list is not a smaller job, it
    # is a different one that some encoders refuse outright.
    queries = (
        embed_cached(texts=corpus.query_texts, kind="query", part="queries", **shared)
        if corpus.query_texts
        else np.empty((0, documents.shape[1]), dtype=np.float32)
    )
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


# ── the document-proxy protocol ───────────────────────────────────────


def _as_queries(encoded: Encoded) -> FloatArray:
    """The documents, encoded the way a query is encoded.

    For a symmetric model those are the same vectors and there is nothing to
    compute; for an asymmetric one they are a second encoding the cache already
    holds, because the ladder needs it to compare fit strategies. Either way a
    proxy is encoded as a query rather than as a document, which is ADR 8's
    decision and what the serving path actually does.
    """
    if encoded.documents_as_queries is None:
        return encoded.documents
    return encoded.documents_as_queries


def knn_qrels(
    corpus: Corpus,
    new: Encoded,
    *,
    truth_k: int,
    depth: int,
    device: str,
) -> dict[str, dict[str, int]]:
    """Judgements from the new model's own exhaustive kNN, as the paper defines them.

    arXiv:2509.23471, section 4: "The ground truth for retrieval (used to
    calculate Recall@k and MRR) is established by performing an exhaustive
    k-nearest neighbor search for each query within the 1M item database using
    embeddings generated by the new model for both queries and database items."

    Exhaustive is the operative word and it is not a detail. An approximate
    index would make this cheaper and the number meaningless, which is why
    ``ROADMAP.md`` lists approximate ground truth among the things this project
    will not do.

    The grades are binary because Recall@k and MRR are set-and-first-hit
    metrics and that is what the paper reports. One consequence is worth
    knowing before reading any row: the ``full_reindex`` configuration scores
    exactly 1.0 here **by construction**, because it is the same computation
    that produced the judgements. That is the paper's ``ARR = 1.0`` oracle, and
    it is a check on the wiring rather than a result.

    **The search runs to ``depth`` and the judgements are its first ``truth_k``,
    rather than a second search at ``truth_k``.** Two searches over the same
    vectors return the same ranking only while no two documents score exactly
    the same, and a corpus holding duplicate documents breaks that: ``argpartition``
    is free to return a different member of a tied group for a different ``k``,
    and the oracle then scores below 1.0 against a ground truth it produced
    itself. Measured, on AG-News replicated eightfold: 0.7997. Deriving both from
    one search makes the ground truth a prefix of the oracle's own answer, which
    is what it was always meant to be.
    """
    from rebasis.compute import resolve_device, top_k_search, using_device

    with using_device(resolve_device(device)):
        ranked, _ = top_k_search(
            new.queries, new.documents, k=max(depth, truth_k), self_mask=corpus.self_mask
        )
    return {
        query_id: {corpus.score_ids[int(position)]: 1 for position in row[:truth_k]}
        for query_id, row in zip(corpus.query_ids, ranked, strict=True)
    }


def knn_truth_view(
    corpus: Corpus, new: Encoded, *, truth_k: int, depth: int, device: str
) -> Corpus:
    """The same queries, judged by the new model instead of by people.

    This protocol exists to answer one question and no other. Between
    ``t1-judged`` and ``t0-knn`` two things change at once — who asks and what
    counts as right — so a difference between them cannot be attributed. Holding
    the query set fixed and changing only the ground truth separates the two,
    and the residual is the query distribution.
    """
    return Corpus(
        name=corpus.name,
        doc_ids=corpus.doc_ids,
        doc_texts=corpus.doc_texts,
        query_ids=corpus.query_ids,
        query_texts=corpus.query_texts,
        qrels=knn_qrels(corpus, new, truth_k=truth_k, depth=depth, device=device),
        protocol="t0-knn-real-queries",
        truth={
            "source": "knn",
            "model": new.profile.model_id,
            "k": truth_k,
            "exact": True,
            "queries": "the judged query set",
            "grades": "binary",
        },
    )


def document_proxy_view(  # noqa: PLR0913 - the split, the truth and where to run it
    corpus: Corpus,
    old: Encoded,
    new: Encoded,
    *,
    n_proxies: int,
    truth_k: int,
    depth: int,
    seed: int,
    device: str,
    replicate: int = 1,
) -> tuple[Corpus, Encoded, Encoded]:
    """Rebuild a corpus the way arXiv:2509.23471 evaluates one.

    Two things change and everything downstream is untouched. **The queries
    become documents** — a random sample of the collection, removed from it, in
    the paper's own arrangement ("we use 10,000 documents from their respective
    test sets as queries. These query documents are distinct from the items in
    the 1M-item database", section 4). And **the judgements become a model's
    own neighbours** — the new model's exhaustive kNN over what is left, which
    is the paper's definition word for word.

    The removal is what makes the self-exclusion structural. A proxy is not in
    ``doc_ids``, so it cannot be retrieved, cannot appear in its own ground
    truth, and cannot be drawn as a fit pair by ``probe_store`` — the paper's
    "query embeddings are strictly held out and are never seen during any phase
    of adapter training" falls out of the data rather than being enforced by a
    check somebody has to remember. ``docs/bridge-band.md`` section 7 records
    what forgetting the equivalent convention cost on ArguAna: 0.2 nDCG.

    ``replicate`` copies each remaining document that many times, and exists to
    test one specific reading of the paper's corpus construction. The paper
    states a database of "1 million items randomly sampled from their respective
    training sets" for AG-News, whose training split holds 120,000 rows — which
    is only reachable by sampling **with replacement**, at roughly eight copies
    per document. Under that reading a query's exact top-10 is mostly copies of
    one underlying document, an adapter that finds it finds all of them at once,
    and Recall@10 stops measuring neighbour-set overlap and starts measuring a
    nearest-neighbour hit rate. The copies are made after the query proxies are
    removed, so no proxy has a twin left in the index — which is the arrangement
    the paper describes, where the queries come from a different split entirely.
    """
    n_documents = len(corpus.doc_ids)
    ceiling = max(1, int(n_documents * T0_MAX_PROXY_SHARE))
    take = min(n_proxies, ceiling)
    if take < 1 or n_documents - take < truth_k:
        msg = (
            f"{corpus.name}: {n_documents} documents cannot support "
            f"{take} query proxies with a depth-{truth_k} ground truth"
        )
        raise RuntimeError(msg)

    rng = np.random.default_rng(seed)
    proxies = np.sort(rng.choice(n_documents, size=take, replace=False))
    kept = np.setdiff1d(np.arange(n_documents), proxies, assume_unique=True)
    # Copies are consecutive rather than interleaved so that a reader of the
    # document ids can see the structure, and so that the tiling below is one
    # contiguous take rather than an index computation to get wrong.
    keep = np.repeat(kept, replicate) if replicate > 1 else kept
    copies = np.tile(np.arange(replicate), len(kept)) if replicate > 1 else None

    def restrict(encoded: Encoded) -> Encoded:
        return Encoded(
            profile=encoded.profile,
            documents=encoded.documents[keep],
            queries=_as_queries(encoded)[proxies],
            documents_as_queries=(
                None if encoded.documents_as_queries is None else encoded.documents_as_queries[keep]
            ),
        )

    old_view, new_view = restrict(old), restrict(new)
    underlying = [corpus.doc_ids[i] for i in keep]
    doc_ids = underlying
    if copies is not None:
        doc_ids = [f"{doc_id}#{copy}" for doc_id, copy in zip(underlying, copies, strict=True)]

    view = Corpus(
        name=corpus.name,
        doc_ids=doc_ids,
        duplicate_of=[] if copies is None else underlying,
        doc_texts=[corpus.doc_texts[i] for i in keep],
        # Prefixed rather than reused bare. A query id that is also a document
        # id invites somebody to line two protocols' per-query arrays up by name.
        query_ids=[f"doc:{corpus.doc_ids[i]}" for i in proxies],
        query_texts=[corpus.doc_texts[i] for i in proxies],
        qrels={},
        protocol="t0-knn",
        truth={
            "source": "knn",
            "model": new.profile.model_id,
            "k": truth_k,
            "exact": True,
            "queries": "held-out documents, encoded as queries",
            "grades": "binary",
            "replicate": replicate,
        },
        queries_removed=True,
    )
    view.qrels = knn_qrels(view, new_view, truth_k=truth_k, depth=depth, device=device)
    return view, old_view, new_view


# ── the run ───────────────────────────────────────────────────────────


@contextlib.contextmanager
def _temporary_directory() -> Iterator[Path]:
    path = Path(tempfile.mkdtemp(prefix="rebasis-band-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@contextlib.contextmanager
def fixed_low_rank(rank: int | None) -> Iterator[None]:
    """Pin the low-rank adapter's rank, which no command line can otherwise do.

    This reaches past the CLI and sets the two module constants that
    ``rebasis.core.linear.default_rank`` reads. That is a liberty this file does
    not take anywhere else, and it is taken here for one reason: arXiv:2509.23471
    reports its Low-Rank Affine results at a fixed **r = 64**, and a reproduction
    that never ran the published setting would have an obvious hole in it.

    rebasis' own default is proportional — a quarter of the input dimension, so
    192 at d=768 — because M0 measured a fixed rank of 64 at d=384 collapsing to
    ARR 0.458 against 0.834 for centred Procrustes. The default is therefore the
    *more* generous configuration of the two, and a run that leaves this alone is
    not handicapping the paper's method.
    """
    if rank is None:
        yield
        return

    from rebasis.core import linear

    fraction, minimum = linear.DEFAULT_RANK_FRACTION, linear.MIN_RANK
    linear.DEFAULT_RANK_FRACTION, linear.MIN_RANK = 0.0, rank
    try:
        yield
    finally:
        linear.DEFAULT_RANK_FRACTION, linear.MIN_RANK = fraction, minimum


def fit_bridge(  # noqa: PLR0913 - the corpus, both encodings, and the fit budget
    corpus: Corpus,
    old: Encoded,
    new: Encoded,
    *,
    seed: int,
    device: str,
    fit_pairs: int = FIT_PAIRS,
    methods: Sequence[str] | None = None,
    with_csls: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Fit an adapter the way ``rebasis fit`` does, and load it as a ``Bridge``.

    Round-tripping through the ``.rbs`` file rather than using the in-memory
    adapter is deliberate: the serialised form is what a user actually holds,
    and a measurement of something else would be measuring something else.

    ``fit_pairs`` and ``methods`` are parameters only so that a published budget
    and a published parameterisation can be matched. The defaults are the shipped
    ones: ADR 10 measured six times the fit data buying one to two points of
    retention, and `auto`'s candidate list beating a hand-chosen method 15 times
    out of 15.

    ``with_csls`` decides only which candidate `auto` *selects*, never what is
    then measured. CSLS is a per-document search-time bias and the serialised
    adapter cannot carry one, so the ``bridged`` configuration is always the
    plain mapping — which is what a user gets from ``Bridge``.
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

    # The held-out share is fixed at a quarter of the fit budget, which is the
    # shipped 4000/1000 split and, at --fit-pairs 16000, the 80/20 split of the
    # 20,000 pairs arXiv:2509.23471 trains on.
    heldout = max(1, fit_pairs // 4)
    result, _ = probe_store(
        store,
        embedder,
        size=fit_pairs + heldout,
        heldout=heldout,
        k=10,
        seed=seed,
        device=device,
        methods=methods,
        with_csls=with_csls,
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
        # `probe`'s own T0 figure, under the name `probe` gives it — and **not**
        # the same quantity as this harness' retention, at either protocol. It
        # counts only the *nearest* neighbour as relevant (`SPARSE_RELEVANT` in
        # `rebasis.probe.groundtruth`; `docs/m0-findings.md` section 3 measured
        # the strict top-k variant 0.26 away from it), and it is measured with
        # the CSLS bias when `auto` chose one, which a serialised adapter cannot
        # carry. Recorded beside the harness' numbers so the two can be read
        # against each other rather than mistaken for each other.
        "arr_r10": round(result.best.arr, 4),
        "used_csls": bool(result.best.used_csls),
        "n_fit_pairs": result.n_fit_pairs,
        "n_params": result.best.n_params,
    }


def _run_dict(
    corpus: Corpus, indices: np.ndarray, scores: np.ndarray
) -> dict[str, dict[str, float]]:
    """Turn a top-k result into the mapping ranx wants.

    Keyed by :attr:`Corpus.score_ids`, which is the document ids except where
    copies exist. Several retrieved rows can then collapse onto one name, and the
    best score wins — the results are already in descending order, so keeping the
    first occurrence is keeping the best.
    """
    names = corpus.score_ids
    run: dict[str, dict[str, float]] = {}
    for query_id, row, score_row in zip(corpus.query_ids, indices, scores, strict=True):
        ranked: dict[str, float] = {}
        for position, score in zip(row, score_row, strict=True):
            ranked.setdefault(names[int(position)], float(score))
        run[query_id] = ranked
    return run


@dataclass(slots=True)
class Scored:
    """What ranx returned: the means the row quotes, and the arrays behind them."""

    #: ``aggregate[configuration][metric]`` — what every published table is built
    #: from.
    aggregate: dict[str, dict[str, float]]
    #: ``per_query[configuration][metric]`` — one value per entry of
    #: ``query_ids``, in that order.
    per_query: dict[str, dict[str, list[float]]]
    query_ids: list[str]


def score(
    corpus: Corpus,
    runs: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    cutoffs: Sequence[int],
) -> Scored:
    """Score every configuration with ranx, at every cut-off.

    The per-query arrays come out of the same call rather than a second one:
    ``ranx.evaluate`` computes them either way and writes them into the ``Run``
    it was handed, keyed by query id. Reading them back by id rather than by
    position is deliberate — ranx sorts a run internally, and an array lined up
    against the wrong ids is a mistake nothing downstream could detect.
    """
    from ranx import Qrels, Run, evaluate

    qrels = Qrels(corpus.qrels)
    metrics = [f"{name}@{k}" for k in cutoffs for name in ("ndcg", "recall", "mrr")]
    query_ids = list(corpus.query_ids)

    aggregate: dict[str, dict[str, float]] = {}
    per_query: dict[str, dict[str, list[float]]] = {}
    for label, (indices, values) in runs.items():
        run = Run(_run_dict(corpus, indices, values))
        measured = evaluate(qrels, run, metrics)
        aggregate[label] = {metric: round(float(measured[metric]), 4) for metric in metrics}
        per_query[label] = {
            metric: [round(float(run.scores[metric][query_id]), 6) for query_id in query_ids]
            for metric in metrics
        }
    return Scored(aggregate=aggregate, per_query=per_query, query_ids=query_ids)


def protocol_tag(protocol: str, truth_k: int, replicate: int = 1) -> str:
    """The protocol, plus the ground-truth depth and any replication.

    The depth is part of a run's identity and not a detail of it. rebasis' own
    T0 counts only the **nearest** neighbour as relevant — ``SPARSE_RELEVANT``
    in ``rebasis.probe.groundtruth``, and ``docs/m0-findings.md`` section 3 for
    why — while arXiv:2509.23471 counts the whole top-10 set. Those are two
    different measurements, they differ by 0.26 on average, and a resume key
    that could not tell them apart would let one stand in for the other.

    Replication is in the tag for the same reason and a sharper one: a
    replicated corpus is a different collection, and a row that did not say so
    would be a claim about AG-News that was measured on eight copies of it.
    """
    if protocol == "t1-judged":
        return protocol
    tag = f"{protocol}@{truth_k}"
    return tag if replicate == 1 else f"{tag}x{replicate}"


def run_key(tag: str, corpus: str, old_model: str, new_model: str, seed: int) -> str:
    """A short, stable name for one run, shared by its row and its sidecar.

    Derived from the run's identity rather than drawn at random so that a
    re-measured run overwrites its own per-query file instead of orphaning it,
    and so that a row and a sidecar can be matched up after the fact by anyone
    holding the same five fields.
    """
    material = "\x1f".join([tag, corpus, old_model, new_model, str(seed)])
    return hashlib.blake2s(material.encode("utf-8"), digest_size=8).hexdigest()


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
    protocol: str = "t1-judged",
    fit_pairs: int = FIT_PAIRS,
    n_proxies: int = T0_PROXIES,
    truth_k: int = T0_TRUTH_K,
    methods: Sequence[str] | None = None,
    with_csls: bool = True,
    replicate: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One row of the band: every configuration over one corpus and model pair.

    Returns the row and its per-query sidecar payload. The two are written by
    the caller so that a row never reaches the output file describing a sidecar
    that was not written.
    """
    from rebasis.compute import resolve_device, top_k_search, using_device

    started = time.perf_counter()
    # Encoding is always of the *whole* corpus, whichever protocol is running.
    # The document-proxy split then slices those arrays: a cache keyed on a
    # corpus name would otherwise hold two different document sets under one
    # file name, and the second protocol to run would read the first's vectors.
    shared = {"corpus": corpus, "cache_dir": cache_dir, "device": device}
    old = encode_corpus(model_id=old_model, encoder_cache=encoder_cache, **shared)
    new = encode_corpus(model_id=new_model, encoder_cache=encoder_cache, **shared)

    depth = max(cutoffs)
    if protocol == "t0-knn":
        corpus, old, new = document_proxy_view(
            corpus,
            old,
            new,
            n_proxies=n_proxies,
            truth_k=truth_k,
            depth=depth,
            seed=seed,
            device=device,
            replicate=replicate,
        )
    elif not corpus.query_ids:
        msg = (
            f"{corpus.name} has no judged queries, so it cannot be run under "
            f"{protocol}; it is a --protocol t0-knn collection"
        )
        raise RuntimeError(msg)
    elif protocol == "t0-knn-real-queries":
        corpus = knn_truth_view(corpus, new, truth_k=truth_k, depth=depth, device=device)

    bridge, fit_summary = fit_bridge(
        corpus,
        old,
        new,
        seed=seed,
        device=device,
        fit_pairs=fit_pairs,
        methods=methods,
        with_csls=with_csls,
    )
    mapped = bridge.to_index_space(new.queries)

    # The pre-fit signal, measured on the same corpus as everything else so the
    # bound and the retention it bounds can be read against each other across
    # the whole ladder. One Gram-matrix difference; no fit.
    from rebasis.core import geometry_bound

    geometry = geometry_bound(new.documents, old.documents, seed=seed)

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
        else:
            # The same configuration under the only convention that defines it
            # when the dimensions do not agree: zero-pad the shorter space up to
            # the longer. Zero-padding leaves every inner product unchanged, so
            # this measures the swap rather than the padding — and it is the
            # configuration a published evaluation of a 384-d to 768-d upgrade
            # must have measured under some convention, without saying which.
            #
            # It is a convention of this harness and not an operation a user
            # has. Where the new model is wider the padding lands on the indexed
            # documents, and widening an index means rewriting it. Nor does
            # `IdentityAdapter` licence it: that pads only when the new model is
            # narrower and truncates otherwise. Read the row as "what a number
            # would be if the configuration existed", never as an option.
            width = max(new.queries.shape[1], old.documents.shape[1])
            runs["naive_swap_padded"] = top_k_search(
                _pad_to(new.queries, width),
                _pad_to(old.documents, width),
                k=depth,
                self_mask=self_mask,
            )

        # What the old space could do if the adapter were perfect. Under a kNN
        # ground truth the answer is knowable, and it bounds every row above it.
        if corpus.protocol != "t1-judged":
            runs["ceiling_old_space"] = top_k_search(
                _target_centroids(corpus, old), old.documents, k=depth, self_mask=self_mask
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
    tag = protocol_tag(protocol, truth_k, replicate)
    key = run_key(tag, corpus.name, old_model, new_model, seed)

    row = {
        "protocol": protocol,
        "protocol_tag": tag,
        "run_key": key,
        "corpus": corpus.name,
        "old_model": old_model,
        "new_model": new_model,
        "old_dim": int(old.documents.shape[1]),
        "new_dim": int(new.documents.shape[1]),
        "n_documents": len(corpus.doc_ids),
        "n_queries": len(corpus.query_ids),
        "self_removal": corpus.self_mask is not None,
        "truth": corpus.truth,
        "cutoffs": list(cutoffs),
        "cascade": list(cascade),
        "fit": fit_summary,
        "geometry": geometry.to_dict(),
        "scores": scored.aggregate,
        "seed": seed,
        "duration_seconds": round(time.perf_counter() - started, 1),
    }
    sidecar = {
        "run_key": key,
        "protocol": protocol,
        "protocol_tag": tag,
        "corpus": corpus.name,
        "old_model": old_model,
        "new_model": new_model,
        "seed": seed,
        "cutoffs": list(cutoffs),
        "query_ids": scored.query_ids,
        "scores": scored.per_query,
    }
    return row, sidecar


def _target_centroids(corpus: Corpus, old: Encoded) -> FloatArray:
    """A cheating query — one built from the answer — expressed in the old space.

    **Nothing a user can run produces this number.** It is not an adapter, not a
    method and not a result: it is the score a query would get if it already knew
    which documents it was supposed to retrieve, and it is here only to bound the
    rows above it. Quoted on its own it would be a claim about a system nobody
    can build.

    What it is for is separating "the adapter is not good enough" from "the old
    space cannot express this", and it can only be built where the ground truth
    is a set of documents rather than a human judgement.

    For each query, take the ground-truth documents' own *old-model* vectors and
    use their normalised mean as the query. Among unit vectors that is exactly
    the one maximising the summed similarity to the target set — the natural
    relaxation of "put as many of them as possible in the top ten" — so what it
    scores is a close estimate of the best any query-side map could achieve
    against that index. It is not a proof of the maximum, and it is not
    available to any real adapter: it is built from the answer.

    Read it as the ceiling the row above it is trying to reach. A ``bridged``
    score far below it says the adapter left something on the table; a
    ``ceiling_old_space`` far below 1.0 says the old space does not hold the
    new model's neighbourhoods at all, and no adapter of any family will find
    them there. ADR 10 reaches the same place from the fit side.
    """
    from rebasis.compute import l2_normalize

    # Keyed on the scoring name rather than the row id, so that a target which
    # exists as several copies contributes once. Averaging eight identical
    # vectors would weight that document eight times over and quietly measure a
    # different query.
    position: dict[str, int] = {}
    for index, name in enumerate(corpus.score_ids):
        position.setdefault(name, index)
    centroids = np.zeros((len(corpus.query_ids), old.documents.shape[1]), dtype=np.float32)
    for row, query_id in enumerate(corpus.query_ids):
        targets = [position[name] for name in corpus.qrels[query_id] if name in position]
        if targets:
            centroids[row] = old.documents[targets].mean(axis=0)
    return l2_normalize(centroids)


def _pad_to(vectors: FloatArray, width: int) -> FloatArray:
    """Zero-pad ``vectors`` out to ``width`` columns, or return them unchanged.

    Zero-padding is the one dimension-matching convention that changes no inner
    product: the extra coordinates contribute nothing to any dot product and
    nothing to any norm. Anything else — truncation, a random projection —
    would make the configuration a measurement of that choice.
    """
    if vectors.shape[1] >= width:
        return vectors
    padded = np.zeros((vectors.shape[0], width), dtype=np.float32)
    padded[:, : vectors.shape[1]] = vectors
    return padded


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


def already_done(out: Path) -> set[tuple[str, str, str, str]]:
    """Rows the output file already holds, so a re-run resumes rather than repeats.

    The harness is hours of GPU time over a ladder; an interrupted run that had
    to start again would mean nobody ever finishes one.

    The protocol — and, under a kNN ground truth, its depth — is part of the key.
    Without it, a file holding a corpus' T1 rows would tell a T0 run that corpus
    was already measured, and the resumed run would skip precisely the rows it
    was started to produce. Rows written before the flag existed carry no
    protocol and are read as ``t1-judged``, which is what they are.
    """
    if not out.exists():
        return set()
    done: set[tuple[str, str, str, str]] = set()
    for line in out.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        with contextlib.suppress(json.JSONDecodeError, KeyError):
            row = json.loads(line)
            done.add(
                (
                    row.get("protocol_tag", row.get("protocol", "t1-judged")),
                    row["corpus"],
                    row["old_model"],
                    row["new_model"],
                )
            )
    return done


def write_sidecar(directory: Path, sidecar: dict[str, Any]) -> str:
    """Write one run's per-query scores, and return the path the row should carry.

    A separate file because the arrays are the wrong shape for the row: a
    thousand queries across six configurations and nine metrics is fifty
    thousand numbers, and ``reports/band/*.jsonl`` is read by hand. The row keeps
    the means it always kept and points at the rest.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{sidecar['run_key']}.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(sidecar), encoding="utf-8")
    tmp.replace(path)
    return f"{directory.name}/{path.name}"


def build_parser() -> argparse.ArgumentParser:
    """Every knob, in one place, so that ``main`` is the run and not the options."""
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
    parser.add_argument(
        "--protocol",
        default="t1-judged",
        help=(
            "Comma-separated, from: t1-judged (real queries, human judgements); "
            "t0-knn-real-queries (real queries, the new model's exact kNN as the "
            "ground truth); t0-knn (held-out documents as queries, the same kNN "
            "ground truth). Default is t1-judged, which is every existing row"
        ),
    )
    parser.add_argument("--out", type=Path, default=Path("reports/band/rows.jsonl"))
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / "band-cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument(
        "--fit-pairs",
        type=int,
        default=FIT_PAIRS,
        help=f"Adapter fit budget; a quarter as much again is held out (default {FIT_PAIRS})",
    )
    parser.add_argument(
        "--t0-queries",
        type=int,
        default=T0_PROXIES,
        help=f"Documents held out as query proxies under t0-knn (default {T0_PROXIES})",
    )
    parser.add_argument(
        "--t0-truth-k",
        type=int,
        default=T0_TRUTH_K,
        help=f"Depth of the t0-knn ground truth (default {T0_TRUTH_K})",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help=(
            "Comma-separated adapter methods to fit instead of `auto`'s list. "
            "The list exists so a published parameterisation can be matched: "
            "procrustes is Orthogonal Procrustes, low_rank_affine is Low-Rank "
            "Affine, residual_mlp is the Residual MLP, procrustes_centered+dsm "
            "adds a diagonal scaling matrix"
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help=(
            "Pin low_rank_affine to this rank. rebasis defaults to a quarter of "
            "the input dimension; arXiv:2509.23471 reports r=64"
        ),
    )
    parser.add_argument(
        "--replicate",
        type=int,
        default=1,
        help=(
            "Copy every indexed document this many times under t0-knn, to test a "
            "corpus stated as larger than the split it was drawn from. The copies "
            "are made after the query proxies are removed, so no proxy has a twin"
        ),
    )
    parser.add_argument(
        "--no-csls",
        action="store_true",
        help=(
            "Select on the plain ARR rather than the better of plain and CSLS. "
            "Only affects which candidate wins: a serialised adapter cannot "
            "carry a search-time bias, so the bridged rows never have one"
        ),
    )
    parser.add_argument(
        "--no-per-query",
        action="store_true",
        help="Skip the per-query sidecar files. They are what a significance test needs",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-measure rows the output already holds"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cutoffs = [int(part) for part in args.k.split(",") if part.strip()]
    cascade = [int(part) for part in args.cascade.split(",") if part.strip()]
    protocols = [part.strip() for part in args.protocol.split(",") if part.strip()]
    unknown = [name for name in protocols if name not in PROTOCOLS]
    if unknown:
        parser.error(f"unknown protocol(s) {unknown}; choose from {list(PROTOCOLS)}")
    methods = (
        None
        if args.methods is None
        else ([part.strip() for part in args.methods.split(",") if part.strip()] or None)
    )
    datasets = resolve_corpora(args.corpus or ["heldout"])
    rungs = LADDERS[args.ladder]
    done = set() if args.force else already_done(args.out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    sidecar_dir = args.out.parent / "per-query"

    planned = [
        (dataset, old, new, protocol)
        for dataset in datasets
        for old, new in rungs
        for protocol in protocols
        if args.force
        or (protocol_tag(protocol, args.t0_truth_k, args.replicate), dataset, old, new) not in done
    ]
    print(
        f"{len(planned)} runs to measure ({len(datasets)} corpora x {len(rungs)} rungs "
        f"x {len(protocols)} protocol(s), {len(done)} already done)",
        flush=True,
    )

    encoder_cache: dict[str, Any] = {}
    for dataset in datasets:
        pending = [(o, n, p) for d, o, n, p in planned if d == dataset]
        if not pending:
            continue
        print(f"\n=== {dataset} ===", flush=True)
        corpus = load_corpus(dataset, limit=args.limit_docs)
        print(
            f"  {len(corpus.doc_ids):,} documents, {len(corpus.query_ids):,} judged queries",
            flush=True,
        )
        for old_model, new_model, protocol in pending:
            print(f"  -- [{protocol}] {old_model} -> {new_model}", flush=True)
            with fixed_low_rank(args.rank):
                row, sidecar = measure(
                    corpus,
                    old_model,
                    new_model,
                    cache_dir=args.cache_dir,
                    cutoffs=cutoffs,
                    device=args.device,
                    seed=args.seed,
                    encoder_cache=encoder_cache,
                    cascade=cascade,
                    protocol=protocol,
                    fit_pairs=args.fit_pairs,
                    n_proxies=args.t0_queries,
                    truth_k=args.t0_truth_k,
                    methods=methods,
                    with_csls=not args.no_csls,
                    replicate=args.replicate,
                )
            # Recorded rather than left to be inferred from the parameter count.
            # A forced rank is not in the resume key, so a file holding both
            # would otherwise be two configurations under one name.
            if args.rank is not None:
                row["fit"]["forced_rank"] = args.rank
            # The sidecar lands first. A row naming a file that is not there is
            # worse than a file nothing names: the second is tidied up, the
            # first is discovered by whatever reads the row next.
            if not args.no_per_query:
                row["per_query"] = write_sidecar(sidecar_dir, sidecar)
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
