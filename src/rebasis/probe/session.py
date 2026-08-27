"""Store and embedder plumbing for the probe pipeline.

:func:`rebasis.probe.run_probe` takes vectors. Everything between "a store URI
and two model names" and "vectors" lives here: draw a sample, read its vectors
out of the index, re-embed the same documents with the candidate model.

Two constraints shape the code.

**Memory stays** ``O(pool × d)``, **never** ``O(N × d)``. Stratification needs
vectors to cluster, but the sampler clusters at most ``min(50k, N)`` of them — so
the corpus is reservoir-sampled down to that pool in a single streaming pass, and
the pool is what gets clustered. On a five-million-chunk index the pool is still
50,000 vectors.

**The store is never written to.** Every call here reads.

**Embeddings are reused when the caller asks for it.** ``cache_dir`` turns on
:mod:`rebasis.storage.embedding_cache`, which memoises the candidate model's
output so that the second probe of a corpus does not embed it a second time —
the most visible everyday cost in the tool. It is off unless a directory is
given, because ``probe_store`` is importable and a library that starts writing
into someone's project directory unasked has taken a liberty; ``rebasis probe``
passes one, because it has already created ``.rebasis/`` for the audit trail.

The cache changes what a run *costs* and must not change what it *reports*, and
there is exactly one number where those meet: the reindex estimate is
extrapolated from the rate this run measured, so it is measured over the
documents actually sent to the model and omitted entirely when the cache
answered all of them. An estimate of zero seconds would be worse than no
estimate.
"""

from __future__ import annotations

import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from rebasis.compute.arrays import l2_normalize
from rebasis.errors import InsufficientSamples, StoreUnsupported
from rebasis.observability import (
    DROPPED_NO_TEXT,
    DROPPED_UNJUDGED,
    Events,
    Spans,
    get_logger,
    instrument,
    measure_resources,
    should_span_batch,
    span,
)
from rebasis.probe.groundtruth import build_tier0, build_tier1, build_tier2
from rebasis.probe.metrics import estimate_reindex_cost
from rebasis.probe.runner import ProbeResult, run_probe
from rebasis.sample import SampleResult, draw_sample, split_disjoint
from rebasis.storage.embedding_cache import open_cached_embedder
from rebasis.store.base import require_capability

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from rebasis.audit import AuditWriter
    from rebasis.probe.groundtruth import GroundTruth
    from rebasis.storage.embedding_cache import CachedEmbedder
    from rebasis.store.base import VectorStore
    from rebasis.types import Embedder, EncodingProfile, FloatArray


def _ignore_stage(_text: str) -> None:
    """The default ``on_stage``: a probe with nobody watching."""


__all__ = ["CorpusSample", "QueryLog", "draw_corpus_sample", "probe_store", "record_decision"]

log = get_logger(__name__)

#: Ceiling on the vectors held in memory for clustering. Above this the
#: corpus is reservoir-sampled down; the sample itself is drawn from the pool.
CLUSTER_POOL_MAX = 50_000

#: Below this a synthesised set is too small to estimate an upgrade from: the
#: confidence interval would be wider than the decision it is meant to inform.
MIN_SYNTHESISED_QUERIES = 50

#: How many texts to hand the embedder at a time. The embedder batches
#: internally too; this bounds the list we build, not the forward pass.
ENCODE_CHUNK = 256


@dataclass(slots=True)
class CorpusSample:
    """A sample of the index, with its own vectors and its text.

    ``query_positions`` and ``fit_positions`` index into this sample, not into
    the corpus, and are disjoint by construction — a query the adapter was fitted
    on would make every ARR number meaningless.
    """

    ids: list[str]
    texts: list[str]
    old_vectors: FloatArray
    query_positions: np.ndarray
    fit_positions: np.ndarray
    n_total: int
    strategy: str
    seed: int
    pool_size: int = 0
    #: Whether the query proxies were actually drawn weighted. False when no
    #: access log was given *and* when one was given that named nothing here.
    access_weighted: bool = False
    #: Cluster of each record, when the sample was stratified. This is what
    #: makes ``tail_arr`` computable — without it, heterogeneous drift is
    #: invisible, and one global adapter stalling on part of the corpus while
    #: the overall figure looks healthy goes unreported.
    cluster_labels: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.ids)

    def clusters_of(self, positions: np.ndarray) -> np.ndarray | None:
        """Cluster labels for a subset of this sample, if there are any."""
        if self.cluster_labels is None:
            return None
        subset: np.ndarray = self.cluster_labels[positions]
        return subset


@dataclass(slots=True)
class QueryLog:
    """A real query log with relevance judgements — the T1 tier.

    ``qrels[i]`` holds the record ids judged relevant to ``queries[i]``. Ids that
    are not in the drawn sample are dropped, and a query left with nothing
    relevant is dropped with it: scoring it would count a guaranteed miss
    against every candidate equally and drag every ARR toward zero.
    """

    queries: list[str]
    qrels: list[set[str]]
    metadata: dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.queries)

    @property
    def judged(self) -> bool:
        """Whether any query names a document that was judged relevant.

        A log exported from a search box has no judgements, and that is the
        usual case rather than a broken file. It still measures retention; it
        just cannot measure the upgrade.
        """
        return any(self.qrels)


def draw_corpus_sample(  # noqa: PLR0913 - one argument per sampling input
    store: VectorStore,
    *,
    size: int = 10_000,
    heldout: int = 1_000,
    strategy: str = "stratified",
    seed: int = 0,
    batch_size: int = 1_000,
    need_text: bool = True,
    access_counts: Mapping[str, float] | None = None,
) -> CorpusSample:
    """Draw a sample of the index and read back its vectors and text.

    ``access_counts`` maps record id to how often it was read, and weights which
    sampled records become **query proxies** — leaving the sample itself uniform,
    for the reason :func:`~rebasis.sample.split_disjoint` gives. Records the log
    does not mention count as read once.

    Raises:
        StoreUnsupported: When the store cannot return text and text is needed —
            without it there is nothing to re-embed with the new model.
        InsufficientSamples: When too few usable records came back.
    """
    require_capability(store, "can_read_vectors", operation="probe")
    if need_text:
        require_capability(store, "can_read_text", operation="probe")

    with span(Spans.SAMPLE_DRAW, {"strategy": strategy, "count": size}):
        return _draw(
            store,
            size=size,
            heldout=heldout,
            strategy=strategy,
            seed=seed,
            batch_size=batch_size,
            need_text=need_text,
            access_counts=access_counts,
        )


def _draw(  # noqa: PLR0913 - mirrors its caller's signature
    store: VectorStore,
    *,
    size: int,
    heldout: int,
    strategy: str,
    seed: int,
    batch_size: int,
    need_text: bool,
    access_counts: Mapping[str, float] | None = None,
) -> CorpusSample:
    n_total = store.count()
    pool_ids, pool_vectors = _reservoir(store, n_total=n_total, seed=seed, batch_size=batch_size)
    if not pool_ids:
        raise InsufficientSamples(
            "The collection returned no records with vectors.",
            hint="Check the collection name in the store URI; `rebasis doctor` lists them.",
            context={"count": 0},
        )

    drawn = draw_sample(
        pool_vectors, len(pool_ids), min(size, len(pool_ids)), strategy=strategy, seed=seed
    )
    positions = drawn.indices
    ids = [pool_ids[int(p)] for p in positions]
    vectors = l2_normalize(pool_vectors[positions])

    labels = drawn.cluster_labels[positions] if drawn.cluster_labels is not None else None

    texts = _fetch_texts(store, ids, batch_size=batch_size) if need_text else ["" for _ in ids]
    if need_text:
        ids, texts, vectors, labels = _drop_empty_texts(ids, texts, vectors, labels)

    # Weights go on the **query split**, not on the sample. The sample does two
    # jobs — it is the mini-index every measurement runs against, and it is the
    # pool the queries come out of — and weighting it fills the mini-index with
    # frequently-read documents, which changes the *distractors*: a property of
    # the index rather than of the questions asked of it. Measured over 36
    # cells, weighting the split leaves the estimate half as far from the
    # whole-corpus quantity as weighting the sample does (+0.025 against
    # +0.051). `docs/access-weighting.md` has both.
    access = _access_vector(ids, access_counts)
    query_positions, fit_positions = split_disjoint(
        _positional(len(ids), drawn.strategy, seed), heldout, seed=seed, weights=access
    )

    log.info(
        Events.PROBE_SAMPLE_DRAWN,
        count=len(ids),
        strategy=drawn.strategy,
        seed=seed,
        dim=int(vectors.shape[1]),
    )
    return CorpusSample(
        ids=ids,
        texts=texts,
        old_vectors=vectors,
        query_positions=query_positions,
        fit_positions=fit_positions,
        n_total=n_total,
        strategy=drawn.strategy,
        seed=seed,
        pool_size=len(pool_ids),
        access_weighted=access is not None,
        cluster_labels=labels,
    )


def probe_store(  # noqa: PLR0913 - one argument per pipeline input
    store: VectorStore,
    new_embedder: Embedder,
    *,
    old_embedder: Embedder | None = None,
    sample: CorpusSample | None = None,
    query_log: QueryLog | None = None,
    size: int = 10_000,
    heldout: int = 1_000,
    strategy: str = "stratified",
    k: int = 10,
    seed: int = 0,
    methods: Sequence[str] | None = None,
    with_csls: bool = True,
    batch_size: int = 1_000,
    synth_queries: str | None = None,
    audit: AuditWriter | None = None,
    store_uri: str = "",
    old_model: str = "",
    device: str = "cpu",
    cache_dir: Path | str | None = None,
    fit_migration: bool = False,
    access_counts: Mapping[str, float] | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> tuple[ProbeResult, CorpusSample]:
    """Probe a live store: sample it, re-embed it, and decide.

    ``old_embedder`` is optional and only used at T1. At T0 the query proxies are
    documents that are already in the index, so their old-model vectors are read
    rather than recomputed — which is both faster and exactly what the index
    holds. With a real query log there is no such shortcut: the queries are text
    that was never indexed, and answering "how well does the current model do?"
    means encoding them with it.

    ``cache_dir`` names a directory in which embeddings this run computes are
    kept for the next one, one file per model profile. ``None``, the default,
    means nothing is cached and nothing is written; ``rebasis probe`` passes
    :func:`~rebasis.storage.default_embedding_cache_dir`. See the module
    docstring for why the default is off here and on there.

    ``access_counts`` weights which sampled records become query proxies, so
    ARR describes the questions people actually send rather than a uniform draw
    over the corpus. It changes what is being estimated and says so in the
    result; `docs/access-weighting.md` has what it is worth and what it costs.

    ``fit_migration`` adds a second fit in the opposite direction — the map
    `rebasis migrate` rewrites an index with — scored on what a *completed*
    migration would deliver rather than on what a bridged query retrieves. Off
    by default because the two are different questions and most runs are asking
    the first; see :mod:`rebasis.probe.migration`.
    """
    from rebasis.compute import blas_info, measurement_precision, using_device

    with (
        span(Spans.PROBE, {"seed": seed, "k": k}),
        # Entered once for the run, so the ground-truth kNN reaches the device
        # without a `device=` argument on every function between here and the
        # matmul. TF32 stays off because this output feeds a decision, and 10-bit
        # mantissas are not a precision a threshold should depend on.
        using_device(device),
        measurement_precision(),
        measure_resources(device=device, blas_threads=blas_info()[1]) as usage,
        # Closed here because they were opened here. A clean close checkpoints
        # each cache's write-ahead log, which leaves one file per model where
        # `gc` expects one.
        ExitStack() as caches,
    ):
        new_encoder, new_cache = _with_cache(new_embedder, cache_dir, caches)
        old_encoder: Embedder | None = None
        if old_embedder is not None:
            old_encoder, _ = _with_cache(old_embedder, cache_dir, caches)

        # Announced rather than left to one static spinner. These four stages
        # have very different durations — embedding dominates — and a display
        # that never changes cannot tell a slow stage from a wedged one.
        stage = on_stage if on_stage is not None else _ignore_stage

        stage("Sampling the index")
        corpus = sample or draw_corpus_sample(
            store,
            size=size,
            heldout=heldout,
            strategy=strategy,
            seed=seed,
            batch_size=batch_size,
            access_counts=access_counts,
        )

        # Timed separately from everything else because it is the one step whose
        # rate extrapolates: a full reindex is this, over the whole corpus.
        stage(f"Embedding {len(corpus):,} documents with the new model")
        mark = _encode_mark(new_cache)
        embed_started = time.perf_counter()
        new_doc_vectors = _encode(new_encoder, corpus.texts, kind="document")
        embed_seconds = time.perf_counter() - embed_started
        embedded, model_seconds = _encode_since(new_cache, mark, len(corpus), embed_seconds)

        # For an asymmetric model, encode the same texts a second time the way a
        # query is encoded. That gives `auto` a second candidate list to measure
        # the query-specific strategy against the shared one, instead of picking
        # one on principle. Symmetric models skip it — the two encodings would be
        # identical and the extra pass would buy nothing.
        new_doc_vectors_as_queries = None
        if not new_embedder.profile.symmetric:
            new_doc_vectors_as_queries = _encode(new_encoder, corpus.texts, kind="query")

        stage("Building the ground truth")
        if query_log is not None:
            ground_truth, old_q, new_q, query_clusters = _tier1(
                corpus, query_log, new_encoder, old_encoder, new_doc_vectors, k=k
            )
            # Not for an unjudged log: there the oracle *is* the ground truth,
            # so "how does the old model score against it" measures drift, not
            # improvement, and a number that looks like a gain would be one.
            upgrade_gain = (
                None
                if ground_truth.tier == "t1u"
                else _upgrade_gain(ground_truth, old_q, corpus.old_vectors, k)
            )
        elif synth_queries is not None:
            ground_truth, old_q, new_q, query_clusters = _tier2(
                corpus,
                new_encoder,
                old_encoder,
                new_doc_vectors,
                k=k,
                strategy=synth_queries,
            )
            # Measurable here, unlike T0: the judgement comes from the corpus'
            # own structure rather than from either model's output.
            upgrade_gain = (
                None
                if ground_truth.tier == "t1u"
                else _upgrade_gain(ground_truth, old_q, corpus.old_vectors, k)
            )
        else:
            ground_truth, old_q, new_q, query_clusters = _tier0(
                corpus, new_doc_vectors, new_doc_vectors_as_queries, k=k
            )
            # Not measurable at T0: the ground truth *is* the new model's output,
            # so comparing the new model against it would always score 1.0.
            upgrade_gain = None

        stage("Fitting and scoring the candidate adapters")
        result = run_probe(
            old_doc_vectors=corpus.old_vectors,
            new_doc_vectors=new_doc_vectors,
            fit_indices=corpus.fit_positions,
            ground_truth=ground_truth,
            old_query_vectors=old_q,
            new_query_vectors=new_q,
            k=k,
            seed=seed,
            methods=methods,
            with_csls=with_csls,
            upgrade_gain=upgrade_gain,
            new_doc_vectors_as_queries=new_doc_vectors_as_queries,
            query_clusters=query_clusters,
            # Omitted, not zeroed, when the cache answered the whole corpus:
            # there is no rate to extrapolate from a pass that embedded nothing,
            # and "a full reindex takes no time" is a claim this run did not
            # measure. The same rule as `watts` two lines down — energy is
            # measured or omitted, never estimated, because a figure derived
            # from a nominal TDP would read as a measurement of something nobody
            # asked the hardware.
            reindex_cost=(
                estimate_reindex_cost(
                    corpus.n_total,
                    seconds_per_document=model_seconds / embedded,
                    watts=None,
                )
                if embedded
                else None
            ),
        )

    # Outside the measured block: the summary describes the work, and writing
    # it down is not part of the work.
    result.resources = usage.summary.to_dict()
    # Read off the sample rather than off the argument: an access log that named
    # nothing in this sample weighted nothing, and reporting the flag the user
    # passed instead of the draw that happened would claim a measurement that
    # was not taken.
    result.access_weighted = corpus.access_weighted

    if fit_migration:
        # After the timed block on purpose. This is a second fit over the same
        # pairs, and folding it into the resource summary would make a `probe`
        # that asked for it look more expensive than the one a user compares it
        # against. The reindex-cost rate would be wrong for the same reason.
        from rebasis.probe.migration import fit_migration_adapter

        result.migration = fit_migration_adapter(
            old_doc_vectors=corpus.old_vectors,
            new_doc_vectors=new_doc_vectors,
            new_query_vectors=new_q,
            ground_truth=ground_truth,
            fit_indices=corpus.fit_positions,
            k=k,
            methods=methods,
        )

    if audit is not None:
        record_decision(
            audit,
            result,
            corpus,
            store_uri=store_uri,
            old_model=old_model or (old_embedder.profile.model_id if old_embedder else ""),
            new_profile=new_embedder.profile,
        )
    return result, corpus


def record_decision(  # noqa: PLR0913 - the record needs every input it names
    audit: AuditWriter,
    result: ProbeResult,
    corpus: CorpusSample,
    *,
    store_uri: str,
    old_model: str,
    new_profile: EncodingProfile,
) -> None:
    """Write the decision to the audit trail.

    The decision is made here, so the record is written here: under the layer
    contract ``audit`` is only ever called from above, which is what keeps
    ``core`` free of side effects. The writer is passed in rather than
    constructed so this module never reaches down into ``manifest``.

    The inputs are everything needed to reproduce the decision and nothing else:
    model ids, profile fingerprints, sampling strategy and seed, sizes,
    thresholds. **No document text, no ids, no vectors** — text can be recovered
    from an embedding, so a vector is as sensitive as the text it came from.
    """
    from rebasis.probe.decision import BORDERLINE_BAND, DECISION_BANDS
    from rebasis.probe.metrics import METRIC_VERSION

    audit.write(
        Events.PROBE_DECISION_MADE,
        inputs={
            "old_model": old_model,
            "new_model": new_profile.model_id,
            "new_profile_fingerprint": new_profile.fingerprint(),
            # The whole profile, not just its fingerprint. A fingerprint proves
            # two profiles are the same; it cannot rebuild one. `audit replay`
            # has to reconstruct the exact encoding — prefixes included — and a
            # model that was described by flags is not in any table to look up.
            "new_profile": {
                "model_id": new_profile.model_id,
                "dim": new_profile.dim,
                "query_prefix": new_profile.query_prefix,
                "document_prefix": new_profile.document_prefix,
                "normalize": new_profile.normalize,
                "matryoshka_dim": new_profile.matryoshka_dim,
                "pooling": new_profile.pooling,
            },
            "strategy": corpus.strategy,
            "seed": corpus.seed,
            "count": len(corpus),
            "n_total": corpus.n_total,
            "n_queries": result.n_queries,
            "n_fit_pairs": result.n_fit_pairs,
            "k": result.k,
            "tier": result.ground_truth_tier,
            "bands": {name: threshold for threshold, name in DECISION_BANDS},
            "borderline_band": BORDERLINE_BAND,
            "metric_version": METRIC_VERSION,
        },
        outputs=result.to_dict(),
        subject=store_uri or None,
    )


# ── the embedding cache ──────────────────────────────────────────────────────


def _with_cache(
    embedder: Embedder, cache_dir: Path | str | None, caches: ExitStack
) -> tuple[Embedder, CachedEmbedder | None]:
    """Wrap an embedder so it reuses vectors it has already computed.

    Returns the embedder to encode with and the wrapper behind it — ``None``
    whenever nothing is being cached, which is either because no directory was
    given or because the user set ``REBASIS_EMBED_CACHE=0``. Both are needed:
    one to encode with, and one to ask afterwards how much of the work was real.
    """
    if cache_dir is None:
        return embedder, None
    cached = open_cached_embedder(embedder, directory=cache_dir)
    if cached is None:
        return embedder, None
    caches.callback(cached.close)
    return cached, cached


def _encode_mark(cached: CachedEmbedder | None) -> tuple[int, float]:
    """Snapshot the counters, so one embedding pass can be measured on its own."""
    return (0, 0.0) if cached is None else (cached.stats.encoded, cached.stats.encode_seconds)


def _encode_since(
    cached: CachedEmbedder | None, mark: tuple[int, float], documents: int, seconds: float
) -> tuple[int, float]:
    """How much of one embedding pass the model actually did.

    Without a cache that is the whole pass, and the two numbers are exactly the
    ones this has always reported. With one it is the misses, and the time spent
    inside the model rather than the wall time of the pass — the difference is
    the lookups that avoided the work, and charging those to the model would
    inflate the reindex estimate by however long the cache took to answer.
    """
    if cached is None:
        return documents, seconds
    return cached.stats.encoded - mark[0], cached.stats.encode_seconds - mark[1]


# ── sampling internals ───────────────────────────────────────────────────────


def _reservoir(
    store: VectorStore, *, n_total: int, seed: int, batch_size: int
) -> tuple[list[str], FloatArray]:
    """Stream the collection, keeping at most ``CLUSTER_POOL_MAX`` vectors.

    Reservoir sampling rather than "read the first 50,000": insertion order is
    rarely random, and on a vault that grew by topic the first 50,000 chunks are
    a biased slice of the corpus. It also needs no reliable ``count()``, which
    the bridge backends cannot always give.
    """
    rng = np.random.default_rng(seed)
    capacity = min(n_total, CLUSTER_POOL_MAX) if n_total > 0 else CLUSTER_POOL_MAX

    ids: list[str] = []
    buffer: list[FloatArray] = []
    seen = 0
    started = time.perf_counter()

    for record in store.iter_records(with_vectors=True, with_text=False, batch_size=batch_size):
        if record.vector is None:
            continue
        seen += 1
        if len(ids) < capacity:
            ids.append(record.id)
            buffer.append(np.asarray(record.vector, dtype=np.float32))
            continue
        # Classic Algorithm R: the nth record replaces a uniformly chosen slot
        # with probability capacity/n, which leaves every record equally likely.
        slot = int(rng.integers(0, seen))
        if slot < capacity:
            ids[slot] = record.id
            buffer[slot] = np.asarray(record.vector, dtype=np.float32)

    if not buffer:
        return [], np.empty((0, 0), dtype=np.float32)

    log.debug(
        Events.PROBE_SAMPLE_DRAWN,
        count=len(ids),
        duration_ms=round((time.perf_counter() - started) * 1000, 1),
    )
    return ids, np.vstack(buffer)


def _fetch_texts(store: VectorStore, ids: list[str], *, batch_size: int) -> list[str]:
    """Read text for the sampled ids only.

    Ids come back in whatever order the backend chooses, so the result is mapped
    by id rather than zipped positionally — a silent misalignment here would
    pair every document with someone else's text and produce a plausible-looking
    ARR that means nothing.
    """
    by_id: dict[str, str] = {}
    for record in store.iter_records(
        ids, with_vectors=False, with_text=True, batch_size=batch_size
    ):
        by_id[record.id] = record.text or ""
    return [by_id.get(record_id, "") for record_id in ids]


def _drop_empty_texts(
    ids: list[str], texts: list[str], vectors: FloatArray, labels: np.ndarray | None
) -> tuple[list[str], list[str], FloatArray, np.ndarray | None]:
    """Drop records with no text, and fail if that leaves nothing.

    A record with a vector but no text cannot be re-embedded, so it can only sit
    in the sample as noise.
    """
    keep = [i for i, text in enumerate(texts) if text.strip()]
    if not keep:
        raise StoreUnsupported(
            "None of the sampled records carry text.",
            hint=(
                "rebasis re-embeds the documents with the candidate model, which "
                "needs the original text. If this store holds vectors only, embed "
                "the corpus yourself and use the `precomputed` embedder."
            ),
            context={"count": len(ids)},
        )
    if len(keep) < len(ids):
        log.warning(
            Events.PROBE_SAMPLE_DRAWN,
            count=len(ids) - len(keep),
            dropped=DROPPED_NO_TEXT,
        )
    index = np.asarray(keep, dtype=np.int64)
    return (
        [ids[i] for i in keep],
        [texts[i] for i in keep],
        vectors[index],
        None if labels is None else labels[index],
    )


def _access_vector(
    ids: Sequence[str], access_counts: Mapping[str, float] | None
) -> FloatArray | None:
    """Access counts as a weight per sampled record, or ``None``.

    A record the log does not mention gets ``1.0`` rather than ``0.0``: a log
    records what *was* read, and absence from it means "not seen in this window"
    rather than "never retrievable". Zero would make such a record ineligible as
    a query proxy, which is a stronger claim than any access log supports.

    ``None`` when the log names nothing in this sample at all, so the caller
    falls back to a uniform split rather than to a vector of identical ones —
    the two behave the same and only the first can be reported honestly.
    """
    if not access_counts:
        return None
    weights = np.array([float(access_counts.get(record_id, 1.0)) for record_id in ids])
    if not np.any(weights != 1.0):
        return None
    return np.asarray(weights, dtype=np.float32)


def _positional(n: int, strategy: str, seed: int) -> SampleResult:
    """A :class:`SampleResult` over positions ``0..n-1``.

    ``split_disjoint`` splits a drawn sample; here the sample has already been
    drawn and reduced, so what it splits is the surviving positions.
    """
    return SampleResult(indices=np.arange(n, dtype=np.int64), strategy=strategy, seed=seed)


# ── ground truth tiers ───────────────────────────────────────────────────────


def _tier0(
    corpus: CorpusSample,
    new_doc_vectors: FloatArray,
    new_doc_vectors_as_queries: FloatArray | None,
    *,
    k: int,
) -> tuple[GroundTruth, FloatArray, FloatArray, np.ndarray | None]:
    """Held-out documents stand in for queries — the T0 tier.

    **The proxies are encoded the way a query would be**, not the way a document
    is. For a symmetric model the two are identical and this is a distinction
    without a difference. For an asymmetric one it is the difference between
    measuring "document retrieves document" and measuring what the tool is
    actually for.

    It also decides whether this tier can see a misconfigured prefix at all.
    With document-encoded proxies the ground truth and the measurement share the
    same wrong prefix, so the error cancels and T0 reports nothing — which is
    what section 4 of `docs/golden-findings.md` measured before this changed.

    The proxies *are* sample records, so their clusters are the sample's own —
    which is what makes ``tail_arr`` straightforward at this tier.
    """
    as_queries = (
        new_doc_vectors_as_queries if new_doc_vectors_as_queries is not None else new_doc_vectors
    )
    with span(Spans.GROUNDTRUTH_KNN, {"tier": "t0", "count": int(corpus.query_positions.size)}):
        ground_truth = build_tier0(
            new_doc_vectors, as_queries[corpus.query_positions], corpus.query_positions, k=k
        )
    return (
        ground_truth,
        corpus.old_vectors[corpus.query_positions],
        as_queries[corpus.query_positions],
        corpus.clusters_of(corpus.query_positions),
    )


def _tier2(  # noqa: PLR0913 - one argument per input the tier needs
    corpus: CorpusSample,
    new_embedder: Embedder,
    old_embedder: Embedder | None,
    new_doc_vectors: FloatArray,
    *,
    k: int,
    strategy: str,
) -> tuple[GroundTruth, FloatArray, FloatArray, np.ndarray | None]:
    """Queries synthesised from the documents — the T2 tier.

    Raises:
        StoreUnsupported: When no old-model embedder was supplied. The synthesised
            queries are text that was never indexed, so answering "how well does
            the current model do?" means encoding them with it — which is also
            what makes ``upgrade_gain`` computable at this tier.
        InsufficientSamples: When too few documents yielded a usable query.
    """
    from rebasis.probe.synthesis import synthesize_queries

    if old_embedder is None:
        raise StoreUnsupported(
            "Synthesised queries need the current model as well as the candidate.",
            hint=(
                "Pass --old so the queries can be encoded with the model the index "
                "was built with; without it the upgrade cannot be measured, which "
                "is the reason to synthesise queries at all."
            ),
            context={"tier": "t2"},
        )

    held_out = corpus.query_positions
    synthesised = synthesize_queries(
        [corpus.texts[int(p)] for p in held_out], held_out.tolist(), strategy=strategy
    )
    if len(synthesised) < MIN_SYNTHESISED_QUERIES:
        raise InsufficientSamples(
            f"Only {len(synthesised)} usable queries could be synthesised from "
            f"{held_out.size} held-out documents.",
            hint=(
                "Try --synth-queries title, or supply a real query log with "
                "--queries. Documents shorter than a sentence cannot produce one."
            ),
            context={"count": len(synthesised), "tier": "t2"},
        )
    if len(synthesised) < held_out.size:
        log.info(
            Events.PROBE_GROUNDTRUTH_COMPUTED,
            count=held_out.size - len(synthesised),
            tier="t2",
            dropped=DROPPED_NO_TEXT,
        )

    new_q = _encode(new_embedder, synthesised.queries, kind="query")
    old_q = _encode(old_embedder, synthesised.queries, kind="query")
    with span(Spans.GROUNDTRUTH_KNN, {"tier": "t2", "count": len(synthesised)}):
        ground_truth = build_tier2(
            new_doc_vectors, new_q, synthesised.source_positions, k=k, strategy=strategy
        )

    clusters = corpus.clusters_of(np.asarray(synthesised.source_positions, dtype=np.int64))
    return ground_truth, old_q, new_q, clusters


def _tier1(  # noqa: PLR0913 - one argument per input the tier needs
    corpus: CorpusSample,
    query_log: QueryLog,
    new_embedder: Embedder,
    old_embedder: Embedder | None,
    new_doc_vectors: FloatArray,
    *,
    k: int,
) -> tuple[GroundTruth, FloatArray, FloatArray, np.ndarray | None]:
    """Real queries, with judgements when the log carries them — the T1 tier.

    A log without judgements is not an error: it is what a search box exports.
    It goes to :func:`build_tier1_unjudged`, which measures retention against a
    full reindex and declines to guess at the upgrade.

    Raises:
        StoreUnsupported: When no old-model embedder was supplied and the log is
            judged. The "do nothing" baseline needs the queries under the
            current model, and without it the report cannot say what staying put
            would cost.
        InsufficientSamples: When the log is judged and no query has a judged
            document inside the sample.
    """
    if not query_log.judged:
        return _tier1_unjudged(corpus, query_log, new_embedder, old_embedder, new_doc_vectors, k=k)
    if old_embedder is None:
        raise StoreUnsupported(
            "A real query log needs the current model as well as the candidate.",
            hint=(
                "Pass --old so the queries can be encoded with the model the index "
                "was built with; that is what the 'do nothing' comparison measures."
            ),
            context={"tier": "t1"},
        )

    position_of = {record_id: i for i, record_id in enumerate(corpus.ids)}
    queries: list[str] = []
    qrels: list[set[int]] = []
    for query, relevant in zip(query_log.queries, query_log.qrels, strict=True):
        mapped = {position_of[r] for r in relevant if r in position_of}
        if mapped:
            queries.append(query)
            qrels.append(mapped)

    if not queries:
        raise InsufficientSamples(
            "No query in the log has a judged document inside the drawn sample.",
            hint=(
                "Raise --sample so more of the judged documents are included, or "
                "check that the ids under `relevant` match the ids in the store. "
                "A log with no `relevant` field at all is fine and takes a "
                "different path; this one has judgements that match nothing."
            ),
            context={"count": len(query_log.queries)},
        )
    if len(queries) < len(query_log.queries):
        log.info(
            Events.PROBE_GROUNDTRUTH_COMPUTED,
            count=len(query_log.queries) - len(queries),
            tier="t1",
            dropped=DROPPED_UNJUDGED,
        )

    new_q = _encode(new_embedder, queries, kind="query")
    old_q = _encode(old_embedder, queries, kind="query")
    with span(Spans.GROUNDTRUTH_KNN, {"tier": "t1", "count": len(queries)}):
        ground_truth = build_tier1(new_doc_vectors, new_q, qrels, k=k)

    # A real query is text that was never indexed, so it has no cluster of its
    # own. It inherits the cluster of the document it is judged against, which
    # is the region of the corpus it is actually asking about — and that is the
    # thing `tail_arr` is trying to detect drift in.
    clusters = None
    if corpus.cluster_labels is not None:
        clusters = np.asarray(
            [corpus.cluster_labels[min(mapped)] for mapped in qrels], dtype=np.int64
        )
    return ground_truth, old_q, new_q, clusters


def _tier1_unjudged(  # noqa: PLR0913 - one argument per input the tier needs
    corpus: CorpusSample,
    query_log: QueryLog,
    new_embedder: Embedder,
    old_embedder: Embedder | None,
    new_doc_vectors: FloatArray,
    *,
    k: int,
) -> tuple[GroundTruth, FloatArray, FloatArray, np.ndarray | None]:
    """Real queries with no judgements.

    The old model is encoded when it is available, so the report can still show
    what the current index returns for these queries — but it is not required
    here the way it is at T1, because there is no upgrade to measure either way.
    """
    from rebasis.probe.groundtruth import build_tier1_unjudged

    queries = list(query_log.queries)
    new_q = _encode(new_embedder, queries, kind="query")
    old_q = new_q if old_embedder is None else _encode(old_embedder, queries, kind="query")
    with span(Spans.GROUNDTRUTH_KNN, {"tier": "t1u", "count": len(queries)}):
        ground_truth = build_tier1_unjudged(new_doc_vectors, new_q, k=k)

    # A real query has no cluster of its own and, without a judgement, nothing
    # to inherit one from. `tail_arr` needs the query attributed to a region of
    # the corpus; the nearest oracle hit is the honest stand-in.
    clusters = None
    if corpus.cluster_labels is not None:
        clusters = np.asarray(
            [corpus.cluster_labels[int(row[0])] for row in ground_truth.oracle_indices],
            dtype=np.int64,
        )
    return ground_truth, old_q, new_q, clusters


def _upgrade_gain(
    ground_truth: GroundTruth, old_query_vectors: FloatArray, old_doc_vectors: FloatArray, k: int
) -> float | None:
    """How much better the new model is on this corpus.

    The decisive number when it is available: if the new model is not actually
    better here, the whole question of how to adopt it is moot.
    """
    from rebasis.probe.metrics import recall_at_k, top_k_search

    indices, _ = top_k_search(
        old_query_vectors, old_doc_vectors, k=k, self_mask=ground_truth.self_mask
    )
    old_recall = recall_at_k(indices, ground_truth.relevant_sparse, k)
    oracle = ground_truth.oracle_recall
    if oracle is None or old_recall <= 0:
        return None
    return float(oracle / old_recall)


def _encode(embedder: Embedder, texts: list[str], *, kind: str) -> FloatArray:
    """Encode and normalise, in chunks, logging progress per chunk."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    from rebasis.observability.semconv import (
        GEN_AI_EMBEDDINGS_DIMENSION_COUNT,
        GEN_AI_OPERATION_NAME,
        GEN_AI_REQUEST_MODEL,
        OPERATION_EMBEDDINGS,
    )

    model_id = embedder.profile.model_id
    started = time.perf_counter()
    # The dimension is on the span because it is the one number that explains a
    # duration: the same corpus through a 384-dimension model and a 1024 one is
    # the same `count` and a very different wait.
    corpus_attributes = {
        GEN_AI_OPERATION_NAME: OPERATION_EMBEDDINGS,
        GEN_AI_REQUEST_MODEL: model_id,
        GEN_AI_EMBEDDINGS_DIMENSION_COUNT: embedder.profile.dim,
        "count": len(texts),
    }

    with span(Spans.EMBED_CORPUS, corpus_attributes):
        blocks = [
            _encode_chunk(embedder, texts[i : i + ENCODE_CHUNK], kind=kind, index=index)
            for index, i in enumerate(range(0, len(texts), ENCODE_CHUNK))
        ]

    vectors = l2_normalize(np.vstack(blocks).astype(np.float32, copy=False), copy=False)
    duration_ms = (time.perf_counter() - started) * 1000
    instrument("rebasis.embed.texts").add(len(texts), {"model": model_id, "kind": kind})
    instrument("rebasis.embed.duration").record(duration_ms, {"model": model_id, "kind": kind})
    log.info(
        Events.EMBED_BATCH_COMPLETED,
        count=len(texts),
        model_id=model_id,
        duration_ms=round(duration_ms, 1),
        dim=int(vectors.shape[1]),
    )
    return vectors


def _encode_chunk(embedder: Embedder, texts: list[str], *, kind: str, index: int) -> FloatArray:
    """Encode one chunk, with a span for some of them.

    Sampled rather than one span per chunk: a large corpus produces thousands,
    and a trace where the batches outnumber everything else hides the shape of
    the run — the same reasoning as the hot-loop rule for logs, where a batch is
    logged once rather than once per record.
    """
    if not should_span_batch(index):
        return embedder.encode(texts, kind=kind)  # type: ignore[arg-type]
    with span(Spans.EMBED_BATCH, {"count": len(texts), "batch": index}):
        return embedder.encode(texts, kind=kind)  # type: ignore[arg-type]
