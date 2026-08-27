"""Can an adapter be fitted when the two spaces share no documents at all?

`rebasis fit` never loads the old model. It reads the index's own vectors on one
side and re-embeds the *same* documents with the candidate model on the other,
so the pairs come from the store. That works for every index that kept its text.

It does not work for an index that kept vectors and **discarded the text**.
There is nothing left to re-embed, no correspondence between the two spaces can
be built, and no adapter can be fitted by any means the tool currently has. That
is the limit this spike is about, and the only thing that removes it is a fit
that never needed the correspondence in the first place.

**The method.** Guy Dar, *mini-vec2vec: Scaling Universal Geometry Alignment
with Linear Transformations* (arXiv:2510.02348), reimplemented here from the
paper and from the author's notebook. Three stages: cluster both spaces
independently and match the centroids by a quadratic assignment on their
similarity matrices; turn the matched centroids into relative representations,
whose nearest neighbours across the two spaces are the pseudo-pairs; fit
orthogonal Procrustes on those, then refine by ICP with exponential smoothing
and one clustering-based correction. It needs numpy, scipy and scikit-learn and
nothing else — no torch, no faiss, no optimal-transport package. The rotation is
solved by the same ``scipy.linalg.orthogonal_procrustes`` that
``rebasis.core.procrustes`` already calls, and the preprocessing it asks for —
centre, then ℓ2-normalise — is ADR 1's default.

**The unpaired condition is structural, not a convention.** The corpus is split
into three disjoint document sets: an evaluation hold-out, an *old side* and a
*new side*. Old-model vectors are read only for the old side, new-model vectors
only for the new side, and the two sides are deliberately given **different row
counts and independently permuted row orders**, so no bijection between them
exists to leak even by accident. :func:`mini_vec2vec` receives two float
matrices and never sees an id, a position or a corpus. ``--leak-check`` re-runs
the whole fit with one side's rows permuted again and asserts the answer does
not move; a fit that had found correspondence would not survive that.

**What is measured.** The same ARR the probe reports, through the same
``evaluate_candidate``, against the same T0 ground truth, for four conditions:

    paired_full      procrustes_centered on every document in the pool,
                     both encodings. What `rebasis fit` produces today.
    paired_matched   the same fit restricted to the new side's documents, so
                     the pair count matches the unpaired condition's and the
                     only remaining difference is the correspondence.
    unpaired         the mini-vec2vec map, applied through an adapter that
                     un-centres back into the raw index space.
    unpaired_pairs   mini-vec2vec's final pseudo-pairs handed to the shipped
                     `fit_candidates(methods=["procrustes_centered"])` — the
                     identical call the paired ceiling makes, differing in
                     exactly one thing: whether the pairs are real.

``--random-init`` adds a fifth: the same refinement started from a random
orthogonal matrix instead of from the centroid matching. It is the control that
says whether stage 1 does any work, and it is the cheapest available comparison
point for the Wasserstein-Procrustes family, whose whole content is "start
somewhere, alternate a matching with a Procrustes solve".

Alongside ARR the run reports the paper's own metrics — top-1 accuracy and mean
rank of the true match on the held-out pairs — because a method can recover the
geometry well enough to identify a document and still not preserve a ranking,
and the two failures call for different conclusions.

    ~/rebasis/.venv/bin/python spikes/unpaired_align.py \\
        --corpus beir/scifact/test --ladder default --seed 0,1,2 \\
        --device cuda --out reports/unpaired/rows.jsonl

Numbers, not adjectives: whatever it prints is what goes in the report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.linalg
from scipy.optimize import linear_sum_assignment, quadratic_assignment
from sklearn.cluster import KMeans

from rebasis.compute import top_k_search, using_device
from rebasis.core import fit_candidates, geometry_bound
from rebasis.core.base import BaseAdapter, l2_normalize, pad_or_truncate
from rebasis.core.selection import AdapterCandidate
from rebasis.probe.groundtruth import build_tier0
from rebasis.probe.runner import _oracle_recall_at, evaluate_candidate
from rebasis.types import as_float32

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from rebasis.types import EncodingProfile, FloatArray

# ── the ladder ────────────────────────────────────────────────────────
#
# Copied from `tools/bridge_band.py` rather than imported. That module is being
# rewritten by someone else while this runs, and a spike that fails to start
# because a constant moved is worth less than fifteen duplicated lines. The
# names must stay identical anyway: they are half of the embedding cache key.

LADDERS: dict[str, tuple[tuple[str, str], ...]] = {
    "default": (
        ("minishlab/potion-base-8M", "sentence-transformers/all-MiniLM-L6-v2"),
        ("sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"),
        ("BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5"),
    ),
    "wide": (
        ("minishlab/potion-base-8M", "BAAI/bge-small-en-v1.5"),
        ("minishlab/potion-base-8M", "BAAI/bge-base-en-v1.5"),
        ("sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-base-en-v1.5"),
    ),
}

#: Four ir_datasets collections spanning two orders of magnitude in size, which
#: is the axis this method is most likely to be sensitive to: the published
#: results use ~26,000 vectors per side, and a real index is often smaller than
#: that in total.
CORPORA: dict[str, tuple[str, ...]] = {
    "sizes": (
        "beir/nfcorpus/test",
        "beir/scifact/test",
        "beir/cqadupstack/android",
        "beir/fiqa/test",
    ),
    "large": ("beir/fiqa/test", "beir/trec-covid"),
}


# ── hyperparameters ───────────────────────────────────────────────────
#
# Every default below is the published one. Where the paper and the author's
# notebook disagree the paper's value is taken and the divergence is named, so
# that a run which deviates from the published numbers can be attributed.

#: Ensemble members in the anchor-discovery stage. Each contributes `CLUSTERS`
#: anchors to a concatenated relative representation, so a member whose QAP
#: found the wrong permutation contributes noise that the others outvote.
RUNS = 30

#: Clusters per space per ensemble member. The paper raises this to 30 for one
#: model pair where 20 converged to a sub-optimal solution.
CLUSTERS = 20

#: 2-opt restarts inside one QAP. 2-opt starts from a uniformly random
#: permutation and stops at the first local optimum, so a single run is not
#: enough; the best-scoring of the restarts is kept.
QAP_RESTARTS = 30

#: Vectors each ensemble member clusters. Clustering all of them 30 times is the
#: dominant cost and buys nothing: 20 centroids do not need 25,000 points.
CLUSTER_SAMPLE = 10_000

#: Neighbours averaged to build one pseudo-pair. There is no one-to-one match
#: between the two document sets — that is the whole setting — so a single
#: nearest neighbour would be an arbitrary choice among many equally good ones.
NEIGHBOURS = 50

#: ICP iterations. The paper reports the alignment increasing monotonically with
#: this and 100 being "enough by a large margin".
REFINE1_ITERS = 100

#: Source rows per ICP iteration. **The paper says 10,000 and the notebook runs
#: 1,000.** The paper's appendix says larger is better, so the paper's value is
#: the default and `--refine1-sample` reaches the other.
REFINE1_SAMPLE = 10_000

#: Clusters in the correction stage, and how many times it runs. One, exactly:
#: the paper measures two or more making the alignment slightly worse.
REFINE2_CLUSTERS = 500
REFINE2_ITERS = 1

#: Exponential smoothing on the transform. This is also what lets W leave the
#: orthogonal manifold — orthogonality is enforced softly rather than by
#: re-projecting after every update, and α is what sets how far it drifts.
ALPHA = 0.5

#: Cut-off for every reported metric, and the candidate-set depth at which the
#: two-stage arrangement is reported. Both are the probe's own defaults, so the
#: numbers here can be read against everything else the project has measured.
K = 10
CASCADE_K = 100

#: Documents held out as query proxies, and the ceiling on the share of a
#: collection they may take. A proxy stays in the index and is masked out of its
#: own results, exactly as `build_tier0` does for `rebasis probe`.
HELDOUT = 1_000
HELDOUT_MAX_SHARE = 0.15

#: Vectors per side handed to the unpaired fit. 25,000 is the published scale
#: (the paper splits 51,808 training sentences in half). Capped rather than
#: fixed: a corpus that cannot supply it supplies what it has, and that shortfall
#: is the measurement.
FIT_SIZE = 25_000

#: How unevenly the pool is split between the two sides. Not 0.5 on purpose —
#: with equal counts a row-index leak would still be a bijection, and this
#: removes even the possibility of one. See :class:`Split`.
SPLIT_RATIO = 0.52


# ── corpus and embeddings ─────────────────────────────────────────────
#
# `load_corpus`, `_slug` and `embed_cached` are copied from
# `tools/bridge_band.py` (same reason as `LADDERS` above). `embed_cached` in
# particular has to be byte-identical in its cache key, or the warm `~/band-cache`
# is a cold one. The document order matters for the same reason: the cached
# `.npy` is a matrix of rows, and only the loader's iteration order says which
# row is which document.


@dataclass(slots=True)
class Corpus:
    """A collection's documents. T0 needs no queries — the documents are them."""

    name: str
    doc_ids: list[str]
    doc_texts: list[str]


def load_corpus(dataset: str) -> Corpus:
    """Read an ir_datasets collection, in `tools/bridge_band.py`'s exact order.

    No ``limit`` argument, deliberately. The shared embedding cache is keyed on
    the corpus name and not on any truncation, so a run that trimmed a corpus
    would write vectors for a *different* collection under the full collection's
    name and poison every later run. The way to trim here is a different cache
    directory.
    """
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
    return Corpus(name=dataset, doc_ids=doc_ids, doc_texts=doc_texts)


def _slug(text: str) -> str:
    return text.replace("/", "_").replace(":", "_")


def _encoder(model_id: str, device: str) -> Any:
    """A callable that turns texts into vectors. From `tools/bridge_band.py`."""
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


def embed_cached(
    *,
    model_id: str,
    profile: EncodingProfile,
    texts: Sequence[str],
    corpus_name: str,
    cache_dir: Path,
    device: str,
    encoder_cache: dict[str, Any],
) -> FloatArray:
    """Encode documents once ever. Cache key from `tools/bridge_band.py`.

    Only the ``document`` encoding is needed here: T0's query proxies are
    documents, and ADR 8 says a proxy is encoded the way a query is — but the
    two encodings differ only for an asymmetric model, and every model on this
    ladder is symmetric. An asymmetric pair would need the second encoding and
    this spike would have to grow a strategy comparison to go with it.
    """
    path = cache_dir / f"{_slug(corpus_name)}__{_slug(model_id)}__docs_document.npy"
    if path.exists():
        return np.load(path)

    if model_id not in encoder_cache:
        print(f"    loading {model_id}", flush=True)
        encoder_cache[model_id] = _encoder(model_id, device)

    prefix = profile.prefix_for("document")
    started = time.perf_counter()
    vectors = np.ascontiguousarray(
        np.asarray(encoder_cache[model_id]([prefix + t for t in texts]), dtype=np.float32)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, vectors)
    tmp.replace(path)
    print(
        f"    embedded {len(texts):,} docs with {model_id} "
        f"in {time.perf_counter() - started:.0f}s -> {path.name}",
        flush=True,
    )
    return vectors


def encode_documents(
    *, model_id: str, corpus: Corpus, cache_dir: Path, device: str, encoder_cache: dict[str, Any]
) -> FloatArray:
    """Unit-norm document vectors for one model.

    Normalised here rather than trusted: SentenceTransformer is asked for
    normalised output, but model2vec's static models are not, and the cache
    holds whatever each produced. ADR 1 makes ℓ2 normalisation the
    precondition for every fit in this project, and an unnormalised row would
    silently weight one document more than another in the Procrustes solve.
    """
    from rebasis.embed import profile_for

    vectors = embed_cached(
        model_id=model_id,
        profile=profile_for(model_id),
        texts=corpus.doc_texts,
        corpus_name=corpus.name,
        cache_dir=cache_dir,
        device=device,
        encoder_cache=encoder_cache,
    )
    if len(vectors) != len(corpus.doc_ids):
        msg = (
            f"cache holds {len(vectors):,} vectors for {corpus.name} but the loader "
            f"produced {len(corpus.doc_ids):,} documents"
        )
        raise RuntimeError(msg)
    return l2_normalize(as_float32(vectors))


# ── the split, made structural ────────────────────────────────────────


@dataclass(slots=True)
class Split:
    """Three disjoint document sets, and the proof that they are disjoint.

    ``old_side`` and ``new_side`` are what makes the condition unpaired. Only
    the old model's vectors are read for the first and only the new model's for
    the second, so the two matrices handed to the fit describe **different
    documents**. Three properties are asserted rather than commented, because a
    leak here would not fail — it would produce a good number for the wrong
    reason, which is the one outcome this experiment cannot survive:

    * the three index sets are pairwise disjoint, and so are the id sets they
      name — positions are checked because that is what indexes the matrices,
      ids because that is what identifies a document;
    * the two sides have **different lengths**, so no bijection between their
      rows exists at all;
    * each side's rows are permuted by its own generator, so row *i* of one and
      row *i* of the other are unrelated even before the length check.

    ``heldout`` stays inside the index and is masked out of its own results, the
    way :func:`rebasis.probe.build_tier0` does it.
    """

    heldout: np.ndarray
    old_side: np.ndarray
    new_side: np.ndarray
    pool: np.ndarray

    def check(self, doc_ids: Sequence[str]) -> dict[str, Any]:
        """Assert the disjointness and return what was asserted."""
        parts = {"heldout": self.heldout, "old_side": self.old_side, "new_side": self.new_side}
        names = list(parts)
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                shared = np.intersect1d(parts[first], parts[second])
                if shared.size:
                    msg = f"{first} and {second} share {shared.size} positions"
                    raise AssertionError(msg)
                a = {doc_ids[p] for p in parts[first]}
                b = {doc_ids[p] for p in parts[second]}
                if a & b:
                    msg = f"{first} and {second} share {len(a & b)} document ids"
                    raise AssertionError(msg)
        if len(self.old_side) == len(self.new_side):
            msg = "the two sides have equal length; a row bijection would exist"
            raise AssertionError(msg)
        return {
            "heldout": int(self.heldout.size),
            "old_side": int(self.old_side.size),
            "new_side": int(self.new_side.size),
            "pool": int(self.pool.size),
            "disjoint": True,
        }


def make_split(n_docs: int, *, seed: int, heldout: int, fit_size: int) -> Split:
    """Cut a corpus into an evaluation hold-out and two non-overlapping sides."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_docs)
    n_heldout = min(heldout, int(n_docs * HELDOUT_MAX_SHARE))
    if n_heldout < 50:
        msg = f"{n_docs} documents is too few to hold out an evaluation set from"
        raise RuntimeError(msg)

    held = np.sort(order[:n_heldout])
    pool = order[n_heldout:]
    cut = int(len(pool) * SPLIT_RATIO)
    # Each side gets its own generator, so the row order of one carries no
    # information about the row order of the other even before the sizes differ.
    new_side = np.random.default_rng(seed * 7919 + 1).permutation(pool[:cut])[:fit_size]
    old_side = np.random.default_rng(seed * 7919 + 2).permutation(pool[cut:])[:fit_size]
    if len(new_side) == len(old_side):
        old_side = old_side[:-1]
    return Split(heldout=held, old_side=old_side, new_side=new_side, pool=pool)


# ── mini-vec2vec ──────────────────────────────────────────────────────


@dataclass(slots=True)
class Preprocessed:
    """Centred, ℓ2-normalised vectors, plus what it takes to get back out."""

    hat: FloatArray
    mean: FloatArray
    #: Mean ‖x − μ‖ over the set. The map has to land back in the raw index
    #: space to be searchable, and this is the only thing in the preprocessing
    #: that normalisation destroys. Measured, not chosen.
    radius: float


def preprocess(raw: FloatArray, width: int) -> Preprocessed:
    """§3.1: pad to a common width, centre, then project onto the unit sphere.

    Padding is what lets the method run on a model pair with different
    dimensions, which the paper never has — all five of its encoders are 768-d,
    and ``orthogonal_procrustes`` refuses unequal shapes. Zeros preserve every
    inner product, so k-means, the cosine similarities and the Procrustes solve
    all see exactly the geometry they would have seen without it; it is the same
    convention ``rebasis.core.procrustes`` already uses.
    """
    padded = pad_or_truncate(as_float32(raw), width)
    mean = padded.mean(axis=0)
    centred = padded - mean
    radius = float(np.linalg.norm(centred, axis=1).mean())
    return Preprocessed(hat=l2_normalize(centred), mean=mean, radius=radius)


def _kernel(centroids: FloatArray, *, kind: str) -> FloatArray:
    """The centroid-to-centroid similarity matrix the QAP is solved on.

    ``cosine`` is what the paper writes. ``gram`` is what the author's notebook
    computes — ``H C Cᵀ H`` with ``H`` the centring matrix, which is the centred
    Gram matrix of centroids that are averages of unit vectors and therefore not
    unit vectors themselves. The two are not the same matrix and the QAP is not
    invariant to the difference, so this is a switch rather than a detail; the
    default is the notebook's, because that is what produced the published
    numbers.
    """
    if kind == "cosine":
        unit = l2_normalize(as_float32(centroids))
        return as_float32(unit @ unit.T)
    centred = as_float32(centroids) - as_float32(centroids).mean(axis=0)
    return as_float32(centred @ centred.T)


def _qap(
    kernel_a: FloatArray, kernel_b: FloatArray, *, restarts: int, seed: int
) -> tuple[np.ndarray, float]:
    """Match two similarity matrices under a permutation, best of ``restarts``.

    ``max_π Σ_ij S^A_ij S^B_π(i)π(j)``, which is the Quadratic Assignment
    Problem. 2-opt starts from a uniformly random permutation and climbs to the
    first local optimum, so one run is a lottery ticket; the paper takes the
    best of thirty and the whole cost is under half a second at C=20.

    The score returned is the objective divided by ``‖S^A‖_F ‖S^B‖_F`` — the
    cosine between the two flattened matrices, which is the paper's own reading
    of the objective. It says how *confident* the match is and not whether it is
    *right*: the objective has symmetries, and a wrong permutation can maximise
    it. :func:`_reference_permutation` is the second reading, available here and
    not to the method, and the two are reported side by side because a high
    score with a low agreement is a specific and diagnosable failure.
    """
    best_perm: np.ndarray | None = None
    best = -math.inf
    for restart in range(restarts):
        result = quadratic_assignment(
            kernel_a,
            kernel_b,
            method="2opt",
            options={"maximize": True, "rng": np.random.default_rng(seed * 1_000_003 + restart)},
        )
        if result.fun > best:
            best, best_perm = float(result.fun), result.col_ind
    assert best_perm is not None
    scale = float(np.linalg.norm(kernel_a) * np.linalg.norm(kernel_b))
    return best_perm, (best / scale if scale > 0 else 0.0)


def _kmeans(x: FloatArray, n_clusters: int, *, seed: int, init: FloatArray | None = None) -> Any:
    """k-means with one initialisation, seeded so a run can be replayed.

    The reference implementation seeds nothing, which is why its own paper has
    to report standard deviations over repeats. Seeding here is not a deviation
    from the method — the stochasticity is still there, it is just addressable,
    and ``--seed`` is how the sensitivity gets measured instead of estimated.
    """
    if init is not None:
        return KMeans(n_clusters=n_clusters, init=np.asarray(init, dtype=np.float64), n_init=1).fit(
            x
        )
    return KMeans(n_clusters=n_clusters, n_init=1, random_state=seed).fit(x)


def _procrustes(src: FloatArray, dst: FloatArray) -> FloatArray:
    """``min ‖src·W − dst‖`` over orthogonal W. The same call `core` makes."""
    rotation, _ = scipy.linalg.orthogonal_procrustes(as_float32(src), as_float32(dst))
    return as_float32(rotation)


def _matched_mean(
    source: FloatArray, target: FloatArray, pool: FloatArray, *, neighbours: int
) -> FloatArray:
    """Send each ``source`` row to the mean of its ``neighbours`` nearest in ``target``.

    ``target`` is where the neighbourhood is *found* and ``pool`` is what gets
    averaged; they are the same array in the ambient-space refinement and
    different arrays in the anchor stage, where the neighbourhood comes from the
    shared relative space and the averaging has to happen in the absolute one.

    An average rather than the single nearest neighbour. The two document sets
    do not overlap, so no source row has a correct match to find — only a
    neighbourhood of documents that are about the same thing, and picking one of
    them would be picking noise.
    """
    indices, _ = top_k_search(
        l2_normalize(as_float32(source), copy=True), as_float32(target), k=neighbours
    )
    return as_float32(pool[indices].mean(axis=1))


@dataclass(slots=True)
class Alignment:
    """The learned map, and everything the method can say about itself."""

    rotation: FloatArray
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _reference_permutation(
    centres_a: FloatArray, centres_b: FloatArray, reference: FloatArray
) -> np.ndarray:
    """The centroid correspondence the QAP *should* have found.

    **Nothing the method can see produces this.** ``reference`` is an orthogonal
    map fitted on the paired data the unpaired fit is forbidden to touch, so this
    is a diagnostic in the same declared sense as an oracle: it exists to say
    whether stage 1 succeeded, and it could not be used to make stage 1 succeed.

    It is a reference rather than a truth, and the difference is real. The two
    halves hold *disjoint documents*, so their k-means centroids have no exact
    counterparts — even a perfect method cannot recover a correspondence that
    does not exist. What this measures is whether the QAP found the closest
    thing to one: carry A's centroids into B's space with a map known to be
    right, then take the assignment that maximises total similarity. A method
    that matches centroids well agrees with it; one whose objective is maximised
    somewhere else does not, and the gap between a confident QAP score and a low
    agreement is the shape of that failure.
    """
    mapped = l2_normalize(as_float32(centres_a) @ reference, copy=False)
    similarity = as_float32(mapped @ l2_normalize(as_float32(centres_b), copy=True).T)
    _, columns = linear_sum_assignment(-similarity)
    return np.asarray(columns)


def anchor_pairs(
    x_a: FloatArray,
    x_b: FloatArray,
    *,
    seed: int,
    runs: int,
    clusters: int,
    qap_restarts: int,
    cluster_sample: int,
    neighbours: int,
    kernel: str,
    reference: FloatArray | None = None,
) -> tuple[FloatArray, dict[str, Any]]:
    """§3.2–3.3: find pseudo-pairs without any correspondence to start from.

    The conjecture the whole method rests on: two disjoint samples of the same
    distribution cluster into the *same themes*, so the centroids are landmarks
    that exist in both spaces even though no document does. Matching them is
    then a problem in C×C rather than in n×n, which is what makes it tractable —
    and the matching signal is the similarity *matrix*, since absolute
    coordinates are exactly what differs between the two spaces.

    The ensemble is the noise dilution. One run's permutation can be wrong; it
    contributes ``clusters`` bad coordinates to a relative representation that
    has ``runs × clusters`` of them, and the correct ones outvote it. That is
    cheaper than the alternative in the literature, which is to re-initialise
    the whole pipeline hundreds of times and hope one run is clean.
    """
    rng = np.random.default_rng(seed * 104_729 + 11)
    anchors_a: list[FloatArray] = []
    anchors_b: list[FloatArray] = []
    scores: list[float] = []
    agreements: list[float] = []
    started = time.perf_counter()
    for run in range(runs):
        sample_a = x_a[rng.permutation(len(x_a))[:cluster_sample]]
        sample_b = x_b[rng.permutation(len(x_b))[:cluster_sample]]
        centres_a = _kmeans(sample_a, clusters, seed=seed * 1000 + run).cluster_centers_
        centres_b = _kmeans(sample_b, clusters, seed=seed * 1000 + run + 500).cluster_centers_
        permutation, score = _qap(
            _kernel(centres_a, kind=kernel),
            _kernel(centres_b, kind=kernel),
            restarts=qap_restarts,
            seed=seed * 1000 + run,
        )
        if reference is not None:
            truth = _reference_permutation(centres_a, centres_b, reference)
            agreements.append(float((np.asarray(permutation) == truth).mean()))
        anchors_a.append(as_float32(centres_a))
        anchors_b.append(as_float32(centres_b)[permutation])
        scores.append(score)

    all_a = np.vstack(anchors_a)
    all_b = np.vstack(anchors_b)
    # The relative representation: each vector described by how close it is to
    # every anchor. Absolute coordinates differ between the two spaces; these do
    # not, which is what makes a nearest-neighbour search *across* them mean
    # anything at all.
    relative_a = l2_normalize(as_float32(l2_normalize(x_a, copy=True) @ l2_normalize(all_a).T))
    relative_b = l2_normalize(as_float32(l2_normalize(x_b, copy=True) @ l2_normalize(all_b).T))
    matched = _matched_mean(relative_a, relative_b, x_b, neighbours=neighbours)
    agreement: dict[str, Any] = {}
    if agreements:
        # Chance is 1/C: a uniformly random permutation gets one cluster right
        # on average, so the floor this has to clear is small but not zero.
        agreement = {
            "centroid_agreement_mean": round(float(np.mean(agreements)), 4),
            "centroid_agreement_max": round(float(np.max(agreements)), 4),
            "centroid_agreement_chance": round(1.0 / clusters, 4),
        }
    return matched, {
        **agreement,
        "qap_score_mean": round(float(np.mean(scores)), 4),
        "qap_score_min": round(float(np.min(scores)), 4),
        "qap_score_max": round(float(np.max(scores)), 4),
        "qap_score_std": round(float(np.std(scores)), 4),
        "anchors": int(all_a.shape[0]),
        "anchor_seconds": round(time.perf_counter() - started, 1),
    }


def refine_1(
    x_a: FloatArray,
    x_b: FloatArray,
    rotation: FloatArray,
    *,
    seed: int,
    iterations: int,
    sample: int,
    neighbours: int,
    alpha: float,
) -> tuple[FloatArray, list[float]]:
    """Iterative closest point, in the ambient target space, with smoothing.

    The difference from the anchor stage is one word: the neighbourhood is now
    found in space B itself rather than in the shared relative space. Once a
    coarse map exists, the mapped source vector is already close to where its
    match would be, and B's own geometry is a better guide than a 600-dimensional
    proxy for it.

    ``W ← (1−α)W + αW_new`` leaves the orthogonal manifold and is not projected
    back. That is the method, not an oversight: α is what sets how far the map
    is allowed to drift from a rotation, and the paper measures a larger α
    buying cosine similarity that does not always turn into rank.

    The returned trace is the *unsupervised* objective — mean cosine between the
    mapped sample and its matched average. It is what a user could actually
    watch, unlike the paper's own progress metric, which is computed against
    held-out true pairs that in this setting would not exist.
    """
    rng = np.random.default_rng(seed * 15_485_863 + 3)
    trace: list[float] = []
    for _ in range(iterations):
        rows = rng.permutation(len(x_a))[:sample]
        source = x_a[rows]
        mapped = l2_normalize(as_float32(source @ rotation))
        matched = _matched_mean(mapped, x_b, x_b, neighbours=neighbours)
        trace.append(float(np.sum(mapped * l2_normalize(matched, copy=True), axis=1).mean()))
        rotation = as_float32((1.0 - alpha) * rotation + alpha * _procrustes(source, matched))
    return rotation, trace


def refine_2(
    x_a: FloatArray,
    x_b: FloatArray,
    rotation: FloatArray,
    *,
    seed: int,
    clusters: int,
    iterations: int,
    alpha: float,
) -> tuple[FloatArray, list[float]]:
    """One clustering-based correction, and only one.

    The trick is in the initialisation. Cluster A, push the centroids through W,
    and use *those* as the starting centroids for clustering B. k-means then
    only nudges them, so cluster *j* in B is the same theme as cluster *j* in A
    by construction — a correspondence obtained for free, over sets rather than
    over individual points, which is exactly where ICP's per-point matching is
    weakest.

    The paper measures a second iteration making things slightly worse, and
    conjectures the correction removes a systematic bias that is not there to
    remove twice. Kept as a loop with a default of one so that reading the
    number back out is a run rather than an argument.
    """
    self_consistency: list[float] = []
    for iteration in range(iterations):
        centres_a = as_float32(
            _kmeans(x_a, clusters, seed=seed * 2000 + iteration).cluster_centers_
        )
        transformed = as_float32(centres_a @ rotation)
        centres_b = as_float32(_kmeans(x_b, clusters, seed=0, init=transformed).cluster_centers_)
        self_consistency.append(
            float(
                np.sum(
                    l2_normalize(transformed, copy=True) * l2_normalize(centres_b, copy=True),
                    axis=1,
                ).mean()
            )
        )
        rotation = as_float32((1.0 - alpha) * rotation + alpha * _procrustes(centres_a, centres_b))
    return rotation, self_consistency


def mini_vec2vec(
    x_a: FloatArray,
    x_b: FloatArray,
    *,
    seed: int,
    config: dict[str, Any],
    reference: FloatArray | None = None,
) -> Alignment:
    """arXiv:2510.02348, end to end. Two matrices in, one matrix out.

    The signature is the experiment's guarantee. This function is handed two
    float arrays of different lengths and is told nothing else — not an id, not
    a position, not which corpus either came from — so there is no correspondence
    available to it even in principle. Everything upstream can be checked by
    reading :class:`Split`; everything here is checked by the type of the
    arguments.

    Both inputs must already be preprocessed: padded to a common width, centred,
    ℓ2-normalised. :func:`preprocess` does that and keeps what it takes to
    invert the centring, which :class:`MiniVec2VecAdapter` needs to land back in
    the index's own space.

    ``reference`` is the one exception to the paragraph above, and it is not an
    input to the method: it is a map fitted on paired data, used *after* the
    matching to score it, and it reaches nothing that decides anything. Passing
    it turns "the QAP was confident" into "the QAP was right", which are
    different claims and were being conflated.
    """
    diagnostics: dict[str, Any] = {}
    started = time.perf_counter()

    if config["random_init"]:
        # The control. If the refinement reaches the same place from a random
        # rotation, stage 1 is decoration and the method is ICP with extra steps.
        generator = np.random.default_rng(seed * 6_700_417 + 5)
        rotation = as_float32(
            np.linalg.qr(generator.standard_normal((x_a.shape[1], x_a.shape[1])))[0]
        )
        diagnostics["init"] = "random_orthogonal"
    else:
        matched, anchor_diagnostics = anchor_pairs(
            x_a,
            x_b,
            seed=seed,
            runs=config["runs"],
            clusters=config["clusters"],
            qap_restarts=config["qap_restarts"],
            cluster_sample=config["cluster_sample"],
            neighbours=config["neighbours"],
            kernel=config["kernel"],
            reference=reference,
        )
        rotation = _procrustes(x_a, matched)
        diagnostics.update(anchor_diagnostics)
        diagnostics["init"] = "centroid_matching"
    diagnostics["initial_seconds"] = round(time.perf_counter() - started, 1)

    mark = time.perf_counter()
    rotation, trace = refine_1(
        x_a,
        x_b,
        rotation,
        seed=seed,
        iterations=config["refine1_iters"],
        sample=min(config["refine1_sample"], len(x_a)),
        neighbours=config["refine1_neighbours"],
        alpha=config["alpha"],
    )
    diagnostics["refine1_seconds"] = round(time.perf_counter() - mark, 1)
    diagnostics["refine1_objective"] = [round(v, 4) for v in trace[:: max(1, len(trace) // 10)]]
    diagnostics["refine1_objective_final"] = round(trace[-1], 4) if trace else None
    # How far the smoothed matrix has drifted from a rotation. ‖WᵀW − I‖_F / √d
    # is 0 for an orthogonal matrix; the paper says α controls this and never
    # reports it, and the geometry claim the method rests on is about rotations.
    gram = as_float32(rotation.T @ rotation)
    diagnostics["orthogonality_error"] = round(
        float(
            np.linalg.norm(gram - np.eye(gram.shape[0], dtype=np.float32)) / math.sqrt(len(gram))
        ),
        4,
    )

    mark = time.perf_counter()
    rotation, consistency = refine_2(
        x_a,
        x_b,
        rotation,
        seed=seed,
        clusters=max(2, min(config["refine2_clusters"], min(len(x_a), len(x_b)) // 8)),
        iterations=config["refine2_iters"],
        alpha=config["alpha"],
    )
    diagnostics["refine2_seconds"] = round(time.perf_counter() - mark, 1)
    diagnostics["refine2_self_consistency"] = [round(v, 4) for v in consistency]
    diagnostics["fit_seconds"] = round(time.perf_counter() - started, 1)
    return Alignment(rotation=rotation, diagnostics=diagnostics)


# ── the adapter the map becomes ───────────────────────────────────────


class MiniVec2VecAdapter(BaseAdapter):
    """``g(x) = ‖x_pad − μ_src‖⁻¹(x_pad − μ_src)·W·σ + μ_dst``.

    Not ``CenteredProcrustesAdapter``, and the difference is one operation: this
    normalises the *centred* source before rotating, because that is the space
    mini-vec2vec fitted W in. Skipping it would apply the map to vectors from a
    distribution it never saw.

    ``σ`` is the mean centred norm of the target set, measured at fit time. The
    map lands on the unit sphere of the centred target space, and the index is
    not stored there — it holds raw vectors around μ_dst. Something has to say
    how far from μ_dst to land, and the only honest answer is how far the target
    documents themselves are.
    """

    kind = "minivec2vec"

    def __init__(
        self,
        rotation: FloatArray,
        mean_src: FloatArray,
        mean_dst: FloatArray,
        *,
        input_dim: int,
        output_dim: int,
        scale: float,
    ) -> None:
        super().__init__(input_dim=input_dim, output_dim=output_dim, config={"scale": scale})
        self.rotation = rotation
        self.mean_src = mean_src
        self.mean_dst = mean_dst
        self.scale = scale
        self._work_dim = rotation.shape[0]

    def apply(self, x: FloatArray) -> FloatArray:
        """Pad, centre, normalise, rotate, rescale, un-centre, truncate."""
        z = pad_or_truncate(as_float32(x), self._work_dim) - self.mean_src
        z = l2_normalize(z, copy=True) @ self.rotation
        z *= self.scale
        z += self.mean_dst
        return as_float32(np.ascontiguousarray(z[..., : self.output_dim]))

    def state_dict(self) -> dict[str, FloatArray]:
        """The rotation and the two means; σ is config, not a weight."""
        return {
            "rotation": self.rotation,
            "mean_src": self.mean_src,
            "mean_dst": self.mean_dst,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, FloatArray], config: Mapping[str, Any]) -> Any:
        """Reconstruct from stored weights."""
        return cls(
            as_float32(state["rotation"]),
            as_float32(state["mean_src"]),
            as_float32(state["mean_dst"]),
            input_dim=int(config["input_dim"]),
            output_dim=int(config["output_dim"]),
            scale=float(config["scale"]),
        )


# ── measurement ───────────────────────────────────────────────────────


def _candidate(name: str, adapter: BaseAdapter, seconds: float) -> AdapterCandidate:
    return AdapterCandidate(method=name, adapter=adapter, fit_seconds=seconds)


def _score(
    candidate: AdapterCandidate,
    *,
    new_queries: FloatArray,
    old_docs: FloatArray,
    new_docs: FloatArray,
    ground_truth: Any,
    csls_source: FloatArray,
    with_csls: bool,
    oracle_at_cascade: float | None,
) -> dict[str, Any]:
    """Score one adapter through the probe's own ``evaluate_candidate``.

    One deliberate difference from :func:`rebasis.probe.run_probe`: the CSLS
    sample is built from *this* candidate's adapter rather than from the first
    candidate's. In the probe every candidate is a fit of the same pairs, so
    sharing the sample is a saving; here the candidates are the experiment's
    conditions, and giving one of them another's hubness correction would make
    the comparison meaningless.
    """
    sample = None
    if with_csls:
        sample = l2_normalize(candidate.adapter.apply(csls_source), copy=False)
    metrics = evaluate_candidate(
        candidate,
        query_vectors_new=new_queries,
        old_doc_vectors=old_docs,
        ground_truth=ground_truth,
        k=K,
        csls_sample=sample,
        cascade_k=CASCADE_K,
        oracle_recall_at_cascade=oracle_at_cascade,
    )
    row = metrics.to_dict()
    row["condition"] = candidate.method
    row["fit_seconds"] = round(candidate.fit_seconds, 1)
    row["arr_without_csls"] = round(float(metrics.extras["arr_without_csls"]), 4)
    del row["adapter_type"]
    row.update(
        _paper_metrics(
            candidate.adapter, new_docs=new_docs, old_docs=old_docs, rows=ground_truth.query_indices
        )
    )
    return row


def _paper_metrics(
    adapter: BaseAdapter, *, new_docs: FloatArray, old_docs: FloatArray, rows: np.ndarray
) -> dict[str, Any]:
    """arXiv:2510.02348's own metrics, on the held-out documents.

    Top-1 accuracy and mean rank of the *true* match — computable here only
    because this setting has the text on both sides and could therefore have
    formed the pairs, which is exactly what the fit was forbidden to do. They
    are reported because they answer a different question from ARR: whether the
    map identifies a document, rather than whether it preserves a ranking. A
    method can do the first well and the second badly, and the two failures
    point at different conclusions.

    Not comparable with the paper's figures without the pool size next to them:
    the paper ranks against 8,192 candidates and this ranks against however many
    documents were held out.
    """
    mapped = l2_normalize(adapter.apply(new_docs[rows]), copy=False)
    targets = as_float32(old_docs[rows])
    scores = as_float32(mapped @ targets.T)
    truth = np.diag(scores)
    ranks = (scores > truth[:, None]).sum(axis=1) + 1
    return {
        "pair_pool": int(rows.size),
        "top1": round(float((ranks == 1).mean()), 4),
        "mean_rank": round(float(ranks.mean()), 2),
        "pair_cosine": round(float(truth.mean()), 4),
    }


def run_one(
    *,
    corpus: Corpus,
    old_vectors: FloatArray,
    new_vectors: FloatArray,
    old_model: str,
    new_model: str,
    seed: int,
    fit_size: int,
    heldout: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """One corpus, one model pair, one seed: the ceiling and the unpaired result."""
    split = make_split(len(corpus.doc_ids), seed=seed, heldout=heldout, fit_size=fit_size)
    guarantee = split.check(corpus.doc_ids)

    ground_truth = build_tier0(new_vectors, new_vectors[split.heldout], split.heldout, k=K)
    # The probe's own function, imported rather than reimplemented: `cascade_arr`
    # is a ratio against this, and a denominator computed a second way here
    # would make the spike's numbers incomparable with the tool's.
    oracle_at_cascade = _oracle_recall_at(
        CASCADE_K,
        k=K,
        new_query_vectors=new_vectors[split.heldout],
        new_doc_vectors=new_vectors,
        ground_truth=ground_truth,
    )

    common = {
        "new_queries": new_vectors[split.heldout],
        "old_docs": old_vectors,
        "new_docs": new_vectors,
        "ground_truth": ground_truth,
        "csls_source": new_vectors[split.new_side[: min(4000, len(split.new_side))]],
        "with_csls": config["with_csls"],
        "oracle_at_cascade": oracle_at_cascade,
    }

    base = {
        "corpus": corpus.name,
        "documents": len(corpus.doc_ids),
        "old_model": old_model,
        "new_model": new_model,
        "old_dim": int(old_vectors.shape[1]),
        "new_dim": int(new_vectors.shape[1]),
        "seed": seed,
        "fit_size": fit_size,
        "split": guarantee,
        "init": "random_orthogonal" if config["random_init"] else "centroid_matching",
        "kernel": config["kernel"],
    }
    rows: list[dict[str, Any]] = []

    # ── the paired ceiling ────────────────────────────────────────────
    # What any orthogonal alignment of these two spaces can achieve, before
    # anything is fitted. It bounds every condition below, paired or not.
    bound = geometry_bound(new_vectors[split.pool], old_vectors[split.pool], seed=seed)
    for name, indices in (("paired_full", split.pool), ("paired_matched", split.new_side)):
        started = time.perf_counter()
        fitted = fit_candidates(
            new_vectors[indices],
            old_vectors[indices],
            normalize=False,
            methods=["procrustes_centered"],
        )
        elapsed = time.perf_counter() - started
        row = _score(_candidate(name, fitted[0].adapter, elapsed), **common)
        rows.append({**base, **row, "pairs": int(indices.size)})

    # ── the unpaired condition ────────────────────────────────────────
    width = max(old_vectors.shape[1], new_vectors.shape[1])
    source = preprocess(new_vectors[split.new_side], width)
    target = preprocess(old_vectors[split.old_side], width)

    # The scoring reference for stage 1, in the same preprocessed geometry the
    # method works in and over documents present on both sides — which only a
    # diagnostic may look at. `Split` keeps the fit halves disjoint; this uses
    # the pool, and it is passed to `mini_vec2vec` after the matching rather
    # than into it.
    paired_source = preprocess(new_vectors[split.pool], width)
    paired_target = preprocess(old_vectors[split.pool], width)
    reference = _procrustes(paired_source.hat, paired_target.hat)

    alignment = mini_vec2vec(source.hat, target.hat, seed=seed, config=config, reference=reference)

    adapter = MiniVec2VecAdapter(
        alignment.rotation,
        source.mean,
        target.mean,
        input_dim=int(new_vectors.shape[1]),
        output_dim=int(old_vectors.shape[1]),
        scale=target.radius,
    )
    row = _score(_candidate("unpaired", adapter, alignment.diagnostics["fit_seconds"]), **common)
    rows.append({**base, **row, "pairs": int(split.new_side.size), **alignment.diagnostics})

    # The same pseudo-pairs, handed to the same `fit_candidates` call the
    # ceiling used. Isolates the one variable: real pairs against invented ones.
    started = time.perf_counter()
    pseudo = _matched_mean(
        as_float32(source.hat @ alignment.rotation),
        target.hat,
        old_vectors[split.old_side],
        neighbours=config["refine1_neighbours"],
    )
    fitted = fit_candidates(
        new_vectors[split.new_side], pseudo, normalize=False, methods=["procrustes_centered"]
    )
    elapsed = alignment.diagnostics["fit_seconds"] + (time.perf_counter() - started)
    row = _score(_candidate("unpaired_pairs", fitted[0].adapter, elapsed), **common)
    rows.append({**base, **row, "pairs": int(split.new_side.size)})

    for row in rows:
        row["geometry_delta"] = round(bound.delta, 4)
        row["alignment_bound"] = round(bound.bound, 4)
        row["oracle_recall_at_cascade"] = round(float(oracle_at_cascade), 4)
    return rows


# ── the leak check ────────────────────────────────────────────────────


def leak_check(
    *,
    corpus: Corpus,
    old_vectors: FloatArray,
    new_vectors: FloatArray,
    seed: int,
    fit_size: int,
    heldout: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Re-fit with the target side's rows shuffled, and assert nothing moved.

    A comment that says "these sets are disjoint" is a claim. This is a test of
    it. Row order is the only channel through which a correspondence could
    survive the split — the two sides already hold different documents, so if
    permuting one of them changes the fitted map, the map was reading something
    it should not have been able to see.

    Compared on the fitted matrix rather than on ARR: two ARR values can agree
    to three decimals by coincidence, but two rotations agreeing to float
    tolerance cannot. k-means is seeded per call, so the only difference between
    the two fits is the order of the rows.
    """
    split = make_split(len(corpus.doc_ids), seed=seed, heldout=heldout, fit_size=fit_size)
    width = max(old_vectors.shape[1], new_vectors.shape[1])
    source = preprocess(new_vectors[split.new_side], width)
    target = preprocess(old_vectors[split.old_side], width)

    straight = mini_vec2vec(source.hat, target.hat, seed=seed, config=config)
    shuffled_rows = np.random.default_rng(seed + 424_242).permutation(len(target.hat))
    shuffled = mini_vec2vec(
        source.hat, as_float32(target.hat[shuffled_rows]), seed=seed, config=config
    )
    delta = float(np.abs(straight.rotation - shuffled.rotation).max())
    return {
        "corpus": corpus.name,
        "seed": seed,
        "max_abs_rotation_delta": float(f"{delta:.3e}"),
        "invariant": bool(delta < 1e-4),
    }


# ── driver ────────────────────────────────────────────────────────────


def _key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        row["corpus"],
        row["old_model"],
        row["new_model"],
        str(row["seed"]),
        str(row["fit_size"]),
        str(row.get("init", "")),
    )


def already_done(out: Path | None) -> set[tuple[str, ...]]:
    """Keys already in the output file, so an interrupted run resumes."""
    if out is None or not out.exists():
        return set()
    done: set[tuple[str, ...]] = set()
    for line in out.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(_key(json.loads(line)))
    return done


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="sizes")
    parser.add_argument("--ladder", default="default")
    parser.add_argument("--pair", action="append", default=None, help="old,new")
    parser.add_argument("--seed", default="0,1,2")
    parser.add_argument("--fit-size", default=str(FIT_SIZE))
    parser.add_argument("--heldout", type=int, default=HELDOUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / "band-cache")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--clusters", type=int, default=CLUSTERS)
    parser.add_argument("--qap-restarts", type=int, default=QAP_RESTARTS)
    parser.add_argument("--cluster-sample", type=int, default=CLUSTER_SAMPLE)
    parser.add_argument("--neighbours", type=int, default=NEIGHBOURS)
    parser.add_argument("--refine1-iters", type=int, default=REFINE1_ITERS)
    parser.add_argument("--refine1-sample", type=int, default=REFINE1_SAMPLE)
    parser.add_argument("--refine1-neighbours", type=int, default=NEIGHBOURS)
    parser.add_argument("--refine2-clusters", type=int, default=REFINE2_CLUSTERS)
    parser.add_argument("--refine2-iters", type=int, default=REFINE2_ITERS)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--kernel", default="gram", choices=("gram", "cosine"))
    parser.add_argument("--no-csls", action="store_true")
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Skip the centroid matching and refine from a random rotation",
    )
    parser.add_argument("--leak-check", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Summarise an existing rows file and exit, running nothing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.report is not None:
        summarise(
            [
                json.loads(line)
                for line in args.report.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        )
        return 0
    corpora = CORPORA.get(args.corpus, tuple(args.corpus.split(",")))
    pairs = tuple(tuple(p.split(",")) for p in args.pair) if args.pair else LADDERS[args.ladder]
    seeds = [int(s) for s in str(args.seed).split(",")]
    fit_sizes = [int(s) for s in str(args.fit_size).split(",")]
    config = {
        "runs": args.runs,
        "clusters": args.clusters,
        "qap_restarts": args.qap_restarts,
        "cluster_sample": args.cluster_sample,
        "neighbours": args.neighbours,
        "refine1_iters": args.refine1_iters,
        "refine1_sample": args.refine1_sample,
        "refine1_neighbours": args.refine1_neighbours,
        "refine2_clusters": args.refine2_clusters,
        "refine2_iters": args.refine2_iters,
        "alpha": args.alpha,
        "kernel": args.kernel,
        "random_init": args.random_init,
        "with_csls": not args.no_csls,
    }

    done = already_done(args.out)
    encoder_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    handle = None
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        handle = args.out.open("a", encoding="utf-8")

    try:
        with using_device(args.device):
            for name in corpora:
                print(f"\n{name}", flush=True)
                corpus = load_corpus(name)
                print(f"  {len(corpus.doc_ids):,} documents", flush=True)
                for old_model, new_model in pairs:
                    vectors = {
                        model: encode_documents(
                            model_id=model,
                            corpus=corpus,
                            cache_dir=args.cache_dir,
                            device="cuda" if args.device != "cpu" else "cpu",
                            encoder_cache=encoder_cache,
                        )
                        for model in (old_model, new_model)
                    }
                    for fit_size in fit_sizes:
                        for seed in seeds:
                            stamp = {
                                "corpus": name,
                                "old_model": old_model,
                                "new_model": new_model,
                                "seed": seed,
                                "fit_size": fit_size,
                                "init": "random_orthogonal"
                                if args.random_init
                                else "centroid_matching",
                            }
                            if _key(stamp) in done:
                                print(f"  skip {old_model}->{new_model} seed {seed}", flush=True)
                                continue
                            print(
                                f"  {old_model} -> {new_model}  seed {seed}  fit {fit_size:,}",
                                flush=True,
                            )
                            if args.leak_check:
                                report = leak_check(
                                    corpus=corpus,
                                    old_vectors=vectors[old_model],
                                    new_vectors=vectors[new_model],
                                    seed=seed,
                                    fit_size=fit_size,
                                    heldout=args.heldout,
                                    config=config,
                                )
                                print(f"    leak check {json.dumps(report)}", flush=True)
                                continue
                            produced = run_one(
                                corpus=corpus,
                                old_vectors=vectors[old_model],
                                new_vectors=vectors[new_model],
                                old_model=old_model,
                                new_model=new_model,
                                seed=seed,
                                fit_size=fit_size,
                                heldout=args.heldout,
                                config=config,
                            )
                            for row in produced:
                                rows.append(row)
                                print(
                                    f"    {row['condition']:16s} arr {row['arr_r10']:.3f}  "
                                    f"cascade {row['cascade_arr']}  top1 {row['top1']:.3f}  "
                                    f"rank {row['mean_rank']:.1f}",
                                    flush=True,
                                )
                                if handle is not None:
                                    handle.write(json.dumps(row) + "\n")
                                    handle.flush()
    finally:
        if handle is not None:
            handle.close()

    if rows:
        summarise(rows)
    return 0


def _spread(values: list[float]) -> str:
    """``mean`` with the seed range beside it, because the range is the finding.

    A method that works on one seed and not another is a different result from
    one that works, and a column of means cannot tell them apart. Printed even
    for a single seed, where it collapses to the point estimate and says so.
    """
    if not values:
        return "     -        "
    if len(values) == 1:
        return f"{values[0]:6.3f}         "
    return f"{np.mean(values):6.3f} ({min(values):.3f}-{max(values):.3f})"


def summarise(rows: list[dict[str, Any]]) -> None:
    """One line per (corpus, pair, fit size): ceiling and unpaired side by side."""
    print()
    print(
        f"{'corpus':22s} {'old':>18s} -> {'new':18s} {'fit':>7s} {'n':>2s} "
        f"{'paired ceiling':>15s} {'unpaired (best)':>15s} {'frac':>6s} "
        f"{'top1':>6s} {'rank':>7s}"
    )
    grouped: dict[tuple[str, str, str, int], dict[str, list[float]]] = {}
    for row in rows:
        key = (row["corpus"], row["old_model"], row["new_model"], row["fit_size"])
        bucket = grouped.setdefault(key, {})
        bucket.setdefault(row["condition"], []).append(row["arr_r10"])
        bucket.setdefault(row["condition"] + "/top1", []).append(row["top1"])
        bucket.setdefault(row["condition"] + "/rank", []).append(row["mean_rank"])
    for (corpus, old_model, new_model, fit_size), values in sorted(grouped.items()):
        ceiling = values.get("paired_full", [])
        # The better of the two unpaired parametrisations, per seed rather than
        # per column: taking the max of two means would report a run that never
        # happened.
        direct = values.get("unpaired", [])
        viapairs = values.get("unpaired_pairs", [])
        best = [max(pair) for pair in zip(direct, viapairs, strict=False)] or direct or viapairs
        fraction = [u / c for u, c in zip(best, ceiling, strict=False) if c > 0] if ceiling else []
        print(
            f"{corpus[-22:]:22s} {old_model.split('/')[-1][:18]:>18s} -> "
            f"{new_model.split('/')[-1][:18]:18s} {fit_size:7,d} {len(ceiling):2d} "
            f"{_spread(ceiling)} {_spread(best)} "
            f"{(np.mean(fraction) if fraction else float('nan')):6.2f} "
            f"{np.mean(values.get('unpaired/top1', [float('nan')])):6.3f} "
            f"{np.mean(values.get('unpaired/rank', [float('nan')])):7.1f}"
        )
    _diagnostics(rows)


def _diagnostics(rows: list[dict[str, Any]]) -> None:
    """What the method can say about itself, without the answer key.

    Every column here is computable in the deployment this is for — an index
    with no text — which is the point of printing them next to ARR. If one of
    them tracks ARR, a user could tell a fit that worked from one that did not;
    if none does, the method has no self-diagnosis and that is a finding about
    whether it can be shipped, separate from whether it works.
    """
    carriers = [r for r in rows if r["condition"] == "unpaired" and "refine1_objective" in r]
    if not carriers:
        return
    print()
    print(
        f"{'corpus':22s} {'pair':30s} {'seed':>4s} {'qap':>13s} {'icp':>11s} "
        f"{'ortho':>6s} {'k-means':>7s} {'arr':>6s} {'sec':>6s}"
    )
    for row in sorted(carriers, key=lambda r: (r["corpus"], r["new_model"], r["seed"])):
        trace = row["refine1_objective"]
        consistency = row.get("refine2_self_consistency") or [float("nan")]
        pair = f"{row['old_model'].split('/')[-1][:13]}->{row['new_model'].split('/')[-1][:13]}"
        print(
            f"{row['corpus'][-22:]:22s} {pair[:30]:30s} {row['seed']:4d} "
            f"{row.get('qap_score_mean', float('nan')):.3f}"
            f"±{row.get('qap_score_std', float('nan')):.3f} "
            f"{trace[0]:.3f}->{row['refine1_objective_final']:.3f} "
            f"{row.get('orthogonality_error', float('nan')):6.3f} "
            f"{consistency[0]:7.3f} {row['arr_r10']:6.3f} {row['fit_seconds']:6.0f}"
        )


if __name__ == "__main__":
    sys.exit(main())
