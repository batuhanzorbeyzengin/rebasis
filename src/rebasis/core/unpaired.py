"""Aligning two spaces that share no document.

`fit` never loads the old model: it reads the index's vectors on one side and
re-embeds the **same** documents with the candidate on the other, so the pairs
come from the store. This module is what happens when there are no pairs at all
— two independent samples of the same distribution, in two coordinate systems,
with no correspondence between them and none to be given.

The method is Guy Dar, *mini-vec2vec: Scaling Universal Geometry Alignment with
Linear Transformations* (arXiv:2510.02348). Three stages:

1. Cluster both spaces independently and match the centroids by a quadratic
   assignment on their similarity **matrices** — absolute coordinates are
   exactly what differs between the two spaces, so the matrix is the only thing
   that can be compared. Repeat and pool, because one run's permutation can be
   wrong and the correct ones outvote it.
2. Describe every vector by its similarity to those anchors. Those relative
   representations *are* comparable across the spaces, so a nearest-neighbour
   search across them produces pseudo-pairs; solve orthogonal Procrustes on
   them.
3. Refine: iterative closest point in the target's own space, with exponential
   smoothing, then one clustering-based correction whose initialisation carries
   the correspondence for free.

It needs numpy, scipy and scikit-learn — all three already core dependencies —
and no torch, no faiss and no optimal-transport package. The Procrustes solve is
the same ``scipy.linalg.orthogonal_procrustes`` :mod:`rebasis.core.procrustes`
calls, and the preprocessing it asks for (centre, then ℓ2-normalise) is ADR 1's
default.

**What this is here for.** Not to fit adapters — nothing in ``fit`` calls it,
and the case it would unlock (an index that kept vectors and discarded the text)
cannot be measured on a corpus that still has its text. It is here because the
same arithmetic run *defensively* answers a question an index's owner has and no
tool answers: how alignable is my index. See :mod:`rebasis.probe.exposure`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.linalg
from scipy.optimize import quadratic_assignment
from sklearn.cluster import KMeans

from rebasis.compute.search import top_k_search
from rebasis.core.base import l2_normalize, pad_or_truncate
from rebasis.types import as_float32

if TYPE_CHECKING:
    from rebasis.types import FloatArray

__all__ = [
    "DEFAULTS",
    "Alignment",
    "Preprocessed",
    "align_unpaired",
    "preprocess",
]

#: Every hyperparameter the method has, at the values the paper reports.
#:
#: A dict rather than a dozen keyword arguments: they are one choice — "run the
#: published configuration" — and a caller who wants to vary one of them is
#: doing a measurement rather than using a tool. `spikes/unpaired_align.py` is
#: where that measurement lives.
DEFAULTS: dict[str, Any] = {
    #: Independent clusterings pooled into the anchor set.
    "runs": 5,
    #: Clusters per run. The QAP is solved in C x C rather than n x n, which is
    #: what makes the matching tractable at all.
    "clusters": 20,
    #: 2-opt climbs to the first local optimum, so one run is a lottery ticket.
    "qap_restarts": 30,
    #: Vectors k-means sees per run. Clustering a million rows five times over
    #: is the one place this could become slow, and the centroids of a sample
    #: are the centroids.
    "cluster_sample": 20_000,
    #: Neighbours averaged when forming a pseudo-pair. An average rather than
    #: the single nearest: the two sides share no document, so no source row has
    #: a correct match to find — only a neighbourhood of documents about the
    #: same thing, and picking one of them would be picking noise.
    "neighbours": 5,
    "refine1_iters": 30,
    "refine1_sample": 20_000,
    "refine2_clusters": 200,
    #: One correction, and only one: the paper measures a second making things
    #: slightly worse.
    "refine2_iters": 1,
    #: How far the smoothed map is allowed to drift from a rotation.
    "alpha": 0.3,
    #: ``gram`` is the author's notebook, ``cosine`` is the paper's text. They
    #: are different matrices and the QAP is not invariant to the difference;
    #: the default is what produced the published numbers.
    "kernel": "gram",
}


@dataclass(frozen=True, slots=True)
class Preprocessed:
    """Vectors in the geometry the method works in, plus how to get back."""

    hat: FloatArray
    mean: FloatArray
    #: Mean centred norm. The map lands on the unit sphere of the centred target
    #: space and the index is not stored there — it holds raw vectors around
    #: ``mean``, so something has to say how far from it to land. The only
    #: honest answer is how far the target documents themselves are.
    radius: float


@dataclass(slots=True)
class Alignment:
    """The learned map, and everything the method can say about itself."""

    rotation: FloatArray
    diagnostics: dict[str, Any] = field(default_factory=dict)


def preprocess(raw: FloatArray, width: int) -> Preprocessed:
    """Pad to a common width, centre, then project onto the unit sphere.

    Padding is what lets this run on a model pair whose dimensions differ, which
    the paper never has — all five of its encoders are 768-dimensional and
    ``orthogonal_procrustes`` refuses unequal shapes. Zeros preserve every inner
    product, so k-means, the cosine similarities and the Procrustes solve all
    see the geometry they would have seen without it.
    """
    padded = pad_or_truncate(as_float32(raw), width)
    mean = padded.mean(axis=0)
    centred = padded - mean
    radius = float(np.linalg.norm(centred, axis=1).mean())
    return Preprocessed(hat=l2_normalize(centred), mean=mean, radius=radius)


def align_unpaired(
    source: FloatArray, target: FloatArray, *, seed: int = 0, config: dict[str, Any] | None = None
) -> Alignment:
    """Fit a map from ``source``'s space into ``target``'s, given no pairs.

    The signature is the guarantee. This is handed two float arrays of different
    lengths and told nothing else — not an id, not a position, not which corpus
    either came from — so no correspondence is available to it even in
    principle. Both must already be through :func:`preprocess`.

    Args:
        source: Preprocessed vectors of the space being mapped **from**.
        target: Preprocessed vectors of the space being mapped **into**. It
            shares no document with ``source``, and having a different row count
            is normal rather than a problem.
        seed: Every stochastic step takes it. The reference implementation seeds
            nothing, which is why its own paper reports standard deviations over
            repeats; seeding does not remove the stochasticity, it makes it
            addressable.
        config: Overrides for :data:`DEFAULTS`.
    """
    settings = {**DEFAULTS, **(config or {})}
    diagnostics: dict[str, Any] = {}

    matched, anchor_diagnostics = _anchor_pairs(source, target, seed=seed, settings=settings)
    rotation = _procrustes(source, matched)
    diagnostics.update(anchor_diagnostics)

    rotation, trace = _refine_by_neighbours(source, target, rotation, seed=seed, settings=settings)
    diagnostics["refine_objective_final"] = round(trace[-1], 4) if trace else None
    # How far the smoothed matrix has drifted from a rotation. Zero for an
    # orthogonal one, and the geometry claim the whole method rests on is about
    # rotations — the paper says alpha controls this and never reports it.
    gram = as_float32(rotation.T @ rotation)
    drift = np.linalg.norm(gram - np.eye(gram.shape[0], dtype=np.float32))
    diagnostics["orthogonality_error"] = round(float(drift) / math.sqrt(len(gram)), 4)

    rotation, consistency = _refine_by_clusters(
        source, target, rotation, seed=seed, settings=settings
    )
    diagnostics["cluster_self_consistency"] = [round(v, 4) for v in consistency]
    return Alignment(rotation=rotation, diagnostics=diagnostics)


def _anchor_pairs(
    source: FloatArray, target: FloatArray, *, seed: int, settings: dict[str, Any]
) -> tuple[FloatArray, dict[str, Any]]:
    """Pseudo-pairs, from a correspondence nobody supplied.

    The conjecture the method rests on: two disjoint samples of one distribution
    cluster into the **same themes**, so the centroids are landmarks that exist
    in both spaces even though no document does.

    The ensemble is the noise dilution. One run's permutation can be wrong; it
    contributes ``clusters`` bad coordinates to a relative representation that
    has ``runs x clusters`` of them, and the correct ones outvote it. That is
    cheaper than the alternative in the literature, which is to re-initialise
    the whole pipeline hundreds of times and hope one run is clean.
    """
    rng = np.random.default_rng(seed * 104_729 + 11)
    anchors_source: list[FloatArray] = []
    anchors_target: list[FloatArray] = []
    scores: list[float] = []
    clusters = _usable_clusters(int(settings["clusters"]), source, target)

    for run in range(int(settings["runs"])):
        sample_size = int(settings["cluster_sample"])
        drawn_source = source[rng.permutation(len(source))[:sample_size]]
        drawn_target = target[rng.permutation(len(target))[:sample_size]]
        centres_source = _kmeans(drawn_source, clusters, seed=seed * 1000 + run).cluster_centers_
        centres_target = _kmeans(
            drawn_target, clusters, seed=seed * 1000 + run + 500
        ).cluster_centers_
        permutation, score = _quadratic_assignment(
            _kernel(centres_source, kind=str(settings["kernel"])),
            _kernel(centres_target, kind=str(settings["kernel"])),
            restarts=int(settings["qap_restarts"]),
            seed=seed * 1000 + run,
        )
        anchors_source.append(as_float32(centres_source))
        anchors_target.append(as_float32(centres_target)[permutation])
        scores.append(score)

    all_source = np.vstack(anchors_source)
    all_target = np.vstack(anchors_target)
    # The relative representation: each vector described by how close it is to
    # every anchor. Absolute coordinates differ between the two spaces; these do
    # not, which is what makes a nearest-neighbour search *across* them mean
    # anything at all.
    relative_source = l2_normalize(
        as_float32(l2_normalize(source, copy=True) @ l2_normalize(all_source).T)
    )
    relative_target = l2_normalize(
        as_float32(l2_normalize(target, copy=True) @ l2_normalize(all_target).T)
    )
    matched = _matched_mean(
        relative_source, relative_target, target, neighbours=int(settings["neighbours"])
    )
    return matched, {
        "clusters": clusters,
        # How *confident* the match is, never whether it is *right*: the
        # objective has symmetries and a wrong permutation can maximise it.
        # Reported for that reason and used for nothing.
        "qap_score_mean": round(float(np.mean(scores)), 4),
        "qap_score_min": round(float(np.min(scores)), 4),
    }


def _refine_by_neighbours(
    source: FloatArray,
    target: FloatArray,
    rotation: FloatArray,
    *,
    seed: int,
    settings: dict[str, Any],
) -> tuple[FloatArray, list[float]]:
    """Iterative closest point in the target's own space, with smoothing.

    One word separates this from the anchor stage: the neighbourhood is found in
    the target space itself rather than in the shared relative space. Once a
    coarse map exists, the mapped source vector is already close to where its
    match would be, and the target's own geometry is a better guide than a
    several-hundred-dimensional proxy for it.

    ``W <- (1-a)W + aW_new`` leaves the orthogonal manifold and is not projected
    back. That is the method rather than an oversight: alpha sets how far the
    map may drift from a rotation.
    """
    rng = np.random.default_rng(seed * 15_485_863 + 3)
    trace: list[float] = []
    sample_size = min(int(settings["refine1_sample"]), len(source))
    for _ in range(int(settings["refine1_iters"])):
        rows = rng.permutation(len(source))[:sample_size]
        drawn = source[rows]
        mapped = l2_normalize(as_float32(drawn @ rotation))
        matched = _matched_mean(mapped, target, target, neighbours=int(settings["neighbours"]))
        trace.append(float(np.sum(mapped * l2_normalize(matched, copy=True), axis=1).mean()))
        rotation = as_float32(
            (1.0 - settings["alpha"]) * rotation + settings["alpha"] * _procrustes(drawn, matched)
        )
    return rotation, trace


def _refine_by_clusters(
    source: FloatArray,
    target: FloatArray,
    rotation: FloatArray,
    *,
    seed: int,
    settings: dict[str, Any],
) -> tuple[FloatArray, list[float]]:
    """One clustering-based correction, and only one.

    The trick is the initialisation. Cluster the source, push the centroids
    through W, and use *those* as the starting centroids for clustering the
    target. k-means then only nudges them, so cluster *j* on the two sides is
    the same theme by construction — a correspondence obtained for free, over
    sets rather than over individual points, which is exactly where ICP's
    per-point matching is weakest.
    """
    clusters = max(2, min(int(settings["refine2_clusters"]), min(len(source), len(target)) // 8))
    consistency: list[float] = []
    for iteration in range(int(settings["refine2_iters"])):
        centres_source = as_float32(
            _kmeans(source, clusters, seed=seed * 2000 + iteration).cluster_centers_
        )
        transformed = as_float32(centres_source @ rotation)
        centres_target = as_float32(
            _kmeans(target, clusters, seed=0, init=transformed).cluster_centers_
        )
        consistency.append(
            float(
                np.sum(
                    l2_normalize(transformed, copy=True) * l2_normalize(centres_target, copy=True),
                    axis=1,
                ).mean()
            )
        )
        rotation = as_float32(
            (1.0 - settings["alpha"]) * rotation
            + settings["alpha"] * _procrustes(centres_source, centres_target)
        )
    return rotation, consistency


def _usable_clusters(requested: int, source: FloatArray, target: FloatArray) -> int:
    """Never more clusters than the smaller side has room for.

    k-means refuses more clusters than points, and a sample small enough to hit
    that is a sample the whole method cannot say anything about — so the number
    is lowered and the caller sees it in the diagnostics rather than meeting a
    scikit-learn exception.
    """
    return max(2, min(requested, min(len(source), len(target)) // 2))


def _kernel(centroids: FloatArray, *, kind: str) -> FloatArray:
    """The centroid-to-centroid similarity matrix the assignment is solved on."""
    if kind == "cosine":
        unit = l2_normalize(as_float32(centroids))
        return as_float32(unit @ unit.T)
    centred = as_float32(centroids) - as_float32(centroids).mean(axis=0)
    return as_float32(centred @ centred.T)


def _quadratic_assignment(
    kernel_a: FloatArray, kernel_b: FloatArray, *, restarts: int, seed: int
) -> tuple[np.ndarray, float]:
    """Match two similarity matrices under a permutation, best of ``restarts``.

    The score is the objective over ``||S_a||_F ||S_b||_F`` — the cosine between
    the two flattened matrices, which is the paper's own reading of it.
    """
    best_permutation: np.ndarray | None = None
    best = -math.inf
    for restart in range(restarts):
        result = quadratic_assignment(
            kernel_a,
            kernel_b,
            method="2opt",
            options={"maximize": True, "rng": np.random.default_rng(seed * 1_000_003 + restart)},
        )
        if result.fun > best:
            best, best_permutation = float(result.fun), result.col_ind
    if best_permutation is None:  # pragma: no cover - `restarts` is always >= 1
        message = "the assignment was asked for zero restarts"
        raise ValueError(message)
    scale = float(np.linalg.norm(kernel_a) * np.linalg.norm(kernel_b))
    return best_permutation, (best / scale if scale > 0 else 0.0)


def _kmeans(x: FloatArray, n_clusters: int, *, seed: int, init: FloatArray | None = None) -> Any:
    """k-means with one initialisation, seeded so a run can be replayed."""
    if init is not None:
        return KMeans(n_clusters=n_clusters, init=np.asarray(init, dtype=np.float64), n_init=1).fit(
            x
        )
    return KMeans(n_clusters=n_clusters, n_init=1, random_state=seed).fit(x)


def _procrustes(source: FloatArray, target: FloatArray) -> FloatArray:
    """``min ||source.W - target||`` over orthogonal W. The same call `core` makes."""
    rotation, _ = scipy.linalg.orthogonal_procrustes(as_float32(source), as_float32(target))
    return as_float32(rotation)


def _matched_mean(
    source: FloatArray, neighbourhood: FloatArray, pool: FloatArray, *, neighbours: int
) -> FloatArray:
    """Send each ``source`` row to the mean of its nearest rows in ``pool``.

    ``neighbourhood`` is where the neighbours are *found* and ``pool`` is what
    gets averaged. They are the same array in the ambient refinement and
    different arrays in the anchor stage, where the neighbourhood comes from the
    shared relative space and the averaging has to happen in the absolute one.
    """
    indices, _ = top_k_search(
        l2_normalize(as_float32(source), copy=True),
        as_float32(neighbourhood),
        k=min(neighbours, len(pool)),
    )
    return as_float32(pool[indices].mean(axis=1))
