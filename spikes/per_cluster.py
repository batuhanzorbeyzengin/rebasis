"""Does one adapter per cluster beat one adapter for the whole corpus?

``ROADMAP.md``, under "Beyond 0.3": *"One global map is leaving quality on the
table where drift is heterogeneous. ``probe`` already reports ``tail_arr`` to
detect that; it cannot yet do anything about it."* That is a claim, and nothing
in this repository has measured it. ``probe`` already clusters its sample, already
reports the sparsest decile's retention, and already warns when the gap to the
overall figure exceeds ``TAIL_GAP_LIMIT`` — and then has nothing to offer. This
spike asks whether there is anything to offer.

The one thing it does **not** lean on is a published result. arXiv:2509.23471,
section 6 and appendix A.4, reports two class-routed MLPs at 0.94 against one
global MLP at 0.85 — but the drift in that experiment was **synthesised**, half
the classes by an affine transform and half by a non-linear warp, and the paper
calls it "a small-scale experiment" and "preliminary". Two routed adapters
beating one global adapter on drift that was built to be two-mode is close to a
tautology. It motivates the direction; it is not evidence for it. Everything
below is measured on rebasis' own corpora, where whatever heterogeneity exists is
whatever the corpus and the two models put there.

Four things decide whether the answer means anything, and each is a column in the
output rather than an assumption:

**1. The fit budget.** ADR 10 measured the fit curve flat by 4,000 pairs. Split
across k clusters that is 4000/k each, and a per-cluster adapter that loses on
500 pairs is indistinguishable from a global adapter starved of data. So both
arms run. ``split`` gives the k adapters 4,000 pairs *between them* — the honest
question a user faces, since the fit budget is what it is. ``full`` gives each
cluster up to 4,000 of its own, which is k times the data and isolates whether
the *shape* helps at all. In both arms the global control is fitted on exactly
the union the per-cluster arm used, so the only difference between the two rows
is one map against k maps.

**2. Routing is measured, not assumed.** A query arrives with no cluster label.
Two routers are measured — ``centroid_new``, nearest cluster centroid in the new
model's space, one extra k×d product; and ``bridge_old``, map the query with the
global adapter first and take the nearest centroid in the old space where the
clustering actually happened, which costs a second full matrix multiply. The
oracle row uses the query's true label and is a **bound, not a result**, in the
sense ``ceiling_old_space`` is one in ``tools/bridge_band.py``: nothing a user
can run produces it.

**2a. Two shapes, not one.** ``replace`` is the arrangement the roadmap
describes: the cluster's map *is* the map. ``residual`` puts the global map in
front of it, so the 4,000 shared pairs still do work in every cluster and each
cluster map only has to correct what is left. The second is strictly the more
capable and strictly the more expensive, and it is measured so that a negative
result cannot be answered with "you only tried the naive one".

**3. ``tail_arr``, not only the mean.** The roadmap's claim is about
*heterogeneous* drift specifically. Every arm reports the sparsest decile's
retention beside the overall figure, every row carries the gap ``probe`` itself
reported for that corpus and model pair, and ``by_cluster`` breaks the difference
down region by region against what each region was fitted on — because an adapter
that lifts the dense clusters and sinks the sparse ones has the same mean as one
that does nothing.

**4. Cost, on both paths.** k orthogonal maps are k closed-form solves and stay
cheap; serving may not. ADR 11 puts the hot-path budget at 15/20/30/40 µs for
d=256/384/768/1024, and routing adds a centroid comparison before the map — plus
k weight matrices where there was one, which is the part that does not fit in
cache. Latency is measured single-query on CPU with the global adapter
interleaved as the control, because ADR 11's absolute numbers belong to one
reference host and only the relative figure travels.

The measurement runs on real machinery throughout: ``stratified_sample`` and
``split_disjoint`` build the sample, ``_allocate`` splits the fit budget the way
the sampler splits a sample budget, ``build_tier0`` builds the ground truth,
``fit_candidates`` fits every adapter and ``evaluate_candidate`` scores every arm.
``probe_store`` runs once per corpus and model pair, unchanged, to supply the
condition. Corpora, the model ladders and the cached loader come from
``tools/bridge_band.py`` — imported, never edited: seven read-only names
(``CORPORA``, ``LADDERS``, ``Corpus``, ``Encoded``, ``load_corpus``,
``encode_corpus``, ``resolve_corpora``), taken rather than reimplemented because
the warm embedding cache is keyed by that module's own naming convention and a
second implementation of the key would write vectors nobody else can find.

One corpus name this file adds: ``mix:a+b+c`` reads several collections as one
index. A single BEIR collection is one topic by construction — a cqadupstack
forum is one StackExchange site — and the roadmap's claim is about corpora that
are not. The mixture is assembled; **the drift is not**, which is the whole
difference between this and the published routed experiment.

**No document limit.** The shared cache at ``~/band-cache`` is keyed on the corpus
name, so a truncated run poisons the full-corpus vectors for every later run.
This spike has no ``--limit-docs`` and takes the corpus as it comes.

    .venv/bin/python spikes/per_cluster.py --corpus heldout --ladder default \\
        --clusters 8,16,64 --out reports/per-cluster/rows.json

``--survey`` stops after the ``probe_store`` pass, which is how the corpora were
chosen: it reports ``arr``, ``tail_arr`` and the gap and fits nothing else.

Numbers, not adjectives: whatever it prints is what goes in the report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Fit budget the tool ships with. ADR 10: six times this buys 0.005-0.025, and
#: `procrustes_centered` won 15 fits out of 15 — which is why it is the only
#: method fitted here. Varying the adapter family as well as the shape would
#: leave a difference with two candidate causes.
FIT_PAIRS = 4000

#: Held-out query proxies. `probe`'s own default, and the count M0 measured the
#: ±0.024 ARR confidence interval on.
HELDOUT = 1000

#: The single adapter family both arms use.
METHOD = "procrustes_centered"

#: Fewest pairs a cluster needs before an orthogonal Procrustes fit is defined at
#: all. Below it the cluster falls back to the global map and the row says how
#: many clusters did — a fallback that is counted is a measurement; one that is
#: silent is a different experiment.
MIN_CLUSTER_FIT = 2

#: ADR 11's per-dimension hot-path budget, microseconds. Interpolated nowhere:
#: a dimension not in this table is reported without a budget rather than
#: against a number nobody measured.
HOT_PATH_BUDGET_US: dict[int, float] = {256: 15.0, 384: 20.0, 768: 30.0, 1024: 40.0}

#: Single-query latency samples per variant. Enough that the median is stable
#: and the whole block still costs under a second.
LATENCY_REPEATS = 4000

#: Every shape-and-router combination measured, in the order the run prints
#: them. ``replace`` is the arrangement the roadmap describes; ``residual`` puts
#: the global map in front of it. ``oracle`` is the query's true cluster and is a
#: **bound, not a result** — nothing a user can run produces it.
ROUTERS = (
    "replace@centroid_new",
    "replace@bridge_old",
    "replace@oracle",
    "residual@bridge_old",
    "residual@oracle",
)


def band_module() -> Any:
    """Import ``tools/bridge_band.py`` for its corpora, ladders and warm cache.

    A read-only coupling to seven names. It is taken rather than reimplemented
    because ``embed_cached`` owns the layout of the shared ``~/band-cache``, and
    a second implementation of that key would either miss the cache or write
    vectors under a name nothing else reads.
    """
    tools = Path(__file__).resolve().parent.parent / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    try:
        import bridge_band
    except ImportError as exc:  # pragma: no cover - a broken checkout, not a case
        msg = f"cannot import tools/bridge_band.py from {tools}: {exc}"
        raise RuntimeError(msg) from exc
    return bridge_band


# ── the routed adapter ────────────────────────────────────────────────


class RoutedAdapter:
    """k maps and a router. **Not a shipped adapter** — a measurement object.

    It carries just enough of the adapter surface for
    :func:`rebasis.probe.runner.evaluate_candidate` to score it exactly the way
    it scores a real one: ``apply``, ``n_params`` and ``type_name``. Nothing
    serialises it and nothing in ``src/`` knows it exists, which is deliberate —
    what is being tested is whether the shape is worth building, and building it
    first would answer the question by assuming it.

    ``trunk`` is the difference between the two shapes measured here. Without
    one, each cluster map replaces the global map outright and sees the raw
    new-model query — the arrangement the roadmap describes. With one, the global
    map runs first and each cluster map is a *correction* to its output, so no
    cluster is ever worse off than the shared map by more than its own fit error
    and the 4,000 pairs are not thrown away. The residual shape is strictly the
    more capable of the two and costs a second full matrix multiply per query; it
    is here so that a negative result cannot be answered with "you only measured
    the naive one".

    ``labels`` bypasses the router with the query's true cluster. That row is a
    bound and is reported under a name that says so.
    """

    kind = "per_cluster"

    def __init__(
        self,
        adapters: Sequence[Any],
        *,
        router: str,
        centroids: np.ndarray | None = None,
        gate: Any = None,
        labels: np.ndarray | None = None,
        trunk: Any = None,
    ) -> None:
        self._adapters = list(adapters)
        self._router = router
        self._centroids = centroids
        self._gate = gate
        self._labels = labels
        self._trunk = trunk
        first = self._adapters[0]
        self.input_dim = int(first.input_dim if trunk is None else trunk.input_dim)
        self.output_dim = int(first.output_dim)
        self.type_name = ("residual" if trunk is not None else "per_cluster") + f"@{router}"

    def _probe(self, rows: np.ndarray) -> np.ndarray:
        """The vectors the cluster maps see: the query, or the trunk's output."""
        if self._trunk is None:
            return rows
        from rebasis.compute.arrays import l2_normalize

        return l2_normalize(self._trunk.apply(rows), copy=False)

    def _slots(self, rows: np.ndarray) -> np.ndarray:
        """Which adapter each already-trunked row is sent to."""
        if self._router == "oracle":
            if self._labels is None:
                msg = "oracle routing needs labels"
                raise RuntimeError(msg)
            return self._labels
        if self._centroids is None:
            msg = f"router {self._router} needs centroids"
            raise RuntimeError(msg)
        # Unnormalised on purpose where a gate is involved: the probe's norm is
        # the same against every centroid, so it cannot move the argmax, and a
        # square root on the hot path would be charged to an arrangement that
        # does not need it.
        probe = rows if self._gate is None else self._gate.apply(rows)
        return np.asarray(np.argmax(probe @ self._centroids.T, axis=-1), dtype=np.int64)

    def route(self, x: np.ndarray) -> np.ndarray:
        """Which adapter each query is sent to — for the routing-accuracy figure."""
        return self._slots(self._probe(np.atleast_2d(np.asarray(x, dtype=np.float32))))

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Route each row, then map it with that cluster's adapter.

        The trunk runs once and its output is both what is routed on and what is
        mapped, because that is what a server would do — computing it twice would
        charge the residual shape for a matrix multiply it does not need.

        The single-row case is the serving case and takes the direct path. Going
        through the grouping below for one query would charge the arrangement for
        a ``np.unique`` no server would run, and the hot-path measurement is the
        whole reason the shape might not be worth having.
        """
        rows = self._probe(np.atleast_2d(np.asarray(x, dtype=np.float32)))
        slots = self._slots(rows)
        if rows.shape[0] == 1:
            return self._adapters[int(slots[0])].apply(rows)
        out = np.empty((rows.shape[0], self.output_dim), dtype=np.float32)
        for slot in np.unique(slots):
            mask = slots == slot
            out[mask] = self._adapters[int(slot)].apply(rows[mask])
        return out

    def state_dict(self) -> dict[str, np.ndarray]:
        """Every distinct map's weights, the trunk's, and the router's centroids.

        Distinct by identity: a cluster too small to fit shares another map
        rather than holding a copy of it, and counting that copy would inflate
        the memory figure for an arrangement nobody would build that way.
        """
        distinct: dict[int, Any] = {}
        for adapter in ([self._trunk] if self._trunk is not None else []) + self._adapters:
            distinct.setdefault(id(adapter), adapter)
        state: dict[str, np.ndarray] = {}
        for i, adapter in enumerate(distinct.values()):
            for name, value in adapter.state_dict().items():
                state[f"{i}.{name}"] = value
        if self._centroids is not None:
            state["centroids"] = self._centroids
        return state

    def n_params(self) -> int:
        """Total parameter count, router included.

        The centroids are counted because they are weights a server holds and
        reads on every query. Leaving them out would make the routed arrangement
        look free of the one thing that distinguishes it.
        """
        return int(sum(v.size for v in self.state_dict().values()))


# ── the sample ────────────────────────────────────────────────────────


def build_sample(
    ids: list[str],
    texts: list[str],
    old_documents: np.ndarray,
    *,
    size: int | None,
    heldout: int,
    seed: int,
    n_clusters: int | None,
) -> tuple[Any, np.ndarray, int]:
    """Draw a stratified sample and split it into queries and fit pairs.

    ``draw_corpus_sample`` would do this, and does it from a store — but it
    offers no way to set the cluster count, which is the axis this spike varies.
    So the two functions it calls are called here directly: ``stratified_sample``
    for the clustering and the proportional draw, ``split_disjoint`` for the
    query/fit split with its runtime leakage check. The ``CorpusSample`` that
    comes out is the same object ``probe_store`` accepts.

    The pool is capped at ``CLUSTER_POOL_MAX`` because that is the tool's memory
    invariant, and a measurement taken outside it would be a measurement of
    something the tool does not do.

    Returns the sample, the corpus row each sample position came from, and the
    cluster count actually used.
    """
    from rebasis.compute.arrays import l2_normalize
    from rebasis.probe.session import CLUSTER_POOL_MAX, CorpusSample
    from rebasis.sample.strategies import (
        SampleResult,
        split_disjoint,
        stratified_sample,
        suggested_cluster_count,
    )

    rng = np.random.default_rng(seed)
    n_total = len(ids)
    pool = np.sort(rng.choice(n_total, size=min(n_total, CLUSTER_POOL_MAX), replace=False))
    pool_vectors = l2_normalize(np.ascontiguousarray(old_documents[pool]))

    k = n_clusters or suggested_cluster_count(int(pool.size))
    drawn = stratified_sample(
        pool_vectors, min(size or int(pool.size), int(pool.size)), seed=seed, n_clusters=k
    )
    positions = drawn.indices
    labels = drawn.cluster_labels[positions] if drawn.cluster_labels is not None else None
    rows = pool[positions]

    placeholder = SampleResult(
        indices=np.arange(positions.size, dtype=np.int64), strategy="stratified", seed=seed
    )
    query_positions, fit_positions = split_disjoint(placeholder, heldout, seed=seed)

    sample = CorpusSample(
        ids=[ids[int(r)] for r in rows],
        texts=[texts[int(r)] for r in rows],
        old_vectors=pool_vectors[positions],
        query_positions=query_positions,
        fit_positions=fit_positions,
        n_total=n_total,
        strategy="stratified",
        seed=seed,
        pool_size=int(pool.size),
        cluster_labels=labels,
    )
    return sample, rows, k


# ── the arms ──────────────────────────────────────────────────────────


def allocate_budget(
    labels: np.ndarray, budget: int, k: int, *, per_cluster: bool
) -> dict[int, int]:
    """How many fit pairs each cluster gets.

    ``per_cluster`` is the ``full`` arm: every cluster takes ``budget`` of its
    own, capped by what it actually holds. That cap is not a detail — a corpus
    does not owe the sparsest decile 4,000 documents, and the row records what
    each cluster actually got.

    Otherwise the ``budget`` is split across clusters by
    :func:`rebasis.sample.strategies._allocate` — the shipped proportional draw
    with a floor, reused rather than restated so that the split is the one the
    sampler would make.
    """
    if per_cluster:
        counts = {int(c): int((labels == c).sum()) for c in range(k)}
        return {c: min(budget, n) for c, n in counts.items() if n > 0}

    from rebasis.sample.strategies import _allocate

    return _allocate(labels, budget, k)


def fit_one(src: np.ndarray, dst: np.ndarray) -> Any:
    """One adapter of the family both shapes use, through ``fit_candidates``."""
    from rebasis.core import fit_candidates

    return fit_candidates(src, dst, normalize=False, methods=[METHOD])[0].adapter


class Arm:
    """One fit budget, fitted every way the comparison needs.

    Holds the global control and the k cluster maps together because the point
    of the comparison is that they were fitted on the same pairs: the global
    adapter takes the **union** of what the k clusters were given. Fitting it on
    a separately drawn 4,000 would leave a difference with two causes.

    Two per-cluster shapes come out of the same pairs:

    ``replace``   ``g_c(x_new)`` — the cluster map is the whole map, which is
                  the arrangement the roadmap describes.
    ``residual``  ``r_c(G(x_new))`` — the global map runs first and the cluster
                  map corrects it, so the shared 4,000 pairs are still doing
                  work in every cluster.

    A cluster with too few pairs to fit falls back: to the global map under
    ``replace``, and to the global *residual* — the same correction fitted on
    the union, which is near-identity by construction — under ``residual``.
    """

    def __init__(
        self,
        new_documents: np.ndarray,
        old_documents: np.ndarray,
        fit_positions: np.ndarray,
        labels: np.ndarray,
        *,
        k: int,
        budget: int,
        per_cluster: bool,
        seed: int,
    ) -> None:
        from rebasis.compute.arrays import l2_normalize

        rng = np.random.default_rng(seed + 1)
        fit_labels = labels[fit_positions]
        allocation = allocate_budget(fit_labels, budget, k, per_cluster=per_cluster)

        chosen: dict[int, np.ndarray] = {}
        for cluster, quota in allocation.items():
            members = fit_positions[fit_labels == cluster]
            take = min(int(quota), int(members.size))
            if take > 0:
                chosen[cluster] = rng.choice(members, size=take, replace=False)

        self.pairs = {c: int(v.size) for c, v in chosen.items()}
        self.union = np.sort(np.concatenate(list(chosen.values()))) if chosen else fit_positions[:0]
        self.global_adapter = fit_one(new_documents[self.union], old_documents[self.union])

        # The trunk's output on every fit pair, computed once: it is the input
        # side of every residual fit below. Only the pairs that were drawn — the
        # held-out queries must never reach a fit, and computing them would be
        # both wasted and a leak waiting to happen.
        bridged = l2_normalize(self.global_adapter.apply(new_documents[self.union]), copy=False)
        global_residual = fit_one(bridged, old_documents[self.union])

        started = time.perf_counter()
        self.adapters: list[Any] = []
        self.residuals: list[Any] = []
        self.fallbacks = 0
        for cluster in range(k):
            pick = chosen.get(cluster)
            if pick is None or pick.size < MIN_CLUSTER_FIT:
                self.adapters.append(self.global_adapter)
                self.residuals.append(global_residual)
                self.fallbacks += 1
                continue
            # `union` is sorted and holds every drawn pair, so `searchsorted`
            # turns a sample position into the row `bridged` put it in.
            rows = np.searchsorted(self.union, pick)
            self.adapters.append(fit_one(new_documents[pick], old_documents[pick]))
            self.residuals.append(fit_one(bridged[rows], old_documents[pick]))
        self.fit_seconds = time.perf_counter() - started
        self.fitted = k - self.fallbacks

    def summary(self) -> dict[str, Any]:
        """What each cluster was actually given."""
        sizes = list(self.pairs.values()) or [0]
        return {
            "total_fit_pairs": int(self.union.size),
            "per_cluster_min": int(min(sizes)),
            "per_cluster_median": int(np.median(sizes)),
            "per_cluster_max": int(max(sizes)),
            "clusters_fitted": int(self.fitted),
            "clusters_fallback_to_global": int(self.fallbacks),
            "fit_seconds_both_shapes": round(self.fit_seconds, 2),
        }


def centroids_of(vectors: np.ndarray, labels: np.ndarray, k: int, dim: int) -> np.ndarray:
    """Unit-norm cluster centroids, in whatever space ``vectors`` are in.

    An empty cluster gets a zero row, which scores zero against every query and
    so is never routed to — the right behaviour, and it needs no special case at
    the argmax.
    """
    from rebasis.compute.arrays import l2_normalize

    centroids = np.zeros((k, dim), dtype=np.float32)
    for cluster in range(k):
        members = vectors[labels == cluster]
        if members.size:
            centroids[cluster] = members.mean(axis=0)
    return l2_normalize(centroids)


def score(
    adapter: Any,
    *,
    new_queries: np.ndarray,
    old_documents: np.ndarray,
    ground_truth: Any,
    k: int,
    query_clusters: np.ndarray | None,
) -> dict[str, Any]:
    """Measure one arm through ``evaluate_candidate`` — the real scorer.

    CSLS is off everywhere. It is a search-time bias rather than part of the
    map, M0 measured it adding +0.103 to weak adapters and costing −0.045 on
    strong ones, and letting it vary between the arms would put a second
    difference into a comparison that has one.
    """
    from rebasis.core.selection import AdapterCandidate
    from rebasis.probe.runner import evaluate_candidate

    candidate = AdapterCandidate(method=METHOD, adapter=adapter, fit_seconds=0.0)
    metrics = evaluate_candidate(
        candidate,
        query_vectors_new=new_queries,
        old_doc_vectors=old_documents,
        ground_truth=ground_truth,
        k=k,
        csls_sample=None,
        query_clusters=query_clusters,
        cascade_k=None,
    )
    return {
        "arr": round(metrics.arr, 4),
        "arr_ci": [round(v, 4) for v in metrics.arr_ci],
        "tail_arr": None if metrics.tail_arr is None else round(metrics.tail_arr, 4),
        "ndcg": round(metrics.ndcg, 4),
        "n_params": int(metrics.n_params),
        "memory_mb": round(metrics.n_params * 4 / 1e6, 2),
    }


def per_query_recall(
    adapter: Any,
    *,
    new_queries: np.ndarray,
    old_documents: np.ndarray,
    ground_truth: Any,
    k: int,
) -> np.ndarray:
    """Per-query recall for one arm, for the by-cluster breakdown.

    The same search ``evaluate_candidate`` runs, repeated because it keeps the
    means and not the array. Cheap beside the embedding, and it is what turns
    "per-cluster wins" into "per-cluster wins *here*".
    """
    from rebasis.compute.arrays import l2_normalize
    from rebasis.probe.metrics import recall_per_query, top_k_search

    mapped = l2_normalize(adapter.apply(new_queries), copy=False)
    indices, _ = top_k_search(mapped, old_documents, k=k, self_mask=ground_truth.self_mask)
    return recall_per_query(indices[:, :k], ground_truth.relevant_sparse, k)


def cluster_detail(
    labels: np.ndarray,
    query_labels: np.ndarray,
    pairs: dict[int, int],
    *,
    global_recall: np.ndarray,
    routed_recall: np.ndarray,
    oracle: float,
    k: int,
) -> list[dict[str, Any]]:
    """Retention per cluster, on ARR's scale, beside what it was fitted on.

    This is what answers "and where?", and it is ordered sparsest first because
    that is the end of the distribution ``tail_arr`` is about.
    """
    rows: list[dict[str, Any]] = []
    for cluster in range(k):
        mask = query_labels == cluster
        n = int(mask.sum())
        if n == 0:
            continue
        base = float(global_recall[mask].mean())
        routed = float(routed_recall[mask].mean())
        rows.append(
            {
                "cluster": cluster,
                "n_documents": int((labels == cluster).sum()),
                "n_fit_pairs": int(pairs.get(cluster, 0)),
                "n_queries": n,
                "global": round(base / oracle, 4),
                "per_cluster": round(routed / oracle, 4),
                "delta": round((routed - base) / oracle, 4),
            }
        )
    return sorted(rows, key=lambda r: r["n_documents"])


# ── the hot path ──────────────────────────────────────────────────────


def measure_latency(
    global_adapter: Any,
    routers: dict[str, RoutedAdapter],
    queries: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Single-query latency for the global map and each routed arrangement.

    On CPU, one query at a time, in a random order. The order matters: k weight
    matrices do not fit where one does, and measuring the same cluster twice in
    a row would report a cache state no server sees. The global adapter is
    measured in the same loop as the control, because ADR 11's absolute figures
    belong to one reference host and only the ratio travels.

    The whole hot path is timed — the map and the ℓ2 normalisation after it —
    because that is what ADR 11's budget is a budget for.
    """
    from rebasis.compute.arrays import l2_normalize

    rng = np.random.default_rng(seed + 2)
    order = rng.integers(0, queries.shape[0], size=repeats)
    rows = [np.ascontiguousarray(queries[int(i)][None, :]) for i in order]

    def timed(call: Any) -> dict[str, float]:
        for row in rows[:64]:
            call(row)
        samples = np.empty(repeats, dtype=np.float64)
        for i, row in enumerate(rows):
            started = time.perf_counter_ns()
            call(row)
            samples[i] = time.perf_counter_ns() - started
        return {
            "median_us": round(float(np.median(samples)) / 1000, 2),
            "p95_us": round(float(np.percentile(samples, 95)) / 1000, 2),
        }

    measured = {"global": timed(lambda x: l2_normalize(global_adapter.apply(x), copy=False))}
    for name, routed in routers.items():
        measured[name] = timed(lambda x, r=routed: l2_normalize(r.apply(x), copy=False))

    dim = int(global_adapter.input_dim)
    return {
        "host": "cpu, single query, interleaved with the global control",
        "input_dim": dim,
        "output_dim": int(global_adapter.output_dim),
        "adr11_budget_us": HOT_PATH_BUDGET_US.get(dim),
        "variants": measured,
        "overhead_us": {
            name: round(value["median_us"] - measured["global"]["median_us"], 2)
            for name, value in measured.items()
            if name != "global"
        },
    }


# ── one corpus, one model pair ────────────────────────────────────────


def probe_condition(
    corpus: Any,
    old: Any,
    new: Any,
    *,
    sample_size: int,
    heldout: int,
    seed: int,
    device: str,
    methods: Sequence[str] | None,
) -> dict[str, Any]:
    """Run ``probe_store`` unchanged and report the gap it finds.

    This is the selection criterion, and it is measured by the tool rather than
    by this file: ``arr_at_k − tail_arr`` against ``TAIL_GAP_LIMIT``. It draws
    its own sample at ``probe``'s own defaults, so the cluster count is the one
    ``suggested_cluster_count`` picks and the number is the one a user sees.
    """
    from rebasis.embed import PrecomputedEmbedder
    from rebasis.probe.decision import TAIL_GAP_LIMIT
    from rebasis.probe.session import probe_store
    from rebasis.store import MemoryStore

    store = MemoryStore(corpus.doc_ids, old.documents, corpus.doc_texts)
    documents = dict(zip(corpus.doc_texts, new.documents, strict=True))
    queries = dict(documents)
    if new.documents_as_queries is not None:
        queries.update(zip(corpus.doc_texts, new.documents_as_queries, strict=True))
    embedder = PrecomputedEmbedder(new.profile, documents, query_vectors=queries)

    started = time.perf_counter()
    result, sample = probe_store(
        store,
        embedder,
        size=sample_size,
        heldout=heldout,
        k=10,
        seed=seed,
        device=device,
        methods=methods,
        with_csls=True,
    )
    tail = result.best.tail_arr
    labels = sample.cluster_labels
    return {
        "decision": result.decision.decision,
        "adapter": result.best.name,
        "arr": round(result.best.arr, 4),
        "tail_arr": None if tail is None else round(tail, 4),
        "tail_gap": None if tail is None else round(result.best.arr - tail, 4),
        "tail_gap_limit": TAIL_GAP_LIMIT,
        "heterogeneous": None if tail is None else bool(result.best.arr - tail > TAIL_GAP_LIMIT),
        "n_clusters": 0 if labels is None else int(np.unique(labels).size),
        "n_fit_pairs": int(result.n_fit_pairs),
        "n_queries": int(result.n_queries),
        "seconds": round(time.perf_counter() - started, 1),
    }


def head_to_head(
    sample: Any,
    new_documents: np.ndarray,
    new_as_queries: np.ndarray,
    *,
    k_clusters: int,
    metric_k: int,
    seed: int,
    fit_pairs: int,
    device: str,
) -> dict[str, Any]:
    """Global against per-cluster, at both budgets, with both routers.

    Everything is scored against one ground truth, over one document set, with
    one query set. The ground truth is built the way ``probe_store``'s T0 tier
    builds it, from the same function.
    """
    from rebasis.compute import resolve_device, using_device
    from rebasis.probe.groundtruth import build_tier0

    old_documents = sample.old_vectors
    query_positions = sample.query_positions
    fit_positions = sample.fit_positions
    labels = sample.cluster_labels
    query_labels = sample.clusters_of(query_positions)
    new_queries = new_as_queries[query_positions]

    with using_device(resolve_device(device)):
        ground_truth = build_tier0(new_documents, new_queries, query_positions, k=metric_k)
    oracle = ground_truth.oracle_recall or 1.0

    arms: dict[str, dict[str, Any]] = {}
    detail: dict[str, list[dict[str, Any]]] = {}
    routing: dict[str, dict[str, float]] = {}
    budgets: dict[str, Any] = {}
    latency: dict[str, Any] = {}

    for name, per_cluster in (("split", False), ("full", True)):
        arm = Arm(
            new_documents,
            old_documents,
            fit_positions,
            labels,
            k=k_clusters,
            budget=fit_pairs,
            per_cluster=per_cluster,
            seed=seed,
        )
        union_labels = labels[arm.union]
        new_centroids = centroids_of(
            new_documents[arm.union], union_labels, k_clusters, new_documents.shape[1]
        )
        old_centroids = centroids_of(
            old_documents[arm.union], union_labels, k_clusters, old_documents.shape[1]
        )
        routers = {
            # The map replaces the global one and sees the raw query.
            "replace@centroid_new": RoutedAdapter(
                arm.adapters, router="centroid_new", centroids=new_centroids
            ),
            "replace@bridge_old": RoutedAdapter(
                arm.adapters,
                router="bridge_old",
                centroids=old_centroids,
                gate=arm.global_adapter,
            ),
            "replace@oracle": RoutedAdapter(arm.adapters, router="oracle", labels=query_labels),
            # The map corrects the global one. Routing on the trunk's output is
            # free here — it is already computed — which is why this shape gets
            # the better router without paying for it twice.
            "residual@bridge_old": RoutedAdapter(
                arm.residuals,
                router="bridge_old",
                centroids=old_centroids,
                trunk=arm.global_adapter,
            ),
            "residual@oracle": RoutedAdapter(
                arm.residuals, router="oracle", labels=query_labels, trunk=arm.global_adapter
            ),
        }

        scoring = {
            "new_queries": new_queries,
            "old_documents": old_documents,
            "ground_truth": ground_truth,
            "k": metric_k,
        }
        with using_device(resolve_device(device)):
            arms[f"global_{name}"] = score(
                arm.global_adapter, query_clusters=query_labels, **scoring
            )
            for router, routed in routers.items():
                arms[f"{name}_{router}"] = score(routed, query_clusters=query_labels, **scoring)
            global_recall = per_query_recall(arm.global_adapter, **scoring)
            routed_recall = per_query_recall(routers["replace@centroid_new"], **scoring)

        detail[name] = cluster_detail(
            labels,
            query_labels,
            arm.pairs,
            global_recall=global_recall,
            routed_recall=routed_recall,
            oracle=oracle,
            k=k_clusters,
        )
        routing[name] = {
            router: round(float((routed.route(new_queries) == query_labels).mean()), 4)
            for router, routed in routers.items()
            if not router.endswith("oracle")
        }
        budgets[name] = arm.summary()
        if per_cluster:
            latency = measure_latency(
                arm.global_adapter,
                {r: a for r, a in routers.items() if not r.endswith("oracle")},
                new_queries,
                repeats=LATENCY_REPEATS,
                seed=seed,
            )

    return {
        "k_clusters": k_clusters,
        "n_documents": int(old_documents.shape[0]),
        "n_queries": int(query_positions.size),
        "n_fit_available": int(fit_positions.size),
        "oracle_recall": round(float(oracle), 4),
        "budgets": budgets,
        "routing_accuracy": routing,
        "arms": arms,
        "by_cluster": detail,
        "latency": latency,
    }


#: Prefix for a corpus that is several corpora read as one index —
#: ``mix:beir/cqadupstack/android+beir/cqadupstack/mathematica``.
#:
#: It exists because the roadmap's claim is about *heterogeneous* drift, and a
#: single BEIR collection is one topic by construction: a cqadupstack forum is
#: one StackExchange site. A vault that grew by department is not, and this is the
#: closest thing to one that can be assembled out of corpora already measured.
#:
#: Note what is and is not synthesised here. The **corpus** is assembled; the
#: **drift** is not — it is whatever the two real models do to real text, which
#: is exactly the thing arXiv:2509.23471's routed experiment could not claim. If
#: per-cluster adapters cannot win where the index is literally several
#: unrelated domains, the condition is not the reason they are not winning.
MIX_PREFIX = "mix:"


def mix_members(dataset: str) -> list[str]:
    """The corpora behind a name — one of them, unless it is a mixture."""
    if not dataset.startswith(MIX_PREFIX):
        return [dataset]
    return [part for part in dataset.removeprefix(MIX_PREFIX).split("+") if part]


def combine(band: Any, dataset: str, parts: list[Any]) -> Any:
    """Read several corpora as one index.

    Document ids are prefixed with the member's position so two forums cannot
    collide on a shared id, and the query set is empty: a mixture is scored at T0
    against the new model's own neighbours, which is the tier ``probe`` defaults
    to and the only one that has a ``tail_arr`` worth reading.
    """
    if len(parts) == 1:
        return parts[0]
    ids: list[str] = []
    texts: list[str] = []
    for position, part in enumerate(parts):
        ids.extend(f"{position}:{doc_id}" for doc_id in part.doc_ids)
        texts.extend(part.doc_texts)
    return band.Corpus(
        name=dataset, doc_ids=ids, doc_texts=texts, query_ids=[], query_texts=[], qrels={}
    )


def encode_parts(band: Any, parts: list[Any], model_id: str, **shared: Any) -> Any:
    """Encode every member and stack the results in member order.

    Each member goes through ``encode_corpus`` under its own name, so a mixture
    reads the same warm ``.npy`` files a plain run of that corpus reads and
    writes no new cache entry of its own.
    """
    encoded = [band.encode_corpus(corpus=part, model_id=model_id, **shared) for part in parts]
    if len(encoded) == 1:
        return encoded[0]
    as_queries = (
        None
        if encoded[0].documents_as_queries is None
        else np.vstack([e.documents_as_queries for e in encoded])
    )
    return band.Encoded(
        profile=encoded[0].profile,
        documents=np.vstack([e.documents for e in encoded]),
        queries=np.empty((0, encoded[0].documents.shape[1]), dtype=np.float32),
        documents_as_queries=as_queries,
    )


def run_pair(
    band: Any,
    corpus: Any,
    parts: list[Any],
    old_model: str,
    new_model: str,
    *,
    args: argparse.Namespace,
    encoder_cache: dict[str, Any],
) -> dict[str, Any]:
    """One row: the condition, then the comparison at every cluster count."""
    from rebasis.compute.arrays import l2_normalize

    started = time.perf_counter()
    shared = {"cache_dir": args.cache_dir, "device": args.device, "encoder_cache": encoder_cache}
    old = encode_parts(band, parts, old_model, **shared)
    new = encode_parts(band, parts, new_model, **shared)

    condition = probe_condition(
        corpus,
        old,
        new,
        sample_size=args.probe_sample,
        heldout=HELDOUT,
        seed=args.seed,
        device=args.device,
        methods=args.probe_methods,
    )
    print(
        f"     probe: arr={condition['arr']} tail={condition['tail_arr']} "
        f"gap={condition['tail_gap']} over {condition['n_clusters']} clusters "
        f"-> {condition['decision']}",
        flush=True,
    )

    row: dict[str, Any] = {
        "corpus": corpus.name,
        "old_model": old_model,
        "new_model": new_model,
        "old_dim": int(old.documents.shape[1]),
        "new_dim": int(new.documents.shape[1]),
        "n_corpus_documents": len(corpus.doc_ids),
        "protocol": "t0-sparse",
        "method": METHOD,
        "fit_pairs": args.fit_pairs,
        "seed": args.seed,
        "probe": condition,
        "comparisons": [],
    }
    if args.survey:
        row["duration_seconds"] = round(time.perf_counter() - started, 1)
        return row

    as_queries = new.documents if new.documents_as_queries is None else new.documents_as_queries
    for requested in args.clusters:
        sample, rows, k_used = build_sample(
            corpus.doc_ids,
            corpus.doc_texts,
            old.documents,
            size=args.sample,
            heldout=HELDOUT,
            seed=args.seed,
            n_clusters=requested,
        )
        comparison = head_to_head(
            sample,
            l2_normalize(np.ascontiguousarray(new.documents[rows])),
            l2_normalize(np.ascontiguousarray(as_queries[rows])),
            k_clusters=k_used,
            metric_k=args.k,
            seed=args.seed,
            fit_pairs=args.fit_pairs,
            device=args.device,
        )
        row["comparisons"].append(comparison)
        print_comparison(comparison)

    row["duration_seconds"] = round(time.perf_counter() - started, 1)
    return row


def print_comparison(comparison: dict[str, Any]) -> None:
    """One line per arm, so a long run can be read while it runs."""
    k = comparison["k_clusters"]
    for name in ("split", "full"):
        budget = comparison["budgets"][name]
        base = comparison["arms"][f"global_{name}"]
        print(
            f"     k={k:<3d} {name:<5s} {budget['total_fit_pairs']:>6d} pairs "
            f"(min {budget['per_cluster_min']}/cluster, "
            f"{budget['clusters_fallback_to_global']} fell back)  "
            f"global arr={base['arr']} tail={base['tail_arr']}",
            flush=True,
        )
        for router in ROUTERS:
            arm = comparison["arms"][f"{name}_{router}"]
            tail = arm["tail_arr"]
            gap = (
                "tail=None"
                if tail is None or base["tail_arr"] is None
                else f"tail={tail} ({tail - base['tail_arr']:+.4f})"
            )
            print(
                f"            {router:<21s} arr={arm['arr']} "
                f"({arm['arr'] - base['arr']:+.4f})  {gap}",
                flush=True,
            )


# ── entry point ───────────────────────────────────────────────────────


def build_parser(band: Any) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        action="append",
        default=None,
        help=(
            f"ir_datasets name, a group ({', '.join(sorted(band.CORPORA))}), or "
            f"'{MIX_PREFIX}a+b+c' to read several collections as one index"
        ),
    )
    parser.add_argument("--ladder", default="default", choices=sorted(band.LADDERS))
    parser.add_argument(
        "--clusters",
        default="8,16,64",
        help="Comma-separated cluster counts; 0 means whatever `probe` would pick",
    )
    parser.add_argument("--k", type=int, default=10, help="Cut-off for every metric")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help=(
            "Documents in the measured index; 0 takes the whole pool, which is "
            "capped at CLUSTER_POOL_MAX either way"
        ),
    )
    parser.add_argument(
        "--probe-sample",
        type=int,
        default=10_000,
        help="Sample for the probe_store condition pass; `probe`'s own default",
    )
    parser.add_argument(
        "--probe-methods",
        default=None,
        help=(
            "Comma-separated adapter methods for the condition pass. The default "
            "is `auto`'s full list, which is what a user runs; a survey across "
            "many corpora is much faster with procrustes_centered alone"
        ),
    )
    parser.add_argument("--fit-pairs", type=int, default=FIT_PAIRS)
    parser.add_argument("--out", type=Path, default=Path("reports/per-cluster/rows.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / "band-cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--survey",
        action="store_true",
        help="Stop after the probe_store pass: report the tail gap and fit nothing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    band = band_module()
    parser = build_parser(band)
    args = parser.parse_args(argv)

    args.clusters = [int(part) or None for part in args.clusters.split(",") if part.strip()]
    args.probe_methods = (
        None
        if args.probe_methods is None
        else [part.strip() for part in args.probe_methods.split(",") if part.strip()] or None
    )
    args.sample = args.sample or None
    datasets = band.resolve_corpora(args.corpus or ["heldout"])
    rungs = band.LADDERS[args.ladder]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    encoder_cache: dict[str, Any] = {}
    for dataset in datasets:
        print(f"\n=== {dataset} ===", flush=True)
        parts = [band.load_corpus(member) for member in mix_members(dataset)]
        corpus = combine(band, dataset, parts)
        print(f"  {len(corpus.doc_ids):,} documents in {len(parts)} collection(s)", flush=True)
        for old_model, new_model in rungs:
            print(f"  -- {old_model} -> {new_model}", flush=True)
            try:
                row = run_pair(
                    band,
                    corpus,
                    parts,
                    old_model,
                    new_model,
                    args=args,
                    encoder_cache=encoder_cache,
                )
            except Exception as exc:
                # The traceback, not only the message. A survey that runs for
                # hours and records `IndexError: list index out of range` has
                # told you a cell failed and nothing about where, which costs
                # more than the four lines it saves.
                trace = traceback.format_exc()
                row = {
                    "corpus": dataset,
                    "old_model": old_model,
                    "new_model": new_model,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": trace.splitlines()[-12:],
                }
                print(f"     {row['error']}\n{trace}", flush=True)
            rows.append(row)
            # Written after every pair: the run is hours, and a row that exists
            # is worth more than a file that would have been complete.
            tmp = args.out.with_name(args.out.name + ".tmp")
            tmp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            tmp.replace(args.out)

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
