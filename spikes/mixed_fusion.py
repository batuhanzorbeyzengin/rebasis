"""Which of the two merges is right on a half-migrated index, and by how much.

`serve/mixed.py` sends two queries at a partially migrated collection and merges
what comes back. `serve/hybrid.py`'s `calibrated_merge` does the merging — by
calibrated score when the `.rbs` carries an isotonic `ScoreCalibrator`, by
**reciprocal rank fusion** when it does not. Both are shipped. Neither has been
measured against retrieval quality, and neither has been measured against the
other.

There is a reason to expect them to differ rather than assert that they do not.
RRF scores a document by ``1/(k + rank)`` summed over the result sets it appears
in, which rewards a document for appearing in **both** lists — and during a
migration a document is, by construction, in exactly one space and returned by
exactly one side. Whether that costs anything is an empirical question. It is
the one this spike exists to answer.

Five configurations, one index, at seven points along a real migration::

    status quo        old query  -> old index          the floor a user has
    full reindex      new query  -> new index          the ceiling
    bridged only      adapter(new query) -> the half-migrated index
    mixed, RRF        MixedSpaceSearch, no calibrator
    mixed, calibrated MixedSpaceSearch, the .rbs calibrator

The third is the silent-failure case: the plain `Bridge`, used as if the index
were not mixed. It is the number the mechanism exists to beat.

**The two endpoints are the check that matters most.** At 0% and at 100% there
is only one space in the index, so both merges must degrade to the single-space
answer exactly. `identical_to_single_space` in every stage row is that test, per
query, on the returned id list.

---

**What is real here and what is not.** The migration is driven through
`MigrationEngine` against a real `MemoryStore`: the queue is filled, the shadow
copy is written, every batch is read back and verified, and the manifest records
what moved. `MixedSpaceSearch` then reads which records moved from that manifest,
which is where it reads it from in production — `mixed.py`'s docstring is
explicit that the store is deliberately not asked.

One substitution, and it is in the adapter rather than in the pipeline. Under
``--migrated-vectors reembed`` (the default) the object handed to the engine is a
lookup that returns the **new model's own vector** for each record. The engine's
own docstring names that path — "map them with the adapter, *or re-embed with the
new model*" — but ``_process_batch_inner`` only ever calls ``adapter.apply``, so
there is nothing in the package to drive. The lookup is how a re-embedding
migration is expressed to an engine that only knows how to apply a transform.
Everything else about the run is the shipped code.

``--migrated-vectors adapter`` is the other half of that: a real
``procrustes_centered`` fitted old-space -> new-space and applied by the engine,
which is what ``rebasis migrate`` does today. It is a weaker two-space index —
the migrated vectors are an adapter's image of the old ones rather than the new
model's — and it is reported separately for exactly that reason.

**Run both.** Which merge wins turns out to depend on that flag and not only on
the corpus: the calibrator was fitted to map bridged scores onto the *new
model's* distribution, so a migrated half that is an adapter's image of the old
vectors is not in the distribution the comparison assumes. Reporting one mode
alone would state a result whose sign the other mode reverses.
``docs/mixed-space-fusion.md`` sections 2 and 6 are the two.

**The dimensions have to agree, and that rules out most of the ladder.** A
migration rewrites vectors inside one collection, every backend rebasis supports
locks the collection's dimension, and `MixedSpaceSearch` sends a raw new-model
query at that same collection. So a mixed index only exists when
``d_old == d_new``. Of the seven model pairs in `tools/bridge_band.py`'s ladders
exactly one qualifies: all-MiniLM-L6-v2 -> bge-small-en-v1.5, both 384. That is
the pair measured here, and it is the default.

Corpora, models, the corpus loader, the fit and the `.rbs` round trip all come
from `tools/bridge_band.py` by **import**, so these numbers sit beside the
existing band rather than floating free. Scoring goes through `ranx` for the
reason that harness gives: grading a tool with its own metric code tests
consistency, not correctness.

    ~/rebasis/.venv/bin/python spikes/mixed_fusion.py \\
        --corpus beir/nfcorpus/test --corpus beir/scifact/test \\
        --corpus beir/cqadupstack/android --corpus beir/trec-covid \\
        --cache-dir ~/band-cache --out reports/mixed-fusion.json

Numbers, not adjectives: whatever it prints is what goes in the docs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rebasis.core import fit_candidates, save_adapter
from rebasis.manifest import ManifestDB, manifest_path
from rebasis.migrate import JobQueue, MigrationEngine
from rebasis.serve import Bridge, MixedSpaceSearch
from rebasis.serve.mixed import MAX_OVER_FETCH
from rebasis.store import MemoryStore

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rebasis.types import FloatArray, Hit

#: The only rung of `tools/bridge_band.py`'s ladders whose two models share a
#: dimension, and therefore the only one a migration can produce a mixed index
#: from. See the module docstring.
DEFAULT_PAIR = ("sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5")

#: Where along the migration to measure. The two ends are not decoration: at 0%
#: and 100% the index holds one space, and a merge that does not reduce to the
#: single-space answer there is broken rather than merely worse.
FRACTIONS = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)

#: Cut-offs handed to `ranx`. 10 because that is what a RAG pipeline consumes,
#: and it is what `docs/bridge-band.md` and `docs/cascade-band.md` report at.
CUTOFFS = (10,)

#: Records per migration batch. Only affects how the queue is drained.
BATCH = 512

#: Placeholder id for a query whose configuration returned nothing at all. It is
#: not a document, so it is judged irrelevant and scores zero — which is the
#: honest grade for an empty result and is not the same as dropping the query.
EMPTY = "__no_result__"


# ── the band harness, imported rather than copied ─────────────────────


def band() -> Any:
    """`tools/bridge_band.py`, imported for its corpora, models and fit.

    Imported rather than copied so that a corpus loaded here is the same corpus
    the band was measured on, the embedding cache is the same cache, and the
    adapter comes from the same ``probe_store`` -> ``save_adapter`` ->
    ``Bridge.load`` path. `tools/` is not a package; the repository root goes on
    the path so the implicit namespace package resolves.
    """
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tools import bridge_band

    return bridge_band


# ── the two things handed to the engine ───────────────────────────────


class ReembedAdapter:
    """Return each record's **new-model** vector. Not an adapter — a lookup.

    `MigrationEngine` hands its adapter a batch of vectors and takes back what
    to write; it never passes the ids. So the mapping is keyed on the source
    vector's bytes, which is what `spikes/index_health.py`'s ``_Shuffle``
    control does for the same reason.

    This is the substitution the module docstring names. What it produces is a
    collection in which the migrated records hold the new model's own vectors,
    which is what `serve/mixed.py`, `migrate/spaces.py` and the README all
    describe a partially migrated index as holding, and what the engine's
    docstring calls re-embedding. The engine, the queue, the shadow copy and the
    read-back verification are untouched.
    """

    kind = "reembed"
    type_name = "reembed"

    def __init__(self, source: FloatArray, target: FloatArray) -> None:
        self._by_row: dict[bytes, int] = {}
        #: Records whose source vector is byte-identical to an earlier record's,
        #: which is what a duplicated document in the corpus looks like. They are
        #: given the earlier record's new vector.
        self.collisions = 0
        #: How many of those got the **wrong** vector — the earlier record's new
        #: vector differs from their own. Zero means the duplicates were
        #: duplicate documents and the lookup was exact for every record, which
        #: is the only outcome under which this substitution is free.
        self.wrong = 0
        for row, vector in enumerate(source):
            first = self._by_row.setdefault(vector.tobytes(), row)
            if first != row:
                self.collisions += 1
                self.wrong += int(not np.array_equal(target[first], target[row]))
        self._target = np.ascontiguousarray(target)
        self.input_dim = int(source.shape[1])
        self.output_dim = int(target.shape[1])

    def apply(self, x: FloatArray) -> FloatArray:
        rows = [self._by_row[vector.tobytes()] for vector in np.atleast_2d(x)]
        return self._target[rows]

    def state_dict(self) -> dict[str, FloatArray]:
        return {"__lookup__": np.zeros(1, dtype=np.float32)}

    def n_params(self) -> int:
        return int(self._target.size)


def migration_adapter(kind: str, old: FloatArray, new: FloatArray, *, seed: int, pairs: int) -> Any:
    """What the engine will apply to every stored vector.

    ``reembed`` is the lookup above. ``adapter`` is a real
    ``procrustes_centered`` fitted old-space -> new-space on a seeded sample and
    applied by the engine — the transform `rebasis migrate` runs today, and a
    weaker two-space index than a re-embed produces.
    """
    if kind == "reembed":
        return ReembedAdapter(old, new)

    rng = np.random.default_rng(seed)
    take = min(pairs, old.shape[0])
    sample = np.sort(rng.choice(old.shape[0], size=take, replace=False))
    candidates = fit_candidates(
        old[sample], new[sample], methods=["procrustes_centered"], fit_kind="document"
    )
    if not candidates:
        msg = "procrustes_centered did not fit"
        raise RuntimeError(msg)
    return candidates[0].adapter


def uncalibrated(bridge: Bridge, old_profile: Any, new_profile: Any, directory: Path) -> Bridge:
    """The same adapter, saved and reloaded **without** its calibrator.

    `calibrated_merge` branches on ``calibrator is None`` and reads nothing else
    from the bridge, so this is the state a user is in when `rebasis fit` found
    no calibration to store — and the RRF fallback's actual input.

    Derived from the calibrated bridge rather than fitted a second time on
    purpose: the two configurations then differ in the **merge and nothing
    else**, with byte-identical adapter weights, which is the comparison this
    spike is for. Reaching for ``_adapter`` is the price of that; `Bridge` has no
    public accessor for the object it wraps.
    """
    path = save_adapter(
        bridge._adapter,  # a private attribute, for the reason the docstring gives
        directory / "plain.rbs",
        direction="query_to_old",
        old_profile=old_profile,
        new_profile=new_profile,
        calibrator=None,
    )
    plain = Bridge.load(path, verify=True)
    if plain.has_calibrator:
        msg = "the plain bridge kept a calibrator"
        raise RuntimeError(msg)
    return plain


# ── scoring ───────────────────────────────────────────────────────────


def as_run(
    query_ids: Sequence[str], results: Sequence[Sequence[Hit]]
) -> dict[str, dict[str, float]]:
    """Turn per-query hit lists into the mapping `ranx` wants."""
    run: dict[str, dict[str, float]] = {}
    for query_id, hits in zip(query_ids, results, strict=True):
        run[query_id] = {hit.id: float(hit.score) for hit in hits} or {EMPTY: 0.0}
    return run


def score(corpus: Any, runs: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    """Score every configuration with `ranx`, at every cut-off.

    The metric list is `tools/bridge_band.py`'s, so a row here is comparable
    with a row there. Its ``score()`` itself is not reused: it takes matrices of
    positions into ``doc_ids``, and a mixed-space result can come back shorter
    than ``k`` — a ragged result is not a matrix, and padding one into shape
    would put documents in a result that the merge did not return.
    """
    from ranx import Qrels, Run, evaluate

    qrels = Qrels(corpus.qrels)
    metrics = [f"{name}@{k}" for k in CUTOFFS for name in ("ndcg", "recall", "mrr")]
    return {
        label: {
            metric: round(float(value), 4)
            for metric, value in evaluate(qrels, Run(run), metrics).items()
        }
        for label, run in runs.items()
    }


# ── one stage of one migration ────────────────────────────────────────


def searched(
    store: MemoryStore, queries: FloatArray, *, k: int
) -> tuple[list[list[Hit]], dict[str, float]]:
    """The plain `Bridge` arrangement: one query, no merge, no manifest."""
    results = [store.search(queries[i], k=k) for i in range(queries.shape[0])]
    return results, {"over_fetch_mean": 1.0, "over_fetch_max": 1.0}


def mixed(
    store: MemoryStore,
    bridge: Bridge,
    queries: FloatArray,
    *,
    k: int,
    job_id: str,
    state_dir: Path,
) -> tuple[list[list[Hit]], dict[str, float]]:
    """`MixedSpaceSearch` over every query, recording what each one cost.

    ``over_fetch`` is read after each query rather than at the end: the property
    describes the query that just ran, and one reading at the end would describe
    the last one.
    """
    results: list[list[Hit]] = []
    costs: list[float] = []
    with MixedSpaceSearch(store, bridge, job_id=job_id, state_dir=state_dir) as search:
        for i in range(queries.shape[0]):
            results.append(search.search(queries[i], k=k))
            costs.append(search.over_fetch)
    return results, {
        "over_fetch_mean": round(float(np.mean(costs)), 3),
        "over_fetch_max": round(float(np.max(costs)), 3),
    }


def composition(results: Sequence[Sequence[Hit]], moved: set[str], *, k: int) -> dict[str, float]:
    """How the merged result is made up, and whether it came back short.

    ``share_from_new_space`` is the direct test of RRF's structural bias. A
    document is in exactly one space during a migration, so neither side can be
    rewarded for appearing in both, and a merge with no preference would return
    the two halves in roughly the proportion the migration has reached.
    """
    lengths = [len(hits) for hits in results]
    from_new = [sum(1 for hit in hits if hit.id in moved) for hits in results]
    total = sum(lengths)
    return {
        "share_from_new_space": round(sum(from_new) / total, 4) if total else 0.0,
        "hits_mean": round(float(np.mean(lengths)), 3),
        "short_queries": sum(1 for n in lengths if n < k),
    }


def collapsed(results: Sequence[Sequence[Hit]], bridge: Bridge) -> dict[str, float]:
    """How many of a result's scores the calibrator maps onto the same value.

    Isotonic regression pools adjacent violators, so its output is a step
    function with far fewer levels than it has inputs, and ``out_of_bounds=clip``
    flattens both tails outright. `calibrated_merge` sorts on ``(-score, id)``,
    so every level shared by two documents hands the choice between them to
    whichever id sorts first.

    Measured on the scores the bridge itself returned, which at 0% migrated is
    exactly the population the merge is about to sort.
    """
    calibrator = bridge.calibrator
    if calibrator is None:
        return {}
    distinct: list[int] = []
    for hits in results:
        if not hits:
            continue
        values = calibrator.transform(np.array([hit.score for hit in hits], dtype=np.float32))
        distinct.append(int(np.unique(values).size))
    lengths = [len(hits) for hits in results if hits]
    return {
        "calibrated_levels_mean": round(float(np.mean(distinct)), 3) if distinct else 0.0,
        "queries_with_a_tie": sum(1 for n, m in zip(distinct, lengths, strict=True) if n < m),
    }


def _live_ceiling() -> int:
    """The over-fetch ceiling `MixedSpaceSearch` is actually using right now."""
    import rebasis.serve.mixed as mixed_module

    return int(mixed_module.MAX_OVER_FETCH)


def identical(left: Sequence[Sequence[Hit]], right: Sequence[Sequence[Hit]]) -> float:
    """Fraction of queries on which two configurations returned the same ids, in order."""
    same = sum(
        1
        for a, b in zip(left, right, strict=True)
        if [hit.id for hit in a] == [hit.id for hit in b]
    )
    return round(same / len(left), 4) if left else 0.0


# ── one corpus, one model pair ────────────────────────────────────────


def run_pair(
    corpus: Any,
    old_model: str,
    new_model: str,
    *,
    root: Path,
    cache_dir: Path,
    device: str,
    seed: int,
    k: int,
    fractions: Sequence[float],
    migrated_vectors: str,
    queue_order: str,
    encoder_cache: dict[str, Any],
) -> dict[str, Any]:
    """Every configuration over one corpus and one model pair, at every stage."""
    harness = band()

    if corpus.name in harness.SELF_REMOVAL:
        msg = (
            f"{corpus.name} is scored with self-removal, which MixedSpaceSearch "
            f"has no way to express. Measuring it here would compare a masked "
            f"configuration against unmasked ones."
        )
        raise RuntimeError(msg)

    started = time.perf_counter()
    shared = {"corpus": corpus, "cache_dir": cache_dir, "device": device}
    old = harness.encode_corpus(model_id=old_model, encoder_cache=encoder_cache, **shared)
    new = harness.encode_corpus(model_id=new_model, encoder_cache=encoder_cache, **shared)

    if old.documents.shape[1] != new.documents.shape[1]:
        msg = (
            f"{old_model} is {old.documents.shape[1]}-dimensional and {new_model} is "
            f"{new.documents.shape[1]}-dimensional. A collection's dimension is "
            f"locked, so this pair cannot produce a mixed index at all."
        )
        raise RuntimeError(msg)

    # The fit, the .rbs and the Bridge all come from the band harness, so the
    # adapter under test is the adapter that document reports.
    bridge, fit_summary = harness.fit_bridge(corpus, old, new, seed=seed, device=device)
    plain = uncalibrated(bridge, old.profile, new.profile, root)
    mapped = bridge.to_index_space(new.queries)

    store = MemoryStore(corpus.doc_ids, old.documents.copy(), corpus.doc_texts)

    # The floor, read off the index before a single vector moves. Taken through
    # `MemoryStore.search` rather than `rebasis.compute.top_k_search` so that
    # every number in this file comes out of one arithmetic path: a GPU matmul
    # and a numpy one disagree in the last bits, and the endpoint check below
    # asks whether two rankings are *identical*, which that noise would answer
    # for it.
    reference_hits = {"status_quo": searched(store, old.queries, k=k)[0]}
    adapter = migration_adapter(
        migrated_vectors, old.documents, new.documents, seed=seed, pairs=harness.FIT_PAIRS
    )
    state_dir = root / "state"
    db = ManifestDB(manifest_path(state_dir))
    engine = MigrationEngine(
        db=db,
        store=store,
        adapter=adapter,
        shadow_root=root / "shadow",
        batch_size=BATCH,
        power_aware=False,
    )
    # The queue drains in record-id order, which is what `--limit` gives a user.
    # `--queue-order random` uses the same priority column `--priority access`
    # uses, to check that a finding is not an artefact of how the ids sort.
    priorities = None
    if queue_order == "random":
        rng = np.random.default_rng(seed)
        priorities = dict(zip(corpus.doc_ids, rng.random(len(corpus.doc_ids)), strict=True))
    engine.prepare(list(corpus.doc_ids), priorities=priorities)
    queue = JobQueue(db, engine.job_id)
    total = queue.stats().total

    stages: list[dict[str, Any]] = []
    for target in fractions:
        wanted = round(target * total)
        already = queue.stats().done
        if wanted > already:
            engine.run(limit=wanted - already)
        stats = queue.stats()
        moved = {record_id for chunk in queue.iter_done() for record_id in chunk}

        results: dict[str, list[list[Hit]]] = {}
        costs: dict[str, dict[str, float]] = {}
        results["bridged_only"], costs["bridged_only"] = searched(store, mapped, k=k)
        results["mixed_rrf"], costs["mixed_rrf"] = mixed(
            store, plain, new.queries, k=k, job_id=engine.job_id, state_dir=state_dir
        )
        results["mixed_calibrated"], costs["mixed_calibrated"] = mixed(
            store, bridge, new.queries, k=k, job_id=engine.job_id, state_dir=state_dir
        )

        # At the ends there is one space in the index, so there is a single
        # right answer and both merges have to reproduce it. In between there is
        # nothing to compare against and the field is null rather than 0.
        #
        # At 0% that answer is what the bridge alone returns — the migrated side
        # is empty. At 100% it is the new query against what is now a new index,
        # which is the ceiling configuration, so it is kept and reported as such.
        single_space: list[list[Hit]] | None = None
        if stats.done == 0:
            single_space = results["bridged_only"]
        elif stats.done == total:
            reference_hits["full_reindex"] = searched(store, new.queries, k=k)[0]
            single_space = reference_hits["full_reindex"]

        stages.append(
            {
                "target_fraction": target,
                "migrated": stats.done,
                "fraction": round(stats.done / total, 4) if total else 0.0,
                "scores": score(
                    corpus,
                    {label: as_run(corpus.query_ids, hits) for label, hits in results.items()},
                ),
                "cost": costs,
                "result": {label: composition(hits, moved, k=k) for label, hits in results.items()},
                "calibration": collapsed(results["bridged_only"], bridge),
                "identical_to_single_space": (
                    None
                    if single_space is None
                    else {
                        label: identical(hits, single_space)
                        for label, hits in results.items()
                        if label != "bridged_only" or stats.done == total
                    }
                ),
            }
        )
        print(json.dumps(stages[-1]["scores"] | {"fraction": stages[-1]["fraction"]}), flush=True)

    # The ceiling. Already taken if the stages reached 100%; otherwise the queue
    # is drained now so it is read off the same index as everything else.
    if "full_reindex" not in reference_hits:
        engine.run()
        reference_hits["full_reindex"] = searched(store, new.queries, k=k)[0]
    reference = score(
        corpus,
        {label: as_run(corpus.query_ids, hits) for label, hits in reference_hits.items()},
    )

    db.close()
    return {
        "corpus": corpus.name,
        "old_model": old_model,
        "new_model": new_model,
        "dim": int(old.documents.shape[1]),
        "n_documents": len(corpus.doc_ids),
        "n_queries": len(corpus.query_ids),
        "k": k,
        "seed": seed,
        "cutoffs": list(CUTOFFS),
        "migrated_vectors": migrated_vectors,
        "queue_order": queue_order,
        "max_over_fetch": _live_ceiling(),
        "batch": BATCH,
        "duplicate_source_vectors": getattr(adapter, "collisions", None),
        "lookup_gave_wrong_vector": getattr(adapter, "wrong", None),
        "adapter_type": getattr(adapter, "type_name", ""),
        "fit": fit_summary,
        "calibrator": None if bridge.calibrator is None else bridge.calibrator.summary(),
        "reference": reference,
        "stages": stages,
        "duration_seconds": round(time.perf_counter() - started, 1),
    }


# ── entry point ───────────────────────────────────────────────────────


def environment() -> dict[str, str]:
    """What produced these numbers, recorded beside them."""
    from importlib import metadata

    versions = {}
    for package in ("rebasis", "ranx", "numpy", "ir_datasets", "sentence-transformers", "torch"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "absent"
    return versions


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        action="append",
        default=None,
        help="ir_datasets name, or one of tools/bridge_band.py's groups",
    )
    parser.add_argument("--old-model", default=DEFAULT_PAIR[0])
    parser.add_argument("--new-model", default=DEFAULT_PAIR[1])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--fractions",
        default=",".join(str(f) for f in FRACTIONS),
        help="Where along the migration to measure",
    )
    parser.add_argument(
        "--migrated-vectors",
        default="reembed",
        choices=("reembed", "adapter"),
        help="What the migrated records end up holding; see the module docstring",
    )
    parser.add_argument(
        "--queue-order",
        default="record_id",
        choices=("record_id", "random"),
        help="record_id is what `--limit` gives; random uses the priority column",
    )
    parser.add_argument(
        "--over-fetch-ceiling",
        type=int,
        default=MAX_OVER_FETCH,
        help=(
            "Overwrite `serve.mixed.MAX_OVER_FETCH` for the run. The shipped "
            "value caps each side at 8x k and its docstring says the result is "
            "'short rather than slow' there; raising it is how much that costs"
        ),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / "band-cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    # Patched on the module rather than passed, because `_depth` reads the
    # constant and there is no seam to inject one through. Recorded on every row
    # so a run at a raised ceiling can never be read as a shipped one.
    if args.over_fetch_ceiling != MAX_OVER_FETCH:
        import rebasis.serve.mixed as mixed_module

        mixed_module.MAX_OVER_FETCH = args.over_fetch_ceiling

    harness = band()
    datasets = harness.resolve_corpora(args.corpus or ["beir/scifact/test"])
    fractions = [float(part) for part in args.fractions.split(",") if part.strip()]
    encoder_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    for dataset in datasets:
        print(f"\n=== {dataset} ===", flush=True)
        corpus = harness.load_corpus(dataset, limit=args.limit_docs)
        print(
            f"  {len(corpus.doc_ids):,} documents, {len(corpus.query_ids):,} judged queries",
            flush=True,
        )
        root = Path(tempfile.mkdtemp(prefix="rebasis-fusion-"))
        try:
            row = run_pair(
                corpus,
                args.old_model,
                args.new_model,
                root=root,
                cache_dir=args.cache_dir,
                device=args.device,
                seed=args.seed,
                k=args.k,
                fractions=fractions,
                migrated_vectors=args.migrated_vectors,
                queue_order=args.queue_order,
                encoder_cache=encoder_cache,
            )
        except Exception as exc:
            row = {"corpus": dataset, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            shutil.rmtree(root, ignore_errors=True)
        rows.append(row)

    report = {
        "spike": "mixed_fusion",
        "seed": args.seed,
        "over_fetch_ceiling": _live_ceiling(),
        "k": args.k,
        "old_model": args.old_model,
        "new_model": args.new_model,
        "migrated_vectors": args.migrated_vectors,
        "queue_order": args.queue_order,
        "fractions": fractions,
        "environment": environment(),
        "runs": rows,
    }

    print()
    for row in rows:
        if "error" in row:
            print(f"{row['corpus']:24s} {row['error']}")
            continue
        print(f"{row['corpus']}  ({row['n_queries']} queries, {row['n_documents']:,} documents)")
        print(
            f"  status quo {row['reference']['status_quo'][f'ndcg@{args.k}']:.4f}"
            f"   full reindex {row['reference']['full_reindex'][f'ndcg@{args.k}']:.4f}"
        )
        metric = f"ndcg@{args.k}"
        print(
            f"  {'migrated':>9s} {'bridged':>9s} {'RRF':>9s} {'calib':>9s} "
            f"{'fetch':>7s} {'short':>7s} {'RRF new':>9s} {'cal new':>9s} {'same':>14s}"
        )
        for stage in row["stages"]:
            scores, result = stage["scores"], stage["result"]
            same = stage["identical_to_single_space"]
            print(
                f"  {stage['fraction']:9.2f} "
                f"{scores['bridged_only'][metric]:9.4f} "
                f"{scores['mixed_rrf'][metric]:9.4f} "
                f"{scores['mixed_calibrated'][metric]:9.4f} "
                f"{stage['cost']['mixed_calibrated']['over_fetch_mean']:7.2f} "
                f"{result['mixed_calibrated']['short_queries']:7d} "
                f"{result['mixed_rrf']['share_from_new_space']:9.3f} "
                f"{result['mixed_calibrated']['share_from_new_space']:9.3f} "
                + (
                    "".rjust(14)
                    if same is None
                    else f"{same['mixed_rrf']:6.3f}/{same['mixed_calibrated']:.3f}".rjust(14)
                )
            )
        print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
