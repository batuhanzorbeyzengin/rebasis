# What a migration does to the index

`migrate` had three checks and all three were about the same thing. The
per-batch read-back proves the store took the write. The end-of-job check on a
fresh connection proves it kept it. The dimension check proves it would accept
it at all. None of them asks whether the record can still be **found**.

That is a separate question with a separate answer, because a graph index picks
a record's edges from the geometry of its neighbours at insert time. Rewriting
the vector does not rewrite the graph. Afterwards the edges describe a
neighbourhood that no longer exists — the counts are right, the payloads are
right, nothing raises, and a search can walk past a document that is sitting in
the index, correct and verified, one hop off the path the graph sends the query
down.

Qdrant states the rule plainly in its incremental-HNSW work: a changed vector
value discards the graph the same way a deletion does
([qdrant#6325](https://github.com/qdrant/qdrant/pull/6325)). A production report
measured search quality at 34% until the collection was force-reindexed
([qdrant#7147](https://github.com/qdrant/qdrant/issues/7147)). The general case
— unreachable points and node isolation under update — is
[arXiv:2407.07871](https://arxiv.org/abs/2407.07871) and
[arXiv:2507.19802](https://arxiv.org/abs/2507.19802).

rebasis writes to five backends and had measured none of them. This is that
measurement.

---

## How it is measured

Take a sample of records, use their own stored vectors as queries, and compare
what the store's index returns against exact nearest neighbours computed by
streaming the corpus:

```
recall@10 of store.search() against exact kNN, over 200 probe records
```

The exact side streams, so peak memory is `O((probes + batch) × d)` and the
check runs on a collection of any size at the cost of a scan. A record retrieves
itself first on both sides, so both ask for one extra neighbour and drop the
probe's own id — otherwise every comparison would carry a free hit.

This is a property of the index **structure**, not of the embedding model, which
is what makes it comparable before and after a migration. It is not comparable
*across* backends: an approximate index is approximate by design, and the number
only means something against where that same index started.

`spikes/index_health.py` is the harness. Everything below is 100,000 records at
384 dimensions on a clustered corpus, on the project's A10G host.

The `qdrant-server` backend needs a Qdrant listening on 6333, which the spike
deliberately does not start — a measurement harness that manages a database is a
harness with a second thing to go wrong. A single binary is enough:

```bash
curl -sL -o qdrant.tar.gz \
  https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-musl.tar.gz
tar xzf qdrant.tar.gz && ./qdrant --config-path /dev/null &
```

Everything else runs against an embedded database or a file and needs nothing.

---

## 1. Most of the backends have no graph to break

| backend | before | after | change |
|---|---|---|---|
| chroma | 0.988 | 0.982 | −0.007 |
| sqlite-vec | 1.000 | 1.000 | 0.000 |
| faiss | 1.000 | 1.000 | 0.000 |

sqlite-vec's `vec0` and a FAISS `IndexFlatIP` both scan. They return the exact
answer before and after, and there is nothing for a migration to disturb.

**The embedded modes are not the server modes, and that turned out to matter.**
Qdrant's local mode and LanceDB's OSS default both scanned at 3,000 records.
Qdrant's local mode also scanned at 20,000 — and checking rather than assuming
was worth it: `indexed_vectors_count` was **0**, because the default
`indexing_threshold` of 10,000 is *per segment* and 20,000 points across four
segments leaves every segment below it. A run that had not looked would have
reported "no degradation" about an index that was never approximate. Section 2's
Qdrant numbers are a real server with 100,000 of 100,000 vectors in an HNSW
graph.

Chroma builds an HNSW graph from the start, and it begins at **0.989** rather
than 1.000. That is not a defect — it is what approximate means, and it is
exactly why the check measures a *before*. A tool that asserted a fixed
threshold would have called a healthy Chroma collection broken.

---

## 2. The loss is proportional to how much the adapter disturbs the geometry

This is the finding. Same corpus, same backend, same migration — only the
transform applied to every vector changes:

| transform | before | after | change |
|---|---|---|---|
| `procrustes` (orthogonal) | 0.989 | 0.984 | **−0.005** |
| `procrustes_centered` (rebasis' default) | 0.989 | 0.976 | **−0.013** |
| `linear` (unconstrained affine) | 0.990 | 0.912 | **−0.078** |
| `low_rank_affine` | 0.990 | 0.871 | **−0.119** |
| *shuffle* (control) | 0.990 | 0.646 | −0.344 |

**How much of that is noise.** Chroma's HNSW construction is not
deterministic, and two independent runs of the same configuration
(`procrustes_centered`, same seed, same corpus) landed at 0.976 and 0.984. So
the repeat spread is about **0.008**, and the first two rows of that table are
not separable from each other: what can be read off it is that an orthogonal or
near-orthogonal map costs something inside the noise, and that `linear` and
`low_rank_affine` cost six to fifteen times it.

The ordering is not a coincidence and it is not about adapter quality. An
orthogonal map preserves every pairwise inner product, so the neighbourhood each
edge encodes is still true after it; the graph does not need rebuilding because
nothing it described has moved. Centring adds a translation, which after
renormalisation does move the angles slightly. An unconstrained affine map moves
them a great deal.

*shuffle* is the control: every record is given another record's vector, so the
distribution is untouched and no norm or dimension check could notice, while the
thing the graph encodes — which record is near which — is destroyed. It exists
to show the measurement can see a broken index at all. At 0.646 rather than 0.0,
it also shows that a broken HNSW graph is still *a* graph: traversal keeps
finding some of the right answers by accident.

### Qdrant and Chroma pay for it in different currencies

The same measurement on a real Qdrant server, 100,000 vectors indexed into a
genuine HNSW graph:

| transform | chroma | qdrant server |
|---|---|---|
| `procrustes` | −0.005 | 0.000 |
| `procrustes_centered` | −0.013 | −0.004 |
| `linear` | −0.052 to −0.078 | **−0.002** |
| `low_rank_affine` | −0.117 | −0.062 |

Qdrant barely moves where Chroma loses five to eight points, and the reason is
visible while the run is happening. Qdrant's own incremental-HNSW work states
that a changed vector value discards the graph the way a deletion does
([qdrant#6325](https://github.com/qdrant/qdrant/pull/6325)) — so the optimizer
rebuilds it. Polled during a migration, the collection reports `status: yellow`
with `indexed_vectors_count` **above** the point count, which is what a
background reindex over overlapping segments looks like.

So the failure is not that one backend is careless. It is that **the cost lands
somewhere different in each**:

- **Chroma** leaves the graph alone. The migration is fast and the recall is
  what pays.
- **Qdrant** rebuilds the graph. The recall holds and the CPU and wall-clock
  are what pay: 181–199 seconds against Chroma's 89–145 for the same 100,000
  records, on the same host, plus background indexing that outlives the run.

Neither is wrong, and neither was written down here before. The practical
consequence is for the plan `migrate` prints before it starts: it counts the
shadow copy, the checkpoint and an estimated wall time from throughput, and a
background reindex is not in that estimate. On a backend that rebuilds, the run
therefore costs more than the plan says, and keeps costing after it returns.
`migrate` now says so up front for a backend that declares it can reindex.

### This is a third, independent reason orthogonal wins

[ADR 10](adr/0010-retention-is-bounded-by-the-source.md) measured
`procrustes_centered` beating every more flexible candidate 15 times out of 15
on retrieval. [Maystre et al.](https://arxiv.org/abs/2510.13406) supply the
mechanism: preserving the stronger model's geometry keeps information an
unconstrained map discards.

The table above is a different argument arriving at the same place. Even setting
retrieval quality aside entirely, a non-orthogonal map costs recall **in the
index itself** — a loss that has nothing to do with how well the adapter
approximates its target, and that the adapter's own ARR cannot see.

`auto` already selects `procrustes_centered`, so a default run pays something
inside the noise. A user who overrides it with `--method linear` pays six times
the repeat spread, and until now nothing would have told them.

---

## 3. Two different losses wear the same number

The obvious follow-up: is the loss recoverable? A migrated collection was read
back out and inserted into a fresh one — which is what any backend's own reindex
does, building the graph from the geometry that is actually there now.

| transform | backend | before | after migrating | after rebuilding |
|---|---|---|---|---|
| `procrustes_centered` | chroma | 0.990 | 0.984 | 0.986 |
| `linear` | chroma | 0.989 | 0.936 | **0.987** |
| `low_rank_affine` | chroma | 0.990 | 0.873 | **0.874** |
| `procrustes_centered` | qdrant | 0.999 | 0.995 | 1.000 |
| `linear` | qdrant | 1.000 | 0.998 | 1.000 |
| `low_rank_affine` | qdrant | 0.999 | 0.936 | **0.948** |

`linear` recovers completely. `low_rank_affine` barely recovers on either
backend, and the two are not different amounts of the same thing — they are
different failures that produce the same-looking number:

- **`linear` lost the graph.** The vectors are as distinguishable as they ever
  were; the edges were describing where they used to be. Rebuilding the index
  fixes it, and the whole 0.052 comes back.
- **`low_rank_affine` lost the vectors.** A low-rank map puts every output in a
  subspace, so documents that were distinct become near-neighbours of each
  other. There is no graph that separates points which are no longer separate,
  and a rebuild reproduces the same 0.873 because it is measuring the same
  collapsed geometry.

Two independent backends agree on that split, which is what makes it a property
of the transform rather than of Chroma. This is the reason `migrate` reports the
number without attaching a remedy to it: "rebuild your index" is right for one of
these rows and useless for the other, and nothing on the migration side can tell
which without trying.

It is also a sharper argument than the previous section made. An orthogonal map
avoids **both** failures at once: it preserves the inner products, so the graph
stays valid *and* the points stay as separable as they were.

---

## 4. What changed in the tool

`migrate` runs the measurement before and after, and names any drop:

```
index  The index returns less of the exact answer than before: recall@10
       against exact kNN fell from 0.989 to 0.912 over 200 probes. The
       vectors are correct and verified — this is the search structure,
       which was built against the geometry the old vectors had.
```

No threshold is applied. What counts as a serious drop depends on the backend
and on the index parameters, and this project does not publish a threshold it
has not measured across enough of both. The number and its direction are the
finding.

It costs two scans of the collection: **18–21 seconds** at 100,000 records
against a migration of 105–170 seconds, so roughly a quarter again. `migrate
--no-health-check` turns it off.

---

## What this does not establish

- **Two graph backends, not five.** Chroma and a Qdrant server. LanceDB OSS
  builds a vector index only when asked and scans otherwise, so an index it was
  never given is not one a migration can damage.
- **Two ways of rebuilding, and they are not the same move.** Chroma is rebuilt
  by inserting into a fresh collection, which is the most thorough thing
  possible and not something rebasis does. Qdrant is rebuilt through
  `rebuild_index()`, which is the shipped code path and the backend's own
  documented mechanism.
- **One corpus shape, one scale, default parameters** — see below.
- **One corpus shape.** Clustered synthetic vectors at 384 dimensions. Real
  embeddings have structure a Gaussian mixture does not.
- **Default index parameters**, except one. Chroma's `M=16`,
  `ef_construction=100`; Qdrant's `m=16`, `ef_construct=100`, with
  `indexing_threshold` lowered to 1,000 so the graph existed at all. A
  collection built with a larger `M` has more redundant edges and should degrade
  less; that is expected rather than measured.
- **One scale.** At 3,000 and 4,000 records every backend including Chroma
  returned 1.000 both sides. The effect needs an index large enough to be
  genuinely approximate before it exists at all.
