# FAISS

The one to read twice. FAISS is an **index, not a database**: it stores vectors
and returns row numbers. Ids, text and metadata are yours to keep, and what you
kept decides how much of rebasis works.

## The URI

```
faiss:///path/to/vectors.faiss
```

No fragment — there is no collection inside a FAISS index to name.

## The sidecar

rebasis expects two files side by side:

```
vectors.faiss        the index
vectors.meta.json    {"ids": [...], "texts": [...]}
```

Both keys are optional and each buys something different. Verified behaviour:

| What you have | Ids reported | `probe` | `migrate` |
|---|---|---|---|
| Index only | Row numbers — `"0"`, `"1"`, … | Vector-only mode | No |
| Index + `ids` | Your own ids | Vector-only mode | No |
| Index + `ids` + `texts` | Your own ids | Fully | If the index can be written — see below |

**"Vector-only mode" is the case to understand.** Without text there is nothing
to re-embed, and the pairs an adapter is fitted on are the index's own vectors on
one side and the same documents re-embedded with the candidate model on the
other. So `probe` can still tell you how the geometry compares, and it cannot
tell you whether the new model is better on your corpus. The report says which of
the two you got rather than presenting one as the other.

This is the case rebasis calls out as the one no adapter survives: an index that
kept its vectors and discarded the text they came from.

## Writing needs an `IndexIDMap2`

A bare `IndexFlatIP` addresses vectors by row number and has no way to replace
one by label. An `IndexIDMap2` addresses them by the label you handed
`add_with_ids`, and that is what `migrate` needs.

```python
import faiss, numpy as np

index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
index.add_with_ids(vectors, np.arange(len(vectors)).astype("int64"))
faiss.write_index(index, "vectors.faiss")
```

Verified: a bare index declares `can_upsert_vectors=False` and `migrate` refuses
**at second zero** rather than halfway through. That is the whole point of a
backend declaring its capabilities honestly — partial support is genuinely
useful, and *silent* partial support fails in the middle of somebody's migration.

## Reconstruction, and the check that does not catch everything

rebasis reads vectors back out of the index, so an index that cannot reconstruct
at all is refused up front. That catches the indexes where `reconstruct` **raises**
— an IVF index with no direct map cannot find a vector by id in the first place.

It does not catch the ones where reconstruction *succeeds and lies*. `IndexPQ`
and `IndexScalarQuantizer` decode their codes and hand back a vector that is not
the one that went in. Those are not refused — they are **declared**, through
`capabilities.quantized`, because an approximation is not a reason to refuse a
migration. It is a reason for you to know what your round trip and your rollback
are worth.

Verified on `IndexIDMap2(IndexPQ(...))`: `quantized=True`, `upsert=True`. Read
[If your index is stored quantized](migration.md#if-your-index-is-stored-quantized)
before you migrate one — the shadow copy holds what the index gave back, which
for a quantized index is already not what you originally wrote.

## No filtering, no rebuild

FAISS has no metadata, so there is nothing to filter on and
`can_filter=False`. There is no `--rebuild-index` either: rebasis does not
retrain your index for you, and choosing an index type is a decision about your
own recall/latency trade-off rather than something a migration tool should make.

If your index is IVF or HNSW, rewriting the vectors leaves the structure built
around the old ones. `migrate` measures the index's own recall against exact kNN
before and after and names any drop — but on FAISS, fixing that drop is your
rebuild, with your parameters.

## The install

```bash
pip install "rebasis[faiss]"
```

`faiss-cpu>=1.9` — not 1.8, which was built against numpy 1 and cannot import
under numpy 2, which this package allows.

**On macOS, do not install `faiss-cpu` and `torch` into the same environment.**
Each links its own OpenMP runtime and a process holding both aborts before either
library does any work. It is not rebasis' bug and there is no caller-side fix; the
documented workaround is itself documented as liable to produce wrong results.
`rebasis doctor` reports the conflicting pair when it sees it. This is also why
the macOS leg was removed from CI.
