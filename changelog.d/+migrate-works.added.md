**`rebasis migrate` can be run again, with an adapter that points the right way — and the measurement says it is worth about as much as bridging, which is usually not much.**

`rebasis fit --direction old_to_new` produces the map a migration needs: out of the index's space and into the new model's, rather than the reverse the query path uses. It is the same `fit_candidates` call with source and target exchanged, but it is **not** the same fit with its arguments swapped, because the evaluation differs and that is the part that matters. A query map is judged on what a bridged query retrieves from an untouched index; a document map is judged on what a **raw** new-model query retrieves from a rewritten one. `rebasis.probe.migration` scores the second, which is the configuration a user is left in once `migrate` finishes and there is no adapter on the hot path at all.

**Both directions are now guarded, in both directions.** `migrate` refuses a `query_to_old` adapter before it opens the store; `Bridge.load` refuses an `old_to_new` one. Each is useless in the other's place and neither check existed a release ago — which is how a query map came to be written over indexes until it was measured at recall@1 0.000 for every query type there is.

**What it is worth, over 51 runs on seventeen corpora with human relevance judgements** ([the band](https://batuhanzorbeyzengin.github.io/rebasis/migration-band/)):

| | |
|---|---|
| a completed migration delivers | **0.727** of a full reindex |
| bridging, on the same runs | **0.719** |
| the two track each other at | Spearman **0.993** (p ≈ 1e-46) |
| paired median difference | **+0.004** in favour of migrating |
| migrating beat leaving the index alone in | **5 of 51** |

So the two are the same number. That is [ADR 10](https://batuhanzorbeyzengin.github.io/rebasis/adr/0010-retention-is-bounded-by-the-source/) reaching the document side for the first time: the same source space under the same family of map carries the same amount whichever end it is applied to. The ADR was measured entirely on the query side and could not have said so.

What migrating buys is the adapter leaving the query path — nothing on the hot path, no `.rbs` shipped with a service, the new model querying its own space. What it costs is rewriting every vector, the shadow copy behind it, and a window in which the index holds two spaces. The guide now says to choose on those grounds, because retrieval quality is not one of them.

The end-to-end suite migrates through the real command now rather than through an adapter built by hand in the test file, which was a workaround for the window in which nothing could produce a forward map.
