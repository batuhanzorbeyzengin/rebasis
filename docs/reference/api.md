# Python API

The public surface is deliberately small. `rebasis` itself exports two names,
because the hot path should not have to import a package that pulls in a store
client.

## The query path

::: rebasis.Bridge

## Querying a half-migrated index

For the window between starting a migration and finishing it, when the
collection holds both models' vectors and no single query is correct against all
of it. See [Migration and rollback](../guides/migration.md#searching-one-anyway).

::: rebasis.serve.MixedSpaceSearch

::: rebasis.serve.calibrated_merge

::: rebasis.serve.reciprocal_rank_fusion

## Two-stage retrieval

The bridge as a recall stage, with the new model reranking its candidate set.
Measured in [the bridge as a recall stage](../cascade-band.md); the cache is part
of the design, because re-embedding N documents per query is what the
arrangement costs.

::: rebasis.serve.Cascade

::: rebasis.serve.CascadeStats

::: rebasis.serve.MemoryVectorCache

::: rebasis.serve.DiskVectorCache

## Probing a store

::: rebasis.probe.session.probe_store

::: rebasis.probe.session.draw_corpus_sample

::: rebasis.probe.session.CorpusSample

::: rebasis.probe.session.QueryLog

## Results

::: rebasis.probe.runner.ProbeResult

::: rebasis.probe.runner.CandidateMetrics

::: rebasis.probe.decision.DecisionResult

## Stores

::: rebasis.store.base.VectorStore

::: rebasis.store.open_store

## Adapters

::: rebasis.core.serialization.save_adapter

::: rebasis.core.serialization.load_adapter

## Reports

::: rebasis.report.render_markdown

::: rebasis.report.render_html
