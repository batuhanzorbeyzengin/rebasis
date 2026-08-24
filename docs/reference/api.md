# Python API

The public surface is deliberately small. `rebasis` itself exports two names,
because the hot path should not have to import a package that pulls in a store
client.

## The query path

::: rebasis.Bridge

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
