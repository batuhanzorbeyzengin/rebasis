<!-- GENERATED FILE — edit the source module, not this page. -->

# Event catalogue

Stable event names emitted by rebasis. The message text of an
event may change between releases; **the event name may not**. Users grep
for these and dashboards group by them.

Naming: `<module>.<object>.<verb-past-tense>`, dot-separated, lowercase.

`Audited` marks the small subset that also produces an audit record.
Log and audit trail are different things: a log may be lost, sampled
and reformatted; an audit record may not.

| Event | Level | Audited | Fields | Description |
|---|---|---|---|---|
| `adapter.applied` | DEBUG | no | `adapter_type`, `count` | Adapter applied to a batch. |
| `adapter.loaded` | INFO | no | `adapter_type`, `direction`, `dim` | Adapter loaded from disk. |
| `adapter.rejected` | ERROR | **yes** | `error_code`, `adapter_type` | Fingerprint mismatch. Audited because it is security-relevant: it means someone tried to apply an adapter to an index it was not built for. |
| `adapter.upgraded` | INFO | **yes** | `adapter_type` | An .rbs file was migrated to a newer schema. |
| `audit.chain.verified` | INFO | no | `count` | Audit hash chain verified end to end. |
| `audit.record.written` | DEBUG | no | `audit_seq`, `action` | An audit record was appended. |
| `compute.device.selected` | INFO | no | `device`, `backend` | Compute device chosen for a sub-task. |
| `compute.oom.recovered` | WARNING | no | `device`, `attempt`, `count` | Device OOM recovered by halving the batch. OOM must never kill a job. |
| `config.unsafe_logging_enabled` | WARNING | **yes** | — | Redaction was disabled via --unsafe-log-content. |
| `embed.backend.unavailable` | WARNING | no | `embed_backend`, `error_code` | Embedding backend unreachable. |
| `embed.batch.completed` | DEBUG | no | `embed_backend`, `model_id`, `batch_index`, `count`, `dim`, `duration_ms` | One embedding batch finished. DEBUG and sampled: per-record logging in a 500k migration buries the real signal. |
| `embed.profile.resolved` | INFO | no | `embed_backend`, `dim`, `symmetric` | Encoding profile resolved. |
| `fit.adapter.fitted` | INFO | **yes** | `adapter_type`, `count`, `dim`, `duration_ms` | An adapter was fitted. |
| `fit.adapter.selected` | INFO | **yes** | `adapter_type`, `arr_r10` | auto selected the best adapter. |
| `fit.calibrator.fitted` | INFO | no | `count`, `score_shift` | Isotonic score calibrator fitted. |
| `migrate.adapter.refitted` | INFO | **yes** | `job_id`, `adapter_type`, `arr_r10` | Adapter refitted mid-migration. |
| `migrate.batch.completed` | INFO | no | `job_id`, `batch_index`, `count`, `duration_ms`, `state` | One migration batch finished. This is the per-batch summary that replaces per-record logging. |
| `migrate.batch.throttled` | WARNING | no | `job_id`, `batch_index`, `peak_rss_bytes` | Batch size halved because memory approached the ceiling. |
| `migrate.durability.verified` | INFO | **yes** | `job_id`, `count` | A fresh connection confirmed the writes are still there. |
| `migrate.index.measured` | INFO | no | `store_backend`, `count`, `ann_recall`, `duration_ms` | How much of the exact answer the store's own search returns, measured against streamed exact kNN over the same collection. Emitted either side of a migration: rewriting a vector does not rewrite the graph that was built around it. |
| `migrate.index.mixed` | WARNING | no | `job_id`, `count`, `state`, `store_backend` | The job stopped with records left, so the index now holds vectors from two embedding spaces and no single query is correct against all of it. `count` is how many records still carry the old model's geometry. |
| `migrate.item.failed` | WARNING | no | `job_id`, `record_id`, `error_code` | A single record failed. |
| `migrate.job.completed` | INFO | **yes** | `job_id`, `count`, `duration_ms` | Migration completed. |
| `migrate.job.paused` | WARNING | **yes** | `job_id`, `state`, `error_code` | Migration paused; resumable. |
| `migrate.job.resumed` | INFO | **yes** | `job_id`, `state` | Migration resumed. |
| `migrate.job.started` | INFO | **yes** | `job_id`, `count`, `adapter_type`, `state` | Migration job started. |
| `observability.redaction.triggered` | WARNING | no | `field_count` | A field outside the allowlist reached the renderer and was redacted. In production code paths this counter must stay at zero — a non-zero value is a bug signal, not a safety net working as intended. |
| `probe.decision.made` | INFO | **yes** | `decision`, `arr_r10`, `borderline`, `count`, `seed`, `device` | The tool's actual output. Audited because this decision must remain defensible six months later. |
| `probe.geometry.measured` | INFO | no | `count`, `dim`, `geometry_delta`, `alignment_bound` | How closely the two models' pairwise similarities agree, and the bound on orthogonal alignment error that implies (Maystre et al., arXiv:2510.13406). Computed before any adapter is fitted. A bound, not a prediction: it says an alignment exists, not that retrieval will use it. |
| `probe.groundtruth.computed` | INFO | no | `count`, `tier`, `duration_ms`, `dropped` | Ground truth established. |
| `probe.metrics.computed` | INFO | no | `arr_r10`, `arr_mrr`, `naive_overlap`, `score_shift`, `tail_arr` | Retrieval metrics computed. |
| `probe.run.completed` | INFO | no | `run_id`, `duration_ms`, `peak_rss_bytes` | Probe run finished. |
| `probe.run.started` | INFO | no | `run_id`, `store_backend`, `embed_backend`, `dim` | Probe run started. |
| `probe.sample.drawn` | INFO | no | `count`, `strategy`, `seed`, `dim`, `dropped` | Sample drawn from the corpus. |
| `retry.attempted` | WARNING | no | `attempt`, `duration_ms`, `error_code` | A transient operation is being retried. |
| `runtime.detected` | INFO | no | `environment`, `log_level`, `log_format`, `blas_threads`, `device` | Detected runtime environment and resolved log configuration. Emitted as the first event of every run so that 'why is it logging at this level' is always answerable. |
| `storage.gc.completed` | INFO | **yes** | `count` | Garbage collection removed artefacts. |
| `storage.integrity.verified` | INFO | no | `count` | Integrity hashes verified. |
| `storage.rollback.completed` | INFO | **yes** | `job_id`, `count` | Rollback restored the original vectors. |
| `storage.shadow.written` | INFO | no | `job_id`, `count` | Shadow copy written for rollback. |
| `storage.write.completed` | DEBUG | no | `duration_ms` | Atomic write finished. |
| `store.capability.missing` | WARNING | no | `store_backend`, `error_code` | Store lacks a required capability. |
| `store.opened` | INFO | no | `store_backend`, `count`, `dim` | Vector store opened. |
| `store.search.completed` | DEBUG | no | `store_backend`, `count`, `duration_ms` | Search completed. |
| `store.write.performed` | INFO | **yes** | `store_backend`, `count` | User data was modified. |
