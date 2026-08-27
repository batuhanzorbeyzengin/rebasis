`rebasis pause <job-id>` stops a running migration after its current batch, and `rebasis resume <job-id>` starts it again.

Killing the process was always safe — the queue is the checkpoint and a shadow copy is written before the vector it copies is overwritten — but it lands mid-batch, and the read-back that verifies a write is a per-batch guarantee that half a batch does not get. `pause` returns immediately and the job stops at the next boundary.

It takes no lock, because the migration it is interrupting holds the state lock for its whole run and a command that waited for it would wait for the thing it is trying to stop. What makes that safe is that it writes one column nothing else writes: `jobs.pause_requested`, new in manifest schema 3, is a **request**. Only the engine ever says what state a job is in — writing `PAUSED` from a second process would claim a stop that had not happened and race the engine over the same column.

`status` shows an outstanding request as `running (pausing)` and carries it in `--json` as a separate `pause_requested` field, so a script branching on `state == "running"` keeps working. A request never outlives the run it was made for: it is cleared when a run ends and again when one starts, so a process killed between `rebasis pause` and the engine reading it cannot leave a flag that silently pauses the next run.

`resume` forwards to `migrate --resume`, which is unchanged. Only the flags that describe *this run* are accepted; `--priority` and `--access-log` are not, because they order the queue, the queue was ordered when the job was created, and re-ordering half a migration would be a different job.
