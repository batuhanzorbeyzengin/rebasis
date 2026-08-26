# CLI reference

## Global options

| Option | Effect |
|---|---|
| `--version` | Print the version and exit |
| `-v`, `-vv` | INFO, then DEBUG |
| `-q` | Errors only |
| `--log-level debug\|info\|warning\|error` | Sets the level outright; overrides `-v` and `-q` |
| `--log-format auto\|console\|json` | JSON when piped, human-readable in a terminal |
| `--log-file PATH` | Also write logs here |
| `--unsafe-log-content` | Disable redaction. Debugging only; warns and is audited. |
| `--install-completion` | Install shell completion |

Progress and diagnostics go to **stderr**; command output goes to **stdout**.
That is what makes `--json` safe to pipe while a person still watches the run.

## Machine-readable output

`probe`, `eval`, `status`, `gc` and `doctor` take `--json`. `status` also takes
`--plain`, which prints one job per line, tab separated — the table truncates job
ids with an ellipsis, and an ellipsis is not a job id `rollback` will accept.

```bash
rebasis probe ... --json | jq -r .decision
rebasis status --json | jq -r '.[] | select(.state=="paused") | .job_id'
rebasis doctor --json          # the version to attach to a bug report
```

## Prompts

Every command that changes something asks first. `--yes` answers for you.
Without it, and with no terminal on stdin, the command **fails with a usage
error** naming the flag rather than prompting into a void. `--no-input` refuses
to prompt even on a terminal.

## Exit codes

A contract for script users. A change here is a breaking change.

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Unexpected error — a bug in rebasis |
| `2` | Usage or configuration |
| `3` | Domain error |
| `130` | Interrupted |

---

## `rebasis probe`

Measure what switching models would cost, and recommend. Reads the index; never
writes to it.

| Option | Default | |
|---|---|---|
| `--store` | required | Store URI |
| `--old` | required | The model the index was built with |
| `--new` | required | The model you are considering |
| `--sample` | 10000 | Documents to embed with the new model |
| `--heldout` | 1000 | Documents held out as query proxies |
| `--k` | 10 | Cut-off for every metric |
| `--queries` | — | Real query log (JSONL). Always preferred when available. |
| `--synth-queries` | — | `title\|lead\|keywords` — estimate the upgrade from the documents when you have no query log |
| `--report` | — | Write a report; `.html` for HTML, otherwise Markdown |
| `--strategy` | stratified | `stratified` or `random` |
| `--seed` | 0 | Recorded so the run can be replayed |
| `--device` | auto | `auto\|cpu\|cuda\|cuda:N\|mps` |
| `--state-dir` | `./.rebasis` | Where the audit trail lives. `REBASIS_STATE_DIR` sets it once. |
| `--old-dim`, `--new-dim` | — | Dimension, for a model rebasis does not know |
| `--query-prefix`, `--document-prefix` | — | The new model's prefixes |
| `--old-query-prefix`, `--old-document-prefix` | — | The old model's prefixes |

### Without a query log

`probe` needs to answer two questions: how well an adapter bridges, and whether
the new model is actually better on your corpus. It can always answer the first.
The second needs queries.

With neither `--queries` nor `--synth-queries`, the run is reported as
**provisional**: it says how well the adapter bridges and declines to recommend
acting on it. Measured on six real corpus/model pairs, a run without an upgrade
estimate recommended bridging in six of six cases where bridging actually lost
ground.

`--synth-queries` builds queries out of the documents themselves, which makes a
rough estimate possible:

| strategy | what it uses | measured error against a real query log |
|---|---|---|
| `lead` | the first sentence | 0.14 |
| `title` | the first line | 0.17 |
| `keywords` | the longest distinct words | 0.22 |

`lead` and `title` usually hand the retriever the answer — a lead sentence is a
literal substring of its own document — so both models find it every time and the
estimate separates nothing. rebasis detects that and keeps the run provisional
rather than reporting the meaningless 1.00x it produces.

Even at its best the estimate carries about 0.22 of error against a real query
log, and the break-even it feeds is compared against 1.0. So a synthesised
estimate settles the question when it lands far from that line and is reported as
provisional when it does not. **A real query log is the only thing that settles
the near cases.**

### Models rebasis does not know

The profile table covers the common models; `rebasis adapter profiles` lists
them. For anything else, give the dimension:

```bash
rebasis probe --store ... --old my-old-model --new my-new-model \
  --old-dim 384 --new-dim 768
```

If the model encodes queries differently from documents, say so. rebasis will
not guess: a wrong prefix produces no error and only lowers quality, which is
the hardest kind of failure to attribute.

```bash
rebasis probe ... --new-dim 768 --query-prefix "query: " --document-prefix "passage: "
```

The old model needs no `--old-dim` when it is only being read from the index —
the index is authoritative about its own dimension.

## `rebasis fit`

Fit an adapter and write a `.rbs` file. Reads the index; never writes to it.

| Option | Default | |
|---|---|---|
| `--out` | required | Where to write the adapter |
| `--store`, `--old`, `--new` | required | As for `probe` |
| `--method` | auto | `auto`, or one adapter type |
| `--direction` | `query_to_old` | Which way the map points. `query_to_old` is what [`Bridge`](../concepts/adapters.md) serves with; `old_to_new` is what [`migrate`](#rebasis-migrate) rewrites the index with, and it is fitted and scored on a different question — see [what a completed migration is worth](../migration-band.md) |
| `--pairs` | 4000 | Matched pairs to fit on. Measured to saturate near 4000. |
| `--heldout` | 1000 | Held out to score candidates on |
| `--state-dir` | `./.rebasis` | Where the audit trail lives — `fit` records which adapter won and on what evidence |
| `--seed`, `--device`, `--k` | | As for `probe` |
| `--old-dim`, `--new-dim`, the four prefix options | | As for `probe`. `--old-dim` is rarely needed: the index is authoritative about its own dimension. |

## `rebasis eval`

Measure an adapter that already exists. With no `--store` it describes the file;
with one it re-scores it against a live index.

| Option | |
|---|---|
| `ADAPTER` | Path to a `.rbs` adapter |
| `--verify` | Recompute every tensor hash before loading |
| `--store`, `--queries`, `--sample`, `--heldout`, `--k`, `--seed`, `--device` | As for `probe` |

## `rebasis migrate`

**Experimental.** The only command that writes. Tested against every supported
backend on every commit, not yet proved at production scale — take a backup
rebasis is not part of, and try `--limit` on a slice first. See
[Migration and rollback](../guides/migration.md).

| Option | Default | |
|---|---|---|
| `--adapter` | required | Path to a `.rbs` adapter |
| `--store` | required | Store URI |
| `--batch` | 256 | Records per batch |
| `--limit` | — | Stop after this many |
| `--max-memory` | — | Ceiling, e.g. `2GB` |
| `--priority` | none | `access` migrates what you read first |
| `--access-log` | — | JSONL of `{"id": ..., "count": ...}` |
| `--keep-original/--no-keep-original` | keep | Shadow copy for rollback |
| `--shadow-precision` | float32 | `float16` halves the shadow's disk cost and makes the rollback close rather than exact — [measured at nDCG@10 within 0.002](../shadow-precision.md) |
| `--power-aware/--no-power-aware` | on | Pause on low battery |
| `--resume` | — | Continue an existing job id. Recovers `--adapter` and `--store` from the job. `rebasis resume JOB_ID` is the same thing under the verb that pairs with `pause` |
| `--health-check/--no-health-check` | on | Measure what the index's own search returns against exact kNN, before and after |
| `--rebuild-index` | off | When the run finishes, ask the store to rebuild its search structure |
| `--refit` | off | Refit the adapter part-way through, on records not yet migrated, adopting only a winner. Opens the new model recorded in the adapter's manifest and re-embeds documents |
| `--refit-every` | 50000 | Records between refit attempts |
| `--refit-pairs` | 1000 | Records sampled and re-embedded per attempt |
| `--device` | auto | Where to run the embedder `--refit` needs |
| `--dry-run`, `-n` | off | Print the plan and stop |
| `--state-dir` | `./.rebasis` | Where the job state, shadow copies and audit trail live |
| `--yes`, `-y` | | Skip the confirmation |
| `--no-input` | off | Never prompt; fail instead of asking |

While it runs, `migrate` shows an `X of Y` progress bar with elapsed and
remaining time — the queue knows the total before the first batch. Off a
terminal it prints one line per decile instead of an animation.

Before it writes anything, `migrate` takes the **exclusive state lock** — two
migrations against one state directory are refused, not interleaved — and prints
a disk-space plan: the shadow copy, the checkpoint and state, and the free space
needed with a safety margin. It stops there if the disk cannot take it. A
migration that fills the disk halfway through takes the shadow copy with it, and
the shadow copy is what makes the job reversible.

`rollback` and `gc --apply` take the same lock. `gc`'s dry run does not: "what
would you delete?" is a read, and making it wait behind a running migration would
be a worse answer than showing it.

**`--refit` is for one situation and declines the rest.** It samples records that
have **not** been migrated yet, re-embeds them with the new model, refits on
those pairs alone, and adopts the result only if it beats the adapter in use by
0.01 on a held-out slice. Measured over 216 cells: on a corpus that has not
changed nothing clears that threshold, and on an index that grew into a domain
the adapter never saw, 1,000 pairs drawn from what is left are worth a median
+0.16 nDCG and beat 8,000 pairs from the migrated half by +0.20. An adopted
adapter is written to the state directory and the job points at it, so `resume`
keeps it. [The numbers](../continuous-refit.md).

**Two things `migrate` reports that nothing else can see.**

`--health-check` measures whether the index can still *find* what was written to
it, which is a different question from whether the write landed. A graph index
picks each record's edges from the geometry of its neighbours at insert time, and
rewriting the vector does not rewrite the graph — so recall can fall while every
vector in the collection is correct and verified. It costs two scans of the
collection (18–21 seconds per 100,000 records) and reports the number without
attaching a threshold to it. `--rebuild-index` asks the store to rebuild
afterwards, where the backend supports that; it is off by default because
rebuilding changes the collection's own configuration, which is the user's index
rather than rebasis'. [The measurement](../index-health.md).

A run that stops short — `--limit`, a pause, a failed batch — leaves the
collection holding **both models' vectors**, and no query is correct against all
of it until the job finishes. `migrate` says so before the confirmation and
again on the way out; `status` says so until it is resolved.
[What that means](../guides/migration.md#stopping-short-leaves-two-spaces-in-one-index).

## `rebasis pause` · `resume`

```bash
rebasis pause JOB_ID [--state-dir DIR]     # takes no lock; safe during a migration
rebasis resume JOB_ID [--yes] [--limit N] [--batch N] [--max-memory 2GB]
                      [--power-aware/--no-power-aware]
                      [--health-check/--no-health-check] [--rebuild-index]
```

`pause` asks a running job to stop **after its current batch** and returns
immediately; the job stops a moment later. Killing the process was already safe
— the queue is the checkpoint and a shadow is written before the vector it
copies is overwritten — but it leaves the store holding a batch nobody verified,
and a stop at a batch boundary does not.

Like `status`, it takes no lock. The migration it is interrupting holds the
state lock for its whole run, so a command that waited for the lock would wait
for the thing it is trying to stop. What makes that safe is that `pause` writes
one column nothing else writes: it records a *request*, and only the engine ever
says what state a job is in. `status` shows an outstanding request as
`running (pausing)`, and `--json` carries it as a separate `pause_requested`
field so a script branching on `state == "running"` keeps working.

A request never outlives the run it was made for. It is cleared when a run ends
and again when one starts, so a request left behind by a killed process cannot
silently pause the next run.

`resume` continues a job from where it stopped, recovering the adapter and store
URI from the job row. It is `migrate --resume JOB_ID`, forwarded — only the
flags that describe *this run* are accepted. `--priority` and `--access-log` are
not among them: they order the queue, the queue was ordered when the job was
created, and re-ordering half a migration would be a different job.

## `rebasis status` · `rollback` · `gc`

```bash
rebasis status [JOB_ID] [--json] [--plain]   # takes no lock; safe during a migration
                                             # reports mixed_space until a partial migration finishes
rebasis rollback JOB_ID [--yes] [--no-input] # restores from the shadow copy
rebasis gc [--apply] [--dry-run] [--json] [--job ID] [--i-understand]
```

`gc` lists the shadow copy of a job that has already been rolled back and
removes it without `--i-understand`: `rolled_back` is terminal, so that copy can
never be used again. Shadows of jobs that can still be rolled back are named
explicitly and confirmed explicitly, as before.

## `rebasis audit`

```bash
rebasis audit list
rebasis audit show SEQ
rebasis audit verify                 # checks the hash chain
rebasis audit export --out trail.jsonl
rebasis audit replay SEQ             # re-runs a decision and compares
rebasis audit prune --before YYYY-MM-DD --i-understand
```

`list` filters with `--action` and `--since` and defaults to the last 20. Every
subcommand takes `--state-dir`.

`prune` removes records older than a date. It runs as a dry run without
`--i-understand`, because the records that go are the evidence and there is no
copy. It writes a record of its own first, inside the chain, saying where the gap
is — so `verify` reads a prune as a prune rather than as tampering.

`replay` re-runs the probe with the recorded inputs and nothing else — the two
models, the sampling strategy and seed, the sample size, the cut-off. It takes
`--store` to point at a store the record does not name, and `--device`.

| Verdict | Meaning | Exit |
|---|---|---|
| `matched` | Same decision, same ARR within tolerance | 0 |
| `matched_cross_device` | Same decision, on different hardware from the original | 0 |
| `differed` | A different answer — a regression, or a changed corpus | 3 |
| `not_replayable` | The record is not a decision, or predates a required field | 2 |

## `rebasis adapter`

```bash
rebasis adapter inspect a.rbs [--verify] [--json]
rebasis adapter upgrade a.rbs [--out b.rbs] [--force]
rebasis adapter profiles [SEARCH]
```

`upgrade` converts a `.rbs` file to the current schema. **The original is never
modified** — it writes a new directory, because schema migrations are
forward-only and the original copy is the only way back. Verify before removing
it:

```bash
rebasis adapter upgrade old.rbs --out new.rbs
rebasis eval new.rbs --verify
```

`profiles` lists the encoding profiles rebasis knows, marking the asymmetric
ones. It is where `RB-E2003` sends you.

## `rebasis doctor`

```bash
rebasis doctor [--store URI] [--calibrate] [--json] [--state-dir DIR]
```

What rebasis can see: backends, embedders, devices, BLAS threads, log level,
telemetry. Written to work in the most broken environment — run it first.
`--json` emits the same facts in a form that can be pasted into an issue.

`--store` points the same diagnostic at a live index: whether the URI parses and
the index opens, which backend it is and what it declares it can do, the record
count and dimensionality, whether document text comes back, SQLite's own
`PRAGMA integrity_check` wherever there is a file to reach, rebasis' manifest
integrity, and the encoding profile recorded against the collection.

The check that earns its place is the last one: **whether the collection holds
two embedding spaces at once**. `migrate --limit`, `--priority access` and every
pause leave it that way, and no query is correct against both — the count is
right, the text is right, the ranking is wrong. It costs one indexed aggregate
per unfinished job in the local manifest: no store is opened for it, no vector
is read, nothing goes over a network.

`--calibrate` times this machine and writes `.rebasis/calibration.json`. The
speedups in `rebasis.compute.thresholds` were measured on one GPU against one
CPU and a faster host narrows every one of them, so a diagnostic that repeated
them would be repeating somebody else's machine. It measures what it can reach
without downloading anything — kNN through the same `top_k_search` a probe
calls, and the residual MLP's fit where torch is installed — and records **only
those**: `embed` needs a model, so it is omitted rather than guessed.

It is a diagnostic and nothing more. Nothing in the runtime dispatches per
operation — a session runs under one ambient device and `top_k_search` consults
no size threshold — so a calibration changes what `doctor` reports about this
machine, not where work runs.

**Read-only in every path except `--calibrate`, which is named as the
exception.** Nothing is opened for writing. The manifest is opened only when this release would not migrate its
schema — `ManifestDB` upgrades on connect and takes a backup on the way, which
is right for `status` and wrong for a command whose whole promise is that it
changes nothing. A manifest from an older release is reported rather than
upgraded, and the schema is read out of the SQLite file header so that even the
decision is a read.

## `rebasis version`
