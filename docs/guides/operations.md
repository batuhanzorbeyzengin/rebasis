# Running rebasis somewhere that is not your laptop

rebasis is a command. This page is about being a good citizen inside the things
that run commands — a Kubernetes `Job`, an Airflow task, an Argo step, a cron
entry — and about the questions that come up the first time somebody tries to put
it behind a change-control process.

## Exit codes

A contract, not an implementation detail. Changing one is a breaking change.

| | | |
|---|---|---|
| `0` | Success | The run did what it was asked |
| `1` | Unexpected | A bug in rebasis. It prints an issue link |
| `2` | Usage | A bad flag, a missing argument, an unparseable query log |
| `3` | Domain | The run completed and the answer is "no" |
| `130` | Aborted | Interrupted, or a confirmation declined |

**`3` is the one worth wiring up.** A `probe` that runs perfectly and decides
against bridging is not a failure, and a pipeline that treats every non-zero code
the same will retry it forever. Airflow's `BashOperator` takes a
`skip_on_exit_code`; a Kubernetes `Job` can distinguish exit codes with a Pod
Failure Policy.

## Machine-readable output

Every command that reports takes `--json`. Every command that prompts takes
`--yes` and refuses to guess without it; `--no-input` makes prompting an error
rather than a hang.

**Progress goes to stderr and `--json` goes to stdout**, so the two never collide.
That is what lets you watch a long run and parse it at the same time:

```bash
decision=$(rebasis probe --store "$STORE" --old "$OLD" --new "$NEW" \
  --queries queries.jsonl --json | jq -r .decision)

# Refuse to serve from an index that is halfway between two models.
rebasis status --json | jq -e 'all(.mixed_space == null)' >/dev/null
```

Adding a key to `--json` is not a breaking change. Removing or renaming one is —
see [stability and support](../stability.md).

## Being terminated

**Kubernetes, Airflow and Argo all end a process the same way: SIGTERM, a grace
period, then SIGKILL.** Kubernetes' default grace period is thirty seconds.

`rebasis migrate` catches the first SIGTERM and turns it into the same request
that `rebasis pause` makes: the run stops **at the next batch boundary**, records
the job as paused, and says which signal asked. `rebasis resume <job-id>` picks it
up. The second signal is not caught — a supervisor escalating, or a second
Ctrl-C, stops the process at once, because a graceful stop that cannot be
interrupted is a hang with better manners.

**Set the grace period above your batch duration.** A batch that outlasts it is
still killed part-way, and no handler changes that. Two ways to make them fit:

```yaml
# Kubernetes: give the batch in flight room to finish.
spec:
  template:
    spec:
      terminationGracePeriodSeconds: 300
```

```bash
# Or make the batch smaller than the period you have.
rebasis migrate --batch 64 ...
```

Being killed outright has always been survivable — the queue is the checkpoint
and the shadow copy is written before the vector it replaces is overwritten. What
the handler adds is that the store is not left holding a batch nobody verified.

**A library caller keeps their own signal handling.** The handler is installed by
the CLI, not by `MigrationEngine`, so embedding rebasis does not take SIGTERM
away from your application.

## Re-running the same thing twice

`migrate` is resumable rather than idempotent, and the distinction matters:

- **The queue is the state.** Each record is `pending`, `shadowed`, `done` or
  `failed` in a local SQLite manifest. A re-run continues from that, it does not
  start over.
- **`rebasis migrate --resume <job-id>`**, or the `resume` command, is how a
  retry should be spelled. Starting a fresh job over the same collection would
  migrate already-migrated vectors a second time, which for a non-orthogonal
  adapter is not a no-op.
- **A failed batch does not fail the job.** Records are marked `failed` with the
  error code and the run continues; `rebasis status` lists them.

So a `backoffLimit` retry that re-runs the same command is safe if the command
names a job id. If it does not, it is a new job.

## Configuration by environment

Everything a long-running invocation needs can be set without a flag, which is
what `12-factor` config means in practice and what a Kubernetes `env:` block
wants.

| | |
|---|---|
| `REBASIS_DEVICE` | `auto`, `cpu`, `cuda`, `mps` |
| `REBASIS_STATE_DIR` | Where the manifest, shadow copies and adapters live |
| `REBASIS_CACHE_DIR` | The embedding cache |
| `REBASIS_LOG_LEVEL`, `REBASIS_LOG_FORMAT` | `json` is the format to want here |
| `REBASIS_MAX_MEMORY` | A ceiling, e.g. `4GB` |
| `REBASIS_ENV` | `server` selects INFO/json defaults |
| `REBASIS_OTEL_ENABLED` | Off unless set |
| `NO_COLOR` | Honoured |

`rebasis doctor --json` prints what it resolved, which is the fastest way to find
out that a variable did not reach the process.

## Secrets

**rebasis holds no credentials of its own and stores none.** It reads what the
libraries it drives read, from the environment:

| | |
|---|---|
| An OpenAI-compatible endpoint | `OPENAI_API_KEY`, `OPENAI_BASE_URL` |
| A hosted Qdrant | the API key in the store URI, or the client's own variable |
| Anything else | that backend's own convention |

Three properties, and each is a rule rather than a habit:

- **A credential never reaches a log.** Logging uses an allowlist: a field that is
  not explicitly permitted is redacted. Credentials, document text, vectors,
  query text and filesystem paths are excluded categorically.
- **A credential in a store URI is redacted even when the URI cannot be parsed.**
  That path used to quote the string it had rejected, which put a password in the
  error panel of every command taking `--store` — and from there into
  `rebasis doctor --json`, which this project tells people to attach to bug
  reports. It is now matched by shape, because the moment a URI most needs
  redacting is the moment it could not be parsed.
- **The manifest records parameters, never content and never credentials.** So
  does the audit trail.

**There is no Vault integration and there should not be.** A secret manager's job
is to put the value in the environment of the process that needs it; every
comparable single-machine tool — Terraform, Flyway, the Pinecone and Qdrant CLIs
— treats that as the orchestrator's work rather than the tool's. If your platform
injects secrets as environment variables, rebasis already works with it, and
adding an SDK would only be a second way to do what already works.

## One writer at a time

`migrate`, `adapter upgrade`, `gc` and schema migrations take an exclusive lock on
the state directory. `probe`, `fit`, `eval`, `audit` and `status` take none —
which is what lets you watch `rebasis status` while a migration runs, exactly
when you most want to.

**A stale lock is never broken automatically.** The holder's PID, the operation
and the start time are recorded, and rebasis will tell you whether that process is
still alive and suggest removing the file — but it will not guess. Being wrong
about a process being dead is how two writers end up in one manifest.

The lock is a file in the state directory. **It does not coordinate across
machines.** If two hosts share a state directory over NFS, `flock` semantics there
are historically inconsistent and this lock cannot be relied on. One writer, one
host.

## Logs

`REBASIS_LOG_FORMAT=json` gives one JSON object per line on **stderr**, keyed by a
stable event name. `docs/reference/events.md` is the generated catalogue of every
event and the fields it carries — generated, and CI fails if it drifts from the
code.

What is never in a log line, at any level, including `DEBUG`: document text,
vectors, query text, credentials, filesystem paths. Text can be reconstructed
from an embedding, which makes a vector in a log file as sensitive as plaintext.
`--unsafe-log-content` turns that off, warns in red on every run, and writes
`config.unsafe_logging_enabled` into the audit trail; it cannot be used quietly.

## Tracing

Optional, off by default, and exports to **your own** collector:

```bash
pip install "rebasis[otel]"
export REBASIS_OTEL_ENABLED=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4317
```

`examples/otel/` has a working collector configuration.

Two things to know before you build a dashboard on it. The `gen_ai.*` namespace
is still development-status upstream and its attribute names may move; the names
rebasis emits are isolated in one module so the blast radius of a spec change is
contained. And **there is no vector-database semantic convention at all** — the
upstream issue asking for one is open and unassigned — so store spans carry the
stable `db.system.name` and rebasis invents no `db.vector.*` namespace of its own.

## Installing where there is no internet

The core install has no torch and no heavyweight dependencies, which is most of
what makes this tractable.

**Mirror the wheels, not the sdists.** Everything rebasis depends on publishes
wheels; a source build behind a firewall is where an air-gapped install usually
fails. `uv export` produces the exact set:

```bash
# On a machine that has the internet:
uv export --all-extras --no-emit-project --format requirements.txt --no-hashes -o rebasis.txt
pip download -r rebasis.txt -d wheels/ --only-binary :all:

# On the machine that does not:
pip install --no-index --find-links wheels/ rebasis[chroma]
```

A private index — Artifactory, Nexus, devpi — works the same way: rebasis makes no
network call of its own at install time or at run time.

**The models are the hard part, not the package.** `sentence-transformers` and
`fastembed` download weights on first use, and that is the call that fails in an
air-gapped environment. Fetch the model on a connected machine, copy the cache,
and point the offline machine at it:

```bash
export HF_HOME=/opt/models/huggingface
export HF_HUB_OFFLINE=1
```

Two alternatives avoid the question entirely: an OpenAI-compatible endpoint inside
your own network — `ollama`, `infinity`, anything that speaks the API — or the
`precomputed` embedder, which takes vectors you already hold and runs no model at
all.

**rebasis itself sends no data anywhere.** No phone-home, no usage counter, no
version check. The only outbound traffic is what your chosen embedding backend and
your chosen store make.
