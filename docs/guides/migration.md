# Migration and rollback

`migrate` is the only rebasis command that writes to your index. Everything on
this page exists because of that.

## Status: experimental

`migrate` works, and every guarantee below is covered by a test that runs against
a real store on every commit. It is marked experimental for one reason: the
evidence behind it is a test suite, not a fleet of production indexes. Nobody has
yet run it against a million-record index they could not rebuild.

What that means in practice:

- **Take a backup you can restore without rebasis.** Not `--keep-original` — that
  is rebasis' own shadow copy, and it protects against a bad adapter, not against
  a bug in rebasis. Copy the directory, snapshot the volume, export the
  collection. Whatever your store offers.
- **Migrate a slice first.** `--limit 5000` on a real collection, then look at
  what it did. `rollback` is one command away.
- **Read the backend table below.** Support is not uniform, and the difference is
  in what has been *tested*, not in what the code claims.

The other commands — `probe`, `fit`, `eval` — never write, and carry no such
caveat.

### Backends

Each of these runs the full migrate-and-rollback suite on every commit: the
vectors actually change, the record count does not, text survives, and `rollback`
restores the originals byte for byte. "The originals" means something narrower
on a store that keeps compressed codes rather than vectors — *If your index is
stored quantized*, below, is what rebasis detects and what it then says.

| Backend | Migrate | Notes |
|---|---|---|
| `chroma` | tested | |
| `lancedb` | tested | |
| `qdrant` | tested | Local mode holds an exclusive lock on its folder; rebasis releases it on `close()`. |
| `sqlite-vec` | tested | |
| `faiss` | tested | Needs an `IndexIDMap2` and a `.meta.json` sidecar. A write reorders the index, so the sidecar is rewritten with it. |
| `memory` | tested | Not persistent; used by the durability tests. |

Scale is the untested dimension, not the backend list. The suite runs on
hundreds of records, not millions.

## What it guarantees

**It only upserts.** It replaces vectors in existing records. It never deletes a
record, never removes a payload field, never drops metadata.

**It copies before it overwrites.** Each batch's original vectors go to a shadow
store *before* the new ones are written. A crash before the write costs nothing;
a crash after the write, without the shadow, would cost the originals.

**The queue is the checkpoint.** Interrupt it — Ctrl-C, a closed laptop, a
kill -9 — and `rebasis resume <job-id>` continues from the last completed batch.
There is no separate checkpoint file to get out of sync with reality.

**It verifies what it wrote.** After each batch, a sample of the written records
is read back from the store and compared. A store that silently fails to write is
the most common source of silent data loss, and this is the check that catches it.

**It checks again on a fresh connection at the end.** The per-batch read-back
goes through the handle that did the writing, and a client that caches will
happily hand back what it has in memory rather than what reached disk. So when
the queue empties, `migrate` reopens the store from its URI and re-checks a
sample. That is the difference between "the store accepted it" and "the store
kept it".

## Running it

`migrate` needs an adapter pointing **out** of the index, not into it. That is
the one thing it cannot be run without, and until this release nothing produced
one — so the documented sequence handed it the query map instead, and every guard
the tool had let that through. The reasoning is in [what changed](#what-changed-and-why),
below; the short version is that an index rewritten with a query map answers
recall@1 **0.000** to every query type there is.

```bash
# The map migrate needs. Note --direction; without it `fit` produces the
# query-side map, which `Bridge` serves with and `migrate` now refuses.
rebasis fit \
  --store chroma:///path/to/db#documents \
  --old <old-model> --new <new-model> \
  --direction old_to_new \
  --out forward.rbs

rebasis migrate --adapter forward.rbs --store chroma:///path/to/db#documents
```

It shows what it will do — how many records, which store, which adapter, which
job id, whether rollback is available — then a disk-space plan: the shadow copy,
the checkpoint and state, and the free space needed with a safety margin. If the
disk cannot take it, it stops there rather than filling the disk halfway through
and taking the shadow copy with it. Then it asks before starting; `--yes` skips
the question for scripts.

It holds the exclusive state lock for the whole run, so a second `migrate`,
`rollback` or `gc --apply` against the same state directory is refused rather
than interleaved. `rebasis status` and a bare `gc` take no lock.

Useful flags:

| Flag | What it does |
|---|---|
| `--batch 256` | Records per batch. Adjusted automatically under memory pressure. |
| `--limit 5000` | Stop after this many. Migrate an evening at a time. |
| `--max-memory 2GB` | A ceiling. The batch size is computed from it — you should not have to. |
| `--priority access --access-log log.jsonl` | Migrate what you actually read first, so quality improves where you will notice. |
| `--power-aware/--no-power-aware` | Pause on battery. On by default. |
| `--refit` | Refit the adapter part-way through, on records not yet migrated. Off by default; see below. |
| `--resume <job-id>` | Continue an interrupted job. `rebasis resume <job-id>` is the same thing. |

### Whether to run it at all

Measured across 51 runs on seventeen corpora with human relevance judgements
([the band](../migration-band.md)):

| | |
|---|---|
| a completed migration delivers | **0.727** of a full reindex, on average |
| bridging, on the same runs | **0.719** |
| the two track each other at | Spearman **0.993** |
| migrating beat leaving the index alone in | **5 of 51** |

So migrating and bridging are worth the same amount, and both are usually worth
less than doing nothing. That is
[ADR 10](../adr/0010-retention-is-bounded-by-the-source.md) reaching the document
side: the same source space under the same family of map carries the same
amount, whichever end you apply it to.

**What migrating buys is the adapter leaving the query path** — no map on the hot
path, no `.rbs` to ship with your service, and the new model querying its own
space. What it costs is rewriting every vector, the shadow copy behind it, and a
window in which the index holds two spaces. Choose on those grounds; quality is
not one of them.

### What changed, and why

An adapter has a direction, and the two are mirror images:

| direction | maps | used by |
|---|---|---|
| `query_to_old` | a new-model **query** into the index | `rebasis.Bridge` |
| `old_to_new` | the **stored vectors** into the new model's space | `rebasis migrate` |

`rebasis fit` produced only the first, `migrate` never checked, and the README
showed one being piped into the other. Applying the query map to document vectors
passed every guard: the write landed, the count held, the text survived, the
read-back compared what was written against what came back, and the index-health
check measured the store's search against exact kNN over the vectors it now held.
None of those asks whether the vectors still mean anything.

Measured on data where both spaces are known exactly and the bridge itself scores
recall@1 1.000 against the untouched index, the index a completed migration left
behind answered **0.000** to a raw new-model query, **0.000** to a bridged query
and **0.000** to an old-model query. Where the two models have the same width it
failed silently; where they differ it failed with a dimension error at query time,
which is why it survived on some ladders and not others.

Both directions are now guarded. `migrate` refuses a `query_to_old` adapter before
it opens the store, and `Bridge.load` refuses an `old_to_new` one.

## Stopping short leaves two spaces in one index

`--limit`, `--priority access` and every pause all do the same thing to the
collection: for as long as the job is unfinished, **some records hold the new
model's vectors and some still hold the old model's**, and there is no query
that is correct against both.

```
bridge.to_index_space(q)   correct for the records that have not moved
f_new(q)                   correct for the records that have
```

Whichever you send, part of the corpus is scored against a geometry it is not
in. Nothing raises. The record count is right, the text is right, the ranking is
wrong — and on a graph index the traversal itself is running over the mixture,
so the damage is not limited to the scores of the records that moved.

This is the reason `--limit` is recommended above as a way to *try* a migration
rather than as a way to *pace* one. It is safe for the data — the shadow copy is
intact either way — and it is not safe for queries in the window before the job
finishes.

rebasis will not let that window be silent. It is named three times:

- in `migrate`'s preview, before you confirm, whenever `--limit` will stop the
  run short;
- at the end of any run that did stop short;
- by `rebasis status`, unprompted, until the job finishes or is rolled back —
  including in `--json`, as `mixed_space`, so a script can refuse to serve.

```
$ rebasis status
...
This index holds two embedding spaces. Search results are not correct until
the migration finishes or is rolled back.
  chroma:///path/to/db#documents holds two embedding spaces: 5,000 of 48,000
  records (10%) have the new model's vectors and 43,000 still have the old
  model's. …
    rebasis migrate --resume job-8f2a1c4e0b73   (finish it)
    rebasis rollback job-8f2a1c4e0b73           (put the index back)
```

### Searching one anyway

Finishing the job or rolling it back are the two ways to *resolve* a mixed
index. There is a third thing you can do while it is mixed, which is search it
correctly:

```python
from rebasis.serve import Bridge, MixedSpaceSearch

bridge = Bridge.load("adapters/minilm-to-bge.rbs")
with MixedSpaceSearch(store, bridge, job_id="job-8f2a1c4e0b73") as search:
    hits = search.search(new_model.encode(["how do I deploy?"])[0], k=10)
```

It sends **both** queries and keeps only the half each one is right about — the
bridged query for the records that have not moved, the raw new-model query for
the ones that have — then merges the two through the isotonic calibrator in the
`.rbs`, which is what makes scores from two spaces comparable at all. Without a
calibrator it falls back to reciprocal rank fusion, which discards the scores
and uses ranks: strictly less information, and correct, where comparing raw
scores across two spaces is not.

Which records have moved is read from the **manifest**, not from the store.
rebasis does not write a `rebasis_space` field into your payloads; the whole
store contract is one write path that only ever replaces vectors, and the
migration queue already knows what it moved.

The cost is over-fetching: each side asks deeper than `k` and discards what
belongs to the other, scaled by how far the migration has got.
`search.over_fetch` reports it after every query, and the cheapest way to bring
it down is to finish the job.

## Watching it

```bash
rebasis status
```

Takes no lock, so it works while a migration is running — which is exactly when
you want it.

## Refitting part-way through

```bash
rebasis migrate --adapter adapters/forward.rbs --store ... --refit
```

Off by default. It samples records that have **not** been migrated yet,
re-embeds them with the new model — the one recorded in the adapter's manifest,
so there is no second flag to get wrong — refits on those pairs alone, and
adopts the result only if it beats the adapter in use by 0.01 on a held-out
slice.

**It is for one situation.** Measured over 216 cells, on a corpus that has not
changed a refit is a pair-count effect worth a median +0.0075 at three times the
fit budget, and nothing clears the adoption threshold. On an index that **grew
into a domain the adapter never saw** — a vault that gained a department while
the migration was running — 1,000 pairs drawn from what is left are worth a
median **+0.16 nDCG** and beat 8,000 pairs drawn from the migrated half by
**+0.20**. [The numbers](../continuous-refit.md).

The guard is why it is safe to leave on in either case: it declines in the first
and adopts in the second, and every attempt is audited with its reason.

An adopted adapter is written to `.rebasis/adapters/<job>-refit-<n>.rbs` and the
job is pointed at it, so `rebasis resume` continues with it rather than
reloading the file the job started with. It needs a store that returns document
text — there is no way to make a real pair without re-embedding one — and it
says so at the start of the run rather than at the first checkpoint.

## Stopping it, and starting it again

```bash
rebasis pause <job-id>     # stops after the current batch
rebasis resume <job-id>    # picks up where it stopped
```

Killing the process is safe and always was: the queue is the checkpoint, and a
shadow copy is written before the vector it copies is overwritten. What it is
not is *clean*. A kill lands in the middle of a batch, and that batch's records
are left in the store without having been read back and compared — verified
writes are a per-batch guarantee, and half a batch does not get one.

`pause` stops at a batch boundary instead. It returns immediately and the job
stops a moment later, at the end of the batch it is in.

Like `status`, it takes no lock — the migration holds the state lock for its
whole run, so waiting for the lock would mean waiting for the thing you are
trying to stop. What makes that safe is that `pause` writes one column nothing
else writes. It records a **request**; only the engine ever says what state a job
is in. While the request is outstanding, `status` shows the job as
`running (pausing)`, and `--json` carries it as a separate `pause_requested`
field.

A request never outlives the run it was made for: it is cleared when a run ends
and again when one starts. A process killed between `rebasis pause` and the
engine reading it cannot leave behind a flag that silently pauses tomorrow's run.

A paused job is a job stopped short, so everything under
[Stopping short leaves two spaces in one index](#stopping-short-leaves-two-spaces-in-one-index)
applies to it. Pausing is not a way to stop safely and walk away; it is a way to
stop safely and come back.

## Rolling back

```bash
rebasis rollback <job-id>
```

Restores the original vectors from the shadow copy. The shadow is written at
float32, the same precision the vectors were read at, so what is *kept* is
bit-identical to what was there.

What is *restored* is that, written back through the store's own upsert — which
is where the guarantee stops being rebasis'. A store that keeps what it is given
returns the same bytes. A store that normalises on write does not: Chroma in
`hnsw:space=cosine` shifts a vector by about 3e-08, and it does that to a plain
write-and-read with rebasis nowhere in it. The same collection at `l2`
round-trips exactly.

In practice that is a cosine similarity of 1.000000 against the original. It is
not quite nothing: measured over three migrate-and-rollback rounds on a 5,183
document Chroma collection, recall@10 for one query in 300 moved, because a
result sitting on the k boundary changed sides. Only ties that close can move at
all — a shift of 3e-08 cannot reorder anything genuinely separated.

It is reported here because "bit for bit" is a claim somebody will check, and on
a normalising store they would find it false.

The job records which store it wrote to, so you do not have to remember the URI
months later — which is when a rollback is actually wanted.

## If your index is stored quantized

Everything above assumes the store keeps what it is given. Increasingly it does
not: int8 scalar quantization, product quantization and binary codes are how a
large index is made to fit, and a store that holds a code cannot hand back the
vector the code was made from.

`migrate` checks before it writes anything and says so in the plan, above the
confirmation. It does not refuse. A quantized index is a deliberate engineering
choice and migrating one is a legitimate thing to want; what you would not have
without the check is a correct reading of the paragraph above.

**What the shadow copy holds.** rebasis shadows what the store *returns*, and a
quantized store returns a value decoded from its stored code. The shadow is
still bit-identical — to that decoded view. It is not a copy of the vectors your
embedding model produced; those stopped being retrievable when the collection
was built.

**What `rollback` therefore restores.** The state the migration replaced,
exactly: the vectors this collection read back the moment before `migrate`
started. It does not recover precision the collection had already spent.

**It can stop the run.** After every batch `migrate` re-reads a sample and
compares it to what it sent, to `VERIFY_ATOL` — the constant is in
`src/rebasis/migrate/engine.py`, and `migrate --dry-run` prints whatever it
currently is rather than a figure copied into prose. That check exists to catch
a store that accepts a write and does not keep it; a store that re-encodes on
write fails it for a different reason. Measured in
`tests/integration/test_quantized_roundtrip.py`, an 8-bit scalar-quantized FAISS
index deviates by more than that tolerance in both directions — so on a codec
that coarse the job stops on its first batch, with the shadow copy already
written and nothing lost.

### What each backend reports

`StoreCapabilities.quantized` has three values, and the third is the point.
`False` is a promise that what you write is what you read back; a backend that
answered `False` without looking would be making a guarantee it could not keep.
So the default is `None` — *not determinable* — and that is what a third-party
store behind the LangChain or LlamaIndex bridge honestly is.

| Backend | Read from | Reports |
|---|---|---|
| `faiss` | `sa_code_size()` against `4 × d`, through the `IndexIDMap2` wrapper | `True` for PQ, scalar-quantized, LSH and friends; `False` for a flat index |
| `sqlite-vec` | `vec_type()` on a stored vector | `float32` → `False`; `int8` and `bit` → `True`; empty table → `None` |
| `qdrant` | `VectorParams.datatype` in the collection config | `float16`, `uint8`, `turbo4` → `True`; otherwise `False` |
| `lancedb` | the Arrow element type of the vector column | narrower than 32 bits → `True` |
| `chroma` | nothing to read | `False` |
| `memory` | nothing to read | `False` |

Two of those answers are deliberately narrower than they first look, and both
are worth knowing if you go looking for a warning that does not appear.

**Qdrant's `quantization_config` is not this.** Qdrant builds the quantized
codes *beside* the vectors rather than instead of them — which is what makes its
own rescoring possible — so a scalar- or binary-quantized Qdrant collection
still returns the original vector, and `rollback` on one is exact. Qdrant draws
the line itself: "datatypes are distinct from the quantization feature.
Quantization creates a separate quantized representation of vectors alongside
the original ones, while datatypes determine the representation of the original
vectors themselves." So it is `datatype` that rebasis reads.

**LanceDB's `IVF_PQ` is not this either.** The compressed copy lives in the
index's own columns and the vector column is untouched, which is why LanceDB
documents `bypass_vector_index` as a way to get ground-truth results. What does
change the round trip there is a vector column that is not float32 — `uint8`
columns are a supported way to store binary embeddings — and that is what is
read.

**FAISS is the one where a quantized index used to pass silently.** rebasis
already refuses a FAISS index it cannot reconstruct from, but that catches only
the ones where `reconstruct` *raises*, such as an IVF index with no direct map.
An `IndexPQ` or an `IndexScalarQuantizer` reconstructs happily and returns a
decode. Those are now declared rather than refused.

### Reading it from a script

`migrate` has no `--json`. The finding is written into the audit trail with the
job, as `store_quantized` among the inputs of the `migrate.job.started` record:

```bash
rebasis audit export --out trail.jsonl
jq 'select(.action == "migrate.job.started") | .inputs.store_quantized' trail.jsonl
```

`true`, `false` and `null` are three different answers there, and `null` means
the store could not be asked — not that it keeps what it is given.

## `--no-keep-original`

Disables the shadow copy. It saves disk and it makes the migration
**irreversible**.

rebasis will not let that happen quietly: it prints a warning before starting
and records the choice in the audit trail. If the result is not what you
expected afterwards, the original vectors are gone.

## Cleaning up

```bash
rebasis gc              # a dry run: lists what could be removed
rebasis gc --apply
```

A dry run by default. A garbage collector that deletes without being asked is
the exact class of data loss it exists to prevent.

Removing a *shadow copy* makes that job permanently irreversible, so it needs
`--i-understand` on top of `--apply`.

## When a batch fails

The batch is marked `FAILED`, the job stops, and the failing records keep their
error code. Everything already written stays written; everything not yet written
stays queued. Fix the cause, then `rebasis resume <job-id>`.

The one thing that never happens is a partially-written record: the shadow is
taken first, the write is one call, and the read-back verifies it.
