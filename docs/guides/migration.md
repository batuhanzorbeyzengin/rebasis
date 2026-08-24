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
restores the originals byte for byte.

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
kill -9 — and `--resume <job-id>` continues from the last completed batch. There
is no separate checkpoint file to get out of sync with reality.

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

```bash
rebasis migrate \
  --adapter adapter.rbs \
  --store chroma:///path/to/db#documents
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
| `--resume <job-id>` | Continue an interrupted job. |

## Watching it

```bash
rebasis status
```

Takes no lock, so it works while a migration is running — which is exactly when
you want it.

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
stays queued. Fix the cause, then `--resume`.

The one thing that never happens is a partially-written record: the shadow is
taken first, the write is one call, and the read-back verifies it.
