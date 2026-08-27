# Qdrant

The one backend of the five that can rebuild its own search structure without
downtime, and the only one where the same class serves a laptop and a cluster.

## The URI

```
qdrant:///path/to/local/db#documents      # embedded, no server
qdrant://localhost:6333#documents         # a running instance
```

The fragment is the collection and it is not optional — rebasis will not guess
which one you meant.

**Local mode takes an exclusive lock on its folder.** That is Qdrant's own
behaviour, not rebasis'. A second process cannot open the same path while the
first holds it — and rebasis holds it **for the whole command**, not just for the
reads. So a `rebasis migrate` against a local Qdrant blocks anything else that
would open that folder for as long as it runs, including a `rebasis probe` you
might want to run alongside it.

`rebasis status` is unaffected: it reads the manifest, not the store.

There is one deliberate exception, and it is worth knowing about because it is
the moment a migration could otherwise look stuck. When the queue empties, the
engine releases its handle, reopens on a **fresh connection**, and re-reads a
sample of what it wrote. That check exists because the handle that did the
writing is exactly the handle a caching client will answer from its own memory —
so it catches a store that took a write and did not keep it, which the per-batch
read-back cannot. Local Qdrant is precisely the backend that would refuse a
second simultaneous connection, which is why the release happens first.

If you need concurrent readers, run a Qdrant server rather than local mode. That
is the difference the two URIs buy you.

## Where your ids and text live

This is the part that needs a decision, and it is Qdrant-specific.

A Qdrant point carries a UUID or an integer id. **Your document id is almost
certainly not that** — it is a field in the payload, put there by whatever wrote
the collection. So is the text. rebasis resolves both:

- It looks for a payload field that looks like a document id, and reports that
  rather than the point id, mapping back when it writes.
- It looks for a payload field that looks like text.

When your payload does not use a conventional name, say so in the URI:

```
qdrant://localhost:6333#documents?id_key=doc_id&text_key=body
```

`vector_name` is there for the same reason. A Qdrant collection can hold several
named vectors per point; if yours does, name the one rebasis should read and
write:

```
qdrant://localhost:6333#documents?vector_name=dense
```

Check what it resolved before you rely on it:

```bash
rebasis doctor --store "qdrant://localhost:6333#documents"
```

If `text` comes back unavailable, `probe` and the bridge still work — the pairs
an adapter is fitted on need the index's vectors on one side and the same
documents re-embedded on the other, and without text there is nothing to
re-embed. `migrate` will refuse up front rather than halfway through.

## Reads stream

rebasis uses `scroll`, Qdrant's cursor, rather than `search`. It returns a page
and an offset to continue from, which is the streaming contract exactly and keeps
peak memory at `O(batch × d)` rather than `O(N × d)`. A 5-million-point
collection costs the same resident memory as a 50-thousand-point one.

## Migrating

```bash
rebasis fit --store "qdrant://localhost:6333#documents" \
  --old sentence-transformers/all-MiniLM-L6-v2 --new BAAI/bge-base-en-v1.5 \
  --direction old_to_new --out adapters/forward.rbs

rebasis migrate --adapter adapters/forward.rbs \
  --store "qdrant://localhost:6333#documents" --dry-run
```

Two things Qdrant gives you that the other backends do not.

**`--rebuild-index` actually works here.** Rewriting vectors does not rewrite the
HNSW graph that was built around them, so a migration can cost recall even when
every vector is correct. Qdrant documents rebuilding the index and documents that
it costs no downtime, and it is the only one of the five backends that declares
`can_rebuild_index`. Measured on 100,000 records, an orthogonal adapter moves the
index's own recall within measurement noise — so this is insurance rather than a
requirement, and `migrate` measures before and after and names any drop either
way.

**Quantization is declared.** If the collection stores narrow vectors, rebasis
says so through the backend's capabilities rather than letting you find out from
a round trip that does not match. Read
[If your index is stored quantized](migration.md#if-your-index-is-stored-quantized)
before migrating one: what it changes is what your rollback is worth.

## Batch sizes

The default `--batch 256` sits inside Qdrant's own documented recommendation of
64 to 256 points per upsert. There is little to gain by going higher and
something to lose: Qdrant's own guidance notes that parallel upload gains are not
linear.

## Server mode is less tested than local mode

Stated plainly. The contract suite and the migrate-and-rollback cycle run against
**local mode** on every commit, because it needs no service to install. A real
cluster differs in the ways a network differs from a function call — timeouts,
retries, a payload index that is being rebuilt while you write.

`rebasis doctor --store <uri>` is read-only in every path and is the cheap way to
find out whether your cluster looks the way rebasis expects before anything
writes.
