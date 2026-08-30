# pgvector

The store most teams already run, and the only one of the six where the
migration's weakest guarantee is the database's rather than rebasis'.

```bash
pip install "rebasis[pgvector]"
```

The driver is **pg8000**, not psycopg, and the reason is a licence rather than
a feature. psycopg is the obvious choice and this backend was written against
it; it is LGPL-3.0-only, and rebasis is Apache-2.0 and meant to sit inside
products a copyleft dependency would exclude it from — so `ci.yml`'s dependency
review denies that family, and caught it on the first pull request. pg8000 is
BSD-3-Clause and pure Python. It has no named-cursor API and no
`connection.transaction()`, so the two places that need them issue
`DECLARE`/`FETCH` and `BEGIN`/`COMMIT` themselves.

## The URI

```
pgvector://user:password@host:5432/dbname#public.documents
pgvector://user@host/dbname#documents          # schema defaults to public
```

The fragment is the table, optionally schema-qualified. The database is the
path. Credentials are parsed out of the URI and **never** reach a log line, an
audit record or an error message — `rebasis doctor --json`, which the README
tells people to attach to a bug report, prints `pgvector://<credentials>@host`.

## Which column is which

This is the part that needs a decision, and it is pgvector-specific.

A Chroma collection has a shape. A Postgres table is whatever its owner made,
so rebasis has to be told — or find the conventional name and say which it
found:

| | tried, in order |
|---|---|
| vector | `embedding`, `vector`, `embeddings`, `vec` |
| id | `id`, `doc_id`, `chunk_id`, `key`, `uuid` |
| text | `text`, `content`, `document`, `chunk`, `body` |

When your schema uses something else, name it:

```
pgvector://…/db#public.chunks?vector=emb&id=chunk_id&text=body
```

**Nothing is inferred from a column merely being the only one of its type.** A
table with two vector columns would otherwise have `migrate` rewrite whichever
one the catalogue happened to return first. If no conventional name is there,
the error lists the table's own columns with their types and names the option
that supplies the missing one.

A missing **text** column is not an error — it is `can_read_text: false`, and it
means `probe` and the bridge still work while `Cascade` and any fit that needs
to re-embed do not. Check what was resolved before relying on it:

```bash
rebasis doctor --store "pgvector://user@host/db#public.documents"
```

## The column type decides almost everything

pgvector has four vector types and they are not interchangeable. rebasis reads
the declared type from the catalogue — through `format_type`, because
`information_schema` reports every extension type as `USER-DEFINED` and cannot
tell `vector` from `halfvec` — and behaves differently per type.

| type | what it holds | read | write | `quantized` |
|---|---|---|---|---|
| `vector` | float32 | yes | yes | `false` |
| `halfvec` | float16 | yes | yes | **`true`** |
| `bit` | a binary code | no | no | `true` |
| `sparsevec` | an index/value map | no | no | `true` |

Measured, not assumed (`tests/integration/test_pgvector_types.py`):

- A `vector` column returns **exactly** what was written. That is the promise
  `rollback` rests on.
- A `halfvec` column rounds, and by **more than `migrate`'s read-back
  tolerance** of 1e-4. So a migration into a `halfvec` column would stop on its
  own first batch — the pre-flight plan says so beforehand rather than letting
  it surface as a failed write. Read
  [If your index is stored quantized](migration.md#if-your-index-is-stored-quantized).
- `bit` and `sparsevec` hold a code rather than a reconstruction of the vector
  that produced it, and pgvector scores them with Hamming, Jaccard or its own
  sparse operators rather than the cosine distance rebasis speaks. Both are
  refused with the reason, at the moment they are asked.

The precedent for reading the type rather than reasoning about it is
`sqlite-vec`, where an `int8` column was once read as if it held float32 and
every number derived from the reported dimension was a quarter of the truth.
Nothing raised.

## What the transaction takes over

`migrate`'s durability chain is four mechanisms: a shadow copy before every
batch, a read-back after, a fresh-connection check when the queue empties, and
`rollback`. All correct, and all durability rebuilt above the storage engine.

On pgvector, **one batch is one transaction**. It lands whole or it does not
exist, so the half-written batch stops being a state anything has to detect or
undo. `migrate`'s pre-flight plan names this, because knowing which layer holds
which guarantee is the difference between trusting a tool and hoping.

**The shadow copy stays**, and that is deliberate. A transaction rolls back one
batch; `rebasis rollback <job-id>` rolls back a finished job three days later.
Different scopes, and the second is the one a user asks for.

The connection is opened in **autocommit**, which is the other half of the same
decision. DB-API's default opens a transaction on the first statement and holds
it until something commits — so a `probe` against a live database would sit
`idle in transaction` for the length of the run, pinning the vacuum horizon of a
production table for a read that needed no transaction at all. The two places
that genuinely need one open it explicitly.

## Rebuilding the index

pgvector is the **second** backend that can rebuild its own search structure,
and the first where the two index types answer differently.

```bash
rebasis migrate --adapter adapters/forward.rbs \
  --store "pgvector://user@host/db#public.documents" --rebuild-index
```

`REINDEX INDEX CONCURRENTLY`, not a plain `REINDEX`. The plain form takes an
ACCESS EXCLUSIVE lock and stops every read of the table for its duration, which
is not a thing a tool should do to somebody's production index. The concurrent
form builds the replacement beside the original and swaps, at the cost of more
disk and a longer run. If it fails part-way it leaves an invalid index behind;
`\d` on the table names it and `DROP INDEX` removes it. The vectors are
untouched either way.

**Which index you built decides whether this is insurance or part of the job.**
Measured on 100,000 records with the orthogonal map `auto` picks
([the numbers](../index-health.md#pgvector-the-index-type-decides-and-the-control-stops-being-one)):

| | before | after | after `REINDEX CONCURRENTLY` |
|---|---|---|---|
| HNSW | 0.970 | 0.913 | 0.956 |
| IVFFlat | 0.853 | **0.308** | 0.838 |

(Re-measured under the shipped `pg8000` driver at 0.896 → 0.322 → 0.875 and
0.964 → 0.893 → 0.960; the table above is the `psycopg` run these were checked
against, and the difference is the repeat spread of a non-deterministic index
build.)

**On IVFFlat a migration costs two thirds of the index's recall**, because its
list centroids were computed once from a distribution the migration rotated.
Every vector is correct and in the wrong list. The rebuild recovers essentially
all of it. On HNSW the loss is 6 points and the rebuild recovers most.

IVFFlat also migrates two to four times faster, because maintaining a list
assignment on write is cheaper than maintaining a graph. That is the trade: a
slower migration that degrades a little, or a faster one that degrades a lot and
must be reindexed.

A table with no index on the vector column has nothing to rebuild, and that is
not a failure — an exact scan cannot lose recall.

## Reads stream

`iter_records` opens a **server-side cursor** and fetches `batch_size` rows at a
time. A client-side cursor reads the whole result set into the
client before yielding the first row, which is exactly the `O(N × d)` peak the
store contract forbids. A five-million-row table costs the same resident memory
as a fifty-thousand-row one.

## What pgvector supports

| Capability | Supported |
|---|---|
| Read vectors | yes, on `vector` and `halfvec` |
| Read text | if a text column was found or named |
| Upsert vectors | yes, on `vector` and `halfvec` |
| Metadata filter | **yes** — a SQL `WHERE`, the richest of the six |
| Dimension locked | **yes** |
| In-place update | yes |
| Rebuild index | **yes** |

The locked dimension is `vector(n)`'s own type modifier. Changing it is DDL, and
DDL is the thing rebasis does not do: `migrate` changes vectors, not schemas. A
move from `vector(1024)` to `vector(256)`, or from `vector` to `halfvec`, is
yours to perform — and [`probe --truncate`](../truncation-band.md) is what tells
you whether it is worth performing.

## Scope, stated plainly

**One table, named columns.** A joined schema, a partitioned table, text in a
second table: all refused explicitly rather than half-supported silently. The
narrowing is the point — a backend that half-works on a schema shape is worse
than one that says which shape it works on.

**This is a real database under a live application**, which the other backends
mostly are not. `--dry-run` prints the plan without writing anything, and
`--limit` migrates a slice. Both are worth using here even if you skip them
elsewhere.
