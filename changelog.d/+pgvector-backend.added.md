A pgvector backend — the store most teams already run, and the first one where `migrate`'s batch integrity is the database's rather than rebasis'.

```
pgvector://user:pass@host:5432/dbname#public.documents
pgvector://…/db#public.chunks?vector=emb&id=chunk_id&text=body
```

`pip install "rebasis[pgvector]"`, and it is in `[all]`.

**A Postgres table has no shape.** A Chroma collection does; a table is whatever its owner made. So the columns are named in the URI, or found among the conventional names (`embedding`/`vector`/…, `id`/`doc_id`/…, `text`/`content`/…) and reported. Nothing is inferred from a column merely being the only one of its type — that is how a backend ends up rewriting the wrong column. Where no conventional name is there, the error lists the table's own columns with their types and names the option that supplies the missing one.

**The column type decides almost everything, and it is read rather than assumed.** pgvector has four vector types and they are not interchangeable. The type comes from the catalogue through `format_type`, because `information_schema` reports every extension type as `USER-DEFINED` and cannot tell `vector` from `halfvec`. Measured, not reasoned about: a `vector` column returns exactly what was written; a `halfvec` column rounds by **more than `migrate`'s 1e-4 read-back tolerance**, so a migration into one would stop on its own first batch and the pre-flight plan says so beforehand; `bit` and `sparsevec` hold a code rather than a reconstruction of the vector that produced it, and both are refused with the reason at the moment they are asked. The precedent is `sqlite-vec`, where an `int8` column was once read as if it held float32 and every number derived from the reported dimension was a quarter of the truth, silently.

**One batch is one transaction.** Everywhere else a partially written batch is a state `migrate` has to detect and undo; here it either commits or does not exist. The shadow copy stays — a transaction rolls back one batch and `rollback <job-id>` rolls back a finished job three days later, which are different scopes — and the plan names which layer holds which guarantee.

The connection is opened in **autocommit**, which is part of the same decision rather than a detail. psycopg's default opens a transaction on the first statement and holds it until something commits, so a `probe` against a live database would sit `idle in transaction` for the length of the run, pinning the vacuum horizon of somebody's production table for a read that needed no transaction at all. The two places that genuinely need one — the batch write, and the server-side cursor `iter_records` streams through — open it explicitly. `REINDEX CONCURRENTLY` needs the opposite and gets it for free.

**It is the second backend that can rebuild its own index, and the first where the two index types answer differently.** `REINDEX INDEX CONCURRENTLY`, never the plain form, which takes an ACCESS EXCLUSIVE lock and stops every read of the table for its duration. What that repair is worth is measured separately for HNSW and IVFFlat in [what a migration does to the index](https://batuhanzorbeyzengin.github.io/rebasis/index-health/).

It also has the richest filter of the six — a SQL `WHERE` — and is the only backend whose reads stream through a server-side cursor rather than a client-side one.

The scope is narrow on purpose: **one table, named columns.** A joined schema, a partitioned table, text in a second table — all refused explicitly rather than half-supported silently. CI runs a real PostgreSQL as a service on the coverage job and on the lowest-direct job, the second so the `psycopg` floor is found by running the suite against it rather than by looking plausible.
