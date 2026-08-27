# sqlite-vec

One file on disk, no server, no dependency beyond a loadable extension. It is
the shape of index that gets abandoned rather than reindexed when a better model
comes out, which makes it the backend that fits this project's premise best.

## The URI

```
sqlite-vec:///path/to/index.db#vec_documents
```

The fragment is the **`vec0` virtual table** — the one holding the embeddings —
not the table holding your text.

## Two tables, and rebasis has to find both

This is the part that needs explaining, and it is a property of sqlite-vec rather
than of rebasis.

A `vec0` virtual table holds an integer `rowid` and the embedding, and nothing
else. Your document id and your text live in an ordinary table beside it, joined
on `rowid`. So rebasis has to locate that second table, and it guesses the way
sqlite-vec's own examples are written:

1. **By name.** `vec_documents` → `documents`. It strips a `vec_` prefix or a
   `_vec` / `_vectors` suffix, then tries that name, that name plus `s`, and that
   name without a trailing `s`.
2. **By shape.** Failing that, the first other table with a column named `id`,
   `doc_id`, `key` or `uuid`.

Within that table it looks for an id column named `id`, `doc_id`, `key` or
`uuid`, and a text column named `text`, `content`, `document`, `chunk` or `body`.

**When your schema does not follow any of that, say so in the URI:**

```
sqlite-vec:///index.db#vec_documents?metadata_table=chunks&id_column=chunk_id&text_column=body
```

`vector_column` is there too, and defaults to `embedding`.

Check what it resolved before you rely on it:

```bash
rebasis doctor --store "sqlite-vec:///index.db#vec_documents"
```

If text comes back unavailable, `probe` and the bridge still work — but `migrate`
will refuse, because the pairs an adapter is fitted on need the same documents
re-embedded, and without text there is nothing to re-embed.

## The element type decides what is possible

A `vec0` column is declared with an element type, and the three sqlite-vec offers
are not equivalent:

| Declared as | Bytes per component | rebasis |
|---|---|---|
| `float` / `f32` | 4 | Everything works |
| `int8` / `i8` | 1 | Reads and writes, and **declares itself quantized** |
| `bit` / `b1` | ⅛ | **Cannot be read** |

`int8` and `bit` are lossy by construction — sqlite-vec's own
`vec_quantize_int8` and `vec_quantize_binary` are what produce them. For `int8`
that is a reason to know what your round trip is worth, not a reason to refuse;
read [If your index is stored
quantized](migration.md#if-your-index-is-stored-quantized) before migrating one.

**`bit` is different, and the reason is worth stating.** A `vec0` `bit[N]` is
legal for any `N` — `bit[7]` and `bit[12]` both create — so the blob's length does
not determine the dimension, and a `vec0` table declares its virtual schema
without a type on the vector column. There is nowhere else to read the width
from. Reading vectors out of a `bit` column is not a lossy operation, it is an
impossible one, and the backend declares `can_read_vectors=False` rather than
returning something wrong.

rebasis asks the extension with `vec_type()` rather than parsing the declaration,
and reads each column at **its own** element width. An empty table declares
nothing, so the answer there is "unknown" rather than "float32".

## Writing

`migrate` deletes and re-inserts the same `rowid` rather than issuing an
`UPDATE`. That is not a preference: a `vec0` table cannot be updated through a
plain `UPDATE` in every version, and delete-then-insert is the portable form. It
happens **inside a transaction**, so an interrupted migration cannot leave a row
with an id and no vector.

## What it cannot do

| | |
|---|---|
| **Filtering** | No. There is no server-side filter on arbitrary metadata, so `probe --filter` is unavailable |
| **Rebuilding the index** | Not applicable. sqlite-vec is a brute-force scan, so there is no graph to invalidate — which is also why a migration cannot cost you recall here the way it can on a graph backend |

That second row is a real advantage and worth reading twice. On Qdrant, Chroma
or LanceDB, rewriting vectors does not rewrite the structure built around them,
and `migrate` measures the index's own recall before and after for that reason.
On sqlite-vec there is nothing to measure: search is exact, so a correct vector
is a correct answer.

## Backing it up

One file. `cp index.db index.db.backup` while nothing is writing, or use
SQLite's own `.backup` for a live copy.

Do it. rebasis takes a shadow copy of every vector before it overwrites it and
`rollback` restores from that, but the advice is still a backup rebasis is not
part of — a shadow copy in the same state directory does not survive the disk it
lives on.
