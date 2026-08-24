# LanceDB

LanceDB is where agent memories tend to live, and changing the embedding model
there conventionally requires a destructive reset — throwing away exactly the
accumulated memory that made the agent useful.

```bash
pip install "rebasis[lancedb]"
```

## URI

```
lancedb:///absolute/path/to/db#table_name
```

## Column names

LanceDB tables do not agree on what the columns are called. rebasis looks for
the vector column among `vector`, `embedding`, `embeddings`, `vec`; the id among
`id`, `_id`, `doc_id`, `pk`; and the text among `text`, `content`, `document`,
`page_content`. That covers what the common tutorials and the LangChain
integration produce.

When it guesses wrong, say so explicitly:

```
lancedb:///path/to/db#documents?vector_column=embeddings&id_column=chunk_id
```

If it cannot identify a column, the error lists both what it tried and what the
table actually has.

## What LanceDB supports

| Capability | Supported |
|---|---|
| Read vectors | yes |
| Read text | when a text column exists |
| Upsert vectors | yes |
| Metadata filter | yes |
| Dimension locked | no |
| In-place update | yes |

Unlike Chroma, LanceDB does not lock the dimension — so a full migration to a
different-dimensional model is possible here, and `probe` will tell you whether
it is worth doing.

## Streaming

LanceDB is Arrow-backed and pages naturally, so reads stream without any special
handling. rebasis never materialises a collection: peak memory is a function of
the batch, not of the table.
