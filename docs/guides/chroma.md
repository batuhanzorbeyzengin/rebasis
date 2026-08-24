# Chroma

Chroma locks a collection's dimension when it is created. Changing embedding
model there means creating a new collection and re-adding everything — which is
precisely the situation rebasis exists for.

```bash
pip install "rebasis[chroma]"
```

## URI

```
chroma:///absolute/path/to/db#collection_name
```

The collection name after `#` is required. Without it rebasis cannot know which
collection you mean, and guessing would be worse than asking.

## Probe

```bash
rebasis probe \
  --store chroma:///Users/me/vault/chroma#notes \
  --old sentence-transformers/all-MiniLM-L6-v2 \
  --new BAAI/bge-base-en-v1.5 \
  --report report.html
```

Reads only. The collection is not modified, and the command works against a
read-only filesystem.

## What Chroma supports

| Capability | Supported |
|---|---|
| Read vectors | yes |
| Read text | yes |
| Upsert vectors | yes |
| Metadata filter | yes |
| Dimension locked | **yes** |
| In-place update | yes |

The locked dimension is the important row. If your new model has a different
dimension, migration cannot write the new vectors into the existing collection —
bridging is the path, and rebasis will tell you so before it starts rather than
halfway through.

## If the collection is not found

```
RB-E3003  Chroma has no collection named 'note'.
          Available: notes, archive, drafts.
```

rebasis lists what is actually there. A collection name typo is the first thing
a new user gets wrong, so the error is written to fix itself.
