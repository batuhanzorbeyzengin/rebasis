<!-- GENERATED FILE — edit the source module, not this page. -->

# Encoding profiles

The models rebasis knows without being told. Values come from
each model card's official retrieval instructions, not from guesswork.

**Why this table exists.** Many retrieval models encode a query
differently from a document — usually with a prefix. Getting that wrong
produces no error and no warning; it only lowers quality, which is the
hardest kind of failure to attribute. So the prefix lives in the
profile, the profile is fingerprinted, and the fingerprint blocks an
adapter from loading against an index it was not built for.

**A model not listed here still works.** Pass `--dim`, and
`--query-prefix` / `--document-prefix` if it is asymmetric. rebasis will
not guess a prefix: being asked once is cheaper than debugging a quality
drop.

`Asymmetric` marks the models where the two prefixes differ. For those,
`auto` measures both a shared adapter and a query-specific one and keeps
whichever scores better on the held-out set — M0 measured the mean
difference at -0.003, with the sign varying by model pair.

| Model | Dim | Asymmetric | Query prefix | Document prefix | Pooling |
|---|---|---|---|---|---|
| `BAAI/bge-base-en-v1.5` | 768 | **yes** | `Represent this sentence for searching relevant passages: ` | — | cls |
| `BAAI/bge-large-en-v1.5` | 1024 | **yes** | `Represent this sentence for searching relevant passages: ` | — | cls |
| `BAAI/bge-m3` | 1024 | no | — | — | cls |
| `BAAI/bge-small-en-v1.5` | 384 | **yes** | `Represent this sentence for searching relevant passages: ` | — | cls |
| `intfloat/e5-base-v2` | 768 | **yes** | `query: ` | `passage: ` | mean |
| `intfloat/e5-large-v2` | 1024 | **yes** | `query: ` | `passage: ` | mean |
| `intfloat/e5-small-v2` | 384 | **yes** | `query: ` | `passage: ` | mean |
| `intfloat/multilingual-e5-base` | 768 | **yes** | `query: ` | `passage: ` | mean |
| `jinaai/jina-embeddings-v2-base-en` | 768 | no | — | — | mean |
| `minishlab/potion-base-8M` | 256 | no | — | — | mean |
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | **yes** | `Represent this sentence for searching relevant passages: ` | — | cls |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | **yes** | `search_query: ` | `search_document: ` | mean |
| `sentence-transformers/all-MiniLM-L12-v2` | 384 | no | — | — | mean |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | no | — | — | mean |
| `sentence-transformers/all-mpnet-base-v2` | 768 | no | — | — | mean |
| `thenlper/gte-base` | 768 | no | — | — | mean |
