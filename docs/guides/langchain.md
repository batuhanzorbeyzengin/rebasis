# LangChain and LlamaIndex

rebasis has a `VectorStore` protocol of its own, and two bridge adapters that
wrap LangChain and LlamaIndex vector stores. One file each, dozens of stores.

```bash
pip install "rebasis[langchain]"    # or [llamaindex]
```

Neither bridge holds a hard dependency on its framework: they duck-type the
wrapped object, so the framework is only needed by someone who already has it.

## Using one

```python
from rebasis.store import LangChainStoreAdapter
from rebasis.probe import probe_store
from rebasis.embed import open_embedder

store = LangChainStoreAdapter(my_langchain_vectorstore)
result, _ = probe_store(store, open_embedder("BAAI/bge-base-en-v1.5"))
print(result.decision.decision)
```

## Partial support is declared, not discovered

This is the important part. The LangChain and LlamaIndex interfaces do not
expose `iter_records(with_vectors=True)` or `upsert_vectors` for every backing
store. A bridge therefore reports **honestly restricted** capabilities — and
"honestly" has to mean tied to the code that would deliver them, not to a
`hasattr` that happens to look related.

Every read the LangChain bridge performs goes through one handle: the wrapped
store's private `_collection`, called the way Chroma's is. So both read
capabilities answer that one question, and nothing else:

```python
store.capabilities
# StoreCapabilities(can_read_vectors=True, can_read_text=True,     # a Chroma-backed store
#                   can_upsert_vectors=False, can_filter=False, ...)

# The same bridge over langchain-core's own InMemoryVectorStore, which keeps
# its vectors in a dict and exposes no such handle:
# StoreCapabilities(can_read_vectors=False, can_read_text=False,
#                   can_upsert_vectors=False, can_filter=False, ...)
```

The LlamaIndex bridge declares all four `False` unconditionally. It can search
and it can sometimes count; it cannot enumerate records at all, and its interface
offers no portable way to.

`probe` and the bridging phase work wherever the records can be read. `migrate`
refuses **up front**, through every bridge, with a message naming what is
missing:

```
RB-E3002  The langchain backend does not support can_upsert_vectors,
          which migrate requires.
          `rebasis doctor` lists what this backend supports. `probe` and the
          bridge phase often work even where `migrate` cannot.
```

Half support beats no support. *Silent* half support is worse than none.

## Two things the bridge asks you for

Both exist because the wrapped object genuinely cannot answer, and guessing
would be worse than asking.

**`dimension=`.** Without it the LangChain bridge establishes the dimensionality
by embedding a short probe string through the store's own model. That is a
billed request against a hosted API. It runs at most once per adapter and the
result is cached, but passing the dimension costs nothing at all:

```python
LangChainStoreAdapter(my_store, dimension=768)
```

The LlamaIndex bridge has nothing to probe, so `dimension=` is the only way it
can answer.

**`score_kind=`.** LangChain has no portable contract for the direction of the
number a by-vector search returns. Chroma's
`similarity_search_by_vector_with_relevance_scores` returns a raw *distance* —
"lower score represents more similarity", in its own docstring, under a name
saying the opposite — while other integrations return a relevance score where
higher is better, and FAISS flips direction on a constructor argument. Nothing
on the object says which.

So by default the bridge reports **no score at all**: every hit is `0.0`, and the
rank carries the ranking. A score that is silently inverted is worse than one
that is visibly absent. Tell it which way yours runs and it will use the number:

```python
LangChainStoreAdapter(my_chroma_store, score_kind="distance")   # or "similarity"
```

The LlamaIndex bridge needs no such flag: `VectorStoreQueryResult.similarities`
has one meaning.

## What is tested, and what is not

The bridges have a contract suite of their own: `tests/contract/test_bridge_stores.py`.
It runs in two layers.

The first is duck-typed fakes, and it always runs. They are built from the
upstream source of `langchain-core`, `langchain-chroma` and `llama-index-core`
at versions the test module names, rather than from a reading of the bridges —
because a fake shaped like the bridge passes by construction and keeps passing
after the real interface has moved. There is a family of them rather than one,
because the cases worth pinning are the ones where capability inference has to
answer "no": a store with a `_client` and no `_collection`, a `_collection`
property that raises, a scored search method that returns `None` where its own
annotation says `float`, a collection handle that accepts a paging window and
ignores it.

The second layer drives the real `langchain_core.vectorstores.InMemoryVectorStore`
and `llama_index.core.vector_stores.SimpleVectorStore` — both in the core
packages, so neither needs a vendor, a server or a network — and skips where the
extras are not installed. That layer is the one that would notice a dependency
bump, which is the whole reason these two files are riskier than a native
backend.

Still not covered, and worth knowing:

- **No real third-party store runs through either bridge in CI.** The reference
  stores in the two core packages are the only real ones exercised, and neither
  of them is one the bridge can read records out of. The readable path is pinned
  by fakes only.
- **`similarity_search_by_vector_with_relevance_scores` and
  `similarity_search_with_score_by_vector` are integration methods**, not part of
  either core package, so no real implementation of either is under test here.
- **Filtering is forwarded, not verified.** `can_filter` is `False` on both
  bridges precisely because no inspection can establish it. A `where` you pass is
  still sent on; a store that cannot take it raises rather than quietly returning
  unfiltered results, and that much is tested.

## Writing your own backend

Three steps:

1. Write a class satisfying the `VectorStore` protocol.
2. Register it under the `rebasis.stores` entry-point group.
3. Make the contract suite pass.

```toml
[project.entry-points."rebasis.stores"]
mystore = "my_package.store:MyStore"
```

The contract suite is `tests/contract/test_vector_store.py`. Its two most
valuable tests are the ones a new backend is most likely to get wrong quietly:
that `iter_records` is genuinely lazy, and that the capability declaration is
true.
