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
store. A bridge therefore reports **honestly restricted** capabilities:

```python
store.capabilities
# StoreCapabilities(can_read_vectors=True, can_read_text=True,
#                   can_upsert_vectors=False, ...)
```

`probe` and the bridging phase work. `migrate` refuses **up front**, with a
message naming what is missing:

```
RB-E3002  The langchain backend does not support can_upsert_vectors,
          which migrate requires.
          `rebasis doctor` lists what this backend supports. `probe` and the
          bridge phase often work even where `migrate` cannot.
```

Half support beats no support. *Silent* half support is worse than none — which
is why the capability declaration is checked by a contract test that every
backend has to pass.

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
