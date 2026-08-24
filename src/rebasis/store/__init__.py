"""Vector store abstraction.

The layer contract puts ``store`` below ``probe`` and ``migrate``, so this
package knows nothing about them.

The rule that matters most here: **rebasis does not take ownership of user
data.** ``probe`` and ``fit`` are read-only and must work against a read-only
filesystem; the only write path is ``migrate``, and it upserts without ever
deleting.
"""

from __future__ import annotations

from rebasis.store.backends.memory import MemoryStore
from rebasis.store.base import VectorStore, require_capability
from rebasis.store.bridges import LangChainStoreAdapter, LlamaIndexStoreAdapter
from rebasis.store.registry import available_backends, open_store, resolve_backend
from rebasis.store.uri import StoreURI, parse_store_uri

__all__ = [
    "LangChainStoreAdapter",
    "LlamaIndexStoreAdapter",
    "MemoryStore",
    "StoreURI",
    "VectorStore",
    "available_backends",
    "open_store",
    "parse_store_uri",
    "require_capability",
    "resolve_backend",
]
