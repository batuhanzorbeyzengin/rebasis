"""Framework bridges, re-exported.

Two adapters, each buying access to dozens of stores for one file — with the
honest caveat that neither framework exposes vector enumeration or in-place
update portably. Both report their capabilities truthfully, so `probe` and the
bridge phase work while `migrate` refuses up front rather than failing halfway.
"""

from __future__ import annotations

from rebasis.store.backends.langchain_bridge import LangChainStoreAdapter
from rebasis.store.backends.llamaindex_bridge import LlamaIndexStoreAdapter

__all__ = ["LangChainStoreAdapter", "LlamaIndexStoreAdapter"]
