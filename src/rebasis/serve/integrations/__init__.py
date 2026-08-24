"""Framework integrations.

One wrapper per framework, each small enough to read in a sitting. The bridge
itself is framework-agnostic; these only translate call shapes.
"""

from __future__ import annotations

from rebasis.serve.integrations.retriever import wrap_retriever

__all__ = ["wrap_retriever"]
