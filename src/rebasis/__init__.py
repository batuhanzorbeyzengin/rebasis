"""rebasis — change the embedding model of your local RAG without deleting the index."""

from __future__ import annotations

__all__ = ["Bridge", "__version__"]

from rebasis.__about__ import __version__
from rebasis.serve.bridge import Bridge
