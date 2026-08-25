"""rebasis — measure whether an embedding upgrade is worth it, bridge it, migrate it."""

from __future__ import annotations

__all__ = ["Bridge", "__version__"]

from rebasis.__about__ import __version__
from rebasis.serve.bridge import Bridge
