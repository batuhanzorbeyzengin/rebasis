"""Version, resolved without importing the package root.

Importing ``rebasis`` pulls in :class:`~rebasis.serve.bridge.Bridge`, so any
module that reads ``__version__`` from the package root creates a cycle — and
worse, a layering violation: ``core`` would depend on ``serve``. Reading it from
here keeps the version available to every layer with no dependency at all.
"""

from __future__ import annotations

__all__ = ["__version__"]

try:
    from rebasis._version import __version__
except ImportError:  # pragma: no cover - only in an uninstalled source tree
    __version__ = "0.0.0.dev0"
