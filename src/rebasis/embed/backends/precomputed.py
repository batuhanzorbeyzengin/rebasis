"""Embedder over vectors that already exist.

Two real uses. In tests it makes the whole pipeline runnable with no model
download and no network. In production it is how a user brings their own
vectors — someone who already embedded a sample with a service rebasis has no
backend for can still run ``probe`` and ``fit``.

It refuses to encode text it was not given, rather than returning a zero vector.
A zero vector would flow through the pipeline and surface as "quality is
strangely bad" instead of "you did not supply this text".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rebasis.errors import EmbedError
from rebasis.types import EncodingProfile, FloatArray, TextKind, as_float32

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = ["PrecomputedEmbedder"]


class PrecomputedEmbedder:
    """Serves vectors from a text-to-vector mapping."""

    def __init__(
        self,
        profile: EncodingProfile,
        vectors: Mapping[str, FloatArray],
        *,
        query_vectors: Mapping[str, FloatArray] | None = None,
    ) -> None:
        self.profile = profile
        self._documents = vectors
        # An asymmetric model needs a separate query-side table; without one,
        # both kinds share the document table and the profile is symmetric by
        # construction.
        self._queries = query_vectors or vectors

    def encode(
        self,
        texts: Sequence[str],
        *,
        kind: TextKind = "document",
        batch_size: int = 32,
        progress: bool = False,
    ) -> FloatArray:
        """Look up each text. Unknown text is an error, not a zero vector."""
        del batch_size, progress
        table = self._queries if kind == "query" else self._documents
        missing = [t for t in texts if t not in table]
        if missing:
            raise EmbedError(
                f"{len(missing)} of {len(texts)} texts have no precomputed vector.",
                hint="Supply a vector for every text, or use a real embedding backend.",
                context={"count": len(missing), "embed_backend": "precomputed"},
            )
        return as_float32(np.vstack([table[t] for t in texts]))
