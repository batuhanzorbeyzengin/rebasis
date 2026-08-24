"""Embedding backends and encoding profiles.

A backend is an ``Embedder``: an encoding profile plus ``encode(texts, *, kind,
...)``. The layer contract places ``embed`` below ``probe``, ``migrate`` and
``serve``, so it is used by all three and depends on none of them.

The distinction this package exists to enforce is asymmetry: a model may encode a
query differently from a document. Ignoring that produces no error, only a
quality drop — so the prefix lives in the profile, the profile is fingerprinted,
and the fingerprint blocks an adapter from loading against an index it was not
built for.
"""

from __future__ import annotations

from rebasis.embed.backends.precomputed import PrecomputedEmbedder
from rebasis.embed.profiles import known_profiles, profile_for, register_profile
from rebasis.embed.registry import available_embedders, open_embedder

__all__ = [
    "PrecomputedEmbedder",
    "available_embedders",
    "known_profiles",
    "open_embedder",
    "profile_for",
    "register_profile",
]
