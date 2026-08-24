"""Encoding profile registry.

A profile records **how** a model encodes queries as opposed to documents — nomic
wants ``search_query:``/``search_document:``, E5 ``query:``/``passage:``, BGE an
instruction prefix. Get it wrong and nothing fails: quality simply drops, with no
error to attribute it to, which makes this the most likely source of silent error
in rebasis. It is also why the profile fingerprint blocks an adapter from loading
against the wrong index.

Known profiles live in ``data/profiles.json``, in a data file rather than in code
so a user can extend it without a release. When a model is unknown, rebasis says
so and asks — it does **not** guess a prefix, because a wrong guess is
indistinguishable from a right one until quality is measured.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

from rebasis.errors import UnknownModelProfile
from rebasis.types import EncodingProfile

__all__ = ["known_profiles", "profile_for", "register_profile"]

_DATA = Path(__file__).parent / "data" / "profiles.json"
_EXTRA: dict[str, EncodingProfile] = {}


@functools.cache
def _builtin() -> dict[str, EncodingProfile]:
    payload = json.loads(_DATA.read_text(encoding="utf-8"))
    return {
        model_id: EncodingProfile(model_id=model_id, **fields)
        for model_id, fields in payload["profiles"].items()
    }


def known_profiles() -> dict[str, EncodingProfile]:
    """Every profile currently registered."""
    return {**_builtin(), **_EXTRA}


def register_profile(profile: EncodingProfile) -> None:
    """Register a profile at runtime — used by ``--profile`` and by plugins."""
    _EXTRA[profile.model_id] = profile


def profile_for(model_id: str, *, dim: int | None = None, **overrides: Any) -> EncodingProfile:
    """Look up a model's encoding profile.

    Args:
        model_id: The model identifier, for example ``BAAI/bge-small-en-v1.5``.
        dim: Dimensionality, required only for an unregistered model.
        **overrides: Fields to override, such as ``query_prefix``.

    Raises:
        UnknownModelProfile: When the model is unknown and no dimension was
            given. rebasis will not invent a prefix: guessing wrong is silent,
            and being asked once is cheaper than debugging a quality drop.
    """
    registry = known_profiles()
    if model_id in registry:
        base = registry[model_id]
        if not overrides and dim is None:
            return base
        fields: dict[str, Any] = {
            "model_id": base.model_id,
            "dim": dim if dim is not None else base.dim,
            "query_prefix": base.query_prefix,
            "document_prefix": base.document_prefix,
            "normalize": base.normalize,
            "matryoshka_dim": base.matryoshka_dim,
            "pooling": base.pooling,
        }
        fields.update(overrides)
        return EncodingProfile(**fields)

    if dim is None:
        raise UnknownModelProfile(
            f"No encoding profile is known for {model_id!r}.",
            hint=(
                "Pass --new-dim (or --old-dim), and --query-prefix / "
                "--document-prefix if the model encodes queries and documents "
                "differently. rebasis will not guess a prefix: guessing wrong "
                "lowers quality with no error to explain it. "
                "`rebasis adapter profiles` lists the models it knows."
            ),
            context={"embed_backend": model_id},
        )

    return EncodingProfile(model_id=model_id, dim=dim, **overrides)
