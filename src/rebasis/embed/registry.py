"""Embedder registry.

Backends are discovered through the ``rebasis.embedders`` entry point group, so a
third-party package can add one without forking rebasis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rebasis.embed.profiles import profile_for
from rebasis.errors import MissingDependency, StoreUnsupported

if TYPE_CHECKING:
    from rebasis.types import Embedder, EncodingProfile

__all__ = ["available_embedders", "open_embedder"]

_ENTRY_POINT_GROUP = "rebasis.embedders"

_BUILTIN: dict[str, str] = {
    "precomputed": "rebasis.embed.backends.precomputed:PrecomputedEmbedder",
    "sentence-transformers": (
        "rebasis.embed.backends.sentence_transformers:SentenceTransformerEmbedder"
    ),
    "fastembed": "rebasis.embed.backends.fastembed:FastEmbedEmbedder",
    "ollama": "rebasis.embed.backends.ollama:OllamaEmbedder",
    "openai": "rebasis.embed.backends.openai_compat:OpenAICompatEmbedder",
    "llama-cpp": "rebasis.embed.backends.llama_cpp:LlamaCppEmbedder",
}


def available_embedders() -> dict[str, str]:
    """Every registered embedder name mapped to its import target."""
    from importlib.metadata import entry_points

    found = dict(_BUILTIN)
    for entry in entry_points(group=_ENTRY_POINT_GROUP):
        found.setdefault(entry.name, entry.value)
    return found


def open_embedder(
    model_id: str,
    *,
    backend: str = "sentence-transformers",
    dim: int | None = None,
    device: str | None = None,
    profile: EncodingProfile | None = None,
    **kwargs: Any,
) -> Embedder:
    """Construct an embedder for a model.

    Args:
        model_id: The model to load.
        backend: Which embedding backend to use.
        dim: Dimensionality, for a model the profile table does not know.
        device: ``cpu``, ``cuda``, ``mps``; ``None`` lets the backend choose.
        profile: A fully resolved profile. Supplied by the CLI when the user
            gave prefix flags, because a prefix cannot be reconstructed from a
            model id and getting it wrong costs quality with no error to show.
        **kwargs: Passed to the backend constructor.

    Raises:
        StoreUnsupported: When no backend is registered under that name.
        MissingDependency: When the backend's package is not installed.
    """
    backends = available_embedders()
    if backend not in backends:
        raise StoreUnsupported(
            f"No embedding backend named {backend!r} is registered.",
            hint=f"Available: {', '.join(sorted(backends))}.",
            context={"embed_backend": backend},
        )

    module_path, _, attribute = backends[backend].partition(":")
    try:
        import importlib

        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise MissingDependency(
            f"The {backend} embedding backend is not installed.",
            hint=f'Install it with `pip install "rebasis[{backend}]"`.',
            context={"embed_backend": backend},
            cause=exc,
        ) from exc

    resolved = profile if profile is not None else profile_for(model_id, dim=dim)
    embedder_class = getattr(module, attribute)
    return embedder_class(resolved, device=device, **kwargs)  # type: ignore[no-any-return]
