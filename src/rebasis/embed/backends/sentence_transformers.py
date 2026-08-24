"""sentence-transformers backend.

The primary backend for local models. The model is loaded lazily, so a code path
that only needs the profile — resolving dimensions, checking a fingerprint —
never pays for a model download.

ℓ2 normalisation is left off in the library call and applied by rebasis instead.
Doing it in one place means the fitting path and the serving path cannot drift
apart, which is exactly the kind of divergence that produces a quality gap nobody
can locate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rebasis.errors import BackendUnavailable, MissingDependency
from rebasis.types import EncodingProfile, FloatArray, TextKind, as_float32

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["SentenceTransformerEmbedder"]


class SentenceTransformerEmbedder:
    """Wraps a ``SentenceTransformer`` with profile-aware prefixing."""

    def __init__(
        self, profile: EncodingProfile, *, device: str | None = None, **model_kwargs: Any
    ) -> None:
        self.profile = profile
        self.device = device
        self._model_kwargs = model_kwargs
        self._model: Any = None

    @property
    def model(self) -> Any:
        """The underlying model, loaded on first use."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise MissingDependency(
                    "This backend needs sentence-transformers.",
                    hint='Install it with `pip install "rebasis[sentence-transformers]"`.',
                    context={"embed_backend": "sentence-transformers"},
                    cause=exc,
                ) from exc
            try:
                self._model = SentenceTransformer(
                    self.profile.model_id, device=self.device, **self._model_kwargs
                )
            except Exception as exc:
                raise BackendUnavailable(
                    f"Could not load {self.profile.model_id!r}.",
                    hint=(
                        "Check the model name and your network connection. If the "
                        "model is already downloaded, set HF_HUB_OFFLINE=1."
                    ),
                    context={"embed_backend": self.profile.model_id},
                    cause=exc,
                ) from exc
        return self._model

    def encode(
        self,
        texts: Sequence[str],
        *,
        kind: TextKind = "document",
        batch_size: int = 32,
        progress: bool = False,
    ) -> FloatArray:
        """Encode texts, applying the profile's prefix for this kind."""
        prefix = self.profile.prefix_for(kind)
        payload = [prefix + t for t in texts] if prefix else list(texts)
        vectors = self.model.encode(
            payload,
            batch_size=batch_size,
            show_progress_bar=progress,
            convert_to_numpy=True,
            # rebasis normalises instead — one implementation, one behaviour.
            normalize_embeddings=False,
        )
        return as_float32(vectors)
