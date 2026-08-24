"""FastEmbed backend.

The lightweight path: ONNX runtime, **no torch**, and native support for the
``query`` / ``passage`` prefix distinction. For a user who only wants to run
`probe` on a laptop, this is the backend that makes
``pip install rebasis[fastembed]`` a small install rather than a 2 GB one.

FastEmbed applies its own prefixes for the models it knows. rebasis applies the
profile's prefix instead, for the same reason it normalises itself: one
implementation of the rule, so the fitting path and the serving path cannot
drift apart. Where the profile has no prefix, nothing is added and FastEmbed's
default behaviour is what runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rebasis.errors import BackendUnavailable, MissingDependency
from rebasis.types import EncodingProfile, FloatArray, TextKind, as_float32

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["FastEmbedEmbedder"]


class FastEmbedEmbedder:
    """Wraps a FastEmbed ``TextEmbedding`` with profile-aware prefixing."""

    def __init__(
        self, profile: EncodingProfile, *, device: str | None = None, **model_kwargs: Any
    ) -> None:
        self.profile = profile
        # FastEmbed selects providers rather than devices. `cuda` maps to the
        # CUDA execution provider, which needs onnxruntime-gpu; anything else
        # runs on CPU, which is FastEmbed's whole point.
        self.device = device
        self._model_kwargs = model_kwargs
        self._model: Any = None

    @property
    def model(self) -> Any:
        """The underlying model, loaded on first use."""
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise MissingDependency(
                    "This backend needs fastembed.",
                    hint='Install it with `pip install "rebasis[fastembed]"`.',
                    context={"embed_backend": "fastembed"},
                    cause=exc,
                ) from exc

            kwargs = dict(self._model_kwargs)
            if self.device and self.device.startswith("cuda"):
                kwargs.setdefault("providers", ["CUDAExecutionProvider"])
            try:
                self._model = TextEmbedding(model_name=self.profile.model_id, **kwargs)
            except Exception as exc:
                raise BackendUnavailable(
                    f"Could not load {self.profile.model_id!r} with fastembed.",
                    hint=(
                        "`TextEmbedding.list_supported_models()` lists what fastembed "
                        "carries. A model it does not know needs the "
                        "sentence-transformers backend instead."
                    ),
                    context={"embed_backend": "fastembed"},
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
        import numpy as np

        del progress  # fastembed has no progress hook on this call
        prefix = self.profile.prefix_for(kind)
        payload = [prefix + t for t in texts] if prefix else list(texts)
        if not payload:
            return np.empty((0, self.profile.dim), dtype=np.float32)

        # `embed` returns a generator, one vector at a time. Materialising it
        # here is bounded by the caller's chunk, not by the corpus.
        vectors = list(self.model.embed(payload, batch_size=batch_size))
        return as_float32(np.vstack(vectors))
