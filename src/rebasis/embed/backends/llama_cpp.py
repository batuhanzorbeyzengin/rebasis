"""llama.cpp backend.

For GGUF models run in-process through ``llama-cpp-python``. The reason to want
this over sentence-transformers is quantisation: a 4-bit GGUF of a large
embedding model runs on hardware where the float16 original does not, and
`probe` is exactly the workload where that trade is worth making — the
measurement only has to be good enough to decide.

The model path is the ``model_id``: GGUF files are files, not hub identifiers,
and pretending otherwise would mean inventing a resolution rule that does not
exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from rebasis.errors import BackendUnavailable, ConfigError, MissingDependency
from rebasis.types import EncodingProfile, FloatArray, TextKind, as_float32

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["LlamaCppEmbedder"]


class LlamaCppEmbedder:
    """Embeds with a local GGUF model through llama.cpp."""

    def __init__(
        self,
        profile: EncodingProfile,
        *,
        device: str | None = None,
        n_gpu_layers: int | None = None,
        **model_kwargs: Any,
    ) -> None:
        self.profile = profile
        # llama.cpp offloads by layer count rather than by device name. Asking
        # for a GPU means "offload everything"; anything else stays on CPU.
        self.n_gpu_layers = (
            n_gpu_layers
            if n_gpu_layers is not None
            else (-1 if device and device.startswith(("cuda", "mps")) else 0)
        )
        self._model_kwargs = model_kwargs
        self._model: Any = None

    @property
    def model(self) -> Any:
        """The underlying model, loaded on first use."""
        if self._model is None:
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise MissingDependency(
                    "This backend needs llama-cpp-python.",
                    hint='Install it with `pip install "rebasis[llama-cpp]"`.',
                    context={"embed_backend": "llama-cpp"},
                    cause=exc,
                ) from exc

            path = Path(self.profile.model_id)
            if not path.exists():
                raise ConfigError(
                    f"No GGUF file at {path}.",
                    hint=(
                        "The llama-cpp backend takes a path to a .gguf file, not a hub model id."
                    ),
                    context={"embed_backend": "llama-cpp", "path": str(path)},
                )
            try:
                self._model = Llama(
                    model_path=str(path),
                    embedding=True,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=False,
                    **self._model_kwargs,
                )
            except Exception as exc:
                raise BackendUnavailable(
                    f"llama.cpp could not load {path.name}.",
                    hint="Check that the file is a GGUF embedding model and not a chat model.",
                    context={"embed_backend": "llama-cpp"},
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

        del batch_size, progress
        prefix = self.profile.prefix_for(kind)
        payload = [prefix + t for t in texts] if prefix else list(texts)
        if not payload:
            return np.empty((0, self.profile.dim), dtype=np.float32)

        response = self.model.create_embedding(payload)
        ordered = sorted(response["data"], key=lambda item: item["index"])
        # llama.cpp returns per-token embeddings for some models and a single
        # vector for others. Mean-pooling the former matches what every profile
        # in the table declares.
        vectors = [_pool(item["embedding"]) for item in ordered]
        return as_float32(np.asarray(vectors))


def _pool(embedding: Any) -> Any:
    """Reduce a per-token embedding to one vector; pass a vector through."""
    import numpy as np

    array = np.asarray(embedding, dtype=np.float32)
    return array.mean(axis=0) if array.ndim > 1 else array
