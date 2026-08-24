"""Ollama backend.

For users already running models locally through Ollama. The model is not
downloaded by rebasis and no weights are loaded in-process: this is an HTTP
client against a daemon the user is already running.

One consequence worth stating: **the request never leaves the machine** by
default, because Ollama listens on localhost. That matters for a tool whose
premise is that a corpus stays local — an embedding backend that quietly posts
document text to a remote endpoint would undo it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rebasis.errors import BackendUnavailable, MissingDependency
from rebasis.observability import retry_transient
from rebasis.types import EncodingProfile, FloatArray, TextKind, as_float32

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["OllamaEmbedder"]

#: Ollama embeds one text per call, so a "batch" here is a loop. Kept small
#: because a failure mid-loop should not have queued thousands of requests.
DEFAULT_BATCH = 16


class OllamaEmbedder:
    """Embeds through a running Ollama daemon."""

    def __init__(
        self,
        profile: EncodingProfile,
        *,
        device: str | None = None,
        host: str | None = None,
        **client_kwargs: Any,
    ) -> None:
        self.profile = profile
        # Ollama chooses its own device; rebasis has no say and should not
        # pretend otherwise.
        del device
        self.host = host
        self._client_kwargs = client_kwargs
        self._client: Any = None

    @property
    def client(self) -> Any:
        """The Ollama client, created on first use."""
        if self._client is None:
            try:
                import ollama
            except ImportError as exc:
                raise MissingDependency(
                    "This backend needs the ollama client.",
                    hint='Install it with `pip install "rebasis[ollama]"`.',
                    context={"embed_backend": "ollama"},
                    cause=exc,
                ) from exc
            self._client = (
                ollama.Client(host=self.host, **self._client_kwargs)
                if self.host
                else ollama.Client(**self._client_kwargs)
            )
        return self._client

    @retry_transient
    def encode(
        self,
        texts: Sequence[str],
        *,
        kind: TextKind = "document",
        batch_size: int = DEFAULT_BATCH,
        progress: bool = False,
    ) -> FloatArray:
        """Encode texts, applying the profile's prefix for this kind.

        Raises:
            BackendUnavailable: When the daemon is unreachable or does not have
                the model. Both are ``transient``, so the call is retried with
                backoff: the user can start Ollama or pull the model meanwhile.
        """
        import numpy as np

        del batch_size, progress
        prefix = self.profile.prefix_for(kind)
        payload = [prefix + t for t in texts] if prefix else list(texts)
        if not payload:
            return np.empty((0, self.profile.dim), dtype=np.float32)

        try:
            response = self.client.embed(model=self.profile.model_id, input=payload)
        except Exception as exc:
            raise BackendUnavailable(
                f"Ollama could not embed with {self.profile.model_id!r}.",
                hint=(
                    "Check that `ollama serve` is running and that the model is "
                    f"pulled: `ollama pull {self.profile.model_id}`."
                ),
                context={"embed_backend": "ollama"},
                cause=exc,
            ) from exc

        embeddings = response.get("embeddings") if isinstance(response, dict) else None
        if embeddings is None:
            embeddings = getattr(response, "embeddings", None)
        if embeddings is None:
            raise BackendUnavailable(
                "Ollama returned a response with no embeddings.",
                hint="This usually means the model is a chat model, not an embedding model.",
                context={"embed_backend": "ollama"},
            )
        return as_float32(np.asarray(embeddings))
