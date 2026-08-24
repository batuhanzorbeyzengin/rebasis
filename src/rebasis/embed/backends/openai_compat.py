"""OpenAI-compatible HTTP backend.

Covers OpenAI itself and everything that speaks its embeddings API — vLLM,
Infinity, LM Studio, TEI, LocalAI, most hosted providers. One client, because
the wire format is the same.

**This backend sends document text to whatever endpoint it is pointed at.**
Every other part of rebasis is built so a corpus never leaves the machine, and
this one deliberately can. That is the user's call to make, so it is stated in
the docstring, printed by `doctor`, and the endpoint is recorded in the audit
trail — but the API key never is: a model API key is never logged, anywhere.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from rebasis.errors import BackendUnavailable, ConfigError, MissingDependency
from rebasis.observability import retry_transient
from rebasis.types import EncodingProfile, FloatArray, TextKind, as_float32

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["OpenAICompatEmbedder"]

#: Providers cap request size. 256 is comfortably inside every limit seen in
#: practice and keeps a failed request cheap to retry.
DEFAULT_BATCH = 256


class OpenAICompatEmbedder:
    """Embeds through any OpenAI-compatible ``/v1/embeddings`` endpoint."""

    def __init__(
        self,
        profile: EncodingProfile,
        *,
        device: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        **client_kwargs: Any,
    ) -> None:
        self.profile = profile
        # The device is the server's business.
        del device
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client_kwargs = client_kwargs
        self._client: Any = None

    @property
    def client(self) -> Any:
        """The HTTP client, created on first use."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise MissingDependency(
                    "This backend needs the openai client.",
                    hint='Install it with `pip install "rebasis[openai]"`.',
                    context={"embed_backend": "openai-compat"},
                    cause=exc,
                ) from exc

            if not self._api_key and not self.base_url:
                raise ConfigError(
                    "No API key and no base URL for the OpenAI-compatible backend.",
                    hint=(
                        "Set OPENAI_API_KEY for a hosted provider, or OPENAI_BASE_URL "
                        "for a local server such as vLLM or Infinity."
                    ),
                    context={"embed_backend": "openai-compat"},
                )
            self._client = OpenAI(
                api_key=self._api_key or "not-needed",
                base_url=self.base_url,
                **self._client_kwargs,
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
            BackendUnavailable: When the endpoint rejects the request or is
                unreachable.
        """
        import numpy as np

        del progress
        prefix = self.profile.prefix_for(kind)
        payload = [prefix + t for t in texts] if prefix else list(texts)
        if not payload:
            return np.empty((0, self.profile.dim), dtype=np.float32)

        blocks: list[Any] = []
        for start in range(0, len(payload), batch_size):
            chunk = payload[start : start + batch_size]
            try:
                response = self.client.embeddings.create(model=self.profile.model_id, input=chunk)
            except Exception as exc:
                raise BackendUnavailable(
                    f"The embeddings endpoint rejected a request for {self.profile.model_id!r}.",
                    hint=(
                        "Check the model name, the base URL and the key. The endpoint "
                        "is recorded in the audit trail; the key never is."
                    ),
                    context={"embed_backend": "openai-compat", "count": len(chunk)},
                    cause=exc,
                ) from exc
            # Providers do not promise response order; index is authoritative.
            ordered = sorted(response.data, key=lambda item: item.index)
            blocks.extend(item.embedding for item in ordered)

        return as_float32(np.asarray(blocks))
