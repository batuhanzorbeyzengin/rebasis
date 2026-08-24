"""Shared types.

This module sits at the bottom of the layer contract and depends on nothing. The
one array alias the codebase allows lives here: ``FloatArray``. dtype conversion
is forbidden inside the core; float64 input is converted once at the boundary
via :func:`as_float32` and never touched again.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "Adapter",
    "AdapterDirection",
    "AdapterKind",
    "Decision",
    "Embedder",
    "EncodingProfile",
    "FloatArray",
    "Hit",
    "IntArray",
    "Record",
    "StoreCapabilities",
    "TextKind",
    "as_float32",
]

# float32 is a contract: promoting to float64 doubles memory and has no
# measurable effect on ARR.
FloatArray = np.ndarray[Any, np.dtype[np.float32]]
IntArray = np.ndarray[Any, np.dtype[np.int32]]

#: Queries and documents may be encoded differently. This distinction is not an
#: implementation detail — it is the most likely source of silent errors.
TextKind = Literal["query", "document"]

#: Direction of the adapter. ``query_to_old`` is the default: it leaves the index
#: untouched. ``old_to_new`` is the virtual backfill.
AdapterDirection = Literal["query_to_old", "old_to_new"]

AdapterKind = Literal[
    "identity",
    "procrustes",
    "procrustes_centered",
    "linear",
    "low_rank_affine",
    "residual_mlp",
]

#: Output of the decision rule.
Decision = Literal[
    "no_upgrade_needed",
    "bridge_sufficient",
    "bridge_and_migrate",
    "caution",
    "full_reindex",
]


def as_float32(x: Any) -> FloatArray:
    """The single dtype conversion, performed at the boundary.

    Uses ``np.asarray`` (not ``np.array``): no copy is made when the input is
    already a contiguous float32 array.
    """
    return np.asarray(x, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class EncodingProfile:
    """How a model encodes queries versus documents.

    ``nomic-embed-text`` (``search_query:``/``search_document:``), E5
    (``query:``/``passage:``) and BGE (instruction prefix) encode queries and
    documents differently. Fitting an adapter on document pairs and applying it
    to query vectors produces a train/serve mismatch that raises no error and
    only degrades quality.

    :meth:`fingerprint` is written into the ``.rbs`` file; on mismatch the
    adapter refuses to load. Applying the wrong adapter to the wrong index is
    therefore structurally impossible rather than merely discouraged.
    """

    model_id: str
    dim: int
    query_prefix: str | None = None
    document_prefix: str | None = None
    normalize: bool = True
    matryoshka_dim: int | None = None
    pooling: str = "mean"

    @property
    def symmetric(self) -> bool:
        """Whether queries and documents are encoded identically."""
        return self.query_prefix == self.document_prefix

    def prefix_for(self, kind: TextKind) -> str:
        """Prefix to prepend for the given text kind; empty string if none."""
        prefix = self.query_prefix if kind == "query" else self.document_prefix
        return prefix or ""

    def fingerprint(self) -> str:
        """Stable sha256 fingerprint of this profile.

        Computed over canonical JSON, so field ordering does not affect the
        result. Adding a new field *does* change the fingerprint — deliberately,
        so that adapters built under older semantics are not silently reused.
        """
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Record:
    """A single record read from a vector store."""

    id: str
    vector: FloatArray | None = None
    text: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Hit:
    """A single search result."""

    id: str
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class StoreCapabilities:
    """What a store actually supports.

    Bridge adapters (LangChain / LlamaIndex) report these *honestly restricted*:
    ``probe`` and the bridge phase work, ``migrate`` may not. Partial support
    beats no support; silent partial support does not — which is why the
    capability declaration is checked by a contract test.
    """

    can_read_vectors: bool
    can_read_text: bool
    can_upsert_vectors: bool
    can_filter: bool
    dimension_locked: bool
    supports_in_place_update: bool
    name: str = ""


class Adapter(Protocol):
    """Adapter protocol.

    Contract: ``fit(src, dst)`` then ``apply(src) ≈ dst``. Under the default
    ``query_to_old`` direction, ``src = f_new(d)`` and ``dst = f_old(d)``.
    """

    kind: AdapterKind
    input_dim: int
    output_dim: int

    def apply(self, x: FloatArray) -> FloatArray:
        """Map vectors into the target space. Hot path — performs no validation."""
        ...

    def state_dict(self) -> Mapping[str, FloatArray]:
        """Weights to serialise into the ``.rbs`` file."""
        ...

    def n_params(self) -> int:
        """Total number of parameters."""
        ...


class Embedder(Protocol):
    """Embedding backend protocol."""

    profile: EncodingProfile

    def encode(
        self,
        texts: Sequence[str],
        *,
        kind: TextKind,
        batch_size: int = 32,
        progress: bool = True,
    ) -> FloatArray:
        """Encode texts. ``kind`` selects the prefix to apply."""
        ...
