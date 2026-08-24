"""Adapter registry: name to class.

Deserialisation needs to turn the ``adapter_type`` string in a ``.rbs`` manifest
back into a class. Keeping that mapping in one place means adding an adapter is a
single-line change rather than a hunt through ``if`` chains.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rebasis.core.base import (
    BaseAdapter,  # noqa: TC001 - used as a runtime value in ADAPTER_CLASSES
)
from rebasis.core.identity import IdentityAdapter
from rebasis.core.linear import LinearAdapter, LowRankAffineAdapter
from rebasis.core.procrustes import CenteredProcrustesAdapter, ProcrustesAdapter
from rebasis.errors import AdapterError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rebasis.types import FloatArray

__all__ = ["ADAPTER_CLASSES", "build_adapter", "resolve_adapter_class"]

#: Every adapter that can be deserialised. ``residual_mlp`` is registered lazily
#: in :func:`resolve_adapter_class` because importing it pulls in torch, and
#: `import rebasis` must stay torch-free.
ADAPTER_CLASSES: dict[str, type[BaseAdapter]] = {
    "identity": IdentityAdapter,
    "procrustes": ProcrustesAdapter,
    "procrustes_centered": CenteredProcrustesAdapter,
    "linear": LinearAdapter,
    "low_rank_affine": LowRankAffineAdapter,
}


def resolve_adapter_class(kind: str) -> type[BaseAdapter]:
    """Look up an adapter class by name.

    Raises:
        AdapterError: When the name is unknown, or when it names an adapter
            whose optional dependency is missing.
    """
    base_kind = kind.removesuffix("+dsm")

    if base_kind in ADAPTER_CLASSES:
        return ADAPTER_CLASSES[base_kind]

    if base_kind == "residual_mlp":
        try:
            from rebasis.core.residual_mlp import ResidualMLPAdapter
        except ImportError as exc:
            raise AdapterError(
                "This adapter is a residual MLP, which needs torch to load.",
                hint=(
                    'Install it with `pip install "rebasis[torch]"`, or refit '
                    "with --method procrustes_centered, which measured as "
                    "equivalent in quality at half the memory."
                ),
                context={"adapter_type": kind},
                cause=exc,
            ) from exc
        return ResidualMLPAdapter

    raise AdapterError(
        f"Unknown adapter type: {kind!r}.",
        hint=f"Known types: {', '.join(sorted([*ADAPTER_CLASSES, 'residual_mlp']))}.",
        context={"adapter_type": kind},
    )


def build_adapter(
    kind: str, state: Mapping[str, FloatArray], config: Mapping[str, Any]
) -> BaseAdapter:
    """Reconstruct an adapter from its stored weights and configuration."""
    if kind.endswith("+dsm"):
        from rebasis.core.scaling import ScaledAdapter

        return ScaledAdapter.from_state_dict(state, config)
    return resolve_adapter_class(kind).from_state_dict(state, config)
