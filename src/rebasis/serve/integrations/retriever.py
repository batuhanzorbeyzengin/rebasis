"""Retriever wrapping.

The one-line adoption path::

    from rebasis.serve.integrations import wrap_retriever

    retriever = wrap_retriever(base_retriever, bridge)

Everything downstream keeps working; only the query vector changes on the way in.
That is the point of the default ``query_to_old`` direction — the index, the
store client and the application code all stay exactly as they were.

The wrapper is deliberately duck-typed rather than importing LangChain or
LlamaIndex. Importing either would make them dependencies of the serving path
for users who have neither, and the surface actually needed is one method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rebasis.serve.bridge import Bridge
    from rebasis.types import FloatArray

__all__ = ["BridgedRetriever", "wrap_retriever"]


class BridgedRetriever:
    """Wraps an embedding function so its output lands in the index's space.

    Holds the wrapped object rather than subclassing it: a framework's retriever
    base class may do a great deal in ``__init__``, and inheriting would couple
    this to a version of an API rebasis does not control.
    """

    __slots__ = ("_bridge", "_embed", "_inner")

    def __init__(self, inner: Any, bridge: Bridge, embed_attr: str = "embed_query") -> None:
        self._inner = inner
        self._bridge = bridge
        embed = getattr(inner, embed_attr, None)
        if embed is None or not callable(embed):
            msg = (
                f"{type(inner).__name__} has no callable {embed_attr!r}. Pass "
                "embed_attr= naming the method that turns a query into a vector."
            )
            raise AttributeError(msg)
        self._embed = embed

    def embed_query(self, text: str) -> FloatArray:
        """Embed with the new model, then map into the index's space."""
        return self._bridge.to_index_space(self._embed(text))

    def __getattr__(self, name: str) -> Any:
        """Forward everything else to the wrapped object.

        Only reached for attributes this class does not define, so the wrapping
        stays invisible to callers.
        """
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        return f"BridgedRetriever({type(self._inner).__name__}, {self._bridge!r})"


def wrap_retriever(retriever: Any, bridge: Bridge, *, embed_attr: str = "embed_query") -> Any:
    """Route a retriever's query embeddings through a bridge.

    Args:
        retriever: Any object with a method that embeds a query string.
        bridge: A loaded :class:`~rebasis.serve.bridge.Bridge`.
        embed_attr: Name of that method. ``embed_query`` covers LangChain;
            LlamaIndex uses ``get_query_embedding``.
    """
    return BridgedRetriever(retriever, bridge, embed_attr=embed_attr)
