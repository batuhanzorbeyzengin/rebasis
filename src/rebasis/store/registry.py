"""Store backend registry.

Backends are discovered through the ``rebasis.stores`` entry point group, so a
third-party package can add one **without forking rebasis**. Inside the repo,
adding a store is three steps: write the file, add the entry point, make the
contract test pass — and CONTRIBUTING says exactly that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rebasis.errors import MissingDependency, StoreUnsupported
from rebasis.observability import Events, get_logger
from rebasis.store.uri import StoreURI, parse_store_uri

if TYPE_CHECKING:
    from rebasis.store.base import VectorStore

__all__ = ["available_backends", "open_store", "resolve_backend"]

log = get_logger(__name__)

_ENTRY_POINT_GROUP = "rebasis.stores"

#: Backends that ship with rebasis and need no entry-point lookup.
_BUILTIN: dict[str, str] = {
    "memory": "rebasis.store.backends.memory:MemoryStore",
    "sqlite-vec": "rebasis.store.backends.sqlite_vec:SqliteVecStore",
    "qdrant": "rebasis.store.backends.qdrant:QdrantStore",
    "faiss": "rebasis.store.backends.faiss:FaissStore",
}

#: Spellings a URI is allowed to use, mapped to the backend they mean.
#:
#: Kept apart from :data:`_BUILTIN` so that they resolve without appearing in
#: :func:`available_backends`. Listing `sqlite-vec` and `sqlite_vec` as two
#: entries makes `rebasis doctor` report a backend that does not exist.
_ALIASES: dict[str, str] = {
    "sqlite_vec": "sqlite-vec",
}


def available_backends() -> dict[str, str]:
    """Every registered backend name mapped to its import target.

    Canonical names only: an alias resolves but is not a backend of its own.
    """
    from importlib.metadata import entry_points

    found = dict(_BUILTIN)
    for entry in entry_points(group=_ENTRY_POINT_GROUP):
        found.setdefault(entry.name, entry.value)
    return found


def resolve_backend(name: str) -> type[Any]:
    """Import the class implementing a backend.

    Raises:
        StoreUnsupported: When no backend is registered under that name.
        MissingDependency: When the backend exists but its client library is not
            installed. The two are separated on purpose: "I do not know this
            backend" and "install this package" call for different actions.
    """
    backends = available_backends()
    name = _ALIASES.get(name, name)
    if name not in backends:
        raise StoreUnsupported(
            f"No store backend named {name!r} is registered.",
            hint=f"Available: {', '.join(sorted(backends))}.",
            context={"store_backend": name},
        )

    module_path, _, attribute = backends[name].partition(":")
    try:
        import importlib

        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise MissingDependency(
            f"The {name} backend needs an extra package that is not installed.",
            hint=f'Install it with `pip install "rebasis[{name}]"`.',
            context={"store_backend": name},
            cause=exc,
        ) from exc

    return getattr(module, attribute)  # type: ignore[no-any-return]


def open_store(uri: str | StoreURI, **kwargs: Any) -> VectorStore:
    """Open the store a URI points at.

    The URI is logged in **redacted** form: the credential portion never reaches
    a log line or an audit record.
    """
    parsed = uri if isinstance(uri, StoreURI) else parse_store_uri(uri)
    backend_class = resolve_backend(parsed.backend)
    store: VectorStore = backend_class.from_uri(parsed, **kwargs)
    log.info(
        Events.STORE_OPENED,
        store_backend=parsed.backend,
        count=store.count(),
        dim=store.dimension(),
    )
    return store
