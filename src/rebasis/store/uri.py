"""Store URI parsing.

A store is addressed as ``backend:///path/to/db#collection``. One string carries
the backend, the location and the collection, so a command line, a config file
and an audit record can all identify the same index unambiguously.

The credential portion of a URI is **never logged**, so parsing keeps it
separate from the parts that are safe to record.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

from rebasis.errors import InvalidStoreURI

__all__ = ["StoreURI", "parse_store_uri"]


@dataclass(frozen=True, slots=True)
class StoreURI:
    """A parsed store address."""

    backend: str
    path: str
    collection: str | None = None
    host: str | None = None
    port: int | None = None
    username: str | None = None
    password: str | None = None
    options: dict[str, str] | None = None

    def redacted(self) -> str:
        """A form safe to log or write into an audit record."""
        location = self.host or self.path or ""
        if self.username:
            location = f"<credentials>@{location}"
        suffix = f"#{self.collection}" if self.collection else ""
        return f"{self.backend}://{location}{suffix}"

    def __str__(self) -> str:
        return self.redacted()


def parse_store_uri(uri: str) -> StoreURI:
    """Parse a store URI.

    Examples::

        chroma:///path/to/db#my_collection
        lancedb:///data/vectors#documents
        qdrant://localhost:6333#docs
        memory://

    Raises:
        InvalidStoreURI: When the string has no scheme or is otherwise unusable.
            The error names the expected shape, because a URI typo is the very
            first thing a new user can get wrong.
    """
    raw = (uri or "").strip()
    if not raw:
        raise InvalidStoreURI(
            "No store URI was given.",
            hint="Pass one like `--store chroma:///path/to/db#my_collection`.",
        )

    if "://" not in raw:
        raise InvalidStoreURI(
            f"{raw!r} has no backend scheme.",
            hint=(
                "A store URI looks like `backend:///path#collection`, for example "
                "`chroma:///path/to/db#my_collection`. Run `rebasis doctor` to "
                "list the backends available."
            ),
        )

    parts = urlsplit(raw)
    backend = parts.scheme.lower()
    if not backend:
        raise InvalidStoreURI(
            f"{raw!r} has an empty backend scheme.",
            hint="Start the URI with a backend name, for example `chroma://`.",
        )

    options: dict[str, str] = {}
    if parts.query:
        for pair in parts.query.split("&"):
            key, _, value = pair.partition("=")
            if key:
                options[unquote(key)] = unquote(value)

    return StoreURI(
        backend=backend,
        path=unquote(parts.path),
        collection=unquote(parts.fragment) or None,
        host=parts.hostname,
        port=parts.port,
        username=parts.username,
        password=parts.password,
        options=options or None,
    )
