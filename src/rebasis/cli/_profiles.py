"""Building an encoding profile from CLI flags.

A model rebasis has never seen still has to be usable. The table covers the
common ones; everything else arrives here, as flags.

The one thing this will not do is **guess a prefix**. Many retrieval models
encode a query differently from a document, usually with a short instruction
prefix, and getting it wrong produces no error and no warning — only worse
results, which is the hardest kind of failure to attribute. Being asked once is
cheaper than debugging that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rebasis.errors import ConfigError

if TYPE_CHECKING:
    from rebasis.types import EncodingProfile

__all__ = ["ProfileOverrides", "resolve_profile"]


class ProfileOverrides:
    """The ``--*-dim`` / ``--*-prefix`` flags, as one object.

    Grouped rather than passed as five loose arguments because they travel
    together through `probe`, `fit` and `eval`, and because a flag that means
    something different in one command from another is a bug waiting to happen.
    """

    __slots__ = ("dim", "document_prefix", "matryoshka_dim", "pooling", "query_prefix")

    def __init__(
        self,
        *,
        dim: int | None = None,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
        pooling: str | None = None,
        matryoshka_dim: int | None = None,
    ) -> None:
        self.dim = dim
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.pooling = pooling
        self.matryoshka_dim = matryoshka_dim

    def as_kwargs(self) -> dict[str, object]:
        """Only the fields the user actually set.

        An unset flag must not overwrite a table entry with ``None`` — that
        would turn "I did not say" into "there is no prefix", which is exactly
        the silent misconfiguration this module exists to prevent.
        """
        fields = {
            "query_prefix": self.query_prefix,
            "document_prefix": self.document_prefix,
            "pooling": self.pooling,
            "matryoshka_dim": self.matryoshka_dim,
        }
        return {k: v for k, v in fields.items() if v is not None}

    def __bool__(self) -> bool:
        return bool(self.as_kwargs()) or self.dim is not None


def resolve_profile(
    model_id: str,
    overrides: ProfileOverrides | None = None,
    *,
    fallback_dim: int | None = None,
) -> EncodingProfile:
    """Resolve a model's profile from the table, the flags, or both.

    Args:
        model_id: The model identifier.
        overrides: Flags the user passed.
        fallback_dim: A dimension known from elsewhere — in practice the
            index's own, which is authoritative for the model it was built
            with. Used only when the model is unregistered and no ``--dim`` was
            given.

    Raises:
        ConfigError: When a prefix was given without a dimension for a model
            the table does not know. A prefix on its own cannot construct a
            profile, and failing here says so; failing inside `profile_for`
            would report the missing dimension without mentioning that the
            prefix was accepted.
        UnknownModelProfile: When the model is unknown and no dimension is
            available from any source.
    """
    from rebasis.embed import profile_for

    overrides = overrides or ProfileOverrides()
    kwargs = overrides.as_kwargs()
    dim = overrides.dim if overrides.dim is not None else fallback_dim

    from rebasis.embed.profiles import known_profiles

    if kwargs and dim is None and model_id not in known_profiles():
        raise ConfigError(
            f"A prefix was given for {model_id!r}, but not its dimension.",
            hint="Add --new-dim (or --old-dim); rebasis has no entry for this model.",
            context={"embed_backend": model_id},
        )

    return profile_for(model_id, dim=dim, **kwargs)
