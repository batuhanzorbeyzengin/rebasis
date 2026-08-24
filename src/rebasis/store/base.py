"""Vector store protocol.

Two design rules govern every backend:

**Reads stream.** ``iter_records`` returns an iterator and no caller ever wraps
it in ``list()``. Peak memory is ``O(batch × d)``, never ``O(N × d)`` — the same
tool must behave identically on a 50,000-chunk vault and a 5,000,000-chunk one.
A single materialising call breaks that for everybody.

**Capabilities are declared honestly.** The LangChain and LlamaIndex bridges buy
dozens of stores for one file each, but those interfaces do not expose
``iter_records(with_vectors=True)`` or ``upsert_vectors`` everywhere. A bridge
reports what it can genuinely do, so ``probe`` and the bridge phase work while
``migrate`` refuses up front with a clear reason. Partial support beats none;
*silent* partial support is worse than none — which is why the declaration is
checked by a contract test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from rebasis.errors import CapabilityMissing

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from rebasis.types import FloatArray, Hit, Record, StoreCapabilities

__all__ = ["VectorStore", "require_capability"]


@runtime_checkable
class VectorStore(Protocol):
    """What every store backend must provide."""

    @property
    def capabilities(self) -> StoreCapabilities:
        """What this store can actually do — declared truthfully."""
        ...

    def count(self) -> int:
        """Number of records in the collection."""
        ...

    def dimension(self) -> int:
        """Vector dimensionality of the collection."""
        ...

    def iter_records(
        self,
        ids: Sequence[str] | None = None,
        *,
        with_vectors: bool = True,
        with_text: bool = True,
        batch_size: int = 1000,
    ) -> Iterator[Record]:
        """Stream records. **Must be lazy** — never materialise the collection."""
        ...

    def search(self, vector: FloatArray, k: int, where: dict[str, Any] | None = None) -> list[Hit]:
        """Nearest neighbours of ``vector``."""
        ...

    def upsert_vectors(self, ids: Sequence[str], vectors: FloatArray) -> None:
        """Replace the vectors of existing records.

        The only write path in rebasis, and it never deletes.
        """
        ...


def require_capability(store: VectorStore, name: str, *, operation: str) -> None:
    """Assert a capability before starting work.

    Raises:
        CapabilityMissing: When the store cannot do what the operation needs.

    Checked **before** the operation begins rather than when it first fails, so
    the user is told at second zero instead of halfway through a migration.
    """
    capabilities = store.capabilities
    if not getattr(capabilities, name, False):
        raise CapabilityMissing(
            f"The {capabilities.name or 'store'} backend does not support {name}, "
            f"which {operation} requires.",
            hint=(
                "`rebasis doctor` lists what this backend supports. `probe` and the "
                "bridge phase often work even where `migrate` cannot."
            ),
            context={"store_backend": capabilities.name, "error_code": name},
        )
