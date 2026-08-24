"""Every backend is reachable, and none of them import at rest.

Adding a backend is "write the file, register the entry point, make the contract
tests pass". These check the middle step — which is the one that fails silently,
because a backend nobody can name is indistinguishable from one that does not
exist.

The second property matters more than it looks: `open_embedder` resolves a
backend by importing its module, and if a module imports its heavy dependency at
the top, `rebasis doctor` on a machine without that dependency stops working.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from rebasis.embed import available_embedders
from rebasis.errors import MissingDependency, StoreUnsupported
from rebasis.store import available_backends

pytestmark = pytest.mark.unit

EXPECTED_STORES = {"memory", "chroma", "lancedb", "qdrant", "sqlite-vec", "faiss"}
EXPECTED_EMBEDDERS = {
    "precomputed",
    "sentence-transformers",
    "fastembed",
    "ollama",
    "openai",
    "llama-cpp",
}


class TestRegistration:
    def test_every_store_backend_is_registered(self) -> None:
        assert set(available_backends()) >= EXPECTED_STORES

    def test_every_embedding_backend_is_registered(self) -> None:
        assert set(available_embedders()) >= EXPECTED_EMBEDDERS

    @pytest.mark.parametrize("name", sorted(EXPECTED_STORES))
    def test_a_store_backend_resolves_to_a_real_class(self, name: str) -> None:
        """A registry entry pointing at a module that does not import is worse
        than a missing one: it fails at the moment a user tries to use it."""
        from rebasis.store.registry import resolve_backend

        try:
            backend = resolve_backend(name)
        except MissingDependency:
            pytest.skip(f"{name} is not installed")
        assert hasattr(backend, "from_uri")

    @pytest.mark.parametrize("name", sorted(EXPECTED_EMBEDDERS))
    def test_an_embedding_backend_module_imports(self, name: str) -> None:
        import importlib

        module_path = available_embedders()[name].partition(":")[0]
        assert importlib.import_module(module_path)

    def test_an_alias_resolves_without_being_listed_as_a_backend(self) -> None:
        """`sqlite_vec://` has to work — people type underscores — but it is a
        spelling, not a second backend. Listing both made `rebasis doctor`
        report seven store backends where there are six, one of them a name
        that appears nowhere else in the docs.
        """
        from rebasis.store.registry import resolve_backend

        assert "sqlite_vec" not in available_backends()
        assert "sqlite-vec" in available_backends()
        assert resolve_backend("sqlite_vec") is resolve_backend("sqlite-vec")

    def test_an_unknown_backend_lists_what_exists(self) -> None:
        """A name that is merely wrong is unhelpful without the names that are
        right, and that list belongs in the hint beside the next step."""
        from rebasis.store.registry import resolve_backend

        with pytest.raises(StoreUnsupported, match="not-a-backend") as caught:
            resolve_backend("not-a-backend")

        assert "memory" in (caught.value.hint or "")


class TestLaziness:
    @pytest.mark.parametrize(
        ("module", "dependency"),
        [
            ("rebasis.embed.backends.fastembed", "fastembed"),
            ("rebasis.embed.backends.ollama", "ollama"),
            ("rebasis.embed.backends.openai_compat", "openai"),
            ("rebasis.embed.backends.llama_cpp", "llama_cpp"),
            ("rebasis.store.backends.faiss", "faiss"),
            ("rebasis.store.backends.qdrant", "qdrant_client"),
            ("rebasis.store.backends.sqlite_vec", "sqlite_vec"),
        ],
    )
    def test_importing_a_backend_does_not_import_its_dependency(
        self, module: str, dependency: str
    ) -> None:
        """Measured in a subprocess rather than inferred from the source.

        A static check cannot tell a function-body import from a module-level
        one once a decorator or a type annotation is involved; `sys.modules`
        can.
        """
        code = (
            f"import importlib, sys; importlib.import_module({module!r}); "
            f"print({dependency!r} in sys.modules)"
        )
        result = subprocess.run(  # noqa: S603 - our own interpreter, fixed argv
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == "False", f"{module} imported {dependency} eagerly"


class TestMissingDependencies:
    @pytest.mark.parametrize("name", ["fastembed", "ollama", "openai", "llama-cpp"])
    def test_an_uninstalled_backend_names_its_extra(self, name: str) -> None:
        """The error a user actually hits first. It has to say what to install,
        not that an import failed.

        Only asserted when the dependency really is absent: with it installed
        the call fails for some other reason (no daemon, no model file), which
        is a different test.
        """
        import importlib

        module_path, _, attribute = available_embedders()[name].partition(":")
        cls = getattr(importlib.import_module(module_path), attribute)

        from rebasis.types import EncodingProfile

        embedder = cls(EncodingProfile(model_id="x", dim=8))
        try:
            embedder.encode(["hello"])
        except MissingDependency as exc:
            hint = exc.hint or ""
        except Exception:  # noqa: BLE001 - installed, so it failed further in
            return
        else:
            return

        assert "pip install" in hint
