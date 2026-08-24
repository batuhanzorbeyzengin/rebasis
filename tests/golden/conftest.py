"""Loading the golden fixtures.

Skips cleanly and says how to build them when they are absent: the fixtures are
about 20 MB each and are not committed, so a fresh clone must not fail — it must
explain.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

HERE = Path(__file__).parent
MANIFEST = HERE / "manifest.json"

BUILD_HINT = (
    "Golden fixtures are not committed. Build them with:\n"
    "  uv run --extra sentence-transformers --with ir-datasets --with model2vec \\\n"
    "      python tools/make_golden.py --out tests/golden/data\n"
    "or point REBASIS_GOLDEN_DIR at an existing copy."
)


def golden_dir() -> Path:
    return Path(os.environ.get("REBASIS_GOLDEN_DIR", str(HERE / "data")))


@pytest.fixture(scope="session")
def manifest() -> dict[str, Any]:
    """The committed manifest: scenarios, model pairs and file hashes."""
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def scenario_names(manifest: dict[str, Any]) -> list[str]:
    return sorted(manifest["scenarios"])


@pytest.fixture(scope="session")
def load_scenario(manifest: dict[str, Any]) -> Any:
    """Return a loader that verifies the hash before handing back a scenario.

    Verifying rather than trusting: the fixtures arrive as a Release asset, and
    a band that shifts because the file changed underneath is the least
    debuggable failure this suite could produce.
    """
    cache: dict[str, dict[str, Any]] = {}

    def load(name: str) -> dict[str, Any]:
        if name in cache:
            return cache[name]

        path = golden_dir() / f"{name}.npz"
        if not path.exists():
            pytest.skip(f"{path} is missing.\n{BUILD_HINT}")

        expected = manifest["files"][f"{name}.npz"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            pytest.fail(
                f"{path.name} does not match the manifest.\n"
                f"  expected {expected}\n  actual   {actual}\n"
                "Rebuild it, or the bands below are measuring a different corpus."
            )

        data = np.load(path, allow_pickle=False)
        loaded = {
            "spec": manifest["scenarios"][name],
            "ids": [str(x) for x in data["ids"]],
            "texts": [str(x) for x in data["texts"]],
            "queries": [str(x) for x in data["queries"]],
            "relevant": [json.loads(str(x)) for x in data["relevant"]],
            "old_documents": data["old_documents"],
            "new_documents": data["new_documents"],
            "old_queries": data["old_queries"],
            "new_queries": data["new_queries"],
        }
        cache[name] = loaded
        return loaded

    return load
