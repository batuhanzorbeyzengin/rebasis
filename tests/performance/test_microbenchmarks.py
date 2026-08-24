"""The absolute micro-benchmarks, measured under instruction counting.

``test_hot_path.py`` deliberately asserts only *relative* cost, because a
wall-clock number on a shared runner is noise: a 7% gate is needed to keep false
positives at 1%, and a 7% gate hides exactly the regressions worth catching. Its
docstring says where the absolute numbers belong — here, where CodSpeed counts
instructions in a simulator and the runner stops mattering.

That job existed before these tests did. ``pytest --codspeed`` collects only what
uses the ``benchmark`` fixture, nothing here did, and the job passed on every
commit having measured nothing: exit code 5, no tests collected.

What is covered is what the development docs already claim is covered: adapter
`apply`, `.rbs` loading, normalisation, and top-k.
"""

from __future__ import annotations

import numpy as np
import pytest

from rebasis.compute.search import top_k_search
from rebasis.core import (
    CenteredProcrustesAdapter,
    ProcrustesAdapter,
    l2_normalize,
    load_adapter,
    save_adapter,
)
from rebasis.types import EncodingProfile

#: The dimension the budgets are quoted at, and the one where the hot path is
#: tightest — ADR 11 replaced the flat 15 µs with a per-dimension budget because
#: a 768×768 multiply alone costs more than the old constant allowed.
DIM = 768
BATCH = 256

pytestmark = pytest.mark.perf


@pytest.fixture(scope="module")
def fitted() -> dict[str, object]:
    """One adapter of each kind that the hot path actually runs."""
    rng = np.random.default_rng(0)
    src = l2_normalize(rng.standard_normal((2000, DIM)).astype(np.float32))
    dst = l2_normalize(rng.standard_normal((2000, DIM)).astype(np.float32))
    return {
        "procrustes": ProcrustesAdapter.fit(src, dst),
        "procrustes_centered": CenteredProcrustesAdapter.fit(src, dst),
    }


@pytest.fixture(scope="module")
def one_query() -> np.ndarray:
    """A single query — the shape the 'per query' budget is about."""
    rng = np.random.default_rng(1)
    return l2_normalize(rng.standard_normal((1, DIM)).astype(np.float32))


@pytest.fixture(scope="module")
def a_batch() -> np.ndarray:
    rng = np.random.default_rng(2)
    return l2_normalize(rng.standard_normal((BATCH, DIM)).astype(np.float32))


def test_apply_one_query(benchmark, fitted, one_query) -> None:  # type: ignore[no-untyped-def]
    """The hot path: one query through the adapter that ships by default."""
    adapter = fitted["procrustes_centered"]
    benchmark(lambda: adapter.apply(one_query))  # type: ignore[attr-defined]


def test_apply_a_batch(benchmark, fitted, a_batch) -> None:  # type: ignore[no-untyped-def]
    """Batched apply: what `migrate` runs, once per record."""
    adapter = fitted["procrustes_centered"]
    benchmark(lambda: adapter.apply(a_batch))  # type: ignore[attr-defined]


def test_apply_uncentred(benchmark, fitted, one_query) -> None:  # type: ignore[no-untyped-def]
    """The plain Procrustes, for the difference the folded offset makes.

    The centred adapter folds `mu_dst - mu_src @ R` into a bias at construction
    so the hot path stays one operation. If that regresses, this pair separates.
    """
    adapter = fitted["procrustes"]
    benchmark(lambda: adapter.apply(one_query))  # type: ignore[attr-defined]


def test_normalise_one_vector(benchmark, one_query) -> None:  # type: ignore[no-untyped-def]
    """`l2_normalize` takes a scalar route for a single vector.

    `np.linalg.norm` re-derives its axis and dtype handling per call, which at
    one query is most of the cost. That optimisation is worth a benchmark that
    would notice it being undone.
    """
    benchmark(lambda: l2_normalize(one_query))


def test_normalise_a_batch(benchmark, a_batch) -> None:  # type: ignore[no-untyped-def]
    benchmark(lambda: l2_normalize(a_batch))


def test_top_k_search(benchmark) -> None:  # type: ignore[no-untyped-def]
    """Chunked matmul plus argpartition, at a size where chunking matters."""
    rng = np.random.default_rng(3)
    queries = l2_normalize(rng.standard_normal((64, DIM)).astype(np.float32))
    docs = l2_normalize(rng.standard_normal((5_000, DIM)).astype(np.float32))
    benchmark(lambda: top_k_search(queries, docs, k=10))


def test_load_an_adapter(benchmark, fitted, tmp_path_factory) -> None:  # type: ignore[no-untyped-def]
    """`.rbs` loading: paid once per process, and on every `Bridge.load`.

    Validation happens here precisely so the hot path can skip it, which makes
    this the place where an added check would show up.
    """
    out = tmp_path_factory.mktemp("rbs") / "adapter.rbs"
    save_adapter(
        fitted["procrustes_centered"],  # type: ignore[arg-type]
        out,
        direction="query_to_old",
        old_profile=EncodingProfile(model_id="benchmark/old", dim=DIM),
        new_profile=EncodingProfile(model_id="benchmark/new", dim=DIM),
    )
    benchmark(lambda: load_adapter(out))
