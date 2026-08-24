"""Hot-path micro-benchmarks.

Excluded from the default run (marker ``perf``). The budget is 15 µs for a
single query and 1.5 ms for a 256-batch, and M0 measured what those are
actually worth:

| | Apple Silicon, d=384 | cloud vCPU, d=384 | cloud vCPU, d=768 |
|---|---|---|---|
| Procrustes | 7.1 µs ✓ | 21.8 µs ✗ | 33.9 µs ✗ |
| Residual MLP | 19.5 µs ✗ | 53.7 µs ✗ | 91.1 µs ✗ |

**No adapter meets 15 µs at d=768 on a cloud vCPU**, and the bottleneck is not
arithmetic: every numpy operation on a 1×384 array costs 2.5–3.5 µs whatever the
FLOPs. Procrustes performs one operation, the MLP five.

M4 went further and found the 15 µs figure is below the floor at d=768: the
768×768 matrix is 2.36 MB, so the multiply alone costs 15.8 µs on the reference
host. ADR 11 replaces the constant with a per-dimension budget. Two hot-path
optimisations came out of that measurement, and the last two tests here are what
keep them: they assert the shape of the code, not a wall-clock number.

So these tests do not assert the absolute budget — that would fail on hardware
the budget was never calibrated for. They assert what is portable:

1. **Relative cost.** Procrustes must stay cheaper than the MLP. If that
   inverts, something has been added to the hot path.
2. **No allocation growth.** The hot-path regression check: an accidental log
   call or dict copy shows up as objects allocated per call.
3. **Batching amortises.** Per-query cost inside a batch must be far below the
   single-query cost, since that is what makes the MLP usable at all.

Absolute numbers belong on the measurement host, where the hardware is known.
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pytest

from rebasis.core import (
    CenteredProcrustesAdapter,
    LinearAdapter,
    LowRankAffineAdapter,
    ProcrustesAdapter,
    l2_normalize,
)

DIM = 768
BATCH = 256


@pytest.fixture(scope="module")
def adapters() -> dict[str, object]:
    """One fitted adapter of each kind, at the dimension the budgets quote."""
    rng = np.random.default_rng(0)
    src = l2_normalize(rng.standard_normal((2000, DIM)).astype(np.float32))
    dst = l2_normalize(rng.standard_normal((2000, DIM)).astype(np.float32))
    return {
        "procrustes": ProcrustesAdapter.fit(src, dst),
        "procrustes_centered": CenteredProcrustesAdapter.fit(src, dst),
        "linear": LinearAdapter.fit(src, dst),
        "low_rank_affine": LowRankAffineAdapter.fit(src, dst),
    }


def _median_microseconds(fn: object, repeats: int = 2000, rounds: int = 5) -> float:
    for _ in range(100):
        fn()  # type: ignore[operator]
    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(repeats):
            fn()  # type: ignore[operator]
        samples.append((time.perf_counter() - start) / repeats * 1e6)
    return float(np.median(samples))


@pytest.mark.perf
def test_procrustes_is_cheaper_than_a_full_affine(adapters: dict[str, object]) -> None:
    """Relative cost, which is portable, rather than an absolute budget.

    Both perform one matrix multiply, so they should be within a small factor.
    A large divergence means something else crept onto the path.
    """
    rng = np.random.default_rng(1)
    query = l2_normalize(rng.standard_normal((1, DIM)).astype(np.float32))

    procrustes = _median_microseconds(lambda: adapters["procrustes"].apply(query))  # type: ignore[attr-defined]
    linear = _median_microseconds(lambda: adapters["linear"].apply(query))  # type: ignore[attr-defined]

    assert procrustes < linear * 2.0, (
        f"Procrustes ({procrustes:.1f} µs) should not cost materially more than "
        f"a full affine ({linear:.1f} µs); both are one matmul"
    )


@pytest.mark.perf
def test_low_rank_buys_memory_not_latency(adapters: dict[str, object]) -> None:
    """Rank truncation halves the parameters; it does **not** reduce latency.

    An earlier version of this test asserted that low-rank was faster. It passes
    on Apple Silicon and fails on a cloud vCPU, and the reason is the same one M0
    found everywhere on this path: at a single query, **call overhead dominates
    arithmetic**. Low-rank halves the FLOPs but doubles the number of numpy calls
    — ``x @ U`` then ``@ V`` instead of one ``x @ W`` — and on hardware with
    higher per-call cost the two cancel exactly.

    The recorded figures agree: low-rank affine at ~5 µs against Procrustes at
    ~3 µs. What low-rank actually buys is memory, and that is what is asserted
    here. Latency is checked only for not having regressed *badly*.
    """
    rng = np.random.default_rng(2)
    query = l2_normalize(rng.standard_normal((1, DIM)).astype(np.float32))

    low_rank_params = adapters["low_rank_affine"].n_params()  # type: ignore[attr-defined]
    full_params = adapters["linear"].n_params()  # type: ignore[attr-defined]
    assert low_rank_params < full_params * 0.75, (
        f"rank truncation saved too little memory: {low_rank_params:,} against "
        f"{full_params:,} parameters"
    )

    low_rank = _median_microseconds(lambda: adapters["low_rank_affine"].apply(query))  # type: ignore[attr-defined]
    full = _median_microseconds(lambda: adapters["linear"].apply(query))  # type: ignore[attr-defined]
    assert low_rank < full * 1.5, (
        f"low-rank {low_rank:.1f} µs against full {full:.1f} µs — the extra "
        "matmul should offset the halved FLOPs, not overwhelm them"
    )


@pytest.mark.perf
def test_batching_amortises_the_per_call_cost(adapters: dict[str, object]) -> None:
    """Per-query cost inside a batch must be far below the single-query cost.

    This is the M0 finding that makes an expensive adapter usable: the overhead
    is per *call*, not per vector, so a 256-batch pays it once.
    """
    rng = np.random.default_rng(3)
    single = l2_normalize(rng.standard_normal((1, DIM)).astype(np.float32))
    batch = l2_normalize(rng.standard_normal((BATCH, DIM)).astype(np.float32))
    adapter = adapters["procrustes_centered"]

    per_single = _median_microseconds(lambda: adapter.apply(single), repeats=2000)  # type: ignore[attr-defined]
    per_batch = _median_microseconds(lambda: adapter.apply(batch), repeats=50) / BATCH  # type: ignore[attr-defined]

    assert per_batch < per_single / 5, (
        f"batching saved too little: {per_single:.1f} µs single vs "
        f"{per_batch:.2f} µs per query in a batch of {BATCH}"
    )


@pytest.mark.perf
def test_normalising_one_vector_takes_the_short_route() -> None:
    """A single vector must cost materially less than the per-row batch route.

    `np.linalg.norm` is a Python function that re-derives its axis and dtype
    handling on every call. At d=768 that is 4.7 µs against 1.4 µs for the dot
    product it ends up performing, and the hot path's whole budget is 15 µs.
    The short route exists for that reason, and this test is what stops someone
    tidying it away — asserted as a ratio against the general route, so it holds
    on hardware the budget was never calibrated for.
    """
    rng = np.random.default_rng(5)
    one = l2_normalize(rng.standard_normal((1, DIM)).astype(np.float32))
    two = l2_normalize(rng.standard_normal((2, DIM)).astype(np.float32))

    single = _median_microseconds(lambda: l2_normalize(one, copy=False))
    general = _median_microseconds(lambda: l2_normalize(two, copy=False))

    assert single < general * 0.75, (
        f"one vector took {single:.1f} µs against {general:.1f} µs for two — the "
        "single-vector route is not being taken"
    )


@pytest.mark.perf
def test_the_centred_adapter_folds_its_offset(adapters: dict[str, object]) -> None:
    """`(x − μs)R + μd` is `xR + b`, and the folded form is what ships.

    One fewer full-length array operation per query. Asserted against plain
    Procrustes, which does the same matmul with nothing added: if the centred
    adapter ever costs materially more, the fold has been undone.
    """
    rng = np.random.default_rng(6)
    query = l2_normalize(rng.standard_normal((1, DIM)).astype(np.float32))

    centred = _median_microseconds(lambda: adapters["procrustes_centered"].apply(query))  # type: ignore[attr-defined]
    plain = _median_microseconds(lambda: adapters["procrustes"].apply(query))  # type: ignore[attr-defined]

    assert centred < plain * 1.5, (
        f"centred Procrustes ({centred:.1f} µs) against plain ({plain:.1f} µs): "
        "the centring should be one added vector, not two"
    )


@pytest.mark.perf
@pytest.mark.parametrize("name", ["procrustes", "procrustes_centered", "linear", "low_rank_affine"])
def test_hot_path_allocates_a_bounded_amount(adapters: dict[str, object], name: str) -> None:
    """Catch a log call or a dict copy sneaking onto the hot path.

    Measured in bytes allocated per call. A stray `log.debug(...)` or a
    `dict(**kwargs)` shows up here immediately, while a wall-clock benchmark
    would bury it in noise.
    """
    adapter = adapters[name]
    rng = np.random.default_rng(4)
    query = l2_normalize(rng.standard_normal((1, DIM)).astype(np.float32))

    for _ in range(50):
        adapter.apply(query)  # type: ignore[attr-defined]

    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()
    for _ in range(200):
        adapter.apply(query)  # type: ignore[attr-defined]
    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    grown = sum(stat.size_diff for stat in snapshot_after.compare_to(snapshot_before, "lineno"))
    per_call = max(0, grown) / 200
    # Generous: the point is to catch an order-of-magnitude regression, not to
    # pin an exact allocator behaviour that varies by interpreter version.
    assert per_call < 64 * 1024, f"{name} allocates {per_call:.0f} bytes per call"


@pytest.mark.perf
def test_adapter_memory_matches_the_design_table(adapters: dict[str, object]) -> None:
    """The recorded memory figures, at the dimension they quote.

    M0 found the Residual MLP figure wrong — 1.6 MB counts only ``W₁+W₂`` and
    omits the linear ``Wx`` term. These are the ones that were correct, and this
    test keeps them so.
    """
    expected_mb = {
        "procrustes": 2.4,
        "linear": 2.4,
        "low_rank_affine": 1.2,
    }
    for name, limit in expected_mb.items():
        actual = adapters[name].memory_bytes / 1024**2  # type: ignore[attr-defined]
        assert actual <= limit, f"{name} is {actual:.2f} MB, over its {limit} MB budget"
