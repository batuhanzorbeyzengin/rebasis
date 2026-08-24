"""``tail_arr`` and ``cost_full_reindex``.

Both were promised and neither was computed. `tail_arr` was the worse gap:
`decide()` accepted it, the `caution` band's rationale told users to look at it,
and the README said it was in the report — so the band's own diagnostic pointed
at a number nothing produced.
"""

from __future__ import annotations

import numpy as np
import pytest

from rebasis.observability import ResourceSummary, measure_resources
from rebasis.probe.metrics import estimate_reindex_cost, tail_recall

pytestmark = pytest.mark.unit


class TestTailRecall:
    def test_it_measures_the_sparsest_clusters(self) -> None:
        """Small clusters are where heterogeneous drift hides: a region with
        fifty documents barely moves the mean and can be losing most of its
        quality unnoticed."""
        # Cluster 9 is the sparsest and scores badly; the rest are large and
        # score well. It still has to hold enough queries to be reportable.
        labels = np.array([0] * 400 + [1] * 400 + [2] * 400 + [9] * 40)
        per_query = np.array([0.9] * 1200 + [0.2] * 40, dtype=np.float32)

        tail = tail_recall(per_query, labels, decile=0.25)

        assert tail == pytest.approx(0.2, abs=1e-5)

    def test_a_uniform_corpus_shows_no_gap(self) -> None:
        labels = np.repeat(np.arange(10), 40)
        per_query = np.full(400, 0.85, dtype=np.float32)

        assert tail_recall(per_query, labels) == pytest.approx(0.85, abs=1e-5)

    def test_too_few_clusters_returns_nothing(self) -> None:
        """With three clusters the "sparsest decile" is one cluster, and one
        cluster's mean is not a distribution's tail."""
        labels = np.array([0] * 10 + [1] * 10 + [2] * 10)

        assert tail_recall(np.full(30, 0.5, dtype=np.float32), labels) is None

    def test_misaligned_labels_return_nothing(self) -> None:
        """Scoring a query against another query's cluster would produce a
        number, which is worse than producing none."""
        assert tail_recall(np.zeros(10, dtype=np.float32), np.zeros(4)) is None

    def test_an_empty_series_returns_nothing(self) -> None:
        assert tail_recall(np.empty(0, dtype=np.float32), np.empty(0)) is None


class TestReindexCost:
    def test_it_extrapolates_from_the_measured_rate(self) -> None:
        """Calibrated on the user's own hardware and their own documents. The
        same corpus takes minutes on a GPU and hours on a laptop, and a figure
        that ignores which one you have is not worth printing."""
        cost = estimate_reindex_cost(200_000, seconds_per_document=0.003)

        assert cost["seconds"] == pytest.approx(600.0)
        assert cost["n_documents"] == 200_000

    def test_energy_is_absent_unless_power_was_measured(self) -> None:
        """Measured or omitted. A watt-hour figure derived from a nominal TDP
        reads as a measurement of something nobody asked the hardware."""
        assert estimate_reindex_cost(1000, seconds_per_document=0.01)["energy_wh"] is None

    def test_energy_is_reported_when_power_was(self) -> None:
        cost = estimate_reindex_cost(3_600_000, seconds_per_document=0.001, watts=100.0)

        assert cost["energy_wh"] == pytest.approx(100.0, rel=1e-3)

    def test_an_empty_corpus_costs_nothing(self) -> None:
        assert estimate_reindex_cost(0, seconds_per_document=0.5)["seconds"] == 0.0


class TestResourceSummary:
    def test_it_records_wall_and_cpu_time(self) -> None:
        with measure_resources(device="cpu") as usage:
            sum(i * i for i in range(200_000))

        assert usage.summary.wall_seconds > 0
        assert usage.summary.cpu_seconds > 0
        assert usage.summary.peak_rss_bytes > 0

    def test_parallelism_is_cpu_over_wall(self) -> None:
        """Their ratio is the useful signal: 4x means the cores are working,
        1x means something is being waited on."""
        summary = ResourceSummary(wall_seconds=10.0, cpu_seconds=35.0)

        assert summary.parallelism == pytest.approx(3.5)

    def test_a_zero_length_run_does_not_divide_by_zero(self) -> None:
        assert ResourceSummary().parallelism == 0.0

    def test_energy_is_none_rather_than_zero_when_unmeasured(self) -> None:
        """Zero is a measurement. None is the absence of one."""
        assert ResourceSummary(wall_seconds=1.0).to_dict()["energy_wh"] is None

    def test_the_device_and_blas_count_come_from_the_caller(self) -> None:
        """`observability` sits below `compute` in the layering, so it cannot
        look them up — and reaching for them would put torch on the import path
        of the logging package."""
        with measure_resources(device="cuda:0", blas_threads=8) as usage:
            pass

        payload = usage.summary.to_dict()
        assert payload["device"] == "cuda:0"
        assert payload["blas_threads"] == 8


class TestTheTailNeedsEnoughQueries:
    """A tail of six queries is noise with a decimal point.

    Measured on real corpora: at T0 the sparsest decile held 42-59 of 1,000
    held-out proxies. At T1 it held **six** — a real query log concentrates on
    the documents people actually ask about, and those sit in dense clusters —
    and produced a `tail_arr` of exactly 0.000. That read as catastrophic
    heterogeneous drift and was six queries missing.

    The diagnostic feeds a warning that would send someone building per-cluster
    adapters, so a false signal here is expensive.
    """

    def test_a_tail_of_six_reports_nothing(self) -> None:
        labels = np.array([0] * 40 + [1] * 40 + [2] * 40 + [3] * 40 + [9] * 6)
        per_query = np.array([0.9] * 160 + [0.0] * 6, dtype=np.float32)

        assert tail_recall(per_query, labels, decile=0.25) is None

    def test_a_tail_of_forty_reports(self) -> None:
        labels = np.array([0] * 400 + [1] * 400 + [2] * 400 + [3] * 400 + [9] * 40)
        per_query = np.array([0.9] * 1600 + [0.2] * 40, dtype=np.float32)

        assert tail_recall(per_query, labels, decile=0.25) == pytest.approx(0.2)

    def test_the_threshold_is_reachable(self) -> None:
        """Set where a mean's standard error falls below the 0.1 gap it is
        compared against, not so high that a real signal is suppressed."""
        from rebasis.probe.metrics import MIN_TAIL_QUERIES

        assert 20 <= MIN_TAIL_QUERIES <= 60

    def test_suppression_is_silent_not_zero(self) -> None:
        """Returning 0.0 would be indistinguishable from a real catastrophic
        tail — which is exactly the confusion that prompted the guard."""
        labels = np.array([0] * 40 + [1] * 40 + [2] * 40 + [3] * 40 + [9] * 3)
        per_query = np.array([0.9] * 160 + [0.0] * 3, dtype=np.float32)

        assert tail_recall(per_query, labels, decile=0.25) is None
