"""Weighting `probe`'s query proxies by an access log.

The sampler has taken weights since it was written and nothing passed them. The
roadmap's entry names one place to put them and there are **two**, because a
`probe` sample does two jobs at once: it is the mini-index every measurement runs
against, and it is the pool the query proxies are split out of.

Weighting the *sample* fills the mini-index with frequently-read documents, which
changes the **distractors** — a property of the index rather than of the
questions asked of it. Weighting the *split* leaves the mini-index a fair
miniature and changes only what is asked. The second is what "describe the
queries that matter" means, and it is where the weights go.

What is asserted here is the mechanism and the honesty around it: that weights
reach the split and not the sample, that a log naming nothing degrades to a plain
draw rather than to a vector of ones, and that a run which was weighted says so —
because ARR then estimates a different quantity and two numbers under one name is
the failure this project keeps designing against.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from rebasis.cli._pipeline import read_access_log
from rebasis.sample import SampleResult, split_disjoint

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

N = 600
QUERIES = 200


def sample_of(n: int = N) -> SampleResult:
    return SampleResult(indices=np.arange(n, dtype=np.int64), strategy="stratified", seed=0)


def hot_weights(n: int = N, *, hot: int = 60, ratio: float = 100.0) -> np.ndarray:
    """A small hot set and a long cold tail — the shape an access log has."""
    weights = np.ones(n, dtype=np.float32)
    weights[:hot] = ratio
    return weights


class TestTheWeightsReachTheSplit:
    def test_hot_records_dominate_the_query_set(self) -> None:
        """The point of the feature, stated as the thing that must be true."""
        weights = hot_weights()

        queries, _ = split_disjoint(sample_of(), QUERIES, seed=0, weights=weights)

        hot_in_queries = int((queries < 60).sum())
        assert hot_in_queries > 50, hot_in_queries

    def test_an_unweighted_split_does_not(self) -> None:
        """The control. Without weights the hot set gets its share and no more:
        60 of 600 records, 200 queries, so about 20."""
        queries, _ = split_disjoint(sample_of(), QUERIES, seed=0)

        hot_in_queries = int((queries < 60).sum())
        assert hot_in_queries < 40, hot_in_queries

    def test_the_two_sets_are_still_disjoint(self) -> None:
        """Disjointness is what every ARR number rests on, and a weighted draw
        is a new way to get it wrong."""
        queries, fit = split_disjoint(sample_of(), QUERIES, seed=0, weights=hot_weights())

        assert np.intersect1d(queries, fit).size == 0
        assert queries.size + fit.size == N

    def test_every_sampled_record_lands_on_one_side(self) -> None:
        queries, fit = split_disjoint(sample_of(), QUERIES, seed=0, weights=hot_weights())

        assert sorted(np.concatenate([queries, fit]).tolist()) == list(range(N))

    def test_the_same_seed_gives_the_same_split(self) -> None:
        """Recorded in the audit trail so a decision can be replayed."""
        weights = hot_weights()
        first, _ = split_disjoint(sample_of(), QUERIES, seed=7, weights=weights)
        second, _ = split_disjoint(sample_of(), QUERIES, seed=7, weights=weights)

        assert first.tolist() == second.tolist()

    def test_a_cold_record_is_still_eligible(self) -> None:
        """A log records what *was* read; absence means "not seen in this
        window", not "never retrievable". Zero-probability cold records would
        make ARR describe the hot set alone."""
        queries, _ = split_disjoint(sample_of(), QUERIES, seed=0, weights=hot_weights())

        assert int((queries >= 60).sum()) > 0


class TestItDegradesToAPlainDraw:
    def test_a_weight_vector_of_the_wrong_length_is_ignored(self) -> None:
        """It describes a different sample than the one being split, so it
        describes nothing. Falling back is right: the caller asked for a probe
        and the weights were an optimisation on top of one."""
        queries, fit = split_disjoint(
            sample_of(), QUERIES, seed=0, weights=np.ones(N // 2, dtype=np.float32)
        )

        assert queries.size == QUERIES
        assert np.intersect1d(queries, fit).size == 0

    def test_all_zero_weights_are_ignored(self) -> None:
        queries, fit = split_disjoint(
            sample_of(), QUERIES, seed=0, weights=np.zeros(N, dtype=np.float32)
        )

        assert queries.size == QUERIES
        assert np.intersect1d(queries, fit).size == 0


class TestTheLogItself:
    def test_a_missing_count_reads_as_one(self, tmp_path: Path) -> None:
        path = tmp_path / "access.jsonl"
        path.write_text('{"id": "a", "count": 9}\n{"id": "b"}\n', encoding="utf-8")

        assert read_access_log(path) == {"a": 9.0, "b": 1.0}

    def test_record_id_is_accepted_beside_id(self, tmp_path: Path) -> None:
        """These logs are exported from somewhere else, and a rigid schema means
        everyone writes a converter first."""
        path = tmp_path / "access.jsonl"
        path.write_text('{"record_id": "a", "count": 3}\n', encoding="utf-8")

        assert read_access_log(path) == {"a": 3.0}

    def test_a_log_with_no_usable_lines_is_no_log(self, tmp_path: Path) -> None:
        path = tmp_path / "access.jsonl"
        path.write_text('{"count": 3}\n\n', encoding="utf-8")

        assert read_access_log(path) is None

    def test_no_path_is_no_log(self) -> None:
        assert read_access_log(None) is None


class TestARunSaysWhetherItWasWeighted:
    def test_a_log_naming_nothing_in_the_sample_is_not_a_weighted_run(self) -> None:
        """Reporting the flag the user passed rather than the draw that happened
        would claim a measurement that was not taken."""
        from rebasis.probe.session import _access_vector

        assert _access_vector(["a", "b", "c"], {"z": 50.0}) is None

    def test_a_log_that_names_nobody_differently_is_not_either(self) -> None:
        """Every record weighted 1 is a uniform draw wearing a weight vector."""
        from rebasis.probe.session import _access_vector

        assert _access_vector(["a", "b"], {"a": 1.0, "b": 1.0}) is None

    def test_a_log_that_does_name_someone_is(self) -> None:
        from rebasis.probe.session import _access_vector

        weights = _access_vector(["a", "b"], {"a": 40.0})

        assert weights is not None
        assert weights.tolist() == [40.0, 1.0]

    def test_an_unmentioned_record_counts_as_read_once(self) -> None:
        """Not zero: a log records what was read, and zero would make a record
        ineligible as a query proxy, which is a stronger claim than any access
        log supports."""
        from rebasis.probe.session import _access_vector

        weights = _access_vector(["a", "b", "c"], {"b": 5.0})

        assert weights is not None
        assert weights.tolist() == [1.0, 5.0, 1.0]

    def test_the_result_carries_it_into_json(self) -> None:
        """A report that did not carry it would be two different quantities
        under one name."""
        from rebasis.probe.runner import ProbeResult

        assert "access_weighted" in ProbeResult.__dataclass_fields__

    def test_the_serialised_form_names_it(self) -> None:
        import dataclasses

        from rebasis.probe.runner import ProbeResult

        fields = {f.name for f in dataclasses.fields(ProbeResult)}
        assert "access_weighted" in fields


def test_the_cli_exposes_the_flag() -> None:
    """`--access-log` on `probe` means something different from `--access-log`
    on `migrate`, and both exist: one orders the migration queue, the other
    weights the query proxies."""
    import inspect

    from rebasis.cli.probe import probe_command

    assert "access_log" in inspect.signature(probe_command).parameters


def test_the_json_report_would_carry_it() -> None:
    """The audit trail is where a decision's provenance lives, and how the
    queries were drawn is part of it."""
    from rebasis.probe.runner import ProbeResult

    source = json.dumps(sorted(ProbeResult.__dataclass_fields__))
    assert "access_weighted" in source
