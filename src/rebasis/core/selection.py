"""``auto``: fit every candidate, measure, pick the winner.

Fitting all three parametrisations and keeping the best is the tool's most
important UX decision, and M0's measurements make the case stronger than
expected: across 4 corpora and 3 model pairs, **which adapter wins changes with
the pair**, and the top four candidates sit inside the measurement uncertainty of
one another (0.834–0.839 against a ±0.042 confidence interval). There is no
adapter that is simply best.

Two M0 findings are wired directly into the candidate list:

* **Centred Procrustes replaces plain Procrustes as the default.** Centering was
  worth +0.26 ARR on average, up to +0.75 (section 4 of ``docs/m0-findings.md``).
  Plain Procrustes stays in the list so the report can show what centering
  bought.
* **CSLS is a candidate, not a correction.** It adds +0.103 on weak adapters and
  *costs* −0.045 on strong ones (section 5 of the findings). Applying it
  unconditionally would degrade exactly the adapters that are working.

Efficiency: normalisation, sampling and ground truth are computed once and shared
by every candidate. Fitting each candidate on its own preprocessing would triple
the cost of the most expensive step in the pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rebasis.core.base import BaseAdapter, l2_normalize
from rebasis.core.identity import IdentityAdapter
from rebasis.core.linear import LinearAdapter, LowRankAffineAdapter
from rebasis.core.procrustes import CenteredProcrustesAdapter, ProcrustesAdapter
from rebasis.core.scaling import ScaledAdapter
from rebasis.observability import Events, get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from rebasis.types import FloatArray

__all__ = ["CANDIDATE_METHODS", "AdapterCandidate", "fit_candidates", "select_best"]

log = get_logger(__name__)

#: Methods `auto` tries, cheapest first so that a run interrupted early still
#: has a usable result.
#:
#: `procrustes_centered` leads deliberately: it is the measured default and, when
#: everything is within noise of everything else, the tie-break is cost —
#: half the MLP's memory, a third of its latency, and a closed form that needs no
#: seed.
CANDIDATE_METHODS: tuple[str, ...] = (
    "procrustes_centered",
    "procrustes",
    "linear",
    "low_rank_affine",
    "procrustes_centered+dsm",
    "residual_mlp",
)

#: Methods available without torch.
CPU_ONLY_METHODS: frozenset[str] = frozenset(CANDIDATE_METHODS) - {"residual_mlp"}


@dataclass(slots=True)
class AdapterCandidate:
    """One fitted candidate together with how it was produced and how it scored."""

    method: str
    adapter: BaseAdapter
    fit_seconds: float
    #: Which encoding the fit pairs used — asymmetric models encode queries and
    #: documents differently. ``"document"`` is the shared strategy, one adapter
    #: for both kinds; ``"query"`` is the query-specific one, fitted on the same
    #: texts encoded the way a query would be.
    fit_kind: str = "document"
    score: float | None = None
    score_with_csls: float | None = None
    use_csls: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def best_score(self) -> float:
        """The better of the plain and CSLS scores, treating unscored as -inf."""
        candidates = [s for s in (self.score, self.score_with_csls) if s is not None]
        return max(candidates) if candidates else float("-inf")

    @property
    def name(self) -> str:
        """Display name, with the CSLS and fit-kind suffixes when they apply.

        The fit kind is in the name because it changes what the adapter *is*,
        not just how it scored: two candidates of the same type fitted on
        differently-encoded pairs are different adapters, and a report that
        called them both ``procrustes`` would be hiding the comparison.
        """
        base = f"{self.adapter.type_name}+csls" if self.use_csls else self.adapter.type_name
        return base if self.fit_kind == "document" else f"{base}@query"


def _fit_one(method: str, src: FloatArray, dst: FloatArray, **kwargs: Any) -> BaseAdapter:
    builders: dict[str, Callable[[], BaseAdapter]] = {
        "identity": lambda: IdentityAdapter.fit(src, dst),
        "procrustes": lambda: ProcrustesAdapter.fit(src, dst),
        "procrustes_centered": lambda: CenteredProcrustesAdapter.fit(src, dst),
        "linear": lambda: LinearAdapter.fit(src, dst),
        "low_rank_affine": lambda: LowRankAffineAdapter.fit(src, dst, rank=kwargs.get("rank")),
    }
    if method in builders:
        return builders[method]()

    if method == "procrustes_centered+dsm":
        base = CenteredProcrustesAdapter.fit(src, dst)
        return ScaledAdapter.fit(base, src, dst)

    if method == "residual_mlp":
        from rebasis.core.residual_mlp import ResidualMLPAdapter

        return ResidualMLPAdapter.fit(src, dst, device=kwargs.get("device", "cpu"))

    msg = f"unknown adapter method: {method!r}"
    raise ValueError(msg)


def fit_candidates(  # noqa: PLR0913 - one argument per documented fitting input
    src: FloatArray,
    dst: FloatArray,
    *,
    methods: Sequence[str] = CANDIDATE_METHODS,
    normalize: bool = True,
    skip_unavailable: bool = True,
    fit_kind: str = "document",
    **kwargs: Any,
) -> list[AdapterCandidate]:
    """Fit every requested method on shared preprocessing.

    Args:
        src: New-model vectors of the matched documents.
        dst: Old-model vectors of the same documents, in the same order.
        methods: Which candidates to fit.
        normalize: ℓ2-normalise both sides before fitting. On by default.
        skip_unavailable: Skip a candidate whose optional dependency is missing
            rather than failing the whole run. `auto` must still produce an
            answer on a torch-free install.
        fit_kind: Which encoding ``src`` used — ``"document"`` or ``"query"``.
            Recorded on each candidate so the shared and query-specific
            strategies stay distinguishable in the report.
        **kwargs: Forwarded to individual fitters (``rank``, ``device``).

    Returns:
        The successfully fitted candidates, unscored.
    """
    # Shared preprocessing: doing it per candidate would triple the cost of
    # `auto`, which is already the largest item in the fit budget.
    if normalize:
        src = l2_normalize(src)
        dst = l2_normalize(dst)

    candidates: list[AdapterCandidate] = []
    for method in methods:
        started = time.perf_counter()
        try:
            adapter = _fit_one(method, src, dst, **kwargs)
        except Exception as exc:
            if not skip_unavailable:
                raise
            log.info(Events.FIT_ADAPTER_FITTED, adapter_type=method, error_code=type(exc).__name__)
            continue
        elapsed = time.perf_counter() - started
        log.info(
            Events.FIT_ADAPTER_FITTED,
            adapter_type=method,
            count=int(src.shape[0]),
            dim=int(src.shape[1]),
            duration_ms=round(elapsed * 1000, 2),
        )
        candidates.append(
            AdapterCandidate(method=method, adapter=adapter, fit_seconds=elapsed, fit_kind=fit_kind)
        )

    return candidates


def select_best(candidates: Sequence[AdapterCandidate]) -> AdapterCandidate:
    """Pick the highest-scoring candidate, breaking ties by cost.

    Ties are common and not incidental: M0 found the top four candidates inside
    one another's confidence intervals. When scores are indistinguishable the
    decision belongs to cost, so the tie-break prefers **fewer parameters** —
    which favours centred Procrustes over the MLP at equal quality, and brings
    with it lower latency and a closed-form fit.

    Raises:
        ValueError: When there are no candidates to choose from.
    """
    scored = [c for c in candidates if c.best_score > float("-inf")]
    if not scored:
        msg = "no candidate was scored; nothing to select"
        raise ValueError(msg)

    #: Scores within this distance count as equal. Derived from M0's median
    #: bootstrap CI half-width for ARR (±0.024): claiming a winner inside that
    #: band would be reading sampling noise as signal.
    tolerance = 0.005
    best_score = max(c.best_score for c in scored)
    tied = [c for c in scored if best_score - c.best_score <= tolerance]
    winner = min(tied, key=lambda c: c.adapter.n_params())

    log.info(
        Events.FIT_ADAPTER_SELECTED,
        adapter_type=winner.name,
        arr_r10=round(winner.best_score, 4),
    )
    return winner
