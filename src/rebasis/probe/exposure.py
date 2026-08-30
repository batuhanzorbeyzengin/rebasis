"""How alignable is this index?

**This is a defensive diagnostic and it returns a number.** It does not return
aligned vectors, does not reconstruct text, and performs no inversion. That
distinction is the whole design and it is not arbitrary: a number grants nobody
a capability — mini-vec2vec is published and pip-installable — while a command
that handed back a translation would package that capability and ship it to this
tool's user base. The first is defence; the second is distribution.

## The question

vec2vec's headline finding is a security finding: embeddings can be translated
from unseen documents by unseen encoders while preserving their geometry, and
from a translation an adversary holding only the vectors can infer things about
the documents. Its own framing is that *vector databases reveal (almost) as much
as their inputs*. mini-vec2vec (arXiv:2510.02348) then reduced the cost of that
translation from a day of adversarial training to an orthogonal solve and an
assignment.

So the operational question is: **if somebody takes my vectors, how well can
they align them to a space they already understand?** Nothing answers it, and
`rebasis` is already connected to the index and already draws a read-only sample.

## What is measured, and why it is an upper bound

1. Draw a sample of the index. Split it in two halves that share no document.
2. Read the index's own vectors for half A.
3. Embed half B's text with a **local** reference model.
4. Fit the unpaired map from B's space into A's — :func:`~rebasis.core.unpaired.align_unpaired`,
   which is handed two float matrices and nothing else.
5. On documents held out of the fit, measure how often the map carries a
   reference-model vector to *its own* index vector before any other. That is
   **alignability**.

An adversary has no access to the corpus — only to their own text — so this
simulates a situation **better** than theirs on the one axis that matters most:
the reference half is drawn from the very distribution being attacked. The
figure is therefore an upper bound, and is reported as one.

## What it deliberately does not do

**It does not predict alignability, it measures it.** An earlier design for this
command reported the *centroid-agreement* diagnostic, which
`spikes/unpaired_align.py` measures ranking the outcome at Spearman +0.833. That
is the right quantity when the outcome is unavailable — and here it is
available. An index's owner holds both the vectors and the text, so they can
compute the map's actual identification accuracy directly. Reporting a predictor
of a number you are holding is strictly worse, and centroid agreement is not
even computable without an oracle map fitted on paired data
(`_reference_permutation` in that spike says so in as many words).

**It offers no band.** low/medium/high would be a classifier, and the evidence
does not support one — see `docs/exposure.md`. The number and the document that
explains it, and nothing between them.

**The reference model must be local.** Sending vectors or text to a hosted
endpoint to measure exposure creates the exposure being measured. The remote
backends are refused rather than warned about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from rebasis.core.base import l2_normalize, pad_or_truncate
from rebasis.core.unpaired import align_unpaired, preprocess
from rebasis.errors import ConfigError, InsufficientSamples
from rebasis.observability import Spans, get_logger, span
from rebasis.probe.session import draw_corpus_sample
from rebasis.types import as_float32

if TYPE_CHECKING:
    from collections.abc import Callable

    from rebasis.probe.session import CorpusSample
    from rebasis.store.base import VectorStore
    from rebasis.types import Embedder, FloatArray

__all__ = ["ExposureResult", "measure_exposure"]

log = get_logger(__name__)

#: Documents held out of the fit to measure the map on.
#:
#: They are scored against each other, so the pool size is part of the number:
#: identifying one document among 500 is a weaker result than among 8,192, which
#: is what the paper ranks against. The figure is reported with the pool beside
#: it for exactly that reason.
HELDOUT = 1_000

#: Fewest documents the measurement will run on.
#:
#: Below this the clustering that the whole method starts from has nothing to
#: find: 20 clusters over a few hundred vectors are 20 arbitrary partitions of
#: noise, and the alignability that comes out describes the sample rather than
#: the index.
MIN_DOCUMENTS = 2_000

#: Spread across seeds above which the run says the attempts disagreed.
#:
#: Measured over 32 cells (`docs/exposure.md`): the median seed-to-seed spread
#: is 0.159 and the maximum 0.969. A tenth is where a reader should be told the
#: attempts did not agree, because at that point the choice of seed is moving
#: the answer more than most of the drivers do.
SPREAD_LIMIT = 0.10

#: Independent alignments run before an answer is given.
#:
#: **Not a refinement — a correction.** The method is stochastic in three places
#: (k-means initialisation, the assignment's restarts, the ICP sampling) and
#: `docs/exposure.md` measures the seed-to-seed spread on one index at a median
#: of 0.159 and a maximum of **0.969**. A single run could therefore report 0.03
#: or 0.99 for the same index, and a security figure that moves that far between
#: runs is not one anybody can act on.
#:
#: Three because it is the smallest number that can show a spread at all, and
#: because the cost is linear in it.
SEEDS = 3

#: Embedding backends that send text off the machine.
#:
#: Refused, not warned about. Every other command in this tool warns; this one
#: is measuring exposure, and creating some in order to measure it is the one
#: failure mode that would make the command worse than not having it.
REMOTE_BACKENDS = frozenset({"openai_compat", "ollama"})


@dataclass(slots=True)
class ExposureResult:
    """One scalar, and everything needed to read it honestly."""

    #: The **best** of :data:`SEEDS` independent alignments: how often the map
    #: put a held-out document's own index vector first, out of ``pool``.
    #:
    #: The best rather than the mean, and that follows from what the number is
    #: for. An adversary can run the method more than once and keep whichever
    #: attempt worked, so the figure that describes their position is the best
    #: one available — the same reason every other caveat on this result points
    #: upward.
    alignability: float
    #: How many documents it was ranked against. Part of the number.
    pool: int
    #: Mean rank of the true match. Reported because a map that ranks the answer
    #: second every time and a map that ranks it five-hundredth are the same
    #: ``alignability`` and are not the same finding.
    mean_rank: float
    #: Every seed's result, in the order they were run. Carried because the
    #: spread is the finding: where these disagree, one run of this method says
    #: much less than its own headline number suggests.
    per_seed: list[float]
    reference_model: str
    n_sampled: int
    n_total: int
    seed: int
    #: What the method could say about itself. Diagnostic, never decisive.
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form. Scalars only — by construction, not by filtering."""
        return {
            "alignability": round(self.alignability, 4),
            "alignability_per_seed": [round(v, 4) for v in self.per_seed],
            "alignability_spread": round(max(self.per_seed) - min(self.per_seed), 4)
            if self.per_seed
            else 0.0,
            "pool": self.pool,
            "mean_rank": round(self.mean_rank, 2),
            "reference_model": self.reference_model,
            "reference_is_local": True,
            "n_sampled": self.n_sampled,
            "n_total": self.n_total,
            "seed": self.seed,
            "upper_bound": True,
            "diagnostics": dict(self.diagnostics),
            "warnings": list(self.warnings),
        }


def refuse_remote_reference(embedder: Embedder) -> None:
    """Refuse a reference model that would send text off this machine.

    Raises:
        ConfigError: When the embedder's backend is a hosted endpoint.

    Read off the embedder's module rather than off the model id, because a model
    id says nothing about where it runs — the same name can be a local
    checkpoint or a hosted endpoint, and it is the backend that decides.
    """
    module = type(embedder).__module__.rsplit(".", 1)[-1]
    if module in REMOTE_BACKENDS:
        message = f"{module} sends text off this machine, and this command measures exposure."
        raise ConfigError(
            message,
            hint=(
                "Use a local reference model — sentence-transformers or fastembed. "
                "Measuring exposure by creating some is the one thing this command "
                "must not do."
            ),
            context={"embed_backend": module},
        )


def measure_exposure(  # noqa: PLR0913 - one argument per pipeline input
    store: VectorStore,
    reference: Embedder,
    *,
    size: int = 20_000,
    heldout: int = HELDOUT,
    strategy: str = "stratified",
    seed: int = 0,
    seeds: int = SEEDS,
    batch_size: int = 1_000,
    config: dict[str, Any] | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> ExposureResult:
    """Measure how well this index aligns to a local reference space.

    Read-only in every path, like every other diagnostic here: the store is
    sampled and nothing is written.

    Args:
        store: The index. Needs vectors and text — the text is what the
            reference half is embedded from.
        reference: A **local** model standing in for the public one an adversary
            would use. Refused if it is a hosted endpoint.
        size: Documents drawn.
        heldout: Documents kept out of the fit and used to measure it. They are
            ranked against each other, so this is the pool the number is
            relative to.
        strategy: How the sample is drawn.
        seed: Recorded; the alignments run at ``seed``, ``seed + 1``, and so on.
        seeds: Independent alignments to run before answering. See :data:`SEEDS`
            — one is not enough, and the measurement that says so is in
            `docs/exposure.md`.
        batch_size: Store read size.
        config: Overrides for :data:`~rebasis.core.unpaired.DEFAULTS`.
        on_stage: Progress callback.
    """
    refuse_remote_reference(reference)
    stage = on_stage if on_stage is not None else (lambda _text: None)

    with span(Spans.PROBE, {"seed": seed}):
        stage("Sampling the index")
        corpus = draw_corpus_sample(
            store,
            size=size,
            heldout=heldout,
            strategy=strategy,
            seed=seed,
            batch_size=batch_size,
        )
        if len(corpus) < MIN_DOCUMENTS:
            message = (
                f"{len(corpus)} documents is too few to measure alignability; "
                f"the method needs at least {MIN_DOCUMENTS}."
            )
            raise InsufficientSamples(
                message,
                hint=(
                    "Increase --sample, or accept that an index this small cannot "
                    "be aligned by this method and therefore cannot be scored by it."
                ),
                context={"count": len(corpus)},
            )

        held, first, second = _split(corpus, seed=seed)
        stage(f"Embedding {second.size + held.size:,} documents with the reference model")
        texts = [corpus.texts[i] for i in np.concatenate([second, held])]
        reference_vectors = as_float32(reference.encode(texts, kind="document", progress=False))
        reference_second = reference_vectors[: second.size]
        reference_held = reference_vectors[second.size :]

        width = max(corpus.old_vectors.shape[1], reference_vectors.shape[1])
        source = preprocess(reference_second, width)
        target = preprocess(corpus.old_vectors[first], width)

        # The preprocessing is shared across seeds because it is deterministic:
        # padding, a mean and a normalisation. Only the alignment is re-run,
        # which is where all three sources of randomness are.
        scores: list[tuple[float, float]] = []
        for offset in range(max(1, seeds)):
            stage(f"Aligning, attempt {offset + 1} of {max(1, seeds)}")
            alignment = align_unpaired(source.hat, target.hat, seed=seed + offset, config=config)
            scores.append(
                _identification(
                    reference_held,
                    corpus.old_vectors[held],
                    alignment.rotation,
                    source=source,
                    target=target,
                    width=width,
                )
            )

        # The best attempt, and the rank that came with *that* attempt rather
        # than the best rank from any of them: the two describe one alignment
        # and pairing them across runs would describe none.
        best, mean_rank = max(scores, key=lambda pair: pair[0])

    return ExposureResult(
        alignability=best,
        per_seed=[value for value, _ in scores],
        pool=int(held.size),
        mean_rank=mean_rank,
        reference_model=reference.profile.model_id,
        n_sampled=len(corpus),
        n_total=corpus.n_total,
        seed=seed,
        diagnostics=alignment.diagnostics,
        warnings=_warnings(corpus, held, [value for value, _ in scores]),
    )


def _split(corpus: CorpusSample, *, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Held-out documents, and two fit halves that share none of them or each other.

    Structural rather than conventional. The two halves are given different row
    counts and independently permuted orders, so no bijection between them
    exists to leak even by accident — the property `spikes/unpaired_align.py`
    asserts and re-checks under a permutation.
    """
    rng = np.random.default_rng(seed * 7_919 + 13)
    order = rng.permutation(len(corpus))
    held = np.sort(corpus.query_positions)
    # `np.isin` rather than a comprehension against a set. The set literal
    # inside a comprehension is rebuilt per item, which made this quadratic in
    # the sample size — 20,000 documents against a 1,000-document hold-out is
    # twenty million membership tests for a mask numpy computes in one pass.
    remaining = order[~np.isin(order, held)]
    # Deliberately unequal, and `(n - 1) // 2` rather than `n // 2` because the
    # second is equal whenever n is even. Equal row counts are the first thing
    # that could be mistaken for a correspondence, and the method must not be
    # able to find one even by accident.
    cut = (len(remaining) - 1) // 2
    first = rng.permutation(remaining[:cut])
    second = rng.permutation(remaining[cut:])
    return held, first, second


def _identification(  # noqa: PLR0913 - the two halves, the map and the geometry it lives in
    reference_held: FloatArray,
    index_held: FloatArray,
    rotation: FloatArray,
    *,
    source: Any,
    target: Any,
    width: int,
) -> tuple[float, float]:
    """How often the map puts a document's own index vector first.

    The map was fitted in the centred, normalised geometry, so the held-out
    vectors are carried into it the same way before being rotated — applying a
    map to vectors from a distribution it never saw is the mistake this exists
    to avoid rather than to make.

    Both sides are scored in the *target's* centred space rather than being
    pushed back out to raw coordinates. Nothing here needs raw coordinates, and
    not computing them is the cheapest possible guarantee that none is returned.
    """
    mapped = l2_normalize(
        as_float32((pad_or_truncate(reference_held, width) - source.mean) @ rotation), copy=False
    )
    targets = l2_normalize(as_float32(pad_or_truncate(index_held, width) - target.mean), copy=True)
    scores = as_float32(mapped @ targets.T)
    truth = np.diag(scores)
    ranks = (scores > truth[:, None]).sum(axis=1) + 1
    return float((ranks == 1).mean()), float(ranks.mean())


def _warnings(corpus: CorpusSample, held: np.ndarray, per_seed: list[float]) -> list[str]:
    """What would make this number read as more than it is."""
    notes = [
        (
            "This is an upper bound. The reference half was drawn from your own "
            "corpus, so the alignment saw the very distribution it was attacking; "
            "an adversary has only their own text. It is also a bound for this "
            "reference model alone — a better one may exist tomorrow."
        ),
        (
            f"Each document was ranked against {held.size:,} others. Identifying one "
            f"among a thousand is a weaker result than among a hundred thousand, so "
            f"the pool is part of the number rather than a footnote to it."
        ),
    ]
    spread = max(per_seed) - min(per_seed) if per_seed else 0.0
    if spread > SPREAD_LIMIT:
        notes.append(
            f"The {len(per_seed)} attempts disagreed by {spread:.2f} "
            f"({', '.join(f'{v:.3f}' for v in per_seed)}). The method is "
            f"stochastic and this index is one where that matters: the figure "
            f"above is the best attempt, which is the right reading for an "
            f"upper bound and a poor one for anything else. Raise --sample, or "
            f"read the spread rather than the number."
        )

    if corpus.n_total > len(corpus) * 10:
        notes.append(
            f"{len(corpus):,} of {corpus.n_total:,} records were sampled. Alignability "
            f"is a property of a distribution rather than of a count, so this is "
            f"usually enough — but a corpus with a large tail the sample missed is "
            f"a corpus this did not measure."
        )
    return notes
