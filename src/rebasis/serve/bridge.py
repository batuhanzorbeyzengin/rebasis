"""``Bridge`` — the serving-time API.

This is the object a user's RAG application holds. Its whole job is one call::

    bridge = Bridge.load("adapters/minilm-to-bgem3.rbs")
    q_old = bridge.to_index_space(new_model.encode(["how do I deploy?"]))
    results = collection.query(query_embeddings=q_old, n_results=10)

**Everything here is shaped by one constraint: this runs in the hot path of a RAG
request**, on a budget of 15 µs per query. Three consequences, all deliberate:

* **Validation happens once, in :meth:`load`.** Fingerprints, dimensions and
  integrity are checked there and never again; ``to_index_space`` performs no
  checks at all.
* **No logging on the hot path.** Not even at DEBUG — a log call costs more than
  the arithmetic here.
* **torch is never imported.** Not lazily, not optionally — this module does not
  reference it. A host→device→host round trip alone exceeds 15 µs, and M0
  confirmed the cost at this size is per-call overhead rather than FLOPs. A GPU
  would make this slower.

The default direction is ``query_to_old``, which is what makes the bridge phase
free of risk: the index is never written, so "rolling back" is just not using the
adapter, and no shadow copy is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rebasis.core.base import l2_normalize
from rebasis.core.serialization import load_adapter

if TYPE_CHECKING:
    from pathlib import Path

    from rebasis.core.base import BaseAdapter
    from rebasis.core.calibration import ScoreCalibrator
    from rebasis.core.serialization import AdapterManifest
    from rebasis.types import EncodingProfile, FloatArray

__all__ = ["Bridge"]


class Bridge:
    """Maps new-model query vectors into an existing index's space."""

    __slots__ = ("_adapter", "_apply", "_calibrator", "_manifest")

    def __init__(
        self,
        adapter: BaseAdapter,
        manifest: AdapterManifest,
        calibrator: ScoreCalibrator | None = None,
    ) -> None:
        self._adapter = adapter
        self._manifest = manifest
        self._calibrator = calibrator
        # Bound once at construction. Resolving `self._adapter.apply` on every
        # call costs an attribute lookup per query, which is measurable against
        # a budget this tight.
        self._apply = adapter.apply

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        expected_old: EncodingProfile | None = None,
        expected_new: EncodingProfile | None = None,
        verify: bool = False,
    ) -> Bridge:
        """Load an adapter and validate it — the only place validation happens.

        Args:
            path: The ``.rbs`` directory.
            expected_old: Profile of the model the index was built with. Passing
                it turns a wrong adapter into a load failure instead of silently
                degraded retrieval.
            expected_new: Profile of the model now producing query vectors.
            verify: Recompute every tensor hash. Milliseconds for a few
                megabytes, and worth it when an adapter has travelled between
                machines.
        """
        from pathlib import Path as _Path

        adapter, manifest, calibrator = load_adapter(
            _Path(path),
            expected_old=expected_old,
            expected_new=expected_new,
            verify=verify,
        )
        return cls(adapter, manifest, calibrator)

    def to_index_space(self, vectors: FloatArray, *, normalize: bool = True) -> FloatArray:
        """Map new-model vectors into the index's space — the hot path.

        No validation, no logging, no dictionary construction. A dimension
        mismatch surfaces as a numpy error rather than a rebasis one, because
        checking on every query would cost more than the mapping itself — and
        :meth:`load` has already established that the dimensions agree.

        Args:
            vectors: Shape ``(n, d_new)`` or ``(d_new,)``.
            normalize: Renormalise the output. Leave it on unless the caller
                normalises downstream; the index expects unit vectors.
        """
        mapped = self._apply(vectors)
        return l2_normalize(mapped, copy=False) if normalize else mapped

    def calibrate_scores(self, scores: FloatArray) -> FloatArray:
        """Map similarity scores back onto the pre-migration scale.

        Ranking is unaffected — the calibrator is monotone — so this is only
        needed by pipelines that compare scores against a fixed threshold. Those
        pipelines need it badly: M0 measured that **every** adapter shifts the
        score distribution far enough to break such a filter.

        Returns the scores unchanged when the adapter carries no calibrator.
        """
        if self._calibrator is None:
            return scores
        return self._calibrator.transform(scores)

    @property
    def input_dim(self) -> int:
        """Dimensionality this bridge expects from the new model."""
        return self._adapter.input_dim

    @property
    def output_dim(self) -> int:
        """Dimensionality of the index this bridge targets."""
        return self._adapter.output_dim

    @property
    def adapter_type(self) -> str:
        """Which adapter is inside, for reports and diagnostics."""
        return self._manifest.adapter_type

    @property
    def calibrator(self) -> ScoreCalibrator | None:
        """The score calibrator, for a caller that has to merge two rankings.

        Exposed because merging results from two embedding spaces needs the
        calibrator itself rather than the mapping :meth:`calibrate_scores`
        applies — :class:`~rebasis.serve.mixed.MixedSpaceSearch` hands it to
        `calibrated_merge`, which decides per document rather than per array.
        ``None`` when the adapter carries no calibrator, which is the signal to
        fall back to rank fusion.
        """
        return self._calibrator

    @property
    def has_calibrator(self) -> bool:
        """Whether score calibration is available."""
        return self._calibrator is not None

    @property
    def manifest(self) -> AdapterManifest:
        """The full manifest — models, direction, dimensions, fingerprints."""
        return self._manifest

    def describe(self) -> dict[str, Any]:
        """A compact summary for logs and reports. Never called on the hot path."""
        return {
            "adapter_type": self._manifest.adapter_type,
            "direction": self._manifest.direction,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "old_model": self._manifest.old_model_id,
            "new_model": self._manifest.new_model_id,
            "has_calibrator": self.has_calibrator,
            "n_params": self._adapter.n_params(),
        }

    def __repr__(self) -> str:
        return (
            f"Bridge({self._manifest.old_model_id} ← {self._manifest.new_model_id}, "
            f"{self._manifest.adapter_type}, {self.input_dim}→{self.output_dim})"
        )
