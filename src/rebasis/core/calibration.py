"""Score calibration.

An adapter preserves *ranking*; it does not preserve the *scale* of similarity
scores. That distinction is invisible until it breaks something, and what it
breaks is every pipeline with a fixed threshold such as ``similarity > 0.7``.

**Measured (M0, 72 configurations):** without calibration the KS distance
between adapted and oracle score distributions has a median of **0.924**, and
**100%** of configurations exceed the decision rule's score_shift warning
threshold of 0.1. A warning that always fires carries no information — so that
check has to be evaluated *after* calibration, not before.

Isotonic regression maps adapted scores onto the oracle distribution. Being
monotone it cannot reorder anything, so ARR is unchanged by construction —
verified in M0: ranking preserved in **100%** of cases, median KS falling to
**0.094**.

It is not a complete fix: **46%** of configurations remain above 0.1 even after
calibration. rebasis therefore calibrates *and* warns, rather than calibrating
and claiming the problem is solved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from rebasis.types import FloatArray, as_float32

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["ScoreCalibrator"]


class ScoreCalibrator:
    """Isotonic mapping from adapted scores onto the oracle distribution.

    Stored inside the ``.rbs`` file so that a bridge deployed months later still
    returns scores on the scale the user's thresholds were tuned against.
    """

    def __init__(self, x_thresholds: FloatArray, y_thresholds: FloatArray) -> None:
        self.x_thresholds = x_thresholds
        self.y_thresholds = y_thresholds

    @classmethod
    def fit(cls, adapted_scores: FloatArray, oracle_scores: FloatArray) -> ScoreCalibrator:
        """Fit the calibrator on a held-out set.

        Both arrays are flattened: the mapping is between score *distributions*,
        not between paired individual results.
        """
        from sklearn.isotonic import IsotonicRegression

        x = as_float32(adapted_scores).ravel()
        y = as_float32(oracle_scores).ravel()
        model = IsotonicRegression(out_of_bounds="clip", increasing=True)
        model.fit(x, y)
        return cls(as_float32(model.X_thresholds_), as_float32(model.y_thresholds_))

    def transform(self, scores: FloatArray) -> FloatArray:
        """Map scores onto the calibrated scale.

        Implemented with ``np.interp`` rather than by keeping the sklearn object
        alive: the ``.rbs`` file stores two arrays, loading needs no sklearn, and
        the transform stays cheap enough for a search result set.
        """
        flat = as_float32(scores).ravel()
        mapped = np.interp(flat, self.x_thresholds, self.y_thresholds)
        return as_float32(mapped).reshape(np.shape(scores))

    def state_dict(self) -> dict[str, FloatArray]:
        """Weights to serialise."""
        return {
            "calibration_x": self.x_thresholds,
            "calibration_y": self.y_thresholds,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, FloatArray]) -> ScoreCalibrator:
        """Reconstruct from stored arrays."""
        return cls(as_float32(state["calibration_x"]), as_float32(state["calibration_y"]))

    def summary(self) -> dict[str, Any]:
        """Compact description for the report and the audit record."""
        return {
            "n_knots": int(self.x_thresholds.size),
            "input_range": [float(self.x_thresholds[0]), float(self.x_thresholds[-1])],
            "output_range": [float(self.y_thresholds[0]), float(self.y_thresholds[-1])],
        }
