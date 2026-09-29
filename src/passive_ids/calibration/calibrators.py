"""Isotonic calibration: maps each model's native raw_score onto a
comparable 0-1 confidence scale.

Keyed by threat_class, not model_name -- verified this matters
concretely, not just in principle: models.c2_beaconing and
models.exfiltration both report model_name="isolation_forest" (same
string), but they are separate fitted IsolationForest instances over
completely unrelated feature distributions (IAT statistics vs.
byte-ratio asymmetry). Keying a calibrator registry by model_name
alone would silently share one calibration curve between them --
exactly the mistake this project's calibration design was built to
avoid. threat_class is unique per detector in this architecture (no
two of the six models share one), so it's the correct key here.

Isotonic regression, not a parametric fit (sigmoid/Platt scaling):
Isolation Forest's raw anomaly score isn't linearly or even simply
monotonically-shaped relative to true attack probability -- isotonic
only assumes monotonicity (higher raw score -> higher or equal true
probability), which is the one assumption that actually holds across
all six models' differently-shaped native outputs (a probability
already in [0,1] from river/sklearn classifiers, or an unbounded,
roughly-zero-centered anomaly score from Isolation Forest).

Verified directly before writing, not assumed: sklearn's
IsotonicRegression fails before fit() with a plain AttributeError
('no attribute X_min_'), NOT the NotFittedError every other sklearn
-family wrapper in this project (IsolationForest, LogisticRegression,
LightGBM, XGBoost) raises. Reusing the established
`except NotFittedError` pattern here would have silently failed to
catch this. This wrapper guards with an explicit _fitted flag checked
before calling predict, rather than relying on catching any specific
exception -- the more defensive of the two options, and correct
regardless of which exception type a future sklearn version happens
to raise.
"""
from __future__ import annotations

from dataclasses import replace

from sklearn.isotonic import IsotonicRegression

from passive_ids.models.base import DetectionResult


class ScoreCalibrator:
    """One isotonic calibrator for one detector's raw_score history."""

    def __init__(self) -> None:
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._fitted = False
        self.training_examples_seen = 0

    def fit(self, raw_scores: list[float], labels: list[int]) -> None:
        """labels are ground truth (1 = confirmed attack, 0 = confirmed
        benign) -- from SOC verdicts, once feedback/ exists to supply
        them, or labeled validation data in the meantime. INCONCLUSIVE
        verdicts must already be filtered out by the caller before
        reaching here (see verdicts/schema.py's VerdictLabel) -- this
        class has no way to detect that on its own; it will happily
        fit on whatever it's given.
        """
        self._iso.fit(raw_scores, labels)
        self._fitted = True
        self.training_examples_seen = len(raw_scores)

    def calibrate(self, raw_score: float) -> float | None:
        """Returns a confidence in [0, 1], or None if unfit.

        None here means "no calibration curve exists yet" -- callers
        should fall back to the model's raw_score with a disclosed
        caveat (see each model's own docstring on what raw_score does
        and doesn't mean), not silently substitute 0.0 or 0.5 as if
        that were a real calibrated judgment.
        """
        if not self._fitted:
            return None
        return float(self._iso.predict([raw_score])[0])


class CalibratorRegistry:
    """One ScoreCalibrator per threat_class, created lazily on first use."""

    def __init__(self) -> None:
        self._calibrators: dict[str, ScoreCalibrator] = {}

    def get(self, threat_class: str) -> ScoreCalibrator:
        if threat_class not in self._calibrators:
            self._calibrators[threat_class] = ScoreCalibrator()
        return self._calibrators[threat_class]

    def fit(self, threat_class: str, raw_scores: list[float], labels: list[int]) -> None:
        self.get(threat_class).fit(raw_scores, labels)

    def calibrate(self, threat_class: str, raw_score: float) -> float | None:
        return self.get(threat_class).calibrate(raw_score)

    def apply(self, result: DetectionResult) -> DetectionResult:
        """Returns a copy of `result` with calibrated_confidence filled
        in, or the same result unchanged if no calibrator is fitted
        yet for its threat_class. DetectionResult is frozen (see
        models/base.py), so this returns a new instance via
        dataclasses.replace rather than mutating in place.
        """
        confidence = self.calibrate(result.threat_class, result.raw_score)
        if confidence is None:
            return result
        return replace(result, calibrated_confidence=confidence)
