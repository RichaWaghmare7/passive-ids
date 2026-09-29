"""C2 beaconing detector: Isolation Forest over Welford IAT statistics.

Consumes flow-level feature vectors -- specifically the nested "iat"
statistics (mean, std, coefficient_of_variation) that
flow_state.confirmed_table.ConfirmedFlowEntry has been computing via
features.welford.WelfordIAT since that module was built. No new
timing computation happens here; this model's entire job is learning
what "suspiciously regular" looks like over data that already exists.

This is deliberately a different input shape from
models.ddos_hoeffding.DDoSHoeffdingDetector, which consumes
source-aggregate-level vectors (from the tumbling-window trigger).
This model expects the flow-level vectors FlowTimeoutTrigger emits --
mixing the two up would silently score the wrong thing, since
FlowTimeoutTrigger's snapshot() nests "iat" as a sub-dict rather than
flattening it.

Batch-retrained, not online -- matches models.base.Detector's
documented is_online_learner=False contract exactly: update() is
inherited as a no-op (unsupervised anomaly detection doesn't take
per-example labels the way the DDoS classifier does anyway). Training
happens through fit(), called with a batch of presumed-mostly-normal
flow vectors, exactly as feedback/retrain.py is meant to eventually
do -- not built yet, so this model ships unfit by default.

Verified directly before writing, not assumed:
  - scikit-learn's IsolationForest raises NotFittedError from both
    decision_function() and predict() before fit() has been called --
    unlike river's HoeffdingTreeClassifier, which returns an empty
    dict gracefully. This model's cold-start handling exists
    specifically because sklearn does NOT fail as gracefully as river
    does here.
  - decision_function() is NEGATIVE for anomalous points and POSITIVE
    for normal ones -- the opposite sign convention from this
    project's "higher raw_score = more attack-like" contract, checked
    directly against a synthetic clean-beacon-vs-normal-traffic
    comparison. This model negates it before returning raw_score.
  - The negated decision_function is NOT a calibrated 0-1 probability
    (unlike the DDoS detector's river-native output) -- it's an
    unbounded, roughly-zero-centered score. Isotonic calibration
    (calibration/, not yet built) is what maps this onto a comparable
    0-1 scale alongside the other five models' native outputs; this
    module does not attempt that itself.
"""
from __future__ import annotations

from typing import Any

from sklearn.exceptions import NotFittedError
from sklearn.ensemble import IsolationForest

from passive_ids.models.base import DetectionResult, Detector

# Pulled from ConfirmedFlowEntry.snapshot()'s top level and its nested
# "iat" sub-dict (features.welford.WelfordStats.snapshot()'s shape).
_TOP_LEVEL_KEYS = ("packet_count", "duration")
_IAT_KEYS = ("mean", "std", "coefficient_of_variation", "n")


def _extract_features(feature_vector: dict[str, Any]) -> list[float]:
    """Flattens the nested iat sub-dict into a fixed-order feature
    list. Missing keys (including a completely absent "iat" dict, if a
    non-flow vector is routed here by mistake) default to 0.0 rather
    than raising -- consistent with the DDoS detector's approach to
    the same problem.
    """
    iat = feature_vector.get("iat") or {}
    top = [float(feature_vector.get(k, 0.0)) for k in _TOP_LEVEL_KEYS]
    iat_vals = [float(iat.get(k, 0.0)) for k in _IAT_KEYS]
    return top + iat_vals


class C2BeaconingDetector(Detector):
    threat_class = "c2_beaconing"
    model_name = "isolation_forest"
    is_online_learner = False

    def __init__(self, **isolation_forest_kwargs: Any) -> None:
        self._model = IsolationForest(**isolation_forest_kwargs)
        self._fitted = False
        self.training_examples_seen = 0

    def fit(self, feature_vectors: list[dict[str, Any]]) -> None:
        """Batch (re)fit on presumed-mostly-normal flow vectors.
        Replaces any previous fit entirely -- IsolationForest has no
        meaningful "continue training" operation, unlike river's
        learn_one; this is what feedback/retrain.py's canary-deploy
        cycle is meant to call with fresh data periodically.
        """
        X = [_extract_features(fv) for fv in feature_vectors]
        self._model.fit(X)
        self._fitted = True
        self.training_examples_seen = len(feature_vectors)

    def score(self, feature_vector: dict[str, Any]) -> DetectionResult:
        x = _extract_features(feature_vector)

        if not self._fitted:
            return DetectionResult(
                threat_class=self.threat_class,
                model_name=self.model_name,
                raw_score=0.0,
                evidence={"status": "untrained", "training_examples_seen": 0},
            )

        try:
            decision = float(self._model.decision_function([x])[0])
            sklearn_label = int(self._model.predict([x])[0])
        except NotFittedError:
            # Shouldn't happen given the _fitted guard above, but this
            # is exactly the exception sklearn is documented to raise,
            # so it's handled explicitly rather than left to propagate
            # as an unhandled crash if that guard is ever bypassed.
            return DetectionResult(
                threat_class=self.threat_class,
                model_name=self.model_name,
                raw_score=0.0,
                evidence={"status": "untrained", "training_examples_seen": 0},
            )

        raw_score = -decision  # sign-flip: sklearn is negative-for-anomaly,
        # this project's convention is higher-raw_score-for-attack-like.

        return DetectionResult(
            threat_class=self.threat_class,
            model_name=self.model_name,
            raw_score=raw_score,
            evidence={
                "status": "trained",
                "training_examples_seen": self.training_examples_seen,
                "sklearn_decision_function": decision,
                "sklearn_is_anomaly": sklearn_label == -1,
                "note": "raw_score is NOT a calibrated 0-1 probability; see module docstring",
            },
        )
