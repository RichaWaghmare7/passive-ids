"""Data exfiltration detector: Isolation Forest over directional
byte/packet ratio asymmetry and flow duration.

Consumes flow-level feature vectors -- same input shape as
models.c2_beaconing.C2BeaconingDetector (both read from
flow_state.confirmed_table.ConfirmedFlowEntry.snapshot() via
FlowTimeoutTrigger), different feature subset: byte_ratio_out /
packet_ratio_out (outbound-heavy asymmetry) and duration (sustained
transfer, not a burst) rather than IAT regularity.

byte_ratio_out and packet_ratio_out only exist, and only mean anything,
because of a fix made specifically to support this model: before
FlowKey.canonical() existed, a connection's forward and reverse packets
landed in two separate, unrelated ConfirmedFlowEntry objects (verified
directly -- a 3+3 packet exchange produced 2 entries instead of 1), so
there was no single place to compare outbound against inbound volume
for the same connection. See flow_state/flow_key.py and
flow_state/confirmed_table.py for that fix; this model is the reason
it needed to happen now rather than later.

Same batch-retrained discipline as models.c2_beaconing.C2BeaconingDetector:
Isolation Forest, unsupervised, ships unfit, fit() replaces rather than
accumulates, decision_function() is negative-for-anomaly and gets
sign-flipped to match this project's higher-raw_score-for-attack-like
convention, and the result is NOT a calibrated 0-1 probability -- all
verified/established when that model was built; nothing new to
re-verify here since it's the same underlying library and the same
sign convention.
"""
from __future__ import annotations

from typing import Any

from sklearn.ensemble import IsolationForest
from sklearn.exceptions import NotFittedError

from passive_ids.models.base import DetectionResult, Detector

# Pulled from ConfirmedFlowEntry.snapshot()'s top level and its nested
# "iat" sub-dict.
_TOP_LEVEL_KEYS = ("byte_ratio_out", "packet_ratio_out", "duration", "byte_count")
_IAT_KEYS = ("mean", "coefficient_of_variation")


def _extract_features(feature_vector: dict[str, Any]) -> list[float]:
    """Missing keys default to 0.0, except byte_ratio_out/
    packet_ratio_out which default to 0.5 (balanced) -- consistent
    with ConfirmedFlowEntry.snapshot()'s own default for a flow with
    no directional data, rather than silently reading as "all inbound."
    """
    top = [
        float(feature_vector.get(k, 0.5 if k.endswith("ratio_out") else 0.0))
        for k in _TOP_LEVEL_KEYS
    ]
    iat = feature_vector.get("iat") or {}
    iat_vals = [float(iat.get(k, 0.0)) for k in _IAT_KEYS]
    return top + iat_vals


class ExfiltrationDetector(Detector):
    threat_class = "exfiltration"
    model_name = "isolation_forest"
    is_online_learner = False

    def __init__(self, **isolation_forest_kwargs: Any) -> None:
        self._model = IsolationForest(**isolation_forest_kwargs)
        self._fitted = False
        self.training_examples_seen = 0

    def fit(self, feature_vectors: list[dict[str, Any]]) -> None:
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
            return DetectionResult(
                threat_class=self.threat_class,
                model_name=self.model_name,
                raw_score=0.0,
                evidence={"status": "untrained", "training_examples_seen": 0},
            )

        raw_score = -decision  # sign-flip, see module docstring

        return DetectionResult(
            threat_class=self.threat_class,
            model_name=self.model_name,
            raw_score=raw_score,
            evidence={
                "status": "trained",
                "training_examples_seen": self.training_examples_seen,
                "sklearn_decision_function": decision,
                "sklearn_is_anomaly": sklearn_label == -1,
                "byte_ratio_out": feature_vector.get("byte_ratio_out"),
                "note": "raw_score is NOT a calibrated 0-1 probability; see module docstring",
            },
        )
