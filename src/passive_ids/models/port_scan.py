"""Port scan detector: Logistic Regression or shallow LightGBM over
source-aggregate fan-out features.

Consumes source-aggregate-level vectors -- same input shape as
models.ddos_hoeffding.DDoSHoeffdingDetector (both read from
flow_state.source_aggregate.SourceAggregateEntry.snapshot() via the
tumbling-window trigger), but a different feature subset: fan-out
(unique_dest_ports_est, dest_port_entropy) rather than volume
(packet_count, byte_count) is the primary signal here.

dest_port_entropy is new as of this model -- added to
SourceAggregateEntry specifically because port-scan detection needs
the distribution shape, not just cardinality: a source hitting 50
distinct ports once each (true horizontal scan) and one hitting 2
ports at very uneven rates can have deceptively similar
unique_dest_ports_est, but very different entropy.

Two backends, both genuinely supported rather than picking one and
dropping the other, since the original design called this out as an
explicit choice ("LightGBM shallow / Logistic Regression"):

  - "logistic_regression" (default): simpler, more interpretable
    coefficients, a reasonable match for what's a fairly low
    -dimensional, close-to-linearly-separable signal (high port
    entropy + high SYN ratio + low protocol entropy is a strong,
    fairly direct combination). Wrapped in a Pipeline with
    StandardScaler -- unlike the tree-based models elsewhere in this
    project, logistic regression is sensitive to feature scale, and
    packet_count/byte_count sit on a wildly different scale than the
    0-2ish entropy and 0-1 ratio features.
  - "lightgbm": shallow (low max_depth, few leaves by default) gradient
    -boosted trees, for when the decision boundary turns out not to be
    as linear as the default backend assumes.

Batch-retrained like models.c2_beaconing.C2BeaconingDetector and
models.dga_cascade.DGACascadeDetector -- update() is the inherited
no-op; fit() is the real training entry point, called with labeled
data whenever feedback/retrain.py exists to provide it. Ships unfit by
default, same discipline as every other model in this project: no
fabricated training data baked in.

Verified directly before writing: both backends raise the same
NotFittedError sklearn's own estimators do, before fit() -- including
through a Pipeline wrapper, and Pipeline forwards .classes_ from its
final step directly (pipe.classes_ works without reaching into
named_steps).
"""
from __future__ import annotations

from typing import Any, Literal

from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from passive_ids.models.base import DetectionResult, Detector

# Must match exactly what SourceAggregateEntry.snapshot() produces --
# see flow_state/source_aggregate.py. Fan-out features first, since
# they're the primary signal for this threat class; volume features
# still included since a scan and a volumetric flood aren't always
# clean opposites in practice.
FEATURE_KEYS = (
    "unique_dest_ports_est",
    "dest_port_entropy",
    "unique_dest_ips_est",
    "syn_ratio",
    "protocol_entropy",
    "packet_count",
)


def _extract_features(feature_vector: dict[str, Any]) -> list[float]:
    return [float(feature_vector.get(key, 0.0)) for key in FEATURE_KEYS]


def _build_model(backend: str, **kwargs: Any) -> Any:
    if backend == "logistic_regression":
        return Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(**kwargs))]
        )
    if backend == "lightgbm":
        import lightgbm as lgb

        kwargs.setdefault("max_depth", 4)  # "shallow" -- see module docstring
        kwargs.setdefault("num_leaves", 15)
        kwargs.setdefault("verbose", -1)
        return lgb.LGBMClassifier(**kwargs)
    raise ValueError(f"Unknown backend '{backend}'. Use 'logistic_regression' or 'lightgbm'.")


class PortScanDetector(Detector):
    threat_class = "port_scanning"
    model_name = "logreg_or_lightgbm"
    is_online_learner = False

    def __init__(
        self, backend: Literal["logistic_regression", "lightgbm"] = "logistic_regression", **kwargs: Any
    ) -> None:
        self.backend = backend
        self._model = _build_model(backend, **kwargs)
        self._fitted = False
        self.training_examples_seen = 0

    def fit(self, feature_vectors: list[dict[str, Any]], labels: list[int]) -> None:
        X = [_extract_features(fv) for fv in feature_vectors]
        self._model.fit(X, labels)
        self._fitted = True
        self.training_examples_seen = len(feature_vectors)

    def score(self, feature_vector: dict[str, Any]) -> DetectionResult:
        if not self._fitted:
            return DetectionResult(
                threat_class=self.threat_class,
                model_name=self.model_name,
                raw_score=0.0,
                evidence={"status": "untrained", "backend": self.backend},
            )

        x = _extract_features(feature_vector)
        try:
            proba_row = self._model.predict_proba([x])[0]
        except NotFittedError:
            return DetectionResult(
                threat_class=self.threat_class,
                model_name=self.model_name,
                raw_score=0.0,
                evidence={"status": "untrained", "backend": self.backend},
            )

        classes = list(self._model.classes_)
        positive_index = classes.index(1) if 1 in classes else classes.index(True)
        raw_score = float(proba_row[positive_index])

        return DetectionResult(
            threat_class=self.threat_class,
            model_name=self.model_name,
            raw_score=raw_score,
            evidence={
                "status": "trained",
                "backend": self.backend,
                "training_examples_seen": self.training_examples_seen,
            },
        )
