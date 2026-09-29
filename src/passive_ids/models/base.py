"""Common interface for the six threat-specific detectors.

Every model consumes a feature vector already computed by the
pre-model pipeline (features/) and returns a raw, model-native score
-- calibration to a common 0-1 scale happens later, in calibration/,
never inside the model itself. Keeping that split is what lets a
Hoeffding Tree's probability and an Isolation Forest's anomaly score
sit in the same alert without one silently dominating the other.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DetectionResult:
    threat_class: str
    model_name: str
    raw_score: float
    # Populated by calibration/, left None until then.
    calibrated_confidence: float | None = None
    evidence: dict[str, Any] | None = None


class Detector(ABC):
    """Base class for all six detectors.

    Subclasses fall into two learning modes, and it matters which:

    - Online / incremental (ddos_hoeffding, the Welford+Markov feature
      layers): `update()` is called on every relevant instance and the
      model is immediately usable -- no separate training job.
    - Batch-retrained (c2's Isolation Forest, dga's LightGBM,
      malware_encrypted, port_scan, exfiltration's Isolation Forest):
      `update()` is a no-op; retraining happens offline via
      feedback/retrain.py and the new model is swapped in through
      canary deployment, never mutated in place while serving traffic.
    """

    threat_class: str
    model_name: str
    is_online_learner: bool = False

    @abstractmethod
    def score(self, feature_vector: dict[str, Any]) -> DetectionResult:
        """Score one feature vector. Must be safe to call at line rate."""
        raise NotImplementedError

    def update(self, feature_vector: dict[str, Any], label: int | None = None) -> None:
        """Incremental learning hook. No-op for batch-retrained models."""
        return None
