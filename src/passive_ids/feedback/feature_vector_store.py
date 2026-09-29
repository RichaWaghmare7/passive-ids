"""Feature-vector store: the join key between verdicts and training data.

Real, necessary gap this fills, surfaced while building feedback/:
verdicts.store.VerdictStore.training_pairs() returns (flow_id, label)
pairs, but a label alone isn't a training example -- retraining needs
the actual feature vector (X) that was scored at alert time, paired
with the confirmed label (y). Nothing in this project persisted that
feature vector anywhere until now; pipeline.py's self.detections held
each DetectionResult (the model's output) but never its input.

Keyed by (flow_id, feature_vector_version) together, not flow_id
alone -- feature-engineering code changes over time (see
verdicts/schema.py's Verdict.feature_vector_version), and silently
joining a verdict to a feature vector produced by DIFFERENT
feature-engineering code would corrupt the training set built from it.

In-memory and unbounded for now -- like verdicts.store.VerdictStore,
the real backing store (a database, a versioned feature store) is a
deployment decision, not something to invent here. An unbounded
in-memory dict never evicts -- a real, disclosed limitation for a
long-running deployment, acceptable at this project's current stage.
"""
from __future__ import annotations

from typing import Any


class FeatureVectorStore:
    def __init__(self) -> None:
        self._store: dict[tuple[str, str], dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self._store)

    def record(self, flow_id: str, feature_vector_version: str, feature_vector: dict[str, Any]) -> None:
        self._store[(flow_id, feature_vector_version)] = feature_vector

    def get(self, flow_id: str, feature_vector_version: str) -> dict[str, Any] | None:
        return self._store.get((flow_id, feature_vector_version))
