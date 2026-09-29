"""DDoS detector: online Hoeffding Tree over per-source aggregate features.

Consumes exactly the tumbling-window feature vectors that
flow_state.source_aggregate + emission.triggers.TumblingWindowTrigger
already produce -- packet_count, byte_count, unique_dest_ips_est,
unique_dest_ports_est, syn_ratio, protocol_entropy. No new feature
engineering happens here; this model's entire job is learning a
decision boundary over data that already exists and is already
flowing through pipeline.py's pending_feature_vectors.

Genuinely online: update() calls river's learn_one() whenever a label
becomes available -- eventually from the SOC verdict feedback loop
(verdicts/ + feedback/), no separate training script, no batch
retrain-and-swap cycle. This matches models.base.Detector's
is_online_learner=True contract.

Honest about cold start, verified directly rather than assumed:
river's HoeffdingTreeClassifier.predict_proba_one() returns an empty
dict before any training -- not a default 0.5, not an error. With a
genuinely small number of examples it can also legitimately return an
even split for every input regardless of that input's values --
checked directly: with just 2 training examples (one clearly
attack-like, one clearly benign), it returns {0: 0.5, 1: 0.5} for
both.

A related finding worth carrying forward operationally, checked
directly rather than discovered the hard way in production: this
default configuration (leaf_prediction="nba", naive-Bayes at the
leaf) separates classes well given training data with realistic
natural variance -- but degenerates to that same even 50/50 split,
for every input, if fed the exact same fixed feature vector repeated
many times per class with zero internal variance. Verified: 1000
zero-variance training examples still produced n_nodes=1 and a 50/50
split; the same class separation with realistic per-example jitter
converged almost immediately. Whatever eventually generates training
examples for this model (feedback/retrain.py, or replayed/synthetic
data used to bootstrap it) needs to preserve realistic variance, not
just realistic class balance, or the model can look like it isn't
learning when the real issue is degenerate input.

This model ships with zero training examples. Every score it produces
today reflects that cold-start state honestly. No synthetic/bootstrap
training data is baked in here, for the same reason intel/ja4_lookup.py
ships with no fabricated threat intel: pretending to have signal that
doesn't exist yet is worse than admitting it's absent.
"""
from __future__ import annotations

from typing import Any

from river import tree

from passive_ids.models.base import DetectionResult, Detector

# Must match exactly what SourceAggregateEntry.snapshot() produces --
# see flow_state/source_aggregate.py.
FEATURE_KEYS = (
    "packet_count",
    "byte_count",
    "unique_dest_ips_est",
    "unique_dest_ports_est",
    "syn_ratio",
    "protocol_entropy",
)

_ATTACK = 1
_BENIGN = 0


def _extract_features(feature_vector: dict[str, Any]) -> dict[str, float]:
    """Pull just this model's inputs out of a (possibly larger) feature
    vector, coercing to float. Missing keys default to 0.0 rather than
    raising -- a feature vector missing one of these fields (e.g. an
    older schema version, or a non-source-aggregate vector routed here
    by mistake) shouldn't crash scoring, just score with less
    information than usual.
    """
    return {key: float(feature_vector.get(key, 0.0)) for key in FEATURE_KEYS}


class DDoSHoeffdingDetector(Detector):
    threat_class = "ddos"
    model_name = "hoeffding_tree"
    is_online_learner = True

    def __init__(self) -> None:
        self._tree = tree.HoeffdingTreeClassifier()
        self.training_examples_seen = 0

    def score(self, feature_vector: dict[str, Any]) -> DetectionResult:
        x = _extract_features(feature_vector)
        proba = self._tree.predict_proba_one(x)

        if not proba:
            raw_score = 0.0
            evidence = {"status": "untrained", "training_examples_seen": 0}
        else:
            raw_score = proba.get(_ATTACK, 0.0)
            evidence = {
                "status": "trained",
                "training_examples_seen": self.training_examples_seen,
                "class_probabilities": {str(k): v for k, v in proba.items()},
            }

        return DetectionResult(
            threat_class=self.threat_class,
            model_name=self.model_name,
            raw_score=raw_score,
            evidence=evidence,
        )

    def update(self, feature_vector: dict[str, Any], label: int | None = None) -> None:
        if label is None:
            return  # no label available yet -- nothing to learn from
        x = _extract_features(feature_vector)
        y = _ATTACK if label else _BENIGN
        self._tree.learn_one(x, y)
        self.training_examples_seen += 1
