"""Canary deployment: validates a retrained candidate against the
currently-serving (incumbent) model before it replaces anything.

Never replaces a live model just because retrain_* succeeded --
producing SOME retrained model is not the same as that model being
better. This module's job is the comparison; the actual swap (e.g.
pipeline.ddos_model = new_model) is left to the caller, since this
project has no model-registry/hot-swap mechanism to perform that
assignment safely on a live pipeline.

Uses average_precision_score, not raw accuracy or a single threshold
comparison -- same reasoning as calibration/threshold_selection.py:
this project's threat classes are realistically imbalanced, and
average precision only scores ranking quality among positives, which
is what actually matters for comparing two detectors.

min_improvement is a deliberate bar, not just "candidate >=
incumbent": a candidate that's only marginally ahead on a modest
validation set may just be noise, not a real improvement. The caller
decides how much improvement justifies an operational deploy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from sklearn.metrics import average_precision_score


class _Scoreable(Protocol):
    def score(self, feature_vector: dict[str, Any]) -> Any: ...


@dataclass
class CanaryResult:
    candidate_average_precision: float
    incumbent_average_precision: float
    improvement: float
    should_deploy: bool


def evaluate_canary(
    candidate: _Scoreable,
    incumbent: _Scoreable,
    validation_feature_vectors: list[dict[str, Any]],
    validation_labels: list[int],
    min_improvement: float = 0.0,
) -> CanaryResult:
    """Scores both models on the SAME held-out validation set --
    never the data the candidate was just retrained on, which would
    trivially favor it. Caller is responsible for that separation
    (e.g. a held-out split of build_training_data's output that never
    reaches retrain_*).
    """
    candidate_scores = [candidate.score(fv).raw_score for fv in validation_feature_vectors]
    incumbent_scores = [incumbent.score(fv).raw_score for fv in validation_feature_vectors]

    candidate_ap = float(average_precision_score(validation_labels, candidate_scores))
    incumbent_ap = float(average_precision_score(validation_labels, incumbent_scores))
    improvement = candidate_ap - incumbent_ap

    return CanaryResult(
        candidate_average_precision=candidate_ap,
        incumbent_average_precision=incumbent_ap,
        improvement=improvement,
        should_deploy=improvement >= min_improvement,
    )
