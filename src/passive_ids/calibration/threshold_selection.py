"""Per-class threshold selection from precision-recall curves.

Implements the four objectives already committed in
config/thresholds.yaml ("f_beta", "min_precision", "f1",
"f_beta_or_min_precision") -- this module is what turns that file's
`null` placeholders into real numbers, once labeled score/label data
exists (public dataset first, then rolling SOC-verdict data via
feedback/, not built yet).

Every class gets its OWN objective because the cost of being wrong
differs by threat: missing a volumetric DDoS attack (favor recall,
f_beta with beta>1) is not the same mistake as flooding a SOC with
false port-scan alerts from legitimate vulnerability scanners (favor
precision). No shared threshold or shared objective across classes --
see docs/architecture.md.

Uses average_precision_score, not ROC-AUC, as the headline quality
metric. This project's threat classes are realistically imbalanced
(exfiltration/C2 positives are a small fraction of total traffic);
ROC-AUC is dominated by the resulting large true-negative count and
looks artificially good under that imbalance, while average precision
only scores ranking quality among the positives, which is what
actually matters for a security detector.

Two tiers are always produced, regardless of a class's own objective:
HIGH is always a min-precision-0.95 point (what "severity: HIGH" is
meant to mean consistently across every threat class in the alert
schema), MEDIUM is whatever that class's own configured objective
produces. Either can be None if the underlying data can't support it
(e.g. no threshold reaches 0.95 precision at all) -- callers must
handle None as "this tier isn't achievable with current data," not
crash or substitute a guessed number.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


@dataclass
class TierResult:
    threshold: float | None
    precision: float | None
    recall: float | None


@dataclass
class ThresholdResult:
    high: TierResult
    medium: TierResult
    average_precision: float


def _threshold_max_fbeta(
    precision: np.ndarray, recall: np.ndarray, thresholds: np.ndarray, beta: float
) -> TierResult:
    if len(thresholds) == 0:
        return TierResult(None, None, None)
    fbeta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-12)
    best_idx = int(np.argmax(fbeta[:-1]))  # last point has no matching threshold
    return TierResult(float(thresholds[best_idx]), float(precision[best_idx]), float(recall[best_idx]))


def _threshold_min_precision(
    precision: np.ndarray, recall: np.ndarray, thresholds: np.ndarray, min_precision: float
) -> TierResult:
    if len(thresholds) == 0:
        return TierResult(None, None, None)
    valid = precision[:-1] >= min_precision
    if not valid.any():
        return TierResult(None, None, None)  # unreachable with current data -- not an error
    candidate_recalls = recall[:-1][valid]
    best_idx = int(np.argmax(candidate_recalls))
    return TierResult(
        float(thresholds[valid][best_idx]),
        float(precision[:-1][valid][best_idx]),
        float(candidate_recalls[best_idx]),
    )


def fit_class_thresholds(
    y_true: list[int], y_scores: list[float], objective_config: dict[str, Any]
) -> ThresholdResult:
    """objective_config matches one class entry from
    config/thresholds.yaml: {"objective": ..., "beta": ..., "min_precision": ...}
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ap = float(average_precision_score(y_true, y_scores))

    high = _threshold_min_precision(precision, recall, thresholds, 0.95)

    objective = objective_config.get("objective", "f_beta")
    if objective == "f_beta":
        medium = _threshold_max_fbeta(precision, recall, thresholds, objective_config.get("beta", 1.0))
    elif objective == "min_precision":
        medium = _threshold_min_precision(
            precision, recall, thresholds, objective_config.get("min_precision", 0.9)
        )
    elif objective == "f1":
        medium = _threshold_max_fbeta(precision, recall, thresholds, beta=1.0)
    elif objective == "f_beta_or_min_precision":
        medium = _threshold_min_precision(
            precision, recall, thresholds, objective_config.get("min_precision", 0.9)
        )
        if medium.threshold is None:  # target precision unreachable -- fall back
            medium = _threshold_max_fbeta(
                precision, recall, thresholds, objective_config.get("beta", 2.0)
            )
    else:
        raise ValueError(
            f"Unknown objective '{objective}'. "
            "Use one of: f_beta, min_precision, f1, f_beta_or_min_precision"
        )

    return ThresholdResult(high=high, medium=medium, average_precision=ap)
