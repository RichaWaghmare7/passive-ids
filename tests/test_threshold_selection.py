"""Tests for calibration/threshold_selection.py.

Uses the exact objective names committed in config/thresholds.yaml,
and includes a deliberately-overlapping-classes case to test the
"target precision unreachable, return None" path -- not just the easy
perfectly-separable case.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from passive_ids.calibration.threshold_selection import fit_class_thresholds


def _perfectly_separable():
    random.seed(1)
    y_true = [0] * 100 + [1] * 100
    y_scores = [random.uniform(0.0, 0.4) for _ in range(100)] + [
        random.uniform(0.6, 1.0) for _ in range(100)
    ]
    return y_true, y_scores


def _heavily_overlapping():
    random.seed(2)
    y_true = [0] * 100 + [1] * 100
    # Identical distribution for both classes -- genuinely zero signal,
    # not just overlapping-but-shifted ranges (which can still leave a
    # thin high-precision sliver at the extreme tail, as the first
    # version of this fixture did -- caught by the test itself).
    y_scores = [random.uniform(0.0, 1.0) for _ in range(200)]
    return y_true, y_scores


def test_f_beta_objective_on_separable_data_achieves_near_perfect_tiers():
    y_true, y_scores = _perfectly_separable()
    result = fit_class_thresholds(y_true, y_scores, {"objective": "f_beta", "beta": 2.0})

    assert result.medium.precision > 0.95
    assert result.medium.recall > 0.95
    assert result.high.threshold is not None
    assert result.average_precision > 0.95


def test_min_precision_objective_respects_target():
    y_true, y_scores = _perfectly_separable()
    result = fit_class_thresholds(
        y_true, y_scores, {"objective": "min_precision", "min_precision": 0.9}
    )
    assert result.medium.precision >= 0.9


def test_f1_objective_balances_precision_and_recall():
    y_true, y_scores = _perfectly_separable()
    result = fit_class_thresholds(y_true, y_scores, {"objective": "f1"})
    assert result.medium.precision > 0.9
    assert result.medium.recall > 0.9


def test_f_beta_or_min_precision_uses_min_precision_when_achievable():
    y_true, y_scores = _perfectly_separable()
    result = fit_class_thresholds(
        y_true, y_scores,
        {"objective": "f_beta_or_min_precision", "min_precision": 0.9, "beta": 2.0},
    )
    assert result.medium.precision >= 0.9


def test_f_beta_or_min_precision_falls_back_when_target_unreachable():
    """The specific fallback path: min_precision=0.95 is not achievable
    on heavily-overlapping data, so this must fall back to f_beta
    rather than returning an unusable None threshold."""
    y_true, y_scores = _heavily_overlapping()
    result = fit_class_thresholds(
        y_true, y_scores,
        {"objective": "f_beta_or_min_precision", "min_precision": 0.95, "beta": 2.0},
    )
    assert result.medium.threshold is not None  # fallback produced a usable threshold


def test_high_tier_is_none_when_precision_095_unreachable():
    y_true, y_scores = _heavily_overlapping()
    result = fit_class_thresholds(y_true, y_scores, {"objective": "f1"})
    assert result.high.threshold is None  # honestly unreachable, not a guessed number


def test_unknown_objective_raises_clear_error():
    import pytest
    y_true, y_scores = _perfectly_separable()
    with pytest.raises(ValueError, match="Unknown objective"):
        fit_class_thresholds(y_true, y_scores, {"objective": "not_a_real_objective"})


def test_matches_all_objective_names_from_thresholds_yaml():
    """Every objective string actually used in config/thresholds.yaml
    must be accepted without raising."""
    y_true, y_scores = _perfectly_separable()
    for objective in ("f_beta", "min_precision", "f1", "f_beta_or_min_precision"):
        result = fit_class_thresholds(
            y_true, y_scores, {"objective": objective, "beta": 2.0, "min_precision": 0.9}
        )
        assert result.average_precision > 0.0  # ran without raising
