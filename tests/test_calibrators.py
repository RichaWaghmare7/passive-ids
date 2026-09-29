"""Tests for calibration/calibrators.py.

The registry test directly recreates the scenario that motivated
keying by threat_class instead of model_name: two detectors
(c2_beaconing, exfiltration) both report model_name="isolation_forest"
but must never share a calibration curve.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from passive_ids.calibration.calibrators import CalibratorRegistry, ScoreCalibrator


def test_cold_start_returns_none():
    cal = ScoreCalibrator()
    assert cal.calibrate(0.5) is None


def test_fit_then_calibrate_is_monotonic_and_bounded():
    cal = ScoreCalibrator()
    # Isolation-Forest-like raw scores: unbounded, roughly zero-centered
    raw_scores = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    labels =     [0,     0,    0,   0,   1,   1,   1]
    cal.fit(raw_scores, labels)

    low = cal.calibrate(-2.0)
    mid = cal.calibrate(0.0)
    high = cal.calibrate(2.0)

    assert 0.0 <= low <= mid <= high <= 1.0


def test_out_of_range_scores_clip_into_bounds():
    cal = ScoreCalibrator()
    cal.fit([-1.0, 0.0, 1.0], [0, 0, 1])
    assert cal.calibrate(-100.0) == 0.0
    assert cal.calibrate(100.0) == 1.0


def test_training_examples_seen_tracked():
    cal = ScoreCalibrator()
    cal.fit([0.1, 0.2, 0.3], [0, 1, 1])
    assert cal.training_examples_seen == 3


def test_registry_does_not_share_calibration_between_same_model_name_detectors():
    """The exact scenario that motivated this design: c2_beaconing and
    exfiltration both report model_name='isolation_forest' but their
    raw score distributions mean completely different things."""
    registry = CalibratorRegistry()

    # c2_beaconing: raw_score around 0.05-0.1 means "very anomalous" (tight IAT)
    registry.fit("c2_beaconing", [0.01, 0.02, 0.05, 0.08, 0.1], [0, 0, 1, 1, 1])
    # exfiltration: raw_score needs to be much higher (e.g. 1.0+) to mean "anomalous"
    registry.fit("exfiltration", [-1.0, -0.5, 0.5, 1.0, 2.0], [0, 0, 0, 1, 1])

    # A raw_score of 0.08 should read as high-confidence for c2_beaconing...
    c2_confidence = registry.calibrate("c2_beaconing", 0.08)
    # ...but the SAME raw value must not be interpreted the same way
    # for exfiltration, which has an entirely different fitted curve.
    exfil_confidence = registry.calibrate("exfiltration", 0.08)

    assert c2_confidence is not None
    assert exfil_confidence is not None
    assert c2_confidence != exfil_confidence


def test_registry_lazy_creation_unfit_class_returns_none():
    registry = CalibratorRegistry()
    assert registry.calibrate("never_fitted_class", 0.5) is None


def test_apply_fills_in_calibrated_confidence_on_frozen_result():
    from passive_ids.models.base import DetectionResult

    registry = CalibratorRegistry()
    registry.fit("ddos", [0.1, 0.5, 0.9], [0, 1, 1])

    result = DetectionResult(threat_class="ddos", model_name="hoeffding_tree", raw_score=0.9)
    calibrated = registry.apply(result)

    assert calibrated is not result  # new instance, original untouched (frozen)
    assert result.calibrated_confidence is None
    assert calibrated.calibrated_confidence is not None
    assert 0.0 <= calibrated.calibrated_confidence <= 1.0


def test_apply_leaves_result_unchanged_when_class_unfit():
    from passive_ids.models.base import DetectionResult

    registry = CalibratorRegistry()
    result = DetectionResult(threat_class="dga", model_name="markov_lightgbm_cascade", raw_score=0.7)
    calibrated = registry.apply(result)

    assert calibrated.calibrated_confidence is None
    assert calibrated == result
