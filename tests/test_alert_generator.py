"""Tests for alerting/generator.py.

Severity ranking is tested directly against the enum's actual
declaration order (CRITICAL, HIGH, MEDIUM, LOW) to guard against the
specific mistake _SEVERITY_RANK exists to avoid: a naive max() over
that declaration order would silently treat LOW as "biggest."
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from passive_ids.alerting.generator import build_detection, generate_alert, _overall_severity
from passive_ids.alerting.schema import Detection, Severity
from passive_ids.flow_state.flow_key import FlowKey
from passive_ids.models.base import DetectionResult

_THRESHOLDS = {
    "ddos": {"high_threshold": 0.9, "medium_threshold": 0.6},
    "port_scanning": {"high_threshold": 0.9, "medium_threshold": 0.6},
    "encrypted_malware": {"high_threshold": 0.9, "medium_threshold": 0.6},
}


def _result(threat_class, raw_score, calibrated=None, evidence=None):
    return DetectionResult(
        threat_class=threat_class, model_name="test_model", raw_score=raw_score,
        calibrated_confidence=calibrated, evidence=evidence or {},
    )


# ---------------------------------------------------------------------------
# build_detection
# ---------------------------------------------------------------------------

def test_below_medium_threshold_produces_no_detection():
    result = _result("ddos", raw_score=0.3)
    assert build_detection(result, _THRESHOLDS) is None


def test_between_medium_and_high_gives_medium_severity():
    result = _result("ddos", raw_score=0.7, calibrated=0.7)
    detection = build_detection(result, _THRESHOLDS)
    assert detection.severity == Severity.MEDIUM


def test_above_high_gives_high_severity():
    result = _result("ddos", raw_score=0.95, calibrated=0.95)
    detection = build_detection(result, _THRESHOLDS)
    assert detection.severity == Severity.HIGH


def test_prefers_calibrated_confidence_over_raw_score():
    # raw_score alone would clear HIGH, but calibrated_confidence
    # (what should actually be used) only clears MEDIUM.
    result = _result("ddos", raw_score=0.95, calibrated=0.65)
    detection = build_detection(result, _THRESHOLDS)
    assert detection.confidence == 0.65
    assert detection.severity == Severity.MEDIUM


def test_uncalibrated_result_discloses_status_in_evidence():
    result = _result("ddos", raw_score=0.95, calibrated=None)
    detection = build_detection(result, _THRESHOLDS)
    assert "uncalibrated" in detection.evidence["calibration_status"]


def test_unknown_threat_class_with_no_configured_threshold_produces_no_detection():
    result = _result("some_future_class", raw_score=0.99)
    assert build_detection(result, _THRESHOLDS) is None


# ---------------------------------------------------------------------------
# _overall_severity -- the rank-order correctness this module exists to get right
# ---------------------------------------------------------------------------

def _detection(threat_class, severity, evidence=None):
    return Detection(threat_class=threat_class, model="m", confidence=0.9, severity=severity, evidence=evidence or {})


def test_single_medium_detection_stays_medium():
    assert _overall_severity([_detection("ddos", Severity.MEDIUM)]) == Severity.MEDIUM


def test_single_high_detection_stays_high_not_escalated():
    assert _overall_severity([_detection("ddos", Severity.HIGH)]) == Severity.HIGH


def test_two_distinct_high_classes_escalate_to_critical():
    detections = [_detection("ddos", Severity.HIGH), _detection("port_scanning", Severity.HIGH)]
    assert _overall_severity(detections) == Severity.CRITICAL


def test_one_high_one_medium_different_classes_does_not_escalate():
    """Only ONE class at HIGH -- compounding requires 2+."""
    detections = [_detection("ddos", Severity.HIGH), _detection("port_scanning", Severity.MEDIUM)]
    assert _overall_severity(detections) == Severity.HIGH


def test_ja4_known_bad_escalates_to_critical_even_at_medium_severity():
    detections = [_detection("encrypted_malware", Severity.MEDIUM, evidence={"ja4_known_bad": True})]
    assert _overall_severity(detections) == Severity.CRITICAL


def test_severity_rank_matches_intuitive_ordering_not_enum_declaration_order():
    """Sanity check on the rank map itself: CRITICAL > HIGH > MEDIUM >
    LOW numerically, regardless of how the Severity enum happens to
    be declared in schema.py."""
    from passive_ids.alerting.generator import _SEVERITY_RANK
    assert _SEVERITY_RANK[Severity.CRITICAL] > _SEVERITY_RANK[Severity.HIGH]
    assert _SEVERITY_RANK[Severity.HIGH] > _SEVERITY_RANK[Severity.MEDIUM]
    assert _SEVERITY_RANK[Severity.MEDIUM] > _SEVERITY_RANK[Severity.LOW]


# ---------------------------------------------------------------------------
# generate_alert
# ---------------------------------------------------------------------------

def test_all_benign_group_produces_no_alert():
    results = [_result("ddos", raw_score=0.1), _result("port_scanning", raw_score=0.05)]
    alert = generate_alert(results, _THRESHOLDS, context={"source_ip": "10.0.0.5"})
    assert alert is None


def test_multi_label_group_produces_multiple_detections_in_one_alert():
    results = [
        _result("ddos", raw_score=0.95, calibrated=0.95),
        _result("port_scanning", raw_score=0.7, calibrated=0.7),
    ]
    alert = generate_alert(results, _THRESHOLDS, context={"source_ip": "10.0.0.5"})
    assert len(alert.detections) == 2
    assert {d.threat_class for d in alert.detections} == {"ddos", "port_scanning"}
    assert alert.summary.threats_detected_count == 2


def test_source_aggregate_context_has_no_destination_or_flow_id():
    results = [_result("ddos", raw_score=0.95, calibrated=0.95)]
    alert = generate_alert(results, _THRESHOLDS, context={"source_ip": "10.0.0.5"})
    assert alert.source_ip == "10.0.0.5"
    assert alert.destination_ip is None
    assert alert.flow_id is None


def test_flow_key_context_populates_source_destination_and_flow_id():
    key = FlowKey(src_ip="10.0.0.5", dst_ip="10.0.0.9", src_port=5000, dst_port=443, protocol="TCP")
    results = [_result("encrypted_malware", raw_score=0.95, calibrated=0.95)]
    alert = generate_alert(results, _THRESHOLDS, context={"flow_key": key})
    assert alert.source_ip == "10.0.0.5"
    assert alert.destination_ip == "10.0.0.9"
    assert alert.flow_id is not None


def test_ja4_known_bad_produces_critical_alert():
    results = [_result("encrypted_malware", raw_score=0.7, calibrated=0.7, evidence={"ja4_known_bad": True})]
    alert = generate_alert(results, _THRESHOLDS, context={"src_ip": "10.0.0.5", "dst_ip": "198.51.100.9"})
    assert alert.summary.overall_severity == Severity.CRITICAL


def test_noisy_or_and_max_fusion_produce_different_scores_for_multiple_detections():
    results = [
        _result("ddos", raw_score=0.9, calibrated=0.9),
        _result("port_scanning", raw_score=0.8, calibrated=0.8),
    ]
    noisy_or_alert = generate_alert(results, _THRESHOLDS, {"source_ip": "10.0.0.5"}, fusion_method="noisy_or")
    max_alert = generate_alert(results, _THRESHOLDS, {"source_ip": "10.0.0.5"}, fusion_method="max")

    assert max_alert.summary.overall_score == 0.9
    assert noisy_or_alert.summary.overall_score > 0.9  # P(A or B) > either alone
    assert noisy_or_alert.summary.overall_score_method == "noisy_or"
    assert max_alert.summary.overall_score_method == "max"


def test_unknown_fusion_method_raises_clear_error():
    import pytest
    results = [_result("ddos", raw_score=0.95, calibrated=0.95)]
    with pytest.raises(ValueError, match="Unknown fusion_method"):
        generate_alert(results, _THRESHOLDS, {"source_ip": "10.0.0.5"}, fusion_method="average")


def test_context_missing_source_ip_raises_clear_error():
    import pytest
    results = [_result("ddos", raw_score=0.95, calibrated=0.95)]
    with pytest.raises(ValueError, match="source_ip"):
        generate_alert(results, _THRESHOLDS, context={"domain": "evil.tk"})
