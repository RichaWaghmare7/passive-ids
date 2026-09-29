"""Tests for models/exfiltration.py."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from passive_ids.models.exfiltration import ExfiltrationDetector, _extract_features


def _flow_vector(byte_ratio_out, packet_ratio_out, duration, byte_count, iat_mean=5.0, iat_cv=0.8):
    return {
        "byte_ratio_out": byte_ratio_out,
        "packet_ratio_out": packet_ratio_out,
        "duration": duration,
        "byte_count": byte_count,
        "iat": {"mean": iat_mean, "coefficient_of_variation": iat_cv},
    }


def _exfil_like():
    # Heavily outbound, large volume, long sustained duration
    return _flow_vector(byte_ratio_out=0.95, packet_ratio_out=0.9, duration=3600.0, byte_count=5_000_000)


def _normal_like(rng):
    # Roughly balanced request/response, short-ish, modest volume
    return _flow_vector(
        byte_ratio_out=rng.uniform(0.3, 0.6),
        packet_ratio_out=rng.uniform(0.3, 0.6),
        duration=rng.uniform(1, 60),
        byte_count=rng.uniform(1000, 50000),
    )


def test_cold_start_returns_zero_and_untrained_status():
    detector = ExfiltrationDetector(random_state=42)
    result = detector.score(_exfil_like())

    assert result.raw_score == 0.0
    assert result.evidence["status"] == "untrained"
    assert result.threat_class == "exfiltration"


def test_fit_then_score_flags_exfil_like_above_normal():
    rng = random.Random(1)
    normal_population = [_normal_like(rng) for _ in range(200)]

    detector = ExfiltrationDetector(random_state=42, contamination=0.05)
    detector.fit(normal_population)

    exfil_result = detector.score(_exfil_like())
    normal_result = detector.score(_normal_like(random.Random(99)))

    assert exfil_result.evidence["status"] == "trained"
    assert exfil_result.raw_score > normal_result.raw_score
    assert exfil_result.evidence["sklearn_is_anomaly"] is True


def test_byte_ratio_out_surfaced_in_evidence():
    rng = random.Random(2)
    detector = ExfiltrationDetector(random_state=42)
    detector.fit([_normal_like(rng) for _ in range(100)])

    result = detector.score(_exfil_like())
    assert result.evidence["byte_ratio_out"] == 0.95


def test_missing_ratio_keys_default_to_balanced_not_zero():
    """0.5 (balanced), not 0.0 -- 0.0 would falsely read as
    'all inbound' for a flow with genuinely no directional data."""
    x = _extract_features({"duration": 10.0, "byte_count": 500.0})
    assert x[0] == 0.5  # byte_ratio_out
    assert x[1] == 0.5  # packet_ratio_out


def test_missing_iat_defaults_gracefully():
    detector = ExfiltrationDetector(random_state=42)
    detector.fit([_normal_like(random.Random(3)) for _ in range(50)])
    result = detector.score({"byte_ratio_out": 0.9, "packet_ratio_out": 0.9})  # no "iat" at all
    assert result.evidence["status"] == "trained"


def test_refit_replaces_not_accumulates():
    detector = ExfiltrationDetector(random_state=42)
    rng = random.Random(4)
    detector.fit([_normal_like(rng) for _ in range(50)])
    detector.fit([_normal_like(rng) for _ in range(75)])
    assert detector.training_examples_seen == 75


def test_update_is_a_noop_inherited_from_base():
    detector = ExfiltrationDetector(random_state=42)
    detector.update(_exfil_like(), label=1)
    assert detector.training_examples_seen == 0
    assert detector.score(_exfil_like()).evidence["status"] == "untrained"
