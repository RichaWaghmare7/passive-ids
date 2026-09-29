"""Tests for models/port_scan.py.

Both backends (logistic_regression, lightgbm) are tested against the
same synthetic scan-vs-normal separation, since the design explicitly
supports both as a genuine choice rather than picking one.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from passive_ids.models.port_scan import PortScanDetector, _extract_features, FEATURE_KEYS


def _scan_like(rng):
    return {
        "unique_dest_ports_est": rng.uniform(40, 60),
        "dest_port_entropy": rng.uniform(3.5, 4.5),  # high -- ports hit ~evenly
        "unique_dest_ips_est": rng.uniform(1, 3),
        "syn_ratio": rng.uniform(0.9, 1.0),
        "protocol_entropy": rng.uniform(0.0, 0.1),
        "packet_count": rng.uniform(40, 70),
    }


def _normal_like(rng):
    return {
        "unique_dest_ports_est": rng.uniform(1, 3),
        "dest_port_entropy": rng.uniform(0.0, 0.3),  # low -- same couple ports repeatedly
        "unique_dest_ips_est": rng.uniform(1, 2),
        "syn_ratio": rng.uniform(0.0, 0.1),
        "protocol_entropy": rng.uniform(0.8, 1.5),
        "packet_count": rng.uniform(10, 30),
    }


def test_cold_start_returns_zero_and_untrained_status():
    for backend in ("logistic_regression", "lightgbm"):
        detector = PortScanDetector(backend=backend)
        result = detector.score(_scan_like(random.Random(1)))
        assert result.raw_score == 0.0
        assert result.evidence["status"] == "untrained"
        assert result.threat_class == "port_scanning"


def test_invalid_backend_raises_clear_error():
    import pytest
    with pytest.raises(ValueError, match="Unknown backend"):
        PortScanDetector(backend="not_a_real_backend")


def test_logistic_regression_backend_separates_scan_from_normal():
    rng = random.Random(1)
    scan_examples = [_scan_like(rng) for _ in range(100)]
    normal_examples = [_normal_like(rng) for _ in range(100)]

    detector = PortScanDetector(backend="logistic_regression", random_state=42)
    detector.fit(scan_examples + normal_examples, [1] * 100 + [0] * 100)

    scan_score = detector.score(_scan_like(random.Random(2))).raw_score
    normal_score = detector.score(_normal_like(random.Random(3))).raw_score

    assert scan_score > 0.8
    assert normal_score < 0.2


def test_lightgbm_backend_separates_scan_from_normal():
    rng = random.Random(4)
    scan_examples = [_scan_like(rng) for _ in range(100)]
    normal_examples = [_normal_like(rng) for _ in range(100)]

    detector = PortScanDetector(backend="lightgbm", random_state=42, min_child_samples=1)
    detector.fit(scan_examples + normal_examples, [1] * 100 + [0] * 100)

    scan_score = detector.score(_scan_like(random.Random(5))).raw_score
    normal_score = detector.score(_normal_like(random.Random(6))).raw_score

    assert scan_score > 0.8
    assert normal_score < 0.2


def test_training_examples_seen_tracked():
    detector = PortScanDetector(random_state=42)
    rng = random.Random(7)
    examples = [_scan_like(rng) for _ in range(30)] + [_normal_like(rng) for _ in range(30)]
    detector.fit(examples, [1] * 30 + [0] * 30)
    assert detector.training_examples_seen == 60


def test_update_is_a_noop_inherited_from_base():
    detector = PortScanDetector(random_state=42)
    detector.update(_scan_like(random.Random(1)), label=1)  # must not raise
    assert detector.score(_scan_like(random.Random(1))).evidence["status"] == "untrained"


def test_extract_features_order_and_defaults():
    x = _extract_features({"unique_dest_ports_est": 42.0})
    assert x[FEATURE_KEYS.index("unique_dest_ports_est")] == 42.0
    assert x[FEATURE_KEYS.index("syn_ratio")] == 0.0  # missing key defaults


def test_extract_features_ignores_extra_pipeline_metadata_keys():
    x = _extract_features({
        "unique_dest_ports_est": 5.0,
        "emission_reason": "tumbling_window",
        "aggregate_type": "source",
        "source_ip": "10.0.0.5",
    })
    assert len(x) == len(FEATURE_KEYS)  # extra keys don't leak into the feature vector
