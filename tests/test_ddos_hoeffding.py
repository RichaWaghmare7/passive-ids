"""Tests for models/ddos_hoeffding.py.

Cold-start and small-sample behavior are checked against exactly what
was verified directly against river before writing the model (empty
dict pre-training; even 50/50 split with too few examples) -- these
are documented, expected states, not things to work around.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from passive_ids.models.ddos_hoeffding import DDoSHoeffdingDetector, _extract_features


def _attack_like():
    return {
        "packet_count": 5000.0, "byte_count": 300000.0,
        "unique_dest_ips_est": 3.0, "unique_dest_ports_est": 1.0,
        "syn_ratio": 0.98, "protocol_entropy": 0.0,
    }


def _benign_like():
    return {
        "packet_count": 12.0, "byte_count": 4000.0,
        "unique_dest_ips_est": 1.0, "unique_dest_ports_est": 1.0,
        "syn_ratio": 0.0, "protocol_entropy": 1.4,
    }


def test_cold_start_returns_zero_score_and_untrained_status():
    detector = DDoSHoeffdingDetector()
    result = detector.score(_attack_like())

    assert result.raw_score == 0.0
    assert result.evidence["status"] == "untrained"
    assert result.threat_class == "ddos"
    assert result.model_name == "hoeffding_tree"


def test_update_with_none_label_is_a_noop():
    detector = DDoSHoeffdingDetector()
    detector.update(_attack_like(), label=None)

    assert detector.training_examples_seen == 0
    assert detector.score(_attack_like()).evidence["status"] == "untrained"


def test_single_labeled_example_produces_high_confidence_for_that_exact_input():
    """Matches river's verified behavior: after one training example,
    scoring that same input again returns near-certainty (the tree has
    only ever seen this one pattern)."""
    detector = DDoSHoeffdingDetector()
    detector.update(_attack_like(), label=1)

    result = detector.score(_attack_like())
    assert result.evidence["status"] == "trained"
    assert result.raw_score > 0.9


def test_two_examples_gives_even_split_not_a_bug():
    """The specific, verified-before-writing behavior: with only 2
    examples (one attack, one benign), the tree hasn't seen enough to
    justify a split yet, so it gives 50/50 for BOTH inputs -- checked
    directly against river, not assumed."""
    detector = DDoSHoeffdingDetector()
    detector.update(_attack_like(), label=1)
    detector.update(_benign_like(), label=0)

    attack_score = detector.score(_attack_like()).raw_score
    benign_score = detector.score(_benign_like()).raw_score

    assert abs(attack_score - 0.5) < 1e-9
    assert abs(benign_score - 0.5) < 1e-9


def test_sufficient_training_separates_attack_from_benign():
    """With realistic per-example variance (NOT exact repeated fixed
    points -- see the next test for why that distinction matters),
    river's default naive-Bayes leaf estimator separates the classes
    sharply, and does so quickly. Verified directly: this does NOT
    require a structural tree split (n_nodes can stay at 1) -- river's
    default leaf_prediction='nba' computes input-dependent probabilities
    from per-feature distributions within a single leaf.
    """
    import random
    random.seed(1)

    def attack_sample():
        return {
            "packet_count": 5000.0 + random.uniform(-500, 500),
            "byte_count": 300000.0 + random.uniform(-20000, 20000),
            "unique_dest_ips_est": 3.0 + random.uniform(-1, 1),
            "unique_dest_ports_est": 1.0,
            "syn_ratio": 0.98 + random.uniform(-0.02, 0.02),
            "protocol_entropy": random.uniform(0, 0.1),
        }

    def benign_sample():
        return {
            "packet_count": 12.0 + random.uniform(-5, 5),
            "byte_count": 4000.0 + random.uniform(-1000, 1000),
            "unique_dest_ips_est": 1.0,
            "unique_dest_ports_est": 1.0,
            "syn_ratio": random.uniform(0, 0.05),
            "protocol_entropy": 1.4 + random.uniform(-0.2, 0.2),
        }

    detector = DDoSHoeffdingDetector()
    for _ in range(100):
        detector.update(attack_sample(), label=1)
        detector.update(benign_sample(), label=0)

    attack_score = detector.score(attack_sample()).raw_score
    benign_score = detector.score(benign_sample()).raw_score

    assert attack_score > 0.8
    assert benign_score < 0.2
    assert detector.training_examples_seen == 200


def test_zero_variance_training_data_can_degenerate_to_even_split():
    """The finding that shaped the test above, preserved rather than
    quietly worked around: training on the exact same two fixed
    feature vectors, repeated hundreds of times with NO variance
    within each class, degenerates river's naive-Bayes leaf estimator
    -- every score comes back 50/50 regardless of input, even after
    1000 examples, even though the classes are trivially separable in
    principle. Checked directly, not assumed. This matters
    operationally: synthetic or replayed training data fed to this
    model needs realistic natural variance, or it can look like the
    model isn't learning when the real issue is degenerate input.
    """
    detector = DDoSHoeffdingDetector()
    for _ in range(300):
        detector.update(_attack_like(), label=1)
        detector.update(_benign_like(), label=0)

    attack_score = detector.score(_attack_like()).raw_score
    benign_score = detector.score(_benign_like()).raw_score

    assert abs(attack_score - 0.5) < 1e-9
    assert abs(benign_score - 0.5) < 1e-9


def test_missing_feature_keys_score_without_raising():
    detector = DDoSHoeffdingDetector()
    detector.update({"packet_count": 100.0}, label=1)  # sparse input
    result = detector.score({"unique_dest_ips_est": 5.0})  # different sparse input
    assert result.evidence["status"] == "trained"  # didn't crash, produced a result


def test_extract_features_defaults_missing_keys_to_zero():
    x = _extract_features({"packet_count": 42.0})
    assert x["packet_count"] == 42.0
    assert x["byte_count"] == 0.0
    assert x["syn_ratio"] == 0.0
    assert set(x.keys()) == {
        "packet_count", "byte_count", "unique_dest_ips_est",
        "unique_dest_ports_est", "syn_ratio", "protocol_entropy",
    }


def test_extract_features_ignores_extra_keys():
    """A real pipeline vector carries emission_reason/aggregate_type/
    source_ip alongside the numeric features -- these must not leak
    into the model's input or break it."""
    x = _extract_features({
        "packet_count": 10.0, "emission_reason": "tumbling_window",
        "aggregate_type": "source", "source_ip": "10.0.0.5",
    })
    assert "emission_reason" not in x
    assert "source_ip" not in x
    assert x["packet_count"] == 10.0
