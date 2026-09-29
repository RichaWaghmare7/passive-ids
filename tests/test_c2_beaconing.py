"""Tests for models/c2_beaconing.py.

Cold-start and sign-convention behavior are checked against exactly
what was verified directly against scikit-learn before writing this
model (NotFittedError pre-fit; decision_function negative-for-anomaly,
which this model must sign-flip).
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from passive_ids.models.c2_beaconing import C2BeaconingDetector, _extract_features


def _flow_vector(mean, std, cv, n, packet_count=None, duration=None):
    packet_count = packet_count if packet_count is not None else n + 1
    duration = duration if duration is not None else mean * n
    return {
        "packet_count": packet_count,
        "duration": duration,
        "iat": {"n": n, "mean": mean, "std": std, "coefficient_of_variation": cv},
    }


def _regular_beacon():
    return _flow_vector(mean=300.0, std=3.0, cv=0.01, n=40)  # very regular -- C2-like


def _irregular_normal(rng):
    mean = rng.uniform(5, 60)
    cv = rng.uniform(0.5, 1.5)
    return _flow_vector(mean=mean, std=mean * cv, cv=cv, n=rng.randint(5, 50))


def test_cold_start_returns_zero_score_and_untrained_status():
    detector = C2BeaconingDetector(random_state=42)
    result = detector.score(_regular_beacon())

    assert result.raw_score == 0.0
    assert result.evidence["status"] == "untrained"
    assert result.threat_class == "c2_beaconing"
    assert result.model_name == "isolation_forest"


def test_fit_then_score_flags_regular_beacon_above_irregular_normal():
    rng = random.Random(1)
    normal_population = [_irregular_normal(rng) for _ in range(200)]

    detector = C2BeaconingDetector(random_state=42, contamination=0.05)
    detector.fit(normal_population)

    beacon_result = detector.score(_regular_beacon())
    normal_result = detector.score(_irregular_normal(random.Random(99)))

    assert beacon_result.evidence["status"] == "trained"
    assert beacon_result.raw_score > normal_result.raw_score


def test_sklearn_is_anomaly_flag_matches_sign_of_raw_score_direction():
    rng = random.Random(2)
    normal_population = [_irregular_normal(rng) for _ in range(200)]
    detector = C2BeaconingDetector(random_state=42, contamination=0.05)
    detector.fit(normal_population)

    beacon_result = detector.score(_regular_beacon())
    # A clear, tight beacon far outside the training population should
    # trip sklearn's own anomaly flag too, not just have a higher score.
    assert beacon_result.evidence["sklearn_is_anomaly"] is True


def test_training_examples_seen_tracks_fit_batch_size():
    detector = C2BeaconingDetector(random_state=42)
    rng = random.Random(3)
    batch = [_irregular_normal(rng) for _ in range(150)]
    detector.fit(batch)
    assert detector.training_examples_seen == 150


def test_refit_replaces_not_accumulates():
    detector = C2BeaconingDetector(random_state=42)
    rng = random.Random(4)
    detector.fit([_irregular_normal(rng) for _ in range(50)])
    detector.fit([_irregular_normal(rng) for _ in range(75)])
    assert detector.training_examples_seen == 75  # replaced, not 125


def test_update_is_a_noop_inherited_from_base():
    detector = C2BeaconingDetector(random_state=42)
    detector.update(_regular_beacon(), label=1)  # must not raise
    assert detector.training_examples_seen == 0  # unaffected -- fit() is the only real trainer
    assert detector.score(_regular_beacon()).evidence["status"] == "untrained"


def test_missing_iat_key_defaults_gracefully():
    detector = C2BeaconingDetector(random_state=42)
    rng = random.Random(5)
    detector.fit([_irregular_normal(rng) for _ in range(50)])

    result = detector.score({"packet_count": 5, "duration": 10.0})  # no "iat" at all
    assert result.evidence["status"] == "trained"  # scored without raising


def test_extract_features_flattens_nested_iat_in_fixed_order():
    vec = _flow_vector(mean=100.0, std=5.0, cv=0.05, n=20, packet_count=21, duration=2000.0)
    features = _extract_features(vec)
    # order: packet_count, duration, iat.mean, iat.std, iat.cv, iat.n
    assert features == [21.0, 2000.0, 100.0, 5.0, 0.05, 20.0]
