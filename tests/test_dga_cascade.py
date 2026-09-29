"""Tests for models/dga_cascade.py.

CharMarkovModel is checked against hand-derived exact values for a
tiny, fully-traceable training case (see the bash probe used to derive
them before this module was written) -- not just "weird strings score
lower than normal ones."
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from passive_ids.models.dga_cascade import (
    CharMarkovModel,
    DGACascadeDetector,
    _build_lgbm_features,
    _digit_ratio,
    _extract_label,
    _vowel_ratio,
)


# ---------------------------------------------------------------------------
# CharMarkovModel -- exact hand-derived values
# ---------------------------------------------------------------------------

def test_markov_unfit_returns_none():
    model = CharMarkovModel(order=1)
    assert model.score("anything") is None


def test_markov_exact_score_matches_hand_derivation():
    model = CharMarkovModel(order=1, smoothing=1.0)
    model.fit(["ab", "ab", "ab"])

    expected_ab = (math.log(4 / 5) + math.log(4 / 5)) / 2
    assert abs(model.score("ab") - expected_ab) < 1e-9

    expected_ba = (math.log(1 / 5) + math.log(1 / 2)) / 2
    assert abs(model.score("ba") - expected_ba) < 1e-9


def test_markov_trained_string_scores_higher_than_novel_transitions():
    model = CharMarkovModel(order=1, smoothing=1.0)
    model.fit(["ab", "ab", "ab"])
    assert model.score("ab") > model.score("ba")


def test_markov_empty_string_returns_none():
    model = CharMarkovModel(order=1)
    model.fit(["ab"])
    assert model.score("") is None


def test_markov_training_examples_seen_tracked():
    model = CharMarkovModel()
    model.fit(["a", "b", "c"])
    assert model.training_examples_seen == 3


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def test_extract_label_strips_tld():
    assert _extract_label("google.com") == "google"
    assert _extract_label("mail.google.com") == "mail"
    assert _extract_label("") == ""


def test_digit_ratio():
    assert _digit_ratio("abc123") == 0.5
    assert _digit_ratio("abcdef") == 0.0
    assert _digit_ratio("") == 0.0


def test_vowel_ratio():
    assert abs(_vowel_ratio("google") - 3 / 6) < 1e-9
    assert _vowel_ratio("xkjq") == 0.0


def test_build_lgbm_features_uses_zero_for_none_markov_score():
    features = _build_lgbm_features("test", markov_score=None)
    assert features[-1] == 0.0
    features_with_score = _build_lgbm_features("test", markov_score=-2.5)
    assert features_with_score[-1] == -2.5


# ---------------------------------------------------------------------------
# DGACascadeDetector -- full cascade
# ---------------------------------------------------------------------------

def test_cascade_untrained_returns_zero_and_untrained_status():
    detector = DGACascadeDetector()
    result = detector.score({"domain": "xk4jq9z1.tk"})

    assert result.raw_score == 0.0
    assert result.evidence["status"] == "untrained"
    assert result.threat_class == "dga"
    assert result.model_name == "markov_lightgbm_cascade"


def test_cascade_markov_score_present_even_when_lgbm_untrained():
    detector = DGACascadeDetector()
    detector.fit_markov_gate(["google", "facebook", "amazon", "github", "microsoft"])

    result = detector.score({"domain": "xk4jq9z1.tk"})
    assert result.evidence["markov_score"] is not None
    assert result.evidence["status"] == "untrained"  # lgbm still not fit


def test_cascade_gate_short_circuits_when_threshold_set_and_cleared():
    detector = DGACascadeDetector(gate_threshold=-3.0)
    detector.fit_markov_gate(["google"] * 20)  # narrow corpus -> "google" scores very high

    result = detector.score({"domain": "google.com"})
    assert result.evidence["status"] == "gated_benign"
    assert result.raw_score == 0.0


def test_cascade_no_gate_by_default_falls_through_to_lgbm():
    """gate_threshold defaults to None -- the gate must never
    short-circuit on its own without an explicit, deliberately-set
    threshold (see module docstring: no invented cutoffs)."""
    detector = DGACascadeDetector()  # gate_threshold not set
    detector.fit_markov_gate(["google"] * 20)
    domains = ["google", "facebook", "amazon"] * 10 + ["xk4jq9z1", "qzx7f2mvk", "vbnmqwrt"] * 10
    labels = [0] * 30 + [1] * 30
    detector.fit_classifier(domains, labels)

    result = detector.score({"domain": "google.com"})
    assert result.evidence["status"] == "trained"  # reached lgbm, wasn't gated


def test_cascade_trained_classifier_separates_dga_from_legitimate():
    detector = DGACascadeDetector(random_state=42, min_child_samples=1)
    detector.fit_markov_gate(["google", "facebook", "amazon", "github"] * 10)

    legit = ["google", "facebook", "amazon", "github", "microsoft", "apple"] * 10
    dga = ["xk4jq9z1", "qzx7f2mvk", "vbnmqwrt", "zxcvbnmq", "plkjhgfd", "qazwsxed"] * 10
    detector.fit_classifier(legit + dga, [0] * len(legit) + [1] * len(dga))

    legit_score = detector.score({"domain": "google.com"}).raw_score
    dga_score = detector.score({"domain": "xk4jq9z1.tk"}).raw_score

    assert dga_score > legit_score


def test_cascade_handles_missing_domain_key_gracefully():
    detector = DGACascadeDetector()
    result = detector.score({})  # no "domain" key at all
    assert result.evidence["status"] == "untrained"  # didn't raise
