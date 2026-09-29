"""Tests for features/welford.py.

Correctness is checked against Python's stdlib `statistics` module as
ground truth, not just "the code runs." Includes the specific
catastrophic-cancellation scenario verified empirically before this
module was written -- see welford.py's module docstring.
"""
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from passive_ids.features.welford import WelfordIAT, WelfordStats


def test_matches_stdlib_mean_and_variance():
    random.seed(1)
    values = [random.uniform(0, 100) for _ in range(500)]

    stats = WelfordStats()
    for v in values:
        stats.update(v)

    assert abs(stats.mean - statistics.mean(values)) < 1e-9
    assert abs(stats.variance - statistics.pvariance(values)) < 1e-6
    assert abs(stats.sample_variance - statistics.variance(values)) < 1e-6


def test_n_zero_and_n_one_edge_cases():
    stats = WelfordStats()
    assert stats.n == 0
    assert stats.variance == 0.0
    assert stats.std == 0.0
    assert stats.coefficient_of_variation == 0.0

    stats.update(42.0)
    assert stats.n == 1
    assert stats.mean == 42.0
    assert stats.variance == 0.0  # no notion of spread from one sample


def test_coefficient_of_variation_low_for_regular_beaconing():
    # Simulates a C2 beacon: ~300s interval, +/- 2s jitter -- low CV
    random.seed(2)
    stats = WelfordStats()
    for _ in range(200):
        stats.update(300.0 + random.uniform(-2.0, 2.0))

    assert stats.coefficient_of_variation < 0.02  # tight, regular timing


def test_coefficient_of_variation_high_for_irregular_traffic():
    # Human-driven traffic: highly variable gaps between requests
    random.seed(3)
    stats = WelfordStats()
    for _ in range(200):
        stats.update(random.expovariate(1 / 5.0))  # mean 5s, exponential spread

    assert stats.coefficient_of_variation > 0.5


def test_coefficient_of_variation_zero_for_nonpositive_mean_no_raise():
    stats = WelfordStats()
    stats.update(-5.0)
    stats.update(5.0)  # mean == 0.0 exactly
    assert stats.coefficient_of_variation == 0.0  # must not raise ZeroDivisionError


def test_naive_formula_fails_at_epoch_scale_welford_does_not():
    """The specific scenario checked empirically before writing this
    module: raw epoch-scale values (~1e9) with tiny true variance make
    the naive E[X^2]-E[X]^2 formula produce a *negative* variance.
    Welford must not have this failure mode."""
    random.seed(42)
    base = 1_700_000_000.0  # realistic unix timestamp scale
    values = [base + random.uniform(-0.001, 0.001) for _ in range(1000)]

    welford = WelfordStats()
    for v in values:
        welford.update(v)

    n = len(values)
    sum_x = sum(values)
    sum_x2 = sum(v * v for v in values)
    naive_mean = sum_x / n
    naive_variance = (sum_x2 / n) - naive_mean**2

    true_variance = statistics.pvariance(values)
    naive_error = abs(naive_variance - true_variance)
    welford_error = abs(welford.variance - true_variance)

    # Catastrophic cancellation doesn't reliably go negative -- it just
    # produces a wrong answer whose error is many orders of magnitude
    # larger than the true variance. That's the deterministic property
    # to check, not the sign (this run: naive lands at ~512 instead of
    # ~3e-7; another seed might land negative instead -- both are the
    # same failure mode).
    assert naive_error > 1.0, (
        "sanity check on the test itself: expected the naive formula to "
        "be wildly off here, or this test isn't demonstrating anything"
    )
    assert welford.variance >= 0.0
    assert welford_error < 1e-6
    assert welford_error < naive_error / 1000


def test_welford_iat_first_observation_returns_none():
    iat = WelfordIAT()
    assert iat.observe(timestamp=100.0) is None
    assert iat.stats.n == 0


def test_welford_iat_computes_correct_deltas():
    iat = WelfordIAT()
    iat.observe(100.0)
    delta1 = iat.observe(103.0)
    delta2 = iat.observe(107.5)

    assert delta1 == 3.0
    assert delta2 == 4.5
    assert iat.stats.n == 2
    assert abs(iat.stats.mean - 3.75) < 1e-9


def test_welford_iat_snapshot_matches_stats_snapshot():
    iat = WelfordIAT()
    for t in [0.0, 1.0, 2.5, 4.0]:
        iat.observe(t)
    assert iat.snapshot() == iat.stats.snapshot()


from passive_ids.features.entropy import (
    StreamingCategoricalEntropy,
    string_char_entropy,
    string_ngram_entropy,
)


def test_shannon_entropy_closed_form_cases():
    # All-identical characters -- zero uncertainty
    assert string_char_entropy("aaaa") == 0.0
    # 4 equally-frequent distinct characters -- exactly log2(4)
    assert abs(string_char_entropy("abcd") - 2.0) < 1e-9
    # 2 equally-frequent distinct characters -- exactly log2(2)
    assert abs(string_char_entropy("aabb") - 1.0) < 1e-9


def test_char_entropy_empty_string_is_zero():
    assert string_char_entropy("") == 0.0


def test_ngram_entropy_matches_hand_computation():
    # bigrams of "abab": "ab","ba","ab" -> ab:2/3, ba:1/3
    expected = -(2 / 3 * math.log2(2 / 3) + 1 / 3 * math.log2(1 / 3))
    assert abs(string_ngram_entropy("abab", n=2) - expected) < 1e-9


def test_ngram_entropy_zero_when_string_shorter_than_n():
    assert string_ngram_entropy("a", n=2) == 0.0


def test_random_looking_domain_scores_higher_than_real_words():
    real_words = [string_char_entropy(w) for w in ("google", "facebook")]
    random_looking = [string_char_entropy(w) for w in ("xk4jq9z1", "qzx7f2mvk")]

    assert max(real_words) < min(random_looking)


def test_dictionary_dga_entropy_is_ambiguous_not_reliably_low():
    """The specific, checked-not-assumed finding from the module
    docstring: a dictionary-based DGA domain's entropy sits in the
    same range as ordinary multi-word domains -- it is NOT reliably
    below them. This is the empirical basis for treating char/n-gram
    entropy as insufficient, alone, for this DGA family."""
    google = string_char_entropy("google")
    facebook = string_char_entropy("facebook")
    dictionary_dga = string_char_entropy("applepine")

    # It must fall inside (or above) the legitimate range, not below it --
    # if this ever breaks, the ambiguity claim in the docstring is wrong
    # and needs updating, not the test loosened to pass.
    assert google <= dictionary_dga
    assert dictionary_dga <= facebook or abs(dictionary_dga - facebook) < 0.5


def test_streaming_categorical_entropy_low_for_single_protocol_flood():
    tracker = StreamingCategoricalEntropy()
    for _ in range(100):
        tracker.observe("UDP")  # pure UDP flood -- no diversity at all

    snap = tracker.snapshot()
    assert snap["entropy"] == 0.0
    assert snap["distinct_categories"] == 1
    assert snap["total_observations"] == 100


def test_streaming_categorical_entropy_higher_for_mixed_normal_traffic():
    tracker = StreamingCategoricalEntropy()
    for proto, count in [("TCP", 70), ("UDP", 25), ("ICMP", 5)]:
        for _ in range(count):
            tracker.observe(proto)

    snap = tracker.snapshot()
    assert snap["entropy"] > 0.0
    assert snap["distinct_categories"] == 3


def test_shannon_entropy_empty_frequencies_is_zero():
    from passive_ids.features.entropy import shannon_entropy
    assert shannon_entropy({}) == 0.0
