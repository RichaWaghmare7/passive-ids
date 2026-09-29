"""Tests for verdicts/sampling.py."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from passive_ids.models.base import DetectionResult
from passive_ids.verdicts.sampling import sample_boundary, sample_random


def _pipeline_entry(raw_score, calibrated=None):
    return {
        "detection": DetectionResult(
            threat_class="ddos", model_name="hoeffding_tree",
            raw_score=raw_score, calibrated_confidence=calibrated,
        ),
        "context": {"source_ip": "10.0.0.5"},
    }


def test_sample_random_respects_rate():
    entries = [_pipeline_entry(0.1) for _ in range(1000)]
    sampled = sample_random(entries, sample_rate=0.01, rng=random.Random(1))
    assert len(sampled) == 10
    assert all(s.sample_reason == "random" for s in sampled)


def test_sample_random_empty_input_returns_empty():
    assert sample_random([], sample_rate=0.5) == []


def test_sample_random_rate_over_one_caps_at_population_size():
    entries = [_pipeline_entry(0.1) for _ in range(5)]
    sampled = sample_random(entries, sample_rate=2.0, rng=random.Random(1))
    assert len(sampled) == 5  # never more than the population


def test_sample_boundary_picks_closest_to_threshold():
    entries = [
        _pipeline_entry(0.1, calibrated=0.1),
        _pipeline_entry(0.55, calibrated=0.55),  # closest to threshold 0.6
        _pipeline_entry(0.9, calibrated=0.9),
    ]
    sampled = sample_boundary(entries, threshold=0.6, top_n=1)
    assert len(sampled) == 1
    assert sampled[0].detection.raw_score == 0.55
    assert sampled[0].sample_reason == "boundary"


def test_sample_boundary_prefers_calibrated_confidence_over_raw_score():
    # raw_score suggests this is far from threshold, but calibrated
    # confidence (what should actually be used) is right at it.
    entry = _pipeline_entry(raw_score=0.99, calibrated=0.61)
    sampled = sample_boundary([entry], threshold=0.6, top_n=1)
    assert sampled[0].detection.calibrated_confidence == 0.61


def test_sample_boundary_top_n_larger_than_population_returns_all():
    entries = [_pipeline_entry(0.1), _pipeline_entry(0.2)]
    sampled = sample_boundary(entries, threshold=0.5, top_n=100)
    assert len(sampled) == 2
