"""Tests for verdicts/store.py.

The TRUE_POSITIVE_WRONG_CLASS scenario is the one that needed real
thought, not just plumbing: a single verdict must produce a NEGATIVE
example for the class it was wrongly predicted as, and a POSITIVE
example for the class it was actually confirmed to be -- tested
directly in both directions.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from passive_ids.verdicts.schema import Confidence, SampleReason, Verdict, VerdictLabel
from passive_ids.verdicts.store import VerdictStore


def _verdict(
    flow_id, verdict, model_predicted_class, confirmed_threat_class,
    sample_reason=SampleReason.ALERTED, verdict_id=None,
):
    return Verdict(
        verdict_id=verdict_id or f"VER-{flow_id}",
        alert_id="ALT-1" if sample_reason == SampleReason.ALERTED else None,
        flow_id=flow_id,
        feature_vector_version="fv-1",
        sample_reason=sample_reason,
        analyst_id="analyst_1",
        verdict_timestamp=datetime.now(timezone.utc),
        verdict=verdict,
        confirmed_threat_class=confirmed_threat_class,
        model_predicted_class=model_predicted_class,
        analyst_confidence=Confidence.HIGH,
    )


def test_true_positive_produces_positive_label_for_predicted_class():
    store = VerdictStore()
    store.add(_verdict("f1", VerdictLabel.TRUE_POSITIVE, "ddos", "ddos"))

    flow_ids, labels = store.training_pairs("ddos")
    assert flow_ids == ["f1"]
    assert labels == [1]


def test_false_positive_produces_negative_label_for_predicted_class():
    store = VerdictStore()
    store.add(_verdict("f1", VerdictLabel.FALSE_POSITIVE, "ddos", None))

    flow_ids, labels = store.training_pairs("ddos")
    assert flow_ids == ["f1"]
    assert labels == [0]


def test_wrong_class_produces_negative_for_predicted_and_positive_for_confirmed():
    """The core scenario: DDoS's detector fired, but the analyst
    confirms it was actually a port scan. One verdict, two opposite
    training implications depending on which class is asking."""
    store = VerdictStore()
    store.add(_verdict("f1", VerdictLabel.TRUE_POSITIVE_WRONG_CLASS, "ddos", "port_scanning"))

    ddos_ids, ddos_labels = store.training_pairs("ddos")
    assert ddos_ids == ["f1"]
    assert ddos_labels == [0]  # ddos's own detector was wrong here

    scan_ids, scan_labels = store.training_pairs("port_scanning")
    assert scan_ids == ["f1"]
    assert scan_labels == [1]  # genuinely was a port scan, even though a different detector caught it


def test_inconclusive_excluded_from_every_class_training_pairs():
    store = VerdictStore()
    store.add(_verdict("f1", VerdictLabel.INCONCLUSIVE, "ddos", "ddos"))

    for threat_class in ("ddos", "port_scanning", "c2_beaconing"):
        flow_ids, labels = store.training_pairs(threat_class)
        assert flow_ids == []
        assert labels == []


def test_unrelated_verdict_excluded_not_guessed_at():
    """A verdict about c2_beaconing (predicted and confirmed) tells us
    nothing about exfiltration -- must be skipped, not turned into a
    guessed negative example."""
    store = VerdictStore()
    store.add(_verdict("f1", VerdictLabel.TRUE_POSITIVE, "c2_beaconing", "c2_beaconing"))

    flow_ids, labels = store.training_pairs("exfiltration")
    assert flow_ids == []
    assert labels == []


def test_missed_positive_rate_none_with_no_random_verdicts():
    store = VerdictStore()
    store.add(_verdict("f1", VerdictLabel.TRUE_POSITIVE, "ddos", "ddos", sample_reason=SampleReason.ALERTED))
    assert store.missed_positive_rate("ddos") is None


def test_missed_positive_rate_computed_only_from_random_sample():
    store = VerdictStore()
    # 3 random samples, 1 confirmed as ddos (a miss -- it was sub-threshold)
    store.add(_verdict("f1", VerdictLabel.TRUE_POSITIVE, None, "ddos", sample_reason=SampleReason.RANDOM))
    store.add(_verdict("f2", VerdictLabel.FALSE_POSITIVE, None, None, sample_reason=SampleReason.RANDOM))
    store.add(_verdict("f3", VerdictLabel.FALSE_POSITIVE, None, None, sample_reason=SampleReason.RANDOM))
    # a boundary-sample verdict that must NOT count toward this (biased pool)
    store.add(_verdict("f4", VerdictLabel.TRUE_POSITIVE, "ddos", "ddos", sample_reason=SampleReason.BOUNDARY))

    rate = store.missed_positive_rate("ddos")
    assert abs(rate - (1 / 3)) < 1e-9


def test_by_sample_reason_filters_correctly():
    store = VerdictStore()
    store.add(_verdict("f1", VerdictLabel.TRUE_POSITIVE, "ddos", "ddos", sample_reason=SampleReason.ALERTED))
    store.add(_verdict("f2", VerdictLabel.FALSE_POSITIVE, None, None, sample_reason=SampleReason.RANDOM))

    assert len(store.by_sample_reason(SampleReason.ALERTED)) == 1
    assert len(store.by_sample_reason(SampleReason.RANDOM)) == 1
    assert len(store) == 2
