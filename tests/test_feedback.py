"""Tests for feedback/feature_vector_store.py, retrain.py, canary.py.

retrain_unsupervised_batch's label-filtering (only benign examples
reach the Isolation Forest's fit()) is tested directly against real
model instances, not mocked -- checking training_examples_seen is the
concrete way to confirm attack examples were actually excluded, not
just trusted to be.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from passive_ids.feedback.canary import evaluate_canary
from passive_ids.feedback.feature_vector_store import FeatureVectorStore
from passive_ids.feedback.retrain import (
    build_training_data,
    retrain_dga_classifier,
    retrain_dga_markov_gate,
    retrain_online,
    retrain_supervised_batch,
    retrain_unsupervised_batch,
)
from passive_ids.models.c2_beaconing import C2BeaconingDetector
from passive_ids.models.ddos_hoeffding import DDoSHoeffdingDetector
from passive_ids.models.dga_cascade import DGACascadeDetector
from passive_ids.models.port_scan import PortScanDetector
from passive_ids.verdicts.schema import Confidence, SampleReason, Verdict, VerdictLabel
from passive_ids.verdicts.store import VerdictStore
from datetime import datetime, timezone


def _verdict(flow_id, verdict, model_predicted_class, confirmed_threat_class):
    return Verdict(
        verdict_id=f"VER-{flow_id}", alert_id="ALT-1", flow_id=flow_id,
        feature_vector_version="fv-1", sample_reason=SampleReason.ALERTED,
        analyst_id="a1", verdict_timestamp=datetime.now(timezone.utc),
        verdict=verdict, confirmed_threat_class=confirmed_threat_class,
        model_predicted_class=model_predicted_class, analyst_confidence=Confidence.HIGH,
    )


# ---------------------------------------------------------------------------
# FeatureVectorStore
# ---------------------------------------------------------------------------

def test_record_and_get_round_trip():
    store = FeatureVectorStore()
    store.record("f1", "fv-1", {"packet_count": 10})
    assert store.get("f1", "fv-1") == {"packet_count": 10}


def test_different_versions_do_not_collide():
    store = FeatureVectorStore()
    store.record("f1", "fv-1", {"packet_count": 10})
    store.record("f1", "fv-2", {"packet_count": 999})
    assert store.get("f1", "fv-1") == {"packet_count": 10}
    assert store.get("f1", "fv-2") == {"packet_count": 999}


def test_missing_key_returns_none():
    store = FeatureVectorStore()
    assert store.get("nonexistent", "fv-1") is None


# ---------------------------------------------------------------------------
# build_training_data
# ---------------------------------------------------------------------------

def test_build_training_data_matches_and_counts_skips():
    verdict_store = VerdictStore()
    verdict_store.add(_verdict("f1", VerdictLabel.TRUE_POSITIVE, "ddos", "ddos"))
    verdict_store.add(_verdict("f2", VerdictLabel.FALSE_POSITIVE, "ddos", None))
    verdict_store.add(_verdict("f3", VerdictLabel.TRUE_POSITIVE, "ddos", "ddos"))  # no matching feature vector

    feature_store = FeatureVectorStore()
    feature_store.record("f1", "fv-1", {"packet_count": 100})
    feature_store.record("f2", "fv-1", {"packet_count": 5})
    # f3 deliberately not recorded

    fvs, labels, skipped = build_training_data(verdict_store, feature_store, "ddos", "fv-1")
    assert len(fvs) == 2
    assert labels == [1, 0]
    assert skipped == 1


def test_build_training_data_version_mismatch_counts_as_skipped():
    verdict_store = VerdictStore()
    verdict_store.add(_verdict("f1", VerdictLabel.TRUE_POSITIVE, "ddos", "ddos"))
    feature_store = FeatureVectorStore()
    feature_store.record("f1", "fv-OLD", {"packet_count": 100})  # wrong version

    fvs, labels, skipped = build_training_data(verdict_store, feature_store, "ddos", "fv-1")
    assert len(fvs) == 0
    assert skipped == 1


# ---------------------------------------------------------------------------
# retrain_online
# ---------------------------------------------------------------------------

def test_retrain_online_updates_ddos_model():
    model = DDoSHoeffdingDetector()
    fvs = [{"packet_count": 5000.0, "syn_ratio": 0.98}] * 5
    labels = [1] * 5
    applied = retrain_online(model, fvs, labels)
    assert applied == 5
    assert model.training_examples_seen == 5


# ---------------------------------------------------------------------------
# retrain_supervised_batch
# ---------------------------------------------------------------------------

def test_retrain_supervised_batch_below_min_examples_does_not_fit():
    model = PortScanDetector(random_state=42)
    fvs = [{"unique_dest_ports_est": 40.0}] * 5
    result = retrain_supervised_batch(model, fvs, [1] * 5, min_examples=20)
    assert result is False
    assert model.score(fvs[0]).evidence["status"] == "untrained"


def test_retrain_supervised_batch_above_min_examples_fits():
    import random
    rng = random.Random(1)

    def scan_fv():
        return {"unique_dest_ports_est": 40.0 + rng.uniform(-5, 5), "dest_port_entropy": 5.0,
                "syn_ratio": 0.9, "protocol_entropy": 0.0, "unique_dest_ips_est": 1.0,
                "packet_count": 40.0 + rng.uniform(-5, 5)}

    def benign_fv():
        return {"unique_dest_ports_est": 1.0, "dest_port_entropy": 0.0,
                "syn_ratio": 0.0, "protocol_entropy": 1.4, "unique_dest_ips_est": 1.0,
                "packet_count": 10.0 + rng.uniform(-3, 3)}

    model = PortScanDetector(random_state=42)
    fvs = [scan_fv() for _ in range(15)] + [benign_fv() for _ in range(15)]
    labels = [1] * 15 + [0] * 15
    result = retrain_supervised_batch(model, fvs, labels, min_examples=20)
    assert result is True
    assert model.training_examples_seen == 30


# ---------------------------------------------------------------------------
# retrain_unsupervised_batch -- the critical label-filtering behavior
# ---------------------------------------------------------------------------

def test_retrain_unsupervised_batch_excludes_positive_labels():
    """The behavior that most needed direct verification: attack
    -labeled examples must NEVER reach the Isolation Forest's fit()."""
    model = C2BeaconingDetector(random_state=42)
    fvs = [{"duration": 10.0, "iat": {"mean": 5.0, "std": 1.0, "coefficient_of_variation": 0.2, "n": 10}}] * 15
    labels = [0] * 10 + [1] * 5  # 10 benign, 5 attack

    result = retrain_unsupervised_batch(model, fvs, labels, min_examples=5)
    assert result is True
    assert model.training_examples_seen == 10  # only the benign ones


def test_retrain_unsupervised_batch_below_min_examples_after_filtering():
    """min_examples is checked against the FILTERED count, not the raw
    input count -- 8 benign examples should fail a min_examples=10
    bar even though 20 total examples were passed in."""
    model = C2BeaconingDetector(random_state=42)
    fvs = [{"duration": 10.0}] * 20
    labels = [0] * 8 + [1] * 12
    result = retrain_unsupervised_batch(model, fvs, labels, min_examples=10)
    assert result is False


# ---------------------------------------------------------------------------
# DGA-specific retraining
# ---------------------------------------------------------------------------

def test_retrain_dga_classifier_extracts_domain_strings():
    model = DGACascadeDetector(random_state=42, min_child_samples=1)
    model.fit_markov_gate(["google", "facebook"] * 10)
    fvs = [{"domain": "google.com"}] * 15 + [{"domain": "xk4jq9z1.tk"}] * 15
    labels = [0] * 15 + [1] * 15

    result = retrain_dga_classifier(model, fvs, labels, min_examples=20)
    assert result is True
    assert model.score({"domain": "xk4jq9z1.tk"}).evidence["status"] == "trained"


def test_retrain_dga_classifier_skips_vectors_missing_domain_key():
    model = DGACascadeDetector(random_state=42)
    fvs = [{"domain": "google.com"}] * 25 + [{"no_domain_key": True}] * 5
    labels = [0] * 25 + [1] * 5
    result = retrain_dga_classifier(model, fvs, labels, min_examples=20)
    assert result is True  # 25 valid domains still clears the bar


def test_retrain_dga_markov_gate_fits_from_domain_strings():
    model = DGACascadeDetector()
    fvs = [{"domain": "google.com"}, {"domain": "facebook.com"}]
    count = retrain_dga_markov_gate(model, fvs)
    assert count == 2
    assert model._markov.fitted is True


# ---------------------------------------------------------------------------
# canary.evaluate_canary
# ---------------------------------------------------------------------------

def test_canary_recommends_deploy_when_candidate_clearly_better():
    import random
    rng = random.Random(2)

    def attack():
        return {"packet_count": 5000.0 + rng.uniform(-300, 300), "syn_ratio": 0.98 + rng.uniform(-0.01, 0.01)}

    def benign():
        return {"packet_count": 10.0 + rng.uniform(-3, 3), "syn_ratio": rng.uniform(0.0, 0.03)}

    # Realistic per-example variance, not exact-repeated fixed points --
    # the latter is the documented degenerate case in
    # models/ddos_hoeffding.py's own docstring (naive-Bayes leaf
    # estimator collapses to 50/50 regardless of input on zero-variance
    # data). Repeating that mistake here first is what caused this
    # test to fail initially.
    model_better = DDoSHoeffdingDetector()
    for _ in range(50):
        model_better.update(attack(), label=1)
        model_better.update(benign(), label=0)

    model_worse = DDoSHoeffdingDetector()  # left genuinely untrained -- cold start

    val_fvs = [attack() for _ in range(10)] + [benign() for _ in range(10)]
    val_labels = [1] * 10 + [0] * 10

    result = evaluate_canary(model_better, model_worse, val_fvs, val_labels, min_improvement=0.05)
    assert result.candidate_average_precision > result.incumbent_average_precision
    assert result.should_deploy is True


def test_canary_rejects_deploy_below_min_improvement_bar():
    model = DDoSHoeffdingDetector()
    attack = {"packet_count": 5000.0, "syn_ratio": 0.98}
    for _ in range(20):
        model.update(attack, label=1)
        model.update(attack, label=1)  # identical to incumbent -- no real improvement

    val_fvs = [attack] * 10
    val_labels = [1] * 10

    result = evaluate_canary(model, model, val_fvs, val_labels, min_improvement=0.01)
    assert abs(result.improvement) < 1e-9
    assert result.should_deploy is False
