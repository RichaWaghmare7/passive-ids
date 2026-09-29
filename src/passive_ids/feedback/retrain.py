"""Offline retraining: joins verdict_store to feature_vector_store to
build real (X, y) training pairs, then retrains a model correctly for
*its own* fit signature.

The six models do not share one calling convention -- checked
directly against each model's actual code, not assumed to be uniform:

  - ddos_hoeffding.DDoSHoeffdingDetector: online. update(fv, label)
    once per example, no batch fit exists at all.
  - malware_encrypted.EncryptedMalwareDetector,
    port_scan.PortScanDetector: supervised batch. fit(feature_vectors,
    labels).
  - c2_beaconing.C2BeaconingDetector, exfiltration.ExfiltrationDetector:
    UNSUPERVISED batch Isolation Forests. fit(feature_vectors) --
    no labels parameter exists. These are trained on presumed-normal
    data only; feeding them confirmed-attack examples would corrupt
    what the model considers "normal" rather than teach it anything.
    retrain_unsupervised_batch below filters to label==0 (confirmed
    -benign) examples specifically, and discards the positives rather
    than passing them in.
  - dga_cascade.DGACascadeDetector: two-stage, and its own fit methods
    take domain STRINGS (fit_markov_gate(list[str]),
    fit_classifier(list[str], labels)), not feature-vector dicts --
    genuinely different enough that forcing it through a generic
    retrain_*_batch function here would silently do the wrong thing.
    retrain_dga_classifier below extracts the "domain" key from each
    stored feature vector rather than pretending this model fits the
    same shape as the others.

A single generic `retrain(model, X, y)` covering all six would have
been the more obviously "reusable" design, but it would have silently
mishandled at least three of them (unsupervised models fed positive
examples that corrupt their notion of normal; DGA's fit methods
called with the wrong argument type entirely). Explicit, per-shape
functions here instead.

min_examples guards on the batch functions are a direct response to
this project's own earlier finding: a batch model fit on too little
(or too cleanly separable) data produces exactly the degenerate,
overconfident thresholds discovered when alerting/generator.py's demo
data turned out to yield a threshold of 2.27e-64. Refusing to fit
below a floor is cheaper than discovering that failure mode again in
a real deploy.
"""
from __future__ import annotations

from typing import Any

from passive_ids.feedback.feature_vector_store import FeatureVectorStore
from passive_ids.models.c2_beaconing import C2BeaconingDetector
from passive_ids.models.ddos_hoeffding import DDoSHoeffdingDetector
from passive_ids.models.dga_cascade import DGACascadeDetector
from passive_ids.models.exfiltration import ExfiltrationDetector
from passive_ids.models.malware_encrypted import EncryptedMalwareDetector
from passive_ids.models.port_scan import PortScanDetector
from passive_ids.verdicts.store import VerdictStore


def build_training_data(
    verdict_store: VerdictStore,
    feature_store: FeatureVectorStore,
    threat_class: str,
    feature_vector_version: str,
) -> tuple[list[dict[str, Any]], list[int], int]:
    """Returns (feature_vectors, labels, skipped_count).

    skipped_count is flow_ids from verdict_store.training_pairs() that
    had no matching entry in feature_store -- exposed explicitly so a
    systematic mismatch (e.g. the wrong feature_vector_version passed
    in, or a verdict referencing a flow this store never recorded) is
    visible rather than silently producing an undersized training set
    with no indication why.
    """
    flow_ids, labels = verdict_store.training_pairs(threat_class)
    feature_vectors: list[dict[str, Any]] = []
    matched_labels: list[int] = []
    skipped = 0

    for flow_id, label in zip(flow_ids, labels):
        fv = feature_store.get(flow_id, feature_vector_version)
        if fv is None:
            skipped += 1
            continue
        feature_vectors.append(fv)
        matched_labels.append(label)

    return feature_vectors, matched_labels, skipped


def retrain_online(
    model: DDoSHoeffdingDetector, feature_vectors: list[dict[str, Any]], labels: list[int]
) -> int:
    """update() once per example. No min_examples guard -- online
    learners are designed for incremental updates on whatever arrives,
    including small batches; that's the point of the online/batch
    split in the first place. Returns the number applied."""
    for fv, label in zip(feature_vectors, labels):
        model.update(fv, label=label)
    return len(feature_vectors)


def retrain_supervised_batch(
    model: EncryptedMalwareDetector | PortScanDetector,
    feature_vectors: list[dict[str, Any]],
    labels: list[int],
    min_examples: int = 20,
) -> bool:
    """fit(feature_vectors, labels) -- replaces the model entirely,
    same as calling .fit() directly on these models. Returns False
    (does not retrain) below min_examples, rather than fitting on too
    little data and risking the degenerate-threshold failure mode
    described in this module's docstring.
    """
    if len(feature_vectors) < min_examples:
        return False
    model.fit(feature_vectors, labels)
    return True


def retrain_unsupervised_batch(
    model: C2BeaconingDetector | ExfiltrationDetector,
    feature_vectors: list[dict[str, Any]],
    labels: list[int],
    min_examples: int = 20,
) -> bool:
    """Filters to label==0 (confirmed-benign) examples ONLY, then
    fit(feature_vectors) with no labels -- these are unsupervised
    Isolation Forests; passing confirmed-attack examples into fit()
    would corrupt the model's learned notion of "normal" rather than
    teach it anything, since Isolation Forest has no way to use a
    label even if one were provided. min_examples is checked against
    the FILTERED (benign-only) count, not the raw input count.
    """
    benign_only = [fv for fv, label in zip(feature_vectors, labels) if label == 0]
    if len(benign_only) < min_examples:
        return False
    model.fit(benign_only)
    return True


def retrain_dga_classifier(
    model: DGACascadeDetector,
    feature_vectors: list[dict[str, Any]],
    labels: list[int],
    min_examples: int = 20,
) -> bool:
    """fit_classifier() specifically -- takes domain strings, extracted
    from each feature vector's "domain" key (the same key
    models.dga_cascade.DGACascadeDetector.score() itself expects). A
    feature vector missing that key is treated as a mismatch and
    excluded, same spirit as build_training_data's skipped_count.
    """
    domains: list[str] = []
    matched_labels: list[int] = []
    for fv, label in zip(feature_vectors, labels):
        domain = fv.get("domain")
        if domain is None:
            continue
        domains.append(domain)
        matched_labels.append(label)

    if len(domains) < min_examples:
        return False
    model.fit_classifier(domains, matched_labels)
    return True


def retrain_dga_markov_gate(model: DGACascadeDetector, legitimate_feature_vectors: list[dict[str, Any]]) -> int:
    """fit_markov_gate() -- takes only confirmed-legitimate domains
    (unsupervised plausibility model, see models/dga_cascade.py).
    Caller is responsible for filtering to legitimate-only vectors
    before calling this, same as retrain_unsupervised_batch does
    internally for the Isolation Forests -- kept as the caller's
    responsibility here rather than hidden, since "legitimate" for the
    Markov gate specifically means confirmed-NOT-DGA, which callers
    should already have as label==0 verdicts for the dga class.
    """
    domains = [fv["domain"] for fv in legitimate_feature_vectors if "domain" in fv]
    model.fit_markov_gate(domains)
    return len(domains)
