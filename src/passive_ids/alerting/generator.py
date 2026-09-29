"""Multi-label alert generation.

Takes a group of DetectionResults that all pertain to the same "thing"
(a source IP's aggregate window, a flow, a DNS query, a TLS
connection -- caller decides the grouping, see pipeline.py's
self.detections for how context is currently attached) and assembles
them into one Alert containing multiple Detection entries -- the
multi-label design decided early in this project: a source can
legitimately be scanning AND beaconing at once, and that has to stay
visible as two Detection entries, not get collapsed into one blended
number by weighted-sum fusion.

Per-detection severity comes from config/thresholds.yaml's two-tier
thresholds (HIGH = min-precision-0.95 point, MEDIUM = that class's own
configured objective), once calibration/threshold_selection.py has
actually fit them -- the `null` placeholders can't produce a
Detection at all, by design (see build_detection: no threshold means
no severity means nothing to alert on, not a silent default).

Severity enum declaration order in alerting/schema.py (CRITICAL, HIGH,
MEDIUM, LOW) is NOT severity rank order -- it just happens to be the
order they were listed in. Computing overall severity via
`max(severities)` directly against that declaration order would
silently pick LOW as "biggest." _SEVERITY_RANK exists specifically to
avoid that mistake.

CRITICAL is never assigned by a single class's own threshold (the
two-tier design only ever produces HIGH/MEDIUM per class) -- it's an
alert-level escalation, applied here when either (a) two or more
distinct threat classes clear HIGH in the same group (compounding
signal -- a source that's simultaneously scanning and beaconing is a
different, worse situation than either alone), or (b) any detection
carries a known-bad JA4 exact match (models.malware_encrypted's
ja4_known_bad evidence flag) -- an exact hit against real threat
intel is about as certain as this system gets, independent of
whatever the classifier's own calibrated confidence happens to be.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from passive_ids.alerting.schema import Alert, Detection, Severity, Summary, fuse_noisy_or
from passive_ids.flow_state.flow_key import FlowKey
from passive_ids.models.base import DetectionResult

_SEVERITY_RANK: dict[Severity, int] = {
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


def _severity_for_score(confidence: float, class_thresholds: dict[str, Any]) -> Severity | None:
    """class_thresholds is one class's entry from config/thresholds.yaml
    (or the equivalent dict from a fitted ThresholdResult) -- expects
    keys "high_threshold" / "medium_threshold". Returns None if the
    score doesn't clear even MEDIUM: that result doesn't belong in an
    alert at all, not a LOW-severity placeholder.
    """
    high = class_thresholds.get("high_threshold")
    medium = class_thresholds.get("medium_threshold")

    if high is not None and confidence >= high:
        return Severity.HIGH
    if medium is not None and confidence >= medium:
        return Severity.MEDIUM
    return None


def build_detection(
    result: DetectionResult, thresholds_by_class: dict[str, dict[str, Any]]
) -> Detection | None:
    """Returns a Detection if `result` clears its class's threshold,
    else None.

    Prefers calibrated_confidence when present. Falls back to
    raw_score when no calibrator has been fit yet for this class --
    the threshold itself was fit on that same raw_score scale (see
    calibration/threshold_selection.py), so thresholding raw_score
    directly against it is still meaningful, just disclosed as
    uncalibrated in the evidence rather than silently presented as
    equivalent to a calibrated result.
    """
    score = result.calibrated_confidence
    is_calibrated = score is not None
    if score is None:
        score = result.raw_score

    class_thresholds = thresholds_by_class.get(result.threat_class, {})
    severity = _severity_for_score(score, class_thresholds)
    if severity is None:
        return None

    evidence = dict(result.evidence or {})
    if not is_calibrated:
        evidence["calibration_status"] = "uncalibrated -- thresholded on raw_score directly"

    return Detection(
        threat_class=result.threat_class,
        model=result.model_name,
        confidence=score,
        severity=severity,
        evidence=evidence,
    )


def _overall_severity(detections: list[Detection]) -> Severity:
    base = max(detections, key=lambda d: _SEVERITY_RANK[d.severity]).severity

    distinct_classes_at_high = {
        d.threat_class for d in detections if d.severity == Severity.HIGH
    }
    has_known_bad_match = any(d.evidence.get("ja4_known_bad") is True for d in detections)

    if len(distinct_classes_at_high) >= 2 or has_known_bad_match:
        return Severity.CRITICAL
    return base


def _extract_identity_fields(context: dict[str, Any]) -> dict[str, Any]:
    """Normalizes the different context shapes pipeline.py's
    self.detections currently attaches (source-aggregate: source_ip
    only; flow-level: a FlowKey; DNS: src_ip + domain; TLS: src_ip +
    dst_ip + ja4) into Alert's source_ip/destination_ip/flow_id
    fields. Fields with no natural value in a given context are left
    as None (Alert's schema already makes them Optional for exactly
    this reason) rather than filled with a placeholder.
    """
    flow_key = context.get("flow_key")
    if isinstance(flow_key, FlowKey):
        return {
            "source_ip": flow_key.src_ip,
            "destination_ip": flow_key.dst_ip,
            "flow_id": f"{flow_key.src_ip}:{flow_key.src_port}-{flow_key.dst_ip}:{flow_key.dst_port}-{flow_key.protocol}",
        }

    source_ip = context.get("source_ip") or context.get("src_ip")
    return {
        "source_ip": source_ip,
        "destination_ip": context.get("dst_ip"),
        "flow_id": None,
    }


def generate_alert(
    grouped_results: list[DetectionResult],
    thresholds_by_class: dict[str, dict[str, Any]],
    context: dict[str, Any],
    fusion_method: str = "noisy_or",
) -> Alert | None:
    """Assembles one multi-label Alert from a group of DetectionResults
    sharing one `context` (caller's grouping choice).

    Returns None if nothing in the group clears its threshold -- an
    all-benign group produces no alert at all, not an empty or
    trivially-LOW one.
    """
    detections = [
        d for r in grouped_results if (d := build_detection(r, thresholds_by_class)) is not None
    ]
    if not detections:
        return None

    confidences = [d.confidence for d in detections]
    if fusion_method == "noisy_or":
        overall_score = fuse_noisy_or(confidences)
    elif fusion_method == "max":
        overall_score = max(confidences)
    else:
        raise ValueError(f"Unknown fusion_method '{fusion_method}'. Use 'noisy_or' or 'max'.")

    identity = _extract_identity_fields(context)
    if identity["source_ip"] is None:
        raise ValueError(
            "context must supply at least a source_ip/src_ip or a FlowKey -- "
            f"got context={context!r}"
        )

    return Alert(
        alert_id=f"ALT-{uuid.uuid4().hex[:12]}",
        timestamp=datetime.now(timezone.utc),
        source_ip=identity["source_ip"],
        destination_ip=identity["destination_ip"],
        flow_id=identity["flow_id"],
        detections=detections,
        summary=Summary(
            overall_severity=_overall_severity(detections),
            overall_score_method=fusion_method,
            overall_score=overall_score,
            threats_detected_count=len(detections),
        ),
    )
