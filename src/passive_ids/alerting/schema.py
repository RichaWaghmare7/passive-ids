"""Multi-label alert schema.

A single flow can legitimately trigger more than one threat_class at
once (e.g. reconnaissance immediately followed by a DGA lookup from
the same source). `detections` is a list for exactly that reason --
weighted-sum fusion into one risk score was deliberately rejected
earlier in this project because it hides that kind of co-occurrence.
`summary.overall_score` is a sort key for triage only, never a
substitute for the per-class confidences.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Detection(BaseModel):
    threat_class: str
    model: str
    confidence: float = Field(ge=0.0, le=1.0)
    severity: Severity
    evidence: dict[str, Any] = Field(default_factory=dict)


class Summary(BaseModel):
    overall_severity: Severity
    overall_score_method: str  # e.g. "noisy_or", "max" -- always labeled, never implied
    overall_score: float = Field(ge=0.0, le=1.0)
    threats_detected_count: int


class Alert(BaseModel):
    alert_id: str
    timestamp: datetime
    source_ip: str
    detections: list[Detection]
    summary: Summary
    # Optional because not every detection context has these: a
    # source-aggregate alert (DDoS, port scan) is aggregated across
    # many destinations, so there's no single destination_ip or
    # flow_id to report; feature_vector_version doesn't exist as a
    # real tracked value anywhere in this codebase yet (it was
    # aspirational in the original SOC-feedback design, never
    # actually implemented) -- None here is honest, not a placeholder
    # string standing in for real data that doesn't exist.
    destination_ip: str | None = None
    flow_id: str | None = None
    feature_vector_version: str | None = None


def fuse_noisy_or(confidences: list[float]) -> float:
    """P(at least one detector is right), assuming rough independence.

    Used only for `summary.overall_score` -- a triage sort key, not a
    calibrated probability of any specific attack.
    """
    survival = 1.0
    for p in confidences:
        survival *= (1 - p)
    return 1 - survival
