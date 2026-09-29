"""SOC verdict schema.

`feature_vector_version` must match the version recorded on the
original Alert -- feature engineering code changes over time, and
joining a verdict to a feature vector computed by different code
silently corrupts the training set built from it.

`INCONCLUSIVE` is a first-class outcome, not a default. It must be
excluded from training data entirely rather than folded into either
class -- treating "couldn't determine" as benign is how a false
negative quietly becomes a labeled negative.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class VerdictLabel(str, Enum):
    TRUE_POSITIVE = "TRUE_POSITIVE"
    TRUE_POSITIVE_WRONG_CLASS = "TRUE_POSITIVE_WRONG_CLASS"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    INCONCLUSIVE = "INCONCLUSIVE"


class SampleReason(str, Enum):
    ALERTED = "alerted"          # crossed a detection threshold
    RANDOM = "random"            # unbiased sample of sub-threshold flows
    BOUNDARY = "boundary"        # uncertainty-sampled, near a decision threshold


class Confidence(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Verdict(BaseModel):
    verdict_id: str
    alert_id: str | None  # None when sample_reason != ALERTED (no alert existed)
    flow_id: str
    feature_vector_version: str
    sample_reason: SampleReason
    analyst_id: str
    verdict_timestamp: datetime
    verdict: VerdictLabel
    confirmed_threat_class: str | None
    model_predicted_class: str | None
    analyst_confidence: Confidence
    notes: str = ""
    second_review_required: bool = False
