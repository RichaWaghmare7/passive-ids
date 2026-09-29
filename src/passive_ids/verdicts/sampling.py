"""Random and boundary (uncertainty) sampling of sub-threshold detections.

Fixes the selection-bias problem established early in this project:
reviewing only alerts (things that crossed a threshold) means false
negatives are structurally invisible -- there is no ground truth for
a flow nobody was ever shown. Two complementary strategies feed the
SOC review queue alongside actually-alerted detections, and must stay
tagged separately (see verdicts.schema.SampleReason) rather than
pooled together:

  - random: an unbiased sample of sub-threshold results. The ONLY pool
    that can honestly estimate recall/false-negative rate, precisely
    because it isn't chosen based on how uncertain or borderline a
    result looks.
  - boundary: uncertainty sampling -- results whose score sits closest
    to the class's own threshold. The most informative pool for
    retraining, but deliberately biased toward hard cases -- mixing it
    into a recall estimate would overstate the miss rate.
"""
from __future__ import annotations

import random as random_module
from dataclasses import dataclass
from typing import Any

from passive_ids.models.base import DetectionResult


@dataclass
class SampledResult:
    detection: DetectionResult
    context: dict[str, Any]
    sample_reason: str  # "random" or "boundary" -- matches verdicts.schema.SampleReason


def sample_random(
    sub_threshold_results: list[dict[str, Any]],
    sample_rate: float,
    rng: random_module.Random | None = None,
) -> list[SampledResult]:
    """sub_threshold_results: pipeline-shaped {"detection": ...,
    "context": ...} entries that did NOT clear any alert threshold --
    caller is responsible for that filtering; this function doesn't
    re-check scores against thresholds itself.
    """
    rng = rng or random_module.Random()
    n = min(round(len(sub_threshold_results) * sample_rate), len(sub_threshold_results))
    chosen = rng.sample(sub_threshold_results, k=max(n, 0))
    return [
        SampledResult(detection=r["detection"], context=r["context"], sample_reason="random")
        for r in chosen
    ]


def sample_boundary(
    sub_threshold_results: list[dict[str, Any]], threshold: float, top_n: int
) -> list[SampledResult]:
    """Selects the `top_n` results whose score sits closest to
    `threshold` (typically a class's own MEDIUM threshold) -- most
    useful for retraining specifically because the model is least
    confident about them, not because they represent overall traffic.
    """

    def _distance(r: dict[str, Any]) -> float:
        d = r["detection"]
        score = d.calibrated_confidence if d.calibrated_confidence is not None else d.raw_score
        return abs(score - threshold)

    ranked = sorted(sub_threshold_results, key=_distance)
    return [
        SampledResult(detection=r["detection"], context=r["context"], sample_reason="boundary")
        for r in ranked[:top_n]
    ]
