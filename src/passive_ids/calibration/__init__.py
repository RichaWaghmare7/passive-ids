"""Tier 6: per-class score calibration and threshold selection.

Each model's native output (probability, anomaly score, etc.) is mapped
onto a common 0-1 confidence scale via isotonic regression, then
thresholded independently per class using PR-curve-derived cutoffs
(see config/thresholds.yaml) -- not a single shared threshold.
"""
