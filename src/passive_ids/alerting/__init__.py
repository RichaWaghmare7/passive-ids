"""Tier 7: multi-label alert generation.

A flow can trigger zero, one, or multiple threat_class detections
simultaneously. No weighted-sum fusion into a single risk score --
each detection keeps its own model, confidence, and evidence.
"""
