"""Retrain, recalibrate, canary deploy.

Joins verdict_store to the original feature vector (by flow_id +
feature_vector_version) to build training pairs, recomputes PR curves
and thresholds, and stages retrained models through canary deployment
before full rollout.
"""
