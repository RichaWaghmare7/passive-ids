"""passive_ids: read-only, unidirectional threat detection for passive IP traffic.

Package layout mirrors the pipeline tiers, in order:
  capture       -> flow_state -> features -> emission
  models        (six threat-specific detectors)
  calibration   (PR-curve thresholds, per-class)
  alerting      (multi-label alert JSON)
  application   (dashboard, case management, SIEM export)
  verdicts      (SOC verdict capture + sampling)
  feedback      (retrain, recalibrate, canary deploy)
"""
