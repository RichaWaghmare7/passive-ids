"""Tier 5: the six threat-specific detectors, chosen for streaming compatibility.

  ddos_hoeffding        - Hoeffding Tree, online, per-window aggregate features
  c2_beaconing          - Welford CV features -> Isolation Forest
  dga_cascade           - Markov n-gram gate -> LightGBM on survivors
  malware_encrypted     - XGBoost/LightGBM on TLS metadata + packet-size stats
  port_scan             - LightGBM (shallow) / Logistic Regression
  exfiltration          - Isolation Forest on byte/packet ratios

All implement the common Detector interface in base.py.
"""
