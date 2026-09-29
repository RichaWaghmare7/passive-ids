"""Threat-intelligence lookups: static, loadable reference data that
detection logic checks traffic against -- distinct from flow_state/
(traffic-derived, bounded by observed volume) and models/ (learned
detectors). Currently: JA4 fingerprint matching against known-bad
lists. No real intelligence data ships in this repository -- see
ja4_lookup.py's module docstring for why, and where to source it.
"""
