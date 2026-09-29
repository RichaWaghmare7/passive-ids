"""Tier 4->5 boundary: decides when a flow's feature vector is emitted.

Two independent triggers: flow-level (inactive/active timeout) and
window-level (1s tumbling tick per source IP for aggregate features).
"""
