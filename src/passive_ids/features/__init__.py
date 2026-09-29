"""Tier 4: online (single-pass) statistics engines.

Welford's algorithm for streaming IAT mean/variance, streaming entropy
for domain/TLS-string randomness, HLL for cardinality, Count-Min for
frequency. None of these buffer raw history.
"""
