"""SOC verdict collection: closes the loop from alert to ground truth.

Two sampling arms feed the review queue: alerted flows (precision
signal) and stratified/boundary-sampled sub-threshold flows (recall
signal, without which false negatives are invisible).
"""
