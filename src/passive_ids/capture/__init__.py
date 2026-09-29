"""Tier 0-1: passive packet capture and protocol parsing.

Parses L3/L4 headers, DNS queries, and TLS ClientHello fields off a
mirror port / TAP. Read-only: never writes back to the wire.
"""
