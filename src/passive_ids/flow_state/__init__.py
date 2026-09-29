"""Tier 2-3: dual-key flow state with bounded memory.

- Provisional entries (cheap, ~2s TTL) before a flow earns a full entry.
- Confirmed 5-tuple flow table (capacity-capped, LRU eviction or degrade).
- Always-on per-source-IP aggregate (HyperLogLog + Count-Min), constant
  memory regardless of attack volume -- must never depend on the
  confirmed table having free capacity.
"""
