"""Tier 2: provisional flow entries -- cheap, short-TTL, before promotion.

Filters out the one-off packets a spoofed flood generates: a burst of
single-packet "flows" from randomized source ports/IPs never
accumulates enough packets to promote, and expires on its own via TTL
-- costing almost nothing. Only flows showing real persistence (>=
promotion_threshold packets) earn a full ConfirmedFlowEntry.

Deliberately minimal: a ProvisionalEntry tracks only a first-seen
timestamp and a packet count, not per-packet size/flag detail. That's
what keeps it cheap enough to hand out to every one-off flow a flood
generates. The cost of that decision surfaces at promotion time (see
confirmed_table.py): only the count and first-seen timestamp carry
over, never the byte/SYN detail of the packets seen before promotion.
"""
from __future__ import annotations

from dataclasses import dataclass

from passive_ids.flow_state.flow_key import FlowKey


@dataclass
class ProvisionalEntry:
    first_seen: float
    pkt_count: int = 0

    def is_expired(self, now: float, ttl_seconds: float) -> bool:
        return (now - self.first_seen) > ttl_seconds


class ProvisionalTable:
    def __init__(self, ttl_seconds: float = 2.0, promotion_threshold: int = 3):
        self.ttl_seconds = ttl_seconds
        self.promotion_threshold = promotion_threshold
        self._table: dict[FlowKey, ProvisionalEntry] = {}

    def __len__(self) -> int:
        return len(self._table)

    def __contains__(self, key: FlowKey) -> bool:
        return key in self._table

    def observe(self, key: FlowKey, timestamp: float) -> ProvisionalEntry | None:
        """Record one packet for this flow key.

        Returns the completed ProvisionalEntry (already removed from
        this table) if this packet just crossed promotion_threshold --
        the caller is expected to seed a confirmed entry from it.
        Returns None if the flow remains provisional.

        An expired entry is treated as if it never existed: a new
        ProvisionalEntry starts fresh rather than resuming a stale
        count, since the flow it was tracking is presumed gone.
        """
        entry = self._table.get(key)
        if entry is None or entry.is_expired(timestamp, self.ttl_seconds):
            entry = ProvisionalEntry(first_seen=timestamp)
            self._table[key] = entry

        entry.pkt_count += 1
        if entry.pkt_count >= self.promotion_threshold:
            return self._table.pop(key)
        return None

    def expire_stale(self, now: float) -> int:
        """Remove entries that timed out without reaching promotion.

        Call periodically (e.g. alongside the 1s tumbling-window
        tick) rather than on every packet -- scanning the whole table
        per-packet would defeat the point of keeping it cheap.
        Returns the number of entries removed.
        """
        stale_keys = [k for k, e in self._table.items() if e.is_expired(now, self.ttl_seconds)]
        for k in stale_keys:
            del self._table[k]
        return len(stale_keys)
