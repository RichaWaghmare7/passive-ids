"""Tier 3: bounded confirmed flow table.

Holds full per-flow state for flows that survived the provisional
promotion gate. Capacity-capped: applies eviction.find_evictable's
evict-idle-first-or-degrade policy under load, for the same reason
source_aggregate.SourceAggregateTable does -- a flood is itself a
resource-exhaustion attempt against the detector, not just the target.

pre_promotion_packet_count is a disclosed gap, not a silent one: a
ConfirmedFlowEntry created by promotion carries over the accumulated
packet count and true first-seen timestamp from its ProvisionalEntry,
but byte_count/syn_count only start accumulating from the promoting
packet onward -- the earlier packets' size/flag detail was never
recorded, by design, to keep provisional entries cheap. Anyone reading
byte_count as "total flow bytes" should add pre_promotion_packet_count
into account or read it as "bytes since promotion," not silently
assume it's complete.

The same disclosed gap applies to iat_tracker: it starts accumulating
from the promoting packet's arrival onward, not from first_seen --
there's no recorded arrival time for the individual pre-promotion
packets to difference against, only their count. The "duration" field
below (last_seen - first_seen) is the only signal available for the
pre-promotion period, and it's a coarse total-elapsed-time, not an
IAT distribution.

fwd_/bwd_ packet and byte counts exist because FlowKey.canonical()
normalizes both directions of a connection onto one entry -- see that
method's docstring for the bug this fixes (forward and reverse packets
used to silently create two unrelated entries). "Forward" means
whichever direction the canonical key's own src/dst represents, not
necessarily "client to server" -- it's an arbitrary but consistent
choice (see canonical()'s sort), not a semantic claim about which side
initiated the connection. This is exactly what
models.exfiltration.ExfiltrationDetector needs: outbound:inbound byte
ratio is only meaningful once both directions land in the same place.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

from passive_ids.features.welford import WelfordIAT
from passive_ids.flow_state.eviction import find_evictable
from passive_ids.flow_state.flow_key import FlowKey


@dataclass
class ConfirmedFlowEntry:
    flow_key: FlowKey
    first_seen: float
    pre_promotion_packet_count: int = 0
    last_seen: float = field(init=False)
    packet_count: int = 0
    byte_count: int = 0
    syn_count: int = 0
    fwd_packet_count: int = 0
    fwd_byte_count: int = 0
    bwd_packet_count: int = 0
    bwd_byte_count: int = 0
    iat_tracker: WelfordIAT = field(default_factory=WelfordIAT, init=False)

    def __post_init__(self) -> None:
        self.last_seen = self.first_seen

    def idle_for(self, now: float) -> float:
        return now - self.last_seen

    def update(
        self, timestamp: float, size: int, is_syn: bool = False, is_forward: bool = True
    ) -> None:
        self.packet_count += 1
        self.byte_count += size
        if is_syn:
            self.syn_count += 1
        if is_forward:
            self.fwd_packet_count += 1
            self.fwd_byte_count += size
        else:
            self.bwd_packet_count += 1
            self.bwd_byte_count += size
        self.iat_tracker.observe(timestamp)
        self.last_seen = max(self.last_seen, timestamp)

    def snapshot(self) -> dict:
        total_bytes = self.fwd_byte_count + self.bwd_byte_count
        total_dir_packets = self.fwd_packet_count + self.bwd_packet_count
        return {
            "packet_count": self.packet_count,
            "byte_count": self.byte_count,
            "syn_count": self.syn_count,
            "pre_promotion_packet_count": self.pre_promotion_packet_count,
            "total_packets_seen": self.packet_count + self.pre_promotion_packet_count,
            "duration": self.last_seen - self.first_seen,
            "iat": self.iat_tracker.snapshot(),
            "fwd_packet_count": self.fwd_packet_count,
            "fwd_byte_count": self.fwd_byte_count,
            "bwd_packet_count": self.bwd_packet_count,
            "bwd_byte_count": self.bwd_byte_count,
            # Ratios default to 0.5 (perfectly balanced) when there's no
            # directional data yet, rather than 0.0 or NaN -- 0.0 would
            # falsely read as "all inbound", and NaN would break anything
            # downstream that isn't defending against it.
            "byte_ratio_out": (self.fwd_byte_count / total_bytes) if total_bytes else 0.5,
            "packet_ratio_out": (
                self.fwd_packet_count / total_dir_packets if total_dir_packets else 0.5
            ),
        }


class ConfirmedFlowTable:
    def __init__(
        self,
        max_confirmed: int = 2_000_000,
        idle_evict_threshold_seconds: float = 5.0,
    ):
        self.max_confirmed = max_confirmed
        self.idle_evict_threshold_seconds = idle_evict_threshold_seconds
        self._table: OrderedDict[FlowKey, ConfirmedFlowEntry] = OrderedDict()
        self.degraded_count = 0  # rising rate under normal load means
        # max_confirmed is too low, or idle_evict_threshold_seconds too
        # conservative, for the actual traffic this is deployed against.

    def __len__(self) -> int:
        return len(self._table)

    def __contains__(self, key: FlowKey) -> bool:
        return key in self._table

    def get(self, key: FlowKey) -> ConfirmedFlowEntry | None:
        return self._table.get(key)

    def insert_or_update(
        self,
        key: FlowKey,
        timestamp: float,
        size: int,
        is_syn: bool = False,
        is_forward: bool = True,
        now: float | None = None,
        pre_promotion_packet_count: int = 0,
        first_seen_override: float | None = None,
    ) -> ConfirmedFlowEntry | None:
        """Update an existing confirmed flow, or admit a newly-promoted one.

        `key` must already be canonical (see FlowKey.canonical()) --
        this method itself does not canonicalize; that happens once,
        in BoundedFlowTable.observe_packet, so both directions of a
        connection reliably land on the same entry.

        Returns None if the table is at capacity and nothing is
        evictable -- the table is under genuine, sustained load, and
        the caller should treat this flow as degraded rather than
        force an insert (see eviction.py for why evicting something
        active would be the wrong tradeoff).
        """
        now = now if now is not None else timestamp

        if key in self._table:
            self._table.move_to_end(key)
            entry = self._table[key]
            entry.update(timestamp, size, is_syn, is_forward)
            return entry

        if len(self._table) >= self.max_confirmed:
            evict_key = find_evictable(self._table, now, self.idle_evict_threshold_seconds)
            if evict_key is not None:
                del self._table[evict_key]
            else:
                self.degraded_count += 1
                return None

        first_seen = first_seen_override if first_seen_override is not None else timestamp
        entry = ConfirmedFlowEntry(
            flow_key=key,
            first_seen=first_seen,
            pre_promotion_packet_count=pre_promotion_packet_count,
        )
        entry.update(timestamp, size, is_syn, is_forward)
        self._table[key] = entry
        return entry

    def expire(
        self, now: float, inactive_timeout_seconds: float, active_timeout_seconds: float
    ) -> list[ConfirmedFlowEntry]:
        """Remove and return flows ready for emission: either idle
        beyond inactive_timeout_seconds, or running beyond
        active_timeout_seconds (a NetFlow-style active timer, so a
        very long-lived flow still gets periodic feature vectors
        rather than silently running forever without ever being
        scored). Same shape as ProvisionalTable.expire_stale --
        called periodically by emission/triggers.py, not per-packet.
        """
        expired_keys = [
            key
            for key, entry in self._table.items()
            if entry.idle_for(now) > inactive_timeout_seconds
            or (now - entry.first_seen) > active_timeout_seconds
        ]
        return [self._table.pop(key) for key in expired_keys]
