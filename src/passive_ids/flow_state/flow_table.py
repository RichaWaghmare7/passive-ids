"""Tier 2-3 orchestrator: the "bounded lane" from the architecture diagrams.

Runs alongside (not gating, and not gated by) source_aggregate's
always-on "unbounded lane" -- both lanes update independently from the
same packet, per pipeline.py.
"""
from __future__ import annotations

from passive_ids.flow_state.confirmed_table import ConfirmedFlowEntry, ConfirmedFlowTable
from passive_ids.flow_state.flow_key import FlowKey
from passive_ids.flow_state.provisional_table import ProvisionalTable


class BoundedFlowTable:
    def __init__(
        self,
        provisional_ttl_seconds: float = 2.0,
        promotion_threshold: int = 3,
        max_confirmed: int = 2_000_000,
        idle_evict_threshold_seconds: float = 5.0,
    ):
        self.provisional = ProvisionalTable(
            ttl_seconds=provisional_ttl_seconds, promotion_threshold=promotion_threshold
        )
        self.confirmed = ConfirmedFlowTable(
            max_confirmed=max_confirmed,
            idle_evict_threshold_seconds=idle_evict_threshold_seconds,
        )

    def observe_packet(
        self, key: FlowKey, timestamp: float, size: int, is_syn: bool = False
    ) -> ConfirmedFlowEntry | None:
        """Feed one packet through the bounded lane.

        Canonicalizes `key` first (see FlowKey.canonical()) so both
        directions of a connection accumulate toward the same
        provisional/confirmed entry -- without this, a request and its
        response would silently create two unrelated entries, each
        needing to independently reach promotion_threshold, which can
        delay or prevent promotion for flows that split evenly between
        directions (verified directly before this fix existed).

        Returns the ConfirmedFlowEntry if this flow is (now, or
        already) confirmed. Returns None if it's still provisional
        (hasn't reached promotion_threshold yet) or was degraded
        (confirmed table at capacity, nothing evictable) -- callers
        must not treat None as an error, both are expected outcomes.
        """
        canonical_key, is_forward = key.canonical()

        if canonical_key in self.confirmed:
            return self.confirmed.insert_or_update(
                canonical_key, timestamp, size, is_syn, is_forward, now=timestamp
            )

        promoted = self.provisional.observe(canonical_key, timestamp)
        if promoted is None:
            return None

        # -1 because the promoting packet itself is about to be counted
        # by insert_or_update -- avoids double-counting it.
        return self.confirmed.insert_or_update(
            canonical_key,
            timestamp,
            size,
            is_syn,
            is_forward,
            now=timestamp,
            pre_promotion_packet_count=promoted.pkt_count - 1,
            first_seen_override=promoted.first_seen,
        )

    def expire_provisional(self, now: float) -> int:
        """Periodic maintenance -- see ProvisionalTable.expire_stale."""
        return self.provisional.expire_stale(now)
