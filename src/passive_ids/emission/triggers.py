"""Emission triggers: decides when a feature vector is complete and
ready to hand to models/.

Two independent triggers, matching the architecture diagrams:

1. Flow-level (FlowTimeoutTrigger): a confirmed flow that's gone idle
   beyond an inactive timeout, or has been running beyond an active
   timeout (NetFlow-style, so a very long-lived flow still gets
   periodic feature vectors rather than silently running forever
   unscored), gets its snapshot emitted and is removed from the
   confirmed table -- its lifecycle is considered complete, or
   complete-for-now.

2. Window-level (TumblingWindowTrigger): every tumbling-window
   interval, every tracked source's current aggregate snapshot is
   emitted, then the aggregate is reset. This is the fix for the
   exact gap flagged when dns_aggregate.py was built: reset_all()
   existed on both aggregate tables, but nothing ever called it, so
   cardinality/rate features were lifetime totals rather than
   reflecting recent behavior.

Both triggers are checked opportunistically, on packet arrival,
rather than on a real background timer (EmissionScheduler below) --
this project has no threading/async model yet, so "check elapsed
wall-clock time when a packet happens to arrive" is the honest
implementation, not a preemptive schedule. This means a window can
run slightly long during a lull in traffic (nothing rolls it over
until the next packet arrives) -- disclosed here, not silently
assumed away. A future move to a real scheduler (or async event loop)
would only need to change EmissionScheduler; the two triggers
themselves have no dependency on how they're invoked.
"""
from __future__ import annotations

from typing import Any

from passive_ids.flow_state.confirmed_table import ConfirmedFlowTable
from passive_ids.flow_state.dns_aggregate import DNSAggregateTable
from passive_ids.flow_state.source_aggregate import SourceAggregateTable


class FlowTimeoutTrigger:
    def __init__(
        self,
        confirmed_table: ConfirmedFlowTable,
        inactive_timeout_seconds: float = 30.0,
        active_timeout_seconds: float = 1800.0,  # 30 min, NetFlow-style
    ):
        self.confirmed_table = confirmed_table
        self.inactive_timeout_seconds = inactive_timeout_seconds
        self.active_timeout_seconds = active_timeout_seconds

    def check(self, now: float) -> list[dict[str, Any]]:
        expired = self.confirmed_table.expire(
            now, self.inactive_timeout_seconds, self.active_timeout_seconds
        )
        vectors = []
        for entry in expired:
            vector = entry.snapshot()
            vector["emission_reason"] = "flow_timeout"
            vector["flow_key"] = entry.flow_key
            vectors.append(vector)
        return vectors


class TumblingWindowTrigger:
    def __init__(self, source_aggregate: SourceAggregateTable, dns_aggregate: DNSAggregateTable):
        self.source_aggregate = source_aggregate
        self.dns_aggregate = dns_aggregate

    def check(self, now: float) -> list[dict[str, Any]]:
        vectors: list[dict[str, Any]] = []

        for source_ip, snap in self.source_aggregate.snapshot_all().items():
            snap["emission_reason"] = "tumbling_window"
            snap["aggregate_type"] = "source"
            snap["source_ip"] = source_ip
            vectors.append(snap)

        for client_ip, snap in self.dns_aggregate.snapshot_all().items():
            snap["emission_reason"] = "tumbling_window"
            snap["aggregate_type"] = "dns"
            snap["client_ip"] = client_ip
            vectors.append(snap)

        # Reset only after both snapshots are captured -- resetting
        # between the two snapshot_all() calls would be harmless here
        # (they're independent tables), but resetting before either
        # would silently emit empty vectors instead of real data.
        self.source_aggregate.reset_all()
        self.dns_aggregate.reset_all()
        return vectors


class EmissionScheduler:
    """Opportunistic wall-clock scheduler. Call maybe_emit(now) on
    every packet arrival; each trigger only actually runs once its own
    interval has elapsed since it last ran.
    """

    def __init__(
        self,
        flow_trigger: FlowTimeoutTrigger,
        window_trigger: TumblingWindowTrigger,
        flow_check_interval_seconds: float = 1.0,
        window_interval_seconds: float = 1.0,
    ):
        self.flow_trigger = flow_trigger
        self.window_trigger = window_trigger
        self.flow_check_interval_seconds = flow_check_interval_seconds
        self.window_interval_seconds = window_interval_seconds
        self._last_flow_check: float | None = None
        self._last_window_check: float | None = None

    def maybe_emit(self, now: float) -> list[dict[str, Any]]:
        emitted: list[dict[str, Any]] = []

        if (
            self._last_flow_check is None
            or now - self._last_flow_check >= self.flow_check_interval_seconds
        ):
            emitted.extend(self.flow_trigger.check(now))
            self._last_flow_check = now

        if (
            self._last_window_check is None
            or now - self._last_window_check >= self.window_interval_seconds
        ):
            emitted.extend(self.window_trigger.check(now))
            self._last_window_check = now

        return emitted
