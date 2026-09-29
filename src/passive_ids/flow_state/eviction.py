"""Shared eviction/degrade policy for capacity-bounded tables.

Two tables in this project face the same problem independently --
flow_state.confirmed_table.ConfirmedFlowTable here, and
flow_state.source_aggregate.SourceAggregateTable -- a flood is itself
a resource-exhaustion attempt against the detector, so a fixed
-capacity table needs a defined policy for what happens when it's
full, not just an assumption that it never will be.

The policy: try to evict something genuinely idle first (protects
flows still actively exchanging traffic). If nothing is idle -- the
table is under real, sustained pressure, not just momentarily busy --
report that no safe eviction exists, so the caller can degrade rather
than silently overwrite something that's still active.
"""
from __future__ import annotations

from typing import Protocol, TypeVar


class HasIdleTime(Protocol):
    def idle_for(self, now: float) -> float: ...


K = TypeVar("K")


def find_evictable(
    ordered_entries: dict[K, HasIdleTime],
    now: float,
    idle_threshold_seconds: float,
    scan_limit: int = 100,
) -> K | None:
    """Return the key of the oldest entry that's genuinely idle, or None.

    Assumes `ordered_entries` iterates in least-recently-used-first
    order (an OrderedDict maintained that way by the caller). Only
    scans the oldest `scan_limit` entries rather than the whole table
    -- if the table is large and under pressure, scanning everything
    on every insert would itself become a bottleneck. If nothing in
    that window is idle, the caller should treat the table as under
    genuine load rather than searching further.
    """
    checked = 0
    for key, entry in ordered_entries.items():
        if checked >= scan_limit:
            break
        if entry.idle_for(now) > idle_threshold_seconds:
            return key
        checked += 1
    return None
