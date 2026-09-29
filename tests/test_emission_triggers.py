"""Tests for emission/triggers.py.

Trigger logic itself is tested with explicit `now` values (same
pattern as flow_state's existing tests) -- fully deterministic, no
sleeping needed. Only the final pipeline-level demonstration uses real
small time deltas, matching the precedent set by the Welford IAT demo.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from passive_ids.emission.triggers import (
    EmissionScheduler,
    FlowTimeoutTrigger,
    TumblingWindowTrigger,
)
from passive_ids.flow_state.confirmed_table import ConfirmedFlowTable
from passive_ids.flow_state.dns_aggregate import DNSAggregateTable
from passive_ids.flow_state.flow_key import FlowKey
from passive_ids.flow_state.source_aggregate import SourceAggregateTable


def _key(port=1):
    return FlowKey(src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=port, dst_port=443, protocol="TCP")


# ---------------------------------------------------------------------------
# FlowTimeoutTrigger
# ---------------------------------------------------------------------------

def test_idle_flow_gets_emitted_and_removed():
    table = ConfirmedFlowTable()
    table.insert_or_update(_key(), timestamp=0.0, size=60)
    trigger = FlowTimeoutTrigger(table, inactive_timeout_seconds=30.0, active_timeout_seconds=1800.0)

    # not idle enough yet
    assert trigger.check(now=10.0) == []
    assert _key() in table

    # now past the inactive threshold
    emitted = trigger.check(now=31.0)
    assert len(emitted) == 1
    assert emitted[0]["emission_reason"] == "flow_timeout"
    assert emitted[0]["flow_key"] == _key()
    assert _key() not in table  # removed after emission


def test_active_flow_within_inactive_timeout_is_not_emitted():
    table = ConfirmedFlowTable()
    table.insert_or_update(_key(), timestamp=0.0, size=60)
    table.insert_or_update(_key(), timestamp=5.0, size=60)  # still active
    trigger = FlowTimeoutTrigger(table, inactive_timeout_seconds=30.0, active_timeout_seconds=1800.0)

    assert trigger.check(now=10.0) == []
    assert _key() in table


def test_long_lived_flow_emitted_via_active_timeout_even_if_not_idle():
    table = ConfirmedFlowTable()
    table.insert_or_update(_key(), timestamp=0.0, size=60)
    # kept receiving packets right up to "now" -- idle_for is ~0, but
    # the flow has been running longer than active_timeout (1800s)
    table.insert_or_update(_key(), timestamp=1800.5, size=60)
    trigger = FlowTimeoutTrigger(table, inactive_timeout_seconds=30.0, active_timeout_seconds=1800.0)

    emitted = trigger.check(now=1801.0)
    assert len(emitted) == 1
    assert _key() not in table


def test_multiple_flows_only_expired_ones_removed():
    table = ConfirmedFlowTable()
    stale_key, fresh_key = _key(1), _key(2)
    table.insert_or_update(stale_key, timestamp=0.0, size=60)
    table.insert_or_update(fresh_key, timestamp=50.0, size=60)
    trigger = FlowTimeoutTrigger(table, inactive_timeout_seconds=30.0, active_timeout_seconds=1800.0)

    emitted = trigger.check(now=51.0)
    assert len(emitted) == 1
    assert emitted[0]["flow_key"] == stale_key
    assert stale_key not in table
    assert fresh_key in table


# ---------------------------------------------------------------------------
# TumblingWindowTrigger
# ---------------------------------------------------------------------------

def test_window_trigger_emits_all_sources_then_resets():
    source_agg = SourceAggregateTable()
    dns_agg = DNSAggregateTable()
    source_agg.update(source_ip="10.0.0.5", dest_ip="1.2.3.4", dest_port=443, packet_size=100)

    trigger = TumblingWindowTrigger(source_agg, dns_agg)
    emitted = trigger.check(now=1.0)

    assert len(emitted) == 1
    assert emitted[0]["aggregate_type"] == "source"
    assert emitted[0]["source_ip"] == "10.0.0.5"
    assert emitted[0]["packet_count"] == 1

    # reset must have happened -- table is empty afterward
    assert len(source_agg) == 0
    assert source_agg.snapshot("10.0.0.5") is None


def test_window_trigger_captures_data_before_resetting_not_after():
    """The bug this must never regress to: snapshotting AFTER reset
    would silently emit empty vectors instead of real data."""
    source_agg = SourceAggregateTable()
    dns_agg = DNSAggregateTable()
    for _ in range(7):
        source_agg.update(source_ip="10.0.0.9", dest_ip="1.2.3.4", dest_port=80, packet_size=50)

    trigger = TumblingWindowTrigger(source_agg, dns_agg)
    emitted = trigger.check(now=1.0)

    assert emitted[0]["packet_count"] == 7  # real data, not post-reset zeros


def test_window_trigger_emits_both_source_and_dns_aggregates():
    from passive_ids.capture.dns_parser import DNSQuestion, ParsedDNSMessage

    source_agg = SourceAggregateTable()
    dns_agg = DNSAggregateTable()
    source_agg.update(source_ip="10.0.0.5", dest_ip="1.2.3.4", dest_port=443, packet_size=100)
    dns_agg.observe(
        src_ip="10.0.0.5", dst_ip="8.8.8.8",
        dns_msg=ParsedDNSMessage(
            transaction_id=1, is_response=False, rcode=0, rcode_name="NOERROR",
            questions=[DNSQuestion(qname="example.com", qtype="A")],
        ),
    )

    trigger = TumblingWindowTrigger(source_agg, dns_agg)
    emitted = trigger.check(now=1.0)

    types = {v["aggregate_type"] for v in emitted}
    assert types == {"source", "dns"}


# ---------------------------------------------------------------------------
# EmissionScheduler -- opportunistic timing
# ---------------------------------------------------------------------------

def test_scheduler_fires_immediately_on_first_call():
    table = ConfirmedFlowTable()
    source_agg = SourceAggregateTable()
    dns_agg = DNSAggregateTable()
    source_agg.update(source_ip="10.0.0.5", dest_ip="1.2.3.4", dest_port=443, packet_size=100)

    scheduler = EmissionScheduler(
        FlowTimeoutTrigger(table), TumblingWindowTrigger(source_agg, dns_agg),
        flow_check_interval_seconds=1.0, window_interval_seconds=1.0,
    )
    emitted = scheduler.maybe_emit(now=0.0)
    assert len(emitted) == 1  # the one source aggregate entry


def test_scheduler_does_not_refire_before_interval_elapses():
    table = ConfirmedFlowTable()
    source_agg = SourceAggregateTable()
    dns_agg = DNSAggregateTable()

    scheduler = EmissionScheduler(
        FlowTimeoutTrigger(table), TumblingWindowTrigger(source_agg, dns_agg),
        flow_check_interval_seconds=1.0, window_interval_seconds=1.0,
    )
    scheduler.maybe_emit(now=0.0)
    source_agg.update(source_ip="10.0.0.5", dest_ip="1.2.3.4", dest_port=443, packet_size=100)

    # only 0.3s later -- window interval (1.0s) hasn't elapsed yet
    assert scheduler.maybe_emit(now=0.3) == []

    # now past the interval -- should fire and include the new data
    emitted = scheduler.maybe_emit(now=1.1)
    assert len(emitted) == 1


def test_scheduler_runs_flow_and_window_triggers_on_independent_schedules():
    table = ConfirmedFlowTable()
    source_agg = SourceAggregateTable()
    dns_agg = DNSAggregateTable()
    table.insert_or_update(_key(), timestamp=0.0, size=60)

    scheduler = EmissionScheduler(
        FlowTimeoutTrigger(table, inactive_timeout_seconds=5.0, active_timeout_seconds=1800.0),
        TumblingWindowTrigger(source_agg, dns_agg),
        flow_check_interval_seconds=10.0,  # deliberately much longer than window
        window_interval_seconds=1.0,
    )
    scheduler.maybe_emit(now=0.0)

    # window interval elapsed, but flow-check interval has not -- flow
    # trigger must not run yet even though the flow itself is idle enough
    source_agg.update(source_ip="10.0.0.5", dest_ip="1.2.3.4", dest_port=443, packet_size=50)
    emitted = scheduler.maybe_emit(now=2.0)
    assert all(v.get("emission_reason") != "flow_timeout" for v in emitted)
    assert _key() in table  # untouched -- flow trigger hasn't run yet
