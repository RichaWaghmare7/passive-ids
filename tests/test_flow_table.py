"""Tests for the bounded flow table (provisional -> confirmed, with eviction).

Focus areas: the promotion handoff actually carries over count/first_seen
correctly (and doesn't double-count the promoting packet), TTL expiry
resets stale provisional entries, and capacity eviction protects active
flows while correctly degrading under genuine sustained load.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from passive_ids.flow_state.flow_key import FlowKey
from passive_ids.flow_state.flow_table import BoundedFlowTable
from passive_ids.flow_state.provisional_table import ProvisionalTable


def _key(port=1234):
    return FlowKey(src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=port, dst_port=443, protocol="TCP")


def test_flow_stays_provisional_below_threshold():
    table = BoundedFlowTable(promotion_threshold=3)
    key = _key()
    assert table.observe_packet(key, timestamp=1.0, size=60) is None
    assert table.observe_packet(key, timestamp=1.01, size=60) is None
    assert key not in table.confirmed
    assert len(table.provisional) == 1


def test_flow_promotes_on_threshold_packet():
    table = BoundedFlowTable(promotion_threshold=3)
    key = _key()
    table.observe_packet(key, timestamp=1.0, size=60)
    table.observe_packet(key, timestamp=1.01, size=60)
    entry = table.observe_packet(key, timestamp=1.02, size=60, is_syn=True)

    assert entry is not None
    assert key in table.confirmed
    assert key not in table.provisional  # removed on promotion


def test_promotion_carries_over_count_and_first_seen_without_double_counting():
    table = BoundedFlowTable(promotion_threshold=3)
    key = _key()
    table.observe_packet(key, timestamp=10.0, size=60)   # provisional pkt 1
    table.observe_packet(key, timestamp=10.1, size=60)   # provisional pkt 2
    entry = table.observe_packet(key, timestamp=10.2, size=100, is_syn=True)  # promoting pkt 3

    snap = entry.snapshot()
    # exactly 3 packets total were ever sent -- promotion must not lose
    # or double-count the promoting packet
    assert snap["total_packets_seen"] == 3
    assert snap["pre_promotion_packet_count"] == 2  # the 2 pre-promotion packets
    assert snap["packet_count"] == 1                # only the promoting packet is in full detail
    assert entry.first_seen == 10.0                 # true flow start, not the promotion moment
    assert snap["byte_count"] == 100                # only the promoting packet's size is known


def test_provisional_ttl_expiry_resets_count():
    table = BoundedFlowTable(provisional_ttl_seconds=1.0, promotion_threshold=3)
    key = _key()
    table.observe_packet(key, timestamp=1.0, size=60)
    table.observe_packet(key, timestamp=1.1, size=60)
    # gap > ttl -- the earlier 2 packets should be treated as gone
    entry = table.observe_packet(key, timestamp=5.0, size=60)
    assert entry is None  # this is only the 1st packet of a fresh provisional entry
    assert table.provisional._table[key].pkt_count == 1


def test_expire_stale_removes_only_timed_out_entries():
    ptable = ProvisionalTable(ttl_seconds=1.0, promotion_threshold=10)
    k1, k2 = _key(1), _key(2)
    ptable.observe(k1, timestamp=1.0)   # will be stale by t=5.0
    ptable.observe(k2, timestamp=4.5)   # still fresh at t=5.0

    removed = ptable.expire_stale(now=5.0)
    assert removed == 1
    assert k1 not in ptable
    assert k2 in ptable


def test_confirmed_table_updates_existing_flow_without_reconfirming():
    table = BoundedFlowTable(promotion_threshold=2)
    key = _key()
    table.observe_packet(key, timestamp=1.0, size=60)
    entry = table.observe_packet(key, timestamp=1.1, size=60)  # promotes here
    assert entry.snapshot()["packet_count"] == 1

    entry2 = table.observe_packet(key, timestamp=1.2, size=200)  # already confirmed
    assert entry2.snapshot()["packet_count"] == 2
    assert entry2.snapshot()["byte_count"] == 260


def test_capacity_eviction_protects_active_flow_evicts_idle_one():
    table = BoundedFlowTable(promotion_threshold=1, max_confirmed=2, idle_evict_threshold_seconds=1.0)
    k_idle = _key(1)
    k_active = _key(2)
    k_new = _key(3)

    table.observe_packet(k_idle, timestamp=0.0, size=60)     # confirmed at t=0
    table.observe_packet(k_active, timestamp=5.0, size=60)   # confirmed at t=5 (recently active)
    # table now at capacity (2). k_idle has been idle for 5.0s > threshold 1.0s.
    entry = table.observe_packet(k_new, timestamp=5.1, size=60)

    assert entry is not None, "should evict the idle flow and admit the new one"
    assert k_idle not in table.confirmed
    assert k_active in table.confirmed
    assert k_new in table.confirmed


def test_capacity_degrades_when_nothing_idle():
    table = BoundedFlowTable(promotion_threshold=1, max_confirmed=1, idle_evict_threshold_seconds=100.0)
    k1 = _key(1)
    k2 = _key(2)

    table.observe_packet(k1, timestamp=0.0, size=60)
    # k1 is only 0.1s idle -- well under the 100s threshold, nothing evictable
    result = table.observe_packet(k2, timestamp=0.1, size=60)

    assert result is None
    assert table.confirmed.degraded_count == 1
    assert k1 in table.confirmed  # the active flow must not have been evicted
    assert k2 not in table.confirmed


def test_confirmed_entry_tracks_iat_after_promotion():
    """Integration check: ConfirmedFlowEntry's iat_tracker (wired to
    features.welford.WelfordIAT) actually accumulates real statistics
    from packets seen after promotion, not just a placeholder."""
    table = BoundedFlowTable(promotion_threshold=2)
    key = _key()

    table.observe_packet(key, timestamp=100.0, size=60)         # provisional pkt 1
    table.observe_packet(key, timestamp=100.5, size=60)         # promotes here (pkt 2)
    table.observe_packet(key, timestamp=101.5, size=60)         # confirmed, IAT = 1.0
    entry = table.observe_packet(key, timestamp=103.5, size=60) # confirmed, IAT = 2.0

    iat_snap = entry.snapshot()["iat"]
    # First post-promotion packet establishes the baseline (no IAT yet),
    # so only 2 real IAT observations exist: 1.0 and 2.0.
    assert iat_snap["n"] == 2
    assert abs(iat_snap["mean"] - 1.5) < 1e-9


def test_bidirectional_packets_merge_into_one_confirmed_entry():
    """The specific bug this fixes, verified directly before the fix
    existed: a client->server packet and the server's response used to
    silently create two unrelated ConfirmedFlowEntry objects."""
    table = BoundedFlowTable(promotion_threshold=1)
    fwd_key = FlowKey(src_ip="10.0.0.5", dst_ip="10.0.0.9", src_port=5000, dst_port=443, protocol="TCP")
    rev_key = FlowKey(src_ip="10.0.0.9", dst_ip="10.0.0.5", src_port=443, dst_port=5000, protocol="TCP")

    table.observe_packet(fwd_key, timestamp=0.0, size=40)
    table.observe_packet(rev_key, timestamp=0.1, size=1200)

    assert len(table.confirmed) == 1


def test_directional_byte_counts_attributed_correctly():
    table = BoundedFlowTable(promotion_threshold=1)
    fwd_key = FlowKey(src_ip="10.0.0.5", dst_ip="10.0.0.9", src_port=5000, dst_port=443, protocol="TCP")
    rev_key = FlowKey(src_ip="10.0.0.9", dst_ip="10.0.0.5", src_port=443, dst_port=5000, protocol="TCP")

    entry = table.observe_packet(fwd_key, timestamp=0.0, size=40)
    entry = table.observe_packet(rev_key, timestamp=0.1, size=1200)
    entry = table.observe_packet(rev_key, timestamp=0.2, size=1200)

    snap = entry.snapshot()
    assert snap["fwd_byte_count"] == 40
    assert snap["bwd_byte_count"] == 2400
    assert abs(snap["byte_ratio_out"] - (40 / 2440)) < 1e-9


def test_snapshot_ratio_defaults_to_half_with_no_directional_data():
    """0.5, not 0.0 or NaN, when there's genuinely no directional data
    yet (e.g. a flow confirmed entirely from pre-promotion packets
    whose direction detail was never recorded)."""
    table = BoundedFlowTable(promotion_threshold=3)
    key = _key()
    table.observe_packet(key, timestamp=0.0, size=60)
    table.observe_packet(key, timestamp=0.1, size=60)
    entry = table.observe_packet(key, timestamp=0.2, size=60)  # promotes

    # the promoting packet itself IS forward-direction, so this isn't
    # actually a zero-data case -- covered for completeness elsewhere;
    # the true zero-data case is checked via direct entry construction:
    from passive_ids.flow_state.confirmed_table import ConfirmedFlowEntry
    fresh = ConfirmedFlowEntry(flow_key=key, first_seen=0.0)
    assert fresh.snapshot()["byte_ratio_out"] == 0.5
    assert fresh.snapshot()["packet_ratio_out"] == 0.5


def test_reverse_direction_packet_promotes_same_flow_as_forward_ones():
    """The promotion-delay bug this fix also resolves: a flow that
    splits evenly between directions used to need 3 packets in EACH
    direction to promote (since each direction was tracked separately).
    Now 2 forward + 1 reverse correctly reaches promotion_threshold=3
    as one flow."""
    table = BoundedFlowTable(promotion_threshold=3)
    fwd_key = FlowKey(src_ip="10.0.0.5", dst_ip="10.0.0.9", src_port=5000, dst_port=443, protocol="TCP")
    rev_key = FlowKey(src_ip="10.0.0.9", dst_ip="10.0.0.5", src_port=443, dst_port=5000, protocol="TCP")

    assert table.observe_packet(fwd_key, timestamp=0.0, size=40) is None
    assert table.observe_packet(rev_key, timestamp=0.1, size=1200) is None
    entry = table.observe_packet(fwd_key, timestamp=0.2, size=40)  # 3rd packet overall
    assert entry is not None
