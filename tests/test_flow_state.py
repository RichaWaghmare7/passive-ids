"""Tests for flow_state/source_aggregate.py.

These check the properties the module docstrings claim, not just that
the code runs: cardinality estimates are close to true values, Count-Min
never underestimates, eviction actually caps memory, and reset_all
actually clears accumulated state.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from passive_ids.flow_state.source_aggregate import (
    CountMinSketch,
    SourceAggregateEntry,
    SourceAggregateTable,
)


def test_hll_cardinality_close_to_true_count():
    entry = SourceAggregateEntry(hll_precision=12)
    true_unique_ips = 500
    for i in range(true_unique_ips):
        entry.update(dest_ip=f"10.0.{i // 256}.{i % 256}", dest_port=443, packet_size=100)
    # duplicates -- should not move the estimate much
    for i in range(true_unique_ips):
        entry.update(dest_ip=f"10.0.{i // 256}.{i % 256}", dest_port=443, packet_size=100)

    est = entry.snapshot()["unique_dest_ips_est"]
    error_pct = abs(est - true_unique_ips) / true_unique_ips
    assert error_pct < 0.05, f"HLL estimate {est} too far from true {true_unique_ips}"


def test_cms_never_underestimates():
    cms = CountMinSketch(width=64, depth=4)  # deliberately small -> forces collisions
    true_counts = {"port_80": 1000, "port_443": 500, "port_22": 3}
    for key, count in true_counts.items():
        cms.add(key, count)

    for key, true_count in true_counts.items():
        est = cms.estimate(key)
        assert est >= true_count, (
            f"CMS estimate {est} for {key} is below true count {true_count} -- "
            "Count-Min must never underestimate"
        )


def test_syn_ratio_computed_correctly():
    entry = SourceAggregateEntry()
    for _ in range(8):
        entry.update(dest_ip="10.0.0.1", dest_port=443, packet_size=60, is_syn=True)
    for _ in range(2):
        entry.update(dest_ip="10.0.0.1", dest_port=443, packet_size=1200, is_syn=False)

    snap = entry.snapshot()
    assert snap["packet_count"] == 10
    assert snap["syn_count"] == 8
    assert snap["syn_ratio"] == 0.8


def test_table_eviction_caps_memory():
    table = SourceAggregateTable(max_tracked_sources=3)
    for i in range(5):
        table.update(source_ip=f"192.168.1.{i}", dest_ip="10.0.0.1", dest_port=80, packet_size=100)

    assert len(table) == 3, "table should never exceed max_tracked_sources"
    assert table.evicted_count == 2, "exactly 2 sources should have been evicted"
    # earliest sources evicted first (LRU)
    assert table.snapshot("192.168.1.0") is None
    assert table.snapshot("192.168.1.1") is None
    assert table.snapshot("192.168.1.4") is not None


def test_lru_touch_prevents_eviction_of_active_source():
    table = SourceAggregateTable(max_tracked_sources=2)
    table.update("A", "10.0.0.1", 80, 100)
    table.update("B", "10.0.0.1", 80, 100)
    table.update("A", "10.0.0.2", 80, 100)  # touch A again -> A is now most-recent
    table.update("C", "10.0.0.1", 80, 100)  # should evict B, not A

    assert table.snapshot("A") is not None
    assert table.snapshot("B") is None
    assert table.snapshot("C") is not None


def test_reset_all_clears_state():
    table = SourceAggregateTable()
    table.update("A", "10.0.0.1", 80, 100)
    assert len(table) == 1
    table.reset_all()
    assert len(table) == 0
    assert table.snapshot("A") is None


def test_port_frequency_tracks_repeated_hits():
    entry = SourceAggregateEntry()
    for _ in range(50):
        entry.update(dest_ip="10.0.0.1", dest_port=445, packet_size=60)
    entry.update(dest_ip="10.0.0.1", dest_port=22, packet_size=60)

    assert entry.port_frequency(445) >= 50
    assert entry.port_frequency(9999) < 50  # never-seen port should be low (0, modulo collisions)


def test_protocol_entropy_low_for_single_protocol_flood_high_for_mixed():
    flood_entry = SourceAggregateEntry()
    for i in range(50):
        flood_entry.update(dest_ip=f"10.0.0.{i % 5}", dest_port=80, packet_size=60, protocol="UDP")

    normal_entry = SourceAggregateEntry()
    for proto, n in [("TCP", 30), ("UDP", 10), ("ICMP", 2)]:
        for _ in range(n):
            normal_entry.update(dest_ip="10.0.0.1", dest_port=443, packet_size=60, protocol=proto)

    flood_snap = flood_entry.snapshot()
    normal_snap = normal_entry.snapshot()

    assert flood_snap["protocol_entropy"] == 0.0
    assert normal_snap["protocol_entropy"] > 0.0


def test_dest_port_entropy_distinguishes_true_scan_from_uneven_repeat():
    """The specific case cardinality alone can't tell apart: 50 distinct
    ports hit once each (true horizontal scan) vs. 2 ports hit at very
    uneven rates -- both could have similar unique_dest_ports_est in
    principle, but only entropy captures the distribution shape."""
    true_scan = SourceAggregateEntry()
    for port in range(50):
        true_scan.update(dest_ip="10.0.1.1", dest_port=port, packet_size=60, protocol="TCP")

    uneven_repeat = SourceAggregateEntry()
    for _ in range(45):
        uneven_repeat.update(dest_ip="10.0.1.1", dest_port=80, packet_size=60, protocol="TCP")
    for _ in range(5):
        uneven_repeat.update(dest_ip="10.0.1.1", dest_port=443, packet_size=60, protocol="TCP")

    scan_snap = true_scan.snapshot()
    repeat_snap = uneven_repeat.snapshot()

    assert scan_snap["dest_port_entropy"] > repeat_snap["dest_port_entropy"]
    assert repeat_snap["dest_port_entropy"] > 0.0  # still 2 distinct ports, not zero
