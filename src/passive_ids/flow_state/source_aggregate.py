"""Always-on per-source-IP aggregate: HyperLogLog + Count-Min Sketch.

This is the "unbounded lane" from the architecture diagrams -- it must
update on every packet regardless of whether the confirmed 5-tuple
flow table has capacity, because it's what DDoS and port-scan
detection actually read from, and a flood is exactly the moment the
confirmed table is under the most pressure.

One honest correction to how that lane was originally described,
made explicit here rather than left implicit in the code: memory
*per tracked source* is genuinely O(1) regardless of how many distinct
destinations or packets that source generates -- a HyperLogLog
register array and a Count-Min Sketch counter grid are both fixed size
by construction. But the *number of distinct source IPs being
tracked* is not bounded by that same property. A spoofed-source flood
that varies the source IP on every packet would grow this table
without limit if nothing capped it -- which is the same
resource-exhaustion problem the confirmed flow table already solves,
just one level up. SourceAggregateTable below applies the identical
fix: a capacity cap with LRU eviction, so total memory is bounded even
though evicting a source does mean losing its accumulated aggregate
state early.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import mmh3
from datasketch import HyperLogLog

from passive_ids.features.entropy import StreamingCategoricalEntropy


class CountMinSketch:
    """Frequency estimator with fixed memory regardless of key cardinality.

    Used here for per-source destination-port frequency (e.g. "how many
    times has this source hit port 445") -- a signal port-scan and DDoS
    detection both use, without needing to store a full histogram.

    Estimates are biased high, never low (a hash collision can only
    inflate a count, never deflate it) -- callers should treat
    `estimate()` as an upper bound, not an exact count.
    """

    def __init__(self, width: int = 2048, depth: int = 4):
        self.width = width
        self.depth = depth
        self.table: list[list[int]] = [[0] * width for _ in range(depth)]

    def _indices(self, key: Any) -> list[int]:
        key_str = str(key)
        return [mmh3.hash(key_str, seed) % self.width for seed in range(self.depth)]

    def add(self, key: Any, count: int = 1) -> None:
        for row, idx in enumerate(self._indices(key)):
            self.table[row][idx] += count

    def estimate(self, key: Any) -> int:
        return min(self.table[row][idx] for row, idx in enumerate(self._indices(key)))


@dataclass
class SourceAggregateEntry:
    """Per-source-IP state. Fixed memory footprint regardless of traffic volume."""

    hll_precision: int = 12
    cms_width: int = 2048
    cms_depth: int = 4

    dest_ip_hll: HyperLogLog = field(init=False)
    dest_port_hll: HyperLogLog = field(init=False)
    dest_port_cms: CountMinSketch = field(init=False)
    protocol_entropy: StreamingCategoricalEntropy = field(
        default_factory=StreamingCategoricalEntropy, init=False
    )
    dest_port_entropy: StreamingCategoricalEntropy = field(
        default_factory=StreamingCategoricalEntropy, init=False
    )
    packet_count: int = 0
    byte_count: int = 0
    syn_count: int = 0

    def __post_init__(self) -> None:
        self.dest_ip_hll = HyperLogLog(p=self.hll_precision)
        self.dest_port_hll = HyperLogLog(p=self.hll_precision)
        self.dest_port_cms = CountMinSketch(width=self.cms_width, depth=self.cms_depth)

    def update(
        self,
        dest_ip: str,
        dest_port: int,
        packet_size: int,
        is_syn: bool = False,
        protocol: str = "OTHER",
    ) -> None:
        self.dest_ip_hll.update(str(dest_ip).encode("utf-8"))
        self.dest_port_hll.update(str(dest_port).encode("utf-8"))
        self.dest_port_cms.add(dest_port)
        self.protocol_entropy.observe(protocol)
        self.dest_port_entropy.observe(dest_port)
        self.packet_count += 1
        self.byte_count += packet_size
        if is_syn:
            self.syn_count += 1

    def port_frequency(self, dest_port: int) -> int:
        return self.dest_port_cms.estimate(dest_port)

    def snapshot(self) -> dict[str, Any]:
        """Feature vector fragment fed to the DDoS and port-scan models.

        unique_dest_ips_est / unique_dest_ports_est are HLL estimates,
        not exact counts -- typical error is ~1-2% at p=12 (4096
        registers). That's an acceptable tradeoff against the
        alternative of storing every distinct destination seen.

        protocol_entropy is genuinely bounded-cardinality (TCP/UDP/
        ICMP/OTHER -- 4 categories), which is exactly the case
        StreamingCategoricalEntropy is built for, unlike the
        destination IP/port cardinality above. A single-protocol
        flood (pure UDP amplification, pure ICMP) drives this toward
        0.0; normal mixed traffic keeps it well above zero.

        dest_port_entropy is the same primitive applied to a larger
        but still genuinely bounded space: destination ports are
        capped at 65536 by the 16-bit port field, unlike domain names
        or arbitrary strings, which is what makes this an appropriate
        (if larger) use of StreamingCategoricalEntropy rather than the
        HLL/CM sketches. It captures something cardinality alone
        cannot: a source hitting 50 distinct ports once each (high
        entropy, true horizontal scan) reads very differently from one
        hitting the same 2 ports repeatedly at very uneven rates (low
        entropy), even when unique_dest_ports_est can't tell them
        apart on its own.
        """
        return {
            "unique_dest_ips_est": self.dest_ip_hll.count(),
            "unique_dest_ports_est": self.dest_port_hll.count(),
            "dest_port_entropy": self.dest_port_entropy.entropy(),
            "packet_count": self.packet_count,
            "byte_count": self.byte_count,
            "syn_count": self.syn_count,
            "syn_ratio": (self.syn_count / self.packet_count) if self.packet_count else 0.0,
            "protocol_entropy": self.protocol_entropy.entropy(),
        }


class SourceAggregateTable:
    """Bounded collection of per-source-IP aggregates, LRU-evicted.

    See module docstring: this is the correction to "unbounded lane" --
    each entry is O(1) memory, but the table itself caps total tracked
    sources so a source-spoofing flood can't grow it without limit.
    """

    def __init__(
        self,
        max_tracked_sources: int = 1_000_000,
        hll_precision: int = 12,
        cms_width: int = 2048,
        cms_depth: int = 4,
    ):
        self.max_tracked_sources = max_tracked_sources
        self._hll_precision = hll_precision
        self._cms_width = cms_width
        self._cms_depth = cms_depth
        self._table: OrderedDict[str, SourceAggregateEntry] = OrderedDict()
        self.evicted_count = 0  # exposed for monitoring -- a rising rate
        # under normal traffic means max_tracked_sources is too low for
        # the actual number of distinct sources on this network.

    def __len__(self) -> int:
        return len(self._table)

    def _get_or_create(self, source_ip: str) -> SourceAggregateEntry:
        if source_ip in self._table:
            self._table.move_to_end(source_ip)
            return self._table[source_ip]
        if len(self._table) >= self.max_tracked_sources:
            self._table.popitem(last=False)  # evict least-recently-used source
            self.evicted_count += 1
        entry = SourceAggregateEntry(
            hll_precision=self._hll_precision,
            cms_width=self._cms_width,
            cms_depth=self._cms_depth,
        )
        self._table[source_ip] = entry
        return entry

    def update(
        self,
        source_ip: str,
        dest_ip: str,
        dest_port: int,
        packet_size: int,
        is_syn: bool = False,
        protocol: str = "OTHER",
    ) -> None:
        entry = self._get_or_create(source_ip)
        entry.update(dest_ip, dest_port, packet_size, is_syn, protocol)

    def snapshot(self, source_ip: str) -> dict[str, Any] | None:
        entry = self._table.get(source_ip)
        return entry.snapshot() if entry else None

    def snapshot_all(self) -> dict[str, dict[str, Any]]:
        """All tracked sources' current snapshots, keyed by source IP.
        Used by emission.triggers' tumbling-window trigger, which calls
        this immediately before reset_all() -- snapshotting after reset
        would just return empty data for everything.
        """
        return {source_ip: entry.snapshot() for source_ip, entry in self._table.items()}

    def port_frequency(self, source_ip: str, dest_port: int) -> int:
        entry = self._table.get(source_ip)
        return entry.port_frequency(dest_port) if entry else 0

    def reset_all(self) -> None:
        """Called by emission/triggers.py on each tumbling-window tick.

        Without a periodic reset, unique-destination counts accumulate
        over the source's entire observed lifetime rather than reflecting
        recent behavior -- which would dilute a burst of scanning
        activity into an unremarkable all-time average. Per-window
        aggregates are what the fan-out/DDoS features actually need.
        """
        self._table.clear()
