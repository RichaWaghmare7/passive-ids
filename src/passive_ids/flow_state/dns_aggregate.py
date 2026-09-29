"""Always-on per-source-IP DNS behavior aggregate, for DGA detection.

Tracks the per-source signals DGA detection actually needs: query
volume, distinct-domain cardinality, NXDOMAIN rate, running domain
-entropy statistics, query-type diversity, and fast-flux (low TTL)
signal. Bounded memory with the same LRU-eviction pattern as
flow_state.source_aggregate.SourceAggregateTable, for the same
reason: a DGA-infected host can generate thousands of queries per
minute, which is itself a resource-exhaustion risk against this table
-- the same problem a volumetric flood poses to the confirmed flow
table, one layer up.

Ties together every primitive built earlier in this project:
  - HyperLogLog (already used in source_aggregate) for distinct-domain
    cardinality without storing every domain string ever queried.
  - features.welford.WelfordStats for running domain-entropy mean/
    variance. Per features.entropy's disclosed finding, any single
    dictionary-based DGA domain may not read as anomalous alone --
    but a source whose *running average* entropy sits well outside
    normal browsing's range is a different, and stronger, signal than
    judging one domain in isolation.
  - features.entropy.StreamingCategoricalEntropy for query-type
    (A/AAAA/TXT/...) diversity.

Client-IP resolution, the one subtlety here: a DNS *query* packet's
src_ip is the client, but a DNS *response* packet's src_ip is the
resolving server -- the client being profiled is the response's
dst_ip. Getting this backwards would silently aggregate NXDOMAIN/TTL
behavior onto the DNS server's own address instead of the infected
host's, for every response-side observation.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from datasketch import HyperLogLog

from passive_ids.capture.dns_parser import ParsedDNSMessage
from passive_ids.features.entropy import StreamingCategoricalEntropy, string_char_entropy
from passive_ids.features.welford import WelfordStats


@dataclass
class DNSAggregateEntry:
    hll_precision: int = 12

    domain_hll: HyperLogLog = field(init=False)
    query_type_entropy: StreamingCategoricalEntropy = field(
        default_factory=StreamingCategoricalEntropy, init=False
    )
    domain_entropy_stats: WelfordStats = field(default_factory=WelfordStats, init=False)
    ttl_stats: WelfordStats = field(default_factory=WelfordStats, init=False)
    query_count: int = 0
    response_count: int = 0
    nxdomain_count: int = 0

    def __post_init__(self) -> None:
        self.domain_hll = HyperLogLog(p=self.hll_precision)

    def observe_query(self, dns_msg: ParsedDNSMessage) -> None:
        self.query_count += 1
        for q in dns_msg.questions:
            self.domain_hll.update(q.qname.encode("utf-8"))
            self.query_type_entropy.observe(q.qtype)
            # First label only, same TLD-stripping convention used
            # everywhere else domain entropy is computed in this
            # project -- a simple split, not full public-suffix-aware
            # parsing (doesn't handle multi-label TLDs like "co.uk"
            # correctly, disclosed rather than silently assumed exact).
            label = q.qname.split(".")[0]
            self.domain_entropy_stats.update(string_char_entropy(label))

    def observe_response(self, dns_msg: ParsedDNSMessage) -> None:
        self.response_count += 1
        if dns_msg.is_nxdomain:
            self.nxdomain_count += 1
        ttl = dns_msg.min_answer_ttl
        if ttl is not None:
            self.ttl_stats.update(ttl)

    def snapshot(self) -> dict[str, Any]:
        return {
            "unique_domains_est": self.domain_hll.count(),
            "query_count": self.query_count,
            "response_count": self.response_count,
            "nxdomain_count": self.nxdomain_count,
            "nxdomain_rate": (
                self.nxdomain_count / self.response_count if self.response_count else 0.0
            ),
            "query_type_entropy": self.query_type_entropy.entropy(),
            "domain_entropy": self.domain_entropy_stats.snapshot(),
            "ttl": self.ttl_stats.snapshot(),
        }


class DNSAggregateTable:
    """Bounded collection of per-client-IP DNS aggregates, LRU-evicted --
    same pattern as flow_state.source_aggregate.SourceAggregateTable,
    for the same resource-exhaustion reason.
    """

    def __init__(self, max_tracked_sources: int = 1_000_000, hll_precision: int = 12):
        self.max_tracked_sources = max_tracked_sources
        self._hll_precision = hll_precision
        self._table: OrderedDict[str, DNSAggregateEntry] = OrderedDict()
        self.evicted_count = 0

    def __len__(self) -> int:
        return len(self._table)

    def _get_or_create(self, client_ip: str) -> DNSAggregateEntry:
        if client_ip in self._table:
            self._table.move_to_end(client_ip)
            return self._table[client_ip]
        if len(self._table) >= self.max_tracked_sources:
            self._table.popitem(last=False)
            self.evicted_count += 1
        entry = DNSAggregateEntry(hll_precision=self._hll_precision)
        self._table[client_ip] = entry
        return entry

    def observe(self, src_ip: str, dst_ip: str, dns_msg: ParsedDNSMessage) -> None:
        """src_ip/dst_ip are the packet's IP-layer addresses, not
        necessarily the DNS client -- see module docstring for why
        the client-IP resolution differs between queries and responses.
        """
        client_ip = dst_ip if dns_msg.is_response else src_ip
        entry = self._get_or_create(client_ip)
        if dns_msg.is_response:
            entry.observe_response(dns_msg)
        else:
            entry.observe_query(dns_msg)

    def snapshot(self, client_ip: str) -> dict[str, Any] | None:
        entry = self._table.get(client_ip)
        return entry.snapshot() if entry else None

    def snapshot_all(self) -> dict[str, dict[str, Any]]:
        """All tracked clients' current snapshots -- same rationale as
        SourceAggregateTable.snapshot_all, called immediately before
        reset_all() by the tumbling-window trigger."""
        return {client_ip: entry.snapshot() for client_ip, entry in self._table.items()}

    def reset_all(self) -> None:
        """Periodic maintenance, same rationale as
        SourceAggregateTable.reset_all: without a periodic reset,
        cardinality and rate signals accumulate over the client's
        entire observed lifetime rather than reflecting recent
        behavior."""
        self._table.clear()
