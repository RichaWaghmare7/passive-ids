"""Tests for flow_state/dns_aggregate.py.

The client-IP resolution (query src_ip vs response dst_ip) gets
tested most directly, since getting it backwards silently attributes
DNS server behavior to the wrong host without ever raising an error.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from passive_ids.capture.dns_parser import DNSAnswer, DNSQuestion, ParsedDNSMessage
from passive_ids.flow_state.dns_aggregate import DNSAggregateTable


def _query(qname="example.com", qtype="A", txid=1):
    return ParsedDNSMessage(
        transaction_id=txid, is_response=False, rcode=0, rcode_name="NOERROR",
        questions=[DNSQuestion(qname=qname, qtype=qtype)],
    )


def _response(qname="example.com", rcode=0, ttl=300, txid=1):
    if rcode == 0:
        answers = [DNSAnswer(rrname=qname, rtype="A", ttl=ttl, rdata="1.2.3.4")]
    else:
        answers = []
    rcode_name = "NXDOMAIN" if rcode == 3 else "NOERROR"
    return ParsedDNSMessage(
        transaction_id=txid, is_response=True, rcode=rcode, rcode_name=rcode_name,
        questions=[DNSQuestion(qname=qname, qtype="A")], answers=answers,
    )


def test_query_attributes_to_src_ip_not_dst_ip():
    table = DNSAggregateTable()
    # client 10.0.0.5 queries resolver 8.8.8.8
    table.observe(src_ip="10.0.0.5", dst_ip="8.8.8.8", dns_msg=_query())

    assert table.snapshot("10.0.0.5")["query_count"] == 1
    assert table.snapshot("8.8.8.8") is None  # server must not get an entry


def test_response_attributes_to_dst_ip_not_src_ip():
    table = DNSAggregateTable()
    # resolver 8.8.8.8 responds to client 10.0.0.5 -- src/dst are reversed
    # relative to the query, exactly the case that would silently
    # misattribute if the client-IP resolution were wrong.
    table.observe(src_ip="8.8.8.8", dst_ip="10.0.0.5", dns_msg=_response())

    assert table.snapshot("10.0.0.5")["response_count"] == 1
    assert table.snapshot("8.8.8.8") is None  # server must not get an entry


def test_query_then_response_accumulate_on_same_client():
    table = DNSAggregateTable()
    table.observe(src_ip="10.0.0.5", dst_ip="8.8.8.8", dns_msg=_query())
    table.observe(src_ip="8.8.8.8", dst_ip="10.0.0.5", dns_msg=_response())

    snap = table.snapshot("10.0.0.5")
    assert snap["query_count"] == 1
    assert snap["response_count"] == 1


def test_nxdomain_rate_computed_correctly():
    table = DNSAggregateTable()
    for _ in range(3):
        table.observe("8.8.8.8", "10.0.0.9", _response(rcode=3))  # NXDOMAIN
    table.observe("8.8.8.8", "10.0.0.9", _response(rcode=0))       # resolves

    snap = table.snapshot("10.0.0.9")
    assert snap["response_count"] == 4
    assert snap["nxdomain_count"] == 3
    assert abs(snap["nxdomain_rate"] - 0.75) < 1e-9


def test_dga_like_source_vs_normal_browsing_source():
    """Realistic end-to-end comparison: a simulated DGA-infected host
    (many distinct random-looking domains, high NXDOMAIN rate, low TTL
    on rare successes) vs normal browsing (few well-known domains, no
    failures, normal TTLs)."""
    import random
    random.seed(7)

    table = DNSAggregateTable()

    # DGA-like: 40 distinct random-looking domains, most NXDOMAIN
    dga_client = "10.0.0.66"
    for i in range(40):
        domain = "".join(random.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(10)) + ".tk"
        table.observe(dga_client, "8.8.8.8", _query(qname=domain, txid=i))
        rcode = 0 if i % 10 == 0 else 3  # 1 in 10 resolves
        table.observe("8.8.8.8", dga_client, _response(qname=domain, rcode=rcode, ttl=10, txid=i))

    # Normal browsing: 3 well-known domains, repeated, always resolves
    normal_client = "10.0.0.77"
    for domain in ["google.com", "github.com", "google.com", "github.com", "google.com"]:
        table.observe(normal_client, "8.8.8.8", _query(qname=domain))
        table.observe("8.8.8.8", normal_client, _response(qname=domain, rcode=0, ttl=3600))

    dga_snap = table.snapshot(dga_client)
    normal_snap = table.snapshot(normal_client)

    assert dga_snap["nxdomain_rate"] > 0.8
    assert normal_snap["nxdomain_rate"] == 0.0

    assert dga_snap["unique_domains_est"] > 30  # ~40 distinct domains
    assert normal_snap["unique_domains_est"] < 3  # only 2 distinct domains repeated

    assert dga_snap["ttl"]["mean"] < normal_snap["ttl"]["mean"]  # fast-flux signal


def test_eviction_caps_memory_same_as_source_aggregate():
    table = DNSAggregateTable(max_tracked_sources=2)
    for i in range(4):
        table.observe(f"10.0.0.{i}", "8.8.8.8", _query())

    assert len(table) == 2
    assert table.evicted_count == 2
    assert table.snapshot("10.0.0.0") is None
    assert table.snapshot("10.0.0.3") is not None


def test_reset_all_clears_state():
    table = DNSAggregateTable()
    table.observe("10.0.0.5", "8.8.8.8", _query())
    assert len(table) == 1
    table.reset_all()
    assert len(table) == 0
