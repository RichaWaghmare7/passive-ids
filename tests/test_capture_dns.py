"""Tests for capture/dns_parser.py.

Built directly against scapy's confirmed DNS layer behavior: qname
comes back as bytes with a trailing root dot, qtype/rcode are raw
ints, and .qd/.an are always list-like (checked via probes before
writing the parser, not assumed).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from passive_ids.capture.dns_parser import DNSParser


def _dns_query(qname="xk4jq9z1.tk", qtype="A", txid=0x1234):
    from scapy.all import IP, UDP
    from scapy.layers.dns import DNS, DNSQR
    pkt = IP(src="10.0.0.1", dst="8.8.8.8") / UDP(sport=54321, dport=53) / DNS(
        id=txid, qr=0, qdcount=1, qd=DNSQR(qname=qname, qtype=qtype)
    )
    return bytes(pkt)


def _dns_response(qname="xk4jq9z1.tk", rcode=0, ttl=300, rdata="203.0.113.5", txid=0x1234):
    from scapy.all import IP, UDP
    from scapy.layers.dns import DNS, DNSQR, DNSRR
    ancount = 1 if rcode == 0 else 0
    an = DNSRR(rrname=qname, type="A", ttl=ttl, rdata=rdata) if rcode == 0 else []
    pkt = IP(src="8.8.8.8", dst="10.0.0.1") / UDP(sport=53, dport=54321) / DNS(
        id=txid, qr=1, rcode=rcode, qdcount=1, ancount=ancount,
        qd=DNSQR(qname=qname, qtype="A"), an=an,
    )
    return bytes(pkt)


def test_parses_query():
    parsed = DNSParser().parse(_dns_query(qname="google.com", txid=0xABCD))
    assert parsed is not None
    assert parsed.is_response is False
    assert parsed.transaction_id == 0xABCD
    assert len(parsed.questions) == 1
    assert parsed.questions[0].qname == "google.com"  # trailing dot stripped
    assert parsed.questions[0].qtype == "A"


def test_parses_successful_response_with_answer():
    parsed = DNSParser().parse(_dns_response(qname="google.com", rcode=0, ttl=300, rdata="1.2.3.4"))
    assert parsed.is_response is True
    assert parsed.rcode_name == "NOERROR"
    assert parsed.is_nxdomain is False
    assert len(parsed.answers) == 1
    assert parsed.answers[0].rrname == "google.com"
    assert parsed.answers[0].ttl == 300
    assert parsed.answers[0].rdata == "1.2.3.4"
    assert parsed.min_answer_ttl == 300


def test_parses_nxdomain_response_with_no_answers():
    parsed = DNSParser().parse(_dns_response(qname="qzx7f2mvk.com", rcode=3))
    assert parsed.is_response is True
    assert parsed.rcode_name == "NXDOMAIN"
    assert parsed.is_nxdomain is True
    assert parsed.answers == []
    assert parsed.min_answer_ttl is None  # nothing to compute a min TTL over


def test_low_ttl_is_visible_for_fast_flux_detection():
    parsed = DNSParser().parse(_dns_response(qname="evil.tk", ttl=5))
    assert parsed.min_answer_ttl == 5


def test_min_answer_ttl_picks_the_lowest_across_multiple_answers():
    from scapy.all import IP, UDP
    from scapy.layers.dns import DNS, DNSQR, DNSRR
    pkt = IP() / UDP(sport=53, dport=54321) / DNS(
        id=1, qr=1, rcode=0, qdcount=1, ancount=2,
        qd=DNSQR(qname="cdn.example.com", qtype="A"),
        an=DNSRR(rrname="cdn.example.com", type="A", ttl=300, rdata="1.1.1.1")
        / DNSRR(rrname="cdn.example.com", type="A", ttl=15, rdata="2.2.2.2"),
    )
    parsed = DNSParser().parse(bytes(pkt))
    assert len(parsed.answers) == 2
    assert parsed.min_answer_ttl == 15


def test_multiple_questions_all_captured():
    from scapy.all import IP, UDP
    from scapy.layers.dns import DNS, DNSQR
    pkt = IP() / UDP(sport=12345, dport=53) / DNS(
        id=1, qr=0, qdcount=2,
        qd=DNSQR(qname="a.com", qtype="A") / DNSQR(qname="b.com", qtype="AAAA"),
    )
    parsed = DNSParser().parse(bytes(pkt))
    assert len(parsed.questions) == 2
    assert parsed.questions[0].qname == "a.com"
    assert parsed.questions[0].qtype == "A"
    assert parsed.questions[1].qname == "b.com"
    assert parsed.questions[1].qtype == "AAAA"


def test_non_dns_udp_packet_returns_none():
    from scapy.all import IP, UDP
    pkt = IP(src="10.0.0.1", dst="10.0.0.2") / UDP(sport=5353, dport=1234) / b"not dns"
    assert DNSParser().parse(bytes(pkt)) is None


def test_non_ip_bytes_return_none():
    assert DNSParser().parse(b"") is None
    assert DNSParser().parse(b"\xff\xff\xff\xff") is None


def test_unmapped_qtype_falls_back_to_other():
    parsed = DNSParser().parse(_dns_query(qname="weird.example", qtype=999))
    assert parsed.questions[0].qtype == "OTHER"
