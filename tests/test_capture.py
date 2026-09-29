"""Tests for capture/l3l4_parser.py.

Covers both framing cases the parser has to guess between (raw IP vs
Ethernet-framed), all three explicitly-handled protocols, and the
malformed/empty-input paths.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from passive_ids.capture.l3l4_parser import L3L4Parser


def _make_raw_ip_tcp(flags="S", sport=1234, dport=443):
    from scapy.all import IP, TCP
    pkt = IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=sport, dport=dport, flags=flags)
    return bytes(pkt)


def _make_ethernet_ip_tcp(flags="SA"):
    from scapy.all import Ether, IP, TCP
    pkt = (
        Ether(src="aa:bb:cc:dd:ee:ff", dst="11:22:33:44:55:66")
        / IP(src="10.0.0.5", dst="10.0.0.6")
        / TCP(sport=5555, dport=80, flags=flags)
    )
    return bytes(pkt)


def test_parses_raw_ip_tcp_syn():
    parser = L3L4Parser()
    raw = _make_raw_ip_tcp(flags="S")
    parsed = parser.parse(raw)

    assert parsed is not None
    assert parsed.src_ip == "10.0.0.1"
    assert parsed.dst_ip == "10.0.0.2"
    assert parsed.protocol == "TCP"
    assert parsed.src_port == 1234
    assert parsed.dst_port == 443
    assert parsed.syn is True
    assert parsed.ack is False


def test_parses_ethernet_framed_tcp_synack():
    parser = L3L4Parser()
    raw = _make_ethernet_ip_tcp(flags="SA")
    parsed = parser.parse(raw)

    assert parsed is not None
    assert parsed.src_ip == "10.0.0.5"
    assert parsed.dst_ip == "10.0.0.6"
    assert parsed.syn is True
    assert parsed.ack is True


def test_parses_udp():
    from scapy.all import IP, UDP
    raw = bytes(IP(src="10.0.0.1", dst="8.8.8.8") / UDP(sport=5353, dport=53))
    parsed = L3L4Parser().parse(raw)

    assert parsed is not None
    assert parsed.protocol == "UDP"
    assert parsed.dst_port == 53


def test_parses_icmp():
    from scapy.all import ICMP, IP
    raw = bytes(IP(src="10.0.0.1", dst="10.0.0.2") / ICMP())
    parsed = L3L4Parser().parse(raw)

    assert parsed is not None
    assert parsed.protocol == "ICMP"
    assert parsed.src_port is None


def test_empty_bytes_returns_none():
    assert L3L4Parser().parse(b"") is None


def test_garbage_bytes_do_not_raise():
    # Should return None, not throw -- a malformed capture must never
    # crash the pipeline.
    result = L3L4Parser().parse(b"\xff\xff\xff\xff\xff\xff\xff\xff")
    assert result is None or result.protocol == "OTHER"


def test_packet_size_matches_raw_length():
    raw = _make_raw_ip_tcp()
    parsed = L3L4Parser().parse(raw)
    assert parsed.packet_size == len(raw)


def test_flags_all_false_for_ack_only():
    raw = _make_raw_ip_tcp(flags="A")
    parsed = L3L4Parser().parse(raw)
    assert parsed.syn is False
    assert parsed.ack is True
    assert parsed.fin is False
    assert parsed.rst is False
