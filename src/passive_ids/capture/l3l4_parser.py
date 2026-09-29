"""Tier 0-1: L3/L4 header parsing.

Parses raw captured bytes into the fields flow_state/ and features/
need: 5-tuple, protocol, TCP flags (for SYN-ratio features), TTL, and
packet size. Uses scapy for dissection -- already a project dependency
via the pcap replay harness, and far less error-prone here than hand
-rolling struct.unpack over Ethernet/IP/TCP headers. Production
capture should still move off scapy for raw throughput (see the note
in requirements.txt); this module's *interface* (bytes in,
ParsedPacket out) doesn't need to change when that happens -- only
what's inside `parse()`.

Known limitation, stated plainly rather than hidden: the `bytes`-only
interface on `pipeline.on_packet` carries no link-type metadata, so
this parser can't be *told* whether a capture is Ethernet-framed or
raw IP -- it has to guess from the first byte's IP-version nibble
(0x4x/0x6x for raw IPv4/IPv6, anything else assumed Ethernet). This is
correct for every case tested so far (real mirror-port captures are
Ethernet-framed; the synthetic pcap fixtures used elsewhere in this
project are raw IP) but is theoretically ambiguous if a real Ethernet
frame's destination MAC happens to start with a byte whose high
nibble is 4 or 6 -- a genuine edge case, not a hypothetical one, and
worth revisiting if `parse()` ever silently returns None at a higher
rate than expected on real traffic.

Arrival timestamp is intentionally NOT derived from packet bytes here.
In live capture, "now" is attached by the caller (pipeline.on_packet)
at the moment the packet is handed off. In replay,
scripts/replay_pcap.py's --speed realtime pacing exists specifically
so wall-clock arrival during replay reproduces the original capture's
inter-packet timing -- that's what makes Welford IAT features
meaningful on replayed data without threading timestamps through
every layer.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParsedPacket:
    src_ip: str
    dst_ip: str
    protocol: str  # "TCP" | "UDP" | "ICMP" | "OTHER"
    packet_size: int  # total captured bytes (len of raw_packet)
    ttl: int | None = None
    src_port: int | None = None
    dst_port: int | None = None
    syn: bool = False
    ack: bool = False
    fin: bool = False
    rst: bool = False
    psh: bool = False


class L3L4Parser:
    """Stateless -- one instance can be reused across all packets."""

    def parse(self, raw_packet: bytes) -> ParsedPacket | None:
        from scapy.layers.inet import ICMP, IP, TCP, UDP

        from passive_ids.capture._ip_dissect import dissect_ip_layer

        pkt = dissect_ip_layer(raw_packet)
        if pkt is None:
            return None

        ip_layer = pkt[IP]
        packet_size = len(raw_packet)

        if TCP in pkt:
            tcp_layer = pkt[TCP]
            flags = tcp_layer.flags
            return ParsedPacket(
                src_ip=ip_layer.src,
                dst_ip=ip_layer.dst,
                protocol="TCP",
                packet_size=packet_size,
                ttl=ip_layer.ttl,
                src_port=int(tcp_layer.sport),
                dst_port=int(tcp_layer.dport),
                syn=bool(flags.S),
                ack=bool(flags.A),
                fin=bool(flags.F),
                rst=bool(flags.R),
                psh=bool(flags.P),
            )

        if UDP in pkt:
            udp_layer = pkt[UDP]
            return ParsedPacket(
                src_ip=ip_layer.src,
                dst_ip=ip_layer.dst,
                protocol="UDP",
                packet_size=packet_size,
                ttl=ip_layer.ttl,
                src_port=int(udp_layer.sport),
                dst_port=int(udp_layer.dport),
            )

        if ICMP in pkt:
            return ParsedPacket(
                src_ip=ip_layer.src,
                dst_ip=ip_layer.dst,
                protocol="ICMP",
                packet_size=packet_size,
                ttl=ip_layer.ttl,
            )

        return ParsedPacket(
            src_ip=ip_layer.src,
            dst_ip=ip_layer.dst,
            protocol="OTHER",
            packet_size=packet_size,
            ttl=ip_layer.ttl,
        )
