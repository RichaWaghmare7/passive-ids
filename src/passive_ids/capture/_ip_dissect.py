"""Shared raw-bytes-to-scapy-packet dissection, used by every capture/
parser. Internal module (leading underscore) -- not part of the
public capture/ interface, just de-duplicated logic between
l3l4_parser.py and dns_parser.py (and tls_parser.py, when that's built).

See l3l4_parser.py's module docstring for the full explanation of the
version-nibble framing heuristic and its known edge case.
"""
from __future__ import annotations

from scapy.packet import Packet


def dissect_ip_layer(raw_packet: bytes) -> Packet | None:
    """Return a dissected scapy packet with an IP layer, or None.

    None covers: empty input, IPv6 (not yet supported), or bytes that
    don't dissect into anything with a recognizable IPv4 layer.
    """
    if not raw_packet:
        return None

    from scapy.layers.inet import IP
    from scapy.layers.l2 import Ether

    version_nibble = raw_packet[0] >> 4
    try:
        if version_nibble == 4:
            pkt = IP(raw_packet)
        elif version_nibble == 6:
            return None  # IPv6 not yet supported
        else:
            pkt = Ether(raw_packet)
    except Exception:
        return None

    if IP not in pkt:
        return None
    return pkt
