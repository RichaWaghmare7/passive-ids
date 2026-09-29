"""DNS query/response parsing for DGA and DNS-tunneling detection.

Extracts structural DNS fields only -- domain-name processing (TLD
stripping before feeding features.entropy.string_char_entropy /
string_ngram_entropy) is deliberately left to the caller, same
division of responsibility as capture/l3l4_parser.py: this module's
job is wire-format extraction, not feature computation.

Built against scapy's actual DNS layer behavior, checked directly
rather than assumed:
  - `qname` and answer `rrname`/`rdata` come back as `bytes` with a
    trailing root-label dot (b"example.com."), not a clean str --
    decoded and stripped here.
  - `qtype` and answer `type` are raw integers (A=1, AAAA=28, ...),
    not the string mnemonics scapy accepts on construction -- mapped
    to names here via _QTYPE_NAMES.
  - `.qd` and `.an` are always list-like, even for zero or one
    entry (empty list when qdcount/ancount is 0) -- iterated
    uniformly here rather than assuming single-item attribute access,
    which only happens to work by coincidence for exactly one entry.

NXDOMAIN (rcode=3) responses matter specifically for DGA detection:
malware generating algorithmic domains typically has most of them
fail to resolve, so rcode is surfaced on every parsed response, not
just treated as a pass/fail on parsing itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from passive_ids.capture._ip_dissect import dissect_ip_layer

# RFC 1035 section 3.2.2 + common extensions. Anything not listed maps to "OTHER".
_QTYPE_NAMES: dict[int, str] = {
    1: "A",
    2: "NS",
    5: "CNAME",
    6: "SOA",
    12: "PTR",
    15: "MX",
    16: "TXT",
    28: "AAAA",
    33: "SRV",
    255: "ANY",
}

# RFC 1035 section 4.1.1. Anything not listed maps to "OTHER".
_RCODE_NAMES: dict[int, str] = {
    0: "NOERROR",
    1: "FORMERR",
    2: "SERVFAIL",
    3: "NXDOMAIN",
    4: "NOTIMP",
    5: "REFUSED",
}


def _qtype_name(code: int) -> str:
    return _QTYPE_NAMES.get(int(code), "OTHER")


def _rcode_name(code: int) -> str:
    return _RCODE_NAMES.get(int(code), "OTHER")


def _decode_name(raw: bytes | str) -> str:
    """Strip the trailing root-label dot scapy leaves on wire names."""
    s = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    return s[:-1] if s.endswith(".") else s


@dataclass
class DNSQuestion:
    qname: str
    qtype: str
    qclass: str = "IN"


@dataclass
class DNSAnswer:
    rrname: str
    rtype: str
    ttl: int
    rdata: str


@dataclass
class ParsedDNSMessage:
    transaction_id: int
    is_response: bool
    rcode: int
    rcode_name: str
    questions: list[DNSQuestion] = field(default_factory=list)
    answers: list[DNSAnswer] = field(default_factory=list)

    @property
    def is_nxdomain(self) -> bool:
        return self.rcode_name == "NXDOMAIN"

    @property
    def min_answer_ttl(self) -> int | None:
        """Lowest TTL across all answer records, or None if there are
        none (a query, or an NXDOMAIN/empty response). Unusually low
        TTLs (single-digit to low tens of seconds) are a fast-flux DGA
        signal -- infrastructure that rotates IPs faster than a
        legitimate CDN typically would.
        """
        if not self.answers:
            return None
        return min(a.ttl for a in self.answers)


class DNSParser:
    """Stateless -- one instance can be reused across all packets."""

    def parse(self, raw_packet: bytes) -> ParsedDNSMessage | None:
        pkt = dissect_ip_layer(raw_packet)
        if pkt is None:
            return None

        from scapy.layers.dns import DNS

        if DNS not in pkt:
            return None  # not a DNS message

        dns_layer = pkt[DNS]

        questions = [
            DNSQuestion(
                qname=_decode_name(q.qname),
                qtype=_qtype_name(q.qtype),
                qclass="IN",
            )
            for q in dns_layer.qd
        ]
        answers = [
            DNSAnswer(
                rrname=_decode_name(a.rrname),
                rtype=_qtype_name(a.type),
                ttl=int(a.ttl),
                rdata=_decode_name(a.rdata) if isinstance(a.rdata, bytes) else str(a.rdata),
            )
            for a in dns_layer.an
        ]

        rcode = int(dns_layer.rcode)
        return ParsedDNSMessage(
            transaction_id=int(dns_layer.id),
            is_response=bool(dns_layer.qr),
            rcode=rcode,
            rcode_name=_rcode_name(rcode),
            questions=questions,
            answers=answers,
        )
