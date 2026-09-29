"""TLS ClientHello parsing: JA3 and JA4 fingerprinting for encrypted
malware detection without decryption.

Every construction rule here was verified against a primary source
before implementation, not written from memory -- a single wrong
byte-encoding rule would silently produce a fingerprint that looks
plausible but matches nothing in any real threat-intel database:

  - JA3: Salesforce's original specification (github.com/salesforce/ja3),
    including its two published worked examples, used directly as this
    module's test fixtures (known input -> known MD5 output).
  - JA4: FoxIO's technical specification, fetched directly
    (github.com/FoxIO-LLC/ja4/blob/main/technical_details/JA4.md) --
    including its worked example (t13d1516h2_8daaf6152771_e5627efa2ab1)
    and all eight ALPN-encoding edge cases, all used as test fixtures.
  - GREASE values: RFC 8701, fetched directly.

scapy quirks discovered while building this, checked rather than
assumed:
  - scapy.layers.tls only auto-binds the TLS layer under TCP port 443
    by default. Malware and non-standard servers routinely run TLS on
    other ports specifically to evade port-based detection, so this
    module falls back to a content-signature check (first payload byte
    0x16 = TLS handshake record) and attempts dissection regardless
    of port.
  - The ClientHello's supported_versions extension is exposed via the
    class TLS_Ext_SupportedVersion_CH (singular "Version", "_CH"
    suffix) in this scapy version. The more guessable
    TLS_Ext_SupportedVersions (plural, no suffix) exists but is an
    inert placeholder that never actually gets bound during real
    dissection -- confirmed directly, since it fails silently rather
    than raising.

Scope, disclosed rather than silently assumed: this parses TLS over
TCP only. QUIC (TLS 1.3 over UDP) and DTLS ClientHellos are not
supported -- JA4's "q"/"d" transport codes are real but unreachable
from this module today.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from passive_ids.capture._ip_dissect import dissect_ip_layer

# RFC 8701. Identical 16 values are reserved both for cipher
# suites/ALPN identifiers (as 2-byte pairs) and for extensions/named
# groups/signature algorithms/versions (as 16-bit values) -- verified
# directly against the RFC text, not recalled from memory.
GREASE_VALUES: frozenset[int] = frozenset(
    [
        0x0A0A, 0x1A1A, 0x2A2A, 0x3A3A, 0x4A4A, 0x5A5A, 0x6A6A, 0x7A7A,
        0x8A8A, 0x9A9A, 0xAAAA, 0xBABA, 0xCACA, 0xDADA, 0xEAEA, 0xFAFA,
    ]
)


def is_grease(value: int) -> bool:
    return value in GREASE_VALUES


# FoxIO JA4.md "TLS and DTLS Version" table, verified directly against
# the primary source.
_JA4_VERSION_CODES: dict[int, str] = {
    0x0304: "13",
    0x0303: "12",
    0x0302: "11",
    0x0301: "10",
    0x0300: "s3",
    0x0002: "s2",
    0xFEFF: "d1",
    0xFEFD: "d2",
    0xFEFC: "d3",
}

_SNI_EXT_TYPE = 0x0000
_ALPN_EXT_TYPE = 0x0010
_SUPPORTED_GROUPS_EXT_TYPE = 0x000A
_SIGNATURE_ALGORITHMS_EXT_TYPE = 0x000D
_SUPPORTED_VERSIONS_EXT_TYPE = 0x002B


@dataclass
class ParsedClientHello:
    legacy_version: int
    ciphers: list[int]
    extensions: list[int]  # extension type codes, in advertised order
    sni: str | None = None
    alpn_protocols: list[bytes] = field(default_factory=list)
    supported_groups: list[int] = field(default_factory=list)
    ec_point_formats: list[int] = field(default_factory=list)
    signature_algorithms: list[int] = field(default_factory=list)
    supported_versions: list[int] = field(default_factory=list)

    @property
    def effective_version(self) -> int:
        """Highest advertised version, GREASE excluded. Prefers
        supported_versions (how a TLS 1.3 client actually advertises
        its real version) over the legacy version field, which stays
        frozen at TLS 1.2 for backward compatibility -- per JA4's
        documented rule."""
        candidates = [v for v in self.supported_versions if not is_grease(v)]
        if candidates:
            return max(candidates)
        return self.legacy_version


class TLSParser:
    """Stateless -- one instance can be reused across all packets."""

    def parse(self, raw_packet: bytes) -> ParsedClientHello | None:
        pkt = dissect_ip_layer(raw_packet)
        if pkt is None:
            return None

        from scapy.layers.inet import TCP

        if TCP not in pkt:
            return None

        payload = bytes(pkt[TCP].payload)
        if not payload or payload[0] != 0x16:  # TLS handshake record type
            return None  # not a TLS handshake on this segment

        from scapy.layers.tls.all import TLS
        from scapy.layers.tls.handshake import TLSClientHello

        try:
            tls_pkt = TLS(payload)
        except Exception:
            return None

        if not hasattr(tls_pkt, "msg"):
            return None

        client_hello = next(
            (m for m in tls_pkt.msg if isinstance(m, TLSClientHello)), None
        )
        if client_hello is None:
            return None  # a TLS record, but not a ClientHello

        sni: str | None = None
        alpn_protocols: list[bytes] = []
        supported_groups: list[int] = []
        ec_point_formats: list[int] = []
        signature_algorithms: list[int] = []
        supported_versions: list[int] = []

        for ext in client_hello.ext or []:
            ext_type = int(ext.type)
            if ext_type == _SNI_EXT_TYPE and getattr(ext, "servernames", None):
                sni = ext.servernames[0].servername.decode("utf-8", errors="replace")
            elif ext_type == _ALPN_EXT_TYPE and hasattr(ext, "protocols"):
                alpn_protocols = [p.protocol for p in ext.protocols]
            elif ext_type == _SUPPORTED_GROUPS_EXT_TYPE and hasattr(ext, "groups"):
                supported_groups = list(ext.groups)
            elif ext_type == 0x000B and hasattr(ext, "ecpl"):
                ec_point_formats = list(ext.ecpl)
            elif ext_type == _SIGNATURE_ALGORITHMS_EXT_TYPE and hasattr(ext, "sig_algs"):
                signature_algorithms = list(ext.sig_algs)
            elif ext_type == _SUPPORTED_VERSIONS_EXT_TYPE and hasattr(ext, "versions"):
                supported_versions = list(ext.versions)

        return ParsedClientHello(
            legacy_version=int(client_hello.version),
            ciphers=list(client_hello.ciphers or []),
            extensions=[int(e.type) for e in (client_hello.ext or [])],
            sni=sni,
            alpn_protocols=alpn_protocols,
            supported_groups=supported_groups,
            ec_point_formats=ec_point_formats,
            signature_algorithms=signature_algorithms,
            supported_versions=supported_versions,
        )


def compute_ja3(ch: ParsedClientHello) -> tuple[str, str]:
    """Returns (ja3_string, ja3_md5_hex).

    Per Salesforce's spec: GREASE filtered from ciphers, extensions,
    and elliptic curves (supported groups). Point formats are NOT
    GREASE-filtered -- RFC 8701 does not reserve GREASE codepoints for
    that field, so there's nothing to filter there.
    """
    ciphers = [c for c in ch.ciphers if not is_grease(c)]
    extensions = [e for e in ch.extensions if not is_grease(e)]
    groups = [g for g in ch.supported_groups if not is_grease(g)]

    ja3_string = "{},{},{},{},{}".format(
        ch.legacy_version,
        "-".join(str(c) for c in ciphers),
        "-".join(str(e) for e in extensions),
        "-".join(str(g) for g in groups),
        "-".join(str(p) for p in ch.ec_point_formats),
    )
    ja3_hash = hashlib.md5(ja3_string.encode()).hexdigest()
    return ja3_string, ja3_hash


def _ja4_alpn_code(first_alpn: bytes | None) -> str:
    """FoxIO JA4.md "ALPN Extension Value", verified against all eight
    of the spec's worked examples (see test_tls_parser.py).
    """
    if not first_alpn:
        return "00"

    first_byte = first_alpn[0]
    last_byte = first_alpn[-1]  # equals first_byte when len==1 -- this is
    # what implements "a single character is treated as both first and
    # last" for free, without a separate branch.

    def _is_alnum(b: int) -> bool:
        return (0x30 <= b <= 0x39) or (0x41 <= b <= 0x5A) or (0x61 <= b <= 0x7A)

    if _is_alnum(first_byte) and _is_alnum(last_byte):
        return chr(first_byte) + chr(last_byte)

    hex_str = first_alpn.hex()
    return hex_str[0] + hex_str[-1]


def _ja4_cipher_hash(ciphers: list[int]) -> str:
    """FoxIO JA4.md "Cipher hash". GREASE excluded; SCSV (0x00FF,
    0x5600) and experimental/reserved (0xFE00-0xFEFF) values are
    deliberately still included -- they are not GREASE and the spec is
    explicit that they count.
    """
    filtered = [c for c in ciphers if not is_grease(c)]
    if not filtered:
        return "000000000000"  # per spec: a literal sentinel, not a hash of nothing
    sorted_hex = sorted(f"{c:04x}" for c in filtered)
    return hashlib.sha256(",".join(sorted_hex).encode()).hexdigest()[:12]


def _ja4_extension_hash(extensions: list[int], signature_algorithms: list[int]) -> str:
    """FoxIO JA4.md "Extension hash". GREASE, SNI (0x0000), and ALPN
    (0x0010) excluded from the extension list -- SNI/ALPN are already
    captured in the 'a' section, so removing them here means the same
    client produces the same 'c' section regardless of destination or
    ALPN choice. Extensions are sorted; signature algorithms are NOT
    sorted, kept in their advertised order, per the spec.
    """
    filtered = [
        e for e in extensions
        if not is_grease(e) and e not in (_SNI_EXT_TYPE, _ALPN_EXT_TYPE)
    ]
    if not filtered:
        return "000000000000"  # per spec: literal sentinel, not a hash of nothing

    sorted_ext_hex = sorted(f"{e:04x}" for e in filtered)
    sig_alg_hex = [f"{s:04x}" for s in signature_algorithms if not is_grease(s)]

    if sig_alg_hex:
        c_input = ",".join(sorted_ext_hex) + "_" + ",".join(sig_alg_hex)
    else:
        c_input = ",".join(sorted_ext_hex)  # no trailing underscore, per spec

    return hashlib.sha256(c_input.encode()).hexdigest()[:12]


def compute_ja4(ch: ParsedClientHello, transport: str = "t") -> str:
    """Full JA4 fingerprint, verified directly against FoxIO's
    technical specification. `transport` defaults to "t" (TLS over
    TCP) -- this module doesn't parse QUIC or DTLS, so "q"/"d" are
    never produced here, only accepted as a parameter for callers that
    might supply pre-parsed QUIC/DTLS ClientHello data in the future.
    """
    version_code = _JA4_VERSION_CODES.get(ch.effective_version, "00")
    sni_flag = "d" if ch.sni else "i"

    cipher_count = min(len([c for c in ch.ciphers if not is_grease(c)]), 99)
    ext_count = min(len([e for e in ch.extensions if not is_grease(e)]), 99)
    alpn_code = _ja4_alpn_code(ch.alpn_protocols[0] if ch.alpn_protocols else None)

    a_section = f"{transport}{version_code}{sni_flag}{cipher_count:02d}{ext_count:02d}{alpn_code}"
    b_section = _ja4_cipher_hash(ch.ciphers)
    c_section = _ja4_extension_hash(ch.extensions, ch.signature_algorithms)

    return f"{a_section}_{b_section}_{c_section}"
