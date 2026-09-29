"""Tests for capture/tls_parser.py (JA3 + JA4).

Built directly on the official published worked examples -- these are
known-answer tests (exact expected output), not approximations:
  - JA3: both worked examples from github.com/salesforce/ja3
  - JA4: the worked example and all 8 ALPN edge cases from
    github.com/FoxIO-LLC/ja4/blob/main/technical_details/JA4.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from passive_ids.capture.tls_parser import (
    GREASE_VALUES,
    ParsedClientHello,
    TLSParser,
    _ja4_alpn_code,
    _ja4_cipher_hash,
    _ja4_extension_hash,
    compute_ja3,
    compute_ja4,
    is_grease,
)


# ---------------------------------------------------------------------------
# GREASE
# ---------------------------------------------------------------------------

def test_grease_values_match_rfc8701():
    expected = {
        0x0A0A, 0x1A1A, 0x2A2A, 0x3A3A, 0x4A4A, 0x5A5A, 0x6A6A, 0x7A7A,
        0x8A8A, 0x9A9A, 0xAAAA, 0xBABA, 0xCACA, 0xDADA, 0xEAEA, 0xFAFA,
    }
    assert GREASE_VALUES == expected
    assert len(GREASE_VALUES) == 16
    assert is_grease(0x0A0A) is True
    assert is_grease(0x1301) is False  # a real TLS 1.3 cipher, not GREASE


# ---------------------------------------------------------------------------
# JA3 -- both official Salesforce worked examples, exact expected MD5
# ---------------------------------------------------------------------------

def test_ja3_official_example_with_extensions():
    ch = ParsedClientHello(
        legacy_version=769,
        ciphers=[47, 53, 5, 10, 49161, 49162, 49171, 49172, 50, 56, 19, 4],
        extensions=[0, 10, 11],
        supported_groups=[23, 24, 25],
        ec_point_formats=[0],
    )
    ja3_string, ja3_hash = compute_ja3(ch)
    assert ja3_string == "769,47-53-5-10-49161-49162-49171-49172-50-56-19-4,0-10-11,23-24-25,0"
    assert ja3_hash == "ada70206e40642a3e4461f35503241d5"


def test_ja3_official_example_no_extensions():
    ch = ParsedClientHello(
        legacy_version=769,
        ciphers=[4, 5, 10, 9, 100, 98, 3, 6, 19, 18, 99],
        extensions=[],
        supported_groups=[],
        ec_point_formats=[],
    )
    ja3_string, ja3_hash = compute_ja3(ch)
    assert ja3_string == "769,4-5-10-9-100-98-3-6-19-18-99,,,"
    assert ja3_hash == "de350869b8c85de67a350c8d186f11e6"


def test_ja3_filters_grease_from_ciphers_and_extensions():
    ch = ParsedClientHello(
        legacy_version=769,
        ciphers=[0x0A0A, 4, 5],  # leading GREASE cipher
        extensions=[0x1A1A, 0, 10],  # leading GREASE extension
        supported_groups=[0x2A2A, 23],
        ec_point_formats=[0],
    )
    ja3_string, _ = compute_ja3(ch)
    # GREASE values must not appear anywhere in the string
    assert "2570" not in ja3_string  # 0x0A0A decimal = 2570
    assert ja3_string.split(",")[1] == "4-5"
    assert ja3_string.split(",")[2] == "0-10"
    assert ja3_string.split(",")[3] == "23"


# ---------------------------------------------------------------------------
# JA4 ALPN encoding -- all 8 worked examples from the FoxIO spec
# ---------------------------------------------------------------------------

def test_ja4_alpn_no_alpn_is_00():
    assert _ja4_alpn_code(None) == "00"
    assert _ja4_alpn_code(b"") == "00"


def test_ja4_alpn_h2_direct_chars():
    assert _ja4_alpn_code(b"h2") == "h2"


def test_ja4_alpn_http11_gives_h1():
    assert _ja4_alpn_code(b"http/1.1") == "h1"


def test_ja4_alpn_official_hex_edge_cases():
    # every one of these is verbatim from FoxIO's JA4.md
    assert _ja4_alpn_code(bytes([0xAB])) == "ab"
    assert _ja4_alpn_code(bytes([0x20])) == "20"
    assert _ja4_alpn_code(bytes([0xAB, 0xCD])) == "ad"
    assert _ja4_alpn_code(bytes([0x20, 0x61])) == "21"
    assert _ja4_alpn_code(bytes([0x30, 0xAB])) == "3b"
    assert _ja4_alpn_code(bytes([0x61, 0x20])) == "60"
    assert _ja4_alpn_code(bytes([0x30, 0x31, 0xAB, 0xCD])) == "3d"
    assert _ja4_alpn_code(bytes([0x30, 0xAB, 0xCD, 0x31])) == "01"


def test_ja4_alpn_single_alnum_char_doubles():
    assert _ja4_alpn_code(b"h") == "hh"


# ---------------------------------------------------------------------------
# JA4 cipher/extension hash -- the exact FoxIO worked example
# ---------------------------------------------------------------------------

def test_ja4_cipher_hash_matches_official_example():
    ciphers = [
        0x1301, 0x1302, 0x1303, 0xC02B, 0xC02F, 0xC02C, 0xC030, 0xCCA9,
        0xCCA8, 0xC013, 0xC014, 0x009C, 0x009D, 0x002F, 0x0035,
    ]
    assert _ja4_cipher_hash(ciphers) == "8daaf6152771"


def test_ja4_cipher_hash_empty_is_sentinel_not_hash():
    assert _ja4_cipher_hash([]) == "000000000000"
    assert _ja4_cipher_hash([0x0A0A]) == "000000000000"  # only GREASE -> empty after filtering


def test_ja4_extension_hash_matches_official_example():
    extensions = [
        0x001B, 0x0000, 0x0033, 0x0010, 0x4469, 0x0017, 0x002D, 0x000D,
        0x0005, 0x0023, 0x0012, 0x002B, 0xFF01, 0x000B, 0x000A, 0x0015,
    ]
    sig_algs = [0x0403, 0x0804, 0x0401, 0x0503, 0x0805, 0x0501, 0x0806, 0x0601]
    assert _ja4_extension_hash(extensions, sig_algs) == "e5627efa2ab1"


def test_ja4_extension_hash_no_sig_algs_no_trailing_underscore():
    # Same 14-item sorted+filtered extension list as the main worked
    # example (0000/SNI and 0010/ALPN included here since the function
    # filters them internally) -- just without signature algorithms.
    extensions = [
        0x001B, 0x0000, 0x0033, 0x0010, 0x4469, 0x0017, 0x002D, 0x000D,
        0x0005, 0x0023, 0x0012, 0x002B, 0xFF01, 0x000B, 0x000A, 0x0015,
    ]
    assert _ja4_extension_hash(extensions, []) == "6d807ffa2a79"


def test_ja4_extension_hash_empty_is_sentinel():
    assert _ja4_extension_hash([], []) == "000000000000"
    assert _ja4_extension_hash([0x0000, 0x0010], []) == "000000000000"  # only SNI+ALPN


# ---------------------------------------------------------------------------
# JA4 full fingerprint -- the complete official worked example, end to end
# ---------------------------------------------------------------------------

def test_ja4_full_fingerprint_matches_official_example():
    ch = ParsedClientHello(
        legacy_version=0x0303,
        ciphers=[
            0x1301, 0x1302, 0x1303, 0xC02B, 0xC02F, 0xC02C, 0xC030, 0xCCA9,
            0xCCA8, 0xC013, 0xC014, 0x009C, 0x009D, 0x002F, 0x0035,
        ],
        extensions=[
            0x001B, 0x0000, 0x0033, 0x0010, 0x4469, 0x0017, 0x002D, 0x000D,
            0x0005, 0x0023, 0x0012, 0x002B, 0xFF01, 0x000B, 0x000A, 0x0015,
        ],
        sni="example.com",
        alpn_protocols=[b"h2"],
        signature_algorithms=[0x0403, 0x0804, 0x0401, 0x0503, 0x0805, 0x0501, 0x0806, 0x0601],
        supported_versions=[0x0304],  # -> effective_version = TLS 1.3
    )
    assert compute_ja4(ch) == "t13d1516h2_8daaf6152771_e5627efa2ab1"


def test_ja4_effective_version_prefers_supported_versions_over_legacy():
    ch = ParsedClientHello(
        legacy_version=0x0303,  # frozen at TLS 1.2 for compatibility
        ciphers=[],
        extensions=[],
        supported_versions=[0x0304, 0x0303],  # real client is TLS 1.3
    )
    assert ch.effective_version == 0x0304


def test_ja4_effective_version_falls_back_to_legacy_without_extension():
    ch = ParsedClientHello(legacy_version=0x0303, ciphers=[], extensions=[])
    assert ch.effective_version == 0x0303


def test_ja4_sni_flag_reflects_presence():
    with_sni = ParsedClientHello(legacy_version=0x0303, ciphers=[], extensions=[], sni="example.com")
    without_sni = ParsedClientHello(legacy_version=0x0303, ciphers=[], extensions=[], sni=None)
    assert compute_ja4(with_sni).split("_")[0][3] == "d"
    assert compute_ja4(without_sni).split("_")[0][3] == "i"


# ---------------------------------------------------------------------------
# TLSParser -- real scapy-constructed ClientHello, round-tripped through bytes
# ---------------------------------------------------------------------------

def _build_client_hello_packet(dport=443, include_sni=True):
    from scapy.layers.inet import IP, TCP
    from scapy.layers.tls.all import TLS
    from scapy.layers.tls.extensions import (
        ProtocolName,
        TLS_Ext_ALPN,
        TLS_Ext_ServerName,
        TLS_Ext_SupportedGroups,
        TLS_Ext_SupportedVersion_CH,
        ServerName,
    )
    from scapy.layers.tls.handshake import TLSClientHello

    ext = [
        TLS_Ext_SupportedGroups(groups=[23, 24]),
        TLS_Ext_ALPN(protocols=[ProtocolName(protocol=b"h2")]),
        TLS_Ext_SupportedVersion_CH(versions=[0x0304, 0x0303]),
    ]
    if include_sni:
        ext.insert(0, TLS_Ext_ServerName(servernames=[ServerName(servername=b"evil.tk")]))

    ch = TLSClientHello(version=0x0303, ciphers=[0x1301, 0x0A0A, 0xC02F], comp=[0], ext=ext)
    return bytes(IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=54321, dport=dport) / TLS(msg=[ch]))


def test_parser_extracts_real_clienthello_on_standard_port():
    parsed = TLSParser().parse(_build_client_hello_packet(dport=443))
    assert parsed is not None
    assert parsed.sni == "evil.tk"
    assert 0x1301 in parsed.ciphers
    assert 0x0A0A in parsed.ciphers  # GREASE preserved in raw extraction; filtered later at hash time
    assert parsed.alpn_protocols == [b"h2"]
    assert parsed.supported_groups == [23, 24]
    assert parsed.effective_version == 0x0304


def test_parser_falls_back_to_content_signature_on_nonstandard_port():
    # scapy only auto-binds TLS under port 443 -- confirmed this fails
    # without the manual 0x16 signature check this parser implements.
    parsed = TLSParser().parse(_build_client_hello_packet(dport=8443))
    assert parsed is not None
    assert parsed.sni == "evil.tk"


def test_parser_returns_none_for_non_tls_traffic():
    from scapy.layers.inet import IP, TCP
    raw = bytes(IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=1234, dport=443) / b"GET / HTTP/1.1")
    assert TLSParser().parse(raw) is None


def test_parser_returns_none_for_empty_or_garbage_bytes():
    assert TLSParser().parse(b"") is None
    assert TLSParser().parse(b"\xff\xff\xff\xff") is None


def test_end_to_end_parsed_real_packet_produces_valid_ja4():
    parsed = TLSParser().parse(_build_client_hello_packet())
    ja4 = compute_ja4(parsed)
    # 3 ciphers, one is GREASE -> count should be 2; SNI present -> "d"
    parts = ja4.split("_")
    assert parts[0][3] == "d"  # sni flag position within the a-section
    assert len(parts) == 3
    assert len(parts[1]) == 12
    assert len(parts[2]) == 12
