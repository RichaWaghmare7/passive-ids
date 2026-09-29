"""Tests for intel/ja4_lookup.py.

Uses only clearly-synthetic test fingerprints -- these are not real
threat intelligence, just fixtures to prove the lookup mechanism works.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from passive_ids.intel.ja4_lookup import JA4Entry, JA4LookupTable

_SYNTHETIC_JA4 = "t13d1516h2_synthetic0000_testfixture01"  # not a real IOC


def test_empty_table_has_no_matches():
    table = JA4LookupTable()
    assert len(table) == 0
    assert table.lookup("anything") is None
    assert table.hit_rate == 0.0


def test_add_and_lookup_hit():
    table = JA4LookupTable()
    table.add(JA4Entry(ja4=_SYNTHETIC_JA4, malware_family="SYNTHETIC_TEST_FAMILY", confidence="HIGH"))

    result = table.lookup(_SYNTHETIC_JA4)
    assert result is not None
    assert result.malware_family == "SYNTHETIC_TEST_FAMILY"
    assert result.confidence == "HIGH"


def test_lookup_miss_for_unknown_fingerprint():
    table = JA4LookupTable()
    table.add(JA4Entry(ja4=_SYNTHETIC_JA4))
    assert table.lookup("t13d1516h2_deadbeef0000_cafefeed0001") is None


def test_hit_rate_tracks_lookups_and_hits_separately():
    table = JA4LookupTable()
    table.add(JA4Entry(ja4=_SYNTHETIC_JA4))

    table.lookup(_SYNTHETIC_JA4)     # hit
    table.lookup("unknown_1")        # miss
    table.lookup("unknown_2")        # miss
    table.lookup(_SYNTHETIC_JA4)     # hit

    assert table.lookup_count == 4
    assert table.hit_count == 2
    assert abs(table.hit_rate - 0.5) < 1e-9


def test_load_from_json_reads_multiple_entries():
    records = [
        {"ja4": "fp1", "malware_family": "FAM_A", "source": "test_feed"},
        {"ja4": "fp2", "malware_family": "FAM_B", "confidence": "LOW"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(records, f)
        path = f.name

    table = JA4LookupTable()
    loaded_count = table.load_from_json(path)

    assert loaded_count == 2
    assert len(table) == 2
    assert table.lookup("fp1").malware_family == "FAM_A"
    assert table.lookup("fp2").confidence == "LOW"


def test_reloading_json_overwrites_existing_entry():
    table = JA4LookupTable()
    table.add(JA4Entry(ja4="fp1", malware_family="OLD_NAME"))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([{"ja4": "fp1", "malware_family": "UPDATED_NAME"}], f)
        path = f.name
    table.load_from_json(path)

    assert len(table) == 1  # not duplicated
    assert table.lookup("fp1").malware_family == "UPDATED_NAME"


def test_export_then_reimport_round_trips():
    table = JA4LookupTable()
    table.add(JA4Entry(ja4=_SYNTHETIC_JA4, malware_family="SYNTHETIC_TEST_FAMILY", notes="test round-trip"))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        export_path = f.name
    table.export_json(export_path)

    reimported = JA4LookupTable()
    reimported.load_from_json(export_path)

    assert reimported.lookup(_SYNTHETIC_JA4).notes == "test round-trip"
