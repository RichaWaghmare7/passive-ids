"""JA4 threat-intelligence lookup: known-malicious fingerprint matching.

This is reactive, not predictive: it can only catch a JA4 that's
already been logged as malicious somewhere. A zero-day C2 framework
with a novel TLS stack produces a JA4 that matches nothing here --
exactly the gap flagged during the earlier traditional-ML-vs-DL
comparison as the strongest case for eventually adding a slow-path
model on top of pure fingerprint matching. This module is the
"already-known-bad" layer, not a substitute for that.

No real threat-intelligence data ships with this module, deliberately.
The JA4 *algorithm* (capture/tls_parser.py) was verified against
FoxIO's own specification and test vectors before being implemented --
but there is no equivalent way to verify a claim like "this fingerprint
belongs to malware family X" from memory, and a fabricated mapping
would be worse than an empty table: a wrong entry produces false
confidence in either direction (a false match wastes analyst time
chasing nothing; a table that "looks populated" delays realizing
real coverage is thin). Load real entries from an actual source:
FoxIO's own community JA4 lists, an Abuse.ch-style feed, MISP, or --
once the SOC verdict feedback loop (verdicts/) has accumulated real
confirmed-malicious history -- your own organization's incidents.

No bounded-memory/eviction concern here, unlike flow_state's
aggregates: this table's size is controlled by the curated
intelligence feed loaded into it, not by attacker-controlled traffic
volume, so there's nothing analogous to a flood inflating it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class JA4Entry:
    ja4: str
    malware_family: str | None = None
    threat_actor: str | None = None
    confidence: str = "MEDIUM"  # LOW / MEDIUM / HIGH -- the source feed's own confidence
    source: str = "unknown"  # which feed/list this came from
    first_seen: str | None = None  # ISO date string, from the feed's metadata
    notes: str = ""


class JA4LookupTable:
    def __init__(self) -> None:
        self._entries: dict[str, JA4Entry] = {}
        self.lookup_count = 0
        self.hit_count = 0

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, entry: JA4Entry) -> None:
        self._entries[entry.ja4] = entry

    def load_from_json(self, path: str | Path) -> int:
        """Load entries from a JSON file: a list of objects matching
        JA4Entry's fields (only "ja4" is required; everything else
        defaults). Returns the number of entries loaded. Existing
        entries with the same ja4 are overwritten, so re-loading an
        updated feed is safe to do repeatedly.
        """
        with open(path) as f:
            records: list[dict[str, Any]] = json.load(f)
        for record in records:
            self.add(JA4Entry(**record))
        return len(records)

    def lookup(self, ja4: str) -> JA4Entry | None:
        self.lookup_count += 1
        entry = self._entries.get(ja4)
        if entry is not None:
            self.hit_count += 1
        return entry

    @property
    def hit_rate(self) -> float:
        """A near-zero hit rate over real traffic isn't necessarily
        good news -- it may mean coverage is thin rather than that
        no malware is present. This number alone can't distinguish
        those two cases; it's a prompt to check the feed's coverage,
        not a confidence score on its own.
        """
        return self.hit_count / self.lookup_count if self.lookup_count else 0.0

    def export_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump([asdict(e) for e in self._entries.values()], f, indent=2)
