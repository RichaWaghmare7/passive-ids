"""Shannon entropy: per-string DGA scoring, and streaming categorical entropy.

Two distinct use cases, both built on the same `shannon_entropy`
primitive:

1. Per-string entropy (stateless) -- character and n-gram entropy of a
   single domain name, scored fresh each time. There's no "running"
   state across different domains; each is independent.

2. Streaming categorical entropy (stateful) -- entropy of a
   *distribution* built up incrementally across many observations from
   the same source (e.g. protocol-type diversity per source IP),
   tracked without storing every individual observation.

Known limitation, checked empirically rather than assumed from
memory: dictionary-based DGA families (concatenating real words, e.g.
"applepine.com") do NOT reliably produce *low* character entropy the
way a fully random string does. Checked directly: "applepine" scores
2.42 bits/char, which sits between "google" (1.92) and "facebook"
(2.75) -- not clearly below either, and well short of a random
string's ~3.0+. Character/n-gram entropy alone cannot reliably
separate this DGA family from ordinary multi-word domains. This is
exactly the gap flagged earlier (during the traditional-ML-vs-DL
comparison) as the strongest case for an async, slow-path DL addition
to models/dga_cascade.py -- deferred pending real miss-rate data from
the SOC verdict feedback loop, not built preemptively on a guess.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Hashable


def shannon_entropy(frequencies: dict[Any, int]) -> float:
    """Shannon entropy, in bits, of a discrete frequency distribution.

    0.0 for an empty distribution or one with a single category --
    both cases have no uncertainty to measure, not an error.
    """
    total = sum(frequencies.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in frequencies.values():
        if count == 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def string_char_entropy(s: str) -> float:
    """Character-level Shannon entropy of a string.

    Caller is expected to have already stripped the TLD (e.g. pass
    "google" from "google.com") -- entropy over the label is what
    distinguishes domains, not the shared, low-entropy TLD suffix.
    """
    return shannon_entropy(Counter(s))


def string_ngram_entropy(s: str, n: int = 2) -> float:
    """N-gram Shannon entropy -- captures sequential structure that
    per-character entropy misses (English text has common bigrams
    like "th", "er", "ing"; a random string's bigram distribution is
    much flatter). Returns 0.0 if the string is shorter than n.
    """
    if len(s) < n:
        return 0.0
    grams = [s[i : i + n] for i in range(len(s) - n + 1)]
    return shannon_entropy(Counter(grams))


@dataclass
class StreamingCategoricalEntropy:
    """Incremental Shannon entropy over a bounded-cardinality categorical
    stream: protocol type, TCP flag combination, destination-port
    bucket, etc.

    Memory scales with the number of *distinct* categories observed,
    not the number of observations -- appropriate only when that
    category count stays small (protocol type: ~4 values; TCP flags:
    a few dozen combinations at most). For genuinely high-cardinality
    distributions (exact destination IP, arbitrary domain names), use
    the Count-Min/HyperLogLog sketches in
    flow_state.source_aggregate instead -- those are built for
    unbounded-cardinality streams; this class is not, and would grow
    without limit if misused for one.
    """

    counts: dict[Hashable, int] = field(default_factory=dict)

    def observe(self, category: Hashable) -> None:
        self.counts[category] = self.counts.get(category, 0) + 1

    def entropy(self) -> float:
        return shannon_entropy(self.counts)

    def snapshot(self) -> dict:
        return {
            "entropy": self.entropy(),
            "distinct_categories": len(self.counts),
            "total_observations": sum(self.counts.values()),
        }
