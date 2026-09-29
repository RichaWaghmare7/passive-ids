"""DGA detector: Markov character-transition gate cascaded into LightGBM.

Two independent trainable stages, both shipping unfit by default --
same discipline as intel/ja4_lookup.py (no fabricated threat intel)
and models/ddos_hoeffding.py, models/c2_beaconing.py (no fabricated
training data): a plausible-looking baked-in "legitimate English
letter-frequency table" would be exactly the kind of unverified data
this project has consistently refused to invent. Fit CharMarkovModel
from a real corpus of known-legitimate domains (a public top-domains
list, or eventually the SOC verdict feedback loop's confirmed-benign
history) and the LightGBM stage from real labeled DGA/legitimate
examples, whenever that data exists.

Cascade design: the Markov gate is a cheap plausibility score (average
log-probability per character transition, relative to whatever corpus
it was fit on). gate_threshold defaults to None, meaning the gate
never short-circuits on its own -- every domain falls through to the
LightGBM stage once that's fit. This mirrors the project's broader
"don't invent thresholds, derive them from real PR curves" decision
(config/thresholds.yaml): a specific gate cutoff is exactly the kind
of number that needs real labeled data to justify, not a guess baked
into this module.

Reuses features.entropy.string_char_entropy / string_ngram_entropy
directly rather than recomputing anything -- this is the concrete
consumer for those functions, which shipped correctly unwired back
when features/entropy.py was built (no DGA model existed yet to feed
them to).

Domain-label extraction uses the same simple TLD-stripping convention
established in flow_state/dns_aggregate.py: split on "." and take the
first label. Not public-suffix-aware (doesn't handle "co.uk"-style
multi-label TLDs correctly) -- a disclosed simplification, not a
silent one, consistent with every other place this convention is used
in this project.

Verified directly before writing, not assumed:
  - LightGBM's sklearn-API classifier raises the same NotFittedError
    sklearn's own estimators do, before fit() -- checked directly,
    handled the same way as models/c2_beaconing.py's IsolationForest.
  - The Markov gate's math (Laplace-smoothed transition probabilities,
    averaged log-probability per transition) was hand-derived for a
    tiny traceable training case before implementation, and the tests
    assert those exact hand-computed values -- not just "the score for
    a weird string is lower than for a normal one."
"""
from __future__ import annotations

import math
from typing import Any

from sklearn.exceptions import NotFittedError

from passive_ids.features.entropy import string_char_entropy, string_ngram_entropy
from passive_ids.models.base import DetectionResult, Detector


class CharMarkovModel:
    """Order-N character-transition plausibility model.

    score() returns the average log-probability per character
    transition, relative to the training corpus -- higher (closer to
    0, less negative) means the string's letter transitions look like
    the corpus it was trained on; more negative means they don't.
    Not a probability itself, and not bounded to [0, 1].
    """

    def __init__(self, order: int = 2, smoothing: float = 1.0):
        self.order = order
        self.smoothing = smoothing
        self._transition_counts: dict[str, dict[str, int]] = {}
        self._context_totals: dict[str, int] = {}
        self._alphabet: set[str] = set()
        self.fitted = False
        self.training_examples_seen = 0

    def _contexts_and_chars(self, s: str):
        padded = ("^" * self.order) + s
        for i in range(len(s)):
            yield padded[i : i + self.order], s[i]

    def fit(self, strings: list[str]) -> None:
        for s in strings:
            for context, next_char in self._contexts_and_chars(s):
                self._alphabet.add(next_char)
                bucket = self._transition_counts.setdefault(context, {})
                bucket[next_char] = bucket.get(next_char, 0) + 1
                self._context_totals[context] = self._context_totals.get(context, 0) + 1
        self.fitted = True
        self.training_examples_seen = len(strings)

    def score(self, s: str) -> float | None:
        if not self.fitted or not s:
            return None
        alphabet_size = max(len(self._alphabet), 1)
        log_probs = []
        for context, next_char in self._contexts_and_chars(s):
            context_count = self._context_totals.get(context, 0)
            char_count = self._transition_counts.get(context, {}).get(next_char, 0)
            p = (char_count + self.smoothing) / (context_count + self.smoothing * alphabet_size)
            log_probs.append(math.log(p))
        return sum(log_probs) / len(log_probs)


def _digit_ratio(s: str) -> float:
    return sum(c.isdigit() for c in s) / len(s) if s else 0.0


def _vowel_ratio(s: str) -> float:
    return sum(c.lower() in "aeiou" for c in s) / len(s) if s else 0.0


def _extract_label(domain: str) -> str:
    """First label only -- see module docstring re: TLD-stripping."""
    return domain.split(".")[0] if domain else ""


def _build_lgbm_features(label: str, markov_score: float | None) -> list[float]:
    return [
        string_char_entropy(label),
        string_ngram_entropy(label, n=2),
        float(len(label)),
        _digit_ratio(label),
        _vowel_ratio(label),
        markov_score if markov_score is not None else 0.0,
    ]


class DGACascadeDetector(Detector):
    threat_class = "dga"
    model_name = "markov_lightgbm_cascade"
    is_online_learner = False

    def __init__(
        self,
        markov_order: int = 2,
        gate_threshold: float | None = None,
        **lgbm_kwargs: Any,
    ) -> None:
        import lightgbm as lgb

        self._markov = CharMarkovModel(order=markov_order)
        lgbm_kwargs.setdefault("verbose", -1)  # quiet by default; caller can override
        self._lgbm = lgb.LGBMClassifier(**lgbm_kwargs)
        self._lgbm_fitted = False
        self.gate_threshold = gate_threshold

    def fit_markov_gate(self, legitimate_domains: list[str]) -> None:
        self._markov.fit([_extract_label(d) for d in legitimate_domains])

    def fit_classifier(self, domains: list[str], labels: list[int]) -> None:
        markov_scores = [self._markov.score(_extract_label(d)) for d in domains]
        X = [
            _build_lgbm_features(_extract_label(d), ms)
            for d, ms in zip(domains, markov_scores)
        ]
        self._lgbm.fit(X, labels)
        self._lgbm_fitted = True

    def score(self, feature_vector: dict[str, Any]) -> DetectionResult:
        domain = feature_vector.get("domain", "") or ""
        label = _extract_label(domain)
        markov_score = self._markov.score(label)

        if (
            self.gate_threshold is not None
            and markov_score is not None
            and markov_score >= self.gate_threshold
        ):
            return DetectionResult(
                threat_class=self.threat_class,
                model_name=self.model_name,
                raw_score=0.0,
                evidence={
                    "status": "gated_benign",
                    "markov_score": markov_score,
                    "gate_threshold": self.gate_threshold,
                },
            )

        if not self._lgbm_fitted:
            return DetectionResult(
                threat_class=self.threat_class,
                model_name=self.model_name,
                raw_score=0.0,
                evidence={"status": "untrained", "markov_score": markov_score},
            )

        x = _build_lgbm_features(label, markov_score)
        try:
            proba_row = self._lgbm.predict_proba([x])[0]
        except NotFittedError:
            return DetectionResult(
                threat_class=self.threat_class,
                model_name=self.model_name,
                raw_score=0.0,
                evidence={"status": "untrained", "markov_score": markov_score},
            )

        # Look up the positive-class column by class label rather than
        # assuming index 1 -- robust to class ordering regardless of
        # how classes_ happens to be sorted.
        classes = list(self._lgbm.classes_)
        positive_index = classes.index(1) if 1 in classes else classes.index(True)
        raw_score = float(proba_row[positive_index])

        return DetectionResult(
            threat_class=self.threat_class,
            model_name=self.model_name,
            raw_score=raw_score,
            evidence={
                "status": "trained",
                "markov_score": markov_score,
                "domain_label": label,
            },
        )
