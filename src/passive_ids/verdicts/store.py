"""Append-only verdict store.

In-memory (a list) for now -- the real backing store (a database, an
append-only log) is a deployment decision for whoever runs this, not
something to invent here. This module's job is the query surface any
backing store needs to support (add, filter, extract training labels),
kept separate from storage mechanics so swapping the backing store
later doesn't change callers.

training_pairs()'s label-derivation logic is the part that needed real
thought, not just plumbing: Verdict carries both model_predicted_class
and confirmed_threat_class specifically to support
TRUE_POSITIVE_WRONG_CLASS (something bad happened, but not what the
model said). That means a single verdict can be relevant to building
training data for a class it was never predicted as -- e.g. a flow
the DDoS model flagged that the analyst confirms was actually a port
scan is a POSITIVE training example for port_scanning (confirmed
-true, even though port_scanning's own detector never fired) and a
NEGATIVE one for ddos (predicted-but-wrong). training_pairs(target)
only includes a verdict when target is either the model's prediction
or the analyst's confirmed truth -- anything else genuinely isn't
informative about target and is skipped, not guessed at.

Does NOT hold feature vectors, only flow_ids and labels -- joining
flow_id + feature_vector_version back to the actual feature vector
that produced the original score is feedback/retrain.py's job (not
built yet), which is exactly why Verdict carries
feature_vector_version at all (see verdicts/schema.py): a training
pipeline built later needs it to avoid silently joining a label to a
feature vector produced by different feature-engineering code.
"""
from __future__ import annotations

from passive_ids.verdicts.schema import SampleReason, Verdict, VerdictLabel


class VerdictStore:
    def __init__(self) -> None:
        self._verdicts: list[Verdict] = []

    def __len__(self) -> int:
        return len(self._verdicts)

    def add(self, verdict: Verdict) -> None:
        self._verdicts.append(verdict)

    def all(self) -> list[Verdict]:
        return list(self._verdicts)

    def by_sample_reason(self, reason: SampleReason) -> list[Verdict]:
        return [v for v in self._verdicts if v.sample_reason == reason]

    def training_pairs(self, threat_class: str) -> tuple[list[str], list[int]]:
        """Returns (flow_ids, labels) for verdicts relevant to
        threat_class. INCONCLUSIVE is always excluded -- see
        verdicts/schema.py: never folded into either label, not
        defaulted to benign.
        """
        flow_ids: list[str] = []
        labels: list[int] = []

        for v in self._verdicts:
            if v.verdict == VerdictLabel.INCONCLUSIVE:
                continue

            is_confirmed_this_class = v.confirmed_threat_class == threat_class
            is_predicted_this_class = v.model_predicted_class == threat_class
            if not (is_confirmed_this_class or is_predicted_this_class):
                continue  # not informative about this class -- skip, don't guess

            if is_confirmed_this_class:
                label = 1
            elif v.verdict == VerdictLabel.FALSE_POSITIVE and is_predicted_this_class:
                label = 0
            elif v.verdict == VerdictLabel.TRUE_POSITIVE_WRONG_CLASS and is_predicted_this_class:
                label = 0  # this class's detector fired, but it wasn't actually this class
            else:
                continue

            flow_ids.append(v.flow_id)
            labels.append(label)

        return flow_ids, labels

    def missed_positive_rate(self, threat_class: str) -> float | None:
        """Fraction of RANDOM-sampled (unbiased) verdicts confirmed as
        threat_class -- a partial signal toward recall, not a full
        recall calculation. Random samples are drawn specifically from
        sub-threshold traffic (see verdicts/sampling.py), so a
        confirmed-positive result here is, by construction, something
        this class's detector did not flag. Turning this into a full
        recall number additionally needs the sampling rate and total
        traffic volume -- neither of which this store tracks -- so
        that computation belongs one level up (feedback/retrain.py,
        not built yet), not here.

        Returns None if there are no random-sample verdicts yet to
        compute this from, rather than a misleading 0.0.
        """
        random_verdicts = [
            v
            for v in self._verdicts
            if v.sample_reason == SampleReason.RANDOM and v.verdict != VerdictLabel.INCONCLUSIVE
        ]
        if not random_verdicts:
            return None
        confirmed_positive = sum(1 for v in random_verdicts if v.confirmed_threat_class == threat_class)
        return confirmed_positive / len(random_verdicts)
