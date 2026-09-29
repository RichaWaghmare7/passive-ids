"""Welford's online algorithm: streaming mean/variance without storing history.

Single-pass, O(1) memory regardless of how many samples have been
seen -- this is what makes it compatible with the project's
incremental-model constraint (see models/base.py's `is_online_learner`
distinction: this has no separate "training" step, it's simply always
current).

On numerical stability -- checked empirically, not just asserted:
computing variance as E[X^2] - E[X]^2 (the naive, non-Welford way)
loses precision through catastrophic cancellation when the values
being squared are large relative to the true variance. For inter
-arrival times at realistic beacon-interval scale (seconds to
minutes), this isn't actually a problem -- the naive formula agrees
with the true variance there. It becomes a real problem specifically
at large absolute scales: feeding the naive formula raw epoch
timestamps (~1e9) with tiny variance around them produces a variance
estimate that is not just wrong but *negative* (verified: true
variance ~3.4e-7, naive formula gives -128). WelfordStats is safe at
any scale a caller might reasonably use it at, including raw
timestamps directly, without the caller needing to reason about
whether their input happens to be in the danger zone.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WelfordStats:
    """Generic streaming mean/variance/CV for any scalar sequence.

    Not specific to timestamps or IAT -- reusable for packet sizes,
    byte counts, or any other streaming scalar a future feature needs.
    """

    n: int = 0
    mean: float = 0.0
    _m2: float = 0.0

    def update(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self._m2 += delta * delta2

    @property
    def variance(self) -> float:
        """Population variance. 0.0 for n < 2 -- there's no notion of
        spread from a single sample, and returning 0.0 rather than
        raising keeps early-flow feature vectors well-defined."""
        if self.n < 2:
            return 0.0
        return self._m2 / self.n

    @property
    def sample_variance(self) -> float:
        """Bessel-corrected (n-1) variance -- the less-biased estimator,
        worth using over `variance` once n is small enough that the
        correction meaningfully matters (rule of thumb: n < ~30)."""
        if self.n < 2:
            return 0.0
        return self._m2 / (self.n - 1)

    @property
    def std(self) -> float:
        return self.variance**0.5

    @property
    def coefficient_of_variation(self) -> float:
        """std / mean -- the "jitter coefficient" from the architecture
        docs. Low CV = suspiciously regular timing (the C2 beaconing
        signature this feeds); high CV = irregular, human-like
        traffic. Returns 0.0 for a non-positive mean rather than
        raising -- a burst with ~zero mean IAT isn't a computation
        error, it's just not meaningfully describable as "regular."
        """
        if self.mean <= 0:
            return 0.0
        return self.std / self.mean

    def snapshot(self) -> dict:
        return {
            "n": self.n,
            "mean": self.mean,
            "variance": self.variance,
            "std": self.std,
            "coefficient_of_variation": self.coefficient_of_variation,
        }


class WelfordIAT:
    """Converts a stream of arrival timestamps into inter-arrival times
    and feeds them to a WelfordStats instance.

    This is the boundary between "raw arrival times" (what
    flow_state.confirmed_table.ConfirmedFlowEntry tracks via
    last_seen) and "IAT statistics" (what WelfordStats computes) --
    kept as a separate thin wrapper so WelfordStats itself stays
    general-purpose rather than hardcoded to timestamp differencing.
    """

    def __init__(self) -> None:
        self.stats = WelfordStats()
        self._last_ts: float | None = None

    def observe(self, timestamp: float) -> float | None:
        """Record one arrival. Returns the computed IAT, or None on the
        very first observation (no prior timestamp to difference
        against yet -- not an error, just undefined for a single point).
        """
        if self._last_ts is None:
            self._last_ts = timestamp
            return None
        iat = timestamp - self._last_ts
        self._last_ts = timestamp
        self.stats.update(iat)
        return iat

    def snapshot(self) -> dict:
        return self.stats.snapshot()
