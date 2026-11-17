"""Change-point detection on the posterior mixture's prequential log-score (guideline Q2.3).

A one-sided CUSUM watches for a sustained drop of the per-round log-score below its level in the current
segment. On an alarm the caller restarts the posterior a few rounds back (Posterior.restart), so evidence
from before the change stops anchoring the belief.
"""
from __future__ import annotations


class CusumDetector:
    """s_t = max(0, s_{t-1} + (mu - x_t) - k); alarm when s_t > h.

    x_t: log-score of round t; mu: mean log-score of the current segment after `burn_in` rounds.
    k: per-round slack (nats) tolerated before evidence accumulates; h: alarm threshold (nats).
    `back`: how many rounds before the alarm the new segment starts (CUSUM detects with a delay).
    """

    def __init__(self, k: float = 0.3, h: float = 4.0, burn_in: int = 15, back: int = 8):
        self.k, self.h, self.burn_in, self.back = float(k), float(h), int(burn_in), int(back)
        self.reset()

    def reset(self) -> None:
        self.s = 0.0
        self.n = 0        # rounds seen in this segment
        self.total = 0.0  # sum of log-scores after burn-in
        self.n_ref = 0

    def update(self, x: float) -> bool:
        """Feed one round's log-score; True means a change point was detected (state is then reset)."""
        self.n += 1
        if self.n <= self.burn_in:
            return False
        if self.n_ref > 0:
            mu = self.total / self.n_ref
            self.s = max(0.0, self.s + (mu - x) - self.k)
        self.total += x
        self.n_ref += 1
        if self.s > self.h:
            self.reset()
            return True
        return False

    def config(self) -> dict:
        return {"k": self.k, "h": self.h, "burn_in": self.burn_in, "back": self.back}
