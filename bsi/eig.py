"""Expected information gain for choosing the next intervention.

EIG(a) = I(θ; y | a) = H[Σ_h w_h p_h(y|a)] − Σ_h w_h H[p_h(y|a)]   (exact for finite outcomes)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom


def entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * np.log(p), 0.0)
    return -t.sum(axis=axis)


def mutual_information(weights: np.ndarray, probs: np.ndarray) -> float:
    """weights: (H,), probs: (H, Y) rows = p(y | h, a)."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    P = np.asarray(probs, dtype=float)
    return float(max(0.0, entropy(w @ P) - w @ entropy(P, axis=1)))


def binomial_outcomes(p_success: np.ndarray, n: int) -> np.ndarray:
    """Outcome = number of successes in n i.i.d. rollouts. (H,) success probs -> (H, n+1) pmf."""
    k = np.arange(n + 1)
    return binom.pmf(k[None, :], n, np.asarray(p_success, dtype=float)[:, None])


def eig_mc(
    weights: np.ndarray,
    sample: Callable[[int, np.random.Generator], object],
    loglik: Callable[[int, object], float],
    n_samples: int = 64,
    rng: np.random.Generator | None = None,
) -> float:
    """Monte Carlo EIG for outcome spaces too large to enumerate.

    sample(h, rng) draws y ~ p(y | h, a); loglik(h, y) = log p(y | h, a).
    Estimator: E_{h~w, y~p(·|h)} [log p(y|h) − log Σ_h' w_h' p(y|h')].
    """
    rng = rng or np.random.default_rng()
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    lw = np.log(np.maximum(w, 1e-300))
    total = 0.0
    for _ in range(n_samples):
        h = int(rng.choice(len(w), p=w))
        y = sample(h, rng)
        lls = np.array([loglik(j, y) for j in range(len(w))])
        total += lls[h] - logsumexp(lw + lls)
    return max(0.0, total / n_samples)


@dataclass
class Candidate:
    name: str
    probs: np.ndarray  # (H, Y): predicted outcome distribution under each hypothesis
    cost: float = 1.0


@dataclass
class Scored:
    name: str
    eig: float
    cost: float

    @property
    def rate(self) -> float:
        return self.eig / max(self.cost, 1e-12)


def rank_interventions(weights: np.ndarray, candidates: Sequence[Candidate]) -> list[Scored]:
    scored = [Scored(c.name, mutual_information(weights, c.probs), c.cost) for c in candidates]
    return sorted(scored, key=lambda s: s.rate, reverse=True)


def select_intervention(
    weights: np.ndarray, candidates: Sequence[Candidate], min_eig: float = 1e-3
) -> Scored | None:
    """Best EIG-per-cost candidate, or None when nothing is worth running (stop and decide)."""
    ranked = rank_interventions(weights, candidates)
    best = ranked[0] if ranked else None
    return best if best is not None and best.eig >= min_eig else None
