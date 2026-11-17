"""Explicit decision rules: the belief → action step is code, not free LLM choice."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def expected_payoffs(pred_opp: np.ndarray, payoff: np.ndarray) -> np.ndarray:
    """payoff[i, j] = my payoff for my action i vs their action j."""
    return np.asarray(payoff, dtype=float) @ np.asarray(pred_opp, dtype=float)


def best_response(pred_opp: np.ndarray, payoff: np.ndarray, rng: np.random.Generator | None = None) -> int:
    eu = expected_payoffs(pred_opp, payoff)
    best = np.flatnonzero(eu >= eu.max() - 1e-12)
    if len(best) == 1 or rng is None:
        return int(best[0])
    return int(rng.choice(best))


def plan_values(predict_fn, history: tuple, payoff: np.ndarray, depth: int) -> np.ndarray:
    """Expectimax over `depth` rounds: value of each of my actions now, when the opponent's next action
    follows predict_fn(history) and I play optimally afterwards. depth = 1 is the myopic expected payoff."""
    payoff = np.asarray(payoff, dtype=float)
    p = np.asarray(predict_fn(history), dtype=float)
    vals = payoff @ p
    if depth <= 1:
        return vals
    for a in range(payoff.shape[0]):
        for o in np.flatnonzero(p > 1e-9):
            vals[a] += p[o] * plan_values(predict_fn, tuple(history) + ((a, int(o)),), payoff, depth - 1).max()
    return vals


def plan_response(predict_fn, history: tuple, payoff: np.ndarray, depth: int,
                  rng: np.random.Generator | None = None) -> int:
    """Best first action of a depth-limited expectimax plan (ties broken like best_response)."""
    v = plan_values(predict_fn, history, payoff, depth)
    best = np.flatnonzero(v >= v.max() - 1e-9)
    if len(best) == 1 or rng is None:
        return int(best[0])
    return int(rng.choice(best))


def softmax(u: np.ndarray, tau: float) -> np.ndarray:
    z = (np.asarray(u, dtype=float) - np.max(u)) / max(tau, 1e-9)
    e = np.exp(z)
    return e / e.sum()


def softmax_response(
    pred_opp: np.ndarray, payoff: np.ndarray, tau: float, rng: np.random.Generator
) -> tuple[int, np.ndarray]:
    dist = softmax(expected_payoffs(pred_opp, payoff), tau)
    return int(rng.choice(len(dist), p=dist)), dist


def regret(true_opp: np.ndarray, payoff: np.ndarray, action: int) -> float:
    """Myopic regret of `action` against the opponent's true mixed action this round."""
    eu = expected_payoffs(true_opp, payoff)
    return float(eu.max() - eu[action])


@dataclass
class GateDecision:
    commit: bool
    p_positive: float
    mean: float
    reason: str


def gate_commit(delta_samples: np.ndarray, p_threshold: float = 0.9, min_effect: float = 0.0) -> GateDecision:
    """Commit a change only if the posterior over its effect is confidently positive and large enough."""
    d = np.asarray(delta_samples, dtype=float)
    p_pos = float((d > 0).mean())
    mean = float(d.mean())
    if p_pos < p_threshold:
        return GateDecision(False, p_pos, mean, f"P(Δ>0)={p_pos:.3f} < {p_threshold}")
    if mean < min_effect:
        return GateDecision(False, p_pos, mean, f"E[Δ]={mean:.4f} < {min_effect}")
    return GateDecision(True, p_pos, mean, "ok")
