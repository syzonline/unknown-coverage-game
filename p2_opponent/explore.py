"""v6 exploration: in short windows the agent sometimes plays an action chosen to tell its opponent models apart.

Which rounds explore is a pure function of (episode key, round, rate): every arm with the same rate explores in the same
rounds (common random numbers), the rounds of a lower rate are a subset of those of a higher one, and the agent's own
rng is never touched, so agents without exploration behave exactly as before.

The exploring action maximises the one-step expected information gain about the opponent model: our action now does
not change the opponent's current action (moves are simultaneous) but it shapes the context of the next round. For
each action a, EIG(a) = Σ_o p(o) · I(model; next opponent action | history + (a, o)), with p the current predictive.
When no action separates the models (EIG below `min_eig`), the action is uniform at random.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from bsi.eig import mutual_information

# window starts shared by every v6 arm and control, so exploration rounds pair across arms whatever their decision
# schedule (the v6 full arms decide at these rounds)
EXPLORE_STARTS = (10, 24, 38, 52, 66, 80, 94, 108, 122)


def _unit(key: str) -> float:
    return int(hashlib.sha1(key.encode()).hexdigest()[:15], 16) / 16**15


@dataclass(frozen=True)
class ExplorePolicy:
    rate: float
    window: int = 10
    end_t: int = 130
    starts: tuple[int, ...] = EXPLORE_STARTS
    min_eig: float = 1e-3

    def __post_init__(self):
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError(f"explore rate {self.rate} not in [0, 1]")

    @classmethod
    def from_opts(cls, opts: dict | None) -> "ExplorePolicy | None":
        if not opts or not opts.get("rate"):
            return None
        kw = {k: opts[k] for k in ("rate", "window", "end_t", "min_eig") if k in opts}
        if "starts" in opts:
            kw["starts"] = tuple(int(s) for s in opts["starts"])
        return cls(**kw)

    def in_window(self, t: int) -> bool:
        return t < self.end_t and any(s <= t < s + self.window for s in self.starts)

    def explores(self, episode_key: str, t: int) -> bool:
        """t = index of the round about to be played."""
        return self.in_window(t) and _unit(f"explore:{episode_key}:{t}") < self.rate

    def choose(self, episode_key: str, t: int, history: tuple, pred: np.ndarray, n_agent: int,
               models: Callable[[list[tuple]], tuple[np.ndarray, np.ndarray]]) -> tuple[int, float]:
        """models(contexts) -> (weights (H,), probs (H, C, n_opp)) of the opponent models on each context.
        Returns (action, its EIG)."""
        n_opp = len(pred)
        ctx = [tuple(history) + ((a, o),) for a in range(n_agent) for o in range(n_opp)]
        w, P = models(ctx)
        eig = np.zeros(n_agent)
        if len(w) > 1:
            for a in range(n_agent):
                eig[a] = sum(pred[o] * mutual_information(w, P[:, a * n_opp + o, :]) for o in range(n_opp))
        if eig.max() >= self.min_eig:
            best = np.flatnonzero(eig >= eig.max() - 1e-12)
            a = int(best[int(_unit(f"tie:{episode_key}:{t}") * len(best))])
        else:
            a = int(_unit(f"uniform:{episode_key}:{t}") * n_agent)
        return a, float(eig[a])


def model_predictions(hyps: Sequence, weights: Sequence[float], contexts: list[tuple], pool, other=None,
                      other_weight: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Predictions of program hypotheses (one sandbox batch) and optionally the catch-all on every context.
    Programs that fail on any context are left out. Returns (weights, probs (H, C, n_opp))."""
    from .hypotheses import normalize_prediction

    calls = [(h.id, "model", (c, h.params, h.game_info)) for h in hyps for c in contexts]
    res = pool.call_batch(calls) if calls else []
    ws, ps = [], []
    for k, (h, wt) in enumerate(zip(hyps, weights)):
        rows = res[k * len(contexts):(k + 1) * len(contexts)]
        try:
            if any(isinstance(r, Exception) for r in rows):
                continue
            arr = np.array([normalize_prediction(r, h.n) for r in rows])
        except ValueError:
            continue
        ps.append(arr)
        ws.append(wt)
    if other is not None:
        ps.append(np.array([other.predict(c) for c in contexts]))
        ws.append(other_weight)
    if not ws:
        return np.zeros(0), np.zeros((0, len(contexts), 0))
    w = np.asarray(ws, dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
    return w, np.stack(ps)
