"""v6 residual diagnostics: where do the current opponent models predict worst?

The rounds are partitioned by a FIXED list of generic contexts, the same for every game and opponent family
(pre-registered in prereg/v6): it must not be tuned to the families outside the menu. For each cell of each partition
the excess log-loss of a model is how much better the cell's own empirical distribution of the opponent's next action
would have predicted those rounds:

    excess(cell) = Σ_{t in cell} [log q_cell(o_t) − log p_t(o_t)],   q_cell = smoothed empirical distribution

Large excess = the model misses a regularity of that context. The same partitions give compact empirical conditional
tables (P(opponent action | context)) ranked by how much the context tells about the opponent's action.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .games import Game
from .propose import outcome_label

RES = {"opp_won": "W", "opp_lost": "L", "tie": "T"}


def _res(game: Game, a: int, o: int) -> str:
    return RES[outcome_label(game, a, o)]


def _streak(game: Game, h: tuple) -> str:
    """Result of the opponent's last round and how many rounds in a row it has had that result (capped at 4)."""
    r = _res(game, *h[-1])
    k = 1
    while k < min(len(h), 4) and _res(game, *h[-1 - k]) == r:
        k += 1
    return f"{r}x{k}{'+' if k == 4 else ''}"


# name -> (minimum rounds of history, key(game, history before round t, t), description)
PARTITIONS: dict[str, tuple[int, Callable[[Game, tuple, int], object], str]] = {
    "opp_last": (1, lambda g, h, t: h[-1][1], "opponent's previous action"),
    "agent_last": (1, lambda g, h, t: h[-1][0], "agent's previous action"),
    "last_pair": (1, lambda g, h, t: h[-1], "previous round (agent, opponent)"),
    "agent_last2": (2, lambda g, h, t: (h[-2][0], h[-1][0]), "agent's last two actions (older, newer)"),
    "opp_last2": (2, lambda g, h, t: (h[-2][1], h[-1][1]), "opponent's last two actions (older, newer)"),
    "last_result": (1, lambda g, h, t: _res(g, *h[-1]), "opponent's result in the previous round (W/L/T)"),
    "result_streak": (1, lambda g, h, t: _streak(g, h), "opponent's previous result x rounds in a row (cap 4)"),
    "time_block": (0, lambda g, h, t: f"{15 * (t // 15)}-{15 * (t // 15) + 14}", "round range (blocks of 15)"),
    "t_mod_2": (0, lambda g, h, t: t % 2, "round index mod 2"),
    "t_mod_3": (0, lambda g, h, t: t % 3, "round index mod 3"),
    "t_mod_4": (0, lambda g, h, t: t % 4, "round index mod 4"),
    "t_mod_5": (0, lambda g, h, t: t % 5, "round index mod 5"),
}


def context_keys(game: Game, history: tuple) -> dict[str, list]:
    """For every partition, the key of each round t (None where the context is undefined)."""
    out = {}
    for name, (need, fn, _) in PARTITIONS.items():
        out[name] = [fn(game, history[:t], t) if t >= need else None for t in range(len(history))]
    return out


def _smoothed(counts: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return (counts + alpha) / (counts.sum() + alpha * len(counts))


@dataclass
class Cell:
    partition: str
    key: object
    n: int
    counts: list[int]
    mean_pred: list[float]
    excess: float          # total excess log-loss (nats) over the cell's rounds

    def render(self) -> str:
        return (f"{self.partition}={self.key}: n={self.n} observed={self.counts} "
                f"model mean={[round(x, 2) for x in self.mean_pred]} excess={self.excess:.2f}")


def residual_cells(game: Game, history: tuple, preds: np.ndarray, start: int = 0, min_n: int = 4, top: int = 8,
                   keys: dict[str, list] | None = None) -> list[Cell]:
    """Cells with the largest excess log-loss of `preds` (row t = the model's prediction of round t's opponent
    action), over rounds start..T-1."""
    T = len(history)
    preds = np.asarray(preds, dtype=float)
    keys = keys or context_keys(game, history)
    outc = np.array([o for _, o in history])
    cells = []
    for name, ks in keys.items():
        groups: dict[object, list[int]] = {}
        for t in range(start, T):
            if ks[t] is not None and np.all(np.isfinite(preds[t])):
                groups.setdefault(ks[t], []).append(t)
        for key, ts in groups.items():
            if len(ts) < min_n:
                continue
            counts = np.bincount(outc[ts], minlength=game.n_opp)
            q = _smoothed(counts)
            p = np.clip(preds[ts, outc[ts]], 1e-12, None)
            excess = float(np.sum(np.log(q[outc[ts]]) - np.log(p)))
            cells.append(Cell(name, key, len(ts), counts.tolist(), preds[ts].mean(0).tolist(), excess))
    cells.sort(key=lambda c: -c.excess)
    return cells[:top]


def surprising_rounds(game: Game, history: tuple, preds: np.ndarray, start: int = 0, top: int = 5) -> list[str]:
    """The rounds where the model gave the opponent's actual action the lowest probability."""
    preds = np.asarray(preds, dtype=float)
    rows = []
    for t in range(start, len(history)):
        if np.all(np.isfinite(preds[t])):
            rows.append((float(preds[t][history[t][1]]), t))
    rows.sort()
    out = []
    for p, t in rows[:top]:
        prev = history[max(0, t - 3):t]
        out.append(f"t={t}: opponent played {history[t][1]} (model gave it {p:.2f}; model's most likely "
                   f"{int(np.argmax(preds[t]))}); preceding rounds (a, o): {list(prev)}")
    return out


def conditional_tables(game: Game, history: tuple, top: int = 4, min_n: int = 3, max_rows: int = 12,
                       keys: dict[str, list] | None = None) -> list[str]:
    """Empirical P(opponent action | context) for the `top` partitions that tell most about the opponent's action
    (information gain in nats, minus a penalty of (cells − 1)·(n_opp − 1)/(2n) for small-sample optimism)."""
    T = len(history)
    if T < 2:
        return []
    keys = keys or context_keys(game, history)
    outc = np.array([o for _, o in history])
    n_o = game.n_opp
    base = _smoothed(np.bincount(outc, minlength=n_o))
    h0 = float(-(base * np.log(base)).sum())
    scored = []
    for name, ks in keys.items():
        groups: dict[object, list[int]] = {}
        for t in range(T):
            if ks[t] is not None:
                groups.setdefault(ks[t], []).append(t)
        n = sum(len(v) for v in groups.values())
        if n == 0 or len(groups) < 2:
            continue
        h_cond = 0.0
        for ts in groups.values():
            q = _smoothed(np.bincount(outc[ts], minlength=n_o))
            h_cond += len(ts) / n * float(-(q * np.log(q)).sum())
        gain = h0 - h_cond - (len(groups) - 1) * (n_o - 1) / (2 * n)
        scored.append((gain, name, groups))
    scored.sort(key=lambda x: -x[0])
    out = []
    for gain, name, groups in scored[:top]:
        lines = [f"{name} ({PARTITIONS[name][2]}); information about the opponent's action: {gain:.3f} nats"]
        rows = sorted(((k, ts) for k, ts in groups.items() if len(ts) >= min_n), key=lambda kt: -len(kt[1]))
        for k, ts in rows[:max_rows]:
            c = np.bincount(outc[ts], minlength=n_o)
            lines.append(f"  {k}: n={len(ts)} counts={c.tolist()}")
        if len(rows) > max_rows:
            lines.append(f"  ... {len(rows) - max_rows} more contexts")
        out.append("\n".join(lines))
    return out
