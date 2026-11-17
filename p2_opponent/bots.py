"""Ground-truth opponent bots.

Every bot is a *pure function of the history*: `dist(history)` returns its exact mixed action, so
metrics such as KL(true ‖ predicted) and regret are computed exactly, not estimated. Bots with
an explicit model of the agent also expose `belief(history)`, their ground-truth belief over the
agent's next action.

history: tuple of (agent_action, opp_action) pairs, from the agent's perspective.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .games import Game

History = Sequence[tuple[int, int]]


def softmax(u: np.ndarray, tau: float) -> np.ndarray:
    z = (u - u.max()) / max(tau, 1e-6)
    e = np.exp(z)
    return e / e.sum()


def habit(actions: Sequence[int], n: int, w: int) -> np.ndarray:
    """Laplace-smoothed frequency of the last w actions (α = 1)."""
    counts = np.ones(n)
    for x in actions[-w:]:
        counts[x] += 1.0
    return counts / counts.sum()


class Bot:
    family = "base"

    def __init__(self, game: Game):
        self.game = game

    def dist(self, history: History) -> np.ndarray:
        raise NotImplementedError

    def belief(self, history: History) -> np.ndarray | None:
        return None

    def params(self) -> dict:
        return {}

    def describe(self) -> dict:
        return {"family": self.family, "params": self.params()}

    def _uniform(self) -> np.ndarray:
        return np.full(self.game.n_opp, 1.0 / self.game.n_opp)


class Fixed(Bot):
    family = "fixed"

    def __init__(self, game: Game, q: Sequence[float]):
        super().__init__(game)
        q = np.asarray(q, dtype=float)
        self.q = q / q.sum()

    def dist(self, history: History) -> np.ndarray:
        return self.q.copy()

    def params(self) -> dict:
        return {"q": self.q.round(4).tolist()}


class Reactive(Bot):
    """Responds to the agent's previous action through a response table (e.g. tit-for-tat, beat-last)."""

    family = "reactive"

    def __init__(self, game: Game, table: Sequence[int], eps: float, variant: str = ""):
        super().__init__(game)
        self.table = list(table)  # agent's last action -> opponent's response
        self.eps = eps
        self.variant = variant

    def dist(self, history: History) -> np.ndarray:
        if not history:
            return self._uniform()
        det = np.zeros(self.game.n_opp)
        det[self.table[history[-1][0]]] = 1.0
        return (1 - self.eps) * det + self.eps * self._uniform()

    def params(self) -> dict:
        return {"table": self.table, "eps": self.eps, "variant": self.variant}


class WSLS(Bot):
    """Win-stay lose-shift. A 'win' is a last-round payoff above the mean of the bot's payoff matrix."""

    family = "wsls"

    def __init__(self, game: Game, eps: float):
        super().__init__(game)
        self.eps = eps

    def dist(self, history: History) -> np.ndarray:
        if not history:
            return self._uniform()
        a, o = history[-1]
        pay = self.game.payoff_opp
        if pay[a, o] > pay.mean():
            det = np.zeros(self.game.n_opp)
            det[o] = 1.0
        else:
            det = np.ones(self.game.n_opp)
            det[o] = 0.0
            det /= det.sum()
        return (1 - self.eps) * det + self.eps * self._uniform()

    def params(self) -> dict:
        return {"eps": self.eps}


class KLevel(Bot):
    """Level-k reasoning over windowed habits.

    level-0 of a role is its own Laplace-smoothed habit over the last w moves. Level-k softmax-best-responds
    (temperature tau) to the other role at level k-1. k = 1 is fictitious play over a window.
    belief() returns the agent model the bot best-responds to (agent level k-1).
    """

    family = "klevel"

    def __init__(self, game: Game, k: int, w: int, tau: float):
        super().__init__(game)
        self.k, self.w, self.tau = k, w, tau

    def _level(self, j: int, role: str, mine: list[int], theirs: list[int]) -> np.ndarray:
        g = self.game
        if role == "opp":
            if j == 0:
                return habit(theirs, g.n_opp, self.w)
            return softmax(self._level(j - 1, "agent", mine, theirs) @ g.payoff_opp, self.tau)
        if j == 0:
            return habit(mine, g.n_agent, self.w)
        return softmax(g.payoff_agent @ self._level(j - 1, "opp", mine, theirs), self.tau)

    def dist(self, history: History) -> np.ndarray:
        mine = [h[0] for h in history]
        theirs = [h[1] for h in history]
        return self._level(self.k, "opp", mine, theirs)

    def belief(self, history: History) -> np.ndarray | None:
        if self.k == 0:
            return None
        mine = [h[0] for h in history]
        theirs = [h[1] for h in history]
        return self._level(self.k - 1, "agent", mine, theirs)

    def params(self) -> dict:
        return {"k": self.k, "w": self.w, "tau": self.tau}


class Cycle(Bot):
    family = "cycle"

    def __init__(self, game: Game, seq: Sequence[int], eps: float):
        super().__init__(game)
        self.seq = list(seq)
        self.eps = eps

    def dist(self, history: History) -> np.ndarray:
        det = np.zeros(self.game.n_opp)
        det[self.seq[len(history) % len(self.seq)]] = 1.0
        return (1 - self.eps) * det + self.eps * self._uniform()

    def params(self) -> dict:
        return {"seq": self.seq, "eps": self.eps}


class Switching(Bot):
    """Plays `first` until round t_switch, then `second`. Both see the full history."""

    family = "switching"

    def __init__(self, game: Game, first: Bot, second: Bot, t_switch: int):
        super().__init__(game)
        self.first, self.second, self.t_switch = first, second, t_switch

    def active(self, history: History) -> Bot:
        return self.first if len(history) < self.t_switch else self.second

    def dist(self, history: History) -> np.ndarray:
        return self.active(history).dist(history)

    def belief(self, history: History) -> np.ndarray | None:
        return self.active(history).belief(history)

    def params(self) -> dict:
        return {"first": self.first.describe(), "second": self.second.describe(), "t_switch": self.t_switch}


# ---------------------------------------------------------------- v5 families outside the seed menu
# Each one has structure that no SEED_LIBRARY program (at any grid point) and not the last-round catch-all
# express: hidden state, two-round memory, second-order agent model, slow time drift. Their true-class programs
# are seed_programs.NOVEL_PROGRAMS; scripts/gap_check.py verifies the gap before any LLM run.


def opp_won(game: Game, a: int, o: int) -> bool:
    """The WSLS notion of a win: the bot's payoff is above the mean entry of its payoff matrix."""
    return bool(game.payoff_opp[a, o] > game.payoff_opp.mean())


class LoseKShift(Bot):
    """Holds an action; after k consecutive non-wins it switches to its best reply to the agent's last action."""

    family = "lose_k_shift"

    def __init__(self, game: Game, k: int, eps: float, start: int):
        super().__init__(game)
        self.k, self.eps, self.start = k, eps, start

    def current(self, history: History) -> int:
        cur, losses = self.start, 0
        for a, o in history:
            losses = 0 if opp_won(self.game, a, o) else losses + 1
            if losses >= self.k:
                cur, losses = int(np.argmax(self.game.payoff_opp[a])), 0
        return cur

    def dist(self, history: History) -> np.ndarray:
        det = np.zeros(self.game.n_opp)
        det[self.current(history)] = 1.0
        return (1 - self.eps) * det + self.eps * self._uniform()

    def params(self) -> dict:
        return {"k": self.k, "eps": self.eps, "start": self.start}


class React2(Bot):
    """Responds to the agent's last TWO actions through a response table; uniform in the first two rounds."""

    family = "react2"

    def __init__(self, game: Game, table: Sequence[Sequence[int]], eps: float):
        super().__init__(game)
        self.table = [list(r) for r in table]  # table[agent action at t-2][agent action at t-1]
        self.eps = eps

    def dist(self, history: History) -> np.ndarray:
        if len(history) < 2:
            return self._uniform()
        det = np.zeros(self.game.n_opp)
        det[self.table[history[-2][0]][history[-1][0]]] = 1.0
        return (1 - self.eps) * det + self.eps * self._uniform()

    def params(self) -> dict:
        return {"table": self.table, "eps": self.eps}


class SeqBR(Bot):
    """Predicts the agent's next action from its first-order transition counts (Laplace), then softmax-best-responds."""

    family = "seq_br"

    def __init__(self, game: Game, tau: float):
        super().__init__(game)
        self.tau = tau

    def belief(self, history: History) -> np.ndarray | None:
        n = self.game.n_agent
        if not history:
            return np.full(n, 1.0 / n)
        trans = np.ones((n, n))
        for t in range(1, len(history)):
            trans[history[t - 1][0], history[t][0]] += 1.0
        row = trans[history[-1][0]]
        return row / row.sum()

    def dist(self, history: History) -> np.ndarray:
        return softmax(self.belief(history) @ self.game.payoff_opp, self.tau)

    def params(self) -> dict:
        return {"tau": self.tau}


class BlockDrift(Bot):
    """A fixed action distribution q whose labels rotate by one every `block` rounds: P(j at t) = q[(j - t//block) mod n]."""

    family = "block_drift"

    def __init__(self, game: Game, q: Sequence[float], block: int):
        super().__init__(game)
        q = np.asarray(q, dtype=float)
        self.q = q / q.sum()
        self.block = block

    def dist(self, history: History) -> np.ndarray:
        return np.roll(self.q, len(history) // self.block)

    def params(self) -> dict:
        return {"q": self.q.round(4).tolist(), "block": self.block}


# ---------------------------------------------------------------- sampling with disjoint dev/test ranges


@dataclass(frozen=True)
class SplitSpec:
    tau: tuple[float, ...]
    window: tuple[int, ...]
    eps: tuple[float, ...]
    cycle_len: tuple[int, ...]
    dirichlet_alpha: float
    switch_frac: tuple[float, float]
    k: tuple[int, ...]
    lose_k: tuple[int, ...] = (2, 3)  # v5 families
    block: tuple[int, ...] = (8, 12)


SPLITS = {
    "dev": SplitSpec(tau=(0.05, 0.2, 0.5), window=(5, 20), eps=(0.05, 0.15), cycle_len=(2, 3),
                     dirichlet_alpha=0.5, switch_frac=(0.3, 0.5), k=(0, 2)),
    "test": SplitSpec(tau=(0.1, 0.35, 1.0), window=(10, 40), eps=(0.1, 0.2), cycle_len=(4, 5),
                      dirichlet_alpha=0.3, switch_frac=(0.5, 0.7), k=(0, 2, 3), lose_k=(3, 4), block=(15, 20)),
}

FAMILIES = ("fixed", "reactive", "wsls", "fictitious", "klevel", "cycle", "switching")
BASE_FAMILIES = tuple(f for f in FAMILIES if f != "switching")
# v5: families outside the seed menu. Not in BASE_FAMILIES, so switching bots (and every earlier run) are unchanged.
NOVEL_FAMILIES = ("lose_k_shift", "react2", "seq_br", "block_drift")


def _reactive_table(game: Game, variant: str, rng: np.random.Generator) -> list[int]:
    n_a, n_o = game.n_agent, game.n_opp
    if variant == "copy" and n_a == n_o:  # tit-for-tat in IPD
        return list(range(n_a))
    if variant == "beat_last" and game.name.startswith("rps"):
        # the opponent action that beats the agent's last action
        return [int(np.argmax(game.payoff_opp[a])) for a in range(n_a)]
    return [int(rng.integers(n_o)) for _ in range(n_a)]


def sample_bot(game: Game, family: str, split: str, rng: np.random.Generator, horizon: int = 200) -> Bot:
    s = SPLITS[split]
    if family == "fixed":
        q = rng.dirichlet(np.full(game.n_opp, s.dirichlet_alpha))
        return Fixed(game, q)
    if family == "reactive":
        variant = str(rng.choice(["copy", "beat_last", "random_map"]))
        return Reactive(game, _reactive_table(game, variant, rng), float(rng.choice(s.eps)), variant)
    if family == "wsls":
        return WSLS(game, float(rng.choice(s.eps)))
    if family == "fictitious":
        return KLevel(game, 1, int(rng.choice(s.window)), float(rng.choice(s.tau)))
    if family == "klevel":
        return KLevel(game, int(rng.choice(s.k)), int(rng.choice(s.window)), float(rng.choice(s.tau)))
    if family == "cycle":
        L = int(rng.choice(s.cycle_len))
        seq = [int(x) for x in rng.integers(game.n_opp, size=L)]
        while len(set(seq)) == 1 and game.n_opp > 1:
            seq = [int(x) for x in rng.integers(game.n_opp, size=L)]
        return Cycle(game, seq, float(rng.choice(s.eps)))
    if family == "switching":
        f1, f2 = rng.choice(BASE_FAMILIES, size=2, replace=False)
        t = int(horizon * rng.uniform(*s.switch_frac))
        return Switching(game, sample_bot(game, str(f1), split, rng), sample_bot(game, str(f2), split, rng), t)
    if family == "lose_k_shift":
        return LoseKShift(game, int(rng.choice(s.lose_k)), float(rng.choice(s.eps)), int(rng.integers(game.n_opp)))
    if family == "react2":
        table = rng.integers(game.n_opp, size=(game.n_agent, game.n_agent))
        while len({int(x) for x in table.reshape(-1)}) == 1 and game.n_opp > 1:
            table = rng.integers(game.n_opp, size=(game.n_agent, game.n_agent))
        return React2(game, table.tolist(), float(rng.choice(s.eps)))
    if family == "seq_br":
        return SeqBR(game, float(rng.choice(s.tau)))
    if family == "block_drift":
        return BlockDrift(game, rng.dirichlet(np.full(game.n_opp, s.dirichlet_alpha)), int(rng.choice(s.block)))
    raise ValueError(f"unknown family {family!r}")


# ---------------------------------------------------------------- exact rule descriptions (for B4: LLM told the true type)

FAMILY_RULES = {
    "fixed": "Ignores history; every round samples its action from the fixed distribution q (over opponent actions).",
    "reactive": "At t=0: uniform. At t>=1: with prob 1-eps plays table[agent's action at t-1] (an opponent action index), "
                "otherwise uniform over all actions.",
    "wsls": "Win-stay lose-shift. At t=0: uniform. If its payoff at t-1 was above the mean entry of its payoff matrix, "
            "it repeats its last action, otherwise it picks uniformly among its other actions; with prob eps it instead plays uniformly.",
    "klevel": "Level-k reasoning. Level-0 of a player = Laplace-smoothed frequency (+1 per action) of that player's own last w actions. "
              "Level-j opponent = softmax(tau) best response to the agent at level j-1; level-j agent = softmax(tau) best response "
              "to the opponent at level j-1. The bot plays opponent level k (k=1 is fictitious play over a window).",
    "cycle": "Plays seq[t mod len(seq)] at round t (0-based) with prob 1-eps, otherwise uniform.",
    "switching": "Plays strategy `first` for rounds t < t_switch and strategy `second` afterwards (both see the full history).",
    "lose_k_shift": "Holds a current action (initially `start`). After each round it counts consecutive rounds whose payoff was not "
                    "above the mean entry of its payoff matrix; at k of them it switches to its best reply to the agent's "
                    "action in that round and resets the count. Plays the current action with prob 1-eps, otherwise uniform.",
    "react2": "At t<2: uniform. At t>=2: with prob 1-eps plays table[agent's action at t-2][agent's action at t-1], "
              "otherwise uniform.",
    "seq_br": "Predicts the agent's next action from the agent's transition counts (+1 Laplace) out of its last action, "
              "then plays softmax(tau) of its expected payoffs against that prediction.",
    "block_drift": "At round t plays action j with probability q[(j - floor(t / block)) mod n_opp].",
}


def describe_rules(bot: Bot) -> str:
    """Exact natural-language description of a bot's rule and parameters (action indices are 0-based)."""
    if isinstance(bot, Switching):
        return (f"{FAMILY_RULES['switching']}\nt_switch = {bot.t_switch}\n"
                f"first: {describe_rules(bot.first)}\nsecond: {describe_rules(bot.second)}")
    return f"family={bot.family}: {FAMILY_RULES[bot.family]} parameters={bot.params()}"
