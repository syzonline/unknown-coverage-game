"""Two-player simultaneous-move games played repeatedly.

Convention: payoff_agent[i, j] / payoff_opp[i, j] = payoff when the agent plays i and the opponent plays j.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Game:
    name: str
    payoff_agent: np.ndarray
    payoff_opp: np.ndarray
    agent_actions: tuple[str, ...]
    opp_actions: tuple[str, ...]
    description: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def n_agent(self) -> int:
        return len(self.agent_actions)

    @property
    def n_opp(self) -> int:
        return len(self.opp_actions)

    def info(self) -> dict:
        """Plain-Python view handed to sandboxed opponent-model programs."""
        return {
            "name": self.name,
            "n_agent": self.n_agent,
            "n_opp": self.n_opp,
            "payoff_agent": self.payoff_agent.tolist(),
            "payoff_opp": self.payoff_opp.tolist(),
            "agent_actions": list(self.agent_actions),
            "opp_actions": list(self.opp_actions),
        }


def rps(n: int = 3) -> Game:
    """Generalized rock-paper-scissors (n odd): i beats j iff (i - j) mod n ∈ {1, …, (n-1)/2}. Zero-sum."""
    assert n % 2 == 1 and n >= 3
    names = ("rock", "paper", "scissors") if n == 3 else tuple(f"a{i}" for i in range(n))
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = (i - j) % n
            if d == 0:
                continue
            m[i, j] = 1.0 if d <= (n - 1) // 2 else -1.0
    return Game(
        f"rps{n}", m, -m, names, names,
        f"Generalized rock-paper-scissors with {n} actions; win +1, loss -1, tie 0 (zero-sum).",
    )


def ipd() -> Game:
    """Iterated prisoner's dilemma, T=5 > R=3 > P=1 > S=0. Actions: 0 = cooperate, 1 = defect."""
    R, S, T, P = 3.0, 0.0, 5.0, 1.0
    a = np.array([[R, S], [T, P]])
    return Game(
        "ipd", a, a.T.copy(), ("cooperate", "defect"), ("cooperate", "defect"),
        "Iterated prisoner's dilemma: both cooperate 3/3, both defect 1/1, defect vs cooperate 5/0.",
    )


def hide_seek(n: int = 5) -> Game:
    """Agent = seeker (+1 if same location), opponent = hider (+1 if different location)."""
    eye = np.eye(n)
    names = tuple(f"loc{i}" for i in range(n))
    return Game(
        f"hideseek{n}", eye, 1.0 - eye, names, names,
        f"Hide and seek on {n} locations: the agent (seeker) scores 1 when it picks the hider's location; "
        "the opponent (hider) scores 1 otherwise.",
    )


GAMES = {"rps3": lambda: rps(3), "rps5": lambda: rps(5), "ipd": ipd, "hideseek5": lambda: hide_seek(5)}


def make_game(name: str) -> Game:
    return GAMES[name]()
