"""Repeated-game environment. The bot's true mixed action is computed before sampling, for exact metrics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bots import Bot
from .games import Game


@dataclass
class StepResult:
    t: int
    agent_action: int
    opp_action: int
    r_agent: float
    r_opp: float
    true_dist: np.ndarray
    bot_belief: np.ndarray | None
    done: bool


class RepeatedGameEnv:
    def __init__(self, game: Game, bot: Bot, horizon: int, seed: int):
        self.game = game
        self.bot = bot
        self.horizon = horizon
        self.seed = seed
        self.reset()

    def reset(self) -> tuple:
        self.rng = np.random.default_rng(self.seed)
        self._history: list[tuple[int, int]] = []
        return self.history

    @property
    def history(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._history)

    @property
    def t(self) -> int:
        return len(self._history)

    def step(self, agent_action: int) -> StepResult:
        if self.t >= self.horizon:
            raise RuntimeError("episode finished")
        if not 0 <= agent_action < self.game.n_agent:
            raise ValueError(f"invalid agent action {agent_action}")
        h = self.history
        true = self.bot.dist(h)
        belief = self.bot.belief(h)
        o = int(self.rng.choice(self.game.n_opp, p=true))
        self._history.append((int(agent_action), o))
        return StepResult(
            t=len(h),
            agent_action=int(agent_action),
            opp_action=o,
            r_agent=float(self.game.payoff_agent[agent_action, o]),
            r_opp=float(self.game.payoff_opp[agent_action, o]),
            true_dist=true,
            bot_belief=belief,
            done=self.t >= self.horizon,
        )
