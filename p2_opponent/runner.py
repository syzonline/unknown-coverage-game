"""Episode runner with exact metrics against the bot's true mixed action (guideline §3.7)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from bsi.decide import regret
from bsi.stats import ece, kl
from bsi.store import TraceWriter

from .agents import Agent
from .bots import Bot, Switching
from .env import RepeatedGameEnv
from .games import Game


GOOD_TRACKING_KL = 0.25


def adaptation_latency(kls: np.ndarray, t_switch: int, pre_window: int = 20, post_window: int = 10,
                       tol: float = 0.05, good: float = GOOD_TRACKING_KL) -> tuple[float, bool, bool]:
    """Rounds after the switch until the rolling mean KL is back within `tol` of its pre-switch level.

    Returns (latency, censored, applicable). censored=True: never recovered (latency = rounds remaining).
    applicable=False when the method was not tracking the bot before the switch (pre-switch KL > `good`):
    "recovering" to a poor level is trivial, so such episodes must not count as fast adaptation."""
    T = len(kls)
    pre = kls[max(0, t_switch - pre_window): t_switch]
    if t_switch >= T or not np.isfinite(pre).any():
        return float("nan"), False, False
    pre_level = float(np.nanmean(pre))
    if pre_level > good:
        return float("nan"), False, False
    target = pre_level + tol
    for t in range(t_switch, T - post_window + 1):
        w = kls[t: t + post_window]
        if np.isfinite(w).any() and np.nanmean(w) <= target:
            return float(t - t_switch), False, True
    return float(T - t_switch), True, True


def post_switch_excess_kl(kls: np.ndarray, t_switch: int, pre_window: int = 20, post_window: int = 30,
                          tol: float = 0.05) -> tuple[float, bool, float]:
    """(excess, recovered, post) for the rounds around a switch, defined for every method:
    post = mean KL over the `post_window` rounds after the switch (primary: lower = better adaptation);
    excess = post − mean KL over the `pre_window` rounds before it (small also for methods that never tracked
    the bot, so it is not a ranking on its own); recovered: by the end of the episode, i.e. the last 10 rounds'
    mean KL is back within `tol` of the pre-switch level (a window starting right at the switch does not count:
    the new strategy may not have shown yet)."""
    pre = kls[max(0, t_switch - pre_window): t_switch]
    post = kls[t_switch: t_switch + post_window]
    if t_switch >= len(kls) or not np.isfinite(pre).any() or not np.isfinite(post).any():
        return float("nan"), False, float("nan")
    pre_level = float(np.nanmean(pre))
    tail = kls[max(t_switch + 1, len(kls) - 10):]
    rec = bool(len(kls) - t_switch > 10 and np.isfinite(tail).any() and np.nanmean(tail) <= pre_level + tol)
    post_level = float(np.nanmean(post))
    return post_level - pre_level, rec, post_level


@dataclass
class EpisodeResult:
    agent: str
    bot: dict
    game: str
    seed: int
    horizon: int
    actions: list[tuple[int, int]]
    payoff: float
    regret: np.ndarray
    kl: np.ndarray        # per round, NaN when the agent made no prediction
    log_loss: np.ndarray
    preds: np.ndarray     # (T, n_opp), NaN rows when no prediction
    t_switch: int | None = None
    events: list[dict] = field(default_factory=list)
    llm_usage: dict = field(default_factory=dict)
    n_fallbacks: int = 0
    wall_s: float = 0.0
    belief: list[dict] | None = None  # per-round posterior view (PosteriorBRAgent.belief_log)

    def summary(self) -> dict:
        def m(x: np.ndarray) -> float:
            return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")

        half = self.horizon // 2
        has = np.isfinite(self.preds).all(axis=1)
        outcomes = np.array([o for _, o in self.actions])
        out = {
            "agent": self.agent,
            "bot": self.bot["family"],
            "game": self.game,
            "seed": self.seed,
            "horizon": self.horizon,
            "payoff_per_round": self.payoff / self.horizon,
            "regret": m(self.regret),
            "kl": m(self.kl),
            "kl_second_half": m(self.kl[half:]),
            "log_loss": m(self.log_loss),
            "ece": ece(self.preds[has], outcomes[has]) if has.sum() >= 10 else float("nan"),
            "pred_coverage": float(has.mean()),
            "fallback_rate": self.n_fallbacks / self.horizon,
            "llm_calls": int(self.llm_usage.get("calls", 0)),
            "llm_tokens": int(self.llm_usage.get("prompt_tokens", 0) + self.llm_usage.get("completion_tokens", 0)),
            "llm_invalid": int(self.llm_usage.get("invalid_responses", 0)),
            "llm_retries": int(self.llm_usage.get("transport_retries", 0)),
            "llm_degraded": int(self.llm_usage.get("degraded_repairs", 0)),
            "n_events": len(self.events),
            "wall_s": round(self.wall_s, 1),
        }
        if self.t_switch is not None:
            lat, censored, applicable = adaptation_latency(self.kl, self.t_switch)
            excess, recovered, post = post_switch_excess_kl(self.kl, self.t_switch)
            out.update({"t_switch": self.t_switch, "adapt_latency": lat, "adapt_censored": censored,
                        "adapt_applicable": applicable, "post_switch_kl": post, "post_switch_excess_kl": excess,
                        "adapt_recovered": recovered})
        if self.belief:
            out.update({
                "other_mass_mean": float(np.mean([b["other"] for b in self.belief])),
                "proposer_mass_final": float(self.belief[-1]["proposer"]),
                "proposer_map_frac": float(np.mean([b["map_src"] not in (None, "library") for b in self.belief])),
                "n_changepoints": sum(e.get("event") == "changepoint" for e in self.events),
            })
        return out

    def to_record(self) -> dict:
        return {**self.summary(), "bot_full": self.bot, "actions": self.actions, "kl_per_round": self.kl,
                "regret_per_round": self.regret, "preds": self.preds, "events": self.events, "llm_usage": self.llm_usage,
                "belief": self.belief}


def run_episode(game: Game, bot: Bot, agent: Agent, horizon: int, seed: int,
                writer: TraceWriter | None = None, episode_key: str = "") -> EpisodeResult:
    t0 = time.time()
    env = RepeatedGameEnv(game, bot, horizon, seed)
    agent.horizon = horizon
    agent.reset(game, np.random.default_rng(seed + 1_000_003), episode_key=episode_key or f"{game.name}:{seed}")
    regrets, kls, lls = (np.full(horizon, np.nan) for _ in range(3))
    preds = np.full((horizon, game.n_opp), np.nan)
    history = env.history
    for t in range(horizon):
        action, pred = agent.act(history)
        res = env.step(action)
        regrets[t] = regret(res.true_dist, game.payoff_agent, action)
        if pred is not None:
            preds[t] = pred
            kls[t] = kl(res.true_dist, pred)
            lls[t] = -np.log(max(float(pred[res.opp_action]), 1e-12))
        agent.observe(history, (res.agent_action, res.opp_action))
        history = env.history
    payoff = float(sum(game.payoff_agent[a, o] for a, o in history))
    out = EpisodeResult(
        agent=agent.name, bot=bot.describe(), game=game.name, seed=seed, horizon=horizon,
        actions=list(history), payoff=payoff, regret=regrets, kl=kls, log_loss=lls, preds=preds,
        t_switch=bot.t_switch if isinstance(bot, Switching) else None,
        events=list(getattr(agent, "events", [])), llm_usage=dict(agent.llm_meter or {}),
        n_fallbacks=int(getattr(agent, "n_fallbacks", 0)), wall_s=time.time() - t0,
        belief=getattr(agent, "belief_log", None),
    )
    if writer is not None:
        writer.write(out.to_record())
    return out
