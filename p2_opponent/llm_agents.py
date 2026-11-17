"""LLM baselines for Plan 2 (guideline §3.6).

B0 DirectLLMAgent        the LLM sees the history and picks an action
B1 VerbalBeliefAgent     Hypothetical-Minds style: natural-language hypotheses, scored by their hit rate;
                         the LLM states a prediction and picks the action
B3 PosteriorLLMDecideAgent  our posterior (library + proposer), but the LLM picks the action from a posterior summary
B4 OracleTypeLLMAgent    the LLM is told the bot's exact rule and parameters (an upper bound on LLM inference)

Robustness. Every LLM call carries a validator (JSON with a legal action and, where required, a valid
distribution), so malformed answers are repaired or retried inside `LLM.chat`. If a call still fails, the
agent takes a documented fallback, a uniformly random action from the episode's seeded RNG (no prediction
that round), and logs a `fallback` event. The matrix runner discards episodes whose fallback rate is too high.
`LLMUnavailable` and `BudgetExceeded` propagate and stop the run.
"""
from __future__ import annotations

import json
from typing import Callable

import numpy as np

from bsi.llm import LLM, BudgetExceeded, LLMError, LLMResult, LLMUnavailable, extract_json

from .agents import Agent, PosteriorBRAgent
from .bots import Bot, describe_rules
from .games import Game

SYSTEM = "You are a strategic player in a repeated two-player game. Your goal is to maximize your total payoff."


# ---------------------------------------------------------------- formatting / parsing


def game_block(game: Game) -> str:
    rows = []
    for i, a in enumerate(game.agent_actions):
        cells = ", ".join(f"vs {o}: {game.payoff_agent[i, j]:+g}" for j, o in enumerate(game.opp_actions))
        rows.append(f"  you play {a}: {cells}")
    return (f"Game: {game.description}\nYou are the agent. Your actions: {list(game.agent_actions)}. "
            f"Opponent actions: {list(game.opp_actions)}.\nYour payoffs:\n" + "\n".join(rows))


def history_block(game: Game, history: tuple, last: int = 30) -> str:
    T = len(history)
    if T == 0:
        return "No rounds played yet."
    start = max(0, T - last)
    lines = [f"  t={t:>3}: you={game.agent_actions[a]}, opponent={game.opp_actions[o]} (your payoff {game.payoff_agent[a, o]:+g})"
             for t, (a, o) in enumerate(history[start:], start)]
    counts = np.bincount([o for _, o in history], minlength=game.n_opp)
    recent = np.bincount([o for _, o in history[-20:]], minlength=game.n_opp)
    total = sum(game.payoff_agent[a, o] for a, o in history)
    return (f"Rounds played: {T}. Your total payoff: {total:+g}.\n"
            f"Opponent action counts, all rounds: {dict(zip(game.opp_actions, counts.tolist()))}; "
            f"last 20: {dict(zip(game.opp_actions, recent.tolist()))}.\n"
            f"Last {T - start} rounds:\n" + "\n".join(lines))


def _action_index(value, names: list[str]) -> int:
    if isinstance(value, (int, np.integer)) and 0 <= int(value) < len(names):
        return int(value)
    if isinstance(value, str):
        v = value.strip().lower()
        for i, n in enumerate(names):
            if v == n.lower():
                return i
        if v.isdigit() and int(v) < len(names):
            return int(v)
    raise ValueError(f"illegal action {value!r}; must be one of {names}")


def _distribution(value, names: list[str]) -> np.ndarray:
    if isinstance(value, dict):
        lowered = {str(k).strip().lower(): v for k, v in value.items()}
        p = np.array([float(lowered.get(n.lower(), 0.0)) for n in names])
    elif isinstance(value, list) and len(value) == len(names):
        p = np.array([float(x) for x in value])
    else:
        raise ValueError(f"prediction must map each of {names} to a probability")
    if not np.all(np.isfinite(p)) or (p < 0).any() or p.sum() <= 0:
        raise ValueError(f"invalid probabilities {p.tolist()}")
    return p / p.sum()


def parse_decision(text: str, game: Game, need_prediction: bool) -> tuple[int, np.ndarray | None, dict]:
    obj = extract_json(text)
    if not isinstance(obj, dict):
        raise ValueError("expected a JSON object")
    action = _action_index(obj.get("action"), list(game.agent_actions))
    pred = None
    if need_prediction:
        pred = _distribution(obj.get("prediction"), list(game.opp_actions))
    return action, pred, obj


def decision_validator(game: Game, need_prediction: bool, extra: Callable[[dict], None] | None = None):
    def check(res: LLMResult) -> str | None:
        try:
            _, _, obj = parse_decision(res.text, game, need_prediction)
            if extra:
                extra(obj)
        except ValueError as e:
            return str(e)
        return None

    return check


def _fmt_names(names) -> str:
    return "|".join(names)


# ---------------------------------------------------------------- base


class LLMAgentBase(Agent):
    """Shared call path: validated call → parsed decision, or a seeded-random fallback."""

    need_prediction = True

    def __init__(self, llm: LLM, model: str = "decider", temperature: float = 0.3, name: str | None = None):
        self.llm = llm
        self.model = model
        self.temperature = temperature
        if name:
            self.name = name

    def reset(self, game: Game, rng: np.random.Generator, episode_key: str = "") -> None:
        super().reset(game, rng, episode_key)
        self.llm_meter = {}
        self.n_fallbacks = 0
        self.last_pred: np.ndarray | None = None

    def _decide(self, messages: list[dict], history: tuple, extra_check=None) -> tuple[int, np.ndarray | None, dict | None]:
        t = len(history)
        try:
            res = self.llm.chat(
                messages, model=self.model, temperature=self.temperature,
                seed_tag=f"{self.name}:{self.episode_key}:t{t}",
                validate=decision_validator(self.game, self.need_prediction, extra_check), meter=self.llm_meter,
            )
            action, pred, obj = parse_decision(res.text, self.game, self.need_prediction)
            if res.attempts > 1:
                self.events.append({"t": t, "event": "repaired", "attempts": res.attempts})
            return action, pred, obj
        except (LLMUnavailable, BudgetExceeded):
            raise
        except LLMError as e:
            self.n_fallbacks += 1
            self.events.append({"t": t, "event": "fallback", "error": str(e)[:300]})
            return int(self.rng.integers(self.game.n_agent)), None, None


# ---------------------------------------------------------------- B0


class DirectLLMAgent(LLMAgentBase):
    name = "b0_llm_direct"
    need_prediction = False

    def act(self, history: tuple) -> tuple[int, np.ndarray | None]:
        user = (f"{game_block(self.game)}\n\n{history_block(self.game, history)}\n\n"
                "Choose your next action to maximize your total payoff over the remaining rounds.\n"
                f'Reply with JSON only: {{"action": "<{_fmt_names(self.game.agent_actions)}>"}}')
        action, _, _ = self._decide([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], history)
        return action, None


# ---------------------------------------------------------------- B1


class VerbalBeliefAgent(LLMAgentBase):
    """Hypothetical-Minds-style verbal beliefs.

    Each round the LLM gives, for every live hypothesis, the opponent action it predicts; code scores
    hypotheses by Laplace hit rate (hits+1)/(n+2), keeps the best `k_keep`, and the LLM states a
    prediction and an action. Every `refine_every` rounds, or when the best hit rate falls below
    `refine_below`, it may add up to two new or refined hypotheses."""

    name = "b1_verbal"

    def __init__(self, llm: LLM, model: str = "decider", k_keep: int = 3, refine_every: int = 10,
                 refine_below: float = 0.45, **kw):
        super().__init__(llm, model, **kw)
        self.k_keep = k_keep
        self.refine_every = refine_every
        self.refine_below = refine_below

    def reset(self, game: Game, rng: np.random.Generator, episode_key: str = "") -> None:
        super().reset(game, rng, episode_key)
        self.hyps: list[dict] = []  # {id, text, hits, n}
        self._next_id = 1
        self._pending: dict[str, int] = {}  # hypothesis id -> predicted opponent action this round

    @staticmethod
    def _score(h: dict) -> float:
        return (h["hits"] + 1) / (h["n"] + 2)

    def act(self, history: tuple) -> tuple[int, np.ndarray | None]:
        t = len(history)
        best = max((self._score(h) for h in self.hyps), default=0.0)
        refine = not self.hyps or t % self.refine_every == 0 or best < self.refine_below
        hyp_lines = "\n".join(f"  {h['id']}: {h['text']}  (hit rate {h['hits']}/{h['n']}, score {self._score(h):.2f})"
                              for h in self.hyps) or "  (none yet)"
        opp = _fmt_names(self.game.opp_actions)
        new_field = (f'"new_hypotheses": ["<hypothesis about the opponent strategy>", ...up to 2],'
                     if refine else "")
        user = f"""{game_block(self.game)}

{history_block(self.game, history)}

Your current hypotheses about the opponent's strategy:
{hyp_lines}
{"Propose up to 2 NEW or REFINED hypotheses that would explain the opponent's behaviour better." if refine else "Do not add hypotheses this round."}
For EVERY hypothesis (existing and new), state the opponent action it predicts next. Then give your overall
probability prediction for the opponent's next action, based on the most credible hypothesis, and choose your action.

Reply with JSON only:
{{{new_field}
 "hypothesis_predictions": {{"<hypothesis id, or new_1/new_2 for new ones>": "<{opp}>"}},
 "prediction": {{{", ".join(f'"{o}": p' for o in self.game.opp_actions)}}},
 "action": "<{_fmt_names(self.game.agent_actions)}>"}}"""
        action, pred, obj = self._decide([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], history)
        self._pending = {}
        if obj is not None:
            id_map = {}
            if refine:
                for k, text in enumerate(obj.get("new_hypotheses") or [], 1):
                    if isinstance(text, str) and text.strip() and k <= 2:
                        hid = f"H{self._next_id}"
                        self._next_id += 1
                        self.hyps.append({"id": hid, "text": text.strip()[:300], "hits": 0, "n": 0})
                        id_map[f"new_{k}"] = hid
            for key, val in (obj.get("hypothesis_predictions") or {}).items():
                hid = id_map.get(str(key), str(key))
                if any(h["id"] == hid for h in self.hyps):
                    try:
                        self._pending[hid] = _action_index(val, list(self.game.opp_actions))
                    except ValueError:
                        pass
        return action, pred

    def observe(self, history: tuple, rnd: tuple[int, int]) -> None:
        for h in self.hyps:
            if h["id"] in self._pending:
                h["n"] += 1
                h["hits"] += int(self._pending[h["id"]] == rnd[1])
        self.hyps.sort(key=self._score, reverse=True)
        if len(self.hyps) > self.k_keep:
            dropped = self.hyps[self.k_keep:]
            self.hyps = self.hyps[: self.k_keep]
            self.events.append({"t": len(history) + 1, "event": "drop_hypotheses", "ids": [h["id"] for h in dropped]})


# ---------------------------------------------------------------- B3


class PosteriorLLMDecideAgent(PosteriorBRAgent):
    """B3: same posterior as the method (library + proposer); the LLM chooses the action from a posterior summary.
    The reported prediction is the posterior predictive, so B3 and the method differ only in the belief → action step."""

    name = "b3_llm_decide"

    def __init__(self, pool, library, llm: LLM, model: str = "decider", temperature: float = 0.3, **kw):
        if kw.get("plan_depth", 0) > 1:
            raise ValueError("plan_depth has no effect on B3: the LLM chooses the action")
        kw.setdefault("name", self.name)
        super().__init__(pool, library, **kw)
        self.llm = llm
        self.model = model
        self.temperature = temperature

    def reset(self, game: Game, rng: np.random.Generator, episode_key: str = "") -> None:
        super().reset(game, rng, episode_key)
        self.n_fallbacks = 0

    def act(self, history: tuple) -> tuple[int, np.ndarray | None]:
        _, pred = super().act(history)  # refreshes predictions; discard the explicit best response
        rows = []
        for r in self.posterior.summary(top=3):
            label = "catch-all (context-conditioned frequency table)" if r["kind"] == "other" else r["statement"]
            rows.append(f"  weight {r['weight']:.2f}: {label} (fitted params {json.dumps(r['params'])})")
        user = f"""{game_block(self.game)}

{history_block(self.game, history, last=15)}

A Bayesian model of the opponent currently believes:
{chr(10).join(rows)}
Its predicted distribution of the opponent's NEXT action: {json.dumps({o: round(float(p), 3) for o, p in zip(self.game.opp_actions, pred)})}

Choose your next action to maximize your total payoff.
Reply with JSON only: {{"action": "<{_fmt_names(self.game.agent_actions)}>"}}"""
        t = len(history)
        try:
            res = self.llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                model=self.model, temperature=self.temperature, seed_tag=f"{self.name}:{self.episode_key}:t{t}",
                                validate=decision_validator(self.game, False), meter=self.llm_meter)
            action, _, _ = parse_decision(res.text, self.game, False)
        except (LLMUnavailable, BudgetExceeded):
            raise
        except LLMError as e:
            self.n_fallbacks += 1
            self.events.append({"t": t, "event": "fallback", "error": str(e)[:300]})
            action = int(self.rng.integers(self.game.n_agent))
        return action, pred


# ---------------------------------------------------------------- B4


class OracleTypeLLMAgent(LLMAgentBase):
    """B4: the LLM is given the bot's exact rule and parameters and must predict and respond."""

    name = "b4_oracle_type"

    def __init__(self, llm: LLM, bot: Bot, model: str = "decider", **kw):
        super().__init__(llm, model, **kw)
        self.bot = bot

    def act(self, history: tuple) -> tuple[int, np.ndarray | None]:
        user = f"""{game_block(self.game)}

The opponent is a bot whose exact rule is (action indices are 0-based in the order {list(self.game.opp_actions)} for the
opponent and {list(self.game.agent_actions)} for you; round t is 0-based and the next round is t={len(history)}):
{describe_rules(self.bot)}

{history_block(self.game, history)}

Using this rule, compute the probability of each opponent action next round, then choose your action.
Reply with JSON only:
{{"prediction": {{{", ".join(f'"{o}": p' for o in self.game.opp_actions)}}}, "action": "<{_fmt_names(self.game.agent_actions)}>"}}"""
        action, pred, _ = self._decide([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], history)
        return action, pred
