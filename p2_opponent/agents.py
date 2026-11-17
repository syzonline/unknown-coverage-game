"""Agents for the repeated-game environment.

Every agent's act() returns (action, predicted opponent distribution or None). The prediction is
scored against the bot's true distribution.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from bsi.changepoint import CusumDetector
from bsi.decide import best_response, plan_response
from bsi.hypotheses import BackoffDirichletOther, Observation
from bsi.llm import BudgetExceeded, LLMUnavailable
from bsi.posterior import Posterior
from bsi.sandbox import SandboxPool

from .bots import Bot
from .explore import ExplorePolicy, model_predictions
from .games import Game


@dataclass
class ProgramSpec:
    id: str
    src: str
    param_grid: dict[str, list] = field(default_factory=dict)
    statement: str = ""


class Agent:
    name = "agent"
    llm_meter: dict | None = None  # per-episode LLM usage (calls, tokens, retries, ...) for agents that call an LLM
    horizon: int | None = None  # episode length, set by run_episode; planning never looks past it

    def reset(self, game: Game, rng: np.random.Generator, episode_key: str = "") -> None:
        """episode_key identifies the episode; LLM agents put it in cache keys so episodes sample independently
        while re-running the same episode (resume) hits the cache."""
        self.game = game
        self.rng = rng
        self.episode_key = episode_key
        self.events: list[dict] = []

    def act(self, history: tuple) -> tuple[int, np.ndarray | None]:
        raise NotImplementedError

    def observe(self, history: tuple, rnd: tuple[int, int]) -> None:
        """history = rounds before this one; rnd = (agent_action, opp_action) just played."""

    def close(self) -> None:
        pass

    def lookahead(self, history: tuple, depth: int) -> int:
        """Planning depth for this round: `depth`, cut to the rounds left in the episode."""
        return depth if self.horizon is None else min(depth, self.horizon - len(history))


class RandomAgent(Agent):
    name = "random"

    def act(self, history: tuple) -> tuple[int, np.ndarray | None]:
        return int(self.rng.integers(self.game.n_agent)), None


class OracleBRAgent(Agent):
    """B5: best response to the bot's true mixed action (upper bound on myopic play). With plan_depth > 1 it
    plans against the true bot over that many rounds (bots are pure functions of the history)."""

    name = "oracle_br"

    def __init__(self, bot: Bot, plan_depth: int = 0):
        self.bot = bot
        self.plan_depth = plan_depth

    def act(self, history: tuple) -> tuple[int, np.ndarray | None]:
        true = self.bot.dist(history)
        if self.plan_depth > 1:
            depth = self.lookahead(history, self.plan_depth)
            return plan_response(self.bot.dist, history, self.game.payoff_agent, depth, self.rng), true
        return best_response(true, self.game.payoff_agent, self.rng), true


class FrequencyBRAgent(Agent):
    """Non-LLM baseline: best response to the Laplace-smoothed empirical frequency of opponent actions."""

    name = "freq_br"

    def __init__(self, alpha: float = 1.0, window: int | None = None):
        self.alpha = alpha
        self.window = window

    def act(self, history: tuple) -> tuple[int, np.ndarray | None]:
        counts = np.full(self.game.n_opp, self.alpha)
        for _, o in history[-self.window :] if self.window else history:
            counts[o] += 1
        pred = counts / counts.sum()
        return best_response(pred, self.game.payoff_agent, self.rng), pred


Proposer = Callable[..., list[ProgramSpec]]


def last_round(history: tuple) -> tuple | None:
    return history[-1] if history else None


def default_refit_due(t: int) -> bool:
    """Refit at t = 8, 16, 32, 64 and every 64 rounds after that."""
    return t >= 8 and ((t & (t - 1)) == 0 or t % 64 == 0)


class PosteriorBRAgent(Agent):
    """The method (Plan 2): a program-library posterior; best response to the posterior predictive.

    LLM involvement is confined to `proposer`, which is called only when the posterior signals the
    hypothesis space is inadequate (catch-all wins). Without a proposer this is baseline B2.

    Options added for the v2 arms (defaults reproduce v1 exactly):
    - min_obs_propose: the proposer may fire only once the current segment has this many observations;
    - changepoint: CusumDetector kwargs; on an alarm the posterior restarts `back` rounds before it;
    - plan_depth: > 1 plans over that many rounds against the posterior predictive (weights frozen).
    v3 option:
    - propose_at: rounds t at which the proposer is also called regardless of the trigger and cooldown.
    v6 option:
    - explore: ExplorePolicy kwargs (rate, window, end_t, ...); in its rounds the agent plays the action that best
      separates its opponent models instead of the best response (`explore` events). Off by default.
    Every round appends to `belief_log` (read-only view of the posterior; it never changes behaviour).
    Proposed programs' `add` events carry `ll_gain_vs_other`: (fitted log-lik − catch-all prequential log-lik) per
    observation of the current segment; near 0 means the program re-expresses the catch-all (record only).
    """

    name = "bsi"

    def __init__(
        self,
        pool: SandboxPool,
        library: Sequence[ProgramSpec],
        proposer: Proposer | None = None,
        k_max: int = 24,
        mdl_lambda: float = 0.02,
        other_prior: float = 0.05,
        window: int = 20,
        other_mass_threshold: float = 0.3,
        other_margin: float = 0.1,
        propose_cooldown: int = 20,
        max_grid: int = 96,
        max_call_ms: float = 25.0,
        refit_due: Callable[[int], bool] = default_refit_due,
        name: str | None = None,
        min_obs_propose: int = 0,
        changepoint: dict | None = None,
        plan_depth: int = 0,
        propose_at: Sequence[int] = (),
        explore: dict | None = None,
    ):
        self.pool = pool
        self.library = list(library)
        self.proposer = proposer
        self.k_max = k_max
        self.mdl_lambda = mdl_lambda
        self.other_prior = other_prior
        self.window = window
        self.other_mass_threshold = other_mass_threshold
        self.other_margin = other_margin
        self.propose_cooldown = propose_cooldown
        self.max_grid = max_grid
        self.max_call_ms = max_call_ms
        self.refit_due = refit_due
        self.min_obs_propose = min_obs_propose
        self.changepoint = dict(changepoint) if changepoint else None
        self.plan_depth = plan_depth
        self.propose_at = frozenset(int(t) for t in propose_at)
        self.explore = ExplorePolicy.from_opts(explore)
        if name:
            self.name = name
        self.posterior: Posterior | None = None
        self.events: list[dict] = []

    # -- library management

    def _unload_all(self) -> None:
        if self.posterior is not None:
            for hid in self.posterior.ids(include_other=False):
                self.pool.unload(hid)

    def reset(self, game: Game, rng: np.random.Generator, episode_key: str = "") -> None:
        super().reset(game, rng, episode_key)
        self._unload_all()
        self.llm_meter = {}
        self.posterior = Posterior(
            other=BackoffDirichletOther(game.n_opp, key_fn=last_round),
            mdl_lambda=self.mdl_lambda,
            other_prior=self.other_prior,
            window=self.window,
        )
        self._uid = 0
        self._last_propose = -(10**9)
        self._source: dict[str, str] = {}
        self.belief_log: list[dict] = []
        self.detector = CusumDetector(**self.changepoint) if self.changepoint else None
        for spec in self.library:
            self.try_add(spec, history=(), source="library")

    def try_add(self, spec: ProgramSpec, history: tuple, source: str) -> tuple[bool, str]:
        from .hypotheses import ProgramHypothesis  # local import keeps module import order simple

        self._uid += 1
        pid = f"{spec.id}#{self._uid}"
        errs = self.pool.load(pid, spec.src)
        reason = "; ".join(errs)
        if not errs:
            h = ProgramHypothesis(pid, spec.src, self.pool, self.game.info(), spec.param_grid, spec.statement, self.max_grid)
            try:
                h.predict(())
                t0 = time.perf_counter()
                h.predict(history)
                ms = (time.perf_counter() - t0) * 1000
                if ms > self.max_call_ms:  # every refit runs ~|grid| x T calls, so slow programs are unaffordable
                    raise TimeoutError(f"too slow: {ms:.1f} ms per call on {len(history)} rounds (limit {self.max_call_ms} ms)")
                self.posterior.add(h)  # fits parameters on all observations so far
            except Exception as e:  # noqa: BLE001
                reason = f"{type(e).__name__}: {e}"
                self.pool.unload(pid)
                if pid in self.posterior:
                    self.posterior.remove(pid)
        ok = not reason
        if ok:
            self._source[pid] = source
        ev = {"t": len(history), "event": "add", "source": source, "id": pid, "ok": ok, "reason": reason, "statement": spec.statement,
              "src_sha": hashlib.sha1(spec.src.encode()).hexdigest()[:12]}
        if source != "library":  # library programs are in libraries/<game>.json; proposed ones exist only here
            ev.update(src=spec.src[:4096], param_grid=spec.param_grid)
            n = len(self.posterior.segment())
            if ok and n and self.posterior.other_id:
                gain = (self.posterior.ll_sum(pid) - self.posterior.ll_sum(self.posterior.other_id)) / n
                ev["ll_gain_vs_other"] = round(gain, 5)
        self.events.append(ev)
        return ok, reason

    def _drop(self, ids: dict[str, str] | Sequence[str], why: str) -> None:
        for hid in ids:
            if hid in self.posterior:
                self.posterior.remove(hid)
            self.pool.unload(hid)
            self.events.append({"event": "drop", "id": hid, "why": why})

    # -- acting

    def act(self, history: tuple) -> tuple[int, np.ndarray | None]:
        from .hypotheses import predict_batch

        broken = predict_batch(self.posterior.hypotheses(include_other=False), history, self.pool)
        if broken:
            self._drop(broken, "prediction failed")
        pred = self.posterior.predictive(history)
        if self.plan_depth > 1:
            depth = self.lookahead(history, self.plan_depth)
            ids, w = self.posterior.weights_array()  # frozen while planning
            memo = {tuple(history): pred}  # one prediction per distinct context of the expectimax tree

            def plan(ctx: tuple) -> np.ndarray:
                if ctx not in memo:
                    memo[ctx] = self._plan_predict(ctx, ids, w)
                return memo[ctx]

            return self._maybe_explore(history, pred, plan_response(plan, history, self.game.payoff_agent, depth, self.rng)), pred
        return self._maybe_explore(history, pred, best_response(pred, self.game.payoff_agent, self.rng)), pred

    def _maybe_explore(self, history: tuple, pred: np.ndarray, action: int) -> int:
        """v6 exploration: in the policy's rounds, replace the best response by the most informative action. The
        best response is computed first either way, so the agent's rng stream is the same with or without it."""
        t = len(history)
        if self.explore is None or not self.explore.explores(self.episode_key, t):
            return action
        a, eig = self.explore.choose(self.episode_key, t, history, pred, self.game.n_agent, self._explore_models)
        self.events.append({"t": t, "event": "explore", "a": a, "br": int(action), "eig": round(eig, 5)})
        return a

    def _explore_models(self, contexts: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
        """Opponent models whose disagreement exploration resolves: the posterior's programs and its catch-all."""
        ids, w = self.posterior.weights_array()
        wd = dict(zip(ids, w))
        other = self.posterior.other_id
        hyps = self.posterior.hypotheses(include_other=False)
        return model_predictions(hyps, [wd[h.id] for h in hyps], contexts, self.pool,
                                 other=self.posterior.get(other) if other else None, other_weight=wd.get(other, 0.0))

    def _plan_predict(self, context: tuple, ids: list[str], w: np.ndarray) -> np.ndarray:
        """Posterior predictive on a hypothetical history (weights frozen). Programs are called in one sandbox
        batch without touching their prediction caches (which hold the real context for the coming update).
        Hypotheses that fail on it are left out of this prediction only; the real update drops them if they
        also fail on real data."""
        from .hypotheses import normalize_prediction

        hyps = self.posterior.hypotheses(include_other=False)
        res = self.pool.call_batch([(h.id, "model", (context, h.params, h.game_info)) for h in hyps])
        preds: dict[str, np.ndarray] = {}
        for h, r in zip(hyps, res):
            try:
                if not isinstance(r, Exception):
                    preds[h.id] = normalize_prediction(r, h.n)
            except ValueError:
                pass
        other = self.posterior.other_id
        if other is not None:
            preds[other] = self.posterior.get(other).predict(context)
        keep = [(i, wi) for i, wi in zip(ids, w) if i in preds]
        tot = sum(wi for _, wi in keep)
        return sum(wi / tot * preds[i] for i, wi in keep)

    def _log_belief(self) -> None:
        w = self.posterior.weights()
        other = self.posterior.other_id
        named = {i: v for i, v in w.items() if i != other}
        map_id = max(named, key=named.__getitem__) if named else None
        self.belief_log.append({
            "other": round(w.get(other, 0.0), 5),
            "proposer": round(sum(v for i, v in named.items() if self._source.get(i) != "library"), 5),
            "map": map_id,
            "map_w": round(named[map_id], 5) if map_id else None,
            "map_src": self._source.get(map_id) if map_id else None,
            "n": len(named),
        })

    def observe(self, history: tuple, rnd: tuple[int, int]) -> None:
        post = self.posterior
        before = set(post.failed)
        mix = post.update(Observation(history, rnd[1]))
        t = len(history) + 1
        restarted = False
        if self.detector is not None and self.detector.update(mix):
            t0 = max(post.seg_start, t - self.detector.back)
            post.restart(t0)  # refits every hypothesis on the new segment
            restarted = True
            self.events.append({"t": t, "event": "changepoint", "seg_start": t0})
        if self.refit_due(t) and not restarted:  # a scheduled refit right after restart() would repeat it
            post.refit()
        newly_failed = set(post.failed) - before
        if newly_failed:
            self._drop(sorted(newly_failed), "failed during update/refit")
        forced = t in self.propose_at
        if self.proposer is not None and (forced or (
            len(post.observations) - post.seg_start >= self.min_obs_propose
            and t - self._last_propose >= self.propose_cooldown
            and post.other_triggered(self.other_mass_threshold, self.other_margin)
        )):
            self._last_propose = t
            full = history + (tuple(rnd),)
            self.events.append({"t": t, "event": "propose", "other_mass": post.other_mass(),
                                **({"forced": True} if forced else {})})
            plog = getattr(self.proposer, "log", None)
            n_log = len(plog) if plog is not None else 0
            try:
                specs = self.proposer(posterior=post, history=full, game=self.game,
                                      tag=f"{self.episode_key}:t{t}", meter=self.llm_meter)
            except (LLMUnavailable, BudgetExceeded):
                raise  # fatal for the whole run: stop and resume later rather than record a degraded episode
            except Exception as e:  # noqa: BLE001 - the episode must survive proposer/API failures
                self.events.append({"t": t, "event": "propose_error", "error": f"{type(e).__name__}: {e}"})
                specs = []
            reply = {k: plog[-1][k] for k in ("attempts", "finish_reason", "completion_tokens", "reasoning_chars", "degraded")
                     if k in plog[-1]} if plog is not None and len(plog) > n_log else {}
            self.events.append({"t": t, "event": "propose_result", "n_specs": len(specs), **reply})
            for spec in specs:
                self.try_add(spec, full, source="proposer")
            removed = post.prune(self.k_max)
            if removed:
                self._drop(removed, "pruned")
        self._log_belief()

    def close(self) -> None:
        self._unload_all()
