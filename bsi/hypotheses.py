"""Hypotheses over a finite outcome space.

A hypothesis maps a context to a predictive distribution over outcomes. Likelihoods are always
computed by code, never by an LLM. Stateful hypotheses (e.g. the Dirichlet catch-all) update
themselves in `observe` and can be replayed from scratch with `refit`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class Observation:
    context: Any
    outcome: int


class Hypothesis:
    kind = "generic"

    def __init__(
        self,
        id: str,
        statement: str = "",
        params: dict | None = None,
        complexity: float = 0.0,
        n_free_params: int = 0,
    ):
        self.id = id
        self.statement = statement
        self.params = dict(params or {})
        self.complexity = float(complexity)
        self.n_free_params = int(n_free_params)

    def predict(self, context: Any) -> np.ndarray:
        raise NotImplementedError

    def log_lik(self, obs: Observation) -> float:
        p = self.predict(obs.context)
        return float(np.log(max(float(p[obs.outcome]), EPS)))

    def observe(self, obs: Observation) -> None:
        """Hook for stateful hypotheses; called after the posterior has scored `obs`."""

    def reset_state(self) -> None:
        pass

    def refit(self, observations: Sequence[Observation]) -> np.ndarray:
        """Re-estimate free parameters on all observations; return per-observation log-likelihoods.

        The default has no free parameters: it replays the observations prequentially.
        """
        self.reset_state()
        out = np.empty(len(observations))
        for i, o in enumerate(observations):
            out[i] = self.log_lik(o)
            self.observe(o)
        return out

    def describe(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "statement": self.statement,
            "params": self.params,
            "complexity": self.complexity,
            "n_free_params": self.n_free_params,
        }


class FixedDist(Hypothesis):
    """Context-independent outcome distribution (mainly for tests and simple Bernoulli models)."""

    kind = "fixed"

    def __init__(self, id: str, probs: Sequence[float], **kw: Any):
        super().__init__(id, **kw)
        p = np.asarray(probs, dtype=float)
        self.probs = p / p.sum()

    def predict(self, context: Any) -> np.ndarray:
        return self.probs


class FunctionHypothesis(Hypothesis):
    """Wraps a trusted Python callable `fn(context, params) -> probs`."""

    kind = "function"

    def __init__(self, id: str, fn: Callable[[Any, dict], Sequence[float]], **kw: Any):
        super().__init__(id, **kw)
        self.fn = fn

    def predict(self, context: Any) -> np.ndarray:
        p = np.asarray(self.fn(context, self.params), dtype=float)
        return p / p.sum()


class DirichletOther(Hypothesis):
    """Catch-all: Dirichlet-multinomial predictive over outcomes, ignoring context.

    It is a proper sequential predictor, so when it out-predicts every named hypothesis the
    hypothesis space is probably missing something.
    """

    kind = "other"

    def __init__(self, n_outcomes: int, alpha: float = 1.0, id: str = "other"):
        super().__init__(id, statement="none of the named hypotheses (Dirichlet-multinomial baseline)")
        self.n = n_outcomes
        self.alpha = alpha
        self.counts = np.zeros(n_outcomes)

    def predict(self, context: Any) -> np.ndarray:
        c = self.counts + self.alpha
        return c / c.sum()

    def observe(self, obs: Observation) -> None:
        self.counts[obs.outcome] += 1

    def reset_state(self) -> None:
        self.counts = np.zeros(self.n)


class BackoffDirichletOther(Hypothesis):
    """Catch-all conditioned on a discrete key of the context, backing off to the marginal:

        p(y | ctx) = (c[key, y] + beta · m(y)) / (c[key] + beta),   m = Dirichlet(alpha) marginal predictive

    Picks up first-order structure (reactive, cycles, win-stay/lose-shift, …) that a marginal
    catch-all cannot, so "the catch-all wins" still means the named hypotheses miss something.
    """

    kind = "other"

    def __init__(self, n_outcomes: int, key_fn: Callable[[Any], Any], alpha: float = 1.0, beta: float = 2.0, id: str = "other"):
        super().__init__(id, statement="none of the named hypotheses (context-conditioned Dirichlet with backoff)")
        self.n = n_outcomes
        self.key_fn = key_fn
        self.alpha = alpha
        self.beta = beta
        self.reset_state()

    def reset_state(self) -> None:
        self.marginal = np.zeros(self.n)
        self.cond: dict[Any, np.ndarray] = {}

    def predict(self, context: Any) -> np.ndarray:
        m = (self.marginal + self.alpha) / (self.marginal.sum() + self.alpha * self.n)
        c = self.cond.get(self.key_fn(context))
        if c is None:
            return m
        return (c + self.beta * m) / (c.sum() + self.beta)

    def observe(self, obs: Observation) -> None:
        self.marginal[obs.outcome] += 1
        key = self.key_fn(obs.context)
        if key not in self.cond:
            self.cond[key] = np.zeros(self.n)
        self.cond[key][obs.outcome] += 1


HYPOTHESIS_TYPES: dict[str, type[Hypothesis]] = {}


def register(cls: type[Hypothesis]) -> type[Hypothesis]:
    HYPOTHESIS_TYPES[cls.kind] = cls
    return cls


for _cls in (FixedDist, FunctionHypothesis, DirichletOther, BackoffDirichletOther):
    register(_cls)
