"""Opponent-model hypotheses backed by sandboxed programs."""
from __future__ import annotations

import itertools
import zlib
from typing import Sequence

import numpy as np

from bsi.hypotheses import Hypothesis, Observation, register
from bsi.sandbox import SandboxError, SandboxPool, ast_complexity

PRED_EPS = 1e-3  # mix every prediction with uniform so one surprising action is never fatal


def normalize_prediction(raw: np.ndarray, n: int) -> np.ndarray:
    p = np.asarray(raw, dtype=float).reshape(-1)
    if p.shape != (n,) or not np.all(np.isfinite(p)) or (p < 0).any() or p.sum() <= 0:
        raise ValueError(f"invalid prediction {np.round(p, 4).tolist()} (need {n} non-negative finite values)")
    p = p / p.sum()
    return (1 - PRED_EPS) * p + PRED_EPS / n


MAX_VALUES_PER_PARAM = 50


def param_grid(grid: dict[str, Sequence], max_points: int, seed: int = 0) -> list[dict]:
    """Grid points to fit. Grids come from LLM output and can be combinatorially huge (one proposal had
    millions of combinations and OOM-killed the run), so the product is never materialized: when it exceeds
    `max_points`, a deterministic sample of distinct combinations is decoded from mixed-radix indices."""
    keys = sorted(grid)
    values = [list(grid[k])[:MAX_VALUES_PER_PARAM] for k in keys]
    sizes = [len(v) for v in values]
    total = 1
    for n in sizes:
        total *= n
    if total == 0:
        return [{}]
    if total <= max_points:
        return [dict(zip(keys, vals)) for vals in itertools.product(*values)]
    rng = np.random.default_rng(seed)
    chosen: set[tuple[int, ...]] = set()
    while len(chosen) < max_points:
        chosen.add(tuple(int(rng.integers(n)) for n in sizes))
    return [dict(zip(keys, (values[d][i] for d, i in enumerate(idx)))) for idx in sorted(chosen)]


@register
class ProgramHypothesis(Hypothesis):
    """Context = history tuple; outcome = opponent action. Parameters are fit by grid MLE in the sandbox."""

    kind = "opponent_program"

    def __init__(
        self,
        id: str,
        src: str,
        pool: SandboxPool,
        game_info: dict,
        grid: dict[str, Sequence] | None = None,
        statement: str = "",
        max_grid: int = 96,
    ):
        grid = {k: list(v) for k, v in (grid or {}).items() if len(v) > 0}
        points = param_grid(grid, max_grid, seed=zlib.crc32(id.encode()))  # stable across processes
        super().__init__(
            id,
            statement=statement,
            params=points[0],
            complexity=ast_complexity(src),
            n_free_params=sum(1 for v in grid.values() if len(v) > 1),
        )
        self.src = src
        self.pool = pool
        self.game_info = game_info
        self.n = game_info["n_opp"]
        self.grid = grid
        self.points = points
        self._cache: dict[int, np.ndarray] = {}

    # predictions are cached per (history, params) because act() and update() score the same context
    def _key(self, context: tuple) -> int:
        return hash((context, tuple(sorted(self.params.items()))))

    def set_cached(self, context: tuple, pred: np.ndarray) -> None:
        if len(self._cache) > 8:
            self._cache.clear()
        self._cache[self._key(context)] = pred

    def predict(self, context: tuple) -> np.ndarray:
        k = self._key(context)
        if k not in self._cache:
            raw = self.pool.call(self.id, "model", (context, self.params, self.game_info))
            self.set_cached(context, normalize_prediction(raw, self.n))
        return self._cache[k]

    def refit(self, observations: Sequence[Observation]) -> np.ndarray:
        """Grid MLE over all observations; returns per-observation log-likelihoods at the best point."""
        if not observations:
            return np.zeros(0)
        segments = _episodes(observations)
        totals = np.zeros(len(self.points))
        per_obs: list[np.ndarray] = []
        for seg in segments:
            full = seg[-1].context
            start = len(seg[0].context)
            raw = self.pool.prefixes(self.id, "model", full, self.points, extra=(self.game_info,), start=start,
                                     out_len=self.n)
            if raw.shape[:2] != (len(self.points), len(seg)) or raw.shape[-1] != self.n:
                raise SandboxError(f"bad prefix output shape {raw.shape}")
            p = np.clip(raw, 0, None)
            s = p.sum(axis=-1, keepdims=True)
            if not np.all(np.isfinite(p)) or (s <= 0).any():
                raise SandboxError("invalid predictions during refit")
            p = (1 - PRED_EPS) * p / s + PRED_EPS / self.n
            outcomes = np.array([o.outcome for o in seg])
            ll = np.log(p[:, np.arange(len(seg)), outcomes])  # (P, T)
            totals += ll.sum(axis=1)
            per_obs.append(ll)
        best = int(np.argmax(totals))
        self.params = dict(self.points[best])
        self._cache.clear()
        return np.concatenate([ll[best] for ll in per_obs])

    def describe(self) -> dict:
        return {**super().describe(), "grid": self.grid}


def _episodes(observations: Sequence[Observation]) -> list[list[Observation]]:
    """Split observations into runs where each context extends the previous one by exactly one round."""
    segs: list[list[Observation]] = []
    for o in observations:
        if segs and len(o.context) == len(segs[-1][-1].context) + 1:
            segs[-1].append(o)
        else:
            segs.append([o])
    return segs


def predict_batch(hyps: Sequence[ProgramHypothesis], context: tuple, pool: SandboxPool) -> dict[str, str]:
    """Evaluate all program hypotheses on one context in parallel, fill caches; return {id: error} for broken ones."""
    items = [(h.id, "model", (context, h.params, h.game_info)) for h in hyps]
    broken: dict[str, str] = {}
    for h, res in zip(hyps, pool.call_batch(items)):
        if isinstance(res, SandboxError):
            broken[h.id] = str(res)
            continue
        try:
            h.set_cached(context, normalize_prediction(res, h.n))
        except ValueError as e:
            broken[h.id] = str(e)
    return broken
