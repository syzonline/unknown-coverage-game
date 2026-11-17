"""v6 program archive: every program the LLM wrote (or mutated) that loads and runs, scored by exact likelihood.

The archive is the heredity of the v6 loop. Programs stay in it after they leave the active set, so a good variant is
never lost to one bad decision; the LLM sees the best ones with their code and lineage and is asked to mutate them.
Each entry is a ProgramHypothesis loaded in the sandbox under its own id, refit on the whole history at every
decision. Score = fitted log-likelihood − BIC penalty − MDL penalty (the posterior's own prior terms), so it ranks
programs the way the posterior would.
"""
from __future__ import annotations

import difflib
import time
from dataclasses import dataclass, field

import numpy as np

from bsi.hypotheses import Observation
from bsi.sandbox import SandboxPool

from .agents import ProgramSpec
from .hypotheses import ProgramHypothesis


@dataclass
class Entry:
    name: str
    spec: ProgramSpec
    h: ProgramHypothesis
    op: str                    # "create" | "mutate" | "revise" (code written by the LLM for an active model)
    t: int                     # round at which it was added
    parent: str | None = None
    generation: int = 0
    origin_call: str = ""      # "decision" | "variation:<k>"
    similarity: float | None = None  # difflib ratio of the code to the parent's
    ll: float = float("nan")   # fitted log-likelihood over the history at the last rescore
    n_obs: int = 0
    score: float = float("-inf")
    history: list[tuple[int, float]] = field(default_factory=list)  # (t, ll per round) at every rescore

    def mean_ll(self) -> float:
        return self.ll / self.n_obs if self.n_obs else float("nan")


class ProgramArchive:
    def __init__(self, pool: SandboxPool, game_info: dict, capacity: int = 8, max_grid: int = 96,
                 max_call_ms: float = 25.0, mdl_lambda: float = 0.02):
        self.pool = pool
        self.game_info = game_info
        self.capacity = capacity
        self.max_grid = max_grid
        self.max_call_ms = max_call_ms
        self.mdl_lambda = mdl_lambda
        self.entries: dict[str, Entry] = {}
        self.failed: list[dict] = []   # {"t", "name", "op", "statement", "reason"}
        self._uid = 0

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def close(self) -> None:
        for e in self.entries.values():
            self.pool.unload(e.h.id)
        self.entries.clear()

    def _score(self, e: Entry) -> float:
        n = max(e.n_obs, 1)
        return e.ll - 0.5 * e.h.n_free_params * np.log(n) - self.mdl_lambda * e.h.complexity

    def add(self, spec: ProgramSpec, history: tuple, op: str, parent: str | None = None, origin_call: str = "",
            protect: set[str] = frozenset()) -> tuple[bool, str, Entry | None]:
        """Load, smoke-test and time the program (the same checks as PosteriorBRAgent.try_add), fit it on the whole
        history and insert it; then evict the lowest-scoring entries beyond capacity (never one in `protect`).
        Returns (ok, reason, entry)."""
        t = len(history)
        self._uid += 1
        pid = f"arc_{spec.id}#{self._uid}"
        reason = "; ".join(self.pool.load(pid, spec.src))
        h = None
        if not reason:
            h = ProgramHypothesis(pid, spec.src, self.pool, self.game_info, spec.param_grid, spec.statement, self.max_grid)
            try:
                h.predict(())
                t0 = time.perf_counter()
                h.predict(history)
                ms = (time.perf_counter() - t0) * 1000
                if ms > self.max_call_ms:
                    raise TimeoutError(f"too slow: {ms:.1f} ms per call on {t} rounds (limit {self.max_call_ms} ms)")
                ll = h.refit(_observations(history)) if history else np.zeros(0)
            except Exception as ex:  # noqa: BLE001
                reason = f"{type(ex).__name__}: {ex}"
        if reason:
            self.pool.unload(pid)
            self.failed.append({"t": t, "name": spec.id, "op": op, "statement": spec.statement, "reason": reason[:300]})
            return False, reason, None
        gen, sim = 0, None
        if parent is not None and parent in self.entries:
            p = self.entries[parent]
            gen = p.generation + 1
            sim = round(difflib.SequenceMatcher(None, p.spec.src, spec.src).ratio(), 3)
        e = Entry(spec.id, spec, h, op, t, parent, gen, origin_call, sim)
        self.entries[spec.id] = e
        self._set_fit(e, ll)
        self._evict(protect | {spec.id})
        return True, "", e

    def _set_fit(self, e: Entry, ll: np.ndarray) -> None:
        e.ll = float(np.sum(ll))
        e.n_obs = len(ll)
        e.score = self._score(e)
        e.history.append((e.n_obs, round(e.mean_ll(), 5) if e.n_obs else float("nan")))

    def rescore(self, history: tuple) -> list[str]:
        """Refit every entry on the whole history; entries that now fail are removed. Returns the removed names."""
        gone = []
        for name, e in list(self.entries.items()):
            try:
                self._set_fit(e, e.h.refit(_observations(history)) if history else np.zeros(0))
            except Exception as ex:  # noqa: BLE001
                self.failed.append({"t": len(history), "name": name, "op": "rescore", "statement": e.spec.statement,
                                    "reason": f"{type(ex).__name__}: {ex}"[:300]})
                self.pool.unload(e.h.id)
                del self.entries[name]
                gone.append(name)
        return gone

    def _evict(self, protect: set[str]) -> None:
        while len(self.entries) > self.capacity:
            cand = [e for e in self.entries.values() if e.name not in protect]
            if not cand:
                return
            worst = min(cand, key=lambda e: e.score)
            self.pool.unload(worst.h.id)
            del self.entries[worst.name]

    def ranked(self) -> list[Entry]:
        return sorted(self.entries.values(), key=lambda e: -e.score)

    def best(self) -> Entry | None:
        r = self.ranked()
        return r[0] if r else None

    def lineage(self, name: str) -> list[str]:
        """Ancestor chain, newest first (ancestors evicted from the archive end the chain)."""
        out, cur = [], self.entries.get(name)
        while cur is not None and cur.parent is not None:
            out.append(cur.parent)
            cur = self.entries.get(cur.parent)
        return out

    def snapshot(self) -> list[dict]:
        return [{"name": e.name, "op": e.op, "t": e.t, "parent": e.parent, "gen": e.generation, "call": e.origin_call,
                 "mean_ll": round(e.mean_ll(), 5), "score": round(e.score, 3), "params": e.h.params,
                 "sim": e.similarity} for e in self.ranked()]


def _observations(history: tuple) -> list[Observation]:
    return [Observation(tuple(history[:t]), int(history[t][1])) for t in range(len(history))]
