"""Discrete posterior over hypotheses, with an MDL prior, BIC penalty and an "other" bucket."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from scipy.special import logsumexp

from .hypotheses import Hypothesis, Observation


@dataclass
class _Entry:
    h: Hypothesis
    ll_sum: float = 0.0
    recent: deque = field(default_factory=deque)


class Posterior:
    """p(h | data) ∝ prior(h) · exp(lik_temp · Σ log p(obs | h)) · exp(-BIC penalty).

    Prior: the catch-all ("other") gets fixed mass `other_prior`; named hypotheses share the rest
    in proportion to exp(-mdl_lambda · complexity).
    """

    def __init__(
        self,
        hypotheses: Iterable[Hypothesis] = (),
        other: Hypothesis | None = None,
        mdl_lambda: float = 0.02,
        other_prior: float = 0.05,
        lik_temp: float = 1.0,
        bic: bool = True,
        window: int = 20,
    ):
        self.mdl_lambda = mdl_lambda
        self.other_prior = other_prior if other is not None else 0.0
        self.lik_temp = lik_temp
        self.bic = bic
        self.window = window
        self.observations: list[Observation] = []
        self.seg_start = 0  # scores and fits use observations[seg_start:]; moved forward by restart()
        self.mix_ll: list[float] = []  # prequential log-score of the posterior mixture
        self.failed: dict[str, str] = {}
        self._entries: dict[str, _Entry] = {}
        self.other_id: str | None = None
        if other is not None:
            self.other_id = other.id
            self.add(other)
        for h in hypotheses:
            self.add(h)

    # ------------------------------------------------------------ membership

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, hid: str) -> bool:
        return hid in self._entries

    def ids(self, include_other: bool = True) -> list[str]:
        return [i for i in self._entries if include_other or i != self.other_id]

    def get(self, hid: str) -> Hypothesis:
        return self._entries[hid].h

    def hypotheses(self, include_other: bool = True) -> list[Hypothesis]:
        return [self._entries[i].h for i in self.ids(include_other)]

    def add(self, h: Hypothesis) -> None:
        if h.id in self._entries:
            raise ValueError(f"duplicate hypothesis id {h.id!r}")
        arr = h.refit(self.segment())
        self._entries[h.id] = _Entry(h, float(arr.sum()), deque(arr[-self.window :], maxlen=self.window))

    def segment(self) -> list[Observation]:
        """Observations of the current segment (all of them unless restart() was called)."""
        return self.observations[self.seg_start :]

    def remove(self, hid: str) -> None:
        if hid == self.other_id:
            raise ValueError("cannot remove the catch-all hypothesis")
        self._entries.pop(hid, None)

    def prune(self, k_max: int, keep: Iterable[str] = ()) -> list[str]:
        """Keep the `k_max` most probable named hypotheses (plus `keep`); return removed ids."""
        keep = set(keep)
        w = self.weights()
        named = sorted(self.ids(include_other=False), key=lambda i: w[i], reverse=True)
        removed = [i for i in named[k_max:] if i not in keep]
        for i in removed:
            self.remove(i)
        return removed

    # ------------------------------------------------------------ scores

    def _log_scores(self) -> tuple[list[str], np.ndarray]:
        ids = list(self._entries)
        n = len(self.observations) - self.seg_start
        named = [i for i in ids if i != self.other_id]
        log_prior: dict[str, float] = {}
        if named:
            raw = np.array([-self.mdl_lambda * self._entries[i].h.complexity for i in named])
            raw = raw - logsumexp(raw) + np.log1p(-self.other_prior)
            log_prior.update(zip(named, raw))
        if self.other_id is not None:
            log_prior[self.other_id] = np.log(self.other_prior)
        scores = np.empty(len(ids))
        for k, i in enumerate(ids):
            e = self._entries[i]
            pen = 0.5 * e.h.n_free_params * np.log(n) if (self.bic and n > 0) else 0.0
            scores[k] = log_prior[i] + self.lik_temp * e.ll_sum - pen
        return ids, scores

    def weights_array(self) -> tuple[list[str], np.ndarray]:
        ids, s = self._log_scores()
        if not ids:
            return ids, s
        return ids, np.exp(s - logsumexp(s))

    def weights(self) -> dict[str, float]:
        ids, w = self.weights_array()
        return dict(zip(ids, w.tolist()))

    def entropy(self) -> float:
        _, w = self.weights_array()
        w = w[w > 0]
        return float(-(w * np.log(w)).sum())

    def map_id(self, include_other: bool = False) -> str | None:
        w = self.weights()
        cands = [i for i in w if include_other or i != self.other_id]
        return max(cands, key=w.__getitem__) if cands else None

    def other_mass(self) -> float:
        return self.weights().get(self.other_id, 0.0) if self.other_id else 0.0

    def ll_sum(self, hid: str) -> float:
        """Log-likelihood of the current segment under `hid` (fitted for programs, prequential for stateful ones)."""
        return self._entries[hid].ll_sum

    def recent_mean_ll(self, hid: str) -> float:
        r = self._entries[hid].recent
        return float(np.mean(r)) if r else float("nan")

    # ------------------------------------------------------------ inference

    def _drop_failed(self, failures: dict[str, str]) -> None:
        for hid, err in failures.items():
            self.failed[hid] = err
            if hid == self.other_id:
                raise RuntimeError(f"catch-all hypothesis failed: {err}")
            self._entries.pop(hid, None)

    def update(self, obs: Observation) -> float:
        """Score `obs` under every hypothesis, then let stateful ones absorb it.

        Hypotheses that raise are dropped (recorded in `failed`). Returns the mixture log-likelihood.
        """
        ids, w = self.weights_array()
        lls: dict[str, float] = {}
        failures: dict[str, str] = {}
        for i in ids:
            try:
                ll = self._entries[i].h.log_lik(obs)
                if not np.isfinite(ll):
                    raise ValueError(f"non-finite log-likelihood {ll}")
                lls[i] = ll
            except Exception as e:  # noqa: BLE001 - any failure disqualifies the hypothesis
                failures[i] = f"{type(e).__name__}: {e}"
        self._drop_failed(failures)
        if not lls:
            raise RuntimeError("no hypotheses left")
        keep = [k for k, i in enumerate(ids) if i in lls]
        lw = np.log(np.maximum(w[keep], 1e-300))
        lw -= logsumexp(lw)
        mix = float(logsumexp(lw + np.array([lls[ids[k]] for k in keep])))
        self.mix_ll.append(mix)
        for i, ll in lls.items():
            e = self._entries[i]
            e.ll_sum += ll
            e.recent.append(ll)
            e.h.observe(obs)
        self.observations.append(obs)
        return mix

    def refit(self, ids: Iterable[str] | None = None) -> None:
        """Re-estimate parameters of the given hypotheses (default: all) on the current segment."""
        failures: dict[str, str] = {}
        seg = self.segment()
        for i in list(ids) if ids is not None else list(self._entries):
            e = self._entries.get(i)
            if e is None:
                continue
            try:
                arr = e.h.refit(seg)
            except Exception as ex:  # noqa: BLE001
                failures[i] = f"{type(ex).__name__}: {ex}"
                continue
            e.ll_sum = float(arr.sum())
            e.recent = deque(arr[-self.window :], maxlen=self.window)
        self._drop_failed(failures)

    def restart(self, t0: int) -> None:
        """Change point: forget evidence before observation index `t0`. Every hypothesis is refit on
        observations[t0:] (stateful ones, like the catch-all, are rebuilt by replay), so the posterior equals
        one built from scratch on that segment. Contexts keep the full history."""
        self.seg_start = int(min(max(t0, self.seg_start), len(self.observations)))
        self.refit()

    def predictive(self, context: Any) -> np.ndarray:
        ids, w = self.weights_array()
        return sum(wi * self._entries[i].h.predict(context) for i, wi in zip(ids, w))

    def other_triggered(self, mass_threshold: float = 0.3, margin: float = 0.1) -> bool:
        """True when the hypothesis space looks inadequate and new hypotheses should be proposed:
        the catch-all holds > `mass_threshold` of the posterior, or over the last `window`
        observations it beats the best named hypothesis by > `margin` nats per observation."""
        if self.other_id is None:
            return False
        if self.other_mass() > mass_threshold:
            return True
        if len(self._entries[self.other_id].recent) < self.window:
            return False
        best = self.map_id(include_other=False)
        if best is None:
            return True
        return self.recent_mean_ll(self.other_id) - self.recent_mean_ll(best) > margin

    def summary(self, top: int = 5) -> list[dict]:
        w = self.weights()
        rows = []
        for i in sorted(w, key=w.__getitem__, reverse=True)[:top]:
            e = self._entries[i]
            rows.append(
                {
                    **e.h.describe(),
                    "weight": w[i],
                    "ll_sum": e.ll_sum,
                    "recent_mean_ll": self.recent_mean_ll(i),
                }
            )
        return rows
