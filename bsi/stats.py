"""Statistics for honest comparisons: calibration, Bayesian rate tests, paired bootstrap, sequential gates."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-12


def kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) for discrete distributions."""
    p = np.asarray(p, dtype=float)
    q = np.maximum(np.asarray(q, dtype=float), EPS)
    m = p > 0
    return float((p[m] * (np.log(p[m]) - np.log(q[m]))).sum())


def log_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    return float(-np.log(np.maximum(probs[np.arange(len(labels)), labels], EPS)).mean())


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    onehot = np.eye(probs.shape[1])[np.asarray(labels, dtype=int)]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def top_label(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(confidence, correct) of the top label; lets rounds from games with different action counts be pooled."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    return probs.max(axis=1), (probs.argmax(axis=1) == labels).astype(float)


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Top-label reliability: per confidence bin, mean confidence vs accuracy."""
    return reliability_bins_conf(*top_label(probs, labels), n_bins=n_bins)


def reliability_bins_conf(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> list[dict]:
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append({"lo": edges[b], "hi": edges[b + 1], "n": int(m.sum()), "conf": float(conf[m].mean()), "acc": float(correct[m].mean())})
    return rows


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    return ece_conf(*top_label(probs, labels), n_bins=n_bins)


def ece_conf(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    rows = reliability_bins_conf(conf, correct, n_bins)
    n = sum(r["n"] for r in rows)
    return float(sum(r["n"] / n * abs(r["conf"] - r["acc"]) for r in rows)) if n else 0.0


def interval_coverage(lo: np.ndarray, hi: np.ndarray, actual: np.ndarray) -> float:
    lo, hi, actual = (np.asarray(x, dtype=float) for x in (lo, hi, actual))
    return float(((actual >= lo) & (actual <= hi)).mean())


def beta_prob_greater(
    s_a: int, n_a: int, s_b: int, n_b: int, prior: tuple[float, float] = (1.0, 1.0),
    n_samples: int = 20000, rng: np.random.Generator | None = None,
) -> float:
    """P(p_a > p_b) under independent Beta posteriors."""
    rng = rng or np.random.default_rng(0)
    a = rng.beta(prior[0] + s_a, prior[1] + n_a - s_a, n_samples)
    b = rng.beta(prior[0] + s_b, prior[1] + n_b - s_b, n_samples)
    return float((a > b).mean())


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10000, ci: float = 0.95, rng: np.random.Generator | None = None
) -> dict:
    """Paired comparison on matched units (same task × seed). Returns mean diff (a−b), CI, two-sided p."""
    rng = rng or np.random.default_rng(0)
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    boots = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    centered = boots - boots.mean()
    p = float((np.abs(centered) >= abs(d.mean())).mean())
    return {"mean_diff": float(d.mean()), "ci": (float(lo), float(hi)), "p_two_sided": max(p, 1.0 / n_boot), "n": n}


def cluster_bootstrap(
    a: np.ndarray, b: np.ndarray, groups: list, n_boot: int = 10000, ci: float = 0.95,
    rng: np.random.Generator | None = None, ratio_den: tuple[np.ndarray, np.ndarray] | None = None,
    min_clusters: int = 5,
) -> dict:
    """Paired comparison resampling whole clusters (e.g. episodes that share a bot) instead of units, for units
    that are not independent. Statistic: mean of a−b over the resampled units. With `ratio_den=(c, d)` the
    statistic is instead sum(a−b) / sum(c−d) (e.g. a gain as a share of a ceiling gain); its p is not reported, and
    resamples whose denominator is not positive are left out of the CI. With fewer than `min_clusters` clusters the
    resampling distribution is degenerate (one cluster: every draw equals the estimate), so CI and p are nan."""
    rng = rng or np.random.default_rng(0)
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    keys = list(dict.fromkeys(groups))
    idx = {k: i for i, k in enumerate(keys)}
    g = np.array([idx[k] for k in groups])
    k = len(keys)
    num = np.bincount(g, weights=d, minlength=k)
    cnt = np.bincount(g, minlength=k).astype(float)
    den = cnt if ratio_den is None else np.bincount(
        g, weights=np.asarray(ratio_den[0], dtype=float) - np.asarray(ratio_den[1], dtype=float), minlength=k)
    est = float(num.sum() / den.sum())
    out = {"mean_diff": est, "ci": (float("nan"), float("nan")), "n": len(d), "clusters": k}
    if ratio_den is None:
        out["p_two_sided"] = float("nan")
    if k < min_clusters:
        return out
    draw = rng.integers(0, k, size=(n_boot, k))
    bden = den[draw].sum(axis=1)
    boots = num[draw].sum(axis=1)[bden > 0] / bden[bden > 0]
    if len(boots) < n_boot / 2:  # the ratio is not estimable when its denominator is often non-positive
        return out
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    out["ci"] = (float(lo), float(hi))
    if ratio_den is None:
        centered = boots - boots.mean()
        out["p_two_sided"] = max(float((np.abs(centered) >= abs(est)).mean()), 1.0 / n_boot)
    return out


@dataclass
class SequentialBetaGate:
    """Bayesian sequential test of p_new > p_old; safe to check after every batch.

    accept when P(p_new > p_old) ≥ upper, reject when ≤ lower, otherwise continue until max_n per arm.
    """

    upper: float = 0.95
    lower: float = 0.05
    max_n: int = 400
    s_new: int = 0
    n_new: int = 0
    s_old: int = 0
    n_old: int = 0
    history: list[float] = field(default_factory=list)

    def add(self, new_results: np.ndarray, old_results: np.ndarray) -> str:
        new_results = np.asarray(new_results, dtype=int)
        old_results = np.asarray(old_results, dtype=int)
        self.s_new += int(new_results.sum())
        self.n_new += len(new_results)
        self.s_old += int(old_results.sum())
        self.n_old += len(old_results)
        p = beta_prob_greater(self.s_new, self.n_new, self.s_old, self.n_old)
        self.history.append(p)
        if p >= self.upper:
            return "accept"
        if p <= self.lower:
            return "reject"
        if min(self.n_new, self.n_old) >= self.max_n:
            return "inconclusive"
        return "continue"
