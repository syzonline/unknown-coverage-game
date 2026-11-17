import numpy as np
import pytest

from bsi import decide, eig, stats


def test_mutual_information_bounds():
    w = np.array([0.5, 0.5])
    assert eig.mutual_information(w, np.array([[0.3, 0.7], [0.3, 0.7]])) == pytest.approx(0.0, abs=1e-12)
    assert eig.mutual_information(w, np.array([[1.0, 0.0], [0.0, 1.0]])) == pytest.approx(np.log(2))


def test_select_prefers_informative_per_cost():
    w = np.array([0.5, 0.5])
    useless = eig.Candidate("useless", np.array([[0.5, 0.5], [0.5, 0.5]]))
    good = eig.Candidate("good", np.array([[0.9, 0.1], [0.1, 0.9]]), cost=1.0)
    perfect_but_expensive = eig.Candidate("expensive", np.array([[1.0, 0.0], [0.0, 1.0]]), cost=10.0)
    assert eig.select_intervention(w, [useless, good, perfect_but_expensive]).name == "good"
    assert eig.select_intervention(w, [useless]) is None


def test_binomial_outcomes_and_more_rollouts_more_info():
    w = np.array([0.5, 0.5])
    p = np.array([0.4, 0.6])
    pmf = eig.binomial_outcomes(p, 10)
    assert pmf.shape == (2, 11) and pmf.sum(axis=1) == pytest.approx([1, 1])
    assert eig.mutual_information(w, eig.binomial_outcomes(p, 40)) > eig.mutual_information(w, pmf)


def test_eig_mc_matches_exact():
    w = np.array([0.3, 0.7])
    P = np.array([[0.8, 0.2], [0.25, 0.75]])
    exact = eig.mutual_information(w, P)
    mc = eig.eig_mc(w, lambda h, r: int(r.random() < P[h, 1]), lambda h, y: np.log(P[h, y]), n_samples=20000,
                    rng=np.random.default_rng(0))
    assert mc == pytest.approx(exact, abs=0.01)


def test_best_response_and_regret():
    rps = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
    assert decide.best_response(np.array([1.0, 0, 0]), rps) == 1  # rock -> paper
    assert decide.regret(np.array([1.0, 0, 0]), rps, 1) == 0.0
    assert decide.regret(np.array([1.0, 0, 0]), rps, 2) == 2.0


def test_gate_commit():
    rng = np.random.default_rng(0)
    assert decide.gate_commit(rng.normal(0.1, 0.02, 1000), min_effect=0.05).commit
    assert not decide.gate_commit(rng.normal(0.0, 0.1, 1000)).commit


def test_calibration_metrics():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, 20000)
    labels = (rng.random(20000) > conf).astype(int)  # class 0 correct with prob conf
    probs = np.stack([conf, 1 - conf], axis=1)
    assert stats.ece(probs, labels) < 0.02
    overconf = np.stack([np.full(20000, 0.99), np.full(20000, 0.01)], axis=1)
    assert stats.ece(overconf, labels) > 0.15


def test_kl_and_logloss():
    assert stats.kl([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    assert stats.kl([1.0, 0.0], [0.5, 0.5]) == pytest.approx(np.log(2))
    assert stats.log_loss(np.array([[0.5, 0.5]]), np.array([0])) == pytest.approx(np.log(2))


def test_bayesian_and_bootstrap_tests():
    assert stats.beta_prob_greater(80, 100, 50, 100) > 0.99
    rng = np.random.default_rng(1)
    a = rng.normal(0.3, 1, 200)
    r = stats.paired_bootstrap(a + 0.5, a)
    assert r["mean_diff"] == pytest.approx(0.5) and r["p_two_sided"] < 0.01


def test_sequential_gate():
    rng = np.random.default_rng(0)
    gate = stats.SequentialBetaGate()
    decision = "continue"
    while decision == "continue":
        decision = gate.add(rng.random(20) < 0.7, rng.random(20) < 0.4)
    assert decision == "accept"
