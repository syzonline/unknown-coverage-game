import numpy as np
import pytest

from bsi.hypotheses import DirichletOther, FixedDist, FunctionHypothesis, Observation
from bsi.posterior import Posterior


def coin_obs(p, n, seed=0):
    rng = np.random.default_rng(seed)
    return [Observation(None, int(rng.random() < p)) for _ in range(n)]


def test_concentrates_on_true_hypothesis():
    post = Posterior([FixedDist("p20", [0.8, 0.2]), FixedDist("p50", [0.5, 0.5]), FixedDist("p80", [0.2, 0.8])])
    for o in coin_obs(0.8, 200):
        post.update(o)
    assert post.map_id() == "p80"
    assert post.weights()["p80"] > 0.99


def test_add_late_equals_add_early():
    obs = coin_obs(0.7, 50, seed=1)
    a = Posterior([FixedDist("x", [0.3, 0.7]), FixedDist("y", [0.6, 0.4])])
    b = Posterior([FixedDist("x", [0.3, 0.7])])
    for o in obs:
        a.update(o)
        b.update(o)
    b.add(FixedDist("y", [0.6, 0.4]))
    assert a.weights() == pytest.approx(b.weights())


def test_other_bucket_triggers_when_space_is_wrong():
    post = Posterior([FixedDist("p10", [0.9, 0.1]), FixedDist("p30", [0.7, 0.3])], other=DirichletOther(2), window=20)
    for o in coin_obs(0.95, 120, seed=2):
        post.update(o)
    assert post.other_triggered()
    assert post.map_id(include_other=True) == "other"


def test_other_bucket_quiet_when_space_is_right():
    post = Posterior([FixedDist("p95", [0.05, 0.95]), FixedDist("p30", [0.7, 0.3])], other=DirichletOther(2), window=20)
    for o in coin_obs(0.95, 120, seed=3):
        post.update(o)
    assert not post.other_triggered()


def test_mdl_prior_prefers_simpler_when_equal():
    post = Posterior([FixedDist("simple", [0.5, 0.5], complexity=10), FixedDist("complex", [0.5, 0.5], complexity=200)])
    assert post.weights()["simple"] > post.weights()["complex"]


def test_failing_hypothesis_is_dropped():
    def boom(ctx, params):
        raise RuntimeError("bad")

    post = Posterior([FixedDist("ok", [0.5, 0.5]), FunctionHypothesis("bad", boom)])
    post.update(Observation(None, 1))
    assert "bad" not in post and "bad" in post.failed


def test_predictive_and_prune():
    post = Posterior([FixedDist(f"h{i}", [i + 1, 10 - i]) for i in range(6)], other=DirichletOther(2))
    for o in coin_obs(0.2, 60, seed=4):
        post.update(o)
    pred = post.predictive(None)
    assert pred.sum() == pytest.approx(1.0)
    removed = post.prune(2)
    assert len(post.ids(include_other=False)) == 2 and len(removed) == 4 and "other" in post


def test_mixture_loglik_is_prequential():
    post = Posterior([FixedDist("a", [0.5, 0.5])])
    post.update(Observation(None, 0))
    assert post.mix_ll[0] == pytest.approx(np.log(0.5))


def test_backoff_catch_all_learns_first_order_structure():
    from bsi.hypotheses import BackoffDirichletOther

    other = BackoffDirichletOther(3, key_fn=lambda ctx: ctx)
    post = Posterior([FixedDist("uniform", [1, 1, 1])], other=other, window=20)
    rng = np.random.default_rng(0)
    prev = 0
    for _ in range(150):  # outcome = (prev + 1) mod 3: marginal uniform, conditional deterministic
        y = (prev + 1) % 3
        post.update(Observation(prev, y))
        prev = y if rng.random() > 0.05 else int(rng.integers(3))
    assert post.other_triggered()
    assert other.predict(0)[1] > 0.8
