"""Acceptance tests for week 1: seed programs reproduce the bots exactly, and the posterior recovers
the generating family and parameters from play."""
import numpy as np
import pytest

from bsi.hypotheses import DirichletOther, Observation
from bsi.posterior import Posterior
from p2_opponent import bots as B
from p2_opponent.agents import PosteriorBRAgent, RandomAgent
from p2_opponent.env import RepeatedGameEnv
from p2_opponent.games import hide_seek, ipd, rps
from p2_opponent.hypotheses import ProgramHypothesis
from p2_opponent.runner import run_episode
from p2_opponent.seed_programs import CYCLE, HABIT, KLEVEL, SEED_LIBRARY, WSLS


def random_history(game, n, seed):
    rng = np.random.default_rng(seed)
    return tuple((int(rng.integers(game.n_agent)), int(rng.integers(game.n_opp))) for _ in range(n))


@pytest.mark.parametrize("game", [rps(3), rps(5), ipd(), hide_seek(5)], ids=lambda g: g.name)
def test_seed_programs_match_bots_exactly(pool, game):
    cases = [
        (KLEVEL, {"k": 2, "w": 10, "tau": 0.2}, B.KLevel(game, 2, 10, 0.2)),
        (KLEVEL, {"k": 3, "w": 5, "tau": 0.5}, B.KLevel(game, 3, 5, 0.5)),
        (KLEVEL, {"k": 1, "w": 20, "tau": 0.05}, B.KLevel(game, 1, 20, 0.05)),
        (WSLS, {"eps": 0.1}, B.WSLS(game, 0.1)),
        (HABIT, {"w": 5}, B.KLevel(game, 0, 5, 1.0)),
    ]
    for k, (src, params, bot) in enumerate(cases):
        pid = f"match_{game.name}_{k}"
        assert pool.load(pid, src) == []
        for n in (0, 1, 6, 40):
            h = random_history(game, n, seed=n + k)
            got = pool.call(pid, "model", (h, params, game.info()))
            assert np.allclose(got, bot.dist(h), atol=1e-9), (pid, n)
        pool.unload(pid)


def play(game, bot, horizon, seed):
    env = RepeatedGameEnv(game, bot, horizon, seed)
    rng = np.random.default_rng(seed)
    obs = []
    for _ in range(horizon):
        h = env.history
        r = env.step(int(rng.integers(game.n_agent)))
        obs.append(Observation(h, r.opp_action))
    return obs


@pytest.mark.parametrize("make_bot,family,check", [
    (lambda g: B.KLevel(g, 2, 10, 0.2), "klevel", {"k": 2, "w": 10, "tau": 0.2}),
    (lambda g: B.KLevel(g, 1, 20, 0.1), "klevel", {"k": 1, "tau": 0.1}),
    (lambda g: B.Cycle(g, [0, 2, 1], 0.05), "cycle", {"period": 3}),
    (lambda g: B.WSLS(g, 0.1), "wsls", {}),
    (lambda g: B.Reactive(g, [2, 0, 1], 0.1), "reactive", {}),
])
def test_posterior_recovers_family_and_params(pool, make_bot, family, check):
    game = rps(3)
    obs = play(game, make_bot(game), horizon=150, seed=11)
    hyps = []
    for spec in SEED_LIBRARY:
        pid = f"rec_{family}_{spec.id}_{len(check)}"
        assert pool.load(pid, spec.src) == []
        hyps.append(ProgramHypothesis(pid, spec.src, pool, game.info(), spec.param_grid, spec.statement, max_grid=96))
    post = Posterior(other=DirichletOther(game.n_opp))
    for h in hyps:
        post.add(h)
    for o in obs:
        post.update(o)
    post.refit()
    best = post.get(post.map_id())
    assert best.id.startswith(f"rec_{family}_{family}"), post.summary()
    for k, v in check.items():
        assert best.params[k] == v, (best.params, check)
    assert post.weights()[best.id] > 0.9
    for h in hyps:
        pool.unload(h.id)


def test_bsi_agent_tracks_bot_without_llm(pool):
    game = rps(3)
    bot = B.KLevel(game, 1, 10, 0.1)
    agent = PosteriorBRAgent(pool, SEED_LIBRARY, proposer=None)
    res = run_episode(game, bot, agent, horizon=120, seed=5).summary()
    rand = run_episode(game, bot, RandomAgent(), horizon=120, seed=5).summary()
    agent.close()
    assert res["kl_second_half"] < 0.1, res
    assert res["payoff_per_round"] > rand["payoff_per_round"] + 0.2, (res, rand)


def test_agent_rejects_slow_and_invalid_programs(pool):
    from p2_opponent.agents import ProgramSpec

    game = rps(3)
    agent = PosteriorBRAgent(pool, [], max_call_ms=5.0)
    agent.reset(game, np.random.default_rng(0))
    hist = random_history(game, 60, 0)
    slow = "def model(history, params, game):\n    s = 0\n    for i in range(200000):\n        s += i\n    return np.ones(3) / 3\n"
    wrong_len = "def model(history, params, game):\n    return np.ones(5)\n"
    ok, reason = agent.try_add(ProgramSpec("slow", slow), hist, "test")
    assert not ok and "too slow" in reason
    ok, reason = agent.try_add(ProgramSpec("bad", wrong_len), hist, "test")
    assert not ok and "invalid prediction" in reason
    ok, _ = agent.try_add(ProgramSpec("good", "def model(history, params, game):\n    return np.ones(3)\n"), hist, "test")
    assert ok and len(agent.posterior.ids(include_other=False)) == 1
    agent.close()


def test_param_grid_never_materializes_huge_products():
    import resource
    import time

    from p2_opponent.hypotheses import param_grid

    huge = {f"p{i}": list(range(50)) for i in range(8)}  # 50^8 ≈ 3.9e13 combinations
    t0 = time.time()
    pts = param_grid(huge, 96, seed=1)
    assert len(pts) == 96 and len({tuple(sorted(p.items())) for p in pts}) == 96
    assert time.time() - t0 < 1.0
    assert pts == param_grid(huge, 96, seed=1)  # deterministic
    assert len(param_grid({"a": [1, 2], "b": [3, 4, 5]}, 96)) == 6
    assert len(param_grid({"a": list(range(1000))}, 96)) == 50  # values per parameter are capped
    assert param_grid({}, 96) == [{}] and param_grid({"a": []}, 96) == [{}]
