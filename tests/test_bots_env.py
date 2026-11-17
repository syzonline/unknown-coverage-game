import numpy as np
import pytest

from p2_opponent import bots as B
from p2_opponent.agents import FrequencyBRAgent, OracleBRAgent, RandomAgent
from p2_opponent.env import RepeatedGameEnv
from p2_opponent.games import GAMES, hide_seek, ipd, make_game, rps
from p2_opponent.runner import run_episode


def random_history(game, n, seed):
    rng = np.random.default_rng(seed)
    return tuple((int(rng.integers(game.n_agent)), int(rng.integers(game.n_opp))) for _ in range(n))


def test_games_well_formed():
    for name in GAMES:
        g = make_game(name)
        assert g.payoff_agent.shape == g.payoff_opp.shape == (g.n_agent, g.n_opp)
    r = rps(3)
    assert r.payoff_agent[1, 0] == 1 and r.payoff_agent[0, 1] == -1  # paper beats rock
    assert np.allclose(r.payoff_agent, -r.payoff_opp)
    r5 = rps(5)
    assert (np.sign(r5.payoff_agent).sum(axis=1) == 0).all()  # each action beats 2, loses to 2
    assert ipd().payoff_agent[1, 0] == 5 and ipd().payoff_opp[1, 0] == 0
    assert hide_seek(5).payoff_agent[2, 2] == 1


@pytest.mark.parametrize("game_name", list(GAMES))
@pytest.mark.parametrize("family", B.FAMILIES)
def test_bot_dists_valid_and_deterministic(game_name, family):
    g = make_game(game_name)
    bot = B.sample_bot(g, family, "dev", np.random.default_rng(0), horizon=50)
    for n in (0, 1, 7, 30):
        h = random_history(g, n, seed=n)
        d = bot.dist(h)
        assert d.shape == (g.n_opp,) and np.all(d >= 0) and d.sum() == pytest.approx(1.0)
        assert np.array_equal(d, bot.dist(h))


def test_fictitious_is_klevel_one_and_has_belief():
    g = rps(3)
    h = random_history(g, 25, 1)
    fp = B.KLevel(g, 1, 10, 0.2)
    belief = fp.belief(h)
    assert belief is not None and belief.sum() == pytest.approx(1.0)
    expected = B.softmax(belief @ g.payoff_opp, 0.2)
    assert np.allclose(fp.dist(h), expected)


def test_reactive_tit_for_tat():
    g = ipd()
    tft = B.Reactive(g, [0, 1], eps=0.0, variant="copy")
    assert tft.dist(((1, 0),))[1] == 1.0 and tft.dist(((0, 1),))[0] == 1.0


def test_switching_switches():
    g = rps(3)
    sw = B.Switching(g, B.Fixed(g, [1, 0, 0]), B.Fixed(g, [0, 0, 1]), t_switch=5)
    assert sw.dist(random_history(g, 4, 0))[0] == 1.0
    assert sw.dist(random_history(g, 5, 0))[2] == 1.0


def test_dev_test_param_ranges_disjoint():
    d, t = B.SPLITS["dev"], B.SPLITS["test"]
    assert not set(d.tau) & set(t.tau)
    assert not set(d.window) & set(t.window)
    assert not set(d.eps) & set(t.eps)
    assert not set(d.cycle_len) & set(t.cycle_len)
    assert d.switch_frac[1] <= t.switch_frac[0]


def test_env_reproducible_and_bounds():
    g = rps(3)
    bot = B.Fixed(g, [0.6, 0.3, 0.1])
    e1, e2 = RepeatedGameEnv(g, bot, 20, seed=3), RepeatedGameEnv(g, bot, 20, seed=3)
    o1 = [e1.step(0).opp_action for _ in range(20)]
    o2 = [e2.step(0).opp_action for _ in range(20)]
    assert o1 == o2
    with pytest.raises(RuntimeError):
        e1.step(0)


def test_oracle_has_zero_regret_and_freq_beats_random():
    g = rps(3)
    bot = B.Fixed(g, [0.7, 0.2, 0.1])
    oracle = run_episode(g, bot, OracleBRAgent(bot), 200, seed=0)
    assert np.nanmax(oracle.regret) == 0.0 and np.nanmax(oracle.kl) == 0.0
    freq = run_episode(g, bot, FrequencyBRAgent(), 200, seed=0).summary()
    rand = run_episode(g, bot, RandomAgent(), 200, seed=0).summary()
    assert freq["payoff_per_round"] > rand["payoff_per_round"]
    assert freq["regret"] < rand["regret"]
