"""v3 arms: longer-range proposer evidence, forced proposal schedule, redundancy record, cell-restricted jobs."""
import sys
from pathlib import Path

import numpy as np
import pytest

from bsi.sandbox import SandboxPool

from p2_opponent.agents import PosteriorBRAgent
from p2_opponent.bots import Cycle
from p2_opponent.games import make_game
from p2_opponent.hypotheses import normalize_prediction
from p2_opponent.propose import TASK_LAG1, TASK_RICH, build_messages, longer_range_views
from p2_opponent.runner import run_episode
from p2_opponent.seed_programs import SEED_LIBRARY

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_matrix as RM  # noqa: E402


def test_longer_range_views_expose_a_cycle_that_lag1_tables_hide():
    g = make_game("rps3")
    seq = [0, 0, 1]  # after "rock" the next action is rock or paper: not predictable from the last round
    history = tuple((2, seq[t % 3]) for t in range(30))
    text = longer_range_views(history, g)
    lags = dict(x.split(":") for x in text.splitlines()[1].split())
    assert float(lags["3"]) == 1.0 and float(lags["1"]) < 0.6
    for ph, name in enumerate(["rock", "rock", "paper"]):
        line = next(x for x in text.splitlines() if x.strip().startswith(f"t mod 3 = {ph}"))
        assert f"{name}=1.00" in line


@pytest.fixture
def own_pool():
    """A fresh pool: the session pool may still hold ids like habit#1 from agents other tests did not close."""
    with SandboxPool(n_workers=1) as p:
        yield p


def _cycle_agent(pool, **kw):
    lib = [s for s in SEED_LIBRARY if s.id == "habit"]
    return PosteriorBRAgent(pool, lib, **kw)


def test_rich_evidence_is_opt_in(own_pool):
    pool = own_pool
    g = make_game("rps3")
    agent = _cycle_agent(pool)
    res = run_episode(g, Cycle(g, [0, 0, 1], 0.05), agent, 40, seed=3)
    history = tuple(res.actions)
    plain = build_messages(agent.posterior, history, g)[1]["content"]
    rich = build_messages(agent.posterior, history, g, evidence="rich")[1]["content"]
    agent.close()
    assert TASK_LAG1.format(n_new=2) in plain and "Longer-range" not in plain
    assert TASK_RICH.format(n_new=2) in rich and "t mod 3 = 0" in rich


def test_propose_at_forces_proposals_and_records_gain_over_catch_all(own_pool):
    pool = own_pool
    g = make_game("rps3")
    cyc = next(s for s in SEED_LIBRARY if s.id == "cycle")
    calls = []

    def prop(posterior, history, game, tag="", meter=None):
        calls.append(len(history))
        return [cyc] if len(calls) == 1 else []

    # thresholds the trigger can never meet: only the schedule calls the proposer
    agent = _cycle_agent(pool, proposer=prop, propose_at=[20, 45], other_mass_threshold=2.0, other_margin=1e9)
    res = run_episode(g, Cycle(g, [0, 0, 1], 0.05), agent, 60, seed=3)
    assert calls == [20, 45]
    assert all(e.get("forced") for e in res.events if e["event"] == "propose")
    add = next(e for e in res.events if e["event"] == "add" and e["source"] == "proposer")
    assert add["ok"] and add["ll_gain_vs_other"] > 0.1  # the period-3 program out-predicts the last-round table
    agent.close()  # frees the sandbox ids (habit#1, ...) for the next agent, as run_matrix does per job

    agent = _cycle_agent(pool, proposer=prop, other_mass_threshold=2.0, other_margin=1e9)
    calls.clear()
    run_episode(g, Cycle(g, [0, 0, 1], 0.05), agent, 60, seed=3)
    agent.close()
    assert calls == []


def test_cells_restrict_jobs():
    cfg = {"games": ["rps3", "ipd"], "families": ["fixed", "cycle"], "n_params": 2, "seeds": [0],
           "methods": {"a": {"base": "bsi", "cells": [["ipd", "cycle"]]}, "b": None}}
    jobs = RM.build_jobs(cfg, ["a", "b"])
    assert {(j.game, j.family) for j in jobs if j.method == "a"} == {("ipd", "cycle")}
    assert sum(j.method == "a" for j in jobs) == 2 and sum(j.method == "b" for j in jobs) == 8


# ---------------------------------------------------------------- review fixes (v3-evidence)


def test_planning_stops_at_the_horizon():
    from p2_opponent.agents import OracleBRAgent
    from p2_opponent.bots import Reactive

    g = make_game("ipd")
    bot = Reactive(g, [0, 1], 0.0, "copy")  # tit-for-tat: cooperation pays only if a later round exists
    agent = OracleBRAgent(bot, plan_depth=3)
    agent.horizon = 10
    agent.reset(g, np.random.default_rng(0))
    coop = tuple((0, 0) for _ in range(9))
    assert agent.lookahead(coop, 3) == 1
    assert agent.act(coop)[0] == 1  # last round: defect
    assert agent.act(coop[:5])[0] == 0  # mid-episode: keep cooperating


def test_plan_predictions_match_reference_and_keep_real_context_cached(own_pool):
    from bsi.decide import plan_response

    g = make_game("ipd")
    lib = [s for s in SEED_LIBRARY if s.id in ("reactive", "wsls", "habit")]
    agent = PosteriorBRAgent(own_pool, lib, plan_depth=3)
    run_episode(g, Cycle(g, [0, 1, 1], 0.1), agent, 30, seed=5)
    history = tuple((1, 0) for _ in range(12))
    ids, w = agent.posterior.weights_array()

    def reference(ctx):  # posterior predictive with frozen weights, every program called directly
        out = 0
        for i, wi in zip(ids, w):
            h = agent.posterior.get(i)
            if i == agent.posterior.other_id:
                out = out + wi * h.predict(ctx)
            else:
                out = out + wi * normalize_prediction(own_pool.call(h.id, "model", (ctx, h.params, h.game_info)), h.n)
        return out

    a, pred = agent.act(history)
    assert a == plan_response(reference, history, g.payoff_agent, 3)
    for h in agent.posterior.hypotheses(include_other=False):
        assert h._key(history) in h._cache  # planning did not evict the real context
    agent.close()


def test_proposer_evidence_covers_only_the_segment_after_a_restart(own_pool):
    g = make_game("rps3")
    agent = _cycle_agent(own_pool)
    res = run_episode(g, Cycle(g, [0, 0, 1], 0.05), agent, 60, seed=3)
    history = tuple(res.actions)
    agent.posterior.restart(40)
    for ev in ("lag1", "rich"):
        text = build_messages(agent.posterior, history, g, evidence=ev)[1]["content"]
        assert "belief was reset at round 40" in text and "current segment, rounds 40-59" in text
        assert "(all rounds)" not in text
    lines = longer_range_views(history, g, start=40).splitlines()
    n_phase = sum(int(x.split("n=")[1].split()[0]) for x in lines if x.strip().startswith("t mod 3"))
    assert n_phase == 20
    agent.close()


def test_recovered_requires_the_end_of_the_episode():
    from p2_opponent.runner import post_switch_excess_kl

    pre = np.full(50, 0.1)
    late_rise = np.r_[pre, np.full(8, 0.1), np.full(92, 0.8)]  # fine for 8 rounds after the switch, then lost
    assert not post_switch_excess_kl(late_rise, 50)[1]
    back = np.r_[pre, np.full(30, 0.8), np.full(70, 0.12)]
    assert post_switch_excess_kl(back, 50)[1]


def test_config_problems_are_caught_before_running():
    cfg = {"methods": {"a": {"base": "bsi", "library": "seeds"},
                       "b": {"base": "b3_llm_decide", "plan_depth": 3},
                       "c": {"base": "bsi", "proposer_evidence": "rich"}}}
    probs = RM.config_problems(cfg, ["a", "b", "c"])
    assert len(probs) == 2 and probs[0].startswith("a:") and probs[1].startswith("b:")
