"""v2 arms: segment restart, change detection, planning, oracle proposer, method aliases, belief logging."""
import sys
from pathlib import Path

import numpy as np
import pytest

from bsi.changepoint import CusumDetector
from bsi.decide import plan_response, plan_values
from bsi.hypotheses import BackoffDirichletOther, FixedDist, Observation
from bsi.posterior import Posterior
from p2_opponent.agents import OracleBRAgent, PosteriorBRAgent
from p2_opponent.bots import FAMILIES, Reactive, sample_bot
from p2_opponent.games import make_game
from p2_opponent.propose import make_oracle_proposer, oracle_program_ids
from p2_opponent.runner import post_switch_excess_kl, run_episode
from p2_opponent.seed_programs import SEED_LIBRARY

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_matrix as RM  # noqa: E402


def _obs(outcomes, key=lambda i: ((0, 0),)):
    return [Observation(key(i), int(o)) for i, o in enumerate(outcomes)]


def test_restart_equals_fresh_posterior_on_the_segment():
    rng = np.random.default_rng(0)
    data = _obs(np.r_[np.zeros(40, int), rng.integers(0, 3, 40)])

    def build():
        return Posterior([FixedDist("a", [0.8, 0.1, 0.1]), FixedDist("u", [1, 1, 1])],
                         other=BackoffDirichletOther(3, key_fn=lambda c: c[-1]))

    p = build()
    for o in data:
        p.update(o)
    p.restart(50)
    q = build()
    q.observations = list(data[:50])  # contexts before the segment exist but carry no evidence
    q.seg_start = 50
    for o in data[50:]:
        q.update(o)
    pw, qw = p.weights(), q.weights()
    assert pw.keys() == qw.keys()
    assert all(abs(pw[k] - qw[k]) < 1e-9 for k in pw)


def test_cusum_detects_a_drop_and_stays_quiet_when_stationary():
    rng = np.random.default_rng(1)
    d = CusumDetector(k=0.3, h=4.0, burn_in=15)
    quiet = sum(d.update(x) for x in rng.normal(-0.3, 0.2, 200))
    assert quiet == 0
    d = CusumDetector(k=0.3, h=4.0, burn_in=15)
    xs = np.r_[rng.normal(-0.3, 0.2, 60), rng.normal(-2.0, 0.5, 60)]
    alarms = [t for t, x in enumerate(xs) if d.update(x)]
    assert alarms and 60 <= alarms[0] < 70


def test_planning_cooperates_with_tit_for_tat_in_ipd():
    g = make_game("ipd")
    C, D = 0, 1
    assert g.payoff_agent[D, C] > g.payoff_agent[C, C]  # myopic best response is to defect

    def tft(history):
        p = np.zeros(2)
        p[history[-1][0] if history else C] = 1.0
        return p

    assert plan_response(tft, ((C, C),), g.payoff_agent, depth=1) == D
    assert plan_response(tft, ((C, C),), g.payoff_agent, depth=3) == C
    assert np.allclose(plan_values(tft, (), g.payoff_agent, 1), g.payoff_agent @ tft(()))


def test_oracle_plan_beats_myopic_oracle_against_reactive_copy_bot():
    g = make_game("ipd")
    bot = Reactive(g, [0, 1], 0.05, "copy")
    pay = {}
    for depth in (0, 3):
        res = run_episode(g, bot, OracleBRAgent(bot, plan_depth=depth), 60, seed=3)
        pay[depth] = res.payoff
    assert pay[3] > pay[0]


def test_oracle_proposer_covers_every_family():
    g = make_game("rps3")
    seed_ids = {s.id for s in SEED_LIBRARY}
    rng = np.random.default_rng(0)
    for fam in FAMILIES:
        desc = sample_bot(g, fam, "dev", rng, 150).describe()
        ids = oracle_program_ids(desc)
        assert ids and set(ids) <= seed_ids, fam
    prop = make_oracle_proposer(sample_bot(g, "wsls", "dev", rng, 150).describe())
    first = prop(posterior=None, history=(), game=g)
    assert [s.src for s in first] == [s.src for s in SEED_LIBRARY if s.id == "wsls"]
    assert prop(posterior=None, history=(), game=g) == []


def test_method_aliases_resolve_to_registered_bases():
    cfg = {"methods": {"b2_static": {}, "bsi_v2": {"base": "bsi", "min_obs_propose": 20},
                       "bsi_oracle_prop": {"base": "bsi", "oracle_proposer": True},
                       "bsi_minimal_v2": {"base": "bsi_minimal"}, "seedlib_cp": {"base": "bsi_seedlib", "changepoint": {}}}}
    assert RM.resolve_base(cfg, "bsi_v2") == "bsi" and RM.resolve_base(cfg, "b2_static") == "b2_static"
    assert RM.library_kind(cfg, "bsi_v2") == "k0" and RM.library_kind(cfg, "bsi_minimal_v2") == "minimal"
    assert RM.library_kind(cfg, "seedlib_cp") == "seed"
    assert RM.uses_llm_proposer(cfg, "bsi_v2") and not RM.uses_llm_proposer(cfg, "bsi_oracle_prop")
    assert not RM.uses_llm_proposer(cfg, "seedlib_cp")


def test_post_switch_excess_kl_defined_without_pre_switch_tracking():
    kls = np.r_[np.full(50, 0.6), np.full(30, 0.9), np.full(70, 0.55)]
    ex, rec, post = post_switch_excess_kl(kls, 50)
    assert ex == pytest.approx(0.3) and rec and post == pytest.approx(0.9)


def _proposer_after(n_min):
    calls = []

    def proposer(posterior, history, game, tag="", meter=None):
        calls.append(len(history))
        return []

    return proposer, calls


def test_min_obs_propose_and_belief_log(pool):
    g = make_game("rps3")
    bot = sample_bot(g, "cycle", "dev", np.random.default_rng(0), 60)
    lib = [s for s in SEED_LIBRARY if s.id == "habit"]  # cannot explain a cycle: the catch-all wins early
    for n_min in (0, 20):
        prop, calls = _proposer_after(n_min)
        agent = PosteriorBRAgent(pool, lib, proposer=prop, min_obs_propose=n_min, propose_cooldown=5)
        res = run_episode(g, bot, agent, 60, seed=1)
        assert calls and min(calls) >= n_min
        assert len(res.belief) == 60 and {"other", "proposer", "map", "map_src", "n"} <= set(res.belief[0])
        assert 0.0 <= res.summary()["other_mass_mean"] <= 1.0
    assert min(_proposer_calls_without_floor(pool, g, bot, lib)) < 20


def _proposer_calls_without_floor(pool, g, bot, lib):
    prop, calls = _proposer_after(0)
    run_episode(g, bot, PosteriorBRAgent(pool, lib, proposer=prop, propose_cooldown=5), 60, seed=1)
    return calls


def test_changepoint_restarts_posterior_on_switching_bot(pool):
    g = make_game("rps3")
    bot = sample_bot(g, "switching", "dev", np.random.default_rng(4), 150)
    agent = PosteriorBRAgent(pool, SEED_LIBRARY, changepoint={"k": 0.3, "h": 4.0})
    res = run_episode(g, bot, agent, 150, seed=2)
    cps = [e for e in res.events if e["event"] == "changepoint"]
    assert all(e["seg_start"] <= e["t"] for e in cps)
    assert res.summary()["n_changepoints"] == len(cps)


def test_multi_dir_merge_dedupes_identical_and_refuses_conflicts(tmp_path):
    import json

    from analyze_matrix import load

    row = {"job_key": "b2_static|rps3|fixed|p0|s0", "method": "b2_static", "status": "ok",
           "payoff_per_round": 0.5, "kl": 0.1, "regret": 0.02}
    a, b, c = (tmp_path / x for x in "abc")
    for d, extra in ((a, {}), (b, {"other_mass_mean": 0.2}), (c, {"kl": 0.3})):
        d.mkdir()
        (d / "summaries.jsonl").write_text(json.dumps({**row, **extra}) + "\n")
    rows, source = load([a, b])
    assert len(rows) == 1 and rows[0]["other_mass_mean"] == 0.2 and source[row["job_key"]] == b
    with pytest.raises(SystemExit):
        load([a, c])
    rows, source = load([a, c], supersede=("b2_static",))  # fixed and re-run: c replaces a, no conflict
    assert len(rows) == 1 and rows[0]["kl"] == 0.3 and source[row["job_key"]] == c


def test_report_counts_recovery_only_on_applicable_episodes_and_skips_checks_without_ref(tmp_path):
    import json

    import yaml
    from analyze_matrix import analyze

    import test_matrix as TM

    c = TM.cfg(games=["rps3"], families=["switching"], n_params=2, seeds=[0], horizon=120,
               methods={"random": {}, "freq_br": {}, "oracle_br": {}})
    cp = tmp_path / "c.yaml"
    cp.write_text(yaml.safe_dump(c))
    r = TM._run(cp, tmp_path / "run")
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    text = analyze(tmp_path / "run")
    rows = {l.split("|")[1].strip(): [x.strip() for x in l.split("|")[2:-1]] for l in text.splitlines()
            if l.startswith("| ") and "switching" not in l and len(l.split("|")) == 10}
    # random never tracks the bot: no applicable episodes, so no recovery rate (it used to show 1.00 / trivial)
    assert rows["random"][3] == "–" and rows["random"][4] == "0"
    assert rows["oracle_br"][3] == "1.00"
    assert "_skipped: none of these runs contains the method" in text and "`None`" not in text
