"""v4 controls: template proposer (no LLM), K0 with programs excluded, proposer reply provenance, cluster bootstrap."""
import sys
from pathlib import Path

import numpy as np
import pytest

from bsi.llm import LLMResult
from bsi.sandbox import SandboxPool, validate_source
from bsi.stats import cluster_bootstrap

from p2_opponent.agents import PosteriorBRAgent
from p2_opponent.bots import Cycle
from p2_opponent.games import make_game
from p2_opponent.hypotheses import ProgramHypothesis
from p2_opponent.propose import make_llm_proposer, make_template_proposer
from p2_opponent.runner import run_episode
from p2_opponent.seed_programs import SEED_LIBRARY, TEMPLATE_MENUS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_matrix as AM  # noqa: E402
import run_matrix as RM  # noqa: E402


@pytest.fixture
def own_pool():
    with SandboxPool(n_workers=1) as p:
        yield p


@pytest.mark.parametrize("game_name", ["rps3", "ipd", "hideseek5"])
def test_template_programs_are_valid_fast_and_small(own_pool, game_name):
    g = make_game(game_name)
    rng = np.random.default_rng(0)
    history = tuple((int(rng.integers(g.n_agent)), int(rng.integers(g.n_opp))) for _ in range(150))
    for spec in TEMPLATE_MENUS["full"]:
        assert not validate_source(spec.src), spec.id
        assert int(np.prod([len(v) for v in spec.param_grid.values()])) <= 96, spec.id  # agent max_grid: no subsampling
        assert not own_pool.load(spec.id, spec.src)
        h = ProgramHypothesis(spec.id, spec.src, own_pool, g.info(), spec.param_grid, spec.statement)
        for ctx in ((), history[:1], history[:2], history):
            p = h.predict(ctx)
            assert p.shape == (g.n_opp,) and np.all(p >= 0) and abs(p.sum() - 1) < 1e-9, (spec.id, len(ctx))
        own_pool.unload(spec.id)


def test_phase_table_predicts_an_arbitrary_cycle(own_pool):
    g = make_game("hideseek5")
    seq = [3, 0, 0, 4]  # not a rotation: lag-1 tables cannot express it
    history = tuple((0, seq[t % 4]) for t in range(40))
    spec = next(s for s in TEMPLATE_MENUS["periodic"] if s.id == "tpl_phase")
    assert not own_pool.load("ph", spec.src)
    h = ProgramHypothesis("ph", spec.src, own_pool, g.info(), {"period": [4], "alpha": [0.1]})
    p = h.predict(history)
    assert p.argmax() == seq[40 % 4] and p.max() > 0.9
    own_pool.unload("ph")


def test_template_proposer_offers_its_menu_once_at_the_trigger_and_fits_a_cycle(own_pool):
    g = make_game("rps3")
    prop = make_template_proposer("periodic")
    lib = [s for s in SEED_LIBRARY if s.id == "habit"]
    agent = PosteriorBRAgent(own_pool, lib, proposer=prop, min_obs_propose=20, propose_cooldown=30)
    res = run_episode(g, Cycle(g, [0, 0, 1], 0.05), agent, 120, seed=3)
    proposes = [e for e in res.events if e["event"] == "propose"]
    assert proposes and proposes[0]["t"] >= 20
    assert [x["n_specs"] for x in prop.log] == [3] + [0] * (len(prop.log) - 1)
    adds = [e for e in res.events if e["event"] == "add" and e["source"] == "proposer"]
    assert {e["statement"] for e in adds} == {s.statement for s in TEMPLATE_MENUS["periodic"]} and all(e["ok"] for e in adds)
    assert agent.belief_log[-1]["map_src"] == "proposer" and agent.belief_log[-1]["proposer"] > 0.8
    # no LLM proposer reply, so no reply provenance on the result event
    assert all("attempts" not in e for e in res.events if e["event"] == "propose_result")
    agent.close()


def test_llm_proposer_records_whether_the_reply_was_a_reasoning_off_repair():
    reply = "### Hypothesis: uniform\n```json\n{\"param_grid\": {}}\n```\n```python\ndef model(history, params, game):\n    return [0.5, 0.5]\n```"

    class StubLLM:
        def __init__(self, res):
            self.res = res

        def chat(self, *a, **k):
            return self.res

    class Post:  # what build_messages reads
        seg_start, mix_ll, other_id = 0, [], None

        def summary(self, top=4):
            return []

    g = make_game("ipd")
    history = ((0, 0),) * 5
    base = dict(text=reply, model="m", prompt_tokens=1, completion_tokens=2, finish_reason="stop", latency_s=0.1)
    degraded = make_llm_proposer(StubLLM(LLMResult(**base, reasoning="", attempts=2)))
    degraded(Post(), history, g)
    reasoned = make_llm_proposer(StubLLM(LLMResult(**base, reasoning="think", attempts=1)))
    reasoned(Post(), history, g)
    assert degraded.log[-1]["degraded"] and degraded.log[-1]["reasoning_chars"] == 0
    assert not reasoned.log[-1]["degraded"] and reasoned.log[-1]["reasoning_chars"] == 5


def test_propose_result_copies_the_reply_provenance(own_pool):
    g = make_game("rps3")
    log = []

    def prop(posterior, history, game, tag="", meter=None):
        log.append({"t": len(history), "attempts": 2, "degraded": True, "finish_reason": "stop", "n_specs": 0})
        return []

    prop.log = log
    lib = [s for s in SEED_LIBRARY if s.id == "habit"]
    agent = PosteriorBRAgent(own_pool, lib, proposer=prop, propose_at=[10])
    res = run_episode(g, Cycle(g, [0, 0, 1], 0.05), agent, 12, seed=3)
    ev = next(e for e in res.events if e["event"] == "propose_result")
    assert ev["attempts"] == 2 and ev["degraded"] is True and "n_specs" in ev
    agent.close()


def test_library_exclude_drops_programs_and_rejects_unknown_ids(tmp_path):
    cfg = {"name": "t", "split": "dev", "games": ["rps3"], "families": ["fixed"], "n_params": 1, "seeds": [0],
           "horizon": 10, "methods": {"keep": {"base": "bsi_seedlib"},
                                      "drop": {"base": "bsi_seedlib", "library_exclude": {"rps3": ["cycle"]}},
                                      "typo": {"base": "bsi_seedlib", "library_exclude": {"rps3": ["cycel"]}}}}
    mx = RM.Matrix(cfg, tmp_path, None)
    try:
        ids = lambda m: [s.id for s in mx.library(RM.Job(m, "rps3", "fixed", 0, 0))]
        assert "cycle" in ids("keep") and ids("drop") == [s.id for s in SEED_LIBRARY if s.id != "cycle"]
        mx.check_exclusions([RM.Job("drop", "rps3", "fixed", 0, 0)])
        with pytest.raises(SystemExit, match="cycel"):
            mx.check_exclusions([RM.Job("typo", "rps3", "fixed", 0, 0)])
        mx.cfg["methods"]["all"] = {"base": "bsi_seedlib", "library_exclude": {"rps3": [s.id for s in SEED_LIBRARY]}}
        with pytest.raises(SystemExit, match="every program"):
            mx.check_exclusions([RM.Job("all", "rps3", "fixed", 0, 0)])
    finally:
        mx.close()


def test_template_and_exclusion_options_are_validated():
    cfg = {"games": ["ipd", "rps3"], "methods": {"ok": {"base": "bsi", "template_proposer": "periodic", "library_exclude": {"ipd": ["k0_7"]}},
                       "menu": {"base": "bsi", "template_proposer": "periodc"},
                       "both": {"base": "bsi", "template_proposer": "full", "oracle_proposer": True},
                       "nobase": {"base": "b2_static", "template_proposer": "full"},
                       "excl": {"base": "b2_static", "library_exclude": ["k0_7"]},
                       "gamekey": {"base": "b2_static", "library_exclude": {"rps": ["k0_5"]}},
                       "nolib": {"base": "freq_br", "library_exclude": {"rps3": ["k0_5"]}}}}
    probs = RM.config_problems(cfg, list(cfg["methods"]))
    assert sorted(p.split(":")[0] for p in probs) == ["both", "excl", "gamekey", "menu", "nobase", "nolib"]
    assert not RM.uses_llm_proposer(cfg, "ok") and RM.uses_llm_proposer({"methods": {"x": {"base": "bsi"}}}, "x")


def test_cluster_bootstrap_widens_for_duplicated_units_and_computes_ratios():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.1, 1, 40), np.zeros(40)
    g = list(range(40))
    single = cluster_bootstrap(a, b, g)
    assert abs(single["mean_diff"] - a.mean()) < 1e-12 and single["clusters"] == 40
    # the same 40 values repeated 5 times: per-unit inference would claim 200 independent units
    a5, b5, g5 = np.tile(a, 5), np.tile(b, 5), g * 5
    dup = cluster_bootstrap(a5, b5, g5)
    assert dup["clusters"] == 40 and abs(dup["ci"][1] - dup["ci"][0] - (single["ci"][1] - single["ci"][0])) < 0.05
    r = cluster_bootstrap(np.array([3.0, 1.0]), np.array([1.0, 0.0]), ["x", "y"], ratio_den=(np.array([4.0, 2.0]), np.zeros(2)))
    assert abs(r["mean_diff"] - 3 / 6) < 1e-12 and "p_two_sided" not in r


def test_cluster_bootstrap_refuses_degenerate_resampling():
    one = cluster_bootstrap(np.ones(10), np.zeros(10), ["cell"] * 10)  # one cluster: every draw equals the estimate
    assert one["mean_diff"] == 1.0 and np.isnan(one["p_two_sided"]) and np.isnan(one["ci"][0])
    # draws whose ceiling denominator is not positive (a third here) are left out: no inf or sign-flipped share
    num, den = np.ones(6), np.array([6.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    with np.errstate(all="raise"):
        r = cluster_bootstrap(num, np.zeros(6), list(range(6)), ratio_den=(den, np.zeros(6)))
    assert np.all(np.isfinite(r["ci"])) and r["ci"][0] > 0
    ok = cluster_bootstrap(num, np.zeros(6), list(range(6)), ratio_den=(np.full(6, 2.0), np.zeros(6)))
    assert ok["ci"] == (0.5, 0.5)


def test_report_cluster_key_is_the_bot_cell():
    assert AM.cluster_of("rps3|cycle|p3|s4") == "rps3|cycle|p3"
