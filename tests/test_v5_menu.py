"""v5: families outside the seed menu, their true-class programs, the menu-choice prompt/parser and agent."""
import sys
from pathlib import Path

import numpy as np
import pytest

from bsi.hypotheses import Observation
from bsi.llm import LLMResult
from bsi.sandbox import SandboxPool, validate_source
from bsi.stats import kl

from p2_opponent.agents import ProgramSpec
from p2_opponent.bots import BASE_FAMILIES, FAMILIES, NOVEL_FAMILIES, Cycle, describe_rules, sample_bot
from p2_opponent.games import make_game
from p2_opponent.hypotheses import ProgramHypothesis
from p2_opponent.menu_agent import MenuChoiceAgent
from p2_opponent.menu_prompt import ActiveModel, DecisionError, build_messages, parse_decision, render_history
from p2_opponent.propose import oracle_specs
from p2_opponent.runner import run_episode
from p2_opponent.seed_programs import NOVEL_PROGRAMS, SEED_LIBRARY

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_matrix as RM  # noqa: E402


@pytest.fixture
def own_pool():
    with SandboxPool(n_workers=1) as p:
        yield p


def _play(game, bot, T=150, seed=2):
    r = np.random.default_rng(seed)
    h = []
    for _ in range(T):
        p = bot.dist(tuple(h))
        h.append((int(r.integers(game.n_agent)), int(r.choice(game.n_opp, p=p))))
    return tuple(h)


def test_novel_families_leave_earlier_sampling_unchanged():
    assert set(NOVEL_FAMILIES).isdisjoint(FAMILIES) and set(NOVEL_FAMILIES).isdisjoint(BASE_FAMILIES)
    g = make_game("rps3")
    a = sample_bot(g, "switching", "dev", np.random.default_rng(5)).describe()
    b = sample_bot(g, "switching", "dev", np.random.default_rng(5)).describe()
    assert a == b and a["params"]["first"]["family"] in BASE_FAMILIES


@pytest.mark.parametrize("game_name", ["rps3", "ipd", "hideseek5"])
@pytest.mark.parametrize("family", NOVEL_FAMILIES)
def test_novel_bots_are_valid_and_their_programs_recover_them(own_pool, game_name, family):
    g = make_game(game_name)
    bot = sample_bot(g, family, "dev", np.random.default_rng(1))
    assert describe_rules(bot)
    h = _play(g, bot)
    for t in (0, 1, 2, 50, 149):
        p = bot.dist(h[:t])
        assert p.shape == (g.n_opp,) and abs(p.sum() - 1) < 1e-9 and (p >= 0).all()
    spec = NOVEL_PROGRAMS[family]
    assert not validate_source(spec.src)
    assert not own_pool.load(family, spec.src)
    hyp = ProgramHypothesis(family, spec.src, own_pool, g.info(), spec.param_grid)
    hyp.refit([Observation(h[:t], h[t][1]) for t in range(len(h))])
    late = np.mean([kl(bot.dist(h[:t]), hyp.predict(h[:t])) for t in range(100, 150)])
    assert late < 0.06, (family, hyp.params, late)
    own_pool.unload(family)


def test_oracle_specs_cover_novel_and_seed_families():
    g = make_game("rps3")
    for fam in NOVEL_FAMILIES + ("cycle", "fictitious"):
        specs = oracle_specs(sample_bot(g, fam, "dev", np.random.default_rng(0)).describe())
        assert specs and all(s.id.startswith("oracle_") for s in specs)


# ---------------------------------------------------------------- prompt and parser

MENU = [s for s in SEED_LIBRARY]
MENU_IDS = {s.id for s in MENU}
PROG = "def model(history, params, game):\n    n = game['n_opp']\n    return np.ones(n) / n\n"


def _reply(actions: str, models: dict[str, str] | None = None) -> str:
    out = f'```json\n{{"actions": {actions}, "rationale": "r"}}\n```\n'
    for name, code in (models or {}).items():
        out += f"### Model: {name}\n```python\n{code}```\n"
    return out


def test_parse_decision_orders_ops_and_checks_names_grids_and_cap():
    active = {"wsls": ActiveModel("wsls", "menu", MENU[2])}
    ops, why = parse_decision(_reply('[{"op": "create", "id": "u1", "statement": "s", "param_grid": {}},'
                                     ' {"op": "select", "menu_id": "cycle", "param_grid": {"period": [6], "eps": [0.1]}},'
                                     ' {"op": "retire", "target": "wsls"}]', {"u1": PROG}),
                              "select_create", MENU_IDS, active, 4)
    assert [o.op for o in ops] == ["retire", "select", "create"] and why == "r"
    assert ops[1].param_grid == {"period": [6], "eps": [0.1]} and ops[2].code.startswith("def model")
    bad = [('[{"op": "select", "menu_id": "nope"}]', {}, "unknown menu_id"),
           ('[{"op": "create", "id": "wsls"}]', {"wsls": PROG}, "already used"),
           ('[{"op": "create", "id": "u2"}]', {}, "no '### Model: u2'"),
           ('[{"op": "retire", "target": "cycle"}]', {}, "not active"),
           ('[{"op": "select", "menu_id": "klevel", "param_grid": {"k": [1,2,3,4,5], "w": [1,2,3,4,5], "tau": [1,2,3,4,5]}}]', {}, "combinations"),
           ('[{"op": "create", "id": "u3"}]', {"u3": "import os\ndef model(history, params, game):\n    return [1]\n"}, "sandbox"),
           ('[{"op": "select", "menu_id": "habit"}, {"op": "select", "menu_id": "cycle"}, {"op": "select", "menu_id": "klevel"},'
            ' {"op": "select", "menu_id": "reactive"}]', {}, "limit"),
           ('[{"op": "revise", "target": "wsls"}]', {}, "needs a param_grid")]
    for acts, models, msg in bad:
        with pytest.raises(DecisionError, match=msg):
            parse_decision(_reply(acts, models), "select_create", MENU_IDS, active, 4)
    with pytest.raises(DecisionError, match="not available"):
        parse_decision(_reply('[{"op": "select", "menu_id": "cycle"}]'), "create_only", MENU_IDS, {}, 4)


def test_history_window_and_full_rendering():
    g = make_game("rps3")
    h = tuple((t % 3, (t + 1) % 3) for t in range(100))
    full = render_history(h, g, "full", 40)
    win = render_history(h, g, "window", 40)
    assert "All rounds are shown" in full and full.count("\n") > win.count("\n")
    assert "t = 60..99" in win and "\n 59 |" not in win and "\n 60 |" in win and "\n  0 |" in full
    assert "shown rounds" in win and "shown rounds" not in full
    # nothing raw from the hidden rounds: the window's counts cover the 40 shown rounds only
    assert "all rounds" not in win and "shown rounds: [13, 14, 13]" in win and "all rounds: [33, 34, 33]" in full


def test_prompt_counterbalances_the_two_options_and_hides_the_menu_in_create_only():
    from p2_opponent.menu_prompt import PromptState
    g = make_game("rps3")
    st = PromptState(t=10, horizon=150, decision_index=0, n_decisions=6, fits=[], reference=None, mixture_mean_ll_recent=-1.0)
    h = tuple((0, 1) for _ in range(10))
    a = build_messages(g, MENU, {}, st, h, "select_create", "full", 40, 4, flip=False)[1]["content"]
    b = build_messages(g, MENU, {}, st, h, "select_create", "full", 40, 4, flip=True)[1]["content"]
    assert a.index("activate one or more MENU") < a.index("write one or more NEW")
    assert b.index("activate one or more MENU") > b.index("write one or more NEW")
    # the output-format examples follow the same order
    assert a.index('{"op": "select"') < a.index('{"op": "create"') and b.index('{"op": "select"') > b.index('{"op": "create"')
    c = build_messages(g, MENU, {}, st, h, "create_only", "full", 40, 4, flip=False)[1]["content"]
    assert "MENU (prewritten" not in c and "def model(history, params, game):\n    \"\"\"Win-stay" not in c


# ---------------------------------------------------------------- agent with a scripted LLM


class ScriptLLM:
    """Replies from a list, by call order; records every request."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, model="m", temperature=0.7, seed_tag="", validate=None, meter=None, **kw):
        text = self.replies[len(self.calls)] if len(self.calls) < len(self.replies) else _reply('[{"op": "keep"}]')
        self.calls.append({"messages": messages, "seed_tag": seed_tag})
        res = LLMResult(text=text, reasoning="x", model=model, prompt_tokens=10, completion_tokens=5,
                        finish_reason="stop", latency_s=0.0)
        if validate is not None:
            assert validate(res) is None, validate(res)
        if meter is not None:
            meter["calls"] = meter.get("calls", 0) + 1
        return res


PHASE = '''def model(history, params, game):
    n = game["n_opp"]
    p = int(params["period"])
    t = len(history)
    counts = np.full(n, 0.1)
    for r in range(t % p, t, p):
        counts[history[r][1]] += 1.0
    return counts / counts.sum()
'''
SLOW_BROKEN = "def model(history, params, game):\n    return [1.0]\n"  # wrong length: rejected at load/smoke


def test_agent_applies_select_create_revise_retire_and_repairs_a_failed_program(own_pool):
    g = make_game("rps3")
    llm = ScriptLLM([
        _reply('[{"op": "select", "menu_id": "habit"}, {"op": "create", "id": "bad", "statement": "s", "param_grid": {}}]',
               {"bad": SLOW_BROKEN}),
        _reply('[{"op": "create", "id": "bad", "statement": "s", "param_grid": {"period": [3]}}]', {"bad": PHASE}),  # repair turn
        _reply('[{"op": "revise", "target": "bad", "param_grid": {"period": [2, 3, 4]}}, {"op": "retire", "target": "habit"}]'),
    ])
    agent = MenuChoiceAgent(own_pool, MENU, llm, decision_at=[10, 20], k_active=2, name="mc")
    res = run_episode(g, Cycle(g, [0, 0, 1], 0.05), agent, 60, seed=3, episode_key="rps3|cycle|p0|s0")
    dec = [e for e in res.events if e["event"] == "decision"]
    assert [d["t"] for d in dec] == [10, 20]
    first = dec[0]
    assert [(o["op"], o["name"], o["ok"]) for o in first["ops"]] == [("select", "habit", True), ("create", "bad", False),
                                                                    ("create", "bad", True)]
    assert first["ops"][-1]["repair"] and llm.calls[1]["seed_tag"].endswith(":fix")
    assert "rejected" in llm.calls[1]["messages"][-1]["content"]
    assert [(o["op"], o["name"], o["ok"]) for o in dec[1]["ops"]] == [("retire", "habit", True), ("revise", "bad", True)]
    assert [a["name"] for a in dec[1]["active"]] == ["bad"] and dec[1]["active"][0]["revisions"] == 1
    # the second prompt shows the first decision's outcome, the created program and its fit
    p2 = llm.calls[2]["messages"][1]["content"]
    assert "create bad (FAILED" in p2 and "- bad (origin: created)" in p2 and "counts[history[r][1]]" in p2
    b = agent.belief_log[-1]
    assert b["active"] == ["bad"] and b["by_src"].get("revised", 0) > 0.8 and b["map_src"] == "revised"
    assert res.summary()["llm_calls"] == 3
    agent.close()


def test_menu_all_create_starts_with_the_menu_and_create_only_hides_it(own_pool):
    g = make_game("ipd")
    llm = ScriptLLM([])
    agent = MenuChoiceAgent(own_pool, MENU, llm, mode="menu_all_create", decision_at=[5], k_active=2)
    run_episode(g, Cycle(g, [0, 1], 0.05), agent, 8, seed=1, episode_key="e")
    assert sorted(agent.active) == sorted(MENU_IDS) and agent.k_active == len(MENU) + 2
    assert "All menu models were active" in llm.calls[0]["messages"][1]["content"]
    agent.close()
    llm2 = ScriptLLM([])
    agent2 = MenuChoiceAgent(own_pool, MENU, llm2, mode="create_only", decision_at=[5])
    run_episode(g, Cycle(g, [0, 1], 0.05), agent2, 8, seed=1, episode_key="e")
    assert not agent2.active and "MENU (prewritten" not in llm2.calls[0]["messages"][1]["content"]
    agent2.close()


def test_unusable_reply_changes_nothing(own_pool):
    class Broken(ScriptLLM):
        def chat(self, *a, **k):
            raise RuntimeError("invalid")

    g = make_game("rps3")
    agent = MenuChoiceAgent(own_pool, MENU, Broken([]), decision_at=[5])
    res = run_episode(g, Cycle(g, [0, 1], 0.05), agent, 8, seed=1, episode_key="e")
    ev = next(e for e in res.events if e["event"] == "decision")
    assert "error" in ev and ev["ops"] == [] and not agent.active
    agent.close()


# ---------------------------------------------------------------- runner options


def test_v5_runner_options_are_validated_and_shape_the_library(tmp_path):
    cfg = {"name": "t", "split": "dev", "games": ["rps3"], "families": ["react2", "cycle"], "n_params": 1, "seeds": [0],
           "horizon": 30,
           "methods": {"menu_all": {"base": "b2_static", "library": "seed"},
                       "oracle": {"base": "b2_static", "library": "seed", "library_add_oracle": True},
                       "loo": {"base": "b2_static", "library": "seed", "menu_exclude_true_family": True},
                       "choose": {"base": "menu_choice", "history_mode": "window", "history_window": 40, "decision_at": [10, 20]}}}
    assert RM.config_problems(cfg, list(cfg["methods"])) == []
    bad = {"games": ["rps3"], "horizon": 30, "methods": {
        "a": {"base": "menu_choice", "menu_mode": "pick"}, "b": {"base": "menu_choice", "history_mode": "all"},
        "c": {"base": "menu_choice", "decision_at": [0, 40]}, "d": {"base": "b2_static", "history_window": 10},
        "e": {"base": "freq_br", "library_add_oracle": True},
        "f": {"base": "b2_static", "library_add_oracle": True, "menu_exclude_true_family": True}}}
    assert sorted({p.split(":")[0] for p in RM.config_problems(bad, list(bad["methods"]))}) == list("abcdef")
    assert RM.uses_llm(cfg, "choose") and not RM.uses_llm(cfg, "oracle")
    mx = RM.Matrix(cfg, tmp_path, None)
    try:
        job = RM.Job("oracle", "rps3", "react2", 0, 0)
        _, bot = RM.make_bot(cfg, job)
        assert [s.id for s in mx.library_for(job, bot)][-1] == "oracle_react2"
        job = RM.Job("loo", "rps3", "cycle", 0, 0)
        _, bot = RM.make_bot(cfg, job)
        assert "cycle" not in {s.id for s in mx.library_for(job, bot)}
    finally:
        mx.close()


def test_report_has_the_menu_choice_section(own_pool, tmp_path):
    import analyze_matrix as AM
    from bsi.store import TraceWriter, dumps

    g = make_game("rps3")
    tw = TraceWriter(tmp_path / "traces-1.jsonl.gz")
    with open(tmp_path / "summaries.jsonl", "w") as f:
        for p in range(2):
            for fam, reply in (("cycle", '[{"op": "select", "menu_id": "cycle"}]'),
                               ("react2", '[{"op": "create", "id": "u", "statement": "s", "param_grid": {}}]')):
                bot = sample_bot(g, fam, "dev", np.random.default_rng(p))
                ep = f"rps3|{fam}|p{p}|s0"
                for m, agent in (("choose", MenuChoiceAgent(own_pool, MENU, ScriptLLM([_reply(reply, {"u": PROG})]),
                                                            decision_at=[10])),
                                 ("menu_all", None)):
                    if agent is None:
                        from p2_opponent.agents import PosteriorBRAgent
                        agent = PosteriorBRAgent(own_pool, MENU, name=m)
                    res = run_episode(g, bot, agent, 30, seed=p, episode_key=ep)
                    agent.close()
                    s = {**res.summary(), "method": m, "family": fam, "param": p, "env_seed": 0,
                         "job_key": f"{m}|{ep}", "episode": ep, "status": "ok"}
                    f.write(dumps(s) + "\n")
                    tw.write({**res.to_record(), "job_key": s["job_key"], "method": m})
    tw.close()
    text = AM.analyze(tmp_path)
    sec = text.split("## Menu-choice diagnostics (v5)")[1]
    row_in = next(l for l in sec.splitlines() if l.startswith("| choose | in |"))
    row_out = next(l for l in sec.splitlines() if l.startswith("| choose | out |"))
    cells = [c.strip() for c in row_in.split("|")[1:-1]]
    assert cells[3] == "1.00/0.00/0.00/0.00" and cells[9:14] == ["1.00", "1.00", "0.20", "1.00", "10"]
    # one active model, the true class (chance of a random pick of 1 of 5: 0.20), active from t=10
    assert "| 0.00/1.00/0.00/0.00 | 1.00 |" in row_out
    assert "choose − menu_all | in" in sec


# ---------------------------------------------------------------- review regressions (2026-10-28)


def test_revise_records_the_new_pid_and_a_menu_code_revision_counts_as_the_llms(own_pool):
    g = make_game("rps3")
    llm = ScriptLLM([_reply('[{"op": "select", "menu_id": "habit"}]'),
                     _reply('[{"op": "revise", "target": "habit", "new_code": true}]', {"habit": PHASE.replace('params["period"]', "3")})])
    agent = MenuChoiceAgent(own_pool, MENU, llm, decision_at=[10, 20])
    res = run_episode(g, Cycle(g, [0, 0, 1], 0.05), agent, 40, seed=3, episode_key="e")
    rev = [e for e in res.events if e["event"] == "decision"][1]["ops"][0]
    assert rev["ok"] and rev["pid"] == agent._pid["habit"] and rev["ll_gain_vs_other"] is not None
    assert agent._source[agent._pid["habit"]] == "revised" and agent.active["habit"].own_code
    assert "own_code" not in llm.calls[1]["messages"][1]["content"]  # and the prompt shows its code from now on
    agent.close()


def test_a_program_dropped_by_the_posterior_leaves_the_active_set(own_pool):
    late_fail = ("def model(history, params, game):\n    n = game['n_opp']\n"
                 "    if len(history) > 15:\n        return [1.0]\n    return np.ones(n) / n\n")
    llm = ScriptLLM([_reply('[{"op": "create", "id": "frag", "statement": "s", "param_grid": {}}]', {"frag": late_fail})])
    g = make_game("rps3")
    agent = MenuChoiceAgent(own_pool, MENU, llm, decision_at=[10, 30], k_active=1)
    res = run_episode(g, Cycle(g, [0, 1], 0.05), agent, 32, seed=1, episode_key="e")
    assert any(e["event"] == "drop" for e in res.events) and "frag" not in agent.active
    assert [e for e in res.events if e["event"] == "decision"][1]["active"] == []
    agent.close()


def test_create_only_does_not_reserve_the_hidden_menu_ids():
    ops, _ = parse_decision(_reply('[{"op": "create", "id": "wsls", "statement": "s", "param_grid": {}}]', {"wsls": PROG}),
                            "create_only", set(), {}, 4)
    assert ops[0].name == "wsls"


def test_oracle_grid_has_one_point_per_action_and_exclusion_needs_the_seed_library():
    g = make_game("ipd")
    spec = oracle_specs(sample_bot(g, "lose_k_shift", "dev", np.random.default_rng(0)).describe(), g.n_opp)[0]
    assert spec.param_grid["start"] == [0, 1] and {0.1, 0.2} <= set(spec.param_grid["eps"])
    cfg = {"games": ["ipd"], "horizon": 150, "methods": {"x": {"base": "bsi", "menu_exclude_true_family": True}}}
    assert any("needs library seed" in p for p in RM.config_problems(cfg, ["x"]))


def test_menu_record_condition_follows_the_agents_menu():
    import analyze_matrix as AM

    def rec(fam, menu, active):
        ev = {"event": "decision", "t": 10, "menu": menu, "ops": [], "active": [{"name": a} for a in active]}
        return AM.menu_record({"events": [ev], "bot_full": {"family": fam}, "belief": []})

    assert rec("wsls", ["habit", "wsls"], ["wsls"])["cond"] == "in" and rec("wsls", ["habit", "wsls"], ["wsls"])["true_final"]
    assert rec("wsls", ["habit"], ["habit"])["cond"] == "loo" and rec("wsls", ["habit"], [])["true_final"] is None
    assert rec("react2", ["habit"], [])["cond"] == "out" and rec("habit", [], []) is not None


def test_menu_record_reports_active_count_true_share_and_chance():
    import analyze_matrix as AM

    menu = ["habit", "reactive", "wsls", "klevel", "cycle"]
    ev = {"event": "decision", "t": 10, "menu": menu, "ops": [], "active": [{"name": a} for a in ("wsls", "cycle", "own")]}
    r = AM.menu_record({"events": [ev], "bot_full": {"family": "wsls"}, "belief": []})
    assert r["n_active"] == 3 and r["true_share"] == pytest.approx(1 / 3)
    assert r["true_chance"] == pytest.approx(1 - 6 / 10)  # 2 menu models of 5, one is the true class: 1 - C(4,2)/C(5,2)
    assert AM.chance_hit(5, 1, 0) == 0.0 and AM.chance_hit(5, 1, 5) == 1.0


def test_default_schedule_leaves_most_decisions_past_the_window():
    from p2_opponent.menu_agent import DEFAULT_DECISIONS, DEFAULT_K_ACTIVE
    assert sum(t > 40 for t in DEFAULT_DECISIONS) >= 5 and DEFAULT_K_ACTIVE < len(MENU)


def test_arms_never_share_cached_replies(own_pool):
    g = make_game("rps3")
    tags = []
    for name in ("choose_w40", "choose_full"):
        llm = ScriptLLM([])
        agent = MenuChoiceAgent(own_pool, MENU, llm, decision_at=[10], name=name)
        run_episode(g, sample_bot(g, "wsls", "dev", np.random.default_rng(0)), agent, 12, seed=0, episode_key="ep")
        agent.close()
        tags.append(llm.calls[0]["seed_tag"])
    assert tags[0] != tags[1] and all(t.endswith(":ep:t10") for t in tags)


def test_out_gap_subset_keeps_only_out_episodes_with_a_control_gap():
    import analyze_matrix as AM

    def row(m, ep, kl):
        return {"method": m, "episode": ep, "family": ep.split("|")[1], "job_key": f"{m}|{ep}", "kl": kl}

    rows, stats = [], {}
    for ep, gap in (("rps3|react2|p0|s0", 0.0), ("rps3|react2|p1|s0", 0.6)):
        rows += [row("menu_all", ep, 0.1 + gap), row("menu_oracle", ep, 0.1), row("choose", ep, 0.1)]
        stats[f"choose|{ep}"] = {**AM.menu_record({"events": [{"event": "decision", "t": 10, "menu": ["habit"], "ops": [],
                                                               "active": []}], "bot_full": {"family": "react2"}, "belief": []})}
    sec = "\n".join(AM.menu_section(rows, stats))
    assert "| choose | out | 2 |" in sec and "| choose | out_gap | 1 |" in sec
