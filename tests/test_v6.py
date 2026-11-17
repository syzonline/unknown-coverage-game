"""v6: exploration, residual diagnostics, program archive, archive ops in the prompt/parser, variation calls."""
import hashlib
import json
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import yaml

from bsi.llm import LLMResult
from bsi.sandbox import SandboxPool

from p2_opponent.agents import PosteriorBRAgent, ProgramSpec
from p2_opponent.archive import ProgramArchive
from p2_opponent.bots import Cycle, sample_bot
from p2_opponent.diagnostics import PARTITIONS, conditional_tables, residual_cells
from p2_opponent.explore import ExplorePolicy
from p2_opponent.games import make_game
from p2_opponent.menu_agent import MenuChoiceAgent
from p2_opponent.menu_prompt import (ActiveModel, DecisionError, ModelFit, PromptState, V6Prompt, V6Rules,
                                     archive_text, build_messages, parse_decision, parse_variation)
from p2_opponent.runner import run_episode
from p2_opponent.seed_programs import SEED_LIBRARY

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_matrix as RM  # noqa: E402

MENU = list(SEED_LIBRARY)
MENU_IDS = {s.id for s in MENU}
UNIFORM = "def model(history, params, game):\n    n = game['n_opp']\n    return np.ones(n) / n\n"
COPY2 = '''def model(history, params, game):
    n = game["n_opp"]
    eps = float(params["eps"])
    if len(history) < 2:
        return np.ones(n) / n
    p = np.full(n, eps / n)
    p[history[-2][0]] += 1 - eps
    return p
'''
BROKEN = "def model(history, params, game):\n    return [1.0]\n"


@pytest.fixture
def own_pool():
    with SandboxPool(n_workers=2) as p:
        yield p


def _reply(key: str, items: str, models: dict[str, str] | None = None) -> str:
    out = f'```json\n{{"{key}": {items}, "rationale": "r"}}\n```\n'
    for name, code in (models or {}).items():
        out += f"### Model: {name}\n```python\n{code}```\n"
    return out


class TagLLM:
    """Replies chosen by seed tag (parallel variation calls arrive in any order); records every request."""

    def __init__(self, replies: dict[str, str], default: str):
        self.replies, self.default, self.calls = replies, default, []
        self.lock = threading.Lock()

    def chat(self, messages, model="m", temperature=0.7, seed_tag="", validate=None, meter=None, **kw):
        text = next((v for k, v in self.replies.items() if seed_tag.endswith(k)), self.default)
        with self.lock:
            self.calls.append({"messages": messages, "seed_tag": seed_tag})
        res = LLMResult(text=text, reasoning="x", model=model, prompt_tokens=10, completion_tokens=5,
                        finish_reason="stop", latency_s=0.0)
        if validate is not None:
            assert validate(res) is None, validate(res)
        if meter is not None:
            meter["calls"] = meter.get("calls", 0) + 1
        return res


def _copy2_history(T=90, seed=0):
    """The opponent plays the agent's action of two rounds before (else uniform); the agent plays at random."""
    rng = np.random.default_rng(seed)
    h = []
    for t in range(T):
        a = int(rng.integers(3))
        o = h[-2][0] if t >= 2 and rng.random() < 0.9 else int(rng.integers(3))
        h.append((a, o))
    return tuple(h)


# ---------------------------------------------------------------- exploration


def test_exploration_rounds_are_shared_nested_and_inside_windows():
    lo, hi = ExplorePolicy(0.15), ExplorePolicy(0.30)
    r_lo = {t for t in range(150) if lo.explores("ep", t)}
    r_hi = {t for t in range(150) if hi.explores("ep", t)}
    assert r_lo <= r_hi and 0 < len(r_lo) < len(r_hi)
    assert all(hi.in_window(t) and t < 130 for t in r_hi)
    assert r_hi == {t for t in range(150) if ExplorePolicy(0.30).explores("ep", t)}  # a pure function
    assert {t for t in range(150) if hi.explores("other", t)} != r_hi
    assert ExplorePolicy.from_opts({"rate": 0.0}) is None and ExplorePolicy.from_opts(None) is None


def test_exploration_picks_the_action_that_separates_the_models():
    pol = ExplorePolicy(1.0)

    def models(ctx):  # model A predicts the agent's last action, model B the opponent's: only a != o separates them
        n = 3
        P = np.zeros((2, len(ctx), n))
        for c, h in enumerate(ctx):
            a, o = h[-1]
            P[0, c] = np.eye(n)[a] * 0.9 + 0.1 / n
            P[1, c] = np.eye(n)[o] * 0.9 + 0.1 / n
        return np.array([0.5, 0.5]), P

    a, eig = pol.choose("ep", 10, ((0, 0),), np.array([0.98, 0.01, 0.01]), 3, models)
    assert a in (1, 2) and eig > 0.1  # the opponent will very likely play 0, so playing 0 would reveal nothing
    a2, eig2 = pol.choose("ep", 10, ((0, 0),), np.array([0.98, 0.01, 0.01]), 3, lambda c: (np.array([1.0]), np.ones((1, len(c), 3)) / 3))
    assert eig2 == 0.0 and 0 <= a2 < 3  # one model: nothing to separate, uniform fallback


def test_posterior_agent_without_exploration_is_unchanged_and_with_it_explores_in_windows(own_pool):
    g = make_game("rps3")
    bot = Cycle(g, [0, 1, 2], 0.1)
    runs = {}
    for name, ex in (("none", None), ("zero", {"rate": 0.0}), ("e30", {"rate": 0.3})):
        agent = PosteriorBRAgent(own_pool, MENU, name=name, explore=ex)
        runs[name] = run_episode(g, bot, agent, 80, seed=1, episode_key="ep")
        agent.close()
    assert runs["none"].actions == runs["zero"].actions
    ev = [e for e in runs["e30"].events if e["event"] == "explore"]
    pol = ExplorePolicy(0.3)
    assert ev and {e["t"] for e in ev} == {t for t in range(80) if pol.explores("ep", t)}
    assert all(runs["e30"].actions[e["t"]][0] == e["a"] for e in ev)


# ---------------------------------------------------------------- residual diagnostics


def test_residuals_and_tables_point_at_the_hidden_dependence():
    g = make_game("rps3")
    h = _copy2_history()
    uniform = np.full((len(h), 3), 1 / 3)
    cells = residual_cells(g, h, uniform, top=3)
    assert cells and cells[0].partition in ("agent_last2", "last_pair", "agent_last") and cells[0].excess > 5
    assert any(c.partition == "agent_last2" for c in residual_cells(g, h, uniform, top=8))
    tables = conditional_tables(g, h, top=2)
    assert tables[0].startswith("agent_last2")
    # the true model leaves (almost) no excess
    true = np.array([np.full(3, 1 / 3) if t < 2 else np.eye(3)[h[t - 2][0]] * 0.9 + 0.1 / 3 for t in range(len(h))])
    assert residual_cells(g, h, true, top=1)[0].excess < cells[0].excess / 3
    assert set(PARTITIONS) >= {"agent_last2", "result_streak", "time_block", "t_mod_5"}


# ---------------------------------------------------------------- archive


def test_archive_scores_evicts_protects_and_tracks_lineage(own_pool):
    g = make_game("rps3")
    h = _copy2_history()
    ar = ProgramArchive(own_pool, g.info(), capacity=2)
    assert ar.add(ProgramSpec("uni", UNIFORM), h, "create")[0]
    ok, reason, _ = ar.add(ProgramSpec("bad", BROKEN), h, "create")
    assert not ok and ar.failed[-1]["name"] == "bad"
    assert ar.add(ProgramSpec("copy2", COPY2, {"eps": [0.1, 0.3]}), h, "mutate", parent="uni")[0]
    e = ar.entries["copy2"]
    assert e.parent == "uni" and e.generation == 1 and 0 < e.similarity < 1 and e.h.params == {"eps": 0.1}
    assert ar.best().name == "copy2" and ar.entries["copy2"].mean_ll() > ar.entries["uni"].mean_ll()
    # over capacity: the worst unprotected entry goes
    ar.add(ProgramSpec("uni2", UNIFORM.replace("np.ones(n) / n", "np.ones(n) / float(n)")), h, "create", protect={"uni"})
    assert set(ar.entries) == {"uni", "uni2"} or set(ar.entries) == {"copy2", "uni"}
    assert "uni" in ar.entries
    gone = ar.rescore(h[:50])
    assert gone == [] and all(e.n_obs == 50 for e in ar.entries.values())
    ar.close()


# ---------------------------------------------------------------- prompt and parser


def _state():
    return PromptState(t=40, horizon=150, decision_index=1, n_decisions=10, fits=[], reference=None,
                       mixture_mean_ll_recent=-1.0)


def test_v5_prompts_are_byte_identical_without_v6():
    """sha256 of v5 prompts computed before the v6 changes (scratchpad v5_fingerprint.py, 2026-10-29)."""
    g = make_game("hideseek5")
    h = tuple(((t * 3) % 5, (t * 7 + 1) % 5) for t in range(60))
    active = {"wsls": ActiveModel("wsls", "menu", SEED_LIBRARY[2]), "mine": ActiveModel("mine", "created", SEED_LIBRARY[0], 1, True)}
    fits = [ModelFit("wsls", "menu", {"eps": 0.1}, 0.6, -1.2, -1.3), ModelFit("mine", "created", {"w": 5}, 0.3, -1.4, -1.5)]
    st = PromptState(t=60, horizon=150, decision_index=2, n_decisions=7, fits=fits,
                     reference=ModelFit("reference", "reference", {}, 0.1, -1.5, -1.55), mixture_mean_ll_recent=-1.25,
                     log=[{"t": 10, "ops": [{"op": "select", "name": "wsls", "ok": True, "reason": ""}]},
                          {"t": 30, "ops": [{"op": "create", "name": "x", "ok": False, "reason": "boom"}]}])
    expect = {"select_create|window|False": "29421e8cc325b03a", "select_create|full|True": "25e54c3aef8c457c",
              "create_only|full|False": "47fd42129d5c6c8e", "menu_all_create|window|True": "945728370f8df3d3"}
    for key, sha in expect.items():
        mode, hm, flip = key.split("|")
        m = build_messages(g, SEED_LIBRARY, active, st, h, mode, hm, 40, 3, flip == "True")
        assert hashlib.sha256(json.dumps(m).encode()).hexdigest()[:16] == sha, key


def test_v6_prompt_shows_archive_code_only_with_heredity(own_pool):
    g = make_game("rps3")
    h = _copy2_history(40)
    ar = ProgramArchive(own_pool, g.info())
    ar.add(ProgramSpec("copy2", COPY2, {"eps": [0.1]}, "copies the agent two back"), h, "create")
    on = V6Prompt(True, archive_text(ar.ranked(), set(), True), failed="- x", residuals="R", tables="T")
    off = V6Prompt(False, archive_text(ar.ranked(), set(), False), failed="- x", residuals="R", tables="T")
    a = build_messages(g, MENU, {}, _state(), h, "select_create", "full", 40, 3, False, on)[1]["content"]
    b = build_messages(g, MENU, {}, _state(), h, "select_create", "full", 40, 3, False, off)[1]["content"]
    assert "p[history[-2][0]]" in a and "p[history[-2][0]]" not in b
    assert "FAILED ATTEMPTS" in a and "FAILED ATTEMPTS" not in b and "RESIDUAL DIAGNOSTICS" in a and "\nR\n" in a
    assert '"op": "mutate"' in a and '"op": "mutate"' not in b and '"op": "activate"' in b
    ar.close()


def test_parse_decision_archive_ops_and_fresh_limit():
    rules = V6Rules(frozenset({"arc1"}), heredity=True, max_fresh=1)
    acts = ('[{"op": "activate", "archive_id": "arc1"}, {"op": "mutate", "parent": "arc1", "id": "kid", "statement": "s",'
            ' "param_grid": {"eps": [0.1]}}]')
    ops, _ = parse_decision(_reply("actions", acts, {"kid": COPY2}), "select_create", MENU_IDS, {}, 3, rules)
    assert [(o.op, o.name, o.parent) for o in ops] == [("activate", "arc1", None), ("mutate", "kid", "arc1")]
    bad = {
        '[{"op": "activate", "archive_id": "nope"}]': "unknown archive_id",
        '[{"op": "mutate", "parent": "nope", "id": "k", "param_grid": {}}]': "not in the archive",
        '[{"op": "create", "id": "arc1", "param_grid": {}}]': "already used",
        '[{"op": "create", "id": "a1", "param_grid": {}}, {"op": "create", "id": "a2", "param_grid": {}}]': "at most 1",
    }
    for acts, msg in bad.items():
        with pytest.raises(DecisionError, match=msg):
            parse_decision(_reply("actions", acts, {"k": COPY2, "a1": COPY2, "a2": COPY2, "arc1": COPY2}), "select_create",
                           MENU_IDS, {}, 3, rules)
    # without heredity: activate yes, mutate no; without rules (v5): neither
    mutate = '[{"op": "mutate", "parent": "arc1", "id": "k", "param_grid": {}}]'
    with pytest.raises(DecisionError, match="op must be one of"):
        parse_decision(_reply("actions", mutate, {"k": COPY2}), "select_create", MENU_IDS, {}, 3,
                       V6Rules(frozenset({"arc1"}), False))
    with pytest.raises(DecisionError, match="op must be one of"):
        parse_decision(_reply("actions", '[{"op": "activate", "archive_id": "arc1"}]'), "select_create", MENU_IDS, {}, 3)


def test_parse_variation():
    rules = V6Rules(frozenset({"arc1"}), heredity=True)
    progs = ('[{"op": "mutate", "parent": "arc1", "id": "m1", "statement": "s", "param_grid": {"eps": [0.1]}},'
             ' {"op": "create", "id": "c1", "param_grid": {}}]')
    ops, _ = parse_variation(_reply("programs", progs, {"m1": COPY2, "c1": UNIFORM}), {"habit"}, rules, 2)
    assert [(o.op, o.name) for o in ops] == [("mutate", "m1"), ("create", "c1")]
    with pytest.raises(DecisionError, match="at most 1"):
        parse_variation(_reply("programs", progs, {"m1": COPY2, "c1": UNIFORM}), set(), rules, 1)
    with pytest.raises(DecisionError, match="op must be one of"):
        parse_variation(_reply("programs", progs, {"m1": COPY2, "c1": UNIFORM}), set(), V6Rules(frozenset({"arc1"}), False), 2)


# ---------------------------------------------------------------- agent


def test_v6_agent_variation_archive_mutate_activate_and_explore(own_pool):
    g = make_game("rps3")
    replies = {
        # t=10: variation calls write candidates, the decision activates one of them
        "t10:v0": _reply("programs", '[{"op": "create", "id": "uni", "statement": "uniform", "param_grid": {}}]', {"uni": UNIFORM}),
        "t10:v1": _reply("programs", '[{"op": "create", "id": "uni", "statement": "same name", "param_grid": {}}]', {"uni": UNIFORM}),
        "t10": _reply("actions", '[{"op": "select", "menu_id": "habit"}, {"op": "activate", "archive_id": "uni"}]'),
        # t=24: a variation call mutates the best archive program; the decision mutates too, retires the uniform one
        "t24:v0": _reply("programs", '[{"op": "mutate", "parent": "uni", "id": "c2", "statement": "agent two back", '
                                     '"param_grid": {"eps": [0.1, 0.3]}}]', {"c2": COPY2}),
        "t24": _reply("actions", '[{"op": "retire", "target": "uni"}, {"op": "mutate", "parent": "uni", "id": "bad", '
                                 '"statement": "s", "param_grid": {}}]', {"bad": BROKEN}),
        "t24:fix": _reply("actions", '[{"op": "keep"}]'),
    }
    llm = TagLLM(replies, default=_reply("programs", "[]"))
    agent = MenuChoiceAgent(own_pool, MENU, llm, decision_at=[10, 24], archive=True, heredity=True, residuals=True,
                            n_variation=2, explore={"rate": 0.3}, name="full")
    bot = sample_bot(g, "react2", "dev", np.random.default_rng(0))
    res = run_episode(g, bot, agent, 40, seed=0, episode_key="rps3|react2|p0|s0")
    ev = res.events
    adds = [e for e in ev if e["event"] == "archive_add"]
    names = [e["name"] for e in adds if e["ok"]]
    assert names[:2] == ["uni", "uni_2"] and "c2" in names  # parallel calls with one name: the second is renamed
    c2 = next(e for e in adds if e["name"] == "c2")
    assert c2["parent"] == "uni" and c2["gen"] == 1 and c2["src"].startswith("def model") and c2["call"] == "variation:0"
    dec = [e for e in ev if e["event"] == "decision"]
    assert [(o["op"], o["name"], o["ok"]) for o in dec[0]["ops"]] == [("select", "habit", True), ("activate", "uni", True)]
    assert [(o["op"], o["ok"]) for o in dec[1]["ops"]][:2] == [("retire", True), ("mutate", False)]
    assert dec[1]["repair"] and "archive" in dec[1] and dec[1]["archive"][0]["name"]
    assert {e["k"] for e in ev if e["event"] == "variation" and e["t"] == 10} == {0, 1}
    assert any(e["event"] == "explore" for e in ev)
    # the t=24 decision prompt shows the archive with code (heredity), diagnostics and the mutate op
    p = next(c for c in llm.calls if c["seed_tag"].endswith(":t24"))["messages"][1]["content"]
    assert "ARCHIVE" in p and "p[history[-2][0]]" in p and "RESIDUAL DIAGNOSTICS" in p and "CONTEXT TABLES" in p
    assert agent.belief_log[-1]["by_src"].get("archived", 0) + agent.belief_log[-1]["by_src"].get("menu", 0) > 0
    assert res.llm_usage["calls"] == len(llm.calls)
    agent.close()


def test_v6_options_are_validated_and_estimated(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs" / "intern_full.yaml").read_text())
    assert RM.config_problems(cfg, list(cfg["methods"])) == []
    full = RM.method_opts(cfg, "v6_full_e30")
    assert full["heredity"] and full["n_variation"] == 2 and full["residuals"]
    bad = {"games": ["rps3"], "horizon": 150, "methods": {
        "a": {"base": "menu_choice", "heredity": True},
        "b": {"base": "menu_choice", "archive": True, "n_variation": -1},
        "c": {"base": "freq_br", "explore": {"rate": 0.2}},
        "d": {"base": "b2_static", "explore": {"rate": 2}}}}
    probs = RM.config_problems(bad, list(bad["methods"]))
    assert any(p.startswith("a: heredity") for p in probs) and any(p.startswith("b: n_variation") for p in probs)
    assert any(p.startswith("c: explore") for p in probs) and any(p.startswith("d: explore") for p in probs)
    ctl = yaml.safe_load((ROOT / "configs" / "controls.yaml").read_text())
    assert RM.config_problems(ctl, list(ctl["methods"])) == []


def test_report_uses_matched_controls_and_has_the_evolution_section(own_pool, tmp_path):
    import analyze_matrix as AM
    from bsi.store import TraceWriter, dumps

    g = make_game("rps3")
    tw = TraceWriter(tmp_path / "traces-1.jsonl.gz")
    reply = _reply("programs", '[{"op": "create", "id": "uni", "statement": "u", "param_grid": {}}]', {"uni": UNIFORM})
    with open(tmp_path / "summaries.jsonl", "w") as f:
        for p in range(2):
            bot = sample_bot(g, "react2", "dev", np.random.default_rng(p))
            ep = f"rps3|react2|p{p}|s0"
            agents = {
                "v6_full_e30": MenuChoiceAgent(own_pool, MENU, TagLLM({"t10:v0": reply}, _reply("actions", '[{"op": "keep"}]')),
                                               decision_at=[10], archive=True, heredity=True, n_variation=1,
                                               explore={"rate": 0.3}, name="v6_full_e30"),
                "menu_all_e30": PosteriorBRAgent(own_pool, MENU, name="menu_all_e30", explore={"rate": 0.3}),
                "menu_oracle_e30": PosteriorBRAgent(own_pool, MENU, name="menu_oracle_e30", explore={"rate": 0.3}),
            }
            for m, agent in agents.items():
                res = run_episode(g, bot, agent, 30, seed=p, episode_key=ep)
                agent.close()
                s = {**res.summary(), "method": m, "family": "react2", "param": p, "env_seed": 0,
                     "job_key": f"{m}|{ep}", "episode": ep, "status": "ok"}
                f.write(dumps(s) + "\n")
                tw.write({**res.to_record(), "job_key": s["job_key"], "method": m})
    tw.close()
    text = AM.analyze(tmp_path)
    assert "v6_full_e30 − menu_all_e30 | out" in text and "(controls e30)" in text
    evo = text.split("## Evolution (v6)")[1]
    row = next(l for l in evo.splitlines() if l.startswith("| v6_full_e30 | out |"))
    cells = [c.strip() for c in row.split("|")[1:-1]]
    assert cells[3].startswith("1.0") and float(cells[10]) > 0  # one archive insert per episode; explored
