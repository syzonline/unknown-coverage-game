"""LLM baselines and proposer evidence, driven by the scripted fake API (no network)."""
import json

import numpy as np
import pytest

import bsi.llm as L
from bsi.config import APIConfig
from p2_opponent import bots as B
from p2_opponent.agents import PosteriorBRAgent
from p2_opponent.games import ipd, rps
from p2_opponent.llm_agents import (DirectLLMAgent, OracleTypeLLMAgent, PosteriorLLMDecideAgent, VerbalBeliefAgent,
                                    parse_decision)
from p2_opponent.propose import build_messages, evidence_tables
from p2_opponent.runner import adaptation_latency, run_episode
from p2_opponent.seed_programs import MINIMAL_LIBRARY, SEED_LIBRARY
from tests.test_llm_offline import FakeClient, resp, timeout_err


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch):
    monkeypatch.setattr(L, "_backoff", lambda attempt: 0.0)


class Responder(FakeClient):
    """Fake API answering every request with fn(messages)."""

    def __init__(self, fn):
        super().__init__([])
        self.fn = fn

    def create(self, **kw):
        self.calls.append(kw)
        out = self.fn(kw["messages"])
        if isinstance(out, BaseException):
            raise out
        return resp(out)


def fake_llm(tmp_path, fn, **kw):
    kw.setdefault("max_retries", 1)
    llm = L.LLM(api=APIConfig("http://fake/v1", "sk-x"), roles={"decider": "m", "proposer": "m"},
                cache_path=tmp_path / "c.sqlite", rpm=100_000, repair_attempts=1, **kw)
    llm._client = Responder(fn)
    return llm


def test_parse_decision_accepts_names_and_indices():
    g = rps(3)
    a, p, _ = parse_decision('{"action": "Paper", "prediction": {"rock": 2, "paper": 1, "scissors": 1}}', g, True)
    assert a == 1 and np.allclose(p, [0.5, 0.25, 0.25])
    assert parse_decision('{"action": 2}', g, False)[0] == 2
    with pytest.raises(ValueError):
        parse_decision('{"action": "lizard"}', g, False)
    with pytest.raises(ValueError):
        parse_decision('{"action": "rock", "prediction": {"rock": -1}}', g, True)


def test_b0_plays_llm_action_and_meters_usage(tmp_path):
    llm = fake_llm(tmp_path, lambda m: '{"action": "paper"}')
    g = rps(3)
    res = run_episode(g, B.Fixed(g, [1, 0, 0]), DirectLLMAgent(llm), horizon=12, seed=0)
    assert all(a == 1 for a, _ in res.actions)
    s = res.summary()
    assert s["llm_calls"] == 12 and s["fallback_rate"] == 0 and s["payoff_per_round"] == 1.0
    assert np.isnan(s["kl"])  # B0 makes no prediction


def test_b0_falls_back_on_persistent_garbage_and_on_dead_api(tmp_path):
    g = rps(3)
    llm = fake_llm(tmp_path, lambda m: "I think rock is good")  # never valid JSON
    res = run_episode(g, B.Fixed(g, [1, 0, 0]), DirectLLMAgent(llm), horizon=6, seed=0)
    assert res.summary()["fallback_rate"] == 1.0
    assert sum(e["event"] == "fallback" for e in res.events) == 6
    llm2 = fake_llm(tmp_path / "b", lambda m: timeout_err(), breaker_threshold=1000)
    res2 = run_episode(g, B.Fixed(g, [1, 0, 0]), DirectLLMAgent(llm2), horizon=3, seed=0)
    assert res2.summary()["fallback_rate"] == 1.0


def test_fallback_is_reproducible(tmp_path):
    g = rps(3)
    acts = []
    for k in range(2):
        llm = fake_llm(tmp_path / str(k), lambda m: "garbage")
        acts.append(run_episode(g, B.Fixed(g, [1, 0, 0]), DirectLLMAgent(llm), horizon=8, seed=3).actions)
    assert acts[0] == acts[1]


def test_llm_unavailable_propagates(tmp_path):
    g = rps(3)
    llm = fake_llm(tmp_path, lambda m: timeout_err(), max_retries=0, breaker_threshold=1,
                   breaker_cooldown_s=0.01, breaker_max_wait_s=0.05)
    with pytest.raises(L.LLMUnavailable):
        run_episode(g, B.Fixed(g, [1, 0, 0]), DirectLLMAgent(llm), horizon=5, seed=0)


def test_repair_path_recovers(tmp_path):
    g = rps(3)
    n = {"k": 0}

    def fn(m):
        n["k"] += 1
        return "oops" if n["k"] % 2 == 1 else '{"action": "rock"}'

    res = run_episode(g, B.Fixed(g, [0, 0, 1]), DirectLLMAgent(fake_llm(tmp_path, fn)), horizon=4, seed=0)
    s = res.summary()
    assert s["fallback_rate"] == 0 and s["llm_invalid"] == 4
    assert sum(e["event"] == "repaired" for e in res.events) == 4


def test_b1_tracks_and_scores_hypotheses(tmp_path):
    g = rps(3)

    def fn(m):
        user = m[-1]["content"] if m[-1]["role"] == "user" else m[1]["content"]
        out = {"hypothesis_predictions": {}, "prediction": {"rock": 0.8, "paper": 0.1, "scissors": 0.1}, "action": "paper"}
        if "new_hypotheses" in user:
            out["new_hypotheses"] = ["always rock", "always scissors"]
            out["hypothesis_predictions"] = {"new_1": "rock", "new_2": "scissors"}
        else:
            out["hypothesis_predictions"] = {"H1": "rock", "H2": "scissors"}
        return json.dumps(out)

    agent = VerbalBeliefAgent(fake_llm(tmp_path, fn), refine_every=1000, refine_below=0.0)
    res = run_episode(g, B.Fixed(g, [1, 0, 0]), agent, horizon=10, seed=0)
    by_id = {h["id"]: h for h in agent.hyps}
    assert by_id["H1"]["hits"] == by_id["H1"]["n"] == 10 and by_id["H2"]["hits"] == 0
    assert agent.hyps[0]["id"] == "H1"
    assert np.isfinite(res.summary()["kl"])


def test_b4_prompt_contains_exact_rule(tmp_path):
    g = ipd()
    seen = []

    def fn(m):
        seen.append(m[1]["content"])
        return '{"prediction": {"cooperate": 1, "defect": 0}, "action": "defect"}'

    bot = B.Reactive(g, [0, 1], 0.1, "copy")
    run_episode(g, bot, OracleTypeLLMAgent(fake_llm(tmp_path, fn), bot), horizon=3, seed=0)
    assert "family=reactive" in seen[0] and "'table': [0, 1]" in seen[0] and "eps" in seen[0]


def test_b3_reports_posterior_prediction_but_llm_action(pool, tmp_path):
    g = rps(3)
    llm = fake_llm(tmp_path, lambda m: '{"action": "scissors"}')
    bot = B.Fixed(g, [0.6, 0.3, 0.1])
    b3 = PosteriorLLMDecideAgent(pool, SEED_LIBRARY, llm=llm)
    res3 = run_episode(g, bot, b3, horizon=30, seed=1)
    b3.close()
    ref = PosteriorBRAgent(pool, SEED_LIBRARY)
    resr = run_episode(g, bot, ref, horizon=30, seed=1)
    ref.close()
    assert all(a == 2 for a, _ in res3.actions)
    # same environment seed and same posterior inputs on round 0 -> identical first prediction
    assert np.allclose(res3.preds[0], resr.preds[0])


def test_evidence_tables_expose_catch_all_structure(pool):
    g = rps(3)
    bot = B.Reactive(g, [1, 2, 2], 0.05, "random_map")
    agent = PosteriorBRAgent(pool, MINIMAL_LIBRARY)
    res = run_episode(g, bot, agent, horizon=60, seed=0)
    hist = tuple(res.actions)
    text = evidence_tables(agent.posterior, hist, g)
    assert "Catch-all model's learned table" in text and "P(opp next | agent_last)" in text
    # the agent_last view must reveal the deterministic mapping rock->paper
    line = next(l for l in text.splitlines() if "agent_last=rock" in l)
    paper_p = float(line.split("paper=")[1].split()[0])
    assert paper_p > 0.8
    msgs = build_messages(agent.posterior, hist, g)
    assert "EVIDENCE" in msgs[1]["content"]
    agent.close()


def test_adaptation_latency():
    kls = np.r_[np.full(50, 0.02), np.full(15, 0.8), np.full(35, 0.03)]
    assert adaptation_latency(kls, 50) == (15, False, True)
    assert adaptation_latency(np.r_[np.full(50, 0.02), np.full(50, 0.8)], 50) == (50, True, True)
    # never tracked before the switch: not a fast adaptation, just not applicable
    lat, censored, applicable = adaptation_latency(np.r_[np.full(50, 1.4), np.full(50, 1.4)], 50)
    assert not applicable and np.isnan(lat)
