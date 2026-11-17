"""Offline tests of the LLM client's failure handling, driven by a scripted fake API."""
from types import SimpleNamespace

import httpx2 as httpx  # openai>=3 ships its HTTP layer as httpx2
import openai
import pytest

import bsi.llm as L
from bsi.config import APIConfig, load_api_config, normalize_base_url
from bsi.llm import (LLM, BudgetExceeded, LLMCallFailed, LLMInvalidResponse, LLMUnavailable, RateLimiter,
                     extract_code_blocks, extract_json, split_think)

REQ = httpx.Request("POST", "http://fake/v1/chat/completions")


def resp(text="OK", finish="stop", choices=True):
    msg = SimpleNamespace(content=text, model_extra={})
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason=finish)] if choices else [],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def timeout_err():
    return openai.APITimeoutError(request=REQ)


def status_err(code):
    cls = {400: openai.BadRequestError, 401: openai.AuthenticationError, 429: openai.RateLimitError}.get(code, openai.InternalServerError)
    return cls(f"status {code}", response=httpx.Response(code, request=REQ), body=None)


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.chat = self
        self.completions = self

    def with_options(self, **kw):
        return self

    def create(self, **kw):
        self.calls.append(kw)
        item = self.script.pop(0) if self.script else resp()
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch):
    monkeypatch.setattr(L, "_backoff", lambda attempt: 0.0)


def make(tmp_path, script, **kw):
    kw.setdefault("max_retries", 3)
    kw.setdefault("repair_attempts", 2)
    llm = LLM(api=APIConfig("http://fake/v1", "sk-x"), roles={"r": {"model": "m", "max_tokens": 50}},
              cache_path=tmp_path / "c.sqlite", rpm=10_000, **kw)
    llm._client = FakeClient(script)
    return llm


def json_validator(res):
    try:
        extract_json(res.text)
    except ValueError:
        return "not JSON"
    return None


# ---------------------------------------------------------------- config / helpers


def test_normalize_base_url():
    assert normalize_base_url("https://x.cn") == "https://x.cn/v1"
    assert normalize_base_url("https://x.cn/v1/") == "https://x.cn/v1"
    assert normalize_base_url("https://x.cn/v1/chat/completions/") == "https://x.cn/v1"


def test_load_api_config_from_file(tmp_path, monkeypatch):
    monkeypatch.delenv("BSI_API_KEY", raising=False)
    monkeypatch.delenv("BSI_BASE_URL", raising=False)
    f = tmp_path / "api-key.txt"
    f.write_text("key: sk-test123\naddress: https://api.example.org\napi_key=\"<your-api-key>\"\n")
    cfg = load_api_config(f)
    assert cfg.api_key == "sk-test123" and cfg.base_url == "https://api.example.org/v1"
    assert "sk-test123" not in repr(cfg)


def test_named_api_profile_isolated_from_default_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("BSI_API_KEY", "default-env-key")
    monkeypatch.setenv("BSI_BASE_URL", "https://default.example.org")
    f = tmp_path / "api-key.txt"
    f.write_text("key: default-file-key\naddress: https://default-file.example.org\n"
                 "siliconflow-key: profile-secret\nsiliconflow-url: https://siliconflow.example.org/v1/chat/completions\n"
                 "siliconflow-model: example/model\n")
    cfg = load_api_config(f, profile="siliconflow")
    assert cfg.api_key == "profile-secret"
    assert cfg.base_url == "https://siliconflow.example.org/v1"
    assert load_api_config(f).api_key == "default-env-key"
    with pytest.raises(ValueError, match="incomplete"):
        load_api_config(f, profile="missing")


def test_extractors():
    assert extract_json('blah ```json\n{"a": [1, 2]}\n``` x') == {"a": [1, 2]}
    assert extract_json('prefix {"b": 1} suffix') == {"b": 1}
    assert extract_code_blocks("```python\nx = 1\n```\n```python\ny=2\n```") == ["x = 1", "y=2"]
    assert split_think("<think>hmm</think>answer") == ("answer", "hmm")
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_code_blocks_not_confused_by_json_fence():
    text = 'desc\n```json\n{"param_grid": {"a": [1]}}\n```\n```python\ndef model(h, p, g):\n    return [1]\n```\n'
    assert extract_code_blocks(text, "python") == ["def model(h, p, g):\n    return [1]"]
    assert extract_code_blocks(text, "json") == ['{"param_grid": {"a": [1]}}']


def test_rate_limiter_spaces_requests():
    rl = RateLimiter(rpm=3, period=1.0)
    waits = [rl.reserve() for _ in range(6)]
    assert waits[:3] == [0.0, 0.0, 0.0]
    assert all(w > 0.9 for w in waits[3:])


def test_rate_limiter_reserves_rpm_and_tpm_together(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(L.time, "monotonic", lambda: clock[0])
    rl = RateLimiter(rpm=3, period=60, tpm=100)
    assert [rl.reserve(n) for n in (40, 40, 30, 60, 10, 1)] == [0, 0, 60, 60, 60, 120]
    clock[0] = 220.0
    assert rl.reserve(99) == 0
    assert rl.reserve(1) == 60
    with pytest.raises(ValueError, match="exceeds TPM"):
        rl.reserve(101)


def test_tpm_reservations_cover_transport_retries(tmp_path):
    llm = make(tmp_path, [timeout_err(), resp()], tpm=1_250_000)
    messages = [{"role": "user", "content": "hello"}]
    llm.chat(messages, model="r", use_cache=False)
    costs = [n for _, n in llm.limiter._token_slots]
    assert costs == [llm._token_reservation(messages, 50)] * 2


def test_role_params_applied(tmp_path):
    llm = make(tmp_path, [resp()])
    llm.roles["x"] = L.RoleSpec("mx", {"max_tokens": 7, "extra_body": {"thinking": {"type": "disabled"}}})
    llm.chat([{"role": "user", "content": "hi"}], model="x", use_cache=False)
    call = llm._client.calls[0]
    assert call["model"] == "mx" and call["max_tokens"] == 7 and call["extra_body"]["thinking"]["type"] == "disabled"


# ---------------------------------------------------------------- transport


def test_cache_hit_skips_api(tmp_path):
    llm = make(tmp_path, [resp("hello")])
    msgs = [{"role": "user", "content": "hi"}]
    a = llm.chat(msgs, model="r")
    b = llm.chat(msgs, model="r")
    assert not a.cached and b.cached and b.text == "hello" and len(llm._client.calls) == 1


def test_transient_errors_are_retried(tmp_path):
    llm = make(tmp_path, [timeout_err(), status_err(503), status_err(429), resp("fine")])
    meter = {}
    res = llm.chat([{"role": "user", "content": "x"}], model="r", meter=meter)
    assert res.text == "fine"
    assert llm.usage["transport_retries"] == 3 and meter["transport_retries"] == 3 and meter["calls"] == 1


def test_non_retryable_error_fails_fast(tmp_path):
    llm = make(tmp_path, [status_err(400), resp()])
    with pytest.raises(LLMCallFailed):
        llm.chat([{"role": "user", "content": "x"}], model="r")
    assert len(llm._client.calls) == 1


def test_retries_exhausted(tmp_path):
    llm = make(tmp_path, [timeout_err()] * 10, max_retries=2)
    with pytest.raises(LLMCallFailed):
        llm.chat([{"role": "user", "content": "x"}], model="r")
    assert len(llm._client.calls) == 3 and llm._fail_streak["m"] == 1


# ---------------------------------------------------------------- health / repair


def test_invalid_answer_is_repaired_and_only_valid_cached(tmp_path):
    llm = make(tmp_path, [resp("sure, here you go"), resp('{"a": 1}')])
    msgs = [{"role": "user", "content": "give json"}]
    res = llm.chat(msgs, model="r", validate=json_validator)
    assert res.attempts == 2 and extract_json(res.text) == {"a": 1}
    repair_msgs = llm._client.calls[1]["messages"]
    assert repair_msgs[-2]["role"] == "assistant" and "unusable" in repair_msgs[-1]["content"]
    # the original (attempt-0) key holds nothing, so a re-run asks again rather than replaying garbage
    llm._client.script = [resp('{"a": 2}')]
    again = llm.chat(msgs, model="r", validate=json_validator)
    assert extract_json(again.text) == {"a": 2}


def test_empty_and_truncated_answers_are_unhealthy(tmp_path):
    llm = make(tmp_path, [resp("", finish="length"), resp(choices=False), resp("ok now")])
    res = llm.chat([{"role": "user", "content": "x"}], model="r")
    assert res.text == "ok now" and res.attempts == 3 and llm.usage["invalid_responses"] == 2


def test_invalid_after_all_repairs_raises(tmp_path):
    llm = make(tmp_path, [resp("nope")] * 5, repair_attempts=2)
    with pytest.raises(LLMInvalidResponse) as ei:
        llm.chat([{"role": "user", "content": "x"}], model="r", validate=json_validator)
    assert len(ei.value.problems) == 3


def test_crashing_validator_counts_as_invalid(tmp_path):
    llm = make(tmp_path, [resp("a"), resp("b")], repair_attempts=1)
    with pytest.raises(LLMInvalidResponse):
        llm.chat([{"role": "user", "content": "x"}], model="r", validate=lambda r: 1 / 0)


# ---------------------------------------------------------------- breaker / budget


def test_circuit_breaker_waits_for_recovery(tmp_path):
    llm = make(tmp_path, [timeout_err(), timeout_err(), timeout_err(), resp("pong"), resp("after")],
               max_retries=0, breaker_threshold=2, breaker_cooldown_s=0.01, breaker_max_wait_s=5)
    for _ in range(2):
        with pytest.raises(LLMCallFailed):
            llm.chat([{"role": "user", "content": "x"}], model="r", use_cache=False)
    # breaker open: first probe fails, second succeeds, then the real call runs
    res = llm.chat([{"role": "user", "content": "y"}], model="r", use_cache=False)
    assert res.text == "after" and llm.usage["breaker_trips"] == 1 and llm._fail_streak["m"] == 0


def test_circuit_breaker_gives_up(tmp_path):
    llm = make(tmp_path, [timeout_err()] * 1000, max_retries=0, breaker_threshold=1,
               breaker_cooldown_s=0.01, breaker_max_wait_s=0.1)
    with pytest.raises(LLMCallFailed):
        llm.chat([{"role": "user", "content": "x"}], model="r", use_cache=False)
    with pytest.raises(LLMUnavailable):
        llm.chat([{"role": "user", "content": "x"}], model="r", use_cache=False)


def test_budget_enforced(tmp_path):
    llm = make(tmp_path, [resp()], budget_tokens=10)
    llm.usage["prompt_tokens"] = 10
    with pytest.raises(BudgetExceeded):
        llm.chat([{"role": "user", "content": "x"}], model="r")


def test_health_check_reports_without_raising(tmp_path):
    llm = make(tmp_path, [resp("OK"), status_err(401)])
    out = llm.health_check(["r", "r"])
    assert out[0]["ok"] and not out[1]["ok"] and "AuthenticationError" in out[1]["error"]


# ---------------------------------------------------------------- proposer parsing


def test_parse_proposals_roundtrip():
    from p2_opponent.propose import parse_proposals, proposals_validator

    text = """Some preamble.
### Hypothesis: beats the agent's last move
```json
{"param_grid": {"p": [0.6, 0.8]}}
```
```python
def model(history, params, game):
    return np.ones(game["n_opp"]) / game["n_opp"]
```
### Hypothesis: cycles
```json
{"param_grid": {"period": [2, 3]}}
```
```python
def model(history, params, game):
    return np.ones(3) / 3
```
"""
    specs = parse_proposals(text)
    assert [s.param_grid for s in specs] == [{"p": [0.6, 0.8]}, {"period": [2, 3]}]
    assert all(s.src.startswith("def model") for s in specs)
    assert specs[0].statement == "beats the agent's last move"
    check = proposals_validator()
    mk = lambda t: L.LLMResult(t, "", "m", 0, 0, "stop", 0.0)
    assert check(mk(text)) is None
    assert "no '### Hypothesis" in check(mk("just prose"))
    assert "sandbox" in check(mk("### Hypothesis: x\n```python\nimport os\ndef model(h,p,g):\n    return 1\n```"))


def test_role_max_retries_override(tmp_path):
    llm = make(tmp_path, [timeout_err()] * 10, max_retries=4)
    llm.roles["slow"] = L.RoleSpec("m", {"max_retries": 1})
    with pytest.raises(LLMCallFailed):
        llm.chat([{"role": "user", "content": "x"}], model="slow", use_cache=False)
    assert len(llm._client.calls) == 2 and "max_retries" not in llm._client.calls[0]


def test_degraded_repair_after_length_exhaustion(tmp_path):
    llm = make(tmp_path, [resp("", finish="length"), resp("answer")])
    llm.roles["think"] = L.RoleSpec("m", {"max_tokens": 8000, "repair_overrides": {
        "extra_body": {"thinking": {"type": "disabled"}}, "max_tokens": 500}})
    meter = {}
    res = llm.chat([{"role": "user", "content": "x"}], model="think", meter=meter)
    first, second = llm._client.calls
    assert res.text == "answer" and meter["degraded_repairs"] == 1
    assert first["max_tokens"] == 8000 and "extra_body" not in first and "repair_overrides" not in first
    assert second["max_tokens"] == 500 and second["extra_body"]["thinking"]["type"] == "disabled"
    assert res.degraded is True


def test_format_errors_keep_original_params(tmp_path):
    llm = make(tmp_path, [resp("not json", finish="stop"), resp('{"a": 1}')])
    llm.roles["think"] = L.RoleSpec("m", {"max_tokens": 8000, "repair_overrides": {"max_tokens": 500}})
    res = llm.chat([{"role": "user", "content": "x"}], model="think", validate=json_validator)
    assert [c["max_tokens"] for c in llm._client.calls] == [8000, 8000] and llm.usage["degraded_repairs"] == 0
    assert res.degraded is False and res.attempts == 2


def test_meter_counts_cached_calls_as_logical_cost(tmp_path):
    llm = make(tmp_path, [resp("hello")])
    msgs = [{"role": "user", "content": "hi"}]
    llm.chat(msgs, model="r")
    meter = {}
    llm.chat(msgs, model="r", meter=meter)
    assert meter["calls"] == 1 and meter["cached_calls"] == 1 and meter["prompt_tokens"] == 10
    assert llm.usage["calls"] == 1 and llm.usage["cached_calls"] == 1


def test_opt_in_validation_repair_uses_full_answer_and_observer(tmp_path):
    bad = 'not JSON ' + 'x' * 5000
    llm = make(tmp_path, [resp(bad), resp('{"actions": []}')])
    llm.roles['structured'] = L.RoleSpec('m', {
        'max_tokens': 32000, 'repair_context_chars': None,
        'validation_repair_overrides': {'max_tokens': 8000, 'extra_body': {'thinking': {'type': 'disabled'}}}})
    seen = []
    r = llm.chat([{'role': 'user', 'content': 'Return JSON'}], model='structured', validate=json_validator,
                 response_format={'type': 'json_object'}, on_invalid=lambda res, problem, n: seen.append((res.text, problem, n)))
    first, second = llm._client.calls
    assert first['max_tokens'] == 32000 and 'extra_body' not in first
    assert second['max_tokens'] == 8000 and second['extra_body']['thinking']['type'] == 'disabled'
    assert second['messages'][-2]['content'] == bad
    assert second['response_format'] == {'type': 'json_object'}
    assert seen == [(bad, 'not JSON', 1)]
    assert r.repair_mode == 'validation' and r.attempts == 2
    assert llm.usage['validation_repairs'] == 1
    assert not {'repair_context_chars', 'validation_repair_overrides', 'on_invalid'} & first.keys()


def test_opt_in_validation_repair_stays_bounded(tmp_path):
    llm = make(tmp_path, [resp('bad')] * 5)
    llm.roles['r'] = L.RoleSpec('m', {'validation_repair_overrides': {'max_tokens': 8000}})
    failures = []
    with pytest.raises(LLMInvalidResponse):
        llm.chat([{'role': 'user', 'content': 'JSON'}], model='r', validate=json_validator,
                 on_invalid=lambda res, problem, n: failures.append(n))
    assert len(llm._client.calls) == 3 and failures == [1, 2, 3]
