"""LLM client: OpenAI-compatible API with defensive call handling.

Every call goes through four layers:
1. transport retries: timeouts, connection errors, 408/409/425/429/5xx and malformed bodies, with
   exponential backoff and jitter;
2. response health checks: an empty answer always counts as a failure, and callers can pass a
   `validate` function for format checks (for example "JSON with a legal action");
3. repair: when a response is unusable, the model is told what was wrong and asked again, a bounded
   number of times; only responses that pass validation are cached;
4. a per-model circuit breaker: after repeated hard failures, callers pause and probe the endpoint
   until it recovers, or raise `LLMUnavailable` once a maximum wait is exceeded.

Every outcome is bounded: a call returns a validated result or raises a typed `LLMError`, which
callers turn into a documented fallback. Roles map to models plus default request parameters, for
example turning reasoning off for per-round decisions.
"""
from __future__ import annotations

import asyncio
import bisect
import hashlib
import json
import random
import re
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import openai

from .config import PROJECT_ROOT, APIConfig, load_api_config, load_models_config

RETRY_STATUS = {408, 409, 425, 429}


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (openai.APIConnectionError, openai.APIResponseValidationError, json.JSONDecodeError)):
        return True  # APITimeoutError is a subclass of APIConnectionError
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in RETRY_STATUS or exc.status_code >= 500
    return False


class LLMError(RuntimeError):
    """Base class: every failure surfaced by `LLM.chat` is one of these."""


class BudgetExceeded(LLMError):
    pass


class LLMCallFailed(LLMError):
    """Transport-level failure after all retries, or a non-retryable API error (bad request, auth, ...)."""


class LLMInvalidResponse(LLMError):
    def __init__(self, problems: list[str]):
        super().__init__(f"unusable response after {len(problems)} attempt(s): {problems[-1] if problems else ''}")
        self.problems = problems


class LLMUnavailable(LLMError):
    """API unavailability requiring run-level interruption and later resumption."""


@dataclass
class LLMResult:
    text: str
    reasoning: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None
    latency_s: float
    cached: bool = False
    attempts: int = 1
    degraded: bool | None = None  # answered by the repair_overrides fallback (None: cached before this was recorded)
    repair_mode: str | None = None  # "length" or opt-in "validation"; None for original parameters


Validator = Callable[[LLMResult], "str | None"]  # returns a problem description, or None if usable


def check_nonempty(res: LLMResult) -> str | None:
    if not res.text.strip():
        hint = " (output budget exhausted while reasoning; answer briefly)" if res.finish_reason == "length" else ""
        return f"empty answer, finish_reason={res.finish_reason}{hint}"
    return None


@dataclass(frozen=True)
class RoleSpec:
    model: str
    params: dict = field(default_factory=dict)  # default request kwargs (max_tokens, timeout_s, extra_body, ...)


# ---------------------------------------------------------------- rate limit / cache


class RateLimiter:
    """Sliding-window RPM/optional TPM limiter; reserve returns the wait for a booked request."""

    def __init__(self, rpm: int, period: float = 60.0, tpm: int | None = None):
        self.rpm = rpm
        self.period = period
        self.tpm = tpm
        self._slots: list[float] = []
        self._token_slots: list[tuple[float, int]] = []
        self._lock = threading.Lock()

    def reserve(self, tokens: int = 0) -> float:
        if self.tpm is not None and tokens > self.tpm:
            raise ValueError(f"request token reservation {tokens} exceeds TPM allocation {self.tpm}")
        with self._lock:
            now = time.monotonic()
            if self.tpm is not None:
                self._token_slots = [(t, n) for t, n in self._token_slots if t > now - self.period]
                slot = max(now, self._token_slots[-1][0] if self._token_slots else now)
                while True:
                    active = [(t, n) for t, n in self._token_slots if t > slot - self.period]
                    if len(active) < self.rpm and sum(n for _, n in active) + tokens <= self.tpm:
                        break
                    slot = active[0][0] + self.period
                self._token_slots.append((slot, tokens))
                return max(0.0, slot - now)
            del self._slots[: bisect.bisect_right(self._slots, now - self.period)]
            if len(self._slots) < self.rpm:
                slot = now
            else:
                slot = self._slots[-self.rpm] + self.period
            bisect.insort(self._slots, slot)
            return max(0.0, slot - now)


class ResponseCache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL, created REAL NOT NULL)"
            )
            self._conn.commit()

    @staticmethod
    def make_key(payload: dict) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, created) VALUES (?, ?, ?)",
                (key, json.dumps(value, ensure_ascii=False), time.time()),
            )
            self._conn.commit()


# ---------------------------------------------------------------- client


def _backoff(attempt: int) -> float:
    return min(60.0, 2.0**attempt) * (0.5 + random.random())


def _bump(meter: dict | None, key: str, n: int = 1) -> None:
    if meter is not None:
        meter[key] = meter.get(key, 0) + n


class LLM:
    def __init__(
        self,
        api: APIConfig | None = None,
        roles: dict[str, Any] | None = None,
        rpm: int | None = None,
        max_concurrency: int | None = None,
        budget_tokens: int | None = None,
        cache_path: str | Path | None = PROJECT_ROOT / "runs" / "llm_cache.sqlite",
        timeout_s: float | None = None,
        max_retries: int | None = None,
        repair_attempts: int | None = None,
        breaker_threshold: int | None = None,
        breaker_cooldown_s: float | None = None,
        breaker_max_wait_s: float | None = None,
        tpm: int | None = None,
    ):
        cfg = load_models_config()
        a = cfg.get("api", {})
        self.api = api or load_api_config()
        self.roles = {k: self._role(v) for k, v in (roles if roles is not None else cfg.get("roles", {})).items()}
        self.limiter = RateLimiter(rpm or a.get("rpm", 70), tpm=tpm if tpm is not None else a.get("tpm"))
        self.max_concurrency = max_concurrency or a.get("max_concurrency", 8)
        self.timeout_s = timeout_s or a.get("timeout_s", 300)
        self.max_retries = max_retries if max_retries is not None else a.get("max_retries", 5)
        self.repair_attempts = repair_attempts if repair_attempts is not None else a.get("repair_attempts", 2)
        self.breaker_threshold = breaker_threshold or a.get("breaker_threshold", 5)
        self.breaker_cooldown_s = breaker_cooldown_s or a.get("breaker_cooldown_s", 30)
        self.breaker_max_wait_s = breaker_max_wait_s or a.get("breaker_max_wait_s", 1800)
        self.budget_tokens = budget_tokens
        self.cache = ResponseCache(cache_path) if cache_path else None
        self._client: openai.OpenAI | None = None
        self._sems: dict[int, asyncio.Semaphore] = {}
        self._lock = threading.Lock()
        self._fail_streak: dict[str, int] = {}
        self._probe_locks: dict[str, threading.Lock] = {}
        self.usage = {
            "calls": 0, "cached_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "transport_retries": 0, "invalid_responses": 0, "degraded_repairs": 0, "validation_repairs": 0,
            "call_failures": 0, "breaker_trips": 0,
        }

    # -- configuration

    @staticmethod
    def _role(v: Any) -> RoleSpec:
        if isinstance(v, RoleSpec):
            return v
        if isinstance(v, str):
            return RoleSpec(v)
        v = dict(v)
        return RoleSpec(v.pop("model"), v)

    def resolve(self, model: str) -> RoleSpec:
        return self.roles.get(model, RoleSpec(model))

    @property
    def client(self) -> openai.OpenAI:
        if self._client is None:
            self._client = openai.OpenAI(api_key=self.api.api_key, base_url=self.api.base_url, timeout=self.timeout_s, max_retries=0)
        return self._client

    def total_tokens(self) -> int:
        return self.usage["prompt_tokens"] + self.usage["completion_tokens"]

    def _count(self, key: str, n: int = 1, meter: dict | None = None) -> None:
        with self._lock:
            self.usage[key] += n
        _bump(meter, key, n)

    def _check_budget(self) -> None:
        if self.budget_tokens is not None and self.total_tokens() >= self.budget_tokens:
            raise BudgetExceeded(f"token budget {self.budget_tokens} exhausted")

    # -- one HTTP request with transport retries and the circuit breaker

    def _parse(self, resp: Any, model: str, latency: float) -> LLMResult:
        choices = getattr(resp, "choices", None) or []
        if not choices:
            return LLMResult("", "", model, 0, 0, "no_choices", latency)
        choice = choices[0]
        msg = choice.message
        extra = getattr(msg, "model_extra", None) or {}
        reasoning = getattr(msg, "reasoning_content", None) or extra.get("reasoning_content") or ""
        text, think = split_think(msg.content or "")
        usage = getattr(resp, "usage", None)
        return LLMResult(
            text=text, reasoning=reasoning or think, model=model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            finish_reason=choice.finish_reason, latency_s=latency,
        )

    def _request(self, model: str, messages: list[dict], temperature: float, max_tokens: int,
                 timeout: float, params: dict, meter: dict | None, max_retries: int | None = None) -> LLMResult:
        self._await_breaker(model)
        retries = self.max_retries if max_retries is None else max_retries
        last: BaseException | None = None
        for attempt in range(retries + 1):
            self._check_budget()
            time.sleep(self.limiter.reserve(self._token_reservation(messages, max_tokens)))
            t0 = time.monotonic()
            try:
                resp = self.client.with_options(timeout=timeout).chat.completions.create(
                    model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, **params
                )
                res = self._parse(resp, model, time.monotonic() - t0)
                with self._lock:
                    self._fail_streak[model] = 0
                    self.usage["calls"] += 1
                    self.usage["prompt_tokens"] += res.prompt_tokens
                    self.usage["completion_tokens"] += res.completion_tokens
                _bump(meter, "calls")
                _bump(meter, "prompt_tokens", res.prompt_tokens)
                _bump(meter, "completion_tokens", res.completion_tokens)
                return res
            except Exception as e:  # noqa: BLE001 - classified below
                if not is_retryable(e):
                    self._count("call_failures", meter=meter)
                    raise LLMCallFailed(f"{type(e).__name__}: {str(e)[:300]}") from e
                last = e
                if attempt < retries:
                    self._count("transport_retries", meter=meter)
                    time.sleep(_backoff(attempt))
        self._count("call_failures", meter=meter)
        with self._lock:
            self._fail_streak[model] = self._fail_streak.get(model, 0) + 1
        raise LLMCallFailed(f"{retries + 1} attempts failed; last: {type(last).__name__}: {str(last)[:300]}")

    @staticmethod
    def _token_reservation(messages: list[dict], max_tokens: int) -> int:
        # No provider tokenizer is supplied. Reserve UTF-8 bytes (rather than chars/4),
        # message framing, and the entire output cap; do not refund unused capacity.
        return len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) + 16 * len(messages) + 32 + max_tokens

    def _await_breaker(self, model: str) -> None:
        """Block while `model` has failed `breaker_threshold` times in a row; probe it until it answers."""
        if self._fail_streak.get(model, 0) < self.breaker_threshold:
            return
        with self._lock:
            lock = self._probe_locks.setdefault(model, threading.Lock())
        with lock:  # one thread probes; the others wait here and re-check
            if self._fail_streak.get(model, 0) < self.breaker_threshold:
                return
            self._count("breaker_trips")
            deadline = time.monotonic() + self.breaker_max_wait_s
            while time.monotonic() < deadline:
                time.sleep(self.breaker_cooldown_s)
                try:
                    time.sleep(self.limiter.reserve(self._token_reservation([{"role": "user", "content": "ping"}], 8)))
                    self.client.with_options(timeout=60).chat.completions.create(
                        model=model, messages=[{"role": "user", "content": "ping"}], max_tokens=8, temperature=0.0
                    )
                    with self._lock:
                        self._fail_streak[model] = 0
                    return
                except Exception:  # noqa: BLE001 - still down
                    continue
            raise LLMUnavailable(f"{model}: circuit open for more than {self.breaker_max_wait_s:.0f}s")

    # -- public API

    def chat(
        self,
        messages: list[dict],
        model: str = "fast",
        temperature: float = 0.7,
        max_tokens: int | None = None,
        seed_tag: str = "0",
        use_cache: bool = True,
        validate: Validator | None = None,
        repair_attempts: int | None = None,
        timeout: float | None = None,
        meter: dict | None = None,
        on_invalid: Callable[[LLMResult, str, int], None] | None = None,
        **extra: Any,
    ) -> LLMResult:
        """Return a response that passed `check_nonempty` and `validate`, or raise an LLMError."""
        spec = self.resolve(model)
        params = {**spec.params, **extra}
        max_tokens = max_tokens or params.pop("max_tokens", 4096)
        timeout = timeout or params.pop("timeout_s", self.timeout_s)
        max_retries = params.pop("max_retries", None)  # per-role override: long, expensive calls retry less
        # per-role params for a repair after an *empty answer at the length limit* (reasoning used the whole
        # budget): e.g. turn reasoning off so the retry is cheap and returns an answer
        degrade = params.pop("repair_overrides", None)
        validation_overrides = params.pop("validation_repair_overrides", None)
        repair_context_chars = params.pop("repair_context_chars", 4000)
        params.pop("max_tokens", None)
        params.pop("timeout_s", None)
        repairs = self.repair_attempts if repair_attempts is None else repair_attempts

        def check(res: LLMResult) -> str | None:
            problem = check_nonempty(res)
            if problem is None and validate is not None:
                try:
                    problem = validate(res)
                except Exception as e:  # noqa: BLE001 - a crashing validator means the output is unusable
                    problem = f"validator error: {type(e).__name__}: {e}"
            return problem

        msgs = list(messages)
        problems: list[str] = []
        call_params, call_max_tokens = params, max_tokens
        repair_mode = None
        for attempt in range(repairs + 1):
            tag = seed_tag if attempt == 0 else f"{seed_tag}#repair{attempt}"
            payload = {"model": spec.model, "messages": msgs, "temperature": temperature,
                       "max_tokens": call_max_tokens, "seed_tag": tag, "params": call_params}
            key = self.cache.make_key(payload) if (self.cache and use_cache) else None
            res = None
            if key:
                hit = self.cache.get(key)
                if hit is not None:
                    cand = LLMResult(**{**hit, "cached": True})
                    if check(cand) is None:
                        self._count("cached_calls", meter=meter)
                        # per-episode cost is the logical cost: a replayed call still counts as one call with its
                        # original tokens (process-wide `usage` keeps counting only real API traffic)
                        _bump(meter, "calls")
                        _bump(meter, "prompt_tokens", cand.prompt_tokens)
                        _bump(meter, "completion_tokens", cand.completion_tokens)
                        return cand
            res = self._request(spec.model, msgs, temperature, call_max_tokens, timeout, call_params, meter, max_retries)
            problem = check(res)
            if problem is None:
                res.attempts = attempt + 1
                res.degraded = call_params is not params
                res.repair_mode = repair_mode
                if key:
                    self.cache.put(key, asdict(res))
                return res
            problems.append(problem)
            self._count("invalid_responses", meter=meter)
            if on_invalid is not None:
                on_invalid(res, problem, attempt + 1)
            if degrade and not res.text.strip() and res.finish_reason == "length" and call_params is params:
                over = dict(degrade)
                call_max_tokens = over.pop("max_tokens", max_tokens)
                call_params = {**params, **over}
                self._count("degraded_repairs", meter=meter)
                repair_mode = "length"
            elif validation_overrides and res.text.strip() and call_params is params and attempt < repairs:
                over = dict(validation_overrides)
                call_max_tokens = over.pop("max_tokens", max_tokens)
                call_params = {**params, **over}
                self._count("validation_repairs", meter=meter)
                repair_mode = "validation"
            msgs = list(messages) + [
                {"role": "assistant", "content": res.text[:repair_context_chars] or "(no answer)"},
                {"role": "user", "content": f"Your previous reply was unusable: {problem}. "
                                            "Reply again, briefly, following the required output format exactly."},
            ]
        raise LLMInvalidResponse(problems)

    async def achat(self, *args: Any, **kwargs: Any) -> LLMResult:
        loop_id = id(asyncio.get_running_loop())
        sem = self._sems.setdefault(loop_id, asyncio.Semaphore(self.max_concurrency))
        async with sem:
            return await asyncio.to_thread(self.chat, *args, **kwargs)

    async def amap(self, requests: list[dict], return_exceptions: bool = True) -> list[LLMResult | BaseException]:
        """Run many `chat(**req)` concurrently (bounded by max_concurrency and the RPM limiter)."""
        return await asyncio.gather(*(self.achat(**r) for r in requests), return_exceptions=return_exceptions)

    def health_check(self, names: list[str], timeout: float = 120) -> list[dict]:
        """Ping each role/model ("Reply with exactly: OK"), bypassing the cache. Never raises."""
        out = []
        for name in names:
            spec = self.resolve(name)
            t0 = time.monotonic()
            try:
                res = self.chat(
                    [{"role": "user", "content": "Reply with exactly: OK"}], model=name, temperature=0.0,
                    use_cache=False, timeout=timeout, repair_attempts=1,
                    validate=lambda r: None if "ok" in r.text.lower() else f"expected OK, got {r.text[:40]!r}",
                )
                out.append({"name": name, "model": spec.model, "ok": True, "latency_s": round(time.monotonic() - t0, 1),
                            "attempts": res.attempts})
            except LLMError as e:
                out.append({"name": name, "model": spec.model, "ok": False, "latency_s": round(time.monotonic() - t0, 1),
                            "error": str(e)[:300]})
        return out


# ---------------------------------------------------------------- text helpers

_THINK = re.compile(r"<think>(.*?)</think>", re.S)


def split_think(text: str) -> tuple[str, str]:
    """Separate inline <think>…</think> reasoning from the answer."""
    thinks = _THINK.findall(text)
    return _THINK.sub("", text).strip(), "\n".join(t.strip() for t in thinks)


_FENCE = re.compile(r"```([\w+-]*)[ \t]*\n(.*?)```", re.S)


def extract_code_blocks(text: str, lang: str = "python") -> list[str]:
    """Fenced blocks tagged `lang` (or untagged). Fences are consumed in pairs, so a ```json block
    is never mistaken for the gap between two other fences."""
    return [body.strip("\n") for tag, body in _FENCE.findall(text) if tag.lower() in (lang.lower(), "")]


def extract_json(text: str) -> Any:
    """First parseable JSON value: fenced ```json blocks, the whole text, then any embedded object/array."""
    for block in extract_code_blocks(text, "json"):
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                return decoder.raw_decode(text, i)[0]
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON found in text")
