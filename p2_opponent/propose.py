"""LLM proposer: writes opponent-model programs. It never scores them; the posterior does.

Two entry points:
- `make_llm_proposer`: called during play when the catch-all out-predicts the named hypotheses. The
  LLM sees the current hypotheses, the rounds predicted worst, and the evidence the catch-all is winning
  with: its learned conditional table P(opponent next | last round), plus derived views of the same data
  (by the agent's last action, the opponent's last action, the last outcome, and relative shifts), over
  all rounds and the most recent window. Hypotheses are thus grounded in data, not invented from names.
- `generate_initial_library`: K0 programs written from the rules alone, before any play (guideline §3.5).
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from typing import Callable

import numpy as np

from bsi.llm import LLM, LLMResult, extract_code_blocks, extract_json
from bsi.posterior import Posterior
from bsi.sandbox import SandboxPool, validate_source

from .agents import ProgramSpec
from .games import Game

CONTRACT = """\
Write each opponent model as a Python function:

    def model(history, params, game):
        # history: tuple of (agent_action, opponent_action) int pairs, oldest first (may be empty)
        # params:  dict of numeric parameters (values come from your param grid)
        # game:    dict with n_agent, n_opp, payoff_agent, payoff_opp (payoff_X[i][j] = payoff when agent plays i, opponent plays j)
        # return:  probabilities over the opponent's NEXT action (length n_opp, non-negative)

Sandbox rules (violations are rejected): no imports; only `np` (common numpy functions) and `math` are available;
no classes, no names or attributes starting with "_", no file/network access; helper functions are allowed.
Each call must finish in a few milliseconds (it is called ~|grid| x rounds times when fitting). Keep programs short:
shorter programs get a higher prior. A model may learn from `history` itself (e.g. count-based tables)."""

OUTPUT_FORMAT = """\
Output format, repeated for each hypothesis:
### Hypothesis: <one-sentence description>
```json
{{"param_grid": {{"name": [values, ...]}}}}
```
```python
def model(history, params, game):
    ...
```"""

SYSTEM = "You are an expert in game theory and behavioural modelling. You write small, precise Python opponent models."


# ---------------------------------------------------------------- evidence formatting


def _names(game: Game) -> tuple[list[str], list[str]]:
    return list(game.agent_actions), list(game.opp_actions)


def _dist_str(p: np.ndarray, names: list[str]) -> str:
    return " ".join(f"{n}={v:.2f}" for n, v in zip(names, p))


def _cond_table(pairs: list[tuple[str, int]], n: int, names: list[str], key_label: str, min_n: int = 2) -> str:
    """pairs: (condition label, outcome index). Rows sorted by count; Laplace-free empirical frequencies."""
    groups: dict[str, Counter] = {}
    for k, o in pairs:
        groups.setdefault(k, Counter())[o] += 1
    rows = []
    for k, c in sorted(groups.items(), key=lambda kv: -sum(kv[1].values())):
        tot = sum(c.values())
        if tot < min_n:
            continue
        p = np.array([c[i] for i in range(n)], dtype=float) / tot
        rows.append(f"  {key_label}={k:<22} n={tot:<3} -> {_dist_str(p, names)}")
    return "\n".join(rows) if rows else "  (not enough data)"


def outcome_label(game: Game, a: int, o: int) -> str:
    """Last outcome from the opponent's point of view."""
    po, pa = game.payoff_opp[a, o], game.payoff_agent[a, o]
    return "opp_won" if po > pa else ("opp_lost" if po < pa else "tie")


def evidence_tables(posterior: Posterior, history: tuple, game: Game, recent: int = 30) -> str:
    """The catch-all's learned table and derived conditional views, over all rounds and the last `recent`.
    After a change-point restart (seg_start > 0) the views cover the current segment only, like the catch-all."""
    an, on = _names(game)
    n = game.n_opp
    parts = []
    other = posterior.get(posterior.other_id) if posterior.other_id else None
    cond = getattr(other, "cond", None)
    if cond:
        rows = []
        for key, counts in sorted(cond.items(), key=lambda kv: -kv[1].sum()):
            if key is None or counts.sum() < 2:
                continue
            a, o = key
            pred = other.predict((key,))  # the catch-all's actual (backed-off) predictive for this context
            rows.append(f"  last=(agent {an[a]}, opp {on[o]})  n={int(counts.sum()):<3} -> {_dist_str(pred, on)}")
        parts.append(
            "Catch-all model's learned table P(opponent next | last round) [this is what currently out-predicts "
            "your hypotheses]:\n" + ("\n".join(rows) if rows else "  (not enough data)")
        )
    T = len(history)
    seg = posterior.seg_start
    first = ("all rounds", 1) if seg == 0 else (f"current segment, rounds {seg}-{T - 1}", max(1, seg))
    for label, lo in (first, (f"last {recent} rounds", max(1, seg, T - recent))):
        idx = range(lo, T)
        views = [
            ("agent_last", [(an[history[t - 1][0]], history[t][1]) for t in idx]),
            ("opp_last", [(on[history[t - 1][1]], history[t][1]) for t in idx]),
            ("last_outcome", [(outcome_label(game, *history[t - 1]), history[t][1]) for t in idx]),
        ]
        if game.n_agent == game.n_opp:
            views.append(("(opp_next - agent_last) mod n", [(str((history[t][1] - history[t - 1][0]) % n), history[t][1]) for t in idx]))
            views.append(("(opp_next - opp_last) mod n", [(str((history[t][1] - history[t - 1][1]) % n), history[t][1]) for t in idx]))
        block = []
        for key_label, pairs in views:
            if key_label.startswith("("):  # shift views: distribution of the shift itself
                shifts = Counter(k for k, _ in pairs)
                tot = sum(shifts.values()) or 1
                block.append(f" {key_label}: " + " ".join(f"{s}:{shifts[s] / tot:.2f}" for s in sorted(shifts)))
            else:
                block.append(f" P(opp next | {key_label}):\n{_cond_table(pairs, n, on, key_label)}")
        parts.append(f"Derived views of the same data ({label}):\n" + "\n".join(block))
    return "\n\n".join(parts)


def longer_range_views(history: tuple, game: Game, max_lag: int = 6, periods: tuple[int, ...] = (2, 3, 4, 5),
                       start: int = 0) -> str:
    """Structure the catch-all's last-round table cannot show: lag-k repeats, phase (t mod p) tables and
    P(next | opponent's last two actions). Round t's context is history[:t], so a cycle bot plays seq[t % len].
    Only rounds t >= `start` (and their lagged rounds) are used: after a change point, the current segment."""
    on = list(game.opp_actions)
    n = game.n_opp
    T = len(history)
    opp = [o for _, o in history]
    lags = []
    for k in range(1, max_lag + 1):
        same = [opp[t] == opp[t - k] for t in range(start + k, T)]
        lags.append(f"{k}:{np.mean(same):.2f}" if same else f"{k}:–")
    parts = [f"P(opp action at t == opp action at t-k), k = lag:\n  " + " ".join(lags)]
    for p in periods:
        rows = []
        for ph in range(p):
            c = np.bincount([opp[t] for t in range(ph, T, p) if t >= start], minlength=n)
            if c.sum():
                rows.append(f"  t mod {p} = {ph}  n={int(c.sum()):<3} -> {_dist_str(c / c.sum(), on)}")
        parts.append(f"P(opp action | t mod {p}) [round index t from 0]:\n" + "\n".join(rows))
    pairs = [(f"{on[opp[t - 2]]},{on[opp[t - 1]]}", opp[t]) for t in range(start + 2, T)]
    parts.append("P(opp next | opp's last two actions, oldest first):\n" + _cond_table(pairs, n, on, "opp_last2"))
    return "\n\n".join(parts)


TASK_LAG1 = """\
Task: propose {n_new} NEW opponent models that explain the evidence better than the current hypotheses.
Ground every hypothesis in the tables above: name the regularity you see (e.g. "after agent X the opponent plays Y
~80% of the time") and model it. The regularity need not match a textbook strategy: if a conditional table looks
near-deterministic but arbitrary, write a model that encodes or learns that mapping (with a noise parameter).
If recent rounds differ from all rounds, consider a strategy change (e.g. a model fitted on a recent window).
Prefer few parameters; give each a small grid (<= 6 values each, <= 60 combinations); they are fitted by maximum likelihood."""

TASK_RICH = """\
Task: propose {n_new} NEW opponent models that explain the evidence better than the current hypotheses.
The catch-all ALREADY is a learned table P(opponent next | last round). A model that only conditions on the last
round, or that re-learns such a table by counting, cannot beat it and will be wasted. Propose compact models of
structure the catch-all cannot express, for example:
- periodic / phase-dependent play (look at the lag-k repeat rates and the t mod p tables: a lag or period with a
  rate near 1 is a strong sign), e.g. "plays the action from p rounds ago" or a fixed sequence indexed by t mod p;
- dependence on two or more past rounds;
- strategic reasoning about the agent (e.g. best-responding, with a softmax temperature, to the agent's recent
  action frequencies, or to what the agent would best-respond to);
- outcome rules (win-stay / lose-shift) expressed with one or two parameters instead of a table.
Name the regularity you see in the tables and model it with few parameters (explicit parameters beat learned tables).
If recent rounds differ from all rounds, consider a strategy change (e.g. a model fitted on a recent window).
Give each parameter a small grid (<= 6 values each, <= 60 combinations); they are fitted by maximum likelihood."""


def _fmt_rounds(history: tuple, game: Game, rounds: list[int], mix_ll: list[float]) -> str:
    lines = []
    for t in rounds:
        a, o = history[t]
        p = float(np.exp(mix_ll[t])) if t < len(mix_ll) else float("nan")
        lines.append(f"t={t:>3}  agent={game.agent_actions[a]:<10} opponent={game.opp_actions[o]:<10} p_model(opponent action)={p:.2f}")
    return "\n".join(lines)


def build_messages(posterior: Posterior, history: tuple, game: Game, n_new: int = 2, show_last: int = 30,
                   evidence: str = "lag1") -> list[dict]:
    """evidence="lag1" is the v1/v2 prompt (byte-identical, so cached replies stay valid); "rich" adds
    longer-range views and asks for structure the last-round catch-all cannot express."""
    if evidence not in ("lag1", "rich"):
        raise ValueError(f"unknown proposer evidence {evidence!r}")
    T = len(history)
    seg = posterior.seg_start
    mix = posterior.mix_ll
    last = list(range(max(0, T - show_last), T))
    worst = sorted(range(seg, len(mix)), key=lambda t: mix[t])[:10]
    top = []
    for row in posterior.summary(top=4):
        if row["kind"] == "other":
            top.append(f"- [catch-all table model] weight={row['weight']:.2f} recent_ll/round={row['recent_mean_ll']:.3f}")
            continue
        src = getattr(posterior.get(row["id"]), "src", "").strip()
        top.append(
            f"- {row['id']} weight={row['weight']:.2f} recent_ll/round={row['recent_mean_ll']:.3f} "
            f"fitted_params={json.dumps(row['params'])}\n  statement: {row['statement']}\n```python\n{src}\n```"
        )
    counts = np.bincount([o for _, o in history[seg:]], minlength=game.n_opp)
    ev = evidence_tables(posterior, history, game)
    if evidence == "rich":
        scope = "all rounds" if seg == 0 else f"current segment, rounds {seg}-{T - 1}"
        ev += f"\n\nLonger-range views ({scope}):\n" + longer_range_views(history, game, start=seg)
    played = f"Rounds played: {T}. Opponent action counts: {counts.tolist()}." if seg == 0 else (
        f"Rounds played: {T}. A strategy change was detected: the belief was reset at round {seg}, and all evidence "
        f"below covers rounds {seg}-{T - 1} only. Opponent action counts since then: {counts.tolist()}.")
    task = (TASK_RICH if evidence == "rich" else TASK_LAG1).format(n_new=n_new)
    user = f"""\
Game: {game.description}
Agent actions: {list(game.agent_actions)}; opponent actions: {list(game.opp_actions)}.
{played}

Current hypotheses about the opponent (posterior weight, recent mean log-likelihood per round). The catch-all
table model winning means the named hypotheses miss real structure:
{chr(10).join(top)}

EVIDENCE
{ev}

Last {len(last)} rounds:
{_fmt_rounds(history, game, last, mix)}

Rounds the current belief predicted worst{"" if seg == 0 else " (current segment)"}:
{_fmt_rounds(history, game, sorted(worst), mix)}

{task}

{CONTRACT}

{OUTPUT_FORMAT}"""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


# ---------------------------------------------------------------- parsing


def parse_proposals(text: str, prefix: str = "llm") -> list[ProgramSpec]:
    specs = []
    parts = re.split(r"^#{2,4}\s*Hypothesis\s*:?", text, flags=re.M | re.I)
    for k, part in enumerate(parts[1:]):
        statement = part.strip().splitlines()[0].strip() if part.strip() else ""
        code = extract_code_blocks(part, "python")
        if not code:
            continue
        grid: dict = {}
        try:
            obj = extract_json(part.split("```python")[0])
            grid = obj.get("param_grid", obj) if isinstance(obj, dict) else {}
        except ValueError:
            grid = {}
        grid = {str(k2): list(v) for k2, v in grid.items() if isinstance(v, list) and v}
        specs.append(ProgramSpec(f"{prefix}{k}", code[0], grid, statement))
    return specs


def proposals_validator(min_specs: int = 1) -> Callable[[LLMResult], str | None]:
    """Health check for proposer replies: at least `min_specs` hypotheses that pass static sandbox validation."""

    def check(res: LLMResult) -> str | None:
        specs = parse_proposals(res.text)
        if not specs:
            return "no '### Hypothesis:' sections with a ```python block were found"
        ok = [s for s in specs if not validate_source(s.src)]
        if len(ok) < min_specs:
            errs = "; ".join(validate_source(s.src)[0] for s in specs if validate_source(s.src))
            return f"no program passed the sandbox rules ({errs[:300]})"
        return None

    return check


def make_llm_proposer(llm: LLM, model: str = "proposer", n_new: int = 2, temperature: float = 0.7,
                      max_tokens: int | None = None, evidence: str = "lag1"):
    """Returns proposer(posterior, history, game, tag="", meter=None) -> [ProgramSpec].

    Raises LLMError subclasses on failure; PosteriorBRAgent turns those into a recorded no-op
    (fatal ones, LLMUnavailable / BudgetExceeded, propagate and stop the run)."""
    log: list[dict] = []

    def proposer(posterior: Posterior, history: tuple, game: Game, tag: str = "", meter: dict | None = None) -> list[ProgramSpec]:
        n = len(log) + 1
        msgs = build_messages(posterior, history, game, n_new=n_new, evidence=evidence)
        res = llm.chat(msgs, model=model, temperature=temperature, max_tokens=max_tokens, seed_tag=f"propose:{tag}:{n}",
                       validate=proposals_validator(), meter=meter)
        specs = [s for s in parse_proposals(res.text, prefix=f"llm{n}_") if not validate_source(s.src)]
        log.append({"t": len(history), "model": res.model, "finish_reason": res.finish_reason, "cached": res.cached,
                    "attempts": res.attempts, "completion_tokens": res.completion_tokens,
                    "latency_s": round(res.latency_s, 1), "n_specs": len(specs), "reasoning_chars": len(res.reasoning),
                    # answered by the reasoning-off repair (repair_overrides) after reasoning used the whole budget;
                    # replies cached before LLM.chat recorded it: inferred (every first-try reply has reasoning)
                    "degraded": res.degraded if res.degraded is not None else (res.attempts > 1 and not res.reasoning)})
        return specs

    proposer.log = log  # type: ignore[attr-defined]
    return proposer


# ---------------------------------------------------------------- oracle proposer (diagnostic upper bound)

# bot family -> hand-written program(s) of the same model class (seed_programs.SEED_LIBRARY ids).
# fictitious bots are KLevel(k=1) and describe themselves as "klevel"; klevel k=0 is a windowed habit.
ORACLE_PROGRAMS = {
    "fixed": ["habit"],
    "reactive": ["reactive"],
    "wsls": ["wsls"],
    "fictitious": ["klevel"],
    "klevel": ["klevel", "habit"],
    "cycle": ["cycle"],
    # v5 families outside the seed menu (seed_programs.NOVEL_PROGRAMS ids)
    "lose_k_shift": ["lose_k_shift"],
    "react2": ["react2"],
    "seq_br": ["seq_br"],
    "block_drift": ["block_drift"],
}


def oracle_program_ids(bot_desc: dict) -> list[str]:
    fam = bot_desc["family"]
    if fam == "switching":
        p = bot_desc["params"]
        ids = oracle_program_ids(p["first"]) + oracle_program_ids(p["second"])
        return list(dict.fromkeys(ids))
    return list(ORACLE_PROGRAMS[fam])


def oracle_specs(bot_desc: dict, n_opp: int | None = None) -> list[ProgramSpec]:
    """The hand-written programs of the true bot's model class, with ids prefixed `oracle_`. With `n_opp`, a grid
    over opponent action indices ("start") keeps only the game's actions, so no action gets duplicate prior mass."""
    from .seed_programs import NOVEL_PROGRAMS, SEED_LIBRARY

    by_id = {**{s.id: s for s in SEED_LIBRARY}, **NOVEL_PROGRAMS}
    out = []
    for i in oracle_program_ids(bot_desc):
        grid = dict(by_id[i].param_grid)
        if n_opp is not None and "start" in grid:
            grid["start"] = [v for v in grid["start"] if v < n_opp]
        out.append(ProgramSpec(f"oracle_{i}", by_id[i].src, grid, by_id[i].statement))
    return out


def make_oracle_proposer(bot_desc: dict):
    """A proposer that, on its first call, returns the hand-written programs of the true bot's model class
    (parameters still fitted by the posterior); later calls return nothing. No LLM. It measures what the
    library-evolution machinery gains when a correct new hypothesis is offered at the trigger time."""
    log: list[dict] = []

    def proposer(posterior: Posterior, history: tuple, game: Game, tag: str = "", meter: dict | None = None) -> list[ProgramSpec]:
        specs = [] if log else oracle_specs(bot_desc, game.n_opp)
        log.append({"t": len(history), "n_specs": len(specs)})
        return specs

    proposer.log = log  # type: ignore[attr-defined]
    return proposer


def make_template_proposer(menu: str):
    """No-LLM control for the LLM proposer: on its first call (same trigger), returns a fixed menu of generic
    programs for the structures the rich prompt asks for (seed_programs.TEMPLATE_MENUS); later calls return
    nothing. It knows nothing about the bot. If it matches the LLM's gain, writing new programs needs no LLM."""
    from .seed_programs import TEMPLATE_MENUS

    specs_all = TEMPLATE_MENUS[menu]
    log: list[dict] = []

    def proposer(posterior: Posterior, history: tuple, game: Game, tag: str = "", meter: dict | None = None) -> list[ProgramSpec]:
        specs = [] if log else [ProgramSpec(s.id, s.src, s.param_grid, s.statement) for s in specs_all]
        log.append({"t": len(history), "n_specs": len(specs)})
        return specs

    proposer.log = log  # type: ignore[attr-defined]
    return proposer


# ---------------------------------------------------------------- initial library (K0)


def _initial_messages(game: Game, k: int) -> list[dict]:
    user = f"""\
Game: {game.description}
Agent actions: {list(game.agent_actions)}; opponent actions: {list(game.opp_actions)}.
Payoffs: payoff_agent = {game.payoff_agent.tolist()}, payoff_opp = {game.payoff_opp.tolist()} (row = agent action, column = opponent action).

Before any play, write {k} DIVERSE models of how an opponent in this repeated game might choose its actions.
Cover different kinds of behaviour (static biases, reactions to the last round, outcome-dependent rules, learning
from the agent's history, strategic/best-response reasoning, periodic patterns, ...). Give each model a small
parameter grid (<= 6 values per parameter, <= 60 combinations); parameters are fitted by maximum likelihood later.

{CONTRACT}

{OUTPUT_FORMAT}"""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def generate_initial_library(
    llm: LLM, game: Game, pool: SandboxPool, k0: int = 8, model: str = "proposer", max_calls: int = 4,
    max_call_ms: float = 25.0, tag: str = "", meter: dict | None = None,
) -> tuple[list[ProgramSpec], list[dict]]:
    """Ask for programs until k0 pass sandbox loading, a smoke call on empty and random histories, and the
    per-call speed limit. Returns (specs, log). Deterministic given the cache (fixed seed tags)."""
    rng = np.random.default_rng(0)
    probe = tuple((int(rng.integers(game.n_agent)), int(rng.integers(game.n_opp))) for _ in range(150))
    accepted: list[ProgramSpec] = []
    log: list[dict] = []
    for call in range(max_calls):
        need = k0 - len(accepted)
        if need <= 0:
            break
        res = llm.chat(_initial_messages(game, need + 2), model=model, temperature=0.8,
                       seed_tag=f"k0:{game.name}:{tag}:{call}", validate=proposals_validator(), meter=meter)
        for spec in parse_proposals(res.text, prefix=f"k0_{call}_"):
            if len(accepted) >= k0:
                break
            pid = f"k0check:{game.name}:{call}:{spec.id}"
            errs = pool.load(pid, spec.src)
            reason = "; ".join(errs)
            if not errs:
                try:
                    from .hypotheses import ProgramHypothesis

                    h = ProgramHypothesis(pid, spec.src, pool, game.info(), spec.param_grid, spec.statement)
                    h.predict(())
                    t0 = time.perf_counter()
                    h.predict(probe)
                    ms = (time.perf_counter() - t0) * 1000
                    if ms > max_call_ms:
                        raise TimeoutError(f"too slow: {ms:.1f} ms")
                except Exception as e:  # noqa: BLE001
                    reason = f"{type(e).__name__}: {e}"
                pool.unload(pid)
            log.append({"call": call, "id": spec.id, "ok": not reason, "reason": reason, "statement": spec.statement})
            if not reason:
                accepted.append(ProgramSpec(f"k0_{len(accepted)}", spec.src, spec.param_grid, spec.statement))
    return accepted, log


def save_library(path, specs: list[ProgramSpec], log: list[dict]) -> None:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"specs": [s.__dict__ for s in specs], "log": log}, ensure_ascii=False, indent=1))


def load_library(path) -> list[ProgramSpec]:
    from pathlib import Path

    obj = json.loads(Path(path).read_text())
    return [ProgramSpec(**d) for d in obj["specs"]]
