"""v5 prompt for the menu-choice agent: the LLM decides which opponent models are active.

At each decision point the LLM sees the game, a catalogue of prewritten models (the menu), the active models with
their fit, its earlier decisions, and the play history as a structured table. It may activate menu models (setting
their parameter grids), write new models under the same program contract, revise or retire active ones. It never
scores models or chooses actions: the posterior scores the active models exactly, the agent best-responds.

The task text presents the two ways of adding models symmetrically and in a per-episode random order, so the
choice between them is the LLM's (and position bias averages out across episodes).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np

from bsi.llm import LLMResult, extract_code_blocks, extract_json
from bsi.sandbox import validate_source

from .agents import ProgramSpec
from .games import Game
from .propose import CONTRACT, outcome_label

SYSTEM = ("You are an expert in game theory and behavioural modelling. You maintain a small set of executable models "
          "of the opponent in a repeated game; an exact Bayesian scorer weighs them against the observed play.")

MODES = ("select_create", "create_only", "menu_all_create", "select_only")
HISTORY_MODES = ("window", "full")
OPS = ("select", "create", "revise", "retire", "keep")
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
MAX_GRID_POINTS = 96  # PosteriorBRAgent max_grid: larger grids are subsampled, so ask for at most this many


# ---------------------------------------------------------------- state handed to the prompt


@dataclass
class ActiveModel:
    name: str          # how the LLM refers to it (a menu id or the LLM's own name)
    origin: str        # "menu" | "created"
    spec: ProgramSpec
    revisions: int = 0
    own_code: bool = False  # the program is the LLM's (created, or a menu model whose code it replaced)


@dataclass
class ModelFit:
    name: str
    origin: str
    params: dict
    weight: float
    mean_ll_recent: float
    mean_ll_all: float


@dataclass
class PromptState:
    t: int
    horizon: int
    decision_index: int
    n_decisions: int
    fits: list[ModelFit]
    reference: ModelFit | None
    mixture_mean_ll_recent: float
    log: list[dict] = field(default_factory=list)  # earlier decisions: {"t", "ops": [{"op", "name", "ok", "reason"}]}


# ---------------------------------------------------------------- rendering


def _payoff_table(game: Game) -> str:
    on = list(game.opp_actions)
    rows = ["rows = agent action, columns = opponent action; each cell is (agent payoff, opponent payoff)",
            "             " + " ".join(f"{f'{j}:{o}':>14}" for j, o in enumerate(on))]
    for i, a in enumerate(game.agent_actions):
        cells = " ".join(f"{f'({game.payoff_agent[i, j] + 0.0:g}, {game.payoff_opp[i, j] + 0.0:g})':>14}"  # + 0.0: no "-0"
                         for j in range(len(on)))
        rows.append(f"  {f'{i}:{a}':<10} {cells}")
    return "\n".join(rows)


def game_block(game: Game) -> str:
    return (f"{game.description}\n"
            f"Agent actions (index:name): {', '.join(f'{i}:{a}' for i, a in enumerate(game.agent_actions))}\n"
            f"Opponent actions (index:name): {', '.join(f'{j}:{o}' for j, o in enumerate(game.opp_actions))}\n"
            f"{_payoff_table(game)}\n"
            "Programs receive the same payoffs as game['payoff_agent'][i][j] and game['payoff_opp'][i][j] "
            "(i = agent action, j = opponent action).")


RESULT_CODE = {"opp_won": "W", "opp_lost": "L", "tie": "T"}


def render_history(history: tuple, game: Game, mode: str, window: int) -> str:
    """Structured table of rounds. mode="full": every round; "window": the last `window` rounds only (the scorer
    still uses every round, but nothing raw from the hidden rounds is shown: the counts cover the shown rounds).
    Rows: t | a | o | pay_a | pay_o | res, with actions as indices and res the round's result for the opponent
    (W win, L loss, T tie: its payoff above / below / equal to the agent's)."""
    if mode not in HISTORY_MODES:
        raise ValueError(f"unknown history mode {mode!r}")
    T = len(history)
    if T == 0:
        return "No rounds played yet."
    start = 0 if mode == "full" else max(0, T - window)
    shown = range(start, T)
    head = (f"Rounds played: {T} (t = 0..{T - 1}). "
            + ("All rounds are shown." if start == 0 else
               f"Only the last {T - start} rounds (t = {start}..{T - 1}) are shown; earlier rounds are not shown "
               "but every round counts in the model scores."))
    counts = np.bincount([history[t][1] for t in shown], minlength=game.n_opp).tolist()
    summary = f"Opponent action counts by index, {'all' if start == 0 else 'shown'} rounds: {counts}"
    rows = ["  t | a | o | pay_a | pay_o | res"]
    for t in shown:
        a, o = history[t]
        rows.append(f"{t:>3} | {a} | {o} | {game.payoff_agent[a, o] + 0.0:>5g} | {game.payoff_opp[a, o] + 0.0:>5g} | "
                    f"{RESULT_CODE[outcome_label(game, a, o)]}")
    legend = ("Columns: t = round index (from 0); a = agent action index; o = opponent action index; pay_a, pay_o = "
              "payoffs of that round; res = the round's result for the opponent (W: its payoff above the agent's, "
              "L: below, T: equal). The opponent chose o without seeing a of the same round.")
    return "\n".join([head, summary, legend, *rows])


def _grid_str(grid: dict) -> str:
    return json.dumps(grid) if grid else "{} (no parameters)"


def menu_block(menu: list[ProgramSpec]) -> str:
    parts = []
    for s in menu:
        parts.append(f"[{s.id}] {s.statement}\n  default param_grid: {_grid_str(s.param_grid)}\n```python\n{s.src.strip()}\n```")
    return "\n\n".join(parts)


def _ll(v: float) -> str:
    return "n/a" if not np.isfinite(v) else f"{v:.3f}"


def state_block(state: PromptState, game: Game, active: dict[str, ActiveModel]) -> str:
    unif = -np.log(game.n_opp)
    lines = [f"Decision point {state.decision_index + 1} of {state.n_decisions}, after round t = {state.t - 1} "
             f"({state.t} of {state.horizon} rounds played).",
             f"Mean log-likelihood per round of the opponent's actual actions (higher is better; a uniform guess scores "
             f"{unif:.3f}; 0 is perfect). 'recent' = last 20 rounds, 'all' = every round so far.",
             "Active models:"]
    if not state.fits:
        lines.append("  (none)")
    for f in state.fits:
        am = active.get(f.name)
        src = am.spec.src.strip() if am else ""
        rev = am.revisions if am else 0
        lines.append(f"- {f.name} (origin: {f.origin}{f', revised {rev}x' if rev else ''}) weight={f.weight:.2f} "
                     f"recent={_ll(f.mean_ll_recent)} all={_ll(f.mean_ll_all)} fitted_params={json.dumps(f.params)} "
                     f"param_grid={_grid_str(active[f.name].spec.param_grid) if f.name in active else '?'}")
        if am and am.own_code and src:
            lines.append(f"```python\n{src}\n```")
    if state.reference is not None:
        r = state.reference
        lines.append(f"- [reference, always active, not editable] smoothed table P(opponent next | last round) "
                     f"weight={r.weight:.2f} recent={_ll(r.mean_ll_recent)} all={_ll(r.mean_ll_all)}")
    lines.append(f"Posterior mixture (what the agent best-responds to): recent={_ll(state.mixture_mean_ll_recent)}")
    return "\n".join(lines)


def log_block(log: list[dict]) -> str:
    if not log:
        return "No earlier decisions."
    rows = []
    for d in log:
        ops = "; ".join(f"{o['op']} {o.get('name', '')}".strip() + ("" if o.get("ok", True) else f" (FAILED: {o.get('reason', '')[:120]})")
                        for o in d["ops"]) or "keep"
        rows.append(f"- after round t = {d['t'] - 1}: {ops}")
    return "\n".join(rows)


OPTION_SELECT = ("activate one or more MENU models by id (the prewritten programs above), optionally with your "
                 "own param_grid for them")
OPTION_CREATE = ("write one or more NEW models yourself (your own name, statement, param_grid and program, under the "
                 "program contract above)")


def task_block(mode: str, k_active: int, flip: bool) -> str:
    if mode == "select_only":
        how = f"- {OPTION_SELECT}."
    elif mode == "create_only":
        how = f"- {OPTION_CREATE}."
    else:
        opts = [OPTION_SELECT, OPTION_CREATE]
        if flip:
            opts.reverse()
        how = f"- {opts[0]};\n- {opts[1]}."
    pre = ("All menu models were active from the start; you may retire them, revise them, or add others.\n"
           if mode == "menu_all_create" else "")
    revision = ("You can also revise an active model's param_grid or retire it. Model source is fixed: "
                "do not create, mutate, or replace programs."
                if mode == "select_only" else
                "You can also revise an active model (new param_grid and/or new code; the name stays) or retire it.")
    return f"""\
Task: decide which models of the opponent are active from now on. You can
{how}
{revision}
{pre}Use the history and the fit of the active models to decide what the opponent is doing. Any mix of these actions
can be right; nothing is gained by activating models that do not fit, since every active model shares the prior mass.
At most {k_active} models can be active at once (the reference model does not count). At later decision points you
will see how your choice fitted and can change it.
param_grid: lists of candidate values, fitted by maximum likelihood over all rounds; at most {MAX_GRID_POINTS}
combinations per model."""


V6_OP_LINES = {"activate": '  {"op": "activate", "archive_id": "<archive id>"}',
               "mutate": '  {"op": "mutate", "parent": "<archive id>", "id": "<new name>", "statement": "<one sentence>", '
                         '"param_grid": {"<name>": [values]}}'}


def json_output_block(key: str, ops: list[str]) -> str:
    """Single-object envelope; operations and program semantics are unchanged."""
    return f'''Output format (json_v1). Return exactly ONE JSON object, without Markdown or surrounding text.
Required top-level fields:
- "{key}": an array of operation objects. An empty array is allowed.
- "code": an object mapping model names to complete Python source strings, or {{}} when no new code is needed.
- "rationale": a string explaining the evidence for your choices in two or three sentences.
The following are operation formats, NOT a list of operations to perform. Choose only the operations you want:
{chr(10).join(ops)}
Every create and mutate, and every revise with new_code=true, must have a matching entry in "code".
Use JSON string escaping for source newlines and quotes. Do not use ### Model headings or fenced code blocks.
For example, a code entry has the form "new_name": "def model(history, params, game):\\n    ...\\n".
Keep code at the top level under "code", not inside an operation. Names use letters, digits and underscores,
starting with a letter. param_grid values are non-empty numeric lists, with at most {MAX_GRID_POINTS} combinations.
Use only the menu IDs, active targets and archive IDs shown in the current state for the corresponding operations.
Do not put an individual operation, parameter grid, or model source at the root instead of the required envelope.'''


def output_block(mode: str, flip: bool, v6: V6Prompt | None = None, output_protocol: str = "text") -> str:
    """The select / create examples follow the option order of task_block (same flip)."""
    if mode == "select_only":
        return f'''Output format{" (json_v1)" if output_protocol == "json_v1" else ""}. Return exactly ONE JSON object:
{{"actions": [], "code": {{}}, "rationale": "<two or three sentences explaining the evidence>"}}
Choose any number of the following operations for "actions". These are formats, not a list to perform:
  {{"op": "select", "menu_id": "<menu id>", "param_grid": {{"<name>": [values]}}}}
  {{"op": "revise", "target": "<active name>", "param_grid": {{"<name>": [values]}}}}
  {{"op": "retire", "target": "<active name>"}}
  {{"op": "keep"}}
An empty actions array is allowed. param_grid is optional for select and required for revise.
Parameter grids contain non-empty numeric lists, with at most {MAX_GRID_POINTS} combinations per model.
Use only the menu IDs and active targets shown. Keep "code" empty; do not supply new_code=true,
Python source, Markdown fences, or operations that create, mutate, or activate archived programs.'''
    adds = [] if mode == "create_only" else ['  {"op": "select", "menu_id": "<menu id>", "param_grid": {"<name>": [values]}}']
    adds += ['  {"op": "create", "id": "<new name>", "statement": "<one sentence>", "param_grid": {"<name>": [values]}}']
    if flip:
        adds.reverse()
    if v6:
        adds += [V6_OP_LINES["activate"]] + ([V6_OP_LINES["mutate"]] if v6.heredity else [])
    ops = adds + ['  {"op": "revise", "target": "<active name>", "param_grid": {...}, "new_code": true}',
                  '  {"op": "retire", "target": "<active name>"}',
                  '  {"op": "keep"}']
    if output_protocol == "json_v1":
        return json_output_block("actions", ops) + ('\nparam_grid is optional for select and revise; new_code defaults to false. '
                                                    'A revise needs a new param_grid and/or new_code=true.')
    return f"""\
Output format. First one JSON object whose "actions" list holds any number of these entries (comma-separated):
```json
{{"actions": [
{("," + chr(10)).join(ops)}
 ],
 "rationale": "<two or three sentences: what in the history supports this decision>"}}
```
"param_grid" is optional for select (default grid) and for revise (keep the grid); "new_code" is optional (default
false). Names: letters, digits and underscores, starting with a letter. Then, for every create{", every mutate" if v6 and v6.heredity else ""} and every revise with
"new_code": true, the program:
### Model: <name>
```python
def model(history, params, game):
    ...
```"""


@dataclass
class V6Prompt:
    """Extra v6 material for a decision or variation prompt (None in v5 prompts, which stay byte-identical).
    heredity: the archive is shown with code and lineage and `mutate` is available; otherwise one line per entry."""
    heredity: bool
    archive: str
    failed: str = ""
    residuals: str = ""
    tables: str = ""


def _v6_sections(v6: V6Prompt) -> list[str]:
    out = [f"ARCHIVE (programs written so far that run, refit on all rounds; best first)\n{v6.archive}"]
    if v6.heredity and v6.failed:
        out.append(f"FAILED ATTEMPTS (programs rejected when loaded or run)\n{v6.failed}")
    if v6.residuals:
        out.append(f"RESIDUAL DIAGNOSTICS (where the current models predict worst)\n{v6.residuals}")
    if v6.tables:
        out.append(f"CONTEXT TABLES (empirical counts of the opponent's action by context, most informative first)\n{v6.tables}")
    return out


def v6_task_text(v6: V6Prompt, max_fresh: int) -> str:
    text = ("Archive: every program written so far that loads and runs is kept in the archive and refit on all rounds, "
            "also after it leaves the active set. You can activate an archive program by its id without rewriting it.")
    if v6.heredity:
        text += ("\nYou can mutate an archive program: write a new program derived from it (keep what fits, change what "
                 "the diagnostics show it gets wrong). While the archive is not empty, write at most "
                 f"{max_fresh} program(s) from scratch (create) per reply; improve the best archive programs with mutate.")
    return text


def build_messages(game: Game, menu: list[ProgramSpec], active: dict[str, ActiveModel], state: PromptState,
                   history: tuple, mode: str, history_mode: str, window: int, k_active: int, flip: bool,
                   v6: V6Prompt | None = None, max_fresh: int = 1, output_protocol: str = "text") -> list[dict]:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    sections = [
        f"GAME (fixed and fully known)\n{game_block(game)}",
        "SETUP\nThe opponent follows one fixed strategy for the whole match; which strategy, and its parameters, are "
        "unknown to you. You do not choose actions. You choose which opponent models are active. After every round "
        "each active model is scored by the exact likelihood of the opponent's actual action, its parameters are "
        "re-fitted over its param_grid by maximum likelihood, and the agent best-responds to the posterior mixture "
        "of the active models and the reference model.",
    ]
    if mode != "create_only":
        sections.append(f"MENU (prewritten models)\n{menu_block(menu)}")
    sections += [
        f"PROGRAM CONTRACT (menu models follow it too)\n{CONTRACT}",
        f"CURRENT STATE\n{state_block(state, game, active)}",
        *(_v6_sections(v6) if v6 else []),
        f"YOUR EARLIER DECISIONS\n{log_block(state.log)}",
        f"HISTORY\n{render_history(history, game, history_mode, window)}",
        task_block(mode, k_active, flip) + (f"\n{v6_task_text(v6, max_fresh)}"
                                          if v6 and mode != "select_only" else ""),
        output_block(mode, flip, v6, output_protocol),
    ]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(sections)}]


# ---------------------------------------------------------------- parsing and validation


@dataclass
class Op:
    op: str
    name: str = ""                     # menu id (select), new name (create), target (revise / retire)
    param_grid: dict | None = None
    statement: str = ""
    code: str | None = None
    parent: str | None = None          # mutate: the archive program it derives from


@dataclass(frozen=True)
class V6Rules:
    """What a v6 reply may do with the archive: activate any entry; mutate one if heredity is on; at most
    `max_fresh` creates per reply while the archive is not empty (heredity only)."""
    archive_ids: frozenset
    heredity: bool
    max_fresh: int = 1


class DecisionError(ValueError):
    pass


def _clean_grid(grid, where: str) -> dict | None:
    if grid is None:
        return None
    if not isinstance(grid, dict):
        raise DecisionError(f"{where}: param_grid must be an object mapping names to lists")
    out, n = {}, 1
    for k, v in grid.items():
        if not isinstance(v, list) or not v or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
            raise DecisionError(f"{where}: param_grid[{k!r}] must be a non-empty list of numbers")
        out[str(k)] = list(v)
        n *= len(v)
    if n > MAX_GRID_POINTS:
        raise DecisionError(f"{where}: param_grid has {n} combinations (> {MAX_GRID_POINTS})")
    return out


def _code_blocks(text: str) -> dict[str, str]:
    out = {}
    parts = re.split(r"^#{2,4}\s*Model\s*:?\s*", text, flags=re.M | re.I)
    for part in parts[1:]:
        name = part.strip().splitlines()[0].strip().strip("`*[]") if part.strip() else ""
        code = extract_code_blocks(part, "python")
        if name and code:
            out[name] = code[0]
    return out


def _new_name(a: dict, where: str, taken: set[str], codes: dict[str, str]) -> str:
    name = str(a.get("id", ""))
    if not NAME_RE.match(name):
        raise DecisionError(f"{where}: invalid name {name!r} (letters, digits, underscores; start with a letter)")
    if name in taken:
        raise DecisionError(f"{where}: name {name!r} is already used by a menu, active or archive model; pick a new name")
    if name not in codes:
        raise DecisionError(f"{where}: no '### Model: {name}' section with a ```python block")
    return name


def _check_fresh(ops: list[Op], rules: V6Rules | None) -> None:
    if rules and rules.heredity and rules.archive_ids:
        n = sum(o.op == "create" for o in ops)
        if n > rules.max_fresh:
            raise DecisionError(f"{n} programs written from scratch (create); while the archive is not empty at most "
                                f"{rules.max_fresh} is allowed per reply: derive the others from archive programs with mutate")


def _json_payload(text: str, key: str) -> tuple[dict, dict[str, str]]:
    try:
        obj = json.loads(text)
    except ValueError as e:
        raise DecisionError(f"root: expected one JSON object with {key}, code and rationale; {e}") from None
    if not isinstance(obj, dict) or not isinstance(obj.get(key), list):
        raise DecisionError(f"root.{key}: required array; return the complete {key}/code/rationale object")
    codes = obj.get("code")
    if not isinstance(codes, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in codes.items()):
        raise DecisionError("root.code: required object mapping model names to Python source strings ({} if none)")
    if not isinstance(obj.get("rationale"), str):
        raise DecisionError("root.rationale: required string")
    for i, action in enumerate(obj[key]):
        if not isinstance(action, dict):
            raise DecisionError(f"{key}[{i}]: expected an operation object")
        if "new_code" in action and not isinstance(action["new_code"], bool):
            raise DecisionError(f"{key}[{i}].new_code: expected a boolean")
        name = (action.get("id") if action.get("op") in ("create", "mutate") else
                action.get("target") if action.get("op") == "revise" and action.get("new_code") else None)
        if isinstance(name, str) and (name not in codes or not codes[name].strip()):
            raise DecisionError(f"code.{name}: required complete source for {key}[{i}]")
    return obj, codes


def parse_decision(text: str, mode: str, menu_ids: set[str], active: dict[str, ActiveModel], k_active: int,
                   rules: V6Rules | None = None, *, output_protocol: str = "text") -> tuple[list[Op], str]:
    """Parse and check a reply against the current state. Returns (ops, rationale); raises DecisionError with a
    message the LLM can act on. Ops are returned in application order: retire, revise, select, (activate, mutate,)
    create. `rules` (v6) adds the archive ops; without it the accepted replies and the messages are those of v5."""
    if output_protocol == "json_v1":
        obj, codes = _json_payload(text, "actions")
    else:
        head = re.split(r"^#{2,4}\s*Model\b", text, maxsplit=1, flags=re.M | re.I)[0]
        try:
            obj = extract_json(head)
        except ValueError:
            raise DecisionError("no JSON object with an \"actions\" list was found") from None
        codes = _code_blocks(text)
    if not isinstance(obj, dict) or not isinstance(obj.get("actions"), list):
        raise DecisionError("the JSON object must have an \"actions\" list")
    ops: list[Op] = []
    seen: set[str] = set()
    allowed = (("select", "revise", "retire", "keep") if mode == "select_only" else
               OPS if rules is None else OPS + ("activate",) + (("mutate",) if rules.heredity else ()))
    archive_ids = set(rules.archive_ids) if rules else set()
    for k, a in enumerate(obj["actions"]):
        where = f"actions[{k}]"
        if not isinstance(a, dict) or a.get("op") not in allowed:
            raise DecisionError(f"{where}: op must be one of {list(allowed)}")
        op = a["op"]
        if op == "keep":
            continue
        if op == "activate":
            name = str(a.get("archive_id", ""))
            if name not in archive_ids:
                raise DecisionError(f"{where}: unknown archive_id {name!r}; archive ids are {sorted(archive_ids)}")
            if name in active:
                raise DecisionError(f"{where}: archive program {name!r} is already active")
            ops.append(Op("activate", name))
        elif op == "mutate":
            parent = str(a.get("parent", ""))
            if parent not in archive_ids:
                raise DecisionError(f"{where}: mutate parent {parent!r} is not in the archive; archive ids are {sorted(archive_ids)}")
            name = _new_name(a, where, set(menu_ids) | set(active) | archive_ids, codes)
            ops.append(Op("mutate", name, _clean_grid(a.get("param_grid", {}), where), str(a.get("statement", ""))[:300],
                          codes[name], parent))
        elif op == "select":
            if mode == "create_only":
                raise DecisionError(f"{where}: select is not available (no menu)")
            name = str(a.get("menu_id", ""))
            if name not in menu_ids:
                raise DecisionError(f"{where}: unknown menu_id {name!r}; menu ids are {sorted(menu_ids)}")
            ops.append(Op("select", name, _clean_grid(a.get("param_grid"), where)))
        elif op == "create" and rules is not None:
            name = _new_name(a, where, set(menu_ids) | set(active) | archive_ids, codes)
            ops.append(Op("create", name, _clean_grid(a.get("param_grid", {}), where), str(a.get("statement", ""))[:300],
                          codes[name]))
        elif op == "create":
            name = str(a.get("id", ""))
            if not NAME_RE.match(name):
                raise DecisionError(f"{where}: invalid name {name!r} (letters, digits, underscores; start with a letter)")
            if name in menu_ids or name in active:
                raise DecisionError(f"{where}: name {name!r} is already used by a menu or active model; pick a new name")
            if name not in codes:
                raise DecisionError(f"{where}: no '### Model: {name}' section with a ```python block")
            grid = _clean_grid(a.get("param_grid", {}), where)
            ops.append(Op("create", name, grid, str(a.get("statement", ""))[:300], codes[name]))
        else:  # revise / retire
            name = str(a.get("target", ""))
            if name not in active:
                raise DecisionError(f"{where}: {op} target {name!r} is not active; active models are {sorted(active)}")
            if op == "retire":
                ops.append(Op("retire", name))
            else:
                new_code = bool(a.get("new_code", False))
                if new_code and mode == "select_only":
                    raise DecisionError(f"{where}: select_only permits parameter-grid revisions, not new code")
                if new_code and name not in codes:
                    raise DecisionError(f"{where}: new_code is true but there is no '### Model: {name}' python block")
                grid = _clean_grid(a.get("param_grid"), where)
                if grid is None and not new_code:
                    raise DecisionError(f"{where}: revise needs a param_grid or new_code")
                ops.append(Op("revise", name, grid, code=codes[name] if new_code else None))
        if ops and ops[-1].name in seen:
            raise DecisionError(f"{where}: more than one action on {ops[-1].name!r}")
        if ops:
            seen.add(ops[-1].name)
    for o in ops:
        if o.code is not None:
            errs = validate_source(o.code)
            if errs:
                raise DecisionError(f"program {o.name!r} breaks the sandbox rules: {errs[0][:300]}")
    _check_fresh(ops, rules)
    retired = {o.name for o in ops if o.op == "retire"}
    added = {o.name for o in ops if o.op in ("select", "create", "activate", "mutate") and o.name not in active}
    n_after = len(set(active) - retired) + len(added)
    if n_after > k_active:
        raise DecisionError(f"{n_after} models would be active, more than the limit {k_active}; retire some or add fewer")
    order = {"retire": 0, "revise": 1, "select": 2, "activate": 3, "mutate": 4, "create": 5}
    ops.sort(key=lambda o: order[o.op])
    return ops, str(obj.get("rationale", ""))[:1000]


def decision_validator(mode: str, menu_ids: set[str], active: dict[str, ActiveModel], k_active: int,
                       rules: V6Rules | None = None, *, output_protocol: str = "text"):
    def check(res: LLMResult) -> str | None:
        try:
            parse_decision(res.text, mode, menu_ids, active, k_active, rules, output_protocol=output_protocol)
        except DecisionError as e:
            return str(e)
        return None

    return check


# ---------------------------------------------------------------- v6 variation calls (programs only)


def build_variation_messages(game: Game, menu: list[ProgramSpec], active: dict[str, ActiveModel], state: PromptState,
                             history: tuple, v6: V6Prompt, n_programs: int, max_fresh: int = 1,
                             output_protocol: str = "text") -> list[dict]:
    """A variation call writes candidate programs only. Candidates are not activated: those that load and run enter
    the archive, and the decision call that follows chooses the active models from the menu and the archive."""
    if v6.heredity:
        how = (f"Prefer mutating the best archive programs: keep what fits and change what the residual diagnostics "
               f"show they get wrong. While the archive is not empty, at most {max_fresh} of the programs may be written "
               "from scratch (create).")
        ops = [V6_OP_LINES["mutate"],
               '  {"op": "create", "id": "<new name>", "statement": "<one sentence>", "param_grid": {"<name>": [values]}}']
    else:
        how = "Write each program from scratch (create)."
        ops = ['  {"op": "create", "id": "<new name>", "statement": "<one sentence>", "param_grid": {"<name>": [values]}}']
    sections = [
        f"GAME (fixed and fully known)\n{game_block(game)}",
        "SETUP\nThe opponent follows one fixed strategy for the whole match; which strategy, and its parameters, are "
        "unknown. Opponent models are scored by the exact likelihood of the opponent's actual actions, with parameters "
        "re-fitted over each model's param_grid by maximum likelihood. Your job in this step is only to write new "
        "candidate models. Every candidate that loads and runs is added to the archive and refit on all rounds; a later "
        "step chooses the active models from the menu and the archive.",
        f"MENU (prewritten models, for reference)\n{menu_block(menu)}",
        f"PROGRAM CONTRACT\n{CONTRACT}",
        f"CURRENT STATE\n{state_block(state, game, active)}",
        *_v6_sections(v6),
        f"HISTORY\n{render_history(history, game, 'full', len(history))}",
        f"Task: write up to {n_programs} new opponent models that would predict the opponent's actions better than the "
        f"current models and the archive. {how} Do not resubmit a program that is already in the menu or the archive.",
        f"""\
Output format. First one JSON object:
```json
{{"programs": [
{("," + chr(10)).join(ops)}
 ],
 "rationale": "<two or three sentences: what in the history or the diagnostics these programs capture>"}}
```
Names: letters, digits and underscores, starting with a letter, not used by a menu, active or archive model. Then, for
every program:
### Model: <name>
```python
def model(history, params, game):
    ...
```""",
    ]
    if output_protocol == "json_v1":
        sections[-1] = json_output_block("programs", ops)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(sections)}]


def parse_variation(text: str, taken: set[str], rules: V6Rules, n_programs: int,
                    *, output_protocol: str = "text") -> tuple[list[Op], str]:
    """Parse a variation reply into create / mutate ops (at most n_programs); raises DecisionError."""
    if output_protocol == "json_v1":
        obj, codes = _json_payload(text, "programs")
    else:
        head = re.split(r"^#{2,4}\s*Model\b", text, maxsplit=1, flags=re.M | re.I)[0]
        try:
            obj = extract_json(head)
        except ValueError:
            raise DecisionError("no JSON object with a \"programs\" list was found") from None
        codes = _code_blocks(text)
    if not isinstance(obj, dict) or not isinstance(obj.get("programs"), list):
        raise DecisionError("the JSON object must have a \"programs\" list")
    if len(obj["programs"]) > n_programs:
        raise DecisionError(f"{len(obj['programs'])} programs; at most {n_programs} are allowed")
    allowed = ("create", "mutate") if rules.heredity else ("create",)
    ops: list[Op] = []
    names: set[str] = set()
    for k, a in enumerate(obj["programs"]):
        where = f"programs[{k}]"
        if not isinstance(a, dict) or a.get("op") not in allowed:
            raise DecisionError(f"{where}: op must be one of {list(allowed)}")
        name = _new_name(a, where, set(taken) | set(rules.archive_ids) | names, codes)
        names.add(name)
        parent = None
        if a["op"] == "mutate":
            parent = str(a.get("parent", ""))
            if parent not in rules.archive_ids:
                raise DecisionError(f"{where}: mutate parent {parent!r} is not in the archive; archive ids are "
                                    f"{sorted(rules.archive_ids)}")
        errs = validate_source(codes[name])
        if errs:
            raise DecisionError(f"program {name!r} breaks the sandbox rules: {errs[0][:300]}")
        ops.append(Op(a["op"], name, _clean_grid(a.get("param_grid", {}), where), str(a.get("statement", ""))[:300],
                      codes[name], parent))
    _check_fresh(ops, rules)
    return ops, str(obj.get("rationale", ""))[:1000]


def variation_validator(taken: set[str], rules: V6Rules, n_programs: int, *, output_protocol: str = "text"):
    def check(res: LLMResult) -> str | None:
        try:
            parse_variation(res.text, taken, rules, n_programs, output_protocol=output_protocol)
        except DecisionError as e:
            return str(e)
        return None

    return check


# ---------------------------------------------------------------- v6 rendering of archive and diagnostics


def archive_text(entries: list, active_names: set[str], heredity: bool, top_code: int = 3) -> str:
    """entries: archive.Entry list, best first. With heredity the best `top_code` are shown with code and lineage."""
    if not entries:
        return "(empty)"
    lines = []
    for i, e in enumerate(entries):
        act = " [active]" if e.name in active_names else ""
        head = (f"- {e.name}{act}: mean log-lik per round {e.mean_ll():.3f}, score {e.score:.1f}, "
                f"fitted_params={json.dumps(e.h.params)}, param_grid={_grid_str(e.spec.param_grid)}")
        if heredity:
            head += f", from {e.op}{f' of {e.parent}' if e.parent else ''} at t={e.t}, generation {e.generation}"
        lines.append(head + f"\n  statement: {e.spec.statement}")
        if heredity and i < top_code:
            lines.append(f"```python\n{e.spec.src.strip()}\n```")
    return "\n".join(lines)


def failed_text(failed: list[dict], top: int = 6) -> str:
    rows = failed[-top:]
    return "\n".join(f"- t={f['t']} {f['op']} {f['name']}: {f['statement'][:120]} -> {f['reason'][:200]}" for f in rows)


def residual_text(mix_cells: list, best_name: str | None, best_cells: list, surprises: list[str]) -> str:
    out = ["Posterior mixture (what the agent best-responds to), cells with the largest excess log-loss "
           "(excess = how much better the cell's own empirical frequencies would have predicted those rounds):"]
    out += [f"  {c.render()}" for c in mix_cells] or ["  (none)"]
    if best_name is not None:
        out.append(f"Best archive program {best_name}, same measure:")
        out += [f"  {c.render()}" for c in best_cells] or ["  (none)"]
    if surprises:
        out.append("Rounds the mixture found most surprising:")
        out += [f"  {s}" for s in surprises]
    return "\n".join(out)
