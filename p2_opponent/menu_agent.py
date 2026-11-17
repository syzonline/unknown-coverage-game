"""v5 menu-choice agent: the LLM decides which opponent models are active; the posterior scores them exactly.

The agent is PosteriorBRAgent with the proposer replaced by scheduled decisions (rounds `decision_at`). At each one
the LLM may select menu models, create its own, revise or retire active ones (menu_prompt). Everything else is the
parent's: exact posterior over the active models plus the catch-all, best response to the posterior predictive.

Modes: select_create (nothing active at first, menu shown), create_only (no menu), menu_all_create (whole menu
active from the start, menu shown), select_only (menu selection and grid revisions, no program generation).
History shown to the LLM: the last `history_window` rounds or all of them.

v6 options (all off by default; off = the v5 agent, prompts byte-identical):
- archive: every program the LLM writes that runs is kept in a ProgramArchive, refit on all rounds at every
  decision, shown in the prompt and activatable (`activate`);
- heredity: the best archive programs are shown with code and lineage, `mutate` is available, and at most
  `max_fresh` programs per reply are written from scratch while the archive is not empty;
- residuals: residual diagnostics (diagnostics.py) and empirical context tables in the prompt;
- n_variation: before each decision, that many parallel variation calls write candidate programs into the archive;
- explore (parent option): information-seeking actions in fixed windows (explore.py).
"""
from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import numpy as np

from bsi.llm import LLM, BudgetExceeded, LLMCallFailed, LLMUnavailable

from .agents import PosteriorBRAgent, ProgramSpec
from .archive import ProgramArchive
from .diagnostics import conditional_tables, context_keys, residual_cells, surprising_rounds
from .explore import model_predictions
from .menu_prompt import (HISTORY_MODES, MODES, ActiveModel, DecisionError, ModelFit, Op, PromptState, V6Prompt,
                          V6Rules, archive_text, build_messages, build_variation_messages, decision_validator,
                          failed_text, parse_decision, parse_variation, residual_text, variation_validator)

# 5 of 7 points come after round 40, so a 40-round window and the full history differ at most of them
DEFAULT_DECISIONS = (10, 30, 50, 70, 90, 110, 130)
DEFAULT_K_ACTIVE = 3  # of the 5 menu models: activating nearly the whole menu is not an option
# v6 full arms: 10 points, 14 rounds apart (the exploration windows of explore.EXPLORE_STARTS start at the first 9)
V6_DECISIONS = (10, 24, 38, 52, 66, 80, 94, 108, 122, 136)


class MenuChoiceAgent(PosteriorBRAgent):
    name = "menu_choice"

    def __init__(self, pool, menu: Sequence[ProgramSpec], llm: LLM, model: str = "proposer_long",
                 mode: str = "select_create", history_mode: str = "full", history_window: int = 40,
                 decision_at: Sequence[int] = DEFAULT_DECISIONS, k_active: int = DEFAULT_K_ACTIVE,
                 temperature: float = 0.7, name: str | None = None, archive: bool = False, heredity: bool = False,
                 residuals: bool = False, n_variation: int = 0, archive_size: int = 8, max_fresh: int = 1,
                 programs_per_variation: int = 2, output_protocol: str = "text",
                 abort_on_api_failure: bool = False, **kw):
        if (heredity or n_variation) and not archive:
            raise ValueError("heredity and n_variation need archive=True")
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        if history_mode not in HISTORY_MODES:
            raise ValueError(f"unknown history mode {history_mode!r}")
        if output_protocol not in ("text", "json_v1"):
            raise ValueError(f"unknown output protocol {output_protocol!r}")
        self.menu = list(menu)
        if len({s.id for s in self.menu}) != len(self.menu):
            raise ValueError("menu ids must be unique")
        if mode == "menu_all_create":
            k_active += len(self.menu)  # the whole menu starts active; the cap counts models beyond it
        super().__init__(pool, self.menu if mode == "menu_all_create" else [], proposer=None, name=name, **kw)
        self.llm = llm
        self.model = model
        self.mode = mode
        self.history_mode = history_mode
        self.history_window = history_window
        self.decision_at = sorted({int(t) for t in decision_at})
        self.k_active = k_active
        self.temperature = temperature
        self.use_archive = archive
        self.heredity = heredity
        self.residuals = residuals
        self.n_variation = int(n_variation)
        self.archive_size = archive_size
        self.max_fresh = max_fresh
        self.programs_per_variation = programs_per_variation
        self.output_protocol = output_protocol
        self.abort_on_api_failure = abort_on_api_failure

    # -- episode state

    def reset(self, game, rng, episode_key: str = "") -> None:
        self.active: dict[str, ActiveModel] = {}
        self._pid: dict[str, str] = {}  # active name -> posterior id
        super().reset(game, rng, episode_key)
        # parent loaded the initial library (the whole menu in menu_all_create) with source "library"
        for pid, src in list(self._source.items()):
            spec = next(s for s in self.menu if pid.startswith(f"{s.id}#"))
            self._source[pid] = "menu"
            self.active[spec.id] = ActiveModel(spec.id, "menu", spec)
            self._pid[spec.id] = pid
        # option order of the task text: a per-episode coin, the same on every run of this episode
        self.flip = int(hashlib.sha1(f"flip:{episode_key}".encode()).hexdigest(), 16) % 2 == 1
        self.decision_log: list[dict] = []
        if getattr(self, "archive", None) is not None:
            self.archive.close()
        self.archive = ProgramArchive(self.pool, game.info(), self.archive_size, self.max_grid, self.max_call_ms,
                                      self.mdl_lambda) if self.use_archive else None
        self._preds: list[np.ndarray] = []  # posterior predictive of every round (residual diagnostics)

    def close(self) -> None:
        if getattr(self, "archive", None) is not None:
            self.archive.close()
        super().close()

    def act(self, history: tuple):
        action, pred = super().act(history)
        self._preds.append(np.asarray(pred, dtype=float))
        return action, pred

    def _explore_models(self, contexts: list[tuple]):
        """Exploration separates the posterior's models and the best archive programs that are not active. Weights:
        softmax of the fitted log-likelihood on all rounds (posterior programs, catch-all and archive alike)."""
        post = self.posterior
        hyps = post.hypotheses(include_other=False)
        lls = [post.ll_sum(h.id) for h in hyps]
        if self.archive is not None:
            active_src = {am.spec.src for am in self.active.values()}
            extra = [e for e in self.archive.ranked() if e.spec.src not in active_src][:3]
            hyps += [e.h for e in extra]
            lls += [e.ll for e in extra]
        other = post.other_id
        all_ll = np.array(lls + ([post.ll_sum(other)] if other else []), dtype=float)
        w = np.exp(all_ll - all_ll.max()) if len(all_ll) else all_ll
        return model_predictions(hyps, list(w[:len(hyps)]), contexts, self.pool,
                                 other=post.get(other) if other else None,
                                 other_weight=float(w[-1]) if other else 0.0)

    # -- decisions

    def observe(self, history: tuple, rnd: tuple[int, int]) -> None:
        super().observe(history, rnd)  # no proposer: update, refit, belief log
        t = len(history) + 1
        if t in self.decision_at:
            self._decide(history + (tuple(rnd),), t)

    def _fits(self) -> tuple[list[ModelFit], ModelFit | None, float]:
        post = self.posterior
        w = post.weights()
        n = max(1, len(post.segment()))

        def fit(name: str, origin: str, pid: str) -> ModelFit:
            h = post.get(pid)
            return ModelFit(name, origin, dict(getattr(h, "params", {}) or {}), w.get(pid, 0.0),
                            post.recent_mean_ll(pid), post.ll_sum(pid) / n)

        fits = [fit(nm, am.origin, self._pid[nm]) for nm, am in self.active.items() if self._pid.get(nm) in post]
        ref = fit("reference", "reference", post.other_id) if post.other_id else None
        recent = post.mix_ll[-post.window:]
        return fits, ref, float(np.mean(recent)) if recent else float("nan")

    def _state(self, t: int) -> PromptState:
        fits, ref, mix = self._fits()
        return PromptState(t=t, horizon=self.horizon or 0, decision_index=self.decision_at.index(t),
                           n_decisions=len(self.decision_at), fits=fits, reference=ref, mixture_mean_ll_recent=mix,
                           log=self.decision_log)

    def _messages(self, history: tuple, t: int, v6: V6Prompt | None = None) -> list[dict]:
        return build_messages(self.game, self.menu, self.active, self._state(t), history, self.mode,
                              self.history_mode, self.history_window, self.k_active, self.flip, v6, self.max_fresh,
                              self.output_protocol)

    def _chat(self, messages: list[dict], tag: str, validate, meter: dict | None = None):
        extra = {}
        started = time.monotonic()
        if self.output_protocol == "json_v1":
            def rejected(res, problem, attempt):
                self.events.append({"t": len(self.belief_log), "event": "llm_invalid", "tag": tag,
                                    "attempt": attempt, "problem": problem, "text": res.text,
                                    "finish_reason": res.finish_reason, "completion_tokens": res.completion_tokens,
                                    "latency_s": res.latency_s, "output_protocol": self.output_protocol})
            extra = {"response_format": {"type": "json_object"}, "on_invalid": rejected}
        try:
            res = self.llm.chat(messages, model=self.model, temperature=self.temperature, seed_tag=tag,
                                validate=validate, meter=self.llm_meter if meter is None else meter, **extra)
        except LLMCallFailed as e:
            if self.abort_on_api_failure:
                # Preserve the checkpoint: the matrix supervisor resumes the episode through the cache.
                raise LLMUnavailable(f"{tag}: {e}") from e
            raise
        meta = {"attempts": res.attempts, "finish_reason": res.finish_reason, "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens, "reasoning_chars": len(res.reasoning),
                "degraded": res.degraded, "cached": res.cached}
        if self.output_protocol == "json_v1":
            meta.update(output_protocol=self.output_protocol, repair_mode=res.repair_mode,
                        elapsed_s=round(time.monotonic() - started, 3))
        return res, meta

    def _menu_ids(self) -> set[str]:
        """Ids the LLM may select and may not reuse as names; none in create_only (it never sees the menu)."""
        return set() if self.mode == "create_only" else {s.id for s in self.menu}

    # -- v6: archive, diagnostics, variation calls

    def _rules(self) -> V6Rules | None:
        if self.archive is None:
            return None
        return V6Rules(frozenset(self.archive.entries), self.heredity, self.max_fresh)

    def _taken(self) -> set[str]:
        return self._menu_ids() | set(self.active) | set(self.archive.entries if self.archive else ())

    def _v6_prompt(self, history: tuple) -> V6Prompt | None:
        if self.archive is None:
            return None
        ranked = self.archive.ranked()
        residuals = tables = ""
        if self.residuals and history:
            keys = context_keys(self.game, history)
            mix = np.array(self._preds[:len(history)])
            best = ranked[0] if ranked else None
            best_cells = []
            if best is not None:
                raw = self.pool.prefixes(best.h.id, "model", history, [best.h.params], extra=(self.game.info(),),
                                         start=0, out_len=self.game.n_opp)[0, :len(history)]
                p = np.clip(raw, 0, None)
                best_cells = residual_cells(self.game, history, p / p.sum(-1, keepdims=True), top=5, keys=keys)
            residuals = residual_text(residual_cells(self.game, history, mix, top=8, keys=keys),
                                      best.name if best else None, best_cells,
                                      surprising_rounds(self.game, history, mix, top=5))
            tables = "\n".join(conditional_tables(self.game, history, top=4, keys=keys))
        return V6Prompt(self.heredity, archive_text(ranked, set(self.active), self.heredity),
                        failed_text(self.archive.failed), residuals, tables)

    def _archive_add(self, spec: ProgramSpec, history: tuple, op: str, parent: str | None, call: str) -> tuple[bool, str]:
        """Insert into the archive (renamed if the name is taken, e.g. by a parallel variation call) and log it with
        its source, for offline replay (scripts/program_diag.py)."""
        name, k = spec.id, 1
        while name in self._taken():
            k += 1
            name = f"{spec.id}_{k}"
        spec = ProgramSpec(name, spec.src, spec.param_grid, spec.statement)
        protect = {am.name for am in self.active.values()}
        ok, reason, e = self.archive.add(spec, history, op, parent, call, protect)
        ev = {"t": len(history), "event": "archive_add", "name": name, "op": op, "parent": parent, "call": call,
              "ok": ok, "reason": reason[:300], "statement": spec.statement, "src": spec.src[:4096],
              "param_grid": spec.param_grid}
        if e is not None:
            ev.update(mean_ll=round(e.mean_ll(), 5), gen=e.generation, sim=e.similarity,
                      parent_mean_ll=round(self.archive.entries[parent].mean_ll(), 5)
                      if parent in self.archive.entries else None)
        self.events.append(ev)
        return ok, (reason or name)

    def _variation(self, history: tuple, t: int) -> None:
        """n_variation parallel calls write candidate programs; they enter the archive in call order."""
        v6, rules, taken = self._v6_prompt(history), self._rules(), self._taken()
        msgs = build_variation_messages(self.game, self.menu, self.active, self._state(t), history, v6,
                                        self.programs_per_variation, self.max_fresh, self.output_protocol)
        meters = [{} for _ in range(self.n_variation)]

        def call(k: int):
            try:
                res, meta = self._chat(msgs, f"menu:{self.name}:{self.episode_key}:t{t}:v{k}",
                                       variation_validator(taken, rules, self.programs_per_variation,
                                                           output_protocol=self.output_protocol), meters[k])
                return res, meta, parse_variation(res.text, taken, rules, self.programs_per_variation,
                                                  output_protocol=self.output_protocol)
            except (LLMUnavailable, BudgetExceeded):
                raise
            except Exception as e:  # noqa: BLE001 - an unusable variation reply adds nothing
                return None, {}, e

        with ThreadPoolExecutor(max_workers=self.n_variation) as ex:
            outs = list(ex.map(call, range(self.n_variation)))
        for m in meters:
            for key, v in m.items():
                self.llm_meter[key] = self.llm_meter.get(key, 0) + v
        for k, (res, meta, parsed) in enumerate(outs):
            ev = {"t": t, "event": "variation", "k": k, **meta}
            if self.output_protocol == "json_v1":
                ev["output_protocol"] = self.output_protocol
            if isinstance(parsed, Exception):
                ev["error"] = f"{type(parsed).__name__}: {str(parsed)[:300]}"
            else:
                ops, rationale = parsed
                results = []
                for o in ops:
                    parent = o.parent if o.parent in self.archive.entries else None
                    ok, info = self._archive_add(ProgramSpec(o.name, o.code, o.param_grid or {}, o.statement), history,
                                                 o.op, parent, f"variation:{k}")
                    results.append({"op": o.op, "name": info if ok else o.name, "parent": o.parent, "ok": ok,
                                    "reason": "" if ok else info[:200]})
                ev.update(ops=results, rationale=rationale)
            self.events.append(ev)

    def _decide(self, history: tuple, t: int) -> None:
        menu_ids = self._menu_ids()
        if self.archive is not None:
            gone = self.archive.rescore(history)
            if gone:
                self.events.append({"t": t, "event": "archive_drop", "names": gone})
            if self.n_variation and self.mode != "select_only":
                self._variation(history, t)
        rules = self._rules()
        msgs = self._messages(history, t, self._v6_prompt(history))
        ev = {"t": t, "event": "decision", "mode": self.mode, "history_mode": self.history_mode, "flip": self.flip,
              "menu": sorted(s.id for s in self.menu)}
        if self.output_protocol == "json_v1":
            ev["output_protocol"] = self.output_protocol
        try:
            # the method name keeps arms with identical prompts (window vs full before round 40) from sharing cached
            # replies: concurrent misses made that sharing depend on scheduling, and replays irreproducible
            res, meta = self._chat(msgs, f"menu:{self.name}:{self.episode_key}:t{t}",
                                   decision_validator(self.mode, menu_ids, self.active, self.k_active, rules,
                                                      output_protocol=self.output_protocol))
            ops, rationale = parse_decision(res.text, self.mode, menu_ids, self.active, self.k_active, rules,
                                            output_protocol=self.output_protocol)
        except (LLMUnavailable, BudgetExceeded):
            raise  # fatal for the whole run
        except Exception as e:  # noqa: BLE001 - the episode survives an unusable reply: nothing changes
            ev.update(error=f"{type(e).__name__}: {str(e)[:500]}", ops=[], active=self._active_view())
            self.events.append(ev)
            self.decision_log.append({"t": t, "ops": [{"op": "keep", "name": "", "ok": False, "reason": "unusable reply"}]})
            return
        results = self._apply(ops, history)
        failed = [r for r in results if not r["ok"] and r["op"] in ("create", "revise", "mutate")]
        repair = None
        if failed:
            repair = self._repair(msgs, res.text, failed, history, t)
            results += repair.pop("results", [])
        ev.update(ops=results, rationale=rationale, active=self._active_view(), **meta)
        if repair:
            ev["repair"] = repair
        if self.archive is not None:
            ev["archive"] = self.archive.snapshot()
        self.events.append(ev)
        self.decision_log.append({"t": t, "ops": [{k: r[k] for k in ("op", "name", "ok", "reason")} for r in results]})

    def _repair(self, msgs: list[dict], reply: str, failed: list[dict], history: tuple, t: int) -> dict:
        """One follow-up turn: the programs that failed to load or run are reported back, the LLM may resubmit them."""
        names = {r["name"] for r in failed}
        detail = "\n".join(f"- {r['name']}: {r['reason'][:400]}" for r in failed)
        resubmit = ("Resubmit parameter-grid revisions only or " if self.mode == "select_only" else
                    f"Resubmit them (create{' or mutate' if self.heredity else ''} with the same name, or revise) or ")
        follow = msgs + [{"role": "assistant", "content": reply if self.output_protocol == "json_v1" else reply[:8000]},
                         {"role": "user", "content": f"These programs were rejected when loaded or run on the history:\n"
                                                     f"{detail}\n{resubmit}"
                                                     "leave them out. Same output format; only actions on these names."}]
        menu_ids = self._menu_ids()
        rules = self._rules()

        def only_failed(r):
            try:
                ops, _ = parse_decision(r.text, self.mode, menu_ids, self.active, self.k_active, rules,
                                       output_protocol=self.output_protocol)
            except DecisionError as e:
                return str(e)
            bad = [o.name for o in ops if o.name not in names]
            return f"actions on other models are not allowed here: {bad}" if bad else None

        try:
            res, meta = self._chat(follow, f"menu:{self.name}:{self.episode_key}:t{t}:fix", only_failed)
            ops, _ = parse_decision(res.text, self.mode, menu_ids, self.active, self.k_active, rules,
                                   output_protocol=self.output_protocol)
        except (LLMUnavailable, BudgetExceeded):
            raise
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        return {**meta, "results": [{**r, "repair": True} for r in self._apply(ops, history)]}

    # -- applying operations

    def _active_view(self) -> list[dict]:
        return [{"name": n, "origin": a.origin, "revisions": a.revisions, "pid": self._pid.get(n)} for n, a in self.active.items()]

    def _remove(self, name: str, why: str) -> None:
        pid = self._pid.pop(name, None)
        self.active.pop(name, None)
        if pid is not None:
            self._drop([pid], why)

    def _drop(self, ids, why: str) -> None:
        """Also forget the active names of dropped hypotheses (the parent drops programs that fail in act/observe)."""
        gone = set(ids)
        for name in [n for n, pid in self._pid.items() if pid in gone]:
            self._pid.pop(name)
            self.active.pop(name, None)
        super()._drop(ids, why)

    def _add(self, name: str, origin: str, spec: ProgramSpec, history: tuple, revisions: int = 0,
             own_code: bool = False, source: str | None = None) -> tuple[bool, str, dict]:
        """Returns (ok, reason, add event). Posterior source: "menu" for menu code (any grid), "created" for the
        LLM's own code, "revised" for an LLM revision (new code of a menu model, or any revision of its own);
        v6: "mutated" (a mutate op) and "archived" (an archive program activated as it is)."""
        source = source or ("menu" if not own_code else ("revised" if revisions else "created"))
        ok, reason = self.try_add(ProgramSpec(name, spec.src, spec.param_grid, spec.statement), history, source=source)
        add = self.events[-1]
        if ok:
            self.active[name] = ActiveModel(name, origin, spec, revisions, own_code)
            self._pid[name] = add["id"]
        return ok, reason, add

    def _apply(self, ops: list[Op], history: tuple) -> list[dict]:
        menu = {s.id: s for s in self.menu}
        out = []
        for o in ops:
            r = {"op": o.op, "name": o.name, "ok": True, "reason": ""}
            if o.op == "retire":
                self._remove(o.name, "retired by the LLM")
            elif o.op == "select":
                base = menu[o.name]
                spec = ProgramSpec(base.id, base.src, o.param_grid if o.param_grid is not None else base.param_grid, base.statement)
                prev = (self.active.pop(o.name), self._pid.pop(o.name)) if o.name in self.active else None
                r["ok"], r["reason"], add = self._add(o.name, "menu", spec, history)
                if prev is not None:  # re-selecting an active menu model sets a new grid; on failure the old one stays
                    if r["ok"]:
                        self._drop([prev[1]], "re-selected")
                    else:
                        self.active[o.name], self._pid[o.name] = prev
                r["grid_changed"] = o.param_grid is not None and o.param_grid != base.param_grid
            elif o.op == "activate":
                e = self.archive.entries[o.name]
                r["ok"], r["reason"], add = self._add(o.name, "created", e.spec, history, own_code=True, source="archived")
            elif o.op == "mutate" or (o.op == "create" and self.archive is not None):
                # v6: the program enters the archive first (all checks, fit on all rounds), then becomes active
                spec = ProgramSpec(o.name, o.code, o.param_grid or {}, o.statement)
                parent = o.parent if o.parent in self.archive.entries else None
                ok, info = self._archive_add(spec, history, o.op, parent, "decision")
                if ok:
                    r["name"] = info
                    spec = self.archive.entries[info].spec
                    r["ok"], r["reason"], add = self._add(info, "created", spec, history, own_code=True,
                                                          source="mutated" if o.op == "mutate" else "created")
                else:
                    r.update(ok=False, reason=info)
                r["parent"] = o.parent
            elif o.op == "create":
                r["ok"], r["reason"], add = self._add(o.name, "created", ProgramSpec(o.name, o.code, o.param_grid or {}, o.statement),
                                                      history, own_code=True)
            else:  # revise: replace the program and/or its grid, keep the name; on failure the old version stays
                old = self.active[o.name]
                spec = ProgramSpec(o.name, o.code if o.code is not None else old.spec.src,
                                   o.param_grid if o.param_grid is not None else old.spec.param_grid, old.spec.statement)
                old_pid = self._pid[o.name]
                self.active.pop(o.name)
                self._pid.pop(o.name)
                ok, reason, add = self._add(o.name, old.origin, spec, history, revisions=old.revisions + 1,
                                            own_code=old.own_code or o.code is not None)
                if ok:
                    self._drop([old_pid], "revised")
                    if self.archive is not None and o.code is not None:  # new code joins the archive too
                        self._archive_add(ProgramSpec(f"{o.name}_r{old.revisions + 1}", spec.src, spec.param_grid,
                                                      spec.statement), history, "revise",
                                          o.name if o.name in self.archive.entries else None, "decision")
                else:
                    self.active[o.name], self._pid[o.name] = old, old_pid
                r.update(ok=ok, reason=reason, origin=old.origin, new_code=o.code is not None)
            if r["ok"] and o.op in ("select", "create", "revise", "activate", "mutate"):  # `add` is this op's add event, not a later drop
                r["ll_gain_vs_other"] = add.get("ll_gain_vs_other")
                r["pid"] = add["id"]
            out.append(r)
        return out

    def _log_belief(self) -> None:
        super()._log_belief()
        w = self.posterior.weights()
        by: dict[str, float] = {}
        for pid, src in self._source.items():
            if pid in w:
                by[src] = by.get(src, 0.0) + w[pid]
        self.belief_log[-1]["by_src"] = {k: round(v, 5) for k, v in by.items()}
        self.belief_log[-1]["active"] = sorted(getattr(self, "active", {}))
