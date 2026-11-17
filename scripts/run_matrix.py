"""Repeated-game experiment matrix: methods × games × opponent families × parameters × seeds.

    python scripts/run_matrix.py --config configs/intern_full.yaml            # run / resume
    python scripts/run_matrix.py --config configs/intern_full.yaml --dry-run  # job counts
    python scripts/analyze_matrix.py runs/matrix/intern_full                  # per-run report

Design
- Pairing: every method sees the same bot (seeded by game|family|param) and the same environment RNG
  (seeded by game|family|param|seed), so comparisons are paired.
- Resume: finished jobs are lines in summaries.jsonl. Re-running the same command skips them. Changing
  the config of an existing run directory is refused unless --force-config is given.
- Health: roles are health-checked before starting. During the run, LLM failures are retried and repaired
  inside bsi.llm. An agent whose call still fails takes a seeded fallback, and episodes with
  fallback_rate > max_fallback_rate are not accepted (logged to errors.jsonl, retried on the next run, and
  abandoned after max_job_attempts). LLMUnavailable or BudgetExceeded stop the run cleanly (exit code 3);
  re-run later to resume. With api_failure_scope: episode, LLMUnavailable instead defers only that job;
  the queue drains before exit 3, and API failures do not consume max_job_attempts.
- Concurrency: one process, threads. Jobs that need the sandbox run in a small lane, each thread with its
  own single-worker SandboxPool to bound memory. LLM-only jobs run in a wider lane. The RPM limiter is
  per process: concurrent matrices must explicitly partition the provider's RPM/TPM allowances.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
import traceback
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bsi.config import load_api_config  # noqa: E402
from bsi.llm import LLM, BudgetExceeded, LLMUnavailable  # noqa: E402
from bsi.sandbox import SandboxPool  # noqa: E402
from bsi.store import TraceWriter, config_hash, dumps  # noqa: E402
from p2_opponent.agents import FrequencyBRAgent, OracleBRAgent, PosteriorBRAgent, RandomAgent  # noqa: E402
from p2_opponent.bots import sample_bot  # noqa: E402
from p2_opponent.games import make_game  # noqa: E402
from p2_opponent.menu_agent import DEFAULT_DECISIONS, DEFAULT_K_ACTIVE, MenuChoiceAgent  # noqa: E402
from p2_opponent.menu_prompt import HISTORY_MODES, MODES as MENU_MODES  # noqa: E402
from p2_opponent.llm_agents import (DirectLLMAgent, OracleTypeLLMAgent, PosteriorLLMDecideAgent,  # noqa: E402
                                    VerbalBeliefAgent)
from p2_opponent.propose import (generate_initial_library, load_library, make_llm_proposer,  # noqa: E402
                                 make_oracle_proposer, make_template_proposer, oracle_program_ids, oracle_specs,  # noqa: E402
                                 save_library)
from p2_opponent.runner import run_episode  # noqa: E402
from p2_opponent.seed_programs import MINIMAL_LIBRARY, SEED_LIBRARY, TEMPLATE_MENUS  # noqa: E402

SANDBOX_METHODS = {"bsi_seedlib", "b2_static", "bsi", "bsi_minimal", "b3_llm_decide", "menu_choice"}
PROPOSER_METHODS = {"bsi", "bsi_minimal", "b3_llm_decide"}
PER_ROUND_LLM_METHODS = {"b0_llm_direct", "b1_verbal", "b3_llm_decide", "b4_oracle_type"}
ALL_METHODS = ["random", "freq_br", "oracle_br", "bsi_seedlib", "b2_static", "bsi", "bsi_minimal",
               "b3_llm_decide", "b0_llm_direct", "b1_verbal", "b4_oracle_type", "menu_choice"]
# Method variants: `name: {base: <one of ALL_METHODS>, <option>: ...}` in the config. Options passed to the agent:
AGENT_OPTIONS = ("min_obs_propose", "changepoint", "plan_depth", "propose_at", "explore")
# plus `library` (k0 | seed | minimal), `library_exclude: {game: [program ids]}` (drop programs from that library),
# `oracle_proposer: true` (hand-written true-class programs, no LLM), `template_proposer` (periodic | full: a
# fixed menu of generic programs at the first trigger, no LLM),
# `proposer_evidence` (lag1 | rich: what the LLM proposer is shown) and `cells: [[game, family], ...]`
# (run only these game×family cells instead of games × families).
# v5: `library_add_oracle: true` (the true class's programs join the library at t=0, no LLM; a ceiling) and
# `menu_exclude_true_family: true` (drop the true class's programs from the library / menu). Base `menu_choice`
# (the LLM decides which models are active; its library is the menu) takes MENU_OPTIONS.
# v6: `explore: {rate: ...}` (any posterior agent; explore.ExplorePolicy) and the menu_choice options archive,
# heredity, residuals, n_variation, archive_size, max_fresh, programs_per_variation (menu_agent.MenuChoiceAgent).
LIBRARIES = ("k0", "seed", "minimal")
LIBRARY_DEFAULT = {"bsi_seedlib": "seed", "b2_static": "k0", "bsi": "k0", "bsi_minimal": "minimal", "b3_llm_decide": "k0",
                   "menu_choice": "seed"}
MENU_OPTIONS = ("menu_mode", "history_mode", "history_window", "decision_at", "k_active", "archive", "heredity",
                "residuals", "n_variation", "archive_size", "max_fresh", "programs_per_variation", "output_protocol",
                "abort_on_api_failure")


def method_opts(cfg: dict, m: str) -> dict:
    return cfg["methods"].get(m) or {}


def resolve_base(cfg: dict, m: str) -> str:
    return method_opts(cfg, m).get("base", m)


def library_kind(cfg: dict, m: str) -> str | None:
    return method_opts(cfg, m).get("library", LIBRARY_DEFAULT.get(resolve_base(cfg, m)))


def uses_llm_proposer(cfg: dict, m: str) -> bool:
    opt = method_opts(cfg, m)
    return resolve_base(cfg, m) in PROPOSER_METHODS and not (opt.get("oracle_proposer") or opt.get("template_proposer"))


def uses_llm(cfg: dict, m: str) -> bool:
    return resolve_base(cfg, m) in PER_ROUND_LLM_METHODS or uses_llm_proposer(cfg, m) or resolve_base(cfg, m) == "menu_choice"


def decision_rounds(cfg: dict, m: str) -> list[int]:
    return list(method_opts(cfg, m).get("decision_at", DEFAULT_DECISIONS))


def config_problems(cfg: dict, methods: list[str]) -> list[str]:
    """Option values that would otherwise fail per job at run time, or silently do nothing."""
    out = []
    for m in methods:
        opt, base = method_opts(cfg, m), resolve_base(cfg, m)
        kind = library_kind(cfg, m)
        if base in LIBRARY_DEFAULT and kind not in LIBRARIES:
            out.append(f"{m}: library {kind!r} is not one of {sorted(LIBRARIES)}")
        if base == "b3_llm_decide" and opt.get("plan_depth", 0) > 1:
            out.append(f"{m}: plan_depth has no effect on b3 (the LLM chooses the action)")
        if opt.get("proposer_evidence", "lag1") not in ("lag1", "rich"):
            out.append(f"{m}: proposer_evidence must be lag1 or rich")
        tpl = opt.get("template_proposer")
        if tpl is not None and tpl not in TEMPLATE_MENUS:
            out.append(f"{m}: template_proposer must be one of {sorted(TEMPLATE_MENUS)}")
        if tpl is not None and opt.get("oracle_proposer"):
            out.append(f"{m}: template_proposer and oracle_proposer are exclusive")
        if (tpl is not None or opt.get("oracle_proposer")) and base not in PROPOSER_METHODS:
            out.append(f"{m}: a proposer option needs a base with a proposer ({sorted(PROPOSER_METHODS)})")
        excl = opt.get("library_exclude")
        if excl is not None:
            if not isinstance(excl, dict) or not all(isinstance(v, list) for v in excl.values()):
                out.append(f"{m}: library_exclude must map game -> list of program ids")
            elif base not in LIBRARY_DEFAULT:
                out.append(f"{m}: library_exclude needs a base with a program library ({sorted(LIBRARY_DEFAULT)})")
            elif set(excl) - set(cfg.get("games", [])):
                out.append(f"{m}: library_exclude names games not in the config: {sorted(set(excl) - set(cfg.get('games', [])))}")
        for k in ("library_add_oracle", "menu_exclude_true_family"):
            if opt.get(k) and base not in LIBRARY_DEFAULT:
                out.append(f"{m}: {k} needs a base with a program library ({sorted(LIBRARY_DEFAULT)})")
        if opt.get("menu_exclude_true_family") and base in LIBRARY_DEFAULT and library_kind(cfg, m) != "seed":
            # the true-class ids are seed-library ids; in another library the exclusion would silently drop nothing
            out.append(f"{m}: menu_exclude_true_family needs library seed (got {library_kind(cfg, m)!r})")
        if opt.get("library_add_oracle") and opt.get("menu_exclude_true_family"):
            out.append(f"{m}: library_add_oracle and menu_exclude_true_family are exclusive")
        ex = opt.get("explore")
        if ex is not None:
            if not isinstance(ex, dict) or not isinstance(ex.get("rate"), (int, float)) or not 0 <= ex["rate"] <= 1:
                out.append(f"{m}: explore must be a mapping with rate in [0, 1]")
            elif base not in LIBRARY_DEFAULT:
                out.append(f"{m}: explore needs a posterior agent ({sorted(LIBRARY_DEFAULT)})")
        menu_opts = [k for k in MENU_OPTIONS if k in opt]
        if base != "menu_choice":
            if menu_opts:
                out.append(f"{m}: {menu_opts} need base menu_choice")
            continue
        if opt.get("menu_mode", "select_create") not in MENU_MODES:
            out.append(f"{m}: menu_mode must be one of {list(MENU_MODES)}")
        if opt.get("history_mode", "full") not in HISTORY_MODES:
            out.append(f"{m}: history_mode must be one of {list(HISTORY_MODES)}")
        if opt.get("output_protocol", "text") not in ("text", "json_v1"):
            out.append(f"{m}: output_protocol must be text or json_v1")
        w = opt.get("history_window", 40)
        if not isinstance(w, int) or w < 1:
            out.append(f"{m}: history_window must be a positive integer")
        k = opt.get("k_active", DEFAULT_K_ACTIVE)
        if not isinstance(k, int) or k < 1:
            out.append(f"{m}: k_active must be a positive integer")
        if (opt.get("heredity") or opt.get("n_variation")) and not opt.get("archive"):
            out.append(f"{m}: heredity and n_variation need archive: true")
        for key in ("n_variation", "max_fresh"):
            v = opt.get(key, 0)
            if not isinstance(v, int) or v < 0:
                out.append(f"{m}: {key} must be a non-negative integer")
        for key in ("archive_size", "programs_per_variation"):
            v = opt.get(key, 1)
            if not isinstance(v, int) or v < 1:
                out.append(f"{m}: {key} must be a positive integer")
        d = decision_rounds(cfg, m)
        if not d or not all(isinstance(t, int) and 1 <= t < cfg.get("horizon", 10**9) for t in d):
            out.append(f"{m}: decision_at must be rounds in [1, horizon)")
    return out


def crc(s: str) -> int:
    return zlib.crc32(s.encode()) % 2**31


@dataclass(frozen=True)
class Job:
    method: str
    game: str
    family: str
    param: int
    seed: int

    @property
    def episode(self) -> str:
        return f"{self.game}|{self.family}|p{self.param}|s{self.seed}"

    @property
    def key(self) -> str:
        return f"{self.method}|{self.episode}"


def build_jobs(cfg: dict, methods: list[str]) -> list[Job]:
    jobs = []
    for m in methods:
        opt = method_opts(cfg, m)
        params = opt.get("params", list(range(cfg["n_params"])))
        seeds = opt.get("seeds", cfg["seeds"])
        cells = opt.get("cells") or [(g, f) for g in opt.get("games", cfg["games"])
                                     for f in opt.get("families", cfg["families"])]
        for g, f in cells:
            for p in params:
                for s in seeds:
                    jobs.append(Job(m, g, f, p, s))
    if "job_keys" in cfg:
        selected = set(cfg["job_keys"])
        jobs = [j for j in jobs if j.key in selected]
    return jobs


def make_bot(cfg: dict, job: Job):
    game = make_game(job.game)
    rng = np.random.default_rng(crc(f"{job.game}|{job.family}|{job.param}|{cfg['split']}"))
    return game, sample_bot(game, job.family, cfg["split"], rng, cfg["horizon"])


class Matrix:
    def __init__(self, cfg: dict, run_dir: Path, llm: LLM | None):
        self.cfg = cfg
        self.run_dir = run_dir
        self.llm = llm
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.fatal: str | None = None
        self.tls = threading.local()
        self.pools: list[SandboxPool] = []
        self.libraries: dict[str, list] = {}
        self.summaries = open(run_dir / "summaries.jsonl", "a", encoding="utf-8")
        self.errors = open(run_dir / "errors.jsonl", "a", encoding="utf-8")
        self.traces = TraceWriter(run_dir / f"traces-{time.strftime('%Y%m%d-%H%M%S')}.jsonl.gz")
        self.n_done = 0
        self.api_failed: set[str] = set()
        self.t0 = time.time()

    # -- shared resources

    def pool(self) -> SandboxPool:
        p = getattr(self.tls, "pool", None)
        if p is None:
            p = SandboxPool(n_workers=1, op_timeout_cap=self.cfg.get("sandbox_op_timeout_s", 60))
            self.tls.pool = p
            with self.lock:
                self.pools.append(p)
        return p

    def ensure_libraries(self, games: list[str]) -> None:
        k0 = self.cfg.get("k0", 8)
        role = self.cfg.get("proposer", "proposer")
        with SandboxPool(n_workers=1) as pool:
            for g in games:
                path = self.run_dir / "libraries" / f"{g}.json"
                if not path.exists():
                    print(f"[library] generating K0={k0} initial programs for {g} ...", flush=True)
                    specs, log = generate_initial_library(self.llm, make_game(g), pool, k0=k0, model=role)
                    save_library(path, specs, log)
                    print(f"[library] {g}: {len(specs)} accepted / {len(log)} proposed -> {path}", flush=True)
                self.libraries[g] = load_library(path)
                if not self.libraries[g]:
                    raise RuntimeError(f"initial library for {g} is empty; see {path}")

    def _full_library(self, method: str, game: str) -> list:
        kind = library_kind(self.cfg, method)
        return {"seed": SEED_LIBRARY, "minimal": MINIMAL_LIBRARY, "k0": None}[kind] or self.libraries[game]

    def library(self, job: Job) -> list:
        drop = set((method_opts(self.cfg, job.method).get("library_exclude") or {}).get(job.game, []))
        return [s for s in self._full_library(job.method, job.game) if s.id not in drop]

    def check_exclusions(self, pending: list[Job]) -> None:
        """Every excluded id must exist in its library, so a typo cannot silently leave the library unchanged.
        Checked for the (method, game) pairs about to run, whose libraries are loaded."""
        for m, g in sorted({(j.method, j.game) for j in pending}):
            ids = (method_opts(self.cfg, m).get("library_exclude") or {}).get(g)
            if ids:
                lib = {s.id for s in self._full_library(m, g)}
                kind = library_kind(self.cfg, m)
                if set(ids) - lib:
                    raise SystemExit(f"{m}: library_exclude ids not in the {kind} library of {g}: {sorted(set(ids) - lib)}")
                if not lib - set(ids):
                    raise SystemExit(f"{m}: library_exclude removes every program of the {kind} library of {g}")

    def library_for(self, job: Job, bot) -> list:
        """library(job) plus the per-bot v5 options: drop the true class's programs, or add them (ceiling)."""
        opt = method_opts(self.cfg, job.method)
        lib = self.library(job)
        if opt.get("menu_exclude_true_family"):
            drop = set(oracle_program_ids(bot.describe()))
            lib = [s for s in lib if s.id not in drop]
        if opt.get("library_add_oracle"):
            lib = lib + oracle_specs(bot.describe(), bot.game.n_opp)
        return lib

    def make_agent(self, job: Job, bot):
        opt = method_opts(self.cfg, job.method)
        decider = opt.get("decider", self.cfg.get("decider", "decider"))
        proposer_role = opt.get("proposer", self.cfg.get("proposer", "proposer"))
        post_kw = {"propose_cooldown": opt.get("propose_cooldown", self.cfg.get("propose_cooldown", 20)),
                   **{k: opt[k] for k in AGENT_OPTIONS if k in opt}}

        def proposer():
            if opt.get("oracle_proposer"):
                return make_oracle_proposer(bot.describe())
            if opt.get("template_proposer"):
                return make_template_proposer(opt["template_proposer"])
            return make_llm_proposer(self.llm, model=proposer_role, evidence=opt.get("proposer_evidence", "lag1"))

        m = resolve_base(self.cfg, job.method)
        name = job.method
        if m == "random":
            return RandomAgent()
        if m == "freq_br":
            return FrequencyBRAgent()
        if m == "oracle_br":
            return OracleBRAgent(bot, plan_depth=opt.get("plan_depth", 0))
        if m in ("bsi_seedlib", "b2_static"):
            return PosteriorBRAgent(self.pool(), self.library_for(job, bot), name=name, **post_kw)
        if m in ("bsi", "bsi_minimal"):
            return PosteriorBRAgent(self.pool(), self.library_for(job, bot), proposer=proposer(), name=name, **post_kw)
        if m == "menu_choice":
            kw = {k: opt[k] for k in ("min_obs_propose", "changepoint", "plan_depth", "explore", "archive", "heredity",
                                      "residuals", "n_variation", "archive_size", "max_fresh", "programs_per_variation",
                                      "output_protocol", "abort_on_api_failure")
                  if k in opt}
            return MenuChoiceAgent(self.pool(), self.library_for(job, bot), llm=self.llm, model=proposer_role,
                                   mode=opt.get("menu_mode", "select_create"), history_mode=opt.get("history_mode", "full"),
                                   history_window=opt.get("history_window", 40), decision_at=decision_rounds(self.cfg, job.method),
                                   k_active=opt.get("k_active", DEFAULT_K_ACTIVE), name=name, **kw)
        if m == "b3_llm_decide":
            return PosteriorLLMDecideAgent(self.pool(), self.library_for(job, bot), llm=self.llm, model=decider,
                                           proposer=proposer(), name=name, **post_kw)
        if m == "b0_llm_direct":
            return DirectLLMAgent(self.llm, model=decider)
        if m == "b1_verbal":
            return VerbalBeliefAgent(self.llm, model=decider)
        if m == "b4_oracle_type":
            return OracleTypeLLMAgent(self.llm, bot, model=decider)
        raise ValueError(f"unknown method {m!r}")

    # -- execution

    def rss_mb(self) -> float:
        return int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20

    def watchdog(self, limit_mb: float, every_s: float = 1.0) -> None:
        """Stop the run cleanly (exit code 3, resumable) before the kernel OOM-kills it."""
        while not self.stop.wait(every_s):
            if self.rss_mb() > limit_mb:
                self.fatal = f"memory watchdog: main process RSS {self.rss_mb():.0f} MB > {limit_mb:.0f} MB"
                self.stop.set()

    def run_job(self, job: Job, total: int) -> None:
        if self.stop.is_set():
            return
        game, bot = make_bot(self.cfg, job)
        agent = self.make_agent(job, bot)
        try:
            res = run_episode(game, bot, agent, self.cfg["horizon"], seed=crc(job.episode), episode_key=job.episode)
        except (LLMUnavailable, BudgetExceeded) as e:
            if isinstance(e, LLMUnavailable) and self.cfg.get("api_failure_scope", "matrix") == "episode":
                with self.lock:
                    self.api_failed.add(job.key)
                    with (self.run_dir / "api_failures.jsonl").open("a", encoding="utf-8") as out:
                        out.write(dumps({"job_key": job.key, "reason": str(e), "time": time.time()}) + "\n")
                    print(f"[api deferred] {job.key}: {e}; other episodes continue", flush=True)
                    self.write_progress(total)
                return
            self.fatal = f"{type(e).__name__}: {e}"
            self.stop.set()
            return
        except Exception as e:  # noqa: BLE001 - a bug in one job must not kill the matrix
            self._error(job, f"{type(e).__name__}: {e}", traceback.format_exc())
            return
        finally:
            agent.close()
        s = {**res.summary(), "method": job.method, "family": job.family, "param": job.param,
             "env_seed": job.seed, "job_key": job.key, "episode": job.episode, "status": "ok"}
        max_fb = self.cfg.get("max_fallback_rate", 0.2)
        if s["fallback_rate"] > max_fb:
            self._error(job, f"fallback_rate {s['fallback_rate']:.2f} > {max_fb}", "")
            return
        with self.lock:
            self.summaries.write(dumps(s) + "\n")
            self.summaries.flush()
            self.traces.write({**res.to_record(), "job_key": job.key, "method": job.method})
            self.traces.flush()
            self.n_done += 1
            el = time.time() - self.t0
            eta = el / self.n_done * (total - self.n_done)
            u = self.llm.usage if self.llm else {}
            print(f"[{self.n_done}/{total}] {job.key:<44} pay={s['payoff_per_round']:+.3f} kl={s['kl']:.3f} "
                  f"reg={s['regret']:.3f} fb={s['fallback_rate']:.2f} llm={s['llm_calls']} {s['wall_s']:.0f}s | "
                  f"total llm calls={u.get('calls', 0)} invalid={u.get('invalid_responses', 0)} "
                  f"retries={u.get('transport_retries', 0)} failures={u.get('call_failures', 0)} | "
                  f"rss {self.rss_mb():.0f}MB | elapsed {el / 60:.1f}m "
                  f"eta {'unknown (API deferred)' if self.api_failed else f'{eta / 60:.0f}m'}", flush=True)
            self.write_progress(total)

    def write_progress(self, total: int) -> None:
        """Caller holds self.lock; remaining includes API-deferred episodes."""
        elapsed = time.time() - self.t0
        eta = elapsed / self.n_done * (total - self.n_done) if self.n_done and not self.api_failed else None
        (self.run_dir / "progress.json").write_text(dumps(
            {"done_this_run": self.n_done, "remaining": total - self.n_done,
             "api_failed_this_run": len(self.api_failed), "retry_pending": len(self.api_failed),
             "elapsed_s": round(elapsed), "rss_mb": round(self.rss_mb()),
             "eta_s": round(eta) if eta is not None else None,
             "llm_usage": self.llm.usage if self.llm else {}, "updated": time.time()}))

    def _error(self, job: Job, reason: str, tb: str) -> None:
        with self.lock:
            self.errors.write(dumps({"job_key": job.key, "reason": reason, "traceback": tb, "time": time.time()}) + "\n")
            self.errors.flush()
            print(f"[error] {job.key}: {reason}", flush=True)

    def close(self) -> None:
        with self.lock:  # never close a file while a worker thread is writing a record
            for f in (self.summaries, self.errors):
                if not f.closed:
                    f.flush()
                    f.close()
            self.traces.close()
        for p in self.pools:
            p.close()


def _raise_interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a line cut by a crash
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--methods", default=None, help="comma-separated subset of the config's methods")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-config", action="store_true")
    ap.add_argument("--max-jobs-per-method", type=int, default=None, help="run at most this many pending jobs per method (smoke tests)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_dir = Path(args.run_dir or ROOT / "runs" / "matrix" / cfg["name"])
    methods = [m for m in cfg["methods"] if not args.methods or m in args.methods.split(",")]
    unknown = {resolve_base(cfg, m) for m in methods} - set(ALL_METHODS)
    if unknown:
        raise SystemExit(f"unknown methods: {sorted(unknown)}")
    problems = config_problems(cfg, methods)
    if problems:
        raise SystemExit("invalid method options:\n  " + "\n  ".join(problems))
    jobs = build_jobs(cfg, methods)

    if args.dry_run:
        print(f"run dir: {run_dir}\nhorizon {cfg['horizon']}, split {cfg['split']}")
        for m in methods:
            n = sum(j.method == m for j in jobs)
            calls = 0
            if resolve_base(cfg, m) in PER_ROUND_LLM_METHODS:
                calls += n * cfg["horizon"]
            if uses_llm_proposer(cfg, m):
                cd = method_opts(cfg, m).get("propose_cooldown", cfg.get("propose_cooldown", 20))
                calls += n * (cfg["horizon"] // cd + len(method_opts(cfg, m).get("propose_at", [])))
            if resolve_base(cfg, m) == "menu_choice":
                # per decision point: the decision and n_variation variation calls, each with at most one repair
                calls += 2 * n * len(decision_rounds(cfg, m)) * (1 + method_opts(cfg, m).get("n_variation", 0))
            print(f"  {m:<16} {n:>5} episodes   <= {calls:>6} LLM calls")
        total_calls = sum(resolve_base(cfg, j.method) in PER_ROUND_LLM_METHODS for j in jobs) * cfg["horizon"]
        print(f"total {len(jobs)} episodes; per-round LLM calls alone need >= {total_calls / cfg.get('rpm_hint', 70) / 60:.1f} h at 70 RPM")
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.json"
    h = config_hash(cfg)
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        if old["config_hash"] != h and not args.force_config:
            raise SystemExit(f"{run_dir} was created with a different config (hash {old['config_hash']} != {h}). "
                             "Use another --run-dir or --force-config.")
    cfg_path.write_text(dumps({"config": cfg, "config_hash": h, "updated": time.time()}))

    done = {r["job_key"] for r in read_jsonl(run_dir / "summaries.jsonl") if r.get("status") in ("ok", "abandoned")}
    attempts: dict[str, int] = {}
    for r in read_jsonl(run_dir / "errors.jsonl"):
        attempts[r["job_key"]] = attempts.get(r["job_key"], 0) + 1
    max_att = cfg.get("max_job_attempts", 3)
    pending, abandoned = [], []
    for j in jobs:
        if j.key in done:
            continue
        (abandoned if attempts.get(j.key, 0) >= max_att else pending).append(j)
    if args.max_jobs_per_method is not None:
        kept, seen = [], {}
        for j in pending:
            seen[j.method] = seen.get(j.method, 0) + 1
            if seen[j.method] <= args.max_jobs_per_method:
                kept.append(j)
        pending = kept
    print(f"{len(jobs)} jobs: {len(done)} done, {len(pending)} pending, {len(abandoned)} newly abandoned (>= {max_att} failed attempts)")

    k0_games = sorted({j.game for j in pending if library_kind(cfg, j.method) == "k0"})
    needs_llm = (any(uses_llm(cfg, j.method) for j in pending)
                 or any(not (run_dir / "libraries" / f"{g}.json").exists() for g in k0_games))
    llm = None
    if needs_llm:
        api = (load_api_config(cfg.get("api_key_file"), profile=cfg.get("api_profile"))
               if cfg.get("api_profile") or cfg.get("api_key_file") else None)
        llm = LLM(api=api, budget_tokens=cfg.get("budget_tokens"), **cfg.get("llm", {}))
        roles = sorted({cfg.get("proposer", "proposer"), cfg.get("decider", "decider")}
                       | {(cfg["methods"].get(m) or {}).get(k) for m in methods for k in ("decider", "proposer")} - {None})
        for attempt in range(3):
            report = llm.health_check(roles)
            for r in report:
                print(f"[health] {r}", flush=True)
            if all(r["ok"] for r in report):
                break
            if attempt < 2:
                print("[health] some roles failed; retrying in 60 s", flush=True)
                time.sleep(60)
        else:
            print("[health] API unhealthy; not starting. Re-run later.", flush=True)
            return 3

    # SIGTERM (tmux kill-session, kill) behaves like Ctrl-C: stop cleanly, keep finished results
    signal.signal(signal.SIGTERM, _raise_interrupt)
    mx = Matrix(cfg, run_dir, llm)
    executors: list[ThreadPoolExecutor] = []
    with mx.lock:
        for j in abandoned:
            mx.summaries.write(dumps({"job_key": j.key, "method": j.method, "episode": j.episode, "status": "abandoned"}) + "\n")
        mx.summaries.flush()
    try:
        if k0_games:
            mx.ensure_libraries(k0_games)
        mx.check_exclusions(pending)
        # LLM-heavy, long jobs first within each lane so the tail is short
        order = {m: k for k, m in enumerate(["menu_choice", "bsi", "b3_llm_decide", "bsi_minimal", "b1_verbal", "b4_oracle_type",
                                             "b0_llm_direct", "b2_static", "bsi_seedlib", "oracle_br", "freq_br", "random"])}
        pending.sort(key=lambda j: (order.get(resolve_base(cfg, j.method), 99), j.episode))
        sb = [j for j in pending if resolve_base(cfg, j.method) in SANDBOX_METHODS]
        other = [j for j in pending if resolve_base(cfg, j.method) not in SANDBOX_METHODS]
        total = len(pending)
        with mx.lock:
            mx.write_progress(total)
        threading.Thread(target=mx.watchdog, args=(cfg.get("max_rss_mb", 800),), daemon=True).start()
        executors += [ThreadPoolExecutor(cfg.get("sandbox_threads", 3), thread_name_prefix="sb"),
                      ThreadPoolExecutor(cfg.get("llm_threads", 8), thread_name_prefix="llm")]
        futs = [executors[0].submit(mx.run_job, j, total) for j in sb] + [executors[1].submit(mx.run_job, j, total) for j in other]
        for f in as_completed(futs):
            f.result()
            if mx.stop.is_set():
                raise KeyboardInterrupt  # fatal condition: cancel queued jobs and exit without joining blocked threads
        for ex in executors:
            ex.shutdown(wait=True)
    except KeyboardInterrupt:
        # Worker threads may be blocked in an HTTP read that cannot be interrupted, so do not join them:
        # cancel queued jobs, flush finished results, and exit. The interrupted episodes rerun on resume.
        mx.stop.set()
        for ex in executors:
            ex.shutdown(wait=False, cancel_futures=True)
        mx.close()
        if mx.fatal:
            print(f"\n[stopped] {mx.fatal}\nFinished jobs are saved; re-run the same command to resume.", flush=True)
            os._exit(3)
        print("\n[interrupted] finished jobs are saved; re-run the same command to resume.", flush=True)
        os._exit(130)
    except BaseException:
        mx.close()
        raise
    mx.close()

    if mx.fatal:
        print(f"\n[stopped] {mx.fatal}\nFinished jobs are saved; re-run the same command to resume.", flush=True)
        return 3
    if mx.api_failed:
        print(f"\n[incomplete] {len(mx.api_failed)} episodes deferred after API failures; "
              "all other queued jobs processed. Resume retries only unfinished jobs.", flush=True)
        return 3
    print(f"\nall pending jobs processed in {(time.time() - mx.t0) / 60:.1f} min; LLM usage {llm.usage if llm else {}}")
    from analyze_matrix import analyze  # noqa: E402

    print(analyze(run_dir))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
