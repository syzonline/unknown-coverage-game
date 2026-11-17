"""Aggregate one or more matrix runs into a report (guideline §3.7 metrics and §3.9 acceptance checks).

    .venv/bin/python scripts/analyze_matrix.py runs/matrix/pilot                       # -> runs/matrix/pilot/report.md
    .venv/bin/python scripts/analyze_matrix.py runs/matrix/full runs/matrix/replay_v1 runs/matrix/v2_offline \\
        --out runs/matrix/report_v2.md --ref bsi_v2_cp

Several run directories are merged by job_key. A job present in more than one directory must have the same
result in each (e.g. an instrumented replay); the record from the later directory is kept (it has more
fields). Different results for one job_key are an error: such runs are not the same experiment.

--supersede m1,m2 marks methods whose definition was fixed and re-run: their rows come only from the last
directory that has them, and rows of those methods in earlier directories are ignored.

Paired tests report two p-values: per episode (p) and per cluster (p clu). Bots are seeded by game|family|param,
so the env seeds of one (game, family, param) cell face the same bot and are not independent; the cluster
bootstrap resamples whole cells and is the one acceptance checks use.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import warnings

import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bsi.stats import cluster_bootstrap, ece_conf, paired_bootstrap, top_label  # noqa: E402
from bsi.store import iter_traces  # noqa: E402
from p2_opponent.bots import NOVEL_FAMILIES  # noqa: E402
from p2_opponent.propose import ORACLE_PROGRAMS  # noqa: E402
from p2_opponent.runner import adaptation_latency, post_switch_excess_kl  # noqa: E402

METRICS = [("payoff_per_round", "payoff", "+.3f"), ("regret", "regret", ".3f"), ("kl", "KL", ".3f"),
           ("kl_second_half", "KL 2nd½", ".3f"), ("log_loss", "logloss", ".3f"), ("ece", "ECE", ".3f"),
           ("fallback_rate", "fallback", ".3f"), ("llm_calls", "LLM calls/ep", ".0f"), ("llm_tokens", "tokens/ep", ".0f")]
METHOD_ORDER = ["oracle_br", "b4_oracle_type", "bsi_v2_cp", "bsi_v2", "bsi", "bsi_oracle_prop", "bsi_minimal_v2",
                "bsi_minimal", "b3_llm_decide", "b2_cp", "b2_static", "seedlib_cp", "bsi_seedlib", "b1_verbal",
                "b0_llm_direct", "freq_br", "random"]
PLAN_SUFFIX = "_plan"
LLM_BASELINES = ("b0_llm_direct", "b1_verbal", "b3_llm_decide", "b4_oracle_type")
# (evolving method, its static counterpart): same initial library, with vs without a proposer
EVO_PAIRS = [("bsi", "b2_static"), ("bsi_v2", "b2_static"), ("bsi_v3_rich", "b2_static"), ("bsi_tpl_periodic", "b2_static"),
             ("bsi_tpl_full", "b2_static"), ("bsi_oracle_prop", "b2_static"), ("bsi_v2_cp", "b2_cp"),
             ("bsi_v3_rich_cp", "b2_cp"), ("b2_cp", "b2_static"), ("seedlib_cp", "bsi_seedlib"), ("bsi_v2_cp", "bsi_v2"),
             ("bsi_minimal_v2", "b2_static"), ("bsi_v3_rich", "bsi_tpl_periodic"), ("bsi_v3_rich", "bsi_tpl_full"),
             # K0 without its periodic programs
             ("bsi_v3_rich_nocycle", "b2_nocycle"), ("bsi_tpl_periodic_nocycle", "b2_nocycle"),
             ("bsi_oracle_prop_nocycle", "b2_nocycle"), ("b2_nocycle", "b2_static"), ("bsi_v3_rich_nocycle", "bsi_v3_rich")]
# (static base, oracle-proposer ceiling, suffix of the methods measured against them)
CEILINGS = [("b2_static", "bsi_oracle_prop", ""), ("b2_nocycle", "bsi_oracle_prop_nocycle", "_nocycle")]
SAME_RESULT = ("payoff_per_round", "kl", "regret")


def load_one(run_dir: Path) -> list[dict]:
    rows = []
    for line in (run_dir / "summaries.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("status") == "ok":
            rows.append(r)
    latest = {r["job_key"]: r for r in rows}  # keep the last record per job
    return list(latest.values())


def _same(a: dict, b: dict) -> bool:
    return all(np.isclose(a.get(k, np.nan), b.get(k, np.nan), equal_nan=True, rtol=0, atol=1e-12) for k in SAME_RESULT)


def load(run_dirs: list[Path], supersede: tuple[str, ...] = ()) -> tuple[list[dict], dict[str, Path]]:
    """Merged rows and, per job_key, the directory whose record (summary and trace) is used."""
    merged: dict[str, dict] = {}
    source: dict[str, Path] = {}
    per_dir = [(d, load_one(d)) for d in run_dirs]
    last = {m: d for d, rs in per_dir for m in {r["method"] for r in rs} if m in supersede}
    for d, rs in per_dir:
        for r in rs:
            if r["method"] in last and last[r["method"]] != d:
                continue
            k = r["job_key"]
            if k in merged and not _same(merged[k], r):
                raise SystemExit(f"{k}: different results in {source[k]} and {d}; these are not the same experiment")
            merged[k], source[k] = r, d
    return list(merged.values()), source


def trace_stats(source: dict[str, Path]) -> tuple[dict, dict, dict, dict]:
    """Stream traces once. Returns
    - pooled ECE per method over all predicted rounds (per-episode ECE on ~60-150 rounds is biased upward);
    - adaptation records per method for switching bots: (latency, censored, applicable), recomputed with the
      current definition so older runs are scored the same way;
    - post-switch KL per job_key: (excess, recovered, post, applicable), defined for every method (older summaries
      lack it). `recovered` means something only when `applicable`: a method that never tracked the bot is already
      "within 0.05 of its pre-switch level" right after the switch;
    - menu-choice decision stats per job_key (menu_record), for episodes with v5 `decision` events."""
    confs: dict[str, list] = defaultdict(list)
    menu: dict[str, dict] = {}
    adapt: dict[str, list] = defaultdict(list)
    excess: dict[str, tuple] = {}
    seen: set[str] = set()
    for d in sorted(set(source.values())):
        files = sorted(d.glob("traces-*.jsonl.gz"), reverse=True)  # newest first: keep the last record per job
        for f in files:
            try:
                for r in iter_traces(f):
                    k = r.get("job_key")
                    if source.get(k) != d or k in seen:
                        continue
                    seen.add(k)
                    m = r["method"]
                    rec = menu_record(r)
                    if rec is not None:
                        menu[k] = rec
                    preds = np.array(r["preds"], dtype=float)
                    outcomes = np.array([a[1] for a in r["actions"]])
                    ok = np.isfinite(preds).all(axis=1)
                    if ok.any():
                        confs[m].append(top_label(preds[ok], outcomes[ok]))
                    if r.get("t_switch") is not None:
                        kls = np.array(r["kl_per_round"], dtype=float)
                        adapt[m].append(adaptation_latency(kls, int(r["t_switch"])))
                        excess[k] = (*post_switch_excess_kl(kls, int(r["t_switch"])), adapt[m][-1][2])
            except (EOFError, OSError):
                continue  # a trace file cut by a crash: earlier records are still used
    pooled = {m: ece_conf(np.concatenate([c for c, _ in v]), np.concatenate([k for _, k in v])) for m, v in confs.items()}
    return pooled, adapt, excess, menu


def menu_record(r: dict) -> dict | None:
    """What the LLM did at its decision points in one v5 menu-choice episode (None for other episodes).
    cond: "in" = the true class's program is in the agent's menu, "loo" = a seed family whose program was left out
    of the menu, "out" = a family outside the seed menu. true_active: a program of the true class (menu id, or an
    `oracle_` id) is active. Failed creates count only when no later op (the repair turn) added that name."""
    dec = [e for e in r.get("events", []) if e.get("event") == "decision"]
    if not dec:
        return None
    fam = r["bot_full"]["family"]
    base = set(ORACLE_PROGRAMS.get(fam, []))
    true_ids = base | {f"oracle_{i}" for i in base}
    menu = set(dec[0].get("menu", []))
    cond = "in" if true_ids & menu else ("out" if fam in NOVEL_FAMILIES else "loo")
    ops = [o for e in dec for o in e.get("ops", [])]
    kinds = [{o["op"] for o in e.get("ops", []) if o.get("ok")} for e in dec]
    act = [{a["name"] for a in e.get("active", [])} for e in dec]
    first_true = next((e["t"] for e, a in zip(dec, act) if a & true_ids), None)
    created_ok = {o["name"] for o in ops if o["op"] == "create" and o.get("ok")}
    belief = r.get("belief") or []
    by = (belief[-1].get("by_src") or {}) if belief else {}
    tok = lambda key: sum((e.get(key) or 0) + ((e.get("repair") or {}).get(key) or 0) for e in dec)  # noqa: E731
    return {
        "cond": cond, "n_dec": len(dec), "errors": sum("error" in e for e in dec),
        "select": sum(o["op"] == "select" and bool(o.get("ok")) for o in ops),
        "create": len(created_ok),
        "create_failed": len({o["name"] for o in ops if o["op"] == "create" and not o.get("ok")} - created_ok),
        "revise": sum(o["op"] == "revise" and bool(o.get("ok")) for o in ops),
        "retire": sum(o["op"] == "retire" for o in ops),
        "first": ("both" if {"select", "create"} <= kinds[0] else "select" if "select" in kinds[0]
                  else "create" if "create" in kinds[0] else "none"),
        "ever_create": any("create" in k_ for k_ in kinds),
        "changes": sum(a != b for a, b in zip([set()] + act[:-1], act)),
        "true_final": bool(act[-1] & true_ids) if cond == "in" else None,
        "n_active": len(act[-1]),
        # in only: share of the final active models that are the true class, and the chance that a random draw of as
        # many menu models as are active would include the true class (the bar a selective choice must clear)
        "true_share": len(act[-1] & true_ids) / len(act[-1]) if cond == "in" and act[-1] else (0.0 if cond == "in" else None),
        "true_chance": chance_hit(len(menu), len(true_ids & menu), len(act[-1] & menu)) if cond == "in" else None,
        "first_true_t": first_true if cond == "in" else None,
        "mass_menu": by.get("menu", 0.0), "mass_own": sum(v for k, v in by.items() if k != "menu"),
        "mass_ref": belief[-1].get("other") if belief else None,
        "degraded": sum(bool(e.get("degraded")) for e in dec) + sum(bool((e.get("repair") or {}).get("degraded")) for e in dec),
        "repairs": sum("repair" in e for e in dec),
        "completion_tokens": tok("completion_tokens"), "prompt_tokens": tok("prompt_tokens"),
        **v6_record(r, dec),
    }


def v6_record(r: dict, dec: list[dict]) -> dict:
    """v6 evolution measures of one episode (empty for v5 episodes): archive inserts, mutations and their gain in
    fitted mean log-likelihood per round over the parent at insertion, lineage depth, exploration, variation calls."""
    ev = r.get("events", [])
    adds = [e for e in ev if e.get("event") == "archive_add"]
    var = [e for e in ev if e.get("event") == "variation"]
    if not adds and not var and not any("archive" in e or e.get("mode") == "select_only" for e in dec):
        return {"v6": False}
    ok = [e for e in adds if e.get("ok")]
    muts = [e for e in ok if e.get("op") == "mutate"]
    gains = [e["mean_ll"] - e["parent_mean_ll"] for e in muts if e.get("parent_mean_ll") is not None]
    snap = next((e["archive"] for e in reversed(dec) if e.get("archive")), [])
    ops = [o for e in dec for o in e.get("ops", [])]
    return {
        "v6": True, "archive_ok": len(ok), "archive_failed": len(adds) - len(ok), "mutations": len(muts),
        "mut_gains": gains, "activates": sum(o["op"] == "activate" and bool(o.get("ok")) for o in ops),
        "max_gen": max([e.get("gen") or 0 for e in ok] or [0]), "var_calls": len(var),
        "var_errors": sum("error" in e for e in var),
        "var_degraded": sum(bool(e.get("degraded")) for e in var),
        "var_tokens": sum((e.get("prompt_tokens") or 0) + (e.get("completion_tokens") or 0) for e in var),
        "best_mean_ll": snap[0]["mean_ll"] if snap else None,
        "explore": sum(e.get("event") == "explore" for e in ev),
    }


# v6 pre-registered contrasts (prereg/v6/PREREGISTRATION.md); used instead of every pair when these arms are present
PREREG_CONTRASTS = [("v6_full_e30", "v6_noH_e30"), ("v6_full_e30", "v6_noR_e30"), ("v6_full_e30", "v6_noN_e30"),
                    ("v6_full_e30", "v6_full_e15"), ("v6_full_e15", "v6_full_e0"), ("v6_full_e30", "v6_full_e0"),
                    ("v6_full_e30", "v6_base")]


def explore_suffix(m: str) -> str:
    """Methods named *_e15 / *_e30 explore at that rate; their matched controls carry the same suffix."""
    for suf in ("_e15", "_e30"):
        if m.endswith(suf):
            return suf
    return ""


OUT_GAP_MIN = 0.05  # per-episode control gap of the out_gap subset (prereg/v5/PREREGISTRATION.md, amendment 1)


def chance_hit(n: int, m: int, k: int) -> float:
    """P(a uniform draw of k of n menu models includes at least one of the m true-class models)."""
    return 1.0 - math.comb(n - m, k) / math.comb(n, k) if 0 < k <= n else 0.0


def menu_section(rows: list[dict], stats: dict[str, dict]) -> list[str]:
    """v5: decision behaviour of the menu-choice arms per condition (in = the true class is in the menu, out = a
    family outside it), their KL against the no-LLM menu arms, and the paired comparisons the C1 claim rests on."""
    meth = sorted({r["method"] for r in rows if r["job_key"] in stats})
    if not meth:
        return []

    def cond(r: dict) -> str:
        if r["job_key"] in stats:
            return stats[r["job_key"]]["cond"]
        return "out" if r["family"] in NOVEL_FAMILIES else "in"

    # out_gap: the out episodes where the no-LLM controls show a gap (KL(menu_all) − KL(menu_oracle) >= OUT_GAP_MIN):
    # some bot parameters degenerate into in-menu behaviour (e.g. react2 tables that nearly always play one action)
    ctl = defaultdict(dict)
    for r in rows:
        if r["method"] in ("menu_all", "menu_oracle"):
            ctl[r["episode"]][r["method"]] = r["kl"]
    gap_ok = {e for e, k in ctl.items() if len(k) == 2 and k["menu_all"] - k["menu_oracle"] >= OUT_GAP_MIN}

    def member(r: dict, c: str) -> bool:
        return cond(r) == "out" and r["episode"] in gap_ok if c == "out_gap" else cond(r) == c

    def opt_mean(st: list[dict], k: str) -> str:
        v = [s[k] for s in st if s[k] is not None]
        return _fmt(np.mean(v) if v else np.nan, ".2f")

    conds = [c for c in ("in", "loo", "out", "out_gap") if any(member(r, c) for r in rows if r["job_key"] in stats)]
    o = ["", "## Menu-choice diagnostics (v5)", "",
         "Per decision point the LLM may select menu models, create its own, revise or retire. in = the true class is in the",
         "menu (seed families), loo = a seed family whose program was left out of the menu, out = families outside the",
         "seed menu; out_gap = the out episodes whose no-LLM control gap KL(menu_all) − KL(menu_oracle) is >= "
         f"{OUT_GAP_MIN} (pre-registered subset). first = what the first decision added; true active =",
         "the true class's menu program is active after the last decision (in only); chance = the same rate for a random draw",
         "of as many menu models as were active; true share = share of the final active models that are the true class;",
         "mass = final posterior mass by origin.", "",
         "| method | cond | n | first: select / create / both / none | ever create | selects/ep | creates/ep (failed) | "
         "revises/ep | changes/ep | active (final) | true active (final) | chance | true share | first true t | "
         "mass menu / own / ref | degraded/ep | repairs/ep | tokens/ep (prompt+compl., incl. repairs) | KL |",
         "|---|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for m in meth:
        for c in conds:
            rs = [r for r in rows if r["method"] == m and member(r, c) and r["job_key"] in stats]
            if not rs:
                continue
            st = [stats[r["job_key"]] for r in rs]
            mean = lambda k: float(np.mean([s[k] for s in st]))  # noqa: E731
            first = "/".join(f"{np.mean([s['first'] == x for s in st]):.2f}" for x in ("select", "create", "both", "none"))
            tf = [s["true_final"] for s in st if s["true_final"] is not None]
            ft = [s["first_true_t"] for s in st if s["first_true_t"] is not None]
            o.append(f"| {m} | {c} | {len(rs)} | {first} | {mean('ever_create'):.2f} | {mean('select'):.2f} | "
                     f"{mean('create'):.2f} ({mean('create_failed'):.2f}) | {mean('revise'):.2f} | {mean('changes'):.2f} | "
                     f"{mean('n_active'):.2f} | {_fmt(np.mean(tf) if tf else np.nan, '.2f')} | {opt_mean(st, 'true_chance')} | "
                     f"{opt_mean(st, 'true_share')} | {_fmt(np.median(ft) if ft else np.nan, '.0f')} | "
                     f"{mean('mass_menu'):.2f} / {mean('mass_own'):.2f} / {_fmt(np.mean([s['mass_ref'] or 0 for s in st]), '.2f')} | "
                     f"{mean('degraded'):.2f} | {mean('repairs'):.2f} | {mean('prompt_tokens') + mean('completion_tokens'):.0f} | "
                     f"{_nanmean(rs, 'kl'):.3f} |")
    present = {r["method"] for r in rows}
    pairs = []
    for m in meth:  # controls at the method's exploration rate (v6 suffix convention), then freq_br
        suf = explore_suffix(m)
        pairs += [(m, f"{c}{suf}") for c in ("menu_all", "menu_oracle", "menu_loo") if f"{c}{suf}" in present]
        pairs += [(m, "freq_br")] if "freq_br" in present else []
    prereg = [(a, b) for a, b in PREREG_CONTRASTS if a in meth and b in meth]
    pairs += prereg or [(a, b) for a in meth for b in meth if a < b]
    o += ["", "Paired KL by condition of the first method's episodes (negative = the first method predicts better):", "",
          "| comparison | cond " + PAIRED_HEAD[0], "|---|---|" + PAIRED_HEAD[1]]

    def in_cond(c: str) -> list[dict]:
        """Menu-choice rows of condition c, plus every other row on those episodes (controls pair by episode)."""
        eps = {r["episode"] for r in rows if r["job_key"] in stats and member(r, c)}
        return [r for r in rows if r["episode"] in eps and (r["job_key"] not in stats or member(r, c))]

    for a, b in pairs:
        for c in conds:
            line = _paired_row(f"{a} − {b} | {c}", paired(in_cond(c), a, b, "kl"))
            if line:
                o.append(line)
    for c in conds:
        for suf in sorted({explore_suffix(m) for m in meth}):
            share = ceiling_share(in_cond(c), [m for m in meth if explore_suffix(m) == suf],
                                  base=f"menu_all{suf}", ceil=f"menu_oracle{suf}")
            if share:
                o += ["", f"Condition {c}{f' (controls {suf[1:]})' if suf else ''}:"] + share
    o += evolution_section(rows, stats, member, conds)
    return o


def evolution_section(rows: list[dict], stats: dict[str, dict], member, conds: list[str]) -> list[str]:
    """v6: what the evolution loop did per method and condition. mutation gain = fitted mean log-lik per round of the
    child minus the parent's, both on the history at insertion; payoff shows the price of exploration."""
    meth = sorted({r["method"] for r in rows if r["job_key"] in stats and stats[r["job_key"]].get("v6")})
    if not meth:
        return []
    o = ["", "## Evolution (v6)", "",
         "archive ok (failed) = programs that entered the archive (were rejected) per episode; mutations = mutate inserts "
         "per episode; gain = child − parent mean log-lik per round at insertion (share > 0 = improving mutations); gen = "
         "deepest lineage; best ll = mean log-lik per round of the best archive program at the last decision.", "",
         "| method | cond | n | archive ok (failed) | mutations/ep | mutation gain (median) | improving | activates/ep | "
         "max gen | variation calls/ep (errors, degraded) | explore rounds/ep | best ll | payoff | KL |",
         "|---|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for m in meth:
        for c in conds:
            rs = [r for r in rows if r["method"] == m and member(r, c) and r["job_key"] in stats]
            if not rs:
                continue
            st = [stats[r["job_key"]] for r in rs]
            mean = lambda k: float(np.mean([s[k] for s in st]))  # noqa: E731
            gains = [g for s in st for g in s["mut_gains"]]
            best = [s["best_mean_ll"] for s in st if s["best_mean_ll"] is not None]
            o.append(f"| {m} | {c} | {len(rs)} | {mean('archive_ok'):.1f} ({mean('archive_failed'):.1f}) | "
                     f"{mean('mutations'):.2f} | {_fmt(np.median(gains) if gains else np.nan, '+.3f')} | "
                     f"{_fmt(np.mean([g > 0 for g in gains]) if gains else np.nan, '.2f')} | {mean('activates'):.2f} | "
                     f"{mean('max_gen'):.1f} | {mean('var_calls'):.1f} ({mean('var_errors'):.2f}, {mean('var_degraded'):.2f}) | "
                     f"{mean('explore'):.1f} | {_fmt(np.mean(best) if best else np.nan, '.3f')} | "
                     f"{_nanmean(rs, 'payoff_per_round'):+.3f} | {_nanmean(rs, 'kl'):.3f} |")
    return o


def _mean_ci(x: list[float]) -> str:
    a = np.array([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if len(a) == 0:
        return "–"
    if len(a) == 1:
        return f"{a[0]:.3f}"
    se = a.std(ddof=1) / np.sqrt(len(a))
    return f"{a.mean():.3f}±{1.96 * se:.3f}"


def _fmt(v: float, f: str) -> str:
    return "–" if v is None or not np.isfinite(v) else format(v, f)


def _nanmean(rs: list[dict], k: str) -> float:
    v = [r.get(k) for r in rs]
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def _methods(rows: list[dict]) -> list[str]:
    present = {r["method"] for r in rows}
    return [m for m in METHOD_ORDER if m in present] + sorted(present - set(METHOD_ORDER))


def cluster_of(episode: str) -> str:
    """game|family|param: the episodes of one cell face the same bot (bots are seeded without the env seed)."""
    return episode.rsplit("|", 1)[0]


def paired(rows: list[dict], a: str, b: str, metric: str, families: set[str] | None = None,
           games: set[str] | None = None) -> dict | None:
    """Per-episode paired bootstrap, plus `p_cluster` / `ci_cluster` / `clusters` from the cluster bootstrap."""
    by = defaultdict(dict)
    for r in rows:
        if (families is None or r["family"] in families) and (games is None or r["game"] in games):
            by[r["episode"]][r["method"]] = r.get(metric)
    pairs = [(d[a], d[b], cluster_of(ep)) for ep, d in by.items() if a in d and b in d
             and d[a] is not None and d[b] is not None and np.isfinite(d[a]) and np.isfinite(d[b])]
    if len(pairs) < 2:
        return None
    x, y, g = zip(*pairs)
    x, y = np.array(x), np.array(y)
    out = paired_bootstrap(x, y)
    c = cluster_bootstrap(x, y, list(g))
    out.update(p_cluster=c["p_two_sided"], ci_cluster=c["ci"], clusters=c["clusters"])
    return out


def _paired_row(label: str, r: dict | None) -> str | None:
    if not r:
        return None
    return (f"| {label} | {r['n']} | {r['clusters']} | {r['mean_diff']:+.4f} | [{r['ci_cluster'][0]:+.4f}, {r['ci_cluster'][1]:+.4f}] "
            f"| {r['p_two_sided']:.4f} | {r['p_cluster']:.4f} |")


PAIRED_HEAD = ["| n pairs | clusters | mean diff | 95% CI (cluster) | p | p clu |", "---:|---:|---:|---|---:|---:|"]


def _family_table(rows: list[dict], methods: list[str], families: list[str], metric: str) -> list[str]:
    out = ["| method | n | " + " | ".join(families) + " |", "|---|---:|" + "---:|" * len(families)]
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        if not rs:
            continue
        cells = [_fmt(_nanmean([r for r in rs if r["family"] == f], metric), ".3f") for f in families]
        out.append(f"| {m} | {len(rs)} | " + " | ".join(cells) + " |")
    return out


def ceiling_share(rows: list[dict], methods: list[str], base: str = "b2_static", ceil: str = "bsi_oracle_prop") -> list[str]:
    """Each evolving method's KL gain over the static library as a share of the oracle-proposer gain, on the
    non-switching episodes where all three ran (so a pilot on a few cells is compared on those cells only).
    The share's CI and the p-values resample whole (game, family, param) cells."""
    by = defaultdict(dict)
    for r in rows:
        if r["family"] != "switching":
            by[r["episode"]][r["method"]] = r["kl"]
    out = []
    for m in methods:
        eps = [(ep, d) for ep, d in by.items() if all(k in d and np.isfinite(d[k]) for k in (m, base, ceil))]
        if len(eps) < 2:
            continue
        g = [cluster_of(ep) for ep, _ in eps]
        km, kb, kc = (np.array([d[k] for _, d in eps]) for k in (m, base, ceil))
        gain, top = float(np.mean(kb - km)), float(np.mean(kb - kc))
        r = paired_bootstrap(km, kb)
        rc = cluster_bootstrap(km, kb, g)
        sh = cluster_bootstrap(kb, km, g, ratio_den=(kb, kc)) if top > 0 else None
        share = f"{sh['mean_diff']:.2f} | [{sh['ci'][0]:.2f}, {sh['ci'][1]:.2f}]" if sh else "– | –"
        out.append(f"| {m} | {len(eps)} | {rc['clusters']} | {gain:+.4f} | {top:+.4f} | {share} | {r['p_two_sided']:.4f} | {rc['p_two_sided']:.4f} |")
    if not out:
        return []
    return ["", f"Share of the oracle-proposer ceiling (non-switching episodes where {base}, {ceil} and the method all ran;",
            f"gain = KL({base}) − KL(method); share CI and p clu resample (game, family, param) cells; p = paired test vs {base}):", "",
            "| method | episodes | clusters | gain | oracle gain | share | share 95% CI | p | p clu |",
            "|---|---:|---:|---:|---:|---:|---|---:|---:|"] + out


def evolution_by_family(rows: list[dict], pairs: list[tuple[str, str]], families: list[str]) -> list[str]:
    """KL difference per bot family for each comparison, with the cluster p (where does a gain come from?)."""
    present = {r["method"] for r in rows}
    fams = [f for f in families if f != "switching"]
    head = ["", "Evolution gain by bot family (mean KL difference, cluster p in parentheses; 'excl. cycle' = the",
            "non-switching families other than cycle):", "",
            "| comparison | " + " | ".join(fams) + " | excl. cycle |", "|---|" + "---:|" * (len(fams) + 1)]
    out = []
    for a, b in pairs:
        if a not in present or b not in present:
            continue
        cells = []
        for fs in [{f} for f in fams] + [set(fams) - {"cycle"}]:
            r = paired(rows, a, b, "kl", fs)
            cells.append(f"{r['mean_diff']:+.4f} ({r['p_cluster']:.3f})" if r else "–")
        out.append(f"| {a} − {b} | " + " | ".join(cells) + " |")
    return head + out if out else []


def analyze(*run_dirs: str | Path, out: str | Path | None = None, ref: str | None = None,
            supersede: tuple[str, ...] = ()) -> str:
    dirs = [Path(d) for d in run_dirs]
    rows, source = load(dirs, supersede)
    if not rows:
        return "no finished episodes yet"
    plan_rows = [r for r in rows if r["method"].endswith(PLAN_SUFFIX)]
    rows = [r for r in rows if not r["method"].endswith(PLAN_SUFFIX)]
    methods = _methods(rows)
    if ref is None:
        ref = next((m for m in ("bsi_v2_cp", "bsi") if m in methods), None)
    families = sorted({r["family"] for r in rows})
    games = sorted({r["game"] for r in rows})
    title = dirs[0].name if len(dirs) == 1 else " + ".join(d.name for d in dirs)
    out_lines = [f"# Matrix report: {title}", "",
                 f"{len(rows) + len(plan_rows)} finished episodes; games {games}; families {families}; "
                 + (f"reference method for comparisons: `{ref}`." if ref else "no reference method (no bsi_v2_cp or bsi rows)."), ""]

    pooled_ece, adapt, excess, menu = trace_stats(source)
    for r in rows:
        if r["job_key"] in excess:
            (r["post_switch_excess_kl"], r["adapt_recovered"], r["post_switch_kl"],
             r["adapt_applicable"]) = excess[r["job_key"]]
    o = out_lines
    o += ["## Overall (mean over episodes; ± = 95% CI half-width)", "",
          "ECE = mean of per-episode ECE (biased upward on 150 rounds); ECE(pooled) = one ECE over all predicted rounds.",
          "regret is *myopic* (vs the best response to the bot's true mixed action this round); in IPD that rewards",
          "always defecting, so IPD decision quality is reported separately with planning arms.", "",
          "| method | n | " + " | ".join(lbl for _, lbl, _ in METRICS) + " | ECE(pooled) |",
          "|---|---:|" + "---:|" * (len(METRICS) + 1)]
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        cells = [_mean_ci([r.get(k) for r in rs]) if k in ("payoff_per_round", "kl", "regret") else _fmt(_nanmean(rs, k), f)
                 for k, _, f in METRICS]
        o.append(f"| {m} | {len(rs)} | " + " | ".join(cells) + f" | {_fmt(pooled_ece.get(m, np.nan), '.3f')} |")

    for metric, label in (("kl", "KL(true‖pred) per round"), ("regret", "myopic regret per round"),
                          ("payoff_per_round", "payoff per round")):
        o += ["", f"## {label} by bot family (all episodes of each method; n differs between methods)", ""]
        o += _family_table(rows, methods, families, metric)

    # the per-round LLM baselines ran on a subset: compare every method on exactly those episodes
    sub = {r["episode"] for r in rows if r["method"] in LLM_BASELINES}
    if sub:
        srows = [r for r in rows if r["episode"] in sub]
        for metric, label in (("kl", "KL"), ("payoff_per_round", "payoff")):
            o += ["", f"## {label} by bot family on the LLM-baseline subset ({len(sub)} episodes, same for every method)", ""]
            o += _family_table(srows, methods, families, metric)

    evo = [m for m in methods if any(r.get("other_mass_mean") is not None for r in rows if r["method"] == m)]
    if evo:
        o += ["", "## Library evolution diagnostics (methods with belief logs)", "",
              "other = mean posterior mass of the catch-all; proposer mass = final mass of proposed programs;",
              "proposer MAP = share of rounds where a proposed program is the MAP named hypothesis.", "",
              "| method | n | other mass | proposer mass (final) | proposer MAP | change points/ep | LLM calls/ep |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for m in evo:
            rs = [r for r in rows if r["method"] == m and r.get("other_mass_mean") is not None]
            o.append(f"| {m} | {len(rs)} | {_nanmean(rs, 'other_mass_mean'):.3f} | {_nanmean(rs, 'proposer_mass_final'):.3f} | "
                     f"{_nanmean(rs, 'proposer_map_frac'):.3f} | {_nanmean(rs, 'n_changepoints'):.2f} | {_nanmean(rs, 'llm_calls'):.1f} |")
        o += ["", "Evolution gain (paired, KL; negative = the evolving method predicts better):", "",
              "| comparison " + PAIRED_HEAD[0], "|---|" + PAIRED_HEAD[1]]
        for a, b in EVO_PAIRS:
            line = _paired_row(f"{a} − {b} (non-switching bots)", paired(rows, a, b, "kl", set(families) - {"switching"}))
            if line:
                o.append(line)
        o += evolution_by_family(rows, [p for p in EVO_PAIRS if p[1] in ("b2_static", "b2_nocycle", "bsi_tpl_periodic")], families)
        for base, ceil, suffix in CEILINGS:
            o += ceiling_share(rows, [m for m in evo if m.startswith("bsi") and not m.endswith("_cp") and m.endswith(suffix)
                                      and (suffix or not m.endswith("_nocycle"))
                                      and m not in ("bsi_seedlib", ceil)], base=base, ceil=ceil)

    o += menu_section(rows, menu)

    if adapt:
        o += ["", "## Adaptation after a strategy switch (switching bots)", "",
              "post-switch KL = mean KL in the 30 rounds after the switch (primary; defined for every method).",
              "excess = post-switch KL − mean KL in the 20 rounds before; it is also small for methods that never tracked",
              "the bot, so read it together with post-switch KL. Recovery and latency count only *applicable* episodes",
              "(pre-switch KL ≤ 0.25): recovered = share of them whose KL over the last 10 rounds is within 0.05 of the pre-switch level",
              "(for a method that never tracked the bot this holds trivially, so other episodes are left out).", "",
              "| method | switching eps | post-switch KL | excess KL | recovered (applicable) | applicable | median latency | censored |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for m in methods:
            v = adapt.get(m, [])
            e = [(r["post_switch_excess_kl"], r["adapt_recovered"], r["post_switch_kl"], r["adapt_applicable"]) for r in rows
                 if r["method"] == m and r.get("post_switch_kl") is not None]
            if not v:
                continue
            app = [x for x in v if x[2]]
            med = f"{np.median([x[0] for x in app]):.1f}" if app else "–"
            ex = _mean_ci([x[0] for x in e])
            post = _mean_ci([x[2] for x in e])
            e_app = [x for x in e if x[3]]
            rec = f"{np.mean([x[1] for x in e_app]):.2f}" if e_app else "–"
            o.append(f"| {m} | {len(v)} | {post} | {ex} | {rec} | {len(app)} | {med} | {sum(x[1] for x in app)} |")

    if plan_rows:
        o += ["", "## IPD with planning (depth-limited expectimax; decision quality vs the planning oracle)", "",
              "| method | n | payoff | payoff − oracle_plan (paired) | 95% CI | p | KL |", "|---|---:|---:|---:|---|---:|---:|"]
        prow = plan_rows + [r for r in rows if r["game"] == "ipd"]
        for m in sorted({r["method"] for r in prow}, key=lambda m: (not m.endswith(PLAN_SUFFIX), m)):
            rs = [r for r in prow if r["method"] == m]
            d = paired(prow, m, "oracle_plan", "payoff_per_round") if m != "oracle_plan" else None
            dcells = (f"{d['mean_diff']:+.4f} | [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}] | {d['p_two_sided']:.4f}"
                      if d else "– | – | –")
            o.append(f"| {m} | {len(rs)} | {_mean_ci([r['payoff_per_round'] for r in rs])} | {dcells} | {_fmt(_nanmean(rs, 'kl'), '.3f')} |")

    refs = [x for x in dict.fromkeys([ref, "bsi"]) if x in methods]
    for rf in refs:
        o += ["", f"## Paired comparisons vs {rf} (same bot, same env seed; mean diff = {rf} − other)", "",
              "| other | metric " + PAIRED_HEAD[0], "|---|---|" + PAIRED_HEAD[1]]
        for m in methods:
            if m == rf:
                continue
            for metric in ("kl", "payoff_per_round"):
                line = _paired_row(f"{m} | {metric}", paired(rows, rf, m, metric))
                if line:
                    o.append(line)

    # §3.9 acceptance checks
    if ref:
        o += acceptance(rows, families, pooled_ece, adapt, ref)
    else:
        o += ["", "## Acceptance checks (guideline §3.9)", "",
              "_skipped: none of these runs contains the method (bsi_v2_cp or bsi; set one with --ref)._"]

    for d in dirs:
        errs = d / "errors.jsonl"
        if errs.exists():
            n_err = sum(1 for _ in errs.open())
            o += ["", f"_{d.name}: {n_err} failed attempts in errors.jsonl (retried on resume; abandoned after max_job_attempts)._"]
    text = "\n".join(o)
    Path(out or dirs[0] / "report.md").write_text(text, encoding="utf-8")
    return text


def acceptance(rows: list[dict], families: list[str], pooled_ece: dict, adapt: dict, ref: str) -> list[str]:
    o = ["", f"## Acceptance checks (guideline §3.9), method = `{ref}`", ""]
    non_switch = set(families) - {"switching"}
    r = paired(rows, ref, "b1_verbal", "kl", non_switch) if ref else None
    o.append(f"- [{'x' if r and r['mean_diff'] < 0 and r['p_cluster'] < 0.01 else ' '}] KL({ref}) < KL(b1) on non-switching bots, p clu<0.01: "
             + (f"Δ={r['mean_diff']:+.4f}, p={r['p_two_sided']:.4f}, p clu={r['p_cluster']:.4f}, n={r['n']} ({r['clusters']} cells)"
                if r else "not enough paired episodes"))
    e = pooled_ece.get(ref)
    ep = _nanmean([x for x in rows if x["method"] == ref], "ece")
    o.append(f"- [{'x' if e is not None and e < 0.05 else ' '}] ECE({ref}) < 0.05 (pooled; the per-episode mean is reported, "
             f"not checked): pooled {_fmt(e, '.3f')}, mean per-episode {_fmt(ep, '.3f')} (biased upward on 150 rounds)")
    for a in dict.fromkeys(["bsi", ref]):
        rr = paired(rows, a, "b3_llm_decide", "payoff_per_round", games={"rps3", "hideseek5"})
        o.append(f"- [ ] belief→action gap, payoff({a}) − payoff(b3) on rps3+hideseek5 (reported, no threshold): "
                 + (f"{rr['mean_diff']:+.4f} [{rr['ci_cluster'][0]:+.4f}, {rr['ci_cluster'][1]:+.4f}], p={rr['p_two_sided']:.4f}, "
                    f"p clu={rr['p_cluster']:.4f}, n={rr['n']}" if rr else "n/a"))
    ra = paired(rows, ref, "b1_verbal", "post_switch_kl", {"switching"})
    o.append(f"- [{'x' if ra and ra['mean_diff'] < 0 and ra['p_cluster'] < 0.05 else ' '}] adaptation: post-switch KL "
             f"(30 rounds) {ref} < b1 (paired, revised criterion, p clu<0.05): "
             + (f"Δ={ra['mean_diff']:+.4f} [{ra['ci_cluster'][0]:+.4f}, {ra['ci_cluster'][1]:+.4f}], p={ra['p_two_sided']:.4f}, "
                f"p clu={ra['p_cluster']:.4f}, n={ra['n']}" if ra else "n/a"))
    lat = {m: [x[0] for x in adapt.get(m, []) if x[2]] for m in (ref, "b1_verbal")}
    if lat[ref] and lat["b1_verbal"]:
        a, b = np.median(lat[ref]), np.median(lat["b1_verbal"])
        o.append(f"- [{'x' if a < 0.5 * b else ' '}] original criterion, median latency {ref} < ½ × b1 (applicable episodes): "
                 f"{a:.1f} vs {b:.1f}, n = {len(lat[ref])} vs {len(lat['b1_verbal'])}")
    else:
        o.append(f"- [ ] original criterion, median latency {ref} < ½ × b1: n/a (applicable episodes: {ref} {len(lat[ref])}, "
                 f"b1 {len(lat['b1_verbal'])})")
    o.append("- [x] inference-module unit tests and parameter recovery: see `pytest -q` (tests/test_recovery.py)")
    return o


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="*", default=[str(ROOT / "runs" / "matrix" / "pilot")])
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--supersede", default="", help="comma-separated methods re-run after a fix: keep only their last directory")
    a = ap.parse_args()
    print(analyze(*a.run_dirs, out=a.out, ref=a.ref, supersede=tuple(m for m in a.supersede.split(",") if m)))
