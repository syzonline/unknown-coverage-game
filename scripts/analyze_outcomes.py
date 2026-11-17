"""Outcome-first, offline analysis of v6 opponent models (no LLM or sandbox execution).

    python scripts/analyze_outcomes.py runs/matrix/intern_full \
        --export-review reviews/intern.json --review-tail 80

Code reviews are explicit annotations of recorded source, not a name/keyword classifier.
Use --export-review FILE to prepare a source inventory for another run. Unreviewed
structures remain unknown; their stability is not inferred from program identity.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_matrix import load, _same  # noqa: E402
from bsi.store import iter_traces  # noqa: E402
from p2_opponent.propose import ORACLE_PROGRAMS  # noqa: E402
from p2_opponent.seed_programs import SEED_LIBRARY  # noqa: E402

MENU = {s.id: s for s in SEED_LIBRARY}
MATCHES = {"match": "匹配", "partial": "部分匹配", "mismatch": "不匹配", "unknown": "无法判定"}
SOURCES = {"menu": "菜单", "own": "自写", "reference": "参考", "mixed": "并列跨来源"}
TRUTH = {
    "wsls": "依据对手收益判断赢输；赢则保持，否则在其他动作间均匀切换",
    "reactive": "依据我方上一步动作查响应表，并混入噪声",
    "fictitious": "估计我方窗口动作频率，再做 softmax 最优响应（k=1）",
    "react2": "依据我方过去两步动作的有序对查响应表，并混入噪声",
    "lose_k_shift": "累计连续非赢次数；达阈值后改为对我方末步的最优响应，其他时候保持内部动作",
    "seq_br": "用我方一阶转移计数预测我方下一步，再做 softmax 最优响应",
    "block_drift": "按固定时间块循环平移同一个基础动作分布",
}


def read_records(run_dirs: list[Path], method: str) -> list[dict]:
    summaries, sources = load(run_dirs)
    expected = {r["job_key"]: r for r in summaries if r["method"] == method}
    records = {}
    for d in sorted(set(sources[k] for k in expected)):
        for path in sorted(d.glob("traces-*.jsonl.gz")):
            for r in iter_traces(path, resolve_overflow=True):
                k = r.get("job_key")
                if k not in expected or sources[k] != d:
                    continue
                if not _same(r, expected[k]):
                    raise ValueError(f"{k}: trace and successful summary disagree")
                if k in records and records[k] != r:
                    raise ValueError(f"{k}: conflicting duplicate traces")
                records[k] = r
    missing = expected.keys() - records.keys()
    if missing:
        raise ValueError(f"Missing successful traces: {sorted(missing)}")
    if not records:
        raise ValueError(f"No successful episodes for {method}")
    return [records[k] for k in sorted(records)]


def added_models(r: dict) -> dict:
    return {e["id"]: e for e in r["events"] if e.get("event") == "add" and e.get("ok")}


def active_ids(b: dict, adds: dict, round_no: int) -> list[str]:
    # Belief is logged after observing round t, BEFORE the decision at t.
    out = []
    for name in b.get("active", []):
        eligible = [e for e in adds.values() if e["id"].rsplit("#", 1)[0] == name and e["t"] < round_no]
        if eligible:
            out.append(eligible[-1]["id"])
    return out


def dominant(b: dict, adds: dict, round_no: int) -> dict:
    """Recorded MAP including reference, and identifiable two-model rounded ties.

    The trace stores only the named MAP, not all individual weights. A rounded
    50/50 split with exactly two named models is recoverable from their total.
    Other possible named ties cannot be inferred and are explicitly not invented.
    """
    mw, rw = b.get("map_w") or 0.0, b.get("other") or 0.0
    ids = [b["map"]] if b.get("map") else []
    if ids and b.get("n") == 2 and abs(2 * mw + rw - 1) <= 1.5e-5:
        candidates = active_ids(b, adds, round_no)
        if len(candidates) == 2:
            ids = candidates
    if rw > mw or not ids:
        ids = ["reference"]
    elif rw == mw:
        ids.append("reference")
    srcs = {"reference" if pid == "reference" else
            "menu" if adds[pid]["source"] == "menu" else "own" for pid in ids}
    return {"ids": sorted(ids), "source": next(iter(srcs)) if len(srcs) == 1 else "mixed",
            "weight": max(mw, rw), "tie": len(ids) > 1}


def export_review(records: list[dict], path: Path, tail: int) -> None:
    """Unique sources requiring review for terminal and tail-convergence measures."""
    groups = {}
    for r in records:
        adds = added_models(r)
        for t, b in enumerate(r["belief"], 1):
            if t <= len(r["belief"]) - tail:
                continue
            for pid in dominant(b, adds, t)["ids"]:
                if pid == "reference" or adds[pid]["source"] == "menu":
                    continue
                e = adds[pid]
                src = e["src"]
                entry = groups.setdefault(src, {"source": src, "mechanism": "", "label": "", "assessments": {}, "uses": []})
                use = {"job_key": r["job_key"], "pid": pid}
                if use not in entry["uses"]:
                    entry["uses"].append(use)
                fam = r["job_key"].split("|")[2]
                entry["assessments"].setdefault(fam, {"match": "unknown", "reason": "未复核"})
    data = {"description": "Read the source, label its mechanism, then compare against each listed true family. "
                           "Source equality binds the review; model names are not evidence.",
            "entries": [{"id": f"s{i:03d}", **v} for i, v in enumerate(groups.values(), 1)]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def load_reviews(path: Path | None) -> dict:
    if path is None:
        return {}
    out = {}
    for entry in json.loads(path.read_text())["entries"]:
        if entry["source"] in out:
            raise ValueError("Duplicate reviewed source")
        for a in entry["assessments"].values():
            if a["match"] not in MATCHES:
                raise ValueError(f"Unknown match label: {a['match']}")
        out[entry["source"]] = entry
    return out


def model_review(pid: str, adds: dict, family: str, reviews: dict, job_key: str | None = None) -> dict:
    if pid == "reference":
        return {"mechanism": "last_pair_table", "label": "上一回合联合条件表（参考）", "match": "unknown",
                "reason": "参考模型为通用条件频率估计；未据此识别真实生成规则。", "reviewed": True}
    e = adds[pid]
    if e["source"] == "menu":
        name = pid.rsplit("#", 1)[0]
        if name not in MENU or e.get("src") != MENU[name].src:
            raise ValueError(f"{pid}: menu source does not match its recorded template")
        match = name in ORACLE_PROGRAMS.get(family, [])
        # klevel is a family containing k=1, not evidence that its fitted k is 1.
        verdict = "partial" if family == "fictitious" and name == "klevel" else "match" if match else "mismatch"
        return {"mechanism": name, "label": name, "match": verdict, "reviewed": True,
                "reason": ("菜单 klevel 包含 k=1，但日志未保存终局拟合 k，不能确认选中了 fictitious 子结构。" if verdict == "partial"
                           else "菜单程序的核心条件与响应规则对应真实类别；参数估计质量另看预测 KL。" if match
                           else f"菜单 {name} 的生成机制与真实 {family} 不同。")}
    review = reviews.get(e.get("src"))
    in_scope = review is not None and (job_key is None or "uses" not in review or
                                      any(u["job_key"] == job_key for u in review["uses"]))
    if in_scope and review.get("mechanism") and family in review["assessments"]:
        return {"mechanism": review["mechanism"], "label": review["label"],
                **review["assessments"][family], "reviewed": True, "review_id": review["id"]}
    return {"mechanism": None, "label": "未复核", "match": "unknown", "reason": "该版本源码尚未复核。", "reviewed": False}


def annotate_state(state: dict, adds: dict, family: str, reviews: dict, job_key: str | None = None) -> dict:
    models = [model_review(pid, adds, family, reviews, job_key) for pid in state["ids"]]
    known = all(m["mechanism"] is not None for m in models)
    mechanism = "+".join(sorted({m["mechanism"] for m in models})) if known else None
    verdicts = {m["match"] for m in models}
    return {**state, "mechanism": mechanism, "label": " / ".join(sorted({m["label"] for m in models})),
            "match": next(iter(verdicts)) if len(verdicts) == 1 else "unknown",
            "reason": "；".join(dict.fromkeys(m["reason"] for m in models)),
            "reviewed": all(m["reviewed"] for m in models), "models": models}


def mean(values):
    return float(np.mean(values)) if len(values) else None


def median(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def episode(r: dict, reviews: dict, tail: int = 50) -> dict:
    method, game, family, param, seed = r["job_key"].split("|")
    if family not in TRUTH:
        raise ValueError(f"Unsupported true family: {family}")
    adds = added_models(r)
    beliefs, kl = r["belief"], r["kl_per_round"]
    if len(beliefs) != r["horizon"] or len(kl) != r["horizon"] or not np.isfinite(kl).all():
        raise ValueError(f"Incomplete belief/KL series: {r['job_key']}")
    decisions = [e for e in r["events"] if e.get("event") == "decision"]
    if not decisions or decisions[-1]["t"] >= len(kl):
        raise ValueError("A terminal post-decision interval is required")
    states = [annotate_state(dominant(b, adds, t), adds, family, reviews, r["job_key"])
              for t, b in enumerate(beliefs, 1)]
    final, late = states[-1], states[-tail:]
    comparable = final["mechanism"] is not None and all(s["mechanism"] is not None for s in late)
    stability = mean([s["mechanism"] == final["mechanism"] for s in late]) if comparable else None
    weights = [s["weight"] for s in late if s["mechanism"] == final["mechanism"]] if comparable else []
    trajectory, pending_retired, pending_auto = [], [], []
    di = 0
    for e in r["events"]:
        if e.get("event") == "drop":
            (pending_retired if e.get("why") == "retired by the LLM" else pending_auto).append(e)
        if e.get("event") != "decision":
            continue
        t, end = e["t"], decisions[di + 1]["t"] if di + 1 < len(decisions) else len(kl)
        start = decisions[di - 1]["t"] if di else 0
        active = [{**a, "source": "menu" if adds[a["pid"]]["source"] == "menu" else "own"} for a in e.get("active", [])]
        retired_own = [d["id"] for d in pending_retired if adds[d["id"]]["source"] != "menu"]
        code_ops = [o for o in e.get("ops", []) if o["op"] in ("create", "mutate") or
                    (o["op"] == "revise" and o.get("new_code"))]
        gains = [a["mean_ll"] - a["parent_mean_ll"] for a in r["events"] if a.get("event") == "archive_add"
                 and a.get("t") == t and a.get("ok") and a.get("parent_mean_ll") is not None]
        trajectory.append({"t": t, "ops": e.get("ops", []), "rationale": e.get("rationale", ""),
                           "error": e.get("error"), "active": active, "write_attempt": bool(code_ops),
                           "withdraw_own": bool(retired_own) and not any(a["source"] == "own" for a in active),
                           "retired_own": retired_own, "other_drops": pending_auto,
                           "variation_calls": sum(a.get("event") == "variation" and a.get("t") == t for a in r["events"]),
                           "parent_fit_gains": gains, "previous_kl": mean(kl[start:t]), "next_kl": mean(kl[t:end]),
                           "next_rounds": [t + 1, end], "interval_end": states[end - 1]})
        pending_retired, pending_auto = [], []
        di += 1
    final_active = [{"name": pid.rsplit("#", 1)[0], "pid": pid,
                     "source": "menu" if adds[pid]["source"] == "menu" else "own"}
                    for pid in active_ids(beliefs[-1], adds, len(beliefs))]
    return {"job_key": r["job_key"], "method": method, "game": game, "family": family, "param": param, "seed": seed,
            "available": bool(set(ORACLE_PROGRAMS[family]) & set(decisions[0]["menu"])),
            "truth": TRUTH[family], "bot_full": r["bot_full"], "final": final,
            "active_final": final_active, "source_mass_final": beliefs[-1].get("by_src", {}),
            "reference_mass_final": beliefs[-1]["other"], "tail_rounds": len(late), "stability": stability,
            "reviewed_tail_rounds": sum(s["reviewed"] for s in late),
            "tail_dominant_weight": {"first": weights[0] if weights else None, "last": weights[-1] if weights else None,
                                     "median": median(weights), "min": min(weights) if weights else None,
                                     "max": max(weights) if weights else None},
            "kl_all": mean(kl), "kl_stages": [mean(kl[a:b]) for a, b in ((0, 50), (50, 100), (100, 150))],
            "terminal_rounds": [decisions[-1]["t"] + 1, len(kl)], "terminal_kl": mean(kl[decisions[-1]["t"]:]),
            "trajectory": trajectory,
            "tail_states": [{"round": len(states) - len(late) + i + 1, **s} for i, s in enumerate(late)]}


def cross_seed(rows: list[dict]) -> list[dict]:
    cells = defaultdict(list)
    for r in rows:
        cells[(r["game"], r["family"], r["param"])].append(r)
    out = []
    for key, rs in sorted(cells.items()):
        rs = sorted(rs, key=lambda r: r["seed"])
        if len(rs) != 2 or {r["seed"] for r in rs} != {"s0", "s1"}:
            continue
        if rs[0]["bot_full"] != rs[1]["bot_full"]:
            raise ValueError(f"Different true bots across seeds: {key}")
        mechanisms = [r["final"]["mechanism"] for r in rs]
        out.append({"cell": "|".join(key), "keys": [r["job_key"] for r in rs],
                    "same_mechanism": mechanisms[0] == mechanisms[1] if all(mechanisms) else None,
                    "both_match": all(r["final"]["match"] == "match" for r in rs),
                    "mechanisms": mechanisms, "matches": [r["final"]["match"] for r in rs],
                    "terminal_kl": [r["terminal_kl"] for r in rs]})
    return out


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def cell(value):
    return str(value).replace("|", " / ").replace("\n", " ")


def short_key(r):
    return "|".join(r["job_key"].split("|")[1:])


def render(rows: list[dict], pairs: list[dict], run_dirs: list[Path], reviews_path: Path | None,
           out_path: Path, data_dir: Path) -> str:
    def link(path):
        return os.path.relpath(path, out_path.parent)

    out = ["# v6 终局建模与收敛：逐实例初步分析", "",
           f"数据：{', '.join(str(d) for d in run_dirs)}；方法 `{rows[0]['method']}`；{len(rows)} 局。", "",
           f"[逐实例 CSV]({link(data_dir / out_path.with_suffix('.csv').name)}) · "
           f"[完整指标与决策轨迹 JSON]({link(data_dir / out_path.with_suffix('.json').name)}) · "
           + (f"[源码复核表]({link(reviews_path)})" if reviews_path else "源码尚未复核"), "",
           "## 四格结果：来源不是成败标签", "",
           "初始可用性指真实结构类别是否在菜单中；两组均未向 LLM 揭示真实身份及参数。"
           "终局来源按末轮主导单模型划分，预测 KL 评价实际混合预测。结构匹配按源码复核；"
           "匹配只表示核心机制对应，参数及估计误差由 KL 反映。", "",
           "| 真实结构初始可用 | 终局来源 | 局数 | 匹配 / 部分 / 不匹配 / 未定 | 末段 KL 中位数 | 后50轮解释保持率中位数 |",
           "|---|---|---:|---|---:|---:|"]
    for avail in (True, False):
        for source in ("menu", "own", "reference", "mixed"):
            rs = [r for r in rows if r["available"] == avail and r["final"]["source"] == source]
            if not rs and source not in ("menu", "own"):
                continue
            counts = Counter(r["final"]["match"] for r in rs)
            out.append(f"| {'可用' if avail else '不可用'} | {SOURCES[source]} | {len(rs)} | "
                       + " / ".join(str(counts[k]) for k in MATCHES) + " | "
                       + fmt(median([r["terminal_kl"] for r in rs])) + " | "
                       + fmt(median([r["stability"] for r in rs])) + " |")
    out += ["", "小样本格子仅描述实例，不作路径优劣或因果推断。参考主导、跨来源并列如出现则单列。", "",
            "## 预测变化与解释稳定性", "",
            "| 初始可用性 | 局数 | 早期 KL 均值 | 中期 KL 均值 | 晚期 KL 均值 | 终局主导权重中位数 |",
            "|---|---:|---:|---:|---:|---:|"]
    for avail in (True, False):
        rs = [r for r in rows if r["available"] == avail]
        out.append(f"| {'可用' if avail else '不可用'} | {len(rs)} | " + " | ".join(
            fmt(mean([r["kl_stages"][i] for r in rs])) for i in range(3)) + " | "
            + fmt(median([r["final"]["weight"] for r in rs])) + " |")
    scored = [p for p in pairs if p["same_mechanism"] is not None]
    out += ["", f"同一真实对手配置的两个 seed：{sum(p['same_mechanism'] for p in scored)}/{len(scored)} 对终局结构标签一致；"
            f"{sum(p['both_match'] for p in pairs)}/{len(pairs)} 对均为结构匹配。相同标签也可能共同偏离真实结构。", "",
            "### 这些结果支持什么", ""]
    out_own = [r for r in rows if not r["available"] and r["final"]["source"] == "own"]
    matched = [r for r in out_own if r["final"]["match"] == "match"]
    if out_own:
        families = Counter(r["family"] for r in matched)
        detail = "；" + "、".join(f"{f} {n} 局" for f, n in sorted(families.items())) if families else ""
        out.append(f"- **自行扩展的核心结构匹配**：菜单外自写终局中 {len(matched)}/{len(out_own)} 局匹配"
                   + detail + "。这是终局采用结果，不是归档最佳候选。")
    wrong_stable = [r for r in rows if r["final"]["match"] == "mismatch" and r["stability"] == 1]
    if wrong_stable:
        r = sorted(wrong_stable, key=lambda r: r["job_key"])[0]
        out.append(f"- **稳定解释仍可能偏离真实机制**：例如 {cell(short_key(r))} 后50轮保持率为1，末段 KL="
                   f"{fmt(r['terminal_kl'])}，但其解释为“{r['final']['label']}”，真实机制为 {r['family']}。")
    out += ["- **预测变好与结构恢复分开判断**：早中晚 KL 描述实际预测变化；结构匹配来自代码中的条件和更新规则。"
            "同一历史上两种不同机制可能都预测得不错。", "",
            "## 逐实例结果", "",
            "编号按下表排序；完整理由、真实参数、激活集、权重和全部决策轨迹见同名 JSON；平表见同名 CSV。", "",
            "| # | 博弈 / 对手 / 参数 / seed | 可用 | 终局来源 | 终局解释 | 结构匹配 | 末段 KL | 保持率 | 主导权重 |",
            "|---:|---|---|---|---|---|---:|---:|---:|"]
    for i, r in enumerate(rows, 1):
        f = r["final"]
        out.append(f"| {i} | {cell(short_key(r))} | {'是' if r['available'] else '否'} | {SOURCES[f['source']]} | "
                   f"{cell(f['label'])}{'（并列）' if f['tie'] else ''} | {MATCHES[f['match']]} | "
                   f"{fmt(r['terminal_kl'])} | {fmt(r['stability'])} | {fmt(f['weight'])} |")
    out += ["", "### 四格中小样本实例的直接依据", ""]
    for r in rows:
        if r["final"]["source"] == "menu":
            out.append(f"- **{cell(short_key(r))}**：{r['final']['reason']} 末段 KL={fmt(r['terminal_kl'])}。")
    out += ["", "## 对局进展示例", "",
            "菜单内外各取整局 KL 最接近本组中位数的实例（同距按 job_key 排序），不按成功与否挑选。"
            "全部实例的操作及日志中保存的理由保留在 JSON。理由是 LLM 当时的解释，结构标签依据源码与真实规则比较。", ""]
    for avail in (True, False):
        rs = [r for r in rows if r["available"] == avail]
        if not rs:
            continue
        mid = median([r["kl_all"] for r in rs])
        r = min(rs, key=lambda r: (abs(r["kl_all"] - mid), r["job_key"]))
        out += [f"### {cell(short_key(r))}", "", f"真实机制：{r['truth']}。终局复核：{r['final']['reason']}", "",
                "| 决策后轮次 | 操作（失败标 ×） | 下一区间末主导模型 | 区间 KL：前 → 后 | 当次理由摘录 |",
                "|---:|---|---|---|---|"]
        for d in r["trajectory"]:
            ops = "; ".join(f"{o['op']} {o['name']}{'' if o.get('ok') else ' ×'}" for o in d["ops"]) or (
                "回复失败" if d["error"] else "保持")
            if d["withdraw_own"]:
                ops += "；撤回全部自写模型"
            reason = d["rationale"][:180] + ("…" if len(d["rationale"]) > 180 else "")
            out.append(f"| {d['t']} | {cell(ops)} | {cell(', '.join(d['interval_end']['ids']))} | "
                       f"{fmt(d['previous_kl'])} → {fmt(d['next_kl'])} | {cell(reason)} |")
        out.append("")
    out += ["", "## 同一真实配置的跨 seed 对照", "",
            "| 博弈 / 对手 / 参数 | s0 结构 → s1 结构 | 结构匹配 s0 / s1 | 末段 KL s0 / s1 |",
            "|---|---|---|---|"]
    for p in pairs:
        out.append(f"| {cell(p['cell'])} | {' → '.join(x or '未复核' for x in p['mechanisms'])} | "
                   f"{' / '.join(MATCHES[x] for x in p['matches'])} | {' / '.join(fmt(x) for x in p['terminal_kl'])} |")
    incomplete = [r for r in rows if r["stability"] is None]
    out += ["", "## 口径与复现", "",
            "- 早/中/晚为第 1–50 / 51–100 / 101–150 轮。末段从最后一个决策点的下一轮起，主臂为 137–150 轮。",
            "- 解释保持率 = 后 50 轮主导结构与终局主导结构相同的轮次比例。标签按源码机制合并，"
            "不是按名称合并；稳定而不匹配的解释仍照实计入稳定。",
            "- 一个实验实例是给定博弈、真实对手配置及seed的一局150轮对局。信念是系统对候选程序的权重，"
            "混合预测为这些程序输出的加权和；LLM负责生成和选择程序。",
            "- 权重由系统按似然及复杂度计算，不是 LLM 自述置信度。日志保存命名 MAP 和参考权重；"
            "可识别的两模型 50/50 并列保留共同解释，其余未记录的单模型权重不补造。"
            "同机制模型也会分摊权重，表中的单模型权重不是结构类别的总权重。",
            "- 结构匹配是初步源码标签：匹配=核心生成机制对应；部分=含关键成分但有遗漏或额外机制；"
            "不匹配=以其他统计规律代替；未定=信息不足。对可关闭的额外分量，未保存终局拟合参数时按部分匹配计。",
            "- 编写、采用、修改和退休是过程标记；预设 variation 与决策自主编写分开，自动删除不计主动撤回。"
            "父子拟合增益基于同段已有历史；区间 KL 是实际在线混合预测，两者不作为单次修改的因果效应。",
            f"- 后 50 轮结构复核不完整的实例：{len(incomplete)}；信息不足时解释保持率留空。"
            "参考模型的结构匹配记为未定，不能因预测好就判定恢复了真实机制。",
            f"- 源码复核表：`{reviews_path}`。其中保存实际源码、机制标签、各真实家族的比较依据，"
            "以源码内容及已复核实验实例匹配版本；不通过名称猜测结构。",
            "", "```bash", ".venv/bin/python scripts/analyze_outcomes.py " + " ".join(str(d) for d in run_dirs)
            + f" --method {rows[0]['method']}" + (f" --reviews {reviews_path}" if reviews_path else "")
            + f" --out {out_path} --data-dir {data_dir}", "```", ""]
    return "\n".join(out)


def write_outputs(out: Path, rows: list[dict], run_dirs: list[Path], reviews_path: Path | None,
                  data_dir: Path | None = None) -> None:
    pairs = cross_seed(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    data_dir = data_dir or out.parent
    data_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(render(rows, pairs, run_dirs, reviews_path, out, data_dir))
    (data_dir / out.with_suffix(".json").name).write_text(json.dumps({"run_dirs": [str(d) for d in run_dirs], "reviews": str(reviews_path),
                                                  "episodes": rows, "cross_seed": pairs}, ensure_ascii=False, indent=2) + "\n")
    fields = ["job_key", "available", "source", "models", "mechanism", "match", "reason", "terminal_kl", "stability",
              "dominant_weight", "early_kl", "middle_kl", "late_kl"]
    with (data_dir / out.with_suffix(".csv").name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({"job_key": r["job_key"], "available": r["available"], "source": r["final"]["source"],
                             "models": ";".join(r["final"]["ids"]), "mechanism": r["final"]["mechanism"],
                             "match": r["final"]["match"], "reason": r["final"]["reason"], "terminal_kl": r["terminal_kl"],
                             "stability": r["stability"], "dominant_weight": r["final"]["weight"],
                             **dict(zip(("early_kl", "middle_kl", "late_kl"), r["kl_stages"]))})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--method", default="v6_full_e30")
    ap.add_argument("--reviews", type=Path)
    ap.add_argument("--export-review", type=Path)
    ap.add_argument("--review-tail", type=int, choices=(50, 80), default=80,
                    help="Rounds requiring source review (paper: 80)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--data-dir", type=Path, help="JSON/CSV directory (default: alongside Markdown)")
    args = ap.parse_args()
    if not args.out and not args.export_review:
        ap.error("provide --out and/or --export-review")
    records = read_records(args.run_dirs, args.method)
    if any(r["horizon"] != 150 for r in records):
        ap.error("This stage analysis requires 150-round episodes")
    if args.export_review:
        if args.export_review.exists():
            ap.error("Review inventory already exists; choose a new file to preserve completed annotations")
        export_review(records, args.export_review, args.review_tail)
        print(f"Source review inventory -> {args.export_review}")
    if args.out:
        reviews = load_reviews(args.reviews)
        rows = [episode(r, reviews) for r in records]
        write_outputs(args.out, rows, args.run_dirs, args.reviews, args.data_dir)
        print(f"{len(rows)} episodes -> {args.out} (+ JSON, CSV)")


if __name__ == "__main__":
    main()
