"""Outcome analysis: provenance, temporal alignment, review coverage and trace loading."""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_outcomes as A  # noqa: E402
from bsi.store import TraceWriter  # noqa: E402


def add(pid="reactive#1", source="menu", t=0, src=None):
    return {"event": "add", "id": pid, "source": source, "t": t, "ok": True,
            "src": src if src is not None else A.MENU["reactive"].src}


def belief(pid="reactive#1", source="menu", weight=0.9, other=0.1, active=None, n=1):
    return {"map": pid, "map_src": source, "map_w": weight, "other": other,
            "by_src": {source: weight}, "n": n,
            "active": active if active is not None else [pid.rsplit("#", 1)[0]]}


def decision(t, pid="reactive#1", ops=None):
    return {"event": "decision", "t": t, "menu": list(A.MENU), "ops": ops or [],
            "active": [{"name": pid.rsplit("#", 1)[0], "pid": pid, "origin": "menu"}],
            "rationale": "recorded evidence"}


def record():
    kl = [1.0] * 136 + [0.2] * 14
    return {"job_key": "v6_full_e30|rps3|reactive|p0|s0", "method": "v6_full_e30", "horizon": 150,
            "bot_full": {"family": "reactive", "params": {"table": [0, 1, 2]}},
            "events": [add(), decision(10), decision(136)], "belief": [belief() for _ in range(150)],
            "kl_per_round": kl, "kl": sum(kl) / 150, "payoff_per_round": 0.5, "regret": 0.1}


def own_reviews(src="own code", mechanism="reactive", match="match"):
    return {src: {"id": "test", "source": src, "mechanism": mechanism, "label": "已复核解释",
                  "assessments": {"reactive": {"match": match, "reason": "code evidence"}}}}


def test_terminal_interval_and_stages_use_actual_rounds():
    r = A.episode(record(), {})
    assert r["terminal_rounds"] == [137, 150]
    assert r["terminal_kl"] == pytest.approx(0.2)
    assert r["kl_stages"] == pytest.approx([1, 1, (36 + 14 * 0.2) / 50])
    assert r["trajectory"][-1]["previous_kl"] == 1
    assert r["trajectory"][-1]["next_kl"] == pytest.approx(0.2)
    assert r["stability"] == 1


@pytest.mark.parametrize("new_code", [True, False])
def test_menu_rewrite_vs_grid_only_revision_uses_add_source(new_code):
    r = record()
    src, source = ("own code", "revised") if new_code else (A.MENU["reactive"].src, "menu")
    new = add("reactive#2", source, 136, src)
    r["events"][-1:] = [new, {"event": "drop", "id": "reactive#1", "why": "revised"},
                         decision(136, "reactive#2", [{"op": "revise", "name": "reactive", "new_code": new_code, "ok": True}])]
    r["belief"][136:] = [belief("reactive#2", source) for _ in range(14)]
    out = A.episode(r, own_reviews())
    assert out["final"]["source"] == ("own" if new_code else "menu")
    assert out["trajectory"][-1]["write_attempt"] == new_code
    # Menu and rewritten code share a reviewed mechanism: a new pid isn't a new conclusion.
    assert out["stability"] == 1
    assert out["final"]["match"] == "match"


def test_same_name_different_unreviewed_source_does_not_inherit_label():
    r = record()
    r["events"][0] = add(source="created", src="different code")
    out = A.episode(r, own_reviews())
    assert out["final"]["match"] == "unknown"
    assert out["stability"] is None


def test_review_of_specific_response_table_is_not_reused_for_another_instance():
    r = record()
    r["events"][0] = add(source="created", src="own code")
    reviews = own_reviews()
    reviews["own code"]["uses"] = [{"job_key": r["job_key"].replace("p0", "p1"), "pid": "reactive#1"}]
    assert A.episode(r, reviews)["final"]["match"] == "unknown"


def test_family_comes_from_experiment_not_bot_describe_alias():
    r = record()
    r["job_key"] = "v6_full_e30|rps3|fictitious|p0|s0"
    r["bot_full"] = {"family": "klevel", "params": {"k": 1}}
    out = A.episode(r, {})
    assert out["family"] == "fictitious"
    assert out["available"] is True


def test_reference_competes_with_named_map():
    r = record()
    r["belief"][-1] = belief(weight=0.2, other=0.8)
    out = A.episode(r, {})
    assert out["final"]["source"] == "reference"
    assert out["final"]["match"] == "unknown"
    assert out["final"]["weight"] == 0.8


def test_identical_code_tie_does_not_change_structure_identity():
    r = record()
    r["events"][0:1] = [add("a#1", "created", src="own code"), add("b#2", "mutated", src="own code")]
    for d in r["events"]:
        if d["event"] == "decision":
            d["active"] = [{"name": name, "pid": pid, "origin": "created"}
                           for name, pid in (("a", "a#1"), ("b", "b#2"))]
    r["belief"] = [belief("a#1", "created", 0.5, 0, ["a", "b"], 2) for _ in range(150)]
    out = A.episode(r, own_reviews())
    assert out["final"]["tie"]
    assert out["final"]["ids"] == ["a#1", "b#2"]
    assert out["final"]["source"] == "own"
    assert out["final"]["mechanism"] == "reactive"
    assert out["stability"] == 1


def test_decision_round_belief_precedes_replacement():
    adds = {e["id"]: e for e in [add("a#1", "created", 10, "old"), add("b#2", "created", 10, "b"),
                                 add("a#3", "revised", 136, "new")]}
    b = belief("a#1", "created", 0.5, 0, ["a", "b"], 2)
    assert A.dominant(b, adds, 136)["ids"] == ["a#1", "b#2"]
    b["map"] = "a#3"
    assert A.dominant(b, adds, 137)["ids"] == ["a#3", "b#2"]


@pytest.mark.parametrize("reason, keep_own, expected", [
    ("retired by the LLM", False, True),
    ("retired by the LLM", True, False),
    ("failed during update/refit", False, False),
    ("revised", False, False),
])
def test_only_explicit_own_retirement_to_no_own_counts_as_withdrawal(reason, keep_own, expected):
    r = record()
    r["events"].insert(1, add("own#2", "created", 10, "own code"))
    r["events"].insert(-1, {"event": "drop", "id": "own#2", "why": reason})
    if keep_own:
        r["events"][-1]["active"].append({"name": "own", "pid": "own#2", "origin": "created"})
    assert A.episode(r, own_reviews())["trajectory"][-1]["withdraw_own"] == expected


def test_automatic_final_drop_is_not_in_final_active_set():
    r = record()
    r["belief"][-1] = belief(None, "menu", 0, 1, [], 0)
    r["events"].append({"event": "drop", "id": "reactive#1", "why": "failed during update/refit"})
    out = A.episode(r, {})
    assert out["active_final"] == []
    assert not out["trajectory"][-1]["withdraw_own"]


def test_repair_attempts_are_one_decision_and_not_a_variation():
    r = record()
    r["events"][-1]["ops"] = [{"op": "create", "name": "new", "ok": False},
                                {"op": "create", "name": "new", "ok": True}]
    r["events"].insert(-1, {"event": "variation", "t": 136, "ops": [{"op": "mutate", "ok": True}]})
    d = A.episode(r, {})["trajectory"][-1]
    assert d["write_attempt"] is True and d["variation_calls"] == 1
    r["events"][-1]["ops"] = []
    assert A.episode(r, {})["trajectory"][-1]["write_attempt"] is False


def write_run(path, r, overflow=False):
    path.mkdir()
    summary = {**r, "status": "ok"}
    (path / "summaries.jsonl").write_text(json.dumps(summary) + "\n")
    with TraceWriter(path / "traces-1.jsonl.gz", max_bytes=20 if overflow else 1_000_000) as w:
        w.write(r)


def test_loader_resolves_overflow_and_deduplicates_inputs(tmp_path):
    d = tmp_path / "run"
    write_run(d, record(), overflow=True)
    assert len(list((d / "overflow").glob("*"))) == 1
    loaded = A.read_records([d, d], "v6_full_e30")
    assert len(loaded) == 1 and len(loaded[0]["belief"]) == 150


def test_missing_successful_trace_is_an_error(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "summaries.jsonl").write_text(json.dumps({**record(), "status": "ok"}) + "\n")
    with pytest.raises(ValueError, match="Missing successful traces"):
        A.read_records([d], "v6_full_e30")


def test_cross_seed_compares_mechanism_not_source_and_checks_true_bot():
    first = A.episode(record(), {})
    second = copy.deepcopy(first)
    second.update(seed="s1", job_key=first["job_key"].replace("s0", "s1"))
    second["final"]["source"] = "own"
    pair = A.cross_seed([first, second])[0]
    assert pair["same_mechanism"] and pair["both_match"]
    second["bot_full"]["params"]["table"] = [1, 2, 0]
    with pytest.raises(ValueError, match="Different true bots"):
        A.cross_seed([first, second])


def test_report_outputs_and_relative_data_links(tmp_path):
    out = tmp_path / "docs" / "outcomes.md"
    data = tmp_path / "data"
    A.write_outputs(out, [A.episode(record(), {})], [], None, data)
    assert (data / "outcomes.csv").exists()
    assert len(json.loads((data / "outcomes.json").read_text())["episodes"]) == 1
    text = out.read_text()
    assert "../data/outcomes.csv" in text and "137–150" in text
    assert "匹配 / 部分 / 不匹配 / 未定" in text
