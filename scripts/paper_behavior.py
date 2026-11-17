"""Paper checkpoint and candidate-funnel accounting (no stored run inputs)."""
from collections import Counter

BEHAVIORS = ("select", "activate", "write", "code_revision", "parameter_only", "retire")

def behavior_record(r: dict) -> dict:
    """Count checkpoints and link archive versions to their actual active IDs.

    A version is one accepted archive_add event, not a reusable name. Activation
    is linked through the decision op, source, grid and event order. A grid-only
    revision is counted separately and is not a newly generated program.
    """
    from analyze_outcomes import added_models, dominant

    events = r["events"]
    adds = added_models(r)
    add_indices = {e["id"]: i for i, e in enumerate(events) if e["event"] == "add" and e.get("ok")}
    candidates = {i: {**e, "active_ids": [], "dominant_rounds": []} for i, e in enumerate(events)
                  if e["event"] == "archive_add" and e.get("ok")}
    pid_to_version = {}
    parameter_ids = []
    counts = Counter()
    decisions = [e for e in events if e["event"] == "decision"]
    valid = [e for e in decisions if not e.get("error")]
    for i, e in enumerate(events):
        if e["event"] != "decision" or e.get("error"):
            continue
        ops = e["ops"]
        flags = {
            "select": any(o["op"] == "select" for o in ops),
            "activate": any(o["op"] == "activate" for o in ops),
            "write": any(o["op"] in ("create", "mutate") or
                         (o["op"] == "revise" and o.get("new_code")) for o in ops),
            "code_revision": any(o["op"] == "mutate" or
                                 (o["op"] == "revise" and o.get("new_code")) for o in ops),
            "parameter_only": any((o["op"] == "revise" and not o.get("new_code")) or
                                  (o["op"] == "select" and o.get("grid_changed")) for o in ops),
            "retire": any(o["op"] == "retire" for o in ops),
        }
        counts.update(k for k, v in flags.items() if v)
        for o in ops:
            pid = o.get("pid")
            if not o.get("ok") or pid is None or adds[pid]["source"] == "menu":
                continue
            if o["op"] == "revise" and not o.get("new_code"):
                parameter_ids.append(pid)
                continue
            a, ai = adds[pid], add_indices[pid]
            same = [(j, c) for j, c in candidates.items() if j < i and
                    c["src"] == a["src"] and c["param_grid"] == a["param_grid"]]
            if o["op"] == "revise":
                # _apply adds the active revision, drops the old ID, then archives
                # the revised source under a new name, all before decision logging.
                same = [(j, c) for j, c in same if ai < j and c["op"] == "revise" and c["t"] == e["t"]]
                assert len(same) == 1, (r["job_key"], o, "ambiguous revision")
                j, c = same[0]
            else:
                same = [(j, c) for j, c in same if j < ai and c["name"] == o["name"]]
                assert same, (r["job_key"], o, "unlinked activation")
                j, c = max(same, key=lambda item: item[0])
            assert pid not in pid_to_version
            pid_to_version[pid] = j
            c["active_ids"].append(pid)
    own_ids = {pid for pid, a in adds.items() if a["source"] != "menu"}
    assert own_ids == set(pid_to_version) | set(parameter_ids), r["job_key"]
    for t, b in enumerate(r["belief"], 1):
        for pid in dominant(b, adds, t)["ids"]:
            if pid in pid_to_version:
                candidates[pid_to_version[pid]]["dominant_rounds"].append(t)
    funnel = {}
    for stage in ("variation", "decision"):
        cs = [c for c in candidates.values() if c["call"].split(":")[0] == stage]
        funnel[stage] = {"accepted": len(cs), "adopted": sum(bool(c["active_ids"]) for c in cs),
                         "dominant": sum(bool(c["dominant_rounds"]) for c in cs)}
        assert funnel[stage]["dominant"] <= funnel[stage]["adopted"] <= funnel[stage]["accepted"]
    return {"planned": 10, "recorded": len(decisions), "parseable": len(valid),
            "counts": {k: counts[k] for k in BEHAVIORS}, "funnel": funnel,
            "versions": [{"event_index": i, **{k: c[k] for k in
                          ("t", "name", "op", "call", "parent", "active_ids", "dominant_rounds")}}
                         for i, c in candidates.items()], "parameter_only_ids": parameter_ids}
