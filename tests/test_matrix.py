"""Matrix runner: job grid, pairing, resume, and crash tolerance (non-LLM methods only)."""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_matrix as RM  # noqa: E402


def cfg(**kw):
    base = {"name": "t", "split": "dev", "games": ["rps3", "ipd"], "families": ["fixed", "switching"], "n_params": 2,
            "seeds": [0, 1], "horizon": 20, "sandbox_threads": 2, "llm_threads": 2,
            "methods": {"random": {}, "freq_br": {}, "oracle_br": {}, "bsi_seedlib": {}, "b0_llm_direct": {"params": [0], "seeds": [0]}}}
    base.update(kw)
    return base


def test_job_grid_and_method_subsets():
    c = cfg()
    jobs = RM.build_jobs(c, list(c["methods"]))
    assert sum(j.method == "random" for j in jobs) == 2 * 2 * 2 * 2
    assert sum(j.method == "b0_llm_direct" for j in jobs) == 2 * 2
    assert len({j.key for j in jobs}) == len(jobs)


def test_same_bot_and_env_seed_across_methods():
    c = cfg()
    a = RM.Job("random", "rps3", "switching", 1, 0)
    b = RM.Job("bsi", "rps3", "switching", 1, 0)
    _, bot_a = RM.make_bot(c, a)
    _, bot_b = RM.make_bot(c, b)
    assert bot_a.describe() == bot_b.describe() and a.episode == b.episode
    _, bot_c = RM.make_bot(c, RM.Job("random", "rps3", "switching", 0, 0))
    assert bot_c.describe() != bot_a.describe()


def test_read_jsonl_tolerates_truncated_line(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text('{"job_key": "a", "status": "ok"}\n{"job_key": "b", "sta')
    assert [r["job_key"] for r in RM.read_jsonl(p)] == ["a"]


def _run(config_path, run_dir, *extra):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "run_matrix.py"), "--config", str(config_path),
                           "--run-dir", str(run_dir), *extra], capture_output=True, text=True, timeout=600, cwd=ROOT)


def test_end_to_end_resume_and_config_guard(tmp_path):
    c = cfg()
    c["methods"] = {k: v for k, v in c["methods"].items() if k != "b0_llm_direct"}  # no API in unit tests
    cp = tmp_path / "c.yaml"
    cp.write_text(yaml.safe_dump(c))
    run = tmp_path / "run"
    r1 = _run(cp, run)
    assert r1.returncode == 0, r1.stdout[-2000:] + r1.stderr[-2000:]
    rows = [json.loads(l) for l in (run / "summaries.jsonl").read_text().splitlines()]
    assert len(rows) == 4 * 16 and all(r["status"] == "ok" for r in rows)
    assert (run / "report.md").exists()
    # paired: oracle and random share env seeds -> identical opponent actions only if bot ignores agent (fixed)
    fixed = [r for r in rows if r["family"] == "fixed" and r["episode"] == "rps3|fixed|p0|s0"]
    assert len(fixed) == 4
    r2 = _run(cp, run)
    assert "0 pending" in r2.stdout and len((run / "summaries.jsonl").read_text().splitlines()) == 64
    c["horizon"] = 30
    cp.write_text(yaml.safe_dump(c))
    r3 = _run(cp, run)
    assert r3.returncode != 0 and "different config" in (r3.stdout + r3.stderr)


def test_report_pools_ece_across_games_and_scores_adaptation(tmp_path):
    from analyze_matrix import analyze

    c = cfg(games=["rps3", "ipd"], families=["fixed", "switching"], n_params=1, seeds=[0], horizon=40,
            methods={"freq_br": {}, "bsi_seedlib": {}, "oracle_br": {}})
    cp = tmp_path / "c.yaml"
    cp.write_text(yaml.safe_dump(c))
    r = _run(cp, tmp_path / "run")
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    text = analyze(tmp_path / "run")
    assert "ECE(pooled)" in text
    # adaptation row: switching eps | post KL | excess KL | recovered | applicable -- oracle tracks -> applicable
    assert re.search(r"\| oracle_br \| 2 \| [^|]+ \| [^|]+ \| 1\.00 \| 2 \|", text), text


def test_memory_watchdog_stops_cleanly(tmp_path):
    c = cfg(games=["rps3"], families=["fixed", "cycle", "wsls", "klevel"], n_params=2, seeds=[0, 1, 2], horizon=150,
            methods={"bsi_seedlib": {}}, max_rss_mb=10, sandbox_threads=1)
    cp = tmp_path / "c.yaml"
    cp.write_text(yaml.safe_dump(c))
    r = _run(cp, tmp_path / "run")
    assert r.returncode == 3, r.stdout[-1500:] + r.stderr[-1500:]
    assert "memory watchdog" in r.stdout
    done = (tmp_path / "run" / "summaries.jsonl").read_text().splitlines()
    assert len(done) < 24  # stopped early; the rest stays pending for resume
