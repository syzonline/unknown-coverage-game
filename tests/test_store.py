import numpy as np

from bsi.store import TraceWriter, config_hash, iter_traces, make_run_dir


def test_config_hash_stable_and_order_free():
    assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_roundtrip_and_overflow(tmp_path):
    run = make_run_dir(tmp_path, "exp", {"x": 1})
    path = run / "traces.jsonl.gz"
    with TraceWriter(path, max_bytes=1000) as w:
        w.write({"i": 0, "arr": np.arange(3)})
        w.write({"i": 1, "big": "x" * 5000})
    recs = list(iter_traces(path))
    assert recs[0]["arr"] == [0, 1, 2]
    assert recs[1]["_truncated"]
    full = list(iter_traces(path, resolve_overflow=True))
    assert full[1]["big"] == "x" * 5000
