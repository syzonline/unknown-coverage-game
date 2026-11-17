import numpy as np
import pytest

from bsi.sandbox import SandboxError, validate_source

OK = "def model(h, p, g):\n    return np.ones(g['n']) / g['n']\n"


@pytest.mark.parametrize("src,needle", [
    ("import os\ndef model(h,p,g):\n    return [1]\n", "Import"),
    ("def model(h,p,g):\n    return open('x').read()\n", "open"),
    ("def model(h,p,g):\n    return h.__class__\n", "__class__"),
    ("def model(h,p,g):\n    return np.load('x')\n", "load"),
    ("def model(h,p,g):\n    try:\n        return 1\n    except:\n        pass\n", "bare except"),
    ("class A:\n    pass\ndef model(h,p,g):\n    return 1\n", "ClassDef"),
    ("def helper(x):\n    return x\n", "missing"),
    ("def model(h,p,g):\n    return getattr(np, 'ones')(3)\n", "getattr"),
])
def test_validator_rejects(src, needle):
    errs = validate_source(src)
    assert errs and any(needle in e for e in errs), errs


def test_validator_accepts_ok():
    assert validate_source(OK) == []


def test_call_batch_and_prefixes(pool):
    assert pool.load("t_ok", OK) == []
    out = pool.call("t_ok", "model", ((), {}, {"n": 4}))
    assert np.allclose(out, 0.25)
    seq = tuple((i % 2, i % 3) for i in range(10))
    pre = pool.prefixes("t_ok", "model", seq, [{}, {}], extra=({"n": 3},), start=2)
    assert pre.shape == (2, 9, 3)
    pool.unload("t_ok")


def test_timeout_and_recovery(pool):
    assert pool.load("t_loop", "def model(h,p,g):\n    while True:\n        pass\n") == []
    assert pool.load("t_ok2", OK) == []
    res = pool.call_batch([("t_loop", "model", ((), {}, {})), ("t_ok2", "model", ((), {}, {"n": 2}))])
    assert isinstance(res[0], SandboxError) and "timeout" in str(res[0])
    assert np.allclose(res[1], 0.5)
    with pytest.raises(SandboxError):
        pool.prefixes("t_loop", "model", ((0, 0),) * 5, [{}], timeout=0.3)
    assert np.allclose(pool.call("t_ok2", "model", ((), {}, {"n": 2})), 0.5)
    pool.unload("t_loop")
    pool.unload("t_ok2")


def test_memory_and_runtime_errors_are_contained(pool):
    assert pool.load("t_mem", "def model(h,p,g):\n    return np.ones(500000000)[:2]\n") == []
    assert pool.load("t_err", "def model(h,p,g):\n    return 1 / 0\n") == []
    res = pool.call_batch([("t_mem", "model", ((), {}, {})), ("t_err", "model", ((), {}, {}))])
    assert "MemoryError" in str(res[0]) and "ZeroDivisionError" in str(res[1])
    pool.unload("t_mem")
    pool.unload("t_err")


def test_cannot_import_at_runtime(pool):
    # numpy internals may import numpy submodules; anything else must be blocked even if reached indirectly
    from bsi.sandbox import _internal_import
    with pytest.raises(ImportError):
        _internal_import("os")


def test_no_file_writes(pool):
    src = "def model(h,p,g):\n    a = np.ones(3)\n    a.tofile('x')\n    return a\n"
    assert validate_source(src)  # rejected statically


def test_worker_temp_dirs_are_removed():
    import glob
    import tempfile

    from bsi.sandbox import SandboxPool

    pattern = f"{tempfile.gettempdir()}/bsi-sbx-*"
    before = set(glob.glob(pattern))
    with SandboxPool(n_workers=2) as p:
        assert p.load("x", OK) == []
        assert p.load("loop", "def model(h,p,g):\n    while True:\n        pass\n") == []
        p._kill(0)  # simulate a crashed/killed worker; it is respawned lazily
        assert np.allclose(p.call("x", "model", ((), {}, {"n": 2})), 0.5)
    assert set(glob.glob(pattern)) - before == set()


def test_prefixes_rejects_wrong_output_size_early(pool):
    import time

    assert pool.load("t_big", "def model(h, p, g):\n    return np.ones(1000) / 1000\n") == []
    t0 = time.time()
    with pytest.raises(SandboxError, match="expected 3"):
        pool.prefixes("t_big", "model", ((0, 0),) * 150, [{}] * 96, extra=({},), out_len=3)
    assert time.time() - t0 < 2.0  # fails on the first call, nothing large is built or shipped
    with pytest.raises(SandboxError, match="too large"):
        pool.prefixes("t_big", "model", ((0, 0),) * 150, [{}] * 96, extra=({},))
    pool.unload("t_big")
