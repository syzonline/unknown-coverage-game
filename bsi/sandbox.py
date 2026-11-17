"""Restricted execution of LLM-written programs.

Three layers of defense:
1. AST validation: no imports, classes, dunder/private names, bare `except`, or dangerous builtins and attributes.
2. Restricted globals: a builtins whitelist plus a numpy facade exposing only pure numeric functions.
3. Worker processes with rlimits on address space, CPU and file size, a per-op SIGALRM timer, and a
   hard wall-clock kill from the parent.

Only float arrays cross the process boundary, because results are coerced inside the worker.
This blocks accidental damage and casual escapes. It is not a hardened boundary against a
determined adversary: do not grant extra privileges to the worker.
"""
from __future__ import annotations

import ast
import builtins
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import types
from itertools import cycle
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Sequence

import numpy as np

MAX_SOURCE_CHARS = 20_000
MAX_RESULT_SIZE = 10_000          # elements per call
MAX_PREFIX_ELEMENTS = 2_000_000   # elements per prefixes() op (16 MB of float64)

SAFE_BUILTINS = {
    n: getattr(builtins, n)
    for n in (
        "abs all any bool dict enumerate float int len list max min range reversed round set sorted "
        "sum tuple zip map filter isinstance pow divmod str "
        "ValueError IndexError KeyError ZeroDivisionError ArithmeticError Exception"
    ).split()
}

NP_WHITELIST = (
    "array asarray zeros ones full zeros_like ones_like full_like arange linspace eye "
    "exp log log1p log2 sqrt abs maximum minimum clip sum mean max min argmax argmin argsort sort "
    "where dot outer bincount cumsum cumprod prod power isfinite isnan nan_to_num roll concatenate stack "
    "unique count_nonzero any all allclose round floor ceil sign var std median tanh "
    "float64 int64 inf pi e newaxis"
).split()

FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "input", "__import__", "globals", "locals", "vars", "getattr",
    "setattr", "delattr", "breakpoint", "memoryview", "type", "object", "super", "help", "dir", "exit",
    "quit", "BaseException", "SystemExit", "KeyboardInterrupt", "GeneratorExit",
}
FORBIDDEN_ATTRS = {
    "tofile", "dump", "dumps", "ctypes", "load", "loads", "save", "savez", "savetxt", "loadtxt",
    "fromfile", "memmap", "lib", "ctypeslib", "testing", "f2py", "distutils", "os", "sys", "base",
    "gi_frame", "gi_code", "f_globals", "f_locals", "f_back", "tb_frame",
}
FORBIDDEN_NODES = (
    ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef,
    ast.Await, ast.AsyncFor, ast.AsyncWith, ast.With, ast.Delete,
)


class SandboxError(RuntimeError):
    pass


def validate_source(src: str, required: Sequence[str] = ("model",)) -> list[str]:
    """Static checks. Returns a list of problems (empty = ok)."""
    if len(src) > MAX_SOURCE_CHARS:
        return [f"source too long ({len(src)} > {MAX_SOURCE_CHARS} chars)"]
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    errs: list[str] = []
    defined: set[str] = set()
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, FORBIDDEN_NODES):
            errs.append(f"line {line}: {type(node).__name__} not allowed")
        elif isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith("__")):
            errs.append(f"line {line}: name {node.id!r} not allowed")
        elif isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRS):
            errs.append(f"line {line}: attribute {node.attr!r} not allowed")
        elif isinstance(node, ast.ExceptHandler) and node.type is None:
            errs.append(f"line {line}: bare except not allowed")
        elif isinstance(node, ast.FunctionDef):
            if node.name.startswith("_"):
                errs.append(f"line {line}: function name {node.name!r} not allowed")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("__"):
            errs.append(f"line {line}: dunder string constant not allowed")
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            defined.add(node.name)
    for name in required:
        if name not in defined:
            errs.append(f"missing top-level function {name!r}")
    return errs


def ast_complexity(src: str) -> int:
    """Description length proxy: number of AST nodes."""
    return sum(1 for _ in ast.walk(ast.parse(src)))


# ---------------------------------------------------------------- worker side


class _Timeout(BaseException):
    pass


def _on_alarm(signum: int, frame: Any) -> None:
    raise _Timeout()


def _internal_import(name: str, *args: Any, **kwargs: Any) -> Any:
    """numpy's C code lazily imports its own submodules through the *calling frame's* builtins.
    Allow exactly that. Programs cannot name `__import__` (rejected by validate_source)."""
    if name == "math" or name == "numpy" or name.startswith("numpy."):
        return builtins.__import__(name, *args, **kwargs)
    raise ImportError(f"import of {name!r} blocked in sandbox")


def _namespace() -> dict:
    safe_np = types.SimpleNamespace(**{n: getattr(np, n) for n in NP_WHITELIST})
    sb = dict(SAFE_BUILTINS)
    sb["__import__"] = _internal_import
    return {"__builtins__": sb, "np": safe_np, "math": math}


def _numeric(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.size > MAX_RESULT_SIZE:
        raise ValueError(f"result too large ({arr.size} elements)")
    return arr


def _timed(fn: Any, seconds: float) -> Any:
    signal.setitimer(signal.ITIMER_REAL, max(seconds, 1e-3))
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _worker_main(rconn: Any, wconn: Any, mem_mb: int, cpu_s: int) -> None:
    import resource

    # cwd is a private temp dir created (and later removed) by the parent; no file can be written anyway
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 2**20, mem_mb * 2**20))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    except (ValueError, OSError):
        pass
    signal.signal(signal.SIGALRM, _on_alarm)
    progs: dict[str, dict] = {}

    while True:
        try:
            msg = rconn.recv()
        except (EOFError, OSError):
            break
        op = msg[0]
        if op == "close":
            break
        try:
            if op == "load":
                _, pid, src, required, timeout = msg
                ns = _namespace()
                _timed(lambda: exec(compile(src, f"<prog:{pid}>", "exec"), ns), timeout)
                missing = [r for r in required if not callable(ns.get(r))]
                if missing:
                    raise ValueError(f"missing callables: {missing}")
                progs[pid] = ns
                reply: Any = ("ok", None)
            elif op == "unload":
                progs.pop(msg[1], None)
                reply = ("ok", None)
            elif op == "batch":
                _, items, timeout = msg
                results = []
                for pid, fname, args in items:
                    try:
                        fn = progs[pid][fname]
                        results.append(("ok", _timed(lambda: _numeric(fn(*args)), timeout)))
                    except _Timeout:
                        results.append(("err", "timeout"))
                    except Exception as e:  # noqa: BLE001
                        results.append(("err", f"{type(e).__name__}: {e}"))
                reply = ("ok", results)
            elif op == "prefixes":
                _, pid, fname, seq, params_list, extra, start, timeout, out_len = msg
                fn = progs[pid][fname]
                n_steps = len(seq) + 1 - start

                def run() -> np.ndarray:
                    # preallocate; reject a wrong-sized output at the first call instead of materializing it
                    out = None
                    for i, p in enumerate(params_list):
                        for j, t in enumerate(range(start, len(seq) + 1)):
                            v = _numeric(fn(seq[:t], p, *extra)).reshape(-1)
                            if out is None:
                                if out_len is not None and v.size != out_len:
                                    raise ValueError(f"output has {v.size} elements, expected {out_len}")
                                if len(params_list) * n_steps * v.size > MAX_PREFIX_ELEMENTS:
                                    raise ValueError("prefixes result too large")
                                out = np.empty((len(params_list), n_steps, v.size))
                            elif v.size != out.shape[2]:
                                raise ValueError(f"output has {v.size} elements, expected {out.shape[2]}")
                            out[i, j] = v
                    return out if out is not None else np.zeros((len(params_list), 0, out_len or 0))

                reply = ("ok", _timed(run, timeout))
            else:
                reply = ("err", f"unknown op {op!r}")
        except _Timeout:
            reply = ("err", "timeout")
        except Exception as e:  # noqa: BLE001
            reply = ("err", f"{type(e).__name__}: {e}")
        try:
            wconn.send(reply)
        except Exception as e:  # noqa: BLE001 - e.g. unpicklable/too large
            wconn.send(("err", f"reply failed: {type(e).__name__}: {e}"))


# ---------------------------------------------------------------- parent side


class _Proc:
    """A worker interpreter started with `python -m bsi.sandbox`, talking over two OS pipes.

    A fresh interpreter (not fork/spawn) avoids re-importing the caller's __main__ and does not
    inherit the parent's address space, so RLIMIT_AS only counts the worker's own memory.
    """

    def __init__(self, mem_mb: int, cpu_s: int):
        p2c_r, p2c_w = os.pipe()
        c2p_r, c2p_w = os.pipe()
        root = str(Path(__file__).resolve().parents[1])
        env = {
            **os.environ,
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONPATH": root + os.pathsep + os.environ.get("PYTHONPATH", ""),
        }
        self.cwd = tempfile.mkdtemp(prefix="bsi-sbx-")
        self.popen = subprocess.Popen(
            [sys.executable, "-m", "bsi.sandbox", str(p2c_r), str(c2p_w), str(mem_mb), str(cpu_s)],
            pass_fds=(p2c_r, c2p_w),
            cwd=self.cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )
        os.close(p2c_r)
        os.close(c2p_w)
        self.tx = Connection(p2c_w, readable=False)
        self.rx = Connection(c2p_r, writable=False)

    def alive(self) -> bool:
        return self.popen.poll() is None

    def kill(self) -> None:
        if self.alive():
            self.popen.kill()
        self.popen.wait(timeout=5)
        self.tx.close()
        self.rx.close()
        shutil.rmtree(self.cwd, ignore_errors=True)

    def shutdown(self) -> None:
        try:
            self.tx.send(("close",))
            self.popen.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            pass
        self.kill()


class SandboxPool:
    """Worker processes hosting loaded programs. Programs are pinned round-robin to workers;
    `call_batch` fans out to all workers in parallel."""

    def __init__(
        self,
        n_workers: int = 2,
        mem_mb: int = 768,
        cpu_s: int = 3600,
        call_timeout: float = 0.5,
        op_timeout_cap: float = 120.0,
        required: Sequence[str] = ("model",),
    ):
        self.n_workers = n_workers
        self.mem_mb = mem_mb
        self.cpu_s = cpu_s
        self.call_timeout = call_timeout
        self.op_timeout_cap = op_timeout_cap
        self.required = tuple(required)
        self._workers: list[_Proc | None] = [None] * n_workers
        self._assign: dict[str, int] = {}
        self._src: dict[str, str] = {}
        self._rr = cycle(range(n_workers))

    # -- lifecycle

    def _worker(self, i: int) -> _Proc:
        w = self._workers[i]
        if w is None or not w.alive():
            if w is not None:
                w.kill()
            self._workers[i] = _Proc(self.mem_mb, self.cpu_s)
            for pid, wi in list(self._assign.items()):
                if wi == i:
                    self._load_into(i, pid, self._src[pid])
        return self._workers[i]  # type: ignore[return-value]

    def _kill(self, i: int) -> None:
        w = self._workers[i]
        if w is not None:
            w.kill()
        self._workers[i] = None

    def close(self) -> None:
        for i, w in enumerate(self._workers):
            if w is not None:
                w.shutdown()
            self._workers[i] = None

    def __enter__(self) -> "SandboxPool":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # -- messaging

    def _hard(self, soft: float) -> float:
        return min(self.op_timeout_cap, soft) * 2 + 5.0

    def _send(self, i: int, msg: tuple) -> None:
        self._worker(i).tx.send(msg)

    def _recv(self, i: int, hard_timeout: float) -> tuple[str, Any]:
        w = self._workers[i]
        try:
            if w is None:
                raise SandboxError("no worker")
            if not w.rx.poll(hard_timeout):
                raise SandboxError("hard timeout")
            return w.rx.recv()
        except (EOFError, OSError, SandboxError) as e:
            self._kill(i)  # respawned lazily, with its programs reloaded
            return ("err", f"worker lost: {e}")

    def _request(self, i: int, msg: tuple, soft: float) -> tuple[str, Any]:
        try:
            self._send(i, msg)
        except (OSError, BrokenPipeError) as e:
            self._kill(i)
            return ("err", f"worker lost: {e}")
        return self._recv(i, self._hard(soft))

    # -- programs

    def _load_into(self, i: int, pid: str, src: str) -> tuple[str, Any]:
        w = self._workers[i]
        assert w is not None
        w.tx.send(("load", pid, src, self.required, 2.0))
        return self._recv(i, self._hard(2.0))

    def load(self, pid: str, src: str) -> list[str]:
        """Validate and load a program. Returns problems (empty = loaded)."""
        errs = validate_source(src, self.required)
        if errs:
            return errs
        if pid in self._assign:
            return [f"program id {pid!r} already loaded"]
        i = next(self._rr)
        self._worker(i)
        status, payload = self._load_into(i, pid, src)
        if status != "ok":
            return [str(payload)]
        self._assign[pid] = i
        self._src[pid] = src
        return []

    def unload(self, pid: str) -> None:
        i = self._assign.pop(pid, None)
        self._src.pop(pid, None)
        w = self._workers[i] if i is not None else None
        if w is not None and w.alive():
            self._request(i, ("unload", pid), 1.0)

    def loaded(self) -> list[str]:
        return list(self._assign)

    # -- calls

    def call(self, pid: str, fname: str, args: tuple, timeout: float | None = None) -> np.ndarray:
        res = self.call_batch([(pid, fname, args)], timeout)[0]
        if isinstance(res, SandboxError):
            raise res
        return res

    def call_batch(
        self, items: Sequence[tuple[str, str, tuple]], timeout: float | None = None
    ) -> list[np.ndarray | SandboxError]:
        """Evaluate many (program, function, args) calls; workers run their shares in parallel."""
        timeout = timeout or self.call_timeout
        out: list[np.ndarray | SandboxError] = [SandboxError("not run")] * len(items)
        groups: dict[int, list[int]] = {}
        for k, (pid, _, _) in enumerate(items):
            if pid not in self._assign:
                out[k] = SandboxError(f"unknown program {pid!r}")
            else:
                groups.setdefault(self._assign[pid], []).append(k)
        sent: dict[int, list[int]] = {}
        for i, ks in groups.items():
            try:
                self._send(i, ("batch", [items[k] for k in ks], timeout))
                sent[i] = ks
            except (OSError, BrokenPipeError) as e:
                self._kill(i)
                for k in ks:
                    out[k] = SandboxError(f"worker lost: {e}")
        for i, ks in sent.items():
            status, payload = self._recv(i, self._hard(timeout * len(ks)))
            if status != "ok":
                for k in ks:
                    out[k] = SandboxError(str(payload))
                continue
            for k, (st, val) in zip(ks, payload):
                out[k] = val if st == "ok" else SandboxError(val)
        return out

    def prefixes(
        self,
        pid: str,
        fname: str,
        seq: Sequence,
        params_list: Sequence[dict],
        extra: tuple = (),
        start: int = 0,
        timeout: float | None = None,
        out_len: int | None = None,
    ) -> np.ndarray:
        """out[p, t - start] = fname(seq[:t], params_list[p], *extra) for t = start..len(seq).

        out_len: expected number of elements per call; a mismatch aborts the op inside the worker."""
        if pid not in self._assign:
            raise SandboxError(f"unknown program {pid!r}")
        n_calls = len(params_list) * (len(seq) + 1 - start)
        soft = min(self.op_timeout_cap, timeout or self.call_timeout * max(n_calls, 1))
        i = self._assign[pid]
        status, payload = self._request(i, ("prefixes", pid, fname, tuple(seq), list(params_list), extra, start, soft, out_len), soft)
        if status != "ok":
            raise SandboxError(str(payload))
        return payload


if __name__ == "__main__":
    _r, _w, _mem, _cpu = (int(x) for x in sys.argv[1:5])
    _worker_main(Connection(_r, writable=False), Connection(_w, readable=False), _mem, _cpu)
