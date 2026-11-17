"""Run directories and streaming trace storage (gzip JSONL)."""
from __future__ import annotations

import gzip
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

import numpy as np

MAX_RECORD_BYTES = 200_000


def _default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=_default, ensure_ascii=False, sort_keys=True)


def config_hash(cfg: Any, n: int = 12) -> str:
    return hashlib.sha256(dumps(cfg).encode("utf-8")).hexdigest()[:n]


def make_run_dir(root: str | Path, exp: str, cfg: dict, run_id: str | None = None) -> Path:
    run_id = run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{config_hash(cfg, 6)}-{uuid.uuid4().hex[:4]}"
    d = Path(root) / exp / run_id
    d.mkdir(parents=True, exist_ok=False)
    (d / "config.json").write_text(
        dumps({"config": cfg, "config_hash": config_hash(cfg), "created": time.time()}), encoding="utf-8"
    )
    return d


class TraceWriter:
    """Append-only gzip JSONL. Records above `max_bytes` go to an overflow file; a stub stays inline."""

    def __init__(self, path: str | Path, max_bytes: int = MAX_RECORD_BYTES):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self._fh = gzip.open(self.path, "at", encoding="utf-8")
        self._n = 0

    def write(self, record: dict) -> None:
        line = dumps(record)
        if len(line.encode("utf-8")) > self.max_bytes:
            ov_dir = self.path.parent / "overflow"
            ov_dir.mkdir(exist_ok=True)
            ov = ov_dir / f"{self.path.stem}-{self._n}-{uuid.uuid4().hex[:6]}.json.gz"
            with gzip.open(ov, "wt", encoding="utf-8") as f:
                f.write(line)
            line = dumps({"_truncated": True, "_overflow": str(ov), "_preview": line[:2000]})
        self._fh.write(line + "\n")
        self._n += 1

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def iter_traces(path: str | Path, resolve_overflow: bool = False) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if resolve_overflow and rec.get("_truncated"):
                with gzip.open(rec["_overflow"], "rt", encoding="utf-8") as g:
                    rec = json.loads(g.read())
            yield rec


def log_holdout_access(run_dir: str | Path, who: str, note: str = "") -> None:
    """Holdout sets are read once per final evaluation; every access is logged so violations are visible."""
    with open(Path(run_dir) / "holdout_access.log", "a", encoding="utf-8") as f:
        f.write(dumps({"time": time.time(), "who": who, "note": note}) + "\n")
