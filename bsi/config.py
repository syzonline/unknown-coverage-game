"""API and model-role configuration.

The API key is never stored in code or configs. Resolution order:
1. env BSI_API_KEY + BSI_BASE_URL
2. file given by env BSI_API_KEY_FILE
3. the first `api-key.txt` found walking up from the project root

An explicit profile reads `<profile>-key` and `<profile>-url` from that file,
independently of the default provider's environment variables.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class APIConfig:
    base_url: str
    api_key: str

    def __repr__(self) -> str:  # never print the full key
        return f"APIConfig(base_url={self.base_url!r}, api_key='{self.api_key[:5]}…')"


def normalize_base_url(url: str) -> str:
    """Accept a provider root, /v1 base, or full chat-completions endpoint."""
    url = url.rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[:-len("/chat/completions")]
    return url if url.endswith("/v1") else url + "/v1"


def find_key_file() -> Path | None:
    env = os.environ.get("BSI_API_KEY_FILE")
    if env:
        return Path(env)
    for d in (PROJECT_ROOT, *PROJECT_ROOT.parents):
        p = d / "api-key.txt"
        if p.is_file():
            return p
    return None


def load_api_config(path: str | Path | None = None, *, profile: str | None = None) -> APIConfig:
    if profile is not None:
        p = Path(path) if path else find_key_file()
        if p is None or not p.is_file():
            raise FileNotFoundError(f"no API config file for profile {profile!r}")
        text = p.read_text(encoding="utf-8")
        prefix = re.escape(profile)
        mk = re.search(rf"^\s*{prefix}-key:\s*(\S+)", text, re.M)
        mb = re.search(rf"^\s*{prefix}-url:\s*(\S+)", text, re.M)
        if not mk or not mb:
            raise ValueError(f"API profile {profile!r} incomplete (need {profile}-key and {profile}-url)")
        return APIConfig(normalize_base_url(mb.group(1)), mk.group(1))
    key = os.environ.get("BSI_API_KEY")
    base = os.environ.get("BSI_BASE_URL")
    if not (key and base):
        p = Path(path) if path else find_key_file()
        if p is None or not p.is_file():
            raise FileNotFoundError("no API config: set BSI_API_KEY/BSI_BASE_URL or provide api-key.txt")
        text = p.read_text(encoding="utf-8")
        mk = re.search(r"^\s*key:\s*(\S+)", text, re.M)
        mb = re.search(r"^\s*address:\s*(\S+)", text, re.M)
        key = key or (mk.group(1) if mk else None)
        base = base or (mb.group(1) if mb else None)
    if not key or not base:
        raise ValueError("API config incomplete (need key and address)")
    return APIConfig(normalize_base_url(base), key)


def load_models_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else PROJECT_ROOT / "configs" / "models.yaml"
    if not p.is_file():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
