from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import yaml


EXAMPLE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = EXAMPLE_ROOT.parent
EXAMPLE_CONFIG_PATH = EXAMPLE_ROOT / "config" / "config.yaml"

_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_PATH_SUFFIXES = (
    ".csv",
    ".tsv",
    ".txt",
    ".xlsx",
    ".xls",
    ".h5ad",
    ".png",
    ".pdf",
    ".yaml",
    ".yml",
    ".gmt",
    ".svg",
    ".jpg",
    ".jpeg",
)


def _looks_like_path(value: str) -> bool:
    if not value or _URL_RE.match(value):
        return False
    if value.startswith(("/", "./", "../")):
        return True
    if "\\" in value or "/" in value:
        return True
    return value.endswith(_PATH_SUFFIXES)


def _resolve_config_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_config_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_config_paths(item) for item in value]
    if isinstance(value, str) and _looks_like_path(value):
        path = Path(value)
        if path.is_absolute():
            return str(path)
        return str((PROJECT_ROOT / path).resolve(strict=False))
    return value


def load_example_config() -> dict[str, Any]:
    with EXAMPLE_CONFIG_PATH.open(encoding="utf-8") as handle:
        raw_cfg = yaml.safe_load(handle)
    return _resolve_config_paths(raw_cfg)


def write_dir() -> str:
    return f"{Path.home() / 'write'}/"


def repo_parent_dir() -> str:
    return f"{PROJECT_ROOT.parent}/"
