"""YAML config loader. Caches per file so tests + build pipeline stay fast."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


@cache
def load_config(name: str) -> dict[str, Any]:
    """Load `config/<name>.yaml`. Cached. Raises FileNotFoundError if absent."""
    path = CONFIG_DIR / f"{name}.yaml"
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config {name} must be a mapping at root, got {type(data).__name__}")
    return data
