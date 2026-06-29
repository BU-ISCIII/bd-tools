"""Centralized config loading for staged modelling package."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PACKAGE_DIR / "config"


def _load_json(name: str) -> Dict[str, Any]:
    path = CONFIG_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def runtime_defaults() -> Dict[str, Any]:
    return _load_json("runtime_defaults.json")


@lru_cache(maxsize=1)
def algorithm_params() -> Dict[str, Any]:
    return _load_json("algorithm_params.json")
