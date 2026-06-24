"""Centralized config loading for three-level modelling package."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent

MODELLING_CONFIG_DIR = PACKAGE_DIR / "config"
PREPROCESS_CONFIG_DIR = PROJECT_DIR / "preprocess" / "config"


def _load_json(
    name: str,
    *,
    config_dir: Path = MODELLING_CONFIG_DIR,
) -> Dict[str, Any]:
    path = config_dir / name

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Expected a JSON object in {path}, "
            f"received {type(config).__name__}."
        )

    return config


def _load_yaml(
    name: str,
    *,
    config_dir: Path,
) -> Dict[str, Any]:
    path = config_dir / name

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Configuration file is empty: {path}")

    if not isinstance(config, dict):
        raise ValueError(
            f"Expected a YAML mapping in {path}, "
            f"received {type(config).__name__}."
        )

    return config


@lru_cache(maxsize=1)
def runtime_defaults() -> Dict[str, Any]:
    return _load_json("runtime_defaults.json")


@lru_cache(maxsize=1)
def data_processing_config() -> Dict[str, Any]:
    config = _load_json("data_processing.json")

    if "focus_map" in config:
        config["focus_map"] = {
            int(key): value
            for key, value in config["focus_map"].items()
        }

    return config


@lru_cache(maxsize=1)
def algorithm_params() -> Dict[str, Any]:
    return _load_json("algorithm_params.json")


@lru_cache(maxsize=1)
def optuna_search_spaces() -> Dict[str, Any]:
    return _load_json("optuna_search_spaces.json")


@lru_cache(maxsize=1)
def feature_view_definitions() -> Dict[str, Any]:
    """
    Load preprocessing feature-view definitions.

    Expected location:
        <project>/preprocess/config/preprocess_feature_views.yml
    """
    config = _load_yaml(
        "preprocess_feature_views.yml",
        config_dir=PREPROCESS_CONFIG_DIR,
    )

    feature_views = config.get("feature_views")
    feature_groups = config.get("feature_groups")

    if not isinstance(feature_views, list):
        raise ValueError(
            "preprocess_feature_views.yml must contain a list "
            "named 'feature_views'."
        )

    if not isinstance(feature_groups, dict):
        raise ValueError(
            "preprocess_feature_views.yml must contain a mapping "
            "named 'feature_groups'."
        )

    return config


@lru_cache(maxsize=1)
def modelling_feature_views() -> Dict[str, str]:
    """
    Load the feature-view assignment for every modelling stage.

    Expected location:
        <project>/modelling/config/model_feature_views.yml
    """
    config = _load_yaml(
        "model_feature_views.yml",
        config_dir=MODELLING_CONFIG_DIR,
    )

    assignments = config.get("feature_views")

    if not isinstance(assignments, dict):
        raise ValueError(
            "model_feature_views.yml must contain a mapping "
            "named 'feature_views'."
        )

    required_stages = {
        "level1_sepsis",
        "level2_gate",
        "level2_etiology",
        "level3_resistance",
    }

    missing = required_stages.difference(assignments)

    if missing:
        raise ValueError(
            "Missing modelling feature-view assignments for: "
            + ", ".join(sorted(missing))
        )

    invalid = {
        stage: view
        for stage, view in assignments.items()
        if not isinstance(view, str) or not view.strip()
    }

    if invalid:
        raise ValueError(
            "Feature-view assignments must be non-empty strings: "
            f"{invalid}"
        )

    return {
        stage: assignments[stage].strip()
        for stage in required_stages
    }