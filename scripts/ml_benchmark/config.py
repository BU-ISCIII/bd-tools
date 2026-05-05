"""Config loading and normalization for predictive ML benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

from .types import (
    BenchmarkConfig,
    CorrelationFilterSpec,
    FeatureGroupSpec,
    FeaturePolicySpec,
    FeatureSetSpec,
    FeatureViewSpec,
    TargetSpec,
)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_targets(config: Dict[str, Any]) -> List[TargetSpec]:
    targets = []
    for item in config.get("targets", []):
        targets.append(
            TargetSpec(
                name=item["name"],
                task_type=item["type"],
                columns=list(item.get("columns", [])),
                positive_label=item.get("positive_label"),
                main_metric=item.get("main_metric"),
            )
        )
    if not targets:
        raise ValueError("Benchmark config must define at least one target.")
    return targets


def _load_feature_sets(config: Dict[str, Any]) -> List[FeatureSetSpec]:
    feature_sets = []
    for item in config.get("feature_sets", [{"name": "all_features", "strategy": "none"}]):
        feature_sets.append(
            FeatureSetSpec(
                name=item["name"],
                strategy=item.get("strategy", "none"),
                max_features=item.get("max_features"),
                selector=item.get("selector"),
            )
        )
    return feature_sets


def _load_feature_groups(config: Dict[str, Any]) -> Dict[str, FeatureGroupSpec]:
    groups = {}
    for name, item in (config.get("feature_groups") or {}).items():
        groups[name] = FeatureGroupSpec(
            name=name,
            description=item.get("description"),
            alternatives=dict(item.get("alternatives", {})),
        )
    return groups


def _load_feature_views(config: Dict[str, Any]) -> List[FeatureViewSpec]:
    raw_views = config.get("feature_views") or [
        {"name": "default", "description": "Use all configured model features."}
    ]
    views = []
    for item in raw_views:
        views.append(
            FeatureViewSpec(
                name=item["name"],
                description=item.get("description"),
                include_patterns=list(item.get("include_patterns", ["*"])),
                exclude_columns=list(item.get("exclude_columns", [])),
                exclude_patterns=list(item.get("exclude_patterns", [])),
                groups=dict(item.get("groups", {})),
            )
        )
    return views


def _load_models(config: Dict[str, Any]) -> List[str]:
    models = [
        item["name"] if isinstance(item, dict) else item
        for item in config.get("models", [])
    ]
    if not models:
        raise ValueError("Benchmark config must define at least one model.")
    return models


def _load_model_feature_policies(config: Dict[str, Any]) -> Dict[str, str]:
    policies = {}
    for item in config.get("models", []):
        if isinstance(item, dict):
            policies[item["name"]] = item.get("feature_policy", "default")
        else:
            policies[item] = "default"
    return policies


def _load_feature_policies(config: Dict[str, Any]) -> Dict[str, FeaturePolicySpec]:
    raw_policies = config.get("feature_policies") or {
        "default": {
            "require_numeric": True,
            "categorical_handling": "preprocessed",
            "scale": "none",
            "correlation_filter": {"enabled": False, "mode": "report_only"},
        }
    }
    policies = {}
    for name, item in raw_policies.items():
        corr = item.get("correlation_filter", {})
        policies[name] = FeaturePolicySpec(
            name=name,
            require_numeric=item.get("require_numeric", True),
            categorical_handling=item.get(
                "categorical_handling", "preprocessed"
            ),
            scale=item.get("scale", "none"),
            correlation_filter=CorrelationFilterSpec(
                enabled=corr.get("enabled", False),
                threshold=float(corr.get("threshold", 0.90)),
                mode=corr.get("mode", "report_only"),
                manual_groups_first=corr.get("manual_groups_first", True),
            ),
            drop_columns=list(item.get("drop_columns", [])),
            drop_patterns=list(item.get("drop_patterns", [])),
        )
    if "default" not in policies:
        first_policy = next(iter(policies))
        policies["default"] = policies[first_policy]
    return policies


def _validate_model_feature_policies(
    models: List[str],
    model_feature_policies: Dict[str, str],
    feature_policies: Dict[str, FeaturePolicySpec],
) -> None:
    for model_name in models:
        policy_name = model_feature_policies.get(model_name, "default")
        if policy_name not in feature_policies:
            raise ValueError(
                f"Model '{model_name}' references unknown feature policy "
                f"'{policy_name}'."
            )


def load_config(path: str | Path) -> BenchmarkConfig:
    config_path = Path(path)
    raw = load_yaml(config_path)
    models = _load_models(raw)
    model_feature_policies = _load_model_feature_policies(raw)
    feature_policies = _load_feature_policies(raw)
    _validate_model_feature_policies(models, model_feature_policies, feature_policies)
    return BenchmarkConfig(
        raw=raw,
        path=config_path,
        targets=_load_targets(raw),
        models=models,
        model_feature_policies=model_feature_policies,
        feature_policies=feature_policies,
        feature_groups=_load_feature_groups(raw),
        feature_views=_load_feature_views(raw),
        feature_sets=_load_feature_sets(raw),
    )
