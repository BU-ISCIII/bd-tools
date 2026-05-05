"""CatBoost model spec."""

from __future__ import annotations

from catboost import CatBoostClassifier

from ..types import ModelSpec


def build_catboost(task_type: str, random_state: int = 42, n_jobs: int = 1, **kwargs):
    params = {
        "iterations": kwargs.get("iterations", 500),
        "depth": kwargs.get("depth", 6),
        "learning_rate": kwargs.get("learning_rate", 0.05),
        "verbose": False,
        "thread_count": n_jobs,
        "random_state": random_state,
        "auto_class_weights": kwargs.get("auto_class_weights", "Balanced"),
    }
    if task_type == "multiclass":
        params["loss_function"] = "MultiClass"
    return CatBoostClassifier(**params)


def get_catboost_spec() -> ModelSpec:
    return ModelSpec(
        name="catboost",
        supports_binary=True,
        supports_multiclass=True,
        supports_multilabel=False,
        build_estimator=build_catboost,
    )

