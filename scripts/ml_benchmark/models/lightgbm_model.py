"""LightGBM model spec."""

from __future__ import annotations

from lightgbm import LGBMClassifier

from ..types import ModelSpec


def build_lightgbm(task_type: str, random_state: int = 42, n_jobs: int = 1, **kwargs):
    objective = "binary" if task_type == "binary" else "multiclass"
    return LGBMClassifier(
        objective=objective,
        n_estimators=kwargs.get("n_estimators", 500),
        learning_rate=kwargs.get("learning_rate", 0.05),
        num_leaves=kwargs.get("num_leaves", 31),
        class_weight=kwargs.get("class_weight", "balanced"),
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=-1,
    )


def get_lightgbm_spec() -> ModelSpec:
    return ModelSpec(
        name="lightgbm",
        supports_binary=True,
        supports_multiclass=True,
        supports_multilabel=False,
        build_estimator=build_lightgbm,
    )

