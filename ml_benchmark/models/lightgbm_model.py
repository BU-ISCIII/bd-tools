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
        max_depth=kwargs.get("max_depth", -1),
        min_child_samples=kwargs.get("min_child_samples", 20),
        subsample=kwargs.get("subsample", 1.0),
        colsample_bytree=kwargs.get("colsample_bytree", 1.0),
        reg_alpha=kwargs.get("reg_alpha", 0.0),
        reg_lambda=kwargs.get("reg_lambda", 0.0),
        class_weight=kwargs.get("class_weight", "balanced"),
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=-1,
    )


def suggest_lightgbm_params(trial, task_type: str):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 700, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.60, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def get_lightgbm_spec() -> ModelSpec:
    return ModelSpec(
        name="lightgbm",
        supports_binary=True,
        supports_multiclass=True,
        supports_multilabel=False,
        build_estimator=build_lightgbm,
        suggest_params=suggest_lightgbm_params,
    )
