"""Random Forest model spec."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from ..types import ModelSpec


def build_random_forest(random_state: int = 42, n_jobs: int = 1, **kwargs):
    return RandomForestClassifier(
        n_estimators=kwargs.get("n_estimators", 500),
        max_depth=kwargs.get("max_depth"),
        min_samples_leaf=kwargs.get("min_samples_leaf", 1),
        class_weight=kwargs.get("class_weight", "balanced_subsample"),
        random_state=random_state,
        n_jobs=n_jobs,
    )


def suggest_random_forest_params(trial, task_type: str):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 700, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
    }


def get_random_forest_spec() -> ModelSpec:
    return ModelSpec(
        name="random_forest",
        supports_binary=True,
        supports_multiclass=True,
        supports_multilabel=True,
        build_estimator=build_random_forest,
        suggest_params=suggest_random_forest_params,
    )
