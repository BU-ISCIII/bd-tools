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


def get_random_forest_spec() -> ModelSpec:
    return ModelSpec(
        name="random_forest",
        supports_binary=True,
        supports_multiclass=True,
        supports_multilabel=True,
        build_estimator=build_random_forest,
    )

