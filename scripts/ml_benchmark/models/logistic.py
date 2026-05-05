"""Regularized logistic regression model spec."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from ..types import ModelSpec


def build_logistic(task_type: str, random_state: int = 42, n_jobs: int = 1, **kwargs):
    return LogisticRegression(
        penalty=kwargs.get("penalty", "l2"),
        C=kwargs.get("C", 1.0),
        solver=kwargs.get("solver", "saga"),
        l1_ratio=kwargs.get("l1_ratio"),
        class_weight=kwargs.get("class_weight", "balanced"),
        max_iter=kwargs.get("max_iter", 5000),
        random_state=random_state,
        n_jobs=n_jobs,
    )


def get_logistic_spec() -> ModelSpec:
    return ModelSpec(
        name="logistic",
        supports_binary=True,
        supports_multiclass=True,
        supports_multilabel=True,
        build_estimator=build_logistic,
    )
