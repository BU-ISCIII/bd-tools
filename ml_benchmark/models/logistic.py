"""Regularized logistic regression model spec."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from ..types import ModelSpec


def build_logistic(task_type: str, random_state: int = 42, n_jobs: int = 1, **kwargs):
    return LogisticRegression(
        C=kwargs.get("C", 1.0),
        solver=kwargs.get("solver", "saga"),
        l1_ratio=kwargs.get("l1_ratio", 0),
        class_weight=kwargs.get("class_weight", "balanced"),
        max_iter=kwargs.get("max_iter", 5000),
        random_state=random_state,
    )


def suggest_logistic_params(trial, task_type: str):
    return {
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        "l1_ratio": 0,
        "solver": "saga",
        "max_iter": 3000,
    }


def get_logistic_spec() -> ModelSpec:
    return ModelSpec(
        name="logistic",
        supports_binary=True,
        supports_multiclass=True,
        supports_multilabel=True,
        build_estimator=build_logistic,
        suggest_params=suggest_logistic_params,
    )
