"""XGBoost model spec."""

from __future__ import annotations

from xgboost import XGBClassifier

from ..types import ModelSpec


def build_xgboost(task_type: str, random_state: int = 42, n_jobs: int = 1, **kwargs):
    objective = "binary:logistic" if task_type == "binary" else "multi:softprob"
    return XGBClassifier(
        objective=objective,
        n_estimators=kwargs.get("n_estimators", 500),
        learning_rate=kwargs.get("learning_rate", 0.05),
        max_depth=kwargs.get("max_depth", 6),
        tree_method=kwargs.get("tree_method", "hist"),
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=0,
    )


def get_xgboost_spec() -> ModelSpec:
    return ModelSpec(
        name="xgboost",
        supports_binary=True,
        supports_multiclass=True,
        supports_multilabel=False,
        build_estimator=build_xgboost,
    )

