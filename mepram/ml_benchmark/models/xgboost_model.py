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
        subsample=kwargs.get("subsample", 1.0),
        colsample_bytree=kwargs.get("colsample_bytree", 1.0),
        reg_alpha=kwargs.get("reg_alpha", 0.0),
        reg_lambda=kwargs.get("reg_lambda", 1.0),
        tree_method=kwargs.get("tree_method", "hist"),
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=0,
    )


def suggest_xgboost_params(trial, task_type: str):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 700, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.60, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def get_xgboost_spec() -> ModelSpec:
    return ModelSpec(
        name="xgboost",
        supports_binary=True,
        supports_multiclass=True,
        supports_multilabel=False,
        build_estimator=build_xgboost,
        suggest_params=suggest_xgboost_params,
    )
