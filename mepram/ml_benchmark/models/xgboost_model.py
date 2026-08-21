"""XGBoost model spec."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from ..types import ModelSpec


class EncodedXGBClassifier(ClassifierMixin, BaseEstimator):
    """XGBoost classifier that label-encodes targets before fitting.

    XGBoost's sklearn wrapper expects class labels to be contiguous integers
    starting at zero. The benchmark targets are often stored as strings, so this
    wrapper maps them to integer codes during fit and maps predictions back to
    the original labels.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        task_type: str,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        tree_method: str = "hist",
        random_state: int = 42,
        n_jobs: int = 1,
        verbosity: int = 0,
    ):
        self.task_type = task_type
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.tree_method = tree_method
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbosity = verbosity

    def _build_model(self) -> XGBClassifier:
        objective = "binary:logistic" if self.task_type == "binary" else "multi:softprob"
        params = {
            "objective": objective,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "tree_method": self.tree_method,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbosity": self.verbosity,
        }
        if self.task_type != "binary":
            params["objective"] = "multi:softprob"
        return XGBClassifier(**params)

    def fit(self, X, y, sample_weight=None, **fit_params):
        self.label_encoder_ = LabelEncoder()
        y_array = np.asarray(y)
        y_encoded = self.label_encoder_.fit_transform(y_array)
        self.model_ = self._build_model()
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight
        self.model_.fit(X, y_encoded, **fit_params)
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict(self, X):
        encoded = np.asarray(self.model_.predict(X)).ravel().astype(int)
        return self.label_encoder_.inverse_transform(encoded)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def build_xgboost(task_type: str, random_state: int = 42, n_jobs: int = 1, **kwargs):
    return EncodedXGBClassifier(
        task_type=task_type,
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
