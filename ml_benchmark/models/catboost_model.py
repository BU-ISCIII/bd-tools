"""CatBoost model spec."""

from __future__ import annotations

from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

from ..types import ModelSpec


class NativeCatBoostClassifier(ClassifierMixin, BaseEstimator):
    """CatBoost sklearn wrapper that detects categorical pandas columns.

    The benchmark preprocessor keeps native categorical features in a DataFrame
    for CatBoost. This wrapper passes those column names through ``cat_features``
    during fit so CatBoost does not try to parse them as numeric values.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        task_type: str,
        iterations: int = 500,
        depth: int = 6,
        learning_rate: float = 0.05,
        auto_class_weights: str = "Balanced",
        random_state: int = 42,
        thread_count: int = 1,
    ):
        self.task_type = task_type
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.auto_class_weights = auto_class_weights
        self.random_state = random_state
        self.thread_count = thread_count

    def _build_model(self) -> CatBoostClassifier:
        params = {
            "iterations": self.iterations,
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": self.thread_count,
            "random_state": self.random_state,
            "auto_class_weights": self.auto_class_weights,
        }
        if self.task_type == "multiclass":
            params["loss_function"] = "MultiClass"
        return CatBoostClassifier(**params)

    def fit(self, X, y, sample_weight=None, **fit_params):
        self.model_ = self._build_model()
        cat_features = _categorical_columns(X)
        if cat_features:
            fit_params.setdefault("cat_features", cat_features)
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight
        self.model_.fit(X, y, **fit_params)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def get_params(self, deep=True):
        return {
            "task_type": self.task_type,
            "iterations": self.iterations,
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "auto_class_weights": self.auto_class_weights,
            "random_state": self.random_state,
            "thread_count": self.thread_count,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


def build_catboost(task_type: str, random_state: int = 42, n_jobs: int = 1, **kwargs):
    return NativeCatBoostClassifier(
        task_type=task_type,
        iterations=kwargs.get("iterations", 500),
        depth=kwargs.get("depth", 6),
        learning_rate=kwargs.get("learning_rate", 0.05),
        auto_class_weights=kwargs.get("auto_class_weights", "Balanced"),
        random_state=random_state,
        thread_count=n_jobs,
    )


def suggest_catboost_params(trial, task_type: str):
    return {
        "iterations": trial.suggest_int("iterations", 100, 700, step=100),
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
    }


def _categorical_columns(X) -> list[str] | list[int]:
    if hasattr(X, "select_dtypes"):
        return X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return []


def get_catboost_spec() -> ModelSpec:
    return ModelSpec(
        name="catboost",
        supports_binary=True,
        supports_multiclass=True,
        supports_multilabel=False,
        build_estimator=build_catboost,
        suggest_params=suggest_catboost_params,
    )
