"""Leakage-safe benchmark pipeline construction.

Every data-dependent preprocessing step returned here is intended to be fitted
inside the training fold/split only. Validation and test data should only call
``transform`` through the fitted pipeline.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .types import BenchmarkJob, ModelSpec, PipelineBuildResult


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop highly correlated columns using training data only.

    Modes:
    - ``report_only``: learn/report correlations but keep all columns.
    - ``report_then_drop`` or ``auto_drop``: drop redundant columns.
    - ``disabled``: keep all columns without computing correlations.
    """

    def __init__(
        self,
        threshold: float = 0.90,
        mode: str = "report_only",
        enabled: bool = False,
    ):
        self.threshold = threshold
        self.mode = mode
        self.enabled = enabled

    def fit(self, X, y=None):
        self.feature_names_in_ = _feature_names(X)
        self.kept_indices_ = list(range(len(self.feature_names_in_)))
        self.dropped_features_ = []
        if not self.enabled or self.mode == "disabled" or self.threshold >= 1.0:
            return self

        frame = _as_frame(X, self.feature_names_in_)
        corr = frame.corr(method="spearman").abs().fillna(0.0)
        col_order = frame.var().sort_values(ascending=False).index.tolist()

        dropped = set()
        kept_ordered: List[str] = []
        for col in col_order:
            if col in dropped:
                continue
            kept_ordered.append(col)
            redundant = corr.index[corr[col] > self.threshold].tolist()
            for partner in redundant:
                if partner != col:
                    dropped.add(partner)

        self.dropped_features_ = [c for c in self.feature_names_in_ if c in dropped]
        if self.mode in {"report_then_drop", "auto_drop"}:
            kept_set = set(kept_ordered)
            self.kept_indices_ = [
                idx
                for idx, name in enumerate(self.feature_names_in_)
                if name in kept_set
            ]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.kept_indices_]
        return np.asarray(X)[:, self.kept_indices_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray([self.feature_names_in_[idx] for idx in self.kept_indices_])


class FeatureSelectionPlaceholder(BaseEstimator, TransformerMixin):
    """Placeholder for cached feature-selection strategies.

    This keeps the pipeline contract explicit while the actual cached SHAP-RFECV
    implementation is added in a later slice.
    """

    def __init__(self, strategy: str = "none", max_features: int | None = None):
        self.strategy = strategy
        self.max_features = max_features

    def fit(self, X, y=None):
        self.feature_names_in_ = _feature_names(X)
        if self.strategy not in {"none", "all"}:
            self.pending_implementation_ = True
        return self

    def transform(self, X):
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(getattr(self, "feature_names_in_", input_features))


def build_benchmark_pipeline(
    *,
    job: BenchmarkJob,
    model_spec: ModelSpec,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    random_state: int,
    n_jobs: int,
) -> PipelineBuildResult:
    """Build a leakage-safe sklearn pipeline for one benchmark job."""
    notes: List[str] = []
    numeric_columns = list(numeric_columns)
    categorical_columns = list(categorical_columns)

    preprocessor = build_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        categorical_handling=job.feature_policy.categorical_handling,
        scale=job.feature_policy.scale,
    )
    estimator = model_spec.build_estimator(
        task_type=job.target.task_type,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    corr = job.feature_policy.correlation_filter
    if corr.manual_groups_first:
        notes.append("Manual feature groups are resolved before correlation filtering.")
    if job.feature_policy.categorical_handling == "native" and categorical_columns:
        notes.append(
            "Native categorical columns are kept through preprocessing; CatBoost "
            "fit will need categorical feature metadata in the training runner."
        )
    if job.feature_set.strategy != "none":
        notes.append(
            f"Feature-selection strategy '{job.feature_set.strategy}' is declared "
            "but not implemented in this scaffold yet."
        )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "correlation_filter",
                CorrelationFilter(
                    threshold=corr.threshold,
                    mode=corr.mode,
                    enabled=corr.enabled,
                ),
            ),
            (
                "feature_selection",
                FeatureSelectionPlaceholder(
                    strategy=job.feature_set.strategy,
                    max_features=job.feature_set.max_features,
                ),
            ),
            ("model", estimator),
        ]
    )
    return PipelineBuildResult(
        pipeline=pipeline,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        notes=notes,
    )


def build_preprocessor(
    *,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    categorical_handling: str,
    scale: str,
) -> ColumnTransformer:
    transformers = []
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale == "standard":
        numeric_steps.append(("scaler", StandardScaler()))
    transformers.append(("numeric", Pipeline(numeric_steps), list(numeric_columns)))

    categorical_columns = list(categorical_columns)
    if categorical_columns:
        if categorical_handling == "native":
            transformers.append(
                (
                    "categorical",
                    SimpleImputer(strategy="most_frequent"),
                    categorical_columns,
                )
            )
        elif categorical_handling == "one_hot":
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", _one_hot_encoder()),
                        ]
                    ),
                    categorical_columns,
                )
            )
        elif categorical_handling == "preprocessed":
            raise ValueError(
                "Categorical columns were found, but the feature policy expects "
                "preprocessed numeric features. Use categorical_handling='one_hot' "
                "or 'native', or fix the preprocessing output."
            )
        else:
            raise ValueError(
                f"Unsupported categorical_handling='{categorical_handling}'."
            )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def infer_column_types(df: pd.DataFrame) -> tuple[List[str], List[str]]:
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns
    numeric = [col for col in df.columns if col not in set(categorical)]
    return numeric, list(categorical)


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _feature_names(X) -> List[str]:
    if isinstance(X, pd.DataFrame):
        return list(X.columns)
    if hasattr(X, "columns"):
        return list(X.columns)
    return [f"feature_{idx}" for idx in range(np.asarray(X).shape[1])]


def _as_frame(X, columns: Sequence[str]) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(np.asarray(X), columns=list(columns))
