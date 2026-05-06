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
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

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


class BenchmarkPreprocessor(BaseEstimator, TransformerMixin):
    """Model-aware tabular preprocessing fitted inside each benchmark split.

    Supported policies:
    - ``categorical_handling='one_hot'``: impute categoricals then one-hot encode.
    - ``categorical_handling='native'``: impute categoricals and return a
      pandas DataFrame so CatBoost can receive native categorical columns.
    - ``scale='none'|'standard'|'minmax'|'robust'``: scale numeric columns only
      when requested.
    """

    def __init__(
        self,
        numeric_columns: Sequence[str],
        categorical_columns: Sequence[str],
        categorical_handling: str,
        scale: str,
        impute_numeric: str = "median",
        impute_categorical: str = "most_frequent",
        qcut_numeric: str = "none",
        qcut_bins: int = 4,
        iqr_outlier_handling: str = "none",
        iqr_multiplier: float = 3.0,
    ):
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.categorical_handling = categorical_handling
        self.scale = scale
        self.impute_numeric = impute_numeric
        self.impute_categorical = impute_categorical
        self.qcut_numeric = qcut_numeric
        self.qcut_bins = qcut_bins
        self.iqr_outlier_handling = iqr_outlier_handling
        self.iqr_multiplier = iqr_multiplier

    def fit(self, X, y=None):
        # Called by sklearn Pipeline.fit on training rows only. Everything
        # learned here is later reused by transform on validation/test rows.
        numeric_columns = list(self.numeric_columns)
        categorical_columns = list(self.categorical_columns)
        X = _as_frame(X, numeric_columns + categorical_columns)
        self.numeric_pipeline_ = NumericFeatureBuilder(
            impute_numeric=self.impute_numeric,
            scale=self.scale,
            qcut_numeric=self.qcut_numeric,
            qcut_bins=self.qcut_bins,
            iqr_outlier_handling=self.iqr_outlier_handling,
            iqr_multiplier=self.iqr_multiplier,
        )
        self.numeric_pipeline_.fit(X[numeric_columns], y)

        self.categorical_imputer_ = None
        self.one_hot_encoder_ = None
        if categorical_columns:
            self.categorical_imputer_ = _simple_imputer(
                strategy=self.impute_categorical
            )
            cat_frame = X[categorical_columns].where(
                X[categorical_columns].notna(), np.nan
            )
            cat_values = self.categorical_imputer_.fit_transform(cat_frame)
            if self.categorical_handling == "one_hot":
                self.one_hot_encoder_ = _one_hot_encoder()
                self.one_hot_encoder_.fit(cat_values)
            elif self.categorical_handling == "native":
                pass
            elif self.categorical_handling == "preprocessed":
                raise ValueError(
                    "Categorical columns were found, but the feature policy expects "
                    "preprocessed numeric features. Use categorical_handling='one_hot' "
                    "or 'native', or fix the preprocessing output."
                )
            else:
                raise ValueError(
                    f"Unsupported categorical_handling='{self.categorical_handling}'."
                )

        numeric_names = self.numeric_pipeline_.get_feature_names_out(
            numeric_columns
        ).tolist()
        if self.categorical_handling == "one_hot" and self.one_hot_encoder_ is not None:
            categorical_names = self.one_hot_encoder_.get_feature_names_out(
                categorical_columns
            ).tolist()
        elif self.categorical_handling == "native":
            categorical_names = categorical_columns
        else:
            categorical_names = []
        self.feature_names_out_ = numeric_names + categorical_names
        return self

    def transform(self, X):
        # Called by Pipeline.predict/predict_proba for validation/test rows.
        # Use the already-fitted numeric pipeline, imputer, and encoder only.
        numeric_columns = list(self.numeric_columns)
        categorical_columns = list(self.categorical_columns)
        X = _as_frame(X, numeric_columns + categorical_columns)
        numeric = self.numeric_pipeline_.transform(X[numeric_columns])
        numeric_df = _as_frame(
            numeric,
            self.numeric_pipeline_.get_feature_names_out(self.numeric_columns),
        )
        numeric_df.index = X.index
        if not categorical_columns:
            return numeric_df

        cat_frame = X[categorical_columns].where(X[categorical_columns].notna(), np.nan)
        cat_values = self.categorical_imputer_.transform(cat_frame)
        if self.categorical_handling == "native":
            categorical_df = pd.DataFrame(
                cat_values,
                columns=categorical_columns,
                index=X.index,
            ).astype(str)
        elif self.categorical_handling == "one_hot":
            encoded = self.one_hot_encoder_.transform(cat_values)
            categorical_df = pd.DataFrame(
                encoded,
                columns=self.one_hot_encoder_.get_feature_names_out(
                    categorical_columns
                ),
                index=X.index,
            )
        else:
            categorical_df = pd.DataFrame(index=X.index)
        return pd.concat([numeric_df, categorical_df], axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_)


class NumericFeatureBuilder(BaseEstimator, TransformerMixin):
    """Train-fitted numeric preprocessing for benchmark pipelines."""

    def __init__(
        self,
        impute_numeric: str = "median",
        scale: str = "none",
        qcut_numeric: str = "none",
        qcut_bins: int = 4,
        iqr_outlier_handling: str = "none",
        iqr_multiplier: float = 3.0,
    ):
        self.impute_numeric = impute_numeric
        self.scale = scale
        self.qcut_numeric = qcut_numeric
        self.qcut_bins = qcut_bins
        self.iqr_outlier_handling = iqr_outlier_handling
        self.iqr_multiplier = iqr_multiplier

    def fit(self, X, y=None):
        X = _as_frame(X, _feature_names(X)).apply(pd.to_numeric, errors="coerce")
        self.feature_names_in_ = list(X.columns)
        if not self.feature_names_in_:
            self.iqr_bounds_ = {}
            self.qcut_bins_ = {}
            self.imputer_ = None
            self.scaler_ = None
            self.feature_names_out_ = []
            return self

        self.iqr_bounds_ = {}
        if self.iqr_outlier_handling == "train_fit":
            for column in self.feature_names_in_:
                series = X[column].dropna()
                if series.empty:
                    continue
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                self.iqr_bounds_[column] = (
                    q1 - self.iqr_multiplier * iqr,
                    q3 + self.iqr_multiplier * iqr,
                )
        elif self.iqr_outlier_handling != "none":
            raise ValueError(
                "Unsupported iqr_outlier_handling="
                f"'{self.iqr_outlier_handling}'. Use 'none' or 'train_fit'."
            )

        cleaned = self._apply_iqr(X)
        self.qcut_bins_ = {}
        if self.qcut_numeric in {"optional", "append", "quartile"}:
            for column in self.feature_names_in_:
                series = cleaned[column].dropna()
                if series.empty or series.nunique(dropna=True) < 2:
                    self.qcut_bins_[column] = []
                    continue
                try:
                    _, bins = pd.qcut(
                        series,
                        q=self.qcut_bins,
                        labels=False,
                        duplicates="drop",
                        retbins=True,
                    )
                except ValueError:
                    bins = []
                self.qcut_bins_[column] = [float(value) for value in bins]
        elif self.qcut_numeric != "none":
            raise ValueError(
                f"Unsupported qcut_numeric='{self.qcut_numeric}'. "
                "Use 'none', 'optional', 'append', or 'quartile'."
            )

        expanded = self._append_qcut(cleaned)
        self.imputer_ = _simple_imputer(strategy=self.impute_numeric)
        imputed = self.imputer_.fit_transform(expanded)
        self.scaler_ = _numeric_scaler(self.scale)
        if self.scaler_ is not None:
            self.scaler_.fit(imputed)
        self.feature_names_out_ = list(expanded.columns)
        return self

    def transform(self, X):
        # Apply train-fitted IQR bounds, qcut bins, imputer values, and scaler
        # parameters. This method must not learn from validation/test rows.
        X = _as_frame(X, self.feature_names_in_).apply(pd.to_numeric, errors="coerce")
        if not self.feature_names_out_:
            return pd.DataFrame(index=X.index)

        expanded = self._append_qcut(self._apply_iqr(X))
        imputed = self.imputer_.transform(expanded)
        if self.scaler_ is not None:
            imputed = self.scaler_.transform(imputed)
        return pd.DataFrame(imputed, columns=self.feature_names_out_, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_)

    def _apply_iqr(self, X: pd.DataFrame) -> pd.DataFrame:
        cleaned = X.copy()
        for column, (lower, upper) in self.iqr_bounds_.items():
            mask = (cleaned[column] < lower) | (cleaned[column] > upper)
            cleaned.loc[mask, column] = np.nan
        return cleaned

    def _append_qcut(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.qcut_bins_:
            return X.copy()
        qcut_columns = {}
        for column, bins in self.qcut_bins_.items():
            qcut_column = f"{column}_qcut"
            if len(bins) < 2:
                qcut_columns[qcut_column] = pd.Series(np.nan, index=X.index)
                continue
            encoded = pd.cut(
                X[column],
                bins=bins,
                labels=False,
                include_lowest=True,
            )
            qcut_columns[qcut_column] = encoded.astype(float)
        if not qcut_columns:
            return X.copy()
        return pd.concat([X.copy(), pd.DataFrame(qcut_columns, index=X.index)], axis=1)


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
        impute_numeric=job.feature_policy.impute_numeric,
        impute_categorical=job.feature_policy.impute_categorical,
        qcut_numeric=job.feature_policy.qcut_numeric,
        qcut_bins=job.feature_policy.qcut_bins,
        iqr_outlier_handling=job.feature_policy.iqr_outlier_handling,
        iqr_multiplier=job.feature_policy.iqr_multiplier,
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
            "receives their column names during fit."
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
    impute_numeric: str = "median",
    impute_categorical: str = "most_frequent",
    qcut_numeric: str = "none",
    qcut_bins: int = 4,
    iqr_outlier_handling: str = "none",
    iqr_multiplier: float = 3.0,
) -> BenchmarkPreprocessor:
    return BenchmarkPreprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        categorical_handling=categorical_handling,
        scale=scale,
        impute_numeric=impute_numeric,
        impute_categorical=impute_categorical,
        qcut_numeric=qcut_numeric,
        qcut_bins=qcut_bins,
        iqr_outlier_handling=iqr_outlier_handling,
        iqr_multiplier=iqr_multiplier,
    )


def infer_column_types(df: pd.DataFrame) -> tuple[List[str], List[str]]:
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns
    numeric = [col for col in df.columns if col not in set(categorical)]
    return numeric, list(categorical)


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _simple_imputer(strategy: str) -> SimpleImputer:
    try:
        return SimpleImputer(strategy=strategy, keep_empty_features=True)
    except TypeError:
        return SimpleImputer(strategy=strategy)


def _numeric_scaler(scale: str):
    if scale == "none":
        return None
    if scale == "standard":
        return StandardScaler()
    if scale == "minmax":
        return MinMaxScaler()
    if scale == "robust":
        return RobustScaler()
    raise ValueError(
        f"Unsupported scale='{scale}'. Use one of: none, standard, minmax, robust."
    )


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
