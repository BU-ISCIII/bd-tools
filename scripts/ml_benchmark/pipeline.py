"""Leakage-safe benchmark pipeline construction.

Every data-dependent preprocessing step returned here is intended to be fitted
inside the training fold/split only. Validation and test data should only call
``transform`` through the fitted pipeline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from .types import BenchmarkJob, ModelSpec, PipelineBuildResult, ShapRFECVSpec


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
        self.correlation_pairs_ = []
        self.correlation_matrix_ = pd.DataFrame()
        if not self.enabled or self.mode == "disabled" or self.threshold >= 1.0:
            return self

        frame = _as_frame(X, self.feature_names_in_)
        corr = frame.corr(method="spearman").abs().fillna(0.0)
        self.correlation_matrix_ = corr
        self.correlation_pairs_ = _correlation_pairs(corr, self.threshold)
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


def _correlation_pairs(corr: pd.DataFrame, threshold: float) -> list[dict[str, object]]:
    pairs: list[dict[str, object]] = []
    columns = list(corr.columns)
    for left_idx, left in enumerate(columns):
        for right in columns[left_idx + 1 :]:
            value = float(corr.loc[left, right])
            if value > threshold:
                pairs.append(
                    {
                        "feature_1": left,
                        "feature_2": right,
                        "abs_spearman": value,
                    }
                )
    return sorted(pairs, key=lambda item: item["abs_spearman"], reverse=True)


class ShapRFECVSelector(BaseEstimator, TransformerMixin):
    """Train-fitted SHAP recursive feature elimination with internal CV.

    The selector is itself a pipeline step, so sklearn fits it only on the
    current training split. Validation/test rows only pass through transform.
    """

    def __init__(
        self,
        strategy: str = "none",
        max_features: int | None = None,
        estimator=None,
        task_type: str = "binary",
        cv_splits: int = 2,
        step_fraction: float = 0.50,
        min_features_to_select: int = 20,
        max_shap_rows: int = 500,
        max_selector_rows: int | None = None,
        selector_estimator_params: dict | None = None,
        random_state: int = 42,
        cache_dir: str | None = None,
        cache_key: str | None = None,
    ):
        self.strategy = strategy
        self.max_features = max_features
        self.estimator = estimator
        self.task_type = task_type
        self.cv_splits = cv_splits
        self.step_fraction = step_fraction
        self.min_features_to_select = min_features_to_select
        self.max_shap_rows = max_shap_rows
        self.max_selector_rows = max_selector_rows
        self.selector_estimator_params = selector_estimator_params
        self.random_state = random_state
        self.cache_dir = cache_dir
        self.cache_key = cache_key

    def fit(self, X, y=None):
        self.feature_names_in_ = _feature_names(X)
        self.selected_features_ = list(self.feature_names_in_)
        self.dropped_features_ = []
        self.history_ = []
        self.status_ = "not_configured"
        self.cache_status_ = "not_applicable"
        self.cache_path_ = None
        self.ranking_features_ = list(self.feature_names_in_)
        self.best_score_ = None
        self.selector_rows_ = None
        self.selector_estimator_params_ = dict(self.selector_estimator_params or {})
        if self.strategy in {"none", "all"}:
            return self
        if self.strategy != "shap_rfecv":
            self.status_ = "unsupported_strategy"
            return self
        if self.estimator is None or y is None:
            self.status_ = "missing_estimator_or_target"
            return self

        frame = _as_frame(X, self.feature_names_in_)
        if not _all_numeric(frame):
            self.status_ = "skipped_non_numeric_features"
            return self

        selector_frame, selector_y = self._selector_training_data(frame, y)
        self.selector_rows_ = len(selector_frame)
        cached = self._load_cache(frame, y)
        if cached is None:
            cached = self._fit_rfecv(selector_frame, selector_y)
            self._write_cache(cached)
        else:
            self.cache_status_ = "hit"

        self.history_ = cached.get("history", [])
        self.ranking_features_ = [
            feature
            for feature in cached.get("ranking_features", [])
            if feature in self.feature_names_in_
        ]
        self.best_score_ = cached.get("best_score")
        if not self.ranking_features_:
            self.status_ = "empty_cached_ranking"
            return self

        self.selected_features_ = self._apply_max_feature_cap(self.ranking_features_)
        selected_set = set(self.selected_features_)
        self.dropped_features_ = [
            feature for feature in self.feature_names_in_ if feature not in selected_set
        ]
        self.status_ = "completed"
        return self

    def _fit_rfecv(self, frame: pd.DataFrame, y) -> dict[str, object]:
        target_count = self._rfecv_target_feature_count(len(self.feature_names_in_))
        current_features = list(self.feature_names_in_)
        best_features = current_features
        best_score = -np.inf
        round_idx = 0

        while True:
            score = self._cv_score(frame[current_features], y)
            importances = self._shap_importance(frame[current_features], y)
            if score is not None and score > best_score:
                best_score = score
                best_features = list(current_features)
            self.history_.append(
                {
                    "round": round_idx,
                    "n_features": len(current_features),
                    "cv_score": score,
                    "best_score_so_far": None if best_score == -np.inf else best_score,
                    "removed_features": [],
                }
            )
            if len(current_features) <= target_count:
                break
            if importances.empty:
                self.status_ = "shap_importance_failed"
                break

            remove_count = min(
                len(current_features) - target_count,
                max(1, int(len(current_features) * self.step_fraction)),
            )
            removed = importances.tail(remove_count)["feature"].tolist()
            current_features = [f for f in current_features if f not in set(removed)]
            self.history_[-1]["removed_features"] = removed
            round_idx += 1

        final_importances = self._shap_importance(frame[best_features], y)
        if final_importances.empty:
            ranking_features = list(best_features)
        else:
            ranking_features = final_importances["feature"].tolist()
        self.cache_status_ = "created"
        return {
            "ranking_features": ranking_features,
            "history": self.history_,
            "best_score": None if best_score == -np.inf else float(best_score),
            "best_feature_count": len(best_features),
            "metadata": {
                "input_feature_count": len(self.feature_names_in_),
                "selector_rows": len(frame),
                "cv_splits": self.cv_splits,
                "step_fraction": self.step_fraction,
                "min_features_to_select": self.min_features_to_select,
                "max_features_cap": self.max_features,
                "max_shap_rows": self.max_shap_rows,
                "max_selector_rows": self.max_selector_rows,
                "selector_estimator": self.estimator.__class__.__name__,
                "selector_estimator_params": self.selector_estimator_params_,
            },
        }

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        indices = [
            self.feature_names_in_.index(feature) for feature in self.selected_features_
        ]
        return np.asarray(X)[:, indices]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(getattr(self, "selected_features_", input_features))

    def _rfecv_target_feature_count(self, n_features: int) -> int:
        return max(1, min(self.min_features_to_select, n_features))

    def _apply_max_feature_cap(self, ranking_features: list[str]) -> list[str]:
        if self.max_features is None:
            return list(ranking_features)
        return list(ranking_features[: max(1, int(self.max_features))])

    def _load_cache(self, X: pd.DataFrame, y) -> dict[str, object] | None:
        cache_path = self._cache_path(X, y)
        if cache_path is None or not cache_path.exists():
            self.cache_status_ = "miss" if cache_path is not None else "disabled"
            return None
        self.cache_path_ = str(cache_path)
        return json.loads(cache_path.read_text(encoding="utf-8"))

    def _write_cache(self, payload: dict[str, object]) -> None:
        if self.cache_path_ is None:
            return
        cache_path = Path(self.cache_path_)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _cache_path(self, X: pd.DataFrame, y) -> Path | None:
        if not self.cache_dir or not self.cache_key:
            return None
        digest = _selection_fingerprint(
            X=X,
            y=y,
            task_type=self.task_type,
            cv_splits=self.cv_splits,
            step_fraction=self.step_fraction,
            min_features_to_select=self.min_features_to_select,
            max_shap_rows=self.max_shap_rows,
            max_selector_rows=self.max_selector_rows,
            selector_estimator_params=self.selector_estimator_params_,
            estimator=self.estimator,
        )
        path = Path(self.cache_dir) / f"{self.cache_key}__{digest}.json"
        self.cache_path_ = str(path)
        return path

    def _selector_estimator(self):
        estimator = clone(self.estimator)
        params = estimator.get_params(deep=False)
        capped = {}
        if "n_estimators" in params:
            capped["n_estimators"] = min(int(params.get("n_estimators") or 100), 100)
        if "iterations" in params:
            capped["iterations"] = min(int(params.get("iterations") or 100), 100)
        if "max_iter" in params:
            capped["max_iter"] = min(int(params.get("max_iter") or 1000), 1000)
        if capped:
            estimator.set_params(**capped)
        if self.selector_estimator_params_:
            valid_params = estimator.get_params(deep=False)
            overrides = {
                key: value
                for key, value in self.selector_estimator_params_.items()
                if key in valid_params
            }
            if overrides:
                estimator.set_params(**overrides)
        return estimator

    def _selector_training_data(
        self,
        X: pd.DataFrame,
        y,
    ) -> tuple[pd.DataFrame, pd.Series]:
        y_series = pd.Series(y, index=X.index)
        if self.max_selector_rows is None or len(X) <= self.max_selector_rows:
            return X, y_series

        sample_size = max(1, int(self.max_selector_rows))
        sampled_indices = []
        for _, class_indices in y_series.groupby(y_series, observed=True).groups.items():
            class_indices = list(class_indices)
            n_class = min(
                len(class_indices),
                max(1, round(sample_size * len(class_indices) / len(X))),
            )
            sampled_indices.extend(
                pd.Series(class_indices).sample(
                    n=n_class,
                    random_state=self.random_state,
                )
            )
        sampled_indices = pd.Index(sampled_indices)
        if len(sampled_indices) > sample_size:
            sampled_indices = pd.Index(
                pd.Series(sampled_indices).sample(
                    n=sample_size,
                    random_state=self.random_state,
                )
            )
        sampled_indices = sampled_indices.sort_values()
        return X.loc[sampled_indices], y_series.loc[sampled_indices]

    def _cv_score(self, X: pd.DataFrame, y) -> float | None:
        y_series = pd.Series(y)
        splitter = _selector_splitter(
            y_series, self.cv_splits, self.random_state
        )
        if splitter is None:
            return None
        scores = []
        for train_idx, validation_idx in splitter.split(X, y_series):
            estimator = self._selector_estimator()
            estimator.fit(X.iloc[train_idx], y_series.iloc[train_idx])
            y_true = y_series.iloc[validation_idx]
            if self.task_type == "binary" and hasattr(estimator, "predict_proba"):
                proba = estimator.predict_proba(X.iloc[validation_idx])
                scores.append(roc_auc_score(y_true, proba[:, 1]))
            else:
                y_pred = np.asarray(estimator.predict(X.iloc[validation_idx])).ravel()
                scores.append(f1_score(y_true, y_pred, average="macro"))
        return float(np.mean(scores)) if scores else None

    def _shap_importance(self, X: pd.DataFrame, y) -> pd.DataFrame:
        try:
            import shap

            estimator = self._selector_estimator()
            estimator.fit(X, y)
            sample = X.sample(
                n=min(len(X), self.max_shap_rows),
                random_state=self.random_state,
            )
            explainer = shap.Explainer(estimator, sample)
            values = explainer(sample).values
            if isinstance(values, list):
                values = np.asarray(values)
            values = np.asarray(values)
            if values.ndim == 3:
                importance = np.abs(values).mean(axis=(0, 2))
            else:
                importance = np.abs(values).mean(axis=0)
            return pd.DataFrame(
                {"feature": list(X.columns), "mean_abs_shap": importance}
            ).sort_values("mean_abs_shap", ascending=False)
        except Exception as exc:
            self.shap_error_ = str(exc)
            return pd.DataFrame(columns=["feature", "mean_abs_shap"])


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
                if iqr <= 0:
                    continue
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
                if series.empty or series.nunique(dropna=True) < self.qcut_bins:
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
    feature_selection: ShapRFECVSpec | None = None,
    feature_selection_estimator_params: dict | None = None,
    feature_selection_cache_dir: str | None = None,
    feature_selection_cache_key: str | None = None,
) -> PipelineBuildResult:
    """Build a leakage-safe sklearn pipeline for one benchmark job."""
    notes: List[str] = []
    numeric_columns = list(numeric_columns)
    categorical_columns = list(categorical_columns)
    shap_rfecv = feature_selection or ShapRFECVSpec()

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
    if job.feature_set.strategy == "shap_rfecv":
        notes.append("SHAP-RFECV feature selection is fitted inside each split.")
    elif job.feature_set.strategy != "none":
        notes.append(f"Feature-selection strategy '{job.feature_set.strategy}' is unsupported.")

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
                ShapRFECVSelector(
                    strategy=job.feature_set.strategy,
                    max_features=job.feature_set.max_features,
                    estimator=estimator,
                    task_type=job.target.task_type,
                    cv_splits=shap_rfecv.cv_splits,
                    step_fraction=shap_rfecv.step_fraction,
                    min_features_to_select=shap_rfecv.min_features_to_select,
                    max_shap_rows=shap_rfecv.max_shap_rows,
                    max_selector_rows=shap_rfecv.max_selector_rows,
                    selector_estimator_params=feature_selection_estimator_params,
                    random_state=random_state,
                    cache_dir=feature_selection_cache_dir,
                    cache_key=feature_selection_cache_key,
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


def _all_numeric(frame: pd.DataFrame) -> bool:
    return all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes)


def _selector_splitter(y: pd.Series, cv_splits: int, random_state: int):
    n_splits = min(int(cv_splits), int(y.value_counts().min()))
    if n_splits < 2:
        return None
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def _selection_fingerprint(
    *,
    X: pd.DataFrame,
    y,
    task_type: str,
    cv_splits: int,
    step_fraction: float,
    min_features_to_select: int,
    max_shap_rows: int,
    max_selector_rows: int | None,
    selector_estimator_params: dict | None,
    estimator,
) -> str:
    payload = {
        "columns": list(X.columns),
        "shape": list(X.shape),
        "task_type": task_type,
        "cv_splits": int(cv_splits),
        "step_fraction": float(step_fraction),
        "min_features_to_select": int(min_features_to_select),
        "max_shap_rows": int(max_shap_rows),
        "max_selector_rows": (
            None if max_selector_rows is None else int(max_selector_rows)
        ),
        "selector_estimator_params": selector_estimator_params or {},
        "estimator": estimator.__class__.__name__ if estimator is not None else None,
    }
    hasher = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))
    hasher.update(pd.util.hash_pandas_object(X, index=True).values.tobytes())
    hasher.update(pd.util.hash_pandas_object(pd.Series(y), index=True).values.tobytes())
    return hasher.hexdigest()[:24]


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
