"""Feature-filter helpers shared across modelling pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def fit_iqr_bounds(X: pd.DataFrame, iqr_multiplier: float = 5.0) -> Dict[str, Tuple[float, float]]:
    bounds: Dict[str, Tuple[float, float]] = {}
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        s = X[col]
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower = q1 - (iqr_multiplier * iqr)
        upper = q3 + (iqr_multiplier * iqr)
        bounds[col] = (float(lower), float(upper))
    return bounds


def apply_iqr_bounds_to_nan(
    X: pd.DataFrame,
    bounds: Dict[str, Tuple[float, float]],
    output_csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    X = X.copy()
    audit: List[Dict[str, Any]] = []
    for col, (lower, upper) in bounds.items():
        if col not in X.columns:
            continue
        mask = (X[col] < lower) | (X[col] > upper)
        n_outliers = int(mask.sum())
        if n_outliers > 0:
            X.loc[mask, col] = np.nan
            audit.append(
                {
                    "feature": col,
                    "n_outliers": n_outliers,
                    "lower_bound": float(lower),
                    "upper_bound": float(upper),
                }
            )
    if audit and output_csv_path:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(audit).to_csv(output_csv_path, index=False)
    return X


def remove_correlated_features(
    X: pd.DataFrame,
    threshold: float = 0.90,
    output_csv_path: Optional[Path] = None,
) -> List[str]:
    if threshold >= 1.0:
        return X.columns.tolist()

    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    if len(numeric_cols) < len(X.columns):
        dropped_non_numeric = set(X.columns) - set(numeric_cols)
        print(
            f"  Removing {len(dropped_non_numeric)} non-numeric columns before correlation: "
            f"{list(dropped_non_numeric)[:5]}..."
        )

    if len(numeric_cols) < 2:
        return X.columns.tolist()

    X_numeric = X[numeric_cols]
    corr = X_numeric.corr(method="spearman").abs().fillna(0.0)
    variances = X_numeric.var().fillna(0.0)

    col_order = sorted(numeric_cols, key=lambda c: (-float(variances[c]), c))

    dropped: set[str] = set()
    kept_ordered: List[str] = []
    drop_audit: List[Dict[str, Any]] = []
    for col in col_order:
        if col in dropped:
            continue
        kept_ordered.append(col)
        redundant = corr.index[corr[col] > threshold].tolist()
        for partner in redundant:
            if partner == col or partner in dropped:
                continue
            dropped.add(partner)
            drop_audit.append(
                {
                    "kept_col": col,
                    "dropped_col": partner,
                    "spearman_abs_corr": float(corr.loc[col, partner]),
                    "kept_variance": float(variances[col]),
                    "dropped_variance": float(variances[partner]),
                }
            )

    if drop_audit and output_csv_path:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_audit = pd.DataFrame(
            [
                {
                    "feature_dropped": item["dropped_col"],
                    "correlated_feature": item["kept_col"],
                    "abs_spearman": item["spearman_abs_corr"],
                    "var_kept": item["kept_variance"],
                    "var_dropped": item["dropped_variance"],
                }
                for item in drop_audit
            ]
        )
        df_audit.to_csv(output_csv_path, index=False)

    kept_set = set(kept_ordered)
    non_numeric_cols = set(X.columns) - set(numeric_cols)
    kept_set.update(non_numeric_cols)
    return [c for c in X.columns if c in kept_set]


def remove_correlated_features_train_fit(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    kept = remove_correlated_features(X_train, threshold=threshold)
    dropped_cols = [c for c in X_train.columns if c not in kept]
    audit = [{"dropped_feature": c} for c in dropped_cols]
    return (
        X_train[kept].copy(),
        X_test[kept].copy(),
        audit,
    )
