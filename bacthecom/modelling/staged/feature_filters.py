"""Feature-filter utilities for staged modelling."""

from __future__ import annotations

from typing import Any

import pandas as pd

from shared.feature_filters import fit_iqr_bounds


def apply_iqr_bounds_to_nan(
    X: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    X = X.copy()
    audit: list[dict[str, Any]] = []

    for col, (lower, upper) in bounds.items():
        if col not in X.columns:
            continue

        values = pd.to_numeric(X[col], errors="coerce")
        mask = (values < lower) | (values > upper)
        n_outliers = int(mask.sum())

        if n_outliers:
            X.loc[mask, col] = pd.NA

        audit.append(
            {
                "feature": col,
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "n_outliers_to_nan": n_outliers,
            }
        )

    return X, audit


def remove_correlated_features_train_fit(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    if threshold >= 1.0:
        return X_train, X_test, []

    numeric_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    if len(numeric_cols) < 2:
        return X_train, X_test, []

    X_numeric = X_train[numeric_cols]
    corr = X_numeric.corr(method="spearman").abs().fillna(0.0)
    variances = X_numeric.var().fillna(0.0)

    col_order = sorted(numeric_cols, key=lambda c: (-float(variances[c]), c))
    dropped: set[str] = set()
    audit: list[dict[str, Any]] = []

    for col in col_order:
        if col in dropped:
            continue
        redundant = corr.index[corr[col] > threshold].tolist()
        for partner in redundant:
            if partner == col or partner in dropped:
                continue
            dropped.add(partner)
            audit.append(
                {
                    "kept_feature": col,
                    "dropped_feature": partner,
                    "abs_spearman": float(corr.loc[col, partner]),
                    "kept_variance": float(variances[col]),
                    "dropped_variance": float(variances[partner]),
                }
            )

    dropped_cols = sorted(dropped)
    return (
        X_train.drop(columns=dropped_cols, errors="ignore"),
        X_test.drop(columns=dropped_cols, errors="ignore"),
        audit,
    )


__all__ = ["fit_iqr_bounds", "apply_iqr_bounds_to_nan", "remove_correlated_features_train_fit"]
