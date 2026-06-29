"""General and preprocessing utilities for staged mortality pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .feature_filters import (
    apply_iqr_bounds_to_nan,
    fit_iqr_bounds,
    remove_correlated_features_train_fit,
)


def prepare_target_dir(path: Path) -> Path:
    for subdir in ["plots", "predictions", "models", "optuna", "logs"]:
        (path / subdir).mkdir(parents=True, exist_ok=True)
    return path


def prepare_directories(results_dir: Path, job_id: str, target1: str, target2: str) -> tuple[Path, Path, Path]:
    root_dir = results_dir / f"job_{job_id}"
    for subdir in ["logs", "processing", "processing/audits", "processing/datasets", "processing/features"]:
        (root_dir / subdir).mkdir(parents=True, exist_ok=True)
    stage1_dir = prepare_target_dir(root_dir / target1)
    stage2_dir = prepare_target_dir(root_dir / target2)
    return root_dir, stage1_dir, stage2_dir


def setup_logging(run_dir: Path, array_id: str) -> logging.Logger:
    logger = logging.getLogger("mortality_modelling")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(run_dir / "logs" / f"training_array_{array_id}.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=str)


def audit_to_csv(run_dir: Path, name: str, records: list[dict[str, Any]]) -> None:
    path = run_dir / "processing" / "audits" / name
    pd.DataFrame(records).to_csv(path, index=False)


def read_drop_columns(utils_json: Path | None) -> list[str]:
    if utils_json is None or not utils_json.exists():
        return []
    with utils_json.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    return list(cfg.get("drop_columns", []))


def coerce_binary_target(y: pd.Series, name: str) -> pd.Series:
    out = pd.to_numeric(y, errors="coerce")
    if out.isna().any():
        bad = y.loc[out.isna()].dropna().unique()[:10]
        raise ValueError(f"Target {name!r} contains non-numeric values that cannot be coerced: {bad}")
    out = out.astype(int)
    unique = set(out.unique())
    if not unique.issubset({0, 1}):
        raise ValueError(f"Target {name!r} must be binary 0/1. Found: {sorted(unique)}")
    return out


def load_data(data_path: Path, drop_cols: list[str], target1: str, target2: str, logger: logging.Logger | None = None):
    df = pd.read_csv(data_path, low_memory=False)
    original_shape = df.shape
    if logger:
        logger.info("Loaded data: %s rows x %s columns from %s", original_shape[0], original_shape[1], data_path)

    missing_targets = [target for target in [target1, target2] if target not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns in dataset: {missing_targets}")

    requested_drop_cols = list(drop_cols)
    drop_cols = [col for col in drop_cols if col in df.columns and col not in {target1, target2}]
    ignored_drop_cols = [col for col in requested_drop_cols if col not in df.columns or col in {target1, target2}]
    df = df.drop(columns=drop_cols, errors="ignore")

    target_na_mask = df[[target1, target2]].isna().any(axis=1)
    dropped_target_na = int(target_na_mask.sum())
    df = df.dropna(subset=[target1, target2]).copy()

    y1 = coerce_binary_target(df[target1], target1)
    y2 = coerce_binary_target(df[target2], target2)
    X = df.drop(columns=[target1, target2])

    index_like_cols = [c for c in ["index", "Unnamed: 0"] if c in X.columns]
    X = X.drop(columns=index_like_cols, errors="ignore")

    load_info = {
        "data_path": str(data_path),
        "original_rows": int(original_shape[0]),
        "original_columns": int(original_shape[1]),
        "rows_dropped_target_na": dropped_target_na,
        "rows_after_target_na_drop": int(len(df)),
        "manual_drop_columns_applied": drop_cols,
        "manual_drop_columns_ignored": ignored_drop_cols,
        "index_like_columns_dropped": index_like_cols,
        "feature_rows": int(X.shape[0]),
        "feature_columns": int(X.shape[1]),
        "target1_counts": y1.value_counts(dropna=False).to_dict(),
        "target2_counts": y2.value_counts(dropna=False).to_dict(),
    }
    return X, y1, y2, load_info


def train_test_split_data(X: pd.DataFrame, y1: pd.Series, y2: pd.Series, test_size: float, seed: int):
    return train_test_split(X, y1, y2, test_size=test_size, stratify=y1, random_state=seed)


def drop_high_na_features(X_train: pd.DataFrame, X_test: pd.DataFrame, *, na_perc_limit: float):
    if na_perc_limit >= 100:
        return X_train, X_test, []
    missing_pct = X_train.isna().mean().mul(100)
    dropped_cols = missing_pct[missing_pct > na_perc_limit].index.tolist()
    audit = [{"feature": col, "train_missing_percent": float(missing_pct.loc[col])} for col in dropped_cols]
    return (X_train.drop(columns=dropped_cols, errors="ignore"), X_test.drop(columns=dropped_cols, errors="ignore"), audit)


def preprocess_train_test_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    na_perc_limit: float = 80.0,
    iqr_multiplier: float = 5.0,
    max_corr: float = 0.95,
    run_dir: Path | None = None,
    logger: logging.Logger | None = None,
):
    X_train = X_train.copy()
    X_test = X_test.copy()

    preprocessing_info: dict[str, Any] = {
        "initial_train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
        "initial_test_shape": [int(X_test.shape[0]), int(X_test.shape[1])],
        "missing_values_before_preprocessing_train": int(X_train.isna().sum().sum()),
        "missing_values_before_preprocessing_test": int(X_test.isna().sum().sum()),
        "date_columns": [],
        "numeric_columns": [],
        "categorical_columns": [],
        "na_perc_limit": na_perc_limit,
        "na_dropped_features": [],
        "iqr_multiplier": iqr_multiplier,
        "iqr_bounds": {},
        "iqr_outliers_train": [],
        "iqr_outliers_test": [],
        "max_corr": max_corr,
        "correlation_dropped_features": [],
        "train_columns_after_encoding": [],
        "median_imputation_values": {},
        "missing_values_after_iqr_train": None,
        "missing_values_after_iqr_test": None,
        "missing_values_after_imputation_train": None,
        "missing_values_after_imputation_test": None,
        "train_columns_final": [],
        "final_train_shape": None,
        "final_test_shape": None,
    }

    for col in list(X_train.columns):
        if "fecha" in col.lower() or "date" in col.lower():
            tr = pd.to_datetime(X_train[col], errors="coerce")
            te = pd.to_datetime(X_test[col], errors="coerce")
            if tr.notna().sum() > 0 or te.notna().sum() > 0:
                X_train[col] = (tr - pd.Timestamp("1970-01-01")).dt.days
                X_test[col] = (te - pd.Timestamp("1970-01-01")).dt.days
                preprocessing_info["date_columns"].append(col)

    X_train, X_test, na_audit = drop_high_na_features(X_train, X_test, na_perc_limit=na_perc_limit)

    iqr_bounds = fit_iqr_bounds(X_train, iqr_multiplier=iqr_multiplier)
    X_train, iqr_train_audit = apply_iqr_bounds_to_nan(X_train, iqr_bounds)
    X_test, iqr_test_audit = apply_iqr_bounds_to_nan(X_test, iqr_bounds)

    preprocessing_info["iqr_bounds"] = {col: {"lower_bound": lower, "upper_bound": upper} for col, (lower, upper) in iqr_bounds.items()}
    preprocessing_info["iqr_outliers_train"] = iqr_train_audit
    preprocessing_info["iqr_outliers_test"] = iqr_test_audit
    preprocessing_info["na_dropped_features"] = na_audit

    numeric_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]
    preprocessing_info["numeric_columns"] = numeric_cols
    preprocessing_info["categorical_columns"] = categorical_cols

    for col in numeric_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
        median = X_train[col].median()
        if pd.isna(median):
            median = 0
        preprocessing_info["median_imputation_values"][col] = float(median) if pd.notna(median) else 0.0
        X_train[col] = X_train[col].fillna(median)
        X_test[col] = X_test[col].fillna(median)

    for col in categorical_cols:
        X_train[col] = X_train[col].astype("string").fillna("MISSING")
        X_test[col] = X_test[col].astype("string").fillna("MISSING")

    X_train_enc = pd.get_dummies(X_train, columns=categorical_cols, dummy_na=False, dtype=float)
    X_test_enc = pd.get_dummies(X_test, columns=categorical_cols, dummy_na=False, dtype=float)
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)
    X_train_enc = X_train_enc.astype(float)
    X_test_enc = X_test_enc.astype(float)

    X_train_final, X_test_final, corr_audit = remove_correlated_features_train_fit(X_train_enc, X_test_enc, threshold=max_corr)

    preprocessing_info["correlation_dropped_features"] = corr_audit
    preprocessing_info["train_columns_after_encoding"] = X_train_enc.columns.tolist()
    preprocessing_info["train_columns_final"] = X_train_final.columns.tolist()
    preprocessing_info["final_train_shape"] = [int(X_train_final.shape[0]), int(X_train_final.shape[1])]
    preprocessing_info["final_test_shape"] = [int(X_test_final.shape[0]), int(X_test_final.shape[1])]

    if run_dir is not None:
        proc = run_dir / "processing"
        save_json(proc / "preprocessing_info.json", preprocessing_info)
        audit_to_csv(run_dir, "na_dropped_features.csv", na_audit)
        audit_to_csv(run_dir, "iqr_outliers_train.csv", iqr_train_audit)
        audit_to_csv(run_dir, "iqr_outliers_test.csv", iqr_test_audit)
        audit_to_csv(run_dir, "correlation_dropped_features.csv", corr_audit)
        pd.Series(X_train_final.columns, name="feature").to_csv(proc / "features" / "final_feature_list.csv", index=False)
        pd.Series(X_train_enc.columns, name="feature").to_csv(proc / "features" / "encoded_feature_list_before_corr_drop.csv", index=False)
        pd.DataFrame([preprocessing_info["median_imputation_values"]]).T.reset_index().rename(columns={"index": "feature", 0: "median_imputation_value"}).to_csv(proc / "audits" / "median_imputation_values.csv", index=False)
        X_train_final.to_csv(proc / "datasets" / "X_train_processed.csv", index_label="row_index")
        X_test_final.to_csv(proc / "datasets" / "X_test_processed.csv", index_label="row_index")

    return X_train_final, X_test_final, preprocessing_info


def compute_imbalance(y: pd.Series) -> float:
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return float(neg / pos) if pos > 0 else 1.0


def safe_cv_splits(y: pd.Series, requested_splits: int) -> int:
    counts = y.value_counts()
    if len(counts) < 2:
        raise ValueError(f"Cannot train binary classifier with one class only: {counts.to_dict()}")
    return max(2, min(requested_splits, int(counts.min())))


def save_config(args: argparse.Namespace, drop_cols: list[str], preprocessing_info: dict[str, Any], run_dir: Path) -> None:
    run_config = {"args": {k: str(v) for k, v in vars(args).items()}, "drop_columns": drop_cols, "preprocessing_info": preprocessing_info}
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)
