#!/usr/bin/env python3
"""
Three-level independent hierarchical classifier for sepsis outcomes.

Pipeline
--------
  Level 1 – sepsis (binary)
    Trained on original features only.

  Level 2 – resultado_hemo_grouped (multiclass)
    Trained on original features only (no Level-1 OOF cascade).
    Very rare blood-culture classes are dropped automatically.
    SHAP-RFECV is applied to select the best feature subset.

  Level 3 – resistente_cefalosporina (binary)
    Trained on original features only (no Level-1 or Level-2 OOF cascade).
    Row scope: only patients with a positive blood culture
               (resultado_hemo_grouped != "NEGATIVE").
    SHAP-RFECV is applied to select the best feature subset.

Each level is trained independently – OOF predictions from one level are NOT
injected as features into the next level.  This allows a clean comparison
against the cascade variant.

SHAP-RFECV is run at every level; the selected feature list and full score
history are saved in each level's summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import shap

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
JOB_ID = os.environ.get("SLURM_JOB_ID", 1)
TODAY = datetime.today().strftime("%Y%m%d%H%M%S")

FOCUS_MAP = {
    1: "pulmonar",
    2: "intraabdominal",
    3: "biliar",
    4: "urinario",
    5: "cardiovascular",
    6: "piel",
    7: "sistema nervioso central",
    8: "cateter venoso",
    9: "vías altas respiratorias",
    10: "osteoarticular",
    11: "genital",
    12: "desconocido",
}

# All potential target columns – none of these should appear as features
TARGET_REMOVE = [
    "sepsis",
    "resultado_hemo",
    "resultado_hemo_grouped",
    "resultado_hemo_mo",
    "all_cult_org",
    "infected_yes_no",
    "bmr_etiologia",
    "fenotipo_resistencia",
    "fenotipo_resistencia_individual",
    "fenotipo_resistencia_grouped",
    "resistente_cefalosporina",
    "resistente_cefalosporina_multi",
]

DELETE_COLUMNS = []

FOCUS_TO_EXCLUDE = {
    "piel",
    "osteoarticular",
    "biliar",
    "genital",
    "sistema nervioso central",
    "cateter venoso",
    "vías altas respiratorias",
    "cardiovascular",
}

# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------


def safe_drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        try:
            df = df.drop(columns=col)
        except KeyError:
            print(f"Warning: column '{col}' not found – skipping drop.")
    return df


def load_processed_dataframe(csv_path: Path, cols_to_delete: List[str]) -> pd.DataFrame:
    """Load dataset and apply focus filter. No target-based row filtering here."""
    df = pd.read_csv(csv_path)
    if "foco" in df.columns:
        df["foco"] = df["foco"].map(FOCUS_MAP).fillna(df["foco"])
        df = df[~df["foco"].isin(FOCUS_TO_EXCLUDE)]
    df = safe_drop_columns(df, cols_to_delete)
    return df


def impute_missing_values(loaded_df: pd.DataFrame, exclude_cols: set) -> pd.DataFrame:
    """Impute missing values; target/weight columns are kept as-is.

    Strategy
    --------
    * Binary (0/1) numeric columns → mode (SimpleImputer).
    * All other numeric columns (continuous AND low-cardinality ordinal scores
      such as SOFA sub-scores, bilirrubina 0-4, snc_glasgow 0-4, respiracion
      0-3) → KNNImputer(k=10, distance-weighted).  Using KNN for ordinal
      scores is strictly better than mode because it exploits the patient's
      other measurements; mode always collapses to the most common value
      regardless of clinical context.
    * String/category columns → mode (SimpleImputer).
    * Missing-indicator flags: for every numeric column whose missing rate
      exceeds 10 %, a companion ``<col>_missing`` binary column is added
      *before* imputation.  Tree models can use these flags directly as a
      signal that the original value was absent (informative missingness,
      e.g. SOFA not recorded often means the patient was less severe).
    """
    df_copy = loaded_df.drop(columns=list(exclude_cols), errors="ignore").copy()
    numeric_cols = df_copy.select_dtypes(include=["int", "float"]).columns.tolist()
    cat_cols = df_copy.select_dtypes(include=["object", "category"]).columns.tolist()

    binary_cols = [c for c in numeric_cols if set(df_copy[c].dropna().unique()) <= {0, 1}]
    # Ordinal clinical scores (low-cardinality) and continuous values both
    # benefit from multivariate KNN — route all non-binary numeric to KNN.
    knn_cols = [c for c in numeric_cols if c not in binary_cols]

    # Add missing-indicator flags for numerics with >10 % missingness.
    # These flags remain even after imputation so the model can learn from them.
    high_missing = [
        c for c in numeric_cols
        if df_copy[c].isna().mean() > 0.10
    ]
    for col in high_missing:
        df_copy[f"{col}_missing"] = df_copy[col].isna().astype(int)

    if binary_cols:
        imp = SimpleImputer(strategy="most_frequent")
        df_copy[binary_cols] = imp.fit_transform(df_copy[binary_cols]).astype(int)
    if knn_cols:
        # k=10 gives more stable estimates than k=5 for datasets of ~3 000+ rows.
        imp = KNNImputer(n_neighbors=10, weights="distance")
        df_copy[knn_cols] = imp.fit_transform(df_copy[knn_cols])
        # Re-cast originally-integer ordinal columns back to int after KNN.
        for col in knn_cols:
            if loaded_df[col].dropna().astype(float).apply(float.is_integer).all():
                df_copy[col] = df_copy[col].round().astype(int)
    if cat_cols:
        imp = SimpleImputer(strategy="most_frequent")
        df_copy[cat_cols] = imp.fit_transform(df_copy[cat_cols]).astype(str)
    print("Imputed values and included marker columns for high_missings")
    for col in exclude_cols:
        if col in loaded_df.columns:
            df_copy[col] = loaded_df[col]
    return df_copy


def preprocess_train_test_features(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    na_perc_limit: float,
    impute_missing: bool,
    categorical_for_dummies: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Leakage-safe feature preprocessing: fit on train, transform test."""
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()

    # Drop target-related helper columns if present
    cols_to_drop = [c for c in ["resultado_hemo_multilabel", "dominant_all_cult_org"] if c in X_train.columns]
    if cols_to_drop:
        print(f"  Dropping target-related columns: {cols_to_drop}")
        X_train = X_train.drop(columns=cols_to_drop)
        X_test = X_test.drop(columns=cols_to_drop, errors="ignore")

    # Drop high-NA columns based on TRAIN only
    dropped_na = [col for col in X_train.columns if X_train[col].isna().mean() > na_perc_limit]
    if dropped_na:
        print(f"Dropping {len(dropped_na)} high-NA columns (train-based).")
        X_train = X_train.drop(columns=dropped_na)
        X_test = X_test.drop(columns=dropped_na, errors="ignore")

    if not len(X_train.columns):
        raise ValueError("No feature columns remain after NA filtering.")

    numeric_cols = X_train.select_dtypes(include=["int", "float"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    binary_cols = [c for c in numeric_cols if set(X_train[c].dropna().unique()) <= {0, 1}]
    knn_cols = [c for c in numeric_cols if c not in binary_cols]

    # Missingness flags based on TRAIN only
    high_missing = [c for c in numeric_cols if X_train[c].isna().mean() > 0.10]
    for col in high_missing:
        X_train[f"{col}_missing"] = X_train[col].isna().astype(int)
        X_test[f"{col}_missing"] = X_test[col].isna().astype(int)

    if impute_missing:
        if binary_cols:
            imp = SimpleImputer(strategy="most_frequent")
            X_train[binary_cols] = imp.fit_transform(X_train[binary_cols]).astype(int)
            X_test[binary_cols] = imp.transform(X_test[binary_cols]).astype(int)
        if knn_cols:
            imp = KNNImputer(n_neighbors=10, weights="distance")
            X_train[knn_cols] = imp.fit_transform(X_train[knn_cols])
            X_test[knn_cols] = imp.transform(X_test[knn_cols])
            for col in knn_cols:
                if X_train_raw[col].dropna().astype(float).apply(float.is_integer).all():
                    X_train[col] = X_train[col].round().astype(int)
                    X_test[col] = X_test[col].round().astype(int)
        if cat_cols:
            imp = SimpleImputer(strategy="most_frequent")
            X_train[cat_cols] = imp.fit_transform(X_train[cat_cols]).astype(str)
            X_test[cat_cols] = imp.transform(X_test[cat_cols]).astype(str)
        print("Imputed missing values with train-fitted imputers.")
    else:
        train_mask = X_train.notna().all(axis=1)
        test_mask = X_test.notna().all(axis=1)
        X_train = X_train.loc[train_mask].copy()
        X_test = X_test.loc[test_mask].copy()

    # One-hot encode chosen categorical columns with train-fitted schema
    dummies_cols = [c for c in categorical_for_dummies if c in X_train.columns]
    if dummies_cols:
        X_train = pd.get_dummies(X_train, columns=dummies_cols, drop_first=False)
        X_test = pd.get_dummies(X_test, columns=dummies_cols, drop_first=False)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        X_train.columns = X_train.columns.str.replace("[^0-9a-zA-Z_]+", "_", regex=True)
        X_test.columns = X_test.columns.str.replace("[^0-9a-zA-Z_]+", "_", regex=True)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    return X_train, X_test


def compute_balanced_sample_weight(
    labels: pd.Series, base_sample_weight: Optional[pd.Series] = None
) -> pd.Series:
    classes = np.unique(labels)
    cw = compute_class_weight("balanced", classes=classes, y=labels)
    balanced = labels.map(dict(zip(classes, cw)))
    if base_sample_weight is not None:
        balanced = balanced * pd.Series(base_sample_weight, index=labels.index)
    return balanced


def _build_ranking_model(model_type: str, y: pd.Series, random_state: int) -> object:
    """Return a fast, fixed-param model of *model_type* used only for importance ranking.

    Parameters are intentionally conservative (shallow, few estimators) so the
    ranking step finishes quickly.  The result is never used for prediction.
    """
    n_classes = int(y.nunique())
    is_multi = n_classes > 2

    if model_type == "lgbm":
        params: Dict = {
            "n_estimators": 200, "num_leaves": 31, "max_depth": 6,
            "class_weight": "balanced", "n_jobs": N_CPUS,
            "random_state": random_state, "verbosity": -1,
        }
        if is_multi:
            params["objective"] = "multiclass"
            params["num_class"] = n_classes
        else:
            params["objective"] = "binary"
        return LGBMClassifier(**params)

    elif model_type == "xgb":
        if is_multi:
            return XGBClassifier(
                objective="multi:softprob", num_class=n_classes,
                n_estimators=200, max_depth=6,
                random_state=random_state, n_jobs=N_CPUS, verbosity=0,
            )
        n_pos = float((y == 1).sum())
        spw = float(len(y) - n_pos) / max(n_pos, 1.0)
        return XGBClassifier(
            objective="binary:logistic", n_estimators=200, max_depth=6,
            scale_pos_weight=spw,
            random_state=random_state, n_jobs=N_CPUS, verbosity=0,
        )

    elif model_type == "catb":
        return CatBoostClassifier(
            iterations=200, depth=6,
            auto_class_weights="Balanced", verbose=False, random_state=42, thread_count=N_CPUS
        )

    else:  # "rf" or anything else
        return RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight="balanced_subsample",
            random_state=random_state, n_jobs=N_CPUS,
        )


def _shap_importance(model, X: pd.DataFrame) -> np.ndarray:
    """Return mean absolute SHAP values (shape [n_features,]) using TreeExplainer.

    Handles binary (2-D array), multiclass list-of-arrays (XGB/RF style), and
    multiclass 3-D array (LGBM style) transparently.

    Raises ImportError if shap is not installed (caller handles the fallback).
    """
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)

    if isinstance(sv, list):
        # XGB / RF multiclass: list of (n_samples, n_features), one per class
        return np.mean([np.abs(a).mean(axis=0) for a in sv], axis=0)
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        # LGBM multiclass: (n_samples, n_features, n_classes)
        return np.abs(sv).mean(axis=(0, 2))
    # Binary: (n_samples, n_features)
    return np.abs(sv).mean(axis=0)


def cap_features(
    features: List[str],
    X: pd.DataFrame,
    y: pd.Series,
    max_features: Optional[int],
    random_state: int,
    model_type: Optional[str] = "rf",
    rank_model: Optional[BaseEstimator] = None,
) -> List[str]:
    """Trim *features* to the top *max_features* by mean |SHAP| value.

    A lightweight version of *model_type* (200 estimators, fixed depth) is
    fitted on *X[features]* solely to rank candidates – no hyperparameter
    tuning is performed.  Mean absolute SHAP values are used for ranking when
    the ``shap`` package is available; the model's own ``feature_importances_``
    are used as a fallback otherwise.

    Using the same model family as the one being trained avoids the MDI bias
    of Random Forest importance (bias toward high-cardinality / continuous
    features) and makes the selected subset consistent with what the final
    model will actually rely on.

    If *max_features* is None, or the feature list is already within budget,
    the original list is returned unchanged.
    """
    if max_features is None or len(features) <= max_features:
        return features
    if rank_model is None:
        model = _build_ranking_model(model_type, y, random_state)
    else:
        model = copy.deepcopy(rank_model)
    model.fit(X[features], y)

    try:
        raw = _shap_importance(model, X[features])
        method = "mean |SHAP|"
    except ImportError:
        raw = model.feature_importances_
        method = "feature_importances_ (install shap for SHAP-based ranking)"
    except Exception as exc:
        # TreeExplainer can fail for some model configurations; degrade gracefully
        print(f"  SHAP ranking failed ({exc}); falling back to feature_importances_.")
        raw = model.feature_importances_
        method = "feature_importances_"

    importance = pd.Series(raw, index=features)
    top_features = importance.nlargest(max_features).index.tolist()
    print(
        f"  Feature cap applied: {len(features)} → {max_features} features"
        f" (ranked by {method})."
    )
    return top_features


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------


def build_binary_model(model_type: str, params: Dict) -> object:
    cls_map = {
        "xgb": XGBClassifier,
        "lgbm": LGBMClassifier,
        "rf": RandomForestClassifier,
        "catb": CatBoostClassifier,
    }
    if model_type not in cls_map:
        raise ValueError(f"Unknown model_type '{model_type}'.")
    return cls_map[model_type](**params)


def build_multiclass_model(model_type: str, params: Dict, num_classes: int) -> object:
    if model_type == "xgb":
        params["objective"] = "multi:softprob"
        params["num_class"] = num_classes
        return XGBClassifier(**params)
    elif model_type == "lgbm":
        params["objective"] = "multiclass"
        params["num_class"] = num_classes
        return LGBMClassifier(**params)
    elif model_type == "rf":
        params.setdefault("class_weight", "balanced")
        return RandomForestClassifier(**params)
    elif model_type == "catb":
        params.setdefault("loss_function", "MultiClass")
        params.setdefault("auto_class_weights", "Balanced")
        params.setdefault("verbose", False)
        params.setdefault("custom_metric", "PRAUC")
        params.setdefault("thread_count", N_CPUS)
        params.setdefault("random_state", 42)
        return CatBoostClassifier(**params)
    else:
        raise ValueError(f"Unsupported model type for multiclass: '{model_type}'.")


def _fit_model(model, model_type: str, X_tr, y_tr, X_va=None, y_va=None, sample_weight=None) -> None:
    """Fit a model; CatBoost uses early-stopping with the validation fold."""
    sw = sample_weight
    if model_type == "catb" and X_va is not None:
        n_iter = getattr(model, "iterations", 500)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=max(20, int(0.05 * n_iter)),
            verbose=False,
            sample_weight=sw,
        )
    else:
        if sw is not None:
            model.fit(X_tr, y_tr, sample_weight=sw)
        else:
            model.fit(X_tr, y_tr)


def _fit_calibrated_or_base(
    base_model,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    *,
    max_cv: int = 5,
    method: str = "isotonic",
):
    """Fit calibrated model when feasible; otherwise fit and return base model."""
    class_counts = pd.Series(y_tr).value_counts()
    if class_counts.empty or int(class_counts.min()) < 2:
        print(
            "  Warning: skipping calibration because at least one class has <2 samples "
            f"(counts={class_counts.to_dict()})."
        )
        base_model.fit(X_tr, y_tr)
        return base_model

    cal_cv = min(max_cv, int(class_counts.min()))
    model = CalibratedClassifierCV(base_model, cv=cal_cv, method=method)
    model.fit(X_tr, y_tr)
    return model


# ---------------------------------------------------------------------------
# Optuna optimisation
# ---------------------------------------------------------------------------


def _find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Find the probability threshold that maximises Youden's J (sensitivity + specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(np.clip(thresholds[best_idx], 0.05, 0.95))


def _create_optuna_study(
    *,
    direction: str,
    study_name: Optional[str],
    storage: Optional[str],
    load_if_exists: bool,
):
    if storage:
        return optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=storage,
            load_if_exists=load_if_exists,
        )
    return optuna.create_study(direction=direction)


def _binary_params_for_trial(trial: optuna.Trial, model_type: str, scale_pos_weight: float, random_state: int) -> Dict:
    if model_type == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "class_weight": "balanced",
            "n_jobs": N_CPUS,
            "random_state": random_state,
        }
    elif model_type == "xgb":
        return {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",  # PR-AUC for imbalanced optimization
            "tree_method": "hist",
            "n_estimators": trial.suggest_int("n_estimators", 300, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 20.0),
            "random_state": random_state,
            "n_jobs": N_CPUS,
        }
    elif model_type == "lgbm":
        return {
            "objective": "binary",
            "metric": "prauc",  # PRAUC for imbalanced optimization (LGBM uses 'metric' not 'eval_metric')
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int("n_estimators", 300, 4000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 512),
            "max_depth": trial.suggest_int("max_depth", 3, 16),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 200),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "class_weight": "balanced",
            "n_jobs": N_CPUS,
            "random_state": random_state,
            "verbosity": -1,
        }
    elif model_type == "catb":
        return {
            "iterations": trial.suggest_int("iterations", 300, 2000),
            "custom_metric": "PRAUC",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
            "auto_class_weights": "Balanced",
            "verbose": False,
            "thread_count": N_CPUS,
            "random_state": 42,
        }
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'")


def optimise_binary_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int,
    n_trials: int,
    random_state: int,
    sample_weight: Optional[pd.Series],
    model_type: str,
    study_name: Optional[str] = None,
    optuna_storage: Optional[str] = None,
    optuna_load_if_exists: bool = False,
) -> Tuple[Dict, float, optuna.Study]:
    """Tune a binary classifier with Optuna maximising PR-AUC (Average Precision).

    Returns best_params, best_threshold (Youden's J on OOF), study.
    The threshold is found after Optuna by re-running a single OOF pass with
    the best params — this is thread-safe regardless of n_jobs.
    """
    weight_series = (
        sample_weight.reindex(X.index)
        if isinstance(sample_weight, pd.Series)
        else (pd.Series(sample_weight, index=X.index) if sample_weight is not None else None)
    )
    if weight_series is not None:
        pos_mask = y == 1
        spw = float(weight_series[~pos_mask].sum()) / float(weight_series[pos_mask].sum() or 1)
    else:
        n_pos = float((y == 1).sum())
        spw = float(len(y) - n_pos) / (n_pos or 1)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = _binary_params_for_trial(trial, model_type, spw, random_state)
        scores = []
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            w_tr = weight_series.iloc[tr_idx].to_numpy() if weight_series is not None else None
            model = build_binary_model(model_type, params.copy())
            _fit_model(model, model_type, X_tr, y_tr, X_va, y_va, sample_weight=w_tr)
            proba = model.predict_proba(X_va)[:, 1]
            scores.append(average_precision_score(y_va, proba))
        if trial.number % 100 == 0:
            print(f"Trial {trial.number}: score={float(np.mean(scores))}")
        return float(np.mean(scores))

    study = _create_optuna_study(
        direction="maximize",
        study_name=study_name,
        storage=optuna_storage,
        load_if_exists=optuna_load_if_exists,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1, gc_after_trial=True)
    best = study.best_trial.params.copy()

    # Find optimal decision threshold via OOF pass with best hyperparams
    oof_true_parts, oof_proba_parts = [], []
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        w_tr = weight_series.iloc[tr_idx].to_numpy() if weight_series is not None else None
        m = build_binary_model(model_type, best.copy())
        _fit_model(m, model_type, X_tr, y_tr, sample_weight=w_tr)
        oof_proba_parts.append(m.predict_proba(X_va)[:, 1])
        oof_true_parts.append(y_va.to_numpy())
    threshold = _find_best_threshold(
        np.concatenate(oof_true_parts), np.concatenate(oof_proba_parts)
    )
    return best, threshold, study


def _multiclass_params_for_trial(
    trial: optuna.Trial, model_type: str, num_classes: int, random_state: int
) -> Dict:
    if model_type == "xgb":
        return {
            "objective": "multi:softprob",
            "eval_metric": "prauc_mu",  # XGB's multiclass PR-AUC (macro-averaged)
            "tree_method": "hist",
            "num_class": num_classes,
            "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": random_state,
            "n_jobs": N_CPUS,
        }
    elif model_type == "lgbm":
        return {
            "objective": "multiclass",
            "boosting_type": "gbdt",
            "eval_metric": "prauc_mu",  # LGBM's multiclass PR-AUC (macro-averaged)
            "num_class": num_classes,
            "n_estimators": trial.suggest_int("n_estimators", 300, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 14),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 150),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "class_weight": "balanced",
            "n_jobs": N_CPUS,
            "random_state": random_state,
            "verbosity": -1,
        }
    elif model_type == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "class_weight": "balanced",
            "n_jobs": N_CPUS,
            "random_state": random_state,
        }
    elif model_type == "catb":
        return {
            "loss_function": "MultiClass",
            "iterations": trial.suggest_int("iterations", 300, 2000),
            # CatBoost multiclass eval_metric must be a single scalar metric.
            # Keep PRAUC for reporting as custom_metric.
            "eval_metric": "MultiClass",
            "custom_metric": "PRAUC",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
            "auto_class_weights": "Balanced",
            "verbose": False,
            "thread_count": N_CPUS,
            "random_state": 42,
        }
    else:
        raise ValueError(f"Unsupported multiclass model_type: '{model_type}'")


def optimise_multiclass_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int,
    n_trials: int,
    random_state: int,
    sample_weight: Optional[pd.Series],
    model_type: str,
    num_classes: int,
    study_name: Optional[str] = None,
    optuna_storage: Optional[str] = None,
    optuna_load_if_exists: bool = False,
) -> Tuple[Dict, optuna.Study]:
    """Tune a multiclass classifier with Optuna maximising macro F1.

    sample_weight is applied during each fold's fit so that balanced class
    weights actually influence the search (previously they were silently ignored).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    weight_series = (
        sample_weight.reindex(X.index)
        if isinstance(sample_weight, pd.Series)
        else (pd.Series(sample_weight, index=X.index) if sample_weight is not None else None)
    )

    def objective(trial: optuna.Trial) -> float:
        params = _multiclass_params_for_trial(trial, model_type, num_classes, random_state)
        scores = []
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            if y_tr.nunique() < 2:
                # Skip fold with only one class in training
                continue
            w_tr = weight_series.iloc[tr_idx].to_numpy() if weight_series is not None else None
            model = build_multiclass_model(model_type, params.copy(), num_classes)
            try:
                if w_tr is not None:
                    model.fit(X_tr, y_tr, sample_weight=w_tr)
                else:
                    model.fit(X_tr, y_tr)
                y_pred = model.predict(X_va)
                scores.append(f1_score(y_va, y_pred, average="macro", zero_division=0))
            except Exception as e:
                print(f"Skipping fold due to error: {e}")
                continue
        if not scores:
            return 0.0
        if trial.number % 100 == 0:
            print(f"Trial {trial.number}: score={float(np.mean(scores))}")
        return float(np.mean(scores))

    study = _create_optuna_study(
        direction="maximize",
        study_name=study_name,
        storage=optuna_storage,
        load_if_exists=optuna_load_if_exists,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1, gc_after_trial=True)
    return study.best_trial.params.copy(), study


# ---------------------------------------------------------------------------
# Out-of-fold prediction generators
# ---------------------------------------------------------------------------


def generate_oof_probas_binary(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: Dict,
    model_type: str,
    n_splits: int,
    random_state: int,
) -> np.ndarray:
    """Return calibrated OOF positive-class probabilities (shape [n,]) for a binary classifier.

    Within each outer fold the base model is wrapped with CalibratedClassifierCV
    (isotonic, cv=3) so that the probabilities fed to the next cascade level are
    well-calibrated and consistent with what the final calibrated model will produce.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(X))
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        base = build_binary_model(model_type, best_params.copy())
        # cv=3 calibration within the training fold (never touches the OOF validation slice)
        model = _fit_calibrated_or_base(base, X_tr, y_tr, max_cv=3, method="isotonic")
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
    return oof


def generate_oof_probas_multiclass(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: Dict,
    model_type: str,
    num_classes: int,
    n_splits: int,
    random_state: int,
) -> np.ndarray:
    """Return calibrated OOF per-class probabilities (shape [n, num_classes]).

    Same calibration strategy as generate_oof_probas_binary.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros((len(X), num_classes))
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        base = build_multiclass_model(model_type, best_params.copy(), num_classes)
        model = _fit_calibrated_or_base(base, X_tr, y_tr, max_cv=3, method="isotonic")
        oof[va_idx] = model.predict_proba(X_va)
    return oof


def shap_rfecv(
        model: BaseEstimator,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        cv: int = 5,
        min_features: int = 5,
        max_features: int = 50,
        scoring: str = "roc_auc",
        random_state: int = 42,
    ) -> Tuple[List[str], Dict[str, Dict[str, object]]]:
    """
    SHAP-based Recursive Feature Elimination with Cross Validation.

    At each step the CV score is recorded for the *current* feature set,
    then the globally least-important feature (by mean |SHAP| on the full
    training data) is removed.  After all iterations the feature set whose
    CV score was highest is returned together with the full score history.
    """
    remaining_features = list(X.columns)
    y = np.asarray(y)
    n_classes = len(np.unique(y))
    is_multiclass = n_classes > 2

    if n_classes < 2:
        print(
            "  Warning: shap_rfecv received a single-class target; "
            "skipping RFECV and returning all features."
        )
        return remaining_features, {}

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    history: Dict[str, Dict[str, object]] = {}

    def _cv_metrics(features: List[str]) -> Dict[str, float]:
        """Return mean CV metrics for *features* using the current skf splits."""
        if scoring not in ["roc_auc", "pr_auc"]:
            raise ValueError("Only roc_auc and pr_auc are supported for shap_rfecv")
        roc_auc_scores: List[float] = []
        pr_auc_scores: List[float] = []
        f1_scores: List[float] = []
        precision_scores: List[float] = []
        recall_scores: List[float] = []
        for train_idx, val_idx in skf.split(X[features], y):
            X_tr = X.iloc[train_idx][features]
            X_va = X.iloc[val_idx][features]
            y_tr, y_va = y[train_idx], y[val_idx]
            
            # Skip folds where training set has only one unique class (common in imbalanced data)
            if len(np.unique(y_tr)) < 2:
                print(f"    Skipping fold with single class in training set: {np.unique(y_tr)}")
                continue
            
            est = copy.deepcopy(model)
            try:
                est.fit(X_tr, y_tr)
                if is_multiclass:
                    proba = est.predict_proba(X_va)
                    # For multiclass, use macro one-vs-rest averages.
                    pr_auc = float(np.mean([
                        average_precision_score((y_va == i).astype(int), proba[:, i])
                        for i in range(n_classes)
                    ]))
                    roc_auc = float(roc_auc_score(y_va, proba, multi_class="ovr", average="macro"))
                    y_pred = np.argmax(proba, axis=1)
                else:
                    proba = est.predict_proba(X_va)[:, 1]
                    pr_auc = float(average_precision_score(y_va, proba))
                    roc_auc = float(roc_auc_score(y_va, proba))
                    y_pred = (proba >= 0.5).astype(int)
                f1 = f1_score(y_va, y_pred, average="macro", zero_division=0)
                precision = precision_score(y_va, y_pred, average="macro", zero_division=0)
                recall = recall_score(y_va, y_pred, average="macro", zero_division=0)
            except Exception as e:
                print(f"    Exception during fold fit/eval: {type(e).__name__}: {str(e)[:100]}")
                roc_auc = 0.0
                pr_auc = 0.0
                f1 = 0.0
                precision = 0.0
                recall = 0.0
            roc_auc_scores.append(float(roc_auc))
            pr_auc_scores.append(float(pr_auc))
            f1_scores.append(float(f1))
            precision_scores.append(float(precision))
            recall_scores.append(float(recall))
        if not roc_auc_scores:
            print("    Warning: no valid CV folds were available for shap_rfecv; returning zeroed metrics.")
            return {
                "roc_auc": 0.0,
                "pr_auc": 0.0,
                "f1_macro": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
            }
        return {
            "roc_auc": float(np.mean(roc_auc_scores)),
            "pr_auc": float(np.mean(pr_auc_scores)),
            "f1_macro": float(np.mean(f1_scores)),
            "precision_macro": float(np.mean(precision_scores)),
            "recall_macro": float(np.mean(recall_scores)),
        }

    def _global_worst_feature(features: List[str]) -> str:
        """Fit on all training data; return the least-important feature by mean |SHAP|."""
        est = copy.deepcopy(model)
        est.fit(X[features], y)
        try:
            explainer = shap.TreeExplainer(est)
        except Exception:
            explainer = shap.Explainer(est, X[features])
        sv = explainer.shap_values(X[features])
        if isinstance(sv, list):
            # XGB / RF multiclass: list of (n_samples, n_features), one per class
            shap_matrix = np.mean([np.abs(a) for a in sv], axis=0)
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            # LGBM multiclass: (n_samples, n_features, n_classes)
            shap_matrix = np.abs(sv).mean(axis=2)
        else:
            shap_matrix = np.abs(sv)
        importance = np.mean(shap_matrix, axis=0)
        return features[int(np.argmin(importance))]

    while len(remaining_features) > min_features:
        # Score the CURRENT feature set — label and score are in sync
        metrics = _cv_metrics(remaining_features)
        history[str(len(remaining_features))] = {
            "metrics": metrics,
            "features": remaining_features.copy(),
        }
        primary_metric = "pr_auc" if scoring == "pr_auc" else "roc_auc"
        primary_val = metrics.get(primary_metric, 0.0)
        print(
            f"Features: {len(remaining_features)} | "
            f"ROC_AUC: {metrics['roc_auc']:.4f} | PR_AUC: {metrics['pr_auc']:.4f} | "
            f"{primary_metric.upper()}: {primary_val:.4f} | F1: {metrics['f1_macro']:.4f} | "
            f"Precision: {metrics['precision_macro']:.4f} | Recall: {metrics['recall_macro']:.4f}"
        )

        # Remove the globally least important feature
        removed = _global_worst_feature(remaining_features)
        remaining_features.remove(removed)
        print(f"Removed feature: {removed}")

    # Score and record the final minimal feature set
    final_metrics = _cv_metrics(remaining_features)
    history[str(len(remaining_features))] = {
        "metrics": final_metrics,
        "features": remaining_features.copy(),
    }
    primary_metric = "pr_auc" if scoring == "pr_auc" else "roc_auc"
    primary_val = final_metrics.get(primary_metric, 0.0)
    print(
        f"Features: {len(remaining_features)} | "
        f"ROC_AUC: {final_metrics['roc_auc']:.4f} | PR_AUC: {final_metrics['pr_auc']:.4f} | "
        f"{primary_metric.upper()}: {primary_val:.4f} | F1: {final_metrics['f1_macro']:.4f} | "
        f"Precision: {final_metrics['precision_macro']:.4f} | Recall: {final_metrics['recall_macro']:.4f}"
    )

    # Return the feature set whose CV score was highest (true RFECV selection),
    # constrained to at most max_features.  Keys are strings; cast before comparing.
    primary_metric = "pr_auc" if scoring == "pr_auc" else "roc_auc"
    best_n = max(
        history,
        key=lambda k: history[k]["metrics"][primary_metric] if int(k) <= max_features else 0.0,
    )
    best_score = history[best_n]["metrics"][primary_metric]
    best_features = history[best_n]["features"]
    print(
        f"Best CV score ({primary_metric}): {best_score:.4f} at {best_n} features → "
        f"Selected {len(best_features)} features."
    )
    return best_features, history

# ---------------------------------------------------------------------------
# Main training orchestration
# ---------------------------------------------------------------------------


def remove_correlated_features(
    X: pd.DataFrame,
    threshold: float = 0.90,
) -> List[str]:
    """Return the subset of *X.columns* that survives a Spearman correlation filter.

    All feature columns are numeric after ``pd.get_dummies``, so Spearman rank
    correlation is a single valid measure for every column type:

    * **Binary 0/1** (one-hot dummies, presence/absence flags):
      Spearman = phi coefficient for binary pairs, which equals Pearson.
    * **Ordinal** (recoded vital signs 0-3, clinical scores 0-4):
      Spearman is designed for ordered discrete data.
    * **Continuous** (raw vital signs, labs, counts):
      Spearman is valid and more robust to outliers than Pearson.

    No Cramér's V is needed because string/categorical columns have already
    been one-hot encoded before this function is called.

    Algorithm
    ---------
    1. Sort columns by **descending variance** so the more informative
       representation wins when a pair must be pruned.  For example,
       ``temperatura`` (continuous, higher variance) beats
       ``temperatura_recoded`` (0-3 quartile bin, lower variance).
    2. Walk the sorted list; keep each column unless it is already
       scheduled for removal because it is too similar to an earlier-kept
       column.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix built from **training rows only**.  Test rows must
        not be included to avoid test-set leakage into the correlation
        estimates.
    threshold : float
        Absolute Spearman correlation above which a pair is considered
        redundant.  Default 0.90.

    Returns
    -------
    List[str]
        Column names to keep, preserved in their original order.
    """
    if threshold >= 1.0:
        return X.columns.tolist()

    # Filter to numeric columns only (exclude any object/string columns)
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) < len(X.columns):
        dropped_non_numeric = set(X.columns) - set(numeric_cols)
        print(f"  Removing {len(dropped_non_numeric)} non-numeric columns before correlation: {list(dropped_non_numeric)[:5]}...")
    X_numeric = X[numeric_cols]

    # Pairwise Spearman on training data.  Constant columns yield NaN
    # correlations; fill with 0 so they are treated as uncorrelated and
    # kept (SHAP can remove them later if they carry no information).
    corr = X_numeric.corr(method="spearman").abs().fillna(0.0)

    # Higher-variance columns are preferred. For equal variance, prefer
    # *_binary over *_counts so *_counts is dropped first.
    variances = X_numeric.var()
    def _suffix_priority(col: str) -> int:
        if col.endswith("_counts"):
            return 0
        if col.endswith("_binary"):
            return 2
        return 1
    col_order = sorted(
        variances.index.tolist(),
        key=lambda c: (-float(variances[c]), _suffix_priority(c), c),
    )

    dropped: set = set()
    kept_ordered: List[str] = []
    drop_audit: List[Dict[str, object]] = []
    for col in col_order:
        if col in dropped:
            continue
        kept_ordered.append(col)
        # Schedule every column that is too similar to *col* for removal.
        redundant = corr.index[corr[col] > threshold].tolist()
        for partner in redundant:
            if partner != col:
                if partner not in dropped:
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

    if drop_audit:
        print("  Correlation-drop audit (kept -> dropped):")
        drop_audit_sorted = sorted(
            drop_audit,
            key=lambda d: (d["spearman_abs_corr"], d["kept_variance"]),
            reverse=True,
        )
        for item in drop_audit_sorted:
            print(
                "    "
                f"{item['kept_col']} -> {item['dropped_col']} | "
                f"abs_spearman={item['spearman_abs_corr']:.4f} | "
                f"var_kept={item['kept_variance']:.6g} | "
                f"var_dropped={item['dropped_variance']:.6g}"
            )

    # Return names in the original DataFrame column order.
    # Keep all numeric columns that survived correlation filter, plus all non-numeric columns
    kept_set = set(kept_ordered)
    non_numeric_cols = set(X.columns) - set(numeric_cols)
    kept_set.update(non_numeric_cols)
    return [c for c in X.columns if c in kept_set]


def run_training(args: argparse.Namespace) -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("SELECTED ARGS:", args)

    # ------------------------------------------------------------------
    # 1. Load & preprocess
    # ------------------------------------------------------------------
    all_targets = {
        args.sepsis_target,
        args.hemo_target,
        args.cef_target,
        args.bmr_target,
        args.weight_column,
    }
    cols_to_delete = list(DELETE_COLUMNS)
    cols_to_delete.extend([x for x in TARGET_REMOVE if x not in all_targets])

    df = load_processed_dataframe(args.database_file, cols_to_delete)

    missing = [c for c in all_targets if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing from dataframe: {missing}")

    working_df = df.copy()
    working_df[args.weight_column] = pd.to_numeric(
        working_df[args.weight_column], errors="coerce"
    )
    # Drop rows where sepsis label or weight is missing (needed for split stratification)
    working_df = working_df.dropna(subset=[args.sepsis_target, args.weight_column])
    working_df = working_df[working_df[args.weight_column] > 0]

    exclude_cols = all_targets.copy()
    feature_cols = [c for c in working_df.columns if c not in exclude_cols]
    if not feature_cols:
        raise ValueError("No feature columns remain after excluding targets.")

    feature_df_raw = working_df[feature_cols]

    # ------------------------------------------------------------------
    # 2. Train / test split – stratified on sepsis
    # ------------------------------------------------------------------
    y_sepsis = working_df.loc[feature_df_raw.index, args.sepsis_target]
    y_hemo = working_df.loc[feature_df_raw.index, args.hemo_target]
    y_cef = working_df.loc[feature_df_raw.index, args.cef_target]
    y_bmr = working_df.loc[feature_df_raw.index, args.bmr_target]
    w = working_df.loc[feature_df_raw.index, args.weight_column]

    (
        X_train, X_test,
        y_sep_train, y_sep_test,
        y_hemo_train, y_hemo_test,
        y_cef_train, y_cef_test,
        y_bmr_train, y_bmr_test,
        w_train, w_test,
    ) = train_test_split(
        feature_df_raw, y_sepsis, y_hemo, y_cef, y_bmr, w,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_sepsis,
    )

    # ------------------------------------------------------------------
    # 2a. Leakage-safe preprocessing (fit on train, apply on test)
    # ------------------------------------------------------------------
    cat_cols = ["foco", "ultimo_antib"]
    X_train, X_test = preprocess_train_test_features(
        X_train, X_test,
        na_perc_limit=args.na_perc_limit,
        impute_missing=args.impute_missing,
        categorical_for_dummies=cat_cols,
    )
    # Align targets/weights after row filtering caused by no-impute mode
    y_sep_train, y_hemo_train, y_cef_train, y_bmr_train, w_train = (
        y_sep_train.loc[X_train.index],
        y_hemo_train.loc[X_train.index],
        y_cef_train.loc[X_train.index],
        y_bmr_train.loc[X_train.index],
        w_train.loc[X_train.index],
    )
    y_sep_test, y_hemo_test, y_cef_test, y_bmr_test, w_test = (
        y_sep_test.loc[X_test.index],
        y_hemo_test.loc[X_test.index],
        y_cef_test.loc[X_test.index],
        y_bmr_test.loc[X_test.index],
        w_test.loc[X_test.index],
    )
    print(f"Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

    # ------------------------------------------------------------------
    # 2b. Remove highly correlated features (training data only → no leakage)
    # ------------------------------------------------------------------
    if args.max_corr < 1.0:
        print(f"\nRemoving features with |Spearman corr| > {args.max_corr} …")
        kept_cols = remove_correlated_features(X_train, threshold=args.max_corr)
        n_removed = len(X_train.columns) - len(kept_cols)
        if n_removed:
            print(f"  Dropped {n_removed} redundant features → {len(kept_cols)} remain.")
        X_train = X_train[kept_cols]
        X_test = X_test[kept_cols]


    # ------------------------------------------------------------------
    # 3. Output directory
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: Dict = {"args": {str(k): str(v) for k,v in args.__dict__.items()}}
    optuna_storage = args.optuna_storage or None
    study_prefix = args.optuna_study_prefix.strip() if args.optuna_study_prefix else "three_level"
    if optuna_storage:
        print(f"Optuna storage: {optuna_storage}")

    def _study_name(tag: str) -> str:
        return f"{study_prefix}_{tag}_rs{args.random_state}"

    # ==================================================================
    # LEVEL 1 – sepsis
    # ==================================================================
    print("\n" + "=" * 60)
    print("LEVEL 1: sepsis")
    print("=" * 60)
    l1_dir = output_dir / "level1_sepsis"
    l1_dir.mkdir(exist_ok=True)

    le_sep = LabelEncoder()
    y_sep_train_enc = pd.Series(
        le_sep.fit_transform(y_sep_train.astype(str)),
        index=y_sep_train.index,
        name="sepsis_enc",
    )
    y_sep_test_enc = pd.Series(
        le_sep.transform(y_sep_test.astype(str)),
        index=y_sep_test.index,
        name="sepsis_enc",
    )

    sw_l1 = compute_balanced_sample_weight(y_sep_train_enc, w_train)

    rank_model_l1 = _build_ranking_model(args.model_type, y_sep_train_enc, args.random_state)
    if args.skip_rfecv:
        print("  Skipping RFECV for Level 1 – using all features.")
        selected_features = X_train.columns.tolist()
        l1_rfecv_history = {}
    else:
        print("  Selecting Level 1 features by SHAP importance …")
        selected_features, l1_rfecv_history = shap_rfecv(
            rank_model_l1, X_train, y_sep_train_enc, min_features=2, max_features=args.max_features, scoring="pr_auc"
        )
    l1_features = cap_features(
        selected_features, X_train, y_sep_train_enc,
        args.max_features, args.random_state, args.model_type, rank_model=rank_model_l1
    )
    print(f"  SHAP selection kept {len(l1_features)} features.")

    scaler_l1 = MinMaxScaler()
    X_l1_train = pd.DataFrame(
        scaler_l1.fit_transform(X_train[l1_features]),
        columns=l1_features, index=X_train.index,
    )
    X_l1_test = pd.DataFrame(
        scaler_l1.transform(X_test[l1_features]),
        columns=l1_features, index=X_test.index,
    )

    print("  Optimising Level 1 model …")
    l1_params, l1_threshold, l1_study = optimise_binary_model(
        X_l1_train, y_sep_train_enc,
        n_splits=args.cv_splits,
        n_trials=args.binary_trials,
        random_state=args.random_state,
        sample_weight=sw_l1,
        model_type=args.model_type,
        study_name=_study_name("l1"),
        optuna_storage=optuna_storage,
        optuna_load_if_exists=args.optuna_load_if_exists,
    )
    print(f"  Best threshold: {l1_threshold:.3f}")

    # Final Level 1 model trained on all training data, then calibrated
    l1_base = build_binary_model(args.model_type, l1_params.copy())
    l1_model = _fit_calibrated_or_base(l1_base, X_l1_train, y_sep_train_enc, max_cv=5, method="isotonic")

    l1_test_proba = l1_model.predict_proba(X_l1_test)[:, 1]
    l1_test_pred = (l1_test_proba >= l1_threshold).astype(int)

    l1_report = classification_report(
        y_sep_test_enc, l1_test_pred,
        target_names=[str(c) for c in le_sep.classes_],
        zero_division=0,
    )
    l1_auc = roc_auc_score(y_sep_test_enc, l1_test_proba)
    l1_f1 = f1_score(y_sep_test_enc, l1_test_pred, average="macro", zero_division=0)
    print(f"  Level 1 – Macro F1: {l1_f1:.3f}  |  ROC-AUC: {l1_auc:.3f}")

    (l1_dir / "report.txt").write_text(l1_report)
    (l1_dir / "summary.json").write_text(json.dumps({
        "params": l1_params, "threshold": l1_threshold,
        "macro_f1": l1_f1, "roc_auc": l1_auc,
        "classes": le_sep.classes_.tolist(),
        "shaprfecv_features": selected_features,
        "features": l1_features,
    }, indent=2))
    l1_study.trials_dataframe().to_csv(l1_dir / "optuna_trials.csv", index=False)
    pd.DataFrame({
        "true": y_sep_test_enc.values,
        "pred": l1_test_pred,
        "proba": l1_test_proba,
    }, index=X_l1_test.index).to_csv(l1_dir / "predictions.csv", index_label="row_index")

    confusion_matrix(y_sep_test_enc, l1_test_pred, labels=[0, 1])
    all_summaries["level1_sepsis"] = {"macro_f1": l1_f1, "roc_auc": l1_auc}

    # ==================================================================
    # LEVEL 2 – two-stage binary hemo model
    # ==================================================================
    print("\n" + "=" * 60)
    print(
        "LEVEL 2: "
        + ("direct hemo etiology" if args.skip_l2_gate else "two-stage hemo (positive gate -> subtype)")
    )
    print("=" * 60)
    l2_dir = output_dir / "level2_hemo"
    l2_dir.mkdir(exist_ok=True)

    hemo_valid_train = y_hemo_train.notna()
    hemo_valid_test = y_hemo_test.notna()
    if not hemo_valid_train.any():
        raise ValueError("No non-null hemo labels in train.")

    if args.skip_l2_gate:
        # Direct etiology prediction including NEGATIVE as a class.
        X2_train_base = X_train.loc[hemo_valid_train].copy()
        X2_test_base = X_test.loc[hemo_valid_test].copy()
        y2_train_raw = y_hemo_train.loc[hemo_valid_train].astype(str)
        y2_test_raw = y_hemo_test.loc[hemo_valid_test].astype(str)

        le_l2 = LabelEncoder()
        y2_train = pd.Series(
            le_l2.fit_transform(y2_train_raw),
            index=y2_train_raw.index,
            name="hemo_enc",
        )

        test_mask = y2_test_raw.isin(le_l2.classes_)
        X2_test_base = X2_test_base.loc[test_mask].copy()
        y2_test_raw = y2_test_raw.loc[test_mask]
        y2_test = pd.Series(
            le_l2.transform(y2_test_raw),
            index=y2_test_raw.index,
            name="hemo_enc",
        )

        if X2_test_base.empty:
            raise ValueError(
                "No test rows remain for Level 2 direct etiology after filtering to labels seen in training."
            )

        l2_target_names = [str(c) for c in le_l2.classes_]
        is_binary_subtype = len(le_l2.classes_) == 2

        print(f"  Direct hemo etiology classes: {l2_target_names}")
        print(f"  Train rows: {len(X2_train_base)}  |  test rows: {len(X2_test_base)}")

        rank_model_l2 = _build_ranking_model(args.model_type, y2_train, args.random_state)
        if args.skip_rfecv:
            print("  Skipping RFECV for Level 2 direct etiology – using all features.")
            l2_shap_feats = X2_train_base.columns.tolist()
            l2_rfecv_history = {}
        else:
            print("  Selecting Level 2 direct etiology features by SHAP importance …")
            l2_shap_feats, l2_rfecv_history = shap_rfecv(
                rank_model_l2, X2_train_base, y2_train, min_features=2, max_features=args.max_features, scoring="pr_auc"
            )
        l2_features = cap_features(
            l2_shap_feats, X2_train_base, y2_train,
            args.max_features, args.random_state, args.model_type, rank_model=rank_model_l2
        )
        print(f"  Direct Level 2 SHAP selection kept {len(l2_features)} features.")

        scaler_l2 = MinMaxScaler()
        X2_train_scaled = pd.DataFrame(
            scaler_l2.fit_transform(X2_train_base[l2_features]),
            columns=l2_features, index=X2_train_base.index,
        )
        X2_test_scaled = pd.DataFrame(
            scaler_l2.transform(X2_test_base[l2_features]),
            columns=l2_features, index=X2_test_base.index,
        )

        sw_l2 = compute_balanced_sample_weight(y2_train, w_train.reindex(y2_train.index))
        print("  Optimising Level 2 direct etiology model …")
        if is_binary_subtype:
            l2_params, l2_threshold, l2_study = optimise_binary_model(
                X2_train_scaled, y2_train,
                n_splits=args.cv_splits,
                n_trials=args.binary_trials,
                random_state=args.random_state,
                sample_weight=sw_l2,
                model_type=args.model_type,
                study_name=_study_name("l2_direct_bin"),
                optuna_storage=optuna_storage,
                optuna_load_if_exists=args.optuna_load_if_exists,
            )
            l2_base = build_binary_model(args.model_type, l2_params.copy())
        else:
            l2_params, l2_study = optimise_multiclass_model(
                X2_train_scaled, y2_train,
                n_splits=args.cv_splits,
                n_trials=args.binary_trials,
                random_state=args.random_state,
                sample_weight=sw_l2,
                model_type=args.model_type,
                num_classes=len(le_l2.classes_),
                study_name=_study_name("l2_direct_multi"),
                optuna_storage=optuna_storage,
                optuna_load_if_exists=args.optuna_load_if_exists,
            )
            l2_threshold = None
            l2_base = build_multiclass_model(args.model_type, l2_params.copy(), len(le_l2.classes_))

        l2_model = _fit_calibrated_or_base(l2_base, X2_train_scaled, y2_train, max_cv=5, method="isotonic")
        l2_test_proba = l2_model.predict_proba(X2_test_scaled)
        if is_binary_subtype:
            l2_test_pred = (l2_test_proba[:, 1] >= l2_threshold).astype(int)
        else:
            l2_test_pred = np.argmax(l2_test_proba, axis=1)

        l2_f1 = f1_score(y2_test, l2_test_pred, average="macro", zero_division=0)
        l2_auc: Optional[float] = None
        if y2_test.nunique() > 1:
            if is_binary_subtype:
                l2_auc = float(roc_auc_score(y2_test, l2_test_proba[:, 1]))
            else:
                l2_auc = float(roc_auc_score(y2_test, l2_test_proba, multi_class="ovr", average="macro"))

        l2_report = classification_report(
            y2_test, l2_test_pred,
            target_names=l2_target_names,
            zero_division=0,
        )
        print(
            f"  Level 2 direct etiology – Macro F1: {l2_f1:.3f}"
            + (f"  |  ROC-AUC: {l2_auc:.3f}" if l2_auc is not None else "")
        )

        (l2_dir / "report.txt").write_text(l2_report)
        stage2_summary = {
            "mode": "binary" if is_binary_subtype else "multiclass",
            "classes": l2_target_names,
            "params": l2_params,
            "threshold": l2_threshold,
            "macro_f1": l2_f1,
            "roc_auc": l2_auc,
            "shaprfecv_features": l2_shap_feats,
            "features": l2_features,
        }
        if is_binary_subtype:
            stage2_summary["negative_label"] = l2_target_names[0]
            stage2_summary["positive_label"] = l2_target_names[1]

        (l2_dir / "summary.json").write_text(json.dumps({
            "mode": "direct_binary" if is_binary_subtype else "direct_multiclass",
            "params": l2_params,
            "threshold": l2_threshold,
            "macro_f1": l2_f1,
            "roc_auc": l2_auc,
            "classes": l2_target_names,
            "shaprfecv_features": l2_shap_feats,
            "features": l2_features,
        }, indent=2))
        l2_study.trials_dataframe().to_csv(l2_dir / "optuna_trials_direct_etiology.csv", index=False)

        stage_pred_df = pd.DataFrame(index=X2_test_scaled.index)
        stage_pred_df["true"] = y2_test.values
        stage_pred_df["true_label"] = [le_l2.classes_[x] for x in y2_test.values]
        stage_pred_df["pred"] = l2_test_pred
        stage_pred_df["pred_label"] = [le_l2.classes_[x] for x in l2_test_pred]
        if is_binary_subtype:
            stage_pred_df["proba_positive"] = l2_test_proba[:, 1]
        else:
            for idx, class_name in enumerate(le_l2.classes_):
                stage_pred_df[f"proba_{class_name}"] = l2_test_proba[:, idx]
        stage_pred_df.to_csv(l2_dir / "predictions_direct_etiology.csv", index_label="row_index")

        all_summaries["level2_hemo"] = {
            "mode": "direct_binary" if is_binary_subtype else "direct_multiclass",
            "classes": l2_target_names,
            "threshold": l2_threshold,
            "macro_f1": l2_f1,
            "roc_auc": l2_auc,
        }
        all_summaries["l2_gate_rfecv_scores"] = l2_rfecv_history

    else:
        # Stage 1: binary gate (NEGATIVE vs positive)
        y2_gate_train = (y_hemo_train.loc[hemo_valid_train].astype(str) != args.hemo_negative_label).astype(int)
        gate_test_mask = hemo_valid_test
        y2_gate_test = (y_hemo_test.loc[gate_test_mask].astype(str) != args.hemo_negative_label).astype(int)

        X2_gate_train_base = X_train.loc[hemo_valid_train].copy()
        X2_gate_test_base = X_test.loc[gate_test_mask].copy()

        rank_model_l2_gate = _build_ranking_model(args.model_type, y2_gate_train, args.random_state)
        if args.skip_rfecv:
            print("  Skipping RFECV for Level 2 Stage-1 – using all features.")
            l2_gate_shap_feats = X2_gate_train_base.columns.tolist()
            l2_gate_rfecv_history = {}
        else:
            print("  Selecting Level 2 Stage-1 features by SHAP importance …")
            l2_gate_shap_feats, l2_gate_rfecv_history = shap_rfecv(
                rank_model_l2_gate, X2_gate_train_base, y2_gate_train, min_features=2, max_features=args.max_features, scoring="pr_auc"
            )
        l2_gate_features = cap_features(
            l2_gate_shap_feats, X2_gate_train_base, y2_gate_train,
            args.max_features, args.random_state, args.model_type, rank_model=rank_model_l2_gate
        )

        scaler_l2_gate = MinMaxScaler()
        X2_gate_train_scaled = pd.DataFrame(
            scaler_l2_gate.fit_transform(X2_gate_train_base[l2_gate_features]),
            columns=l2_gate_features, index=X2_gate_train_base.index,
        )
        X2_gate_test_scaled = pd.DataFrame(
            scaler_l2_gate.transform(X2_gate_test_base[l2_gate_features]),
            columns=l2_gate_features, index=X2_gate_test_base.index,
        )

        sw_l2_gate = compute_balanced_sample_weight(y2_gate_train, w_train.reindex(y2_gate_train.index))
        print("  Optimising Level 2 Stage-1 binary gate model …")
        l2_gate_params, l2_gate_threshold, l2_gate_study = optimise_binary_model(
            X2_gate_train_scaled, y2_gate_train,
            n_splits=args.cv_splits,
            n_trials=args.binary_trials,
            random_state=args.random_state,
            sample_weight=sw_l2_gate,
            model_type=args.model_type,
            study_name=_study_name("l2_gate"),
            optuna_storage=optuna_storage,
            optuna_load_if_exists=args.optuna_load_if_exists,
        )
        l2_gate_base = build_binary_model(args.model_type, l2_gate_params.copy())
        l2_gate_model = _fit_calibrated_or_base(
            l2_gate_base, X2_gate_train_scaled, y2_gate_train, max_cv=5, method="isotonic"
        )
        l2_gate_test_proba = l2_gate_model.predict_proba(X2_gate_test_scaled)[:, 1]
        l2_gate_test_pred = (l2_gate_test_proba >= l2_gate_threshold).astype(int)
        l2_gate_f1 = f1_score(y2_gate_test, l2_gate_test_pred, average="macro", zero_division=0)
        l2_gate_auc: Optional[float] = (
            float(roc_auc_score(y2_gate_test, l2_gate_test_proba))
            if y2_gate_test.nunique() > 1 else None
        )

        # Stage 2: subtype among positives only
        subtype_train_mask = hemo_valid_train & (y_hemo_train.astype(str) != args.hemo_negative_label)
        subtype_test_mask = hemo_valid_test & (y_hemo_test.astype(str) != args.hemo_negative_label)

        X2_sub_train_base = X_train.loc[subtype_train_mask].copy()
        X2_sub_test_base = X_test.loc[subtype_test_mask].copy()
        y2_sub_train_raw = y_hemo_train.loc[subtype_train_mask].astype(str)
        y2_sub_test_raw = y_hemo_test.loc[subtype_test_mask].astype(str)

        if y2_sub_train_raw.empty:
            raise ValueError(
                "No train rows for Level 2 Stage-2 subtype after excluding NEGATIVE. "
                "Check that the hemo target column contains positive labels."
            )

        le_l2 = LabelEncoder()
        y2_sub_train = pd.Series(
            le_l2.fit_transform(y2_sub_train_raw),
            index=y2_sub_train_raw.index,
            name="hemo_sub_enc",
        )

        subtype_test_mask = subtype_test_mask & y2_sub_test_raw.isin(le_l2.classes_)
        X2_sub_test_base = X_test.loc[subtype_test_mask].copy()
        y2_sub_test_raw = y_hemo_test.loc[subtype_test_mask].astype(str)
        y2_sub_test = pd.Series(
            le_l2.transform(y2_sub_test_raw),
            index=y2_sub_test_raw.index,
            name="hemo_sub_enc",
        )

        if X2_sub_test_base.empty:
            raise ValueError(
                "No test rows for Level 2 Stage-2 subtype after filtering to labels seen in training. "
                "Verify the hemo target values and the train/test split."
            )

        l2_target_names = [str(c) for c in le_l2.classes_]
        is_binary_subtype = len(le_l2.classes_) == 2

        print(f"  Positive hemo classes: {l2_target_names}")
        print(f"  Stage-2 train rows: {len(X2_sub_train_base)}  |  test rows: {len(X2_sub_test_base)}")

        rank_model_l2_sub = _build_ranking_model(args.model_type, y2_sub_train, args.random_state)
        if args.skip_rfecv:
            print("  Skipping RFECV for Level 2 Stage-2 – using all features.")
            l2_shap_feats = X2_sub_train_base.columns.tolist()
            l2_rfecv_history = {}
        else:
            print("  Selecting Level 2 Stage-2 features by SHAP importance …")
            l2_shap_feats, l2_rfecv_history = shap_rfecv(
                rank_model_l2_sub, X2_sub_train_base, y2_sub_train, min_features=2, max_features=args.max_features, scoring="pr_auc"
            )
        l2_features = cap_features(
            l2_shap_feats, X2_sub_train_base, y2_sub_train,
            args.max_features, args.random_state, args.model_type, rank_model=rank_model_l2_sub
        )
        print(f"  Stage-2 SHAP selection kept {len(l2_features)} features.")

        scaler_l2 = MinMaxScaler()
        X2_train_scaled = pd.DataFrame(
            scaler_l2.fit_transform(X2_sub_train_base[l2_features]),
            columns=l2_features, index=X2_sub_train_base.index,
        )
        X2_test_scaled = pd.DataFrame(
            scaler_l2.transform(X2_sub_test_base[l2_features]),
            columns=l2_features, index=X2_sub_test_base.index,
        )

        sw_l2 = compute_balanced_sample_weight(y2_sub_train, w_train.reindex(y2_sub_train.index))
        print("  Optimising Level 2 Stage-2 subtype model …")
        if is_binary_subtype:
            l2_params, l2_threshold, l2_study = optimise_binary_model(
                X2_train_scaled, y2_sub_train,
                n_splits=args.cv_splits,
                n_trials=args.binary_trials,
                random_state=args.random_state,
                sample_weight=sw_l2,
                model_type=args.model_type,
                study_name=_study_name("l2_sub_bin"),
                optuna_storage=optuna_storage,
                optuna_load_if_exists=args.optuna_load_if_exists,
            )
            l2_base = build_binary_model(args.model_type, l2_params.copy())
        else:
            l2_params, l2_study = optimise_multiclass_model(
                X2_train_scaled, y2_sub_train,
                n_splits=args.cv_splits,
                n_trials=args.binary_trials,
                random_state=args.random_state,
                sample_weight=sw_l2,
                model_type=args.model_type,
                num_classes=len(le_l2.classes_),
                study_name=_study_name("l2_sub_multi"),
                optuna_storage=optuna_storage,
                optuna_load_if_exists=args.optuna_load_if_exists,
            )
            l2_threshold = None
            l2_base = build_multiclass_model(args.model_type, l2_params.copy(), len(le_l2.classes_))

        l2_model = _fit_calibrated_or_base(l2_base, X2_train_scaled, y2_sub_train, max_cv=5, method="isotonic")
        l2_test_proba = l2_model.predict_proba(X2_test_scaled)
        if is_binary_subtype:
            l2_test_pred = (l2_test_proba[:, 1] >= l2_threshold).astype(int)
        else:
            l2_test_pred = np.argmax(l2_test_proba, axis=1)

        l2_f1 = f1_score(y2_sub_test, l2_test_pred, average="macro", zero_division=0)
        l2_auc: Optional[float] = None
        if y2_sub_test.nunique() > 1:
            if is_binary_subtype:
                l2_auc = float(roc_auc_score(y2_sub_test, l2_test_proba[:, 1]))
            else:
                l2_auc = float(roc_auc_score(y2_sub_test, l2_test_proba, multi_class="ovr", average="macro"))

        l2_report = classification_report(
            y2_sub_test, l2_test_pred,
            target_names=l2_target_names,
            zero_division=0,
        )
        print(
            f"  Level 2 Stage-1 gate – Macro F1: {l2_gate_f1:.3f}"
            + (f"  |  ROC-AUC: {l2_gate_auc:.3f}" if l2_gate_auc is not None else "")
        )
        print(
            f"  Level 2 Stage-2 subtype – Macro F1: {l2_f1:.3f}"
            + (f"  |  ROC-AUC: {l2_auc:.3f}" if l2_auc is not None else "")
        )

        (l2_dir / "report.txt").write_text(l2_report)
        stage2_summary = {
            "mode": "binary" if is_binary_subtype else "multiclass",
            "classes": l2_target_names,
            "params": l2_params,
            "threshold": l2_threshold,
            "macro_f1": l2_f1,
            "roc_auc": l2_auc,
            "shaprfecv_features": l2_shap_feats,
            "features": l2_features,
        }
        if is_binary_subtype:
            stage2_summary["negative_label"] = l2_target_names[0]
            stage2_summary["positive_label"] = l2_target_names[1]

        (l2_dir / "summary.json").write_text(json.dumps({
            "mode": "two_stage_multiclass" if not is_binary_subtype else "two_stage_binary",
            "stage1_gate": {
                "negative_label": args.hemo_negative_label,
                "params": l2_gate_params,
                "threshold": l2_gate_threshold,
                "macro_f1": l2_gate_f1,
                "roc_auc": l2_gate_auc,
                "shaprfecv_features": l2_gate_shap_feats,
                "features": l2_gate_features,
            },
            "stage2_subtype": stage2_summary,
        }, indent=2))
        l2_gate_study.trials_dataframe().to_csv(l2_dir / "optuna_trials_stage1_gate.csv", index=False)
        l2_study.trials_dataframe().to_csv(l2_dir / "optuna_trials_stage2_subtype.csv", index=False)

        stage2_pred_df = pd.DataFrame(index=X2_test_scaled.index)
        stage2_pred_df["true"] = y2_sub_test.values
        stage2_pred_df["true_label"] = [le_l2.classes_[x] for x in y2_sub_test.values]
        stage2_pred_df["pred"] = l2_test_pred
        stage2_pred_df["pred_label"] = [le_l2.classes_[x] for x in l2_test_pred]
        if is_binary_subtype:
            stage2_pred_df["proba_positive"] = l2_test_proba[:, 1]
        else:
            for idx, class_name in enumerate(le_l2.classes_):
                stage2_pred_df[f"proba_{class_name}"] = l2_test_proba[:, idx]

        pd.DataFrame({
            "true": y2_gate_test.values,
            "pred": l2_gate_test_pred,
            "proba": l2_gate_test_proba,
        }, index=X2_gate_test_scaled.index).to_csv(l2_dir / "predictions_stage1_gate.csv", index_label="row_index")
        stage2_pred_df.to_csv(l2_dir / "predictions_stage2_subtype.csv", index_label="row_index")

        all_summaries["level2_hemo"] = {
            "mode": "two_stage_binary" if is_binary_subtype else "two_stage_multiclass_positive_gate",
            "stage1_gate": {
                "macro_f1": l2_gate_f1,
                "roc_auc": l2_gate_auc,
                "negative_label": args.hemo_negative_label,
                "threshold": l2_gate_threshold,
            },
            "stage2_subtype": {
                "macro_f1": l2_f1,
                "roc_auc": l2_auc,
                "classes": l2_target_names,
                "threshold": l2_threshold,
            },
        }
        all_summaries["l2_gate_rfecv_scores"] = l2_gate_rfecv_history
    # ==================================================================
    # LEVEL 3 – resistente_cefalosporina
    # ==================================================================
    print("\n" + "=" * 60)
    print("LEVEL 3: resistente_cefalosporina")
    print("=" * 60)
    l3_dir = output_dir / "level3_cefalosporina"
    l3_dir.mkdir(exist_ok=True)

    # Level 3 is normally scoped to positive blood culture rows.
    hemo_pos_train_mask = y_hemo_train.notna() & (y_hemo_train != "NEGATIVE")
    hemo_pos_test_mask = y_hemo_test.notna() & (y_hemo_test != "NEGATIVE")
    if args.skip_l3_hemo_filter:
        print("  Level 3: skipping resultado_hemo != NEGATIVE filter before BMR gating.")
        hemo_pos_train_mask = y_hemo_train.notna()
        hemo_pos_test_mask = y_hemo_test.notna()
    else:
        print("  Level 3: filtering to resultado_hemo != NEGATIVE before BMR gating.")

    bmr_valid_train_mask = y_bmr_train.notna() & hemo_pos_train_mask
    bmr_valid_test_mask = y_bmr_test.notna() & hemo_pos_test_mask
    cef_valid_train_mask = y_cef_train.notna() & hemo_pos_train_mask
    cef_valid_test_mask = y_cef_test.notna() & hemo_pos_test_mask

    if not cef_valid_train_mask.any():
        raise ValueError(
            "No training rows with positive blood culture and valid Level-3 label. "
            "Check that 'resultado_hemo_grouped', 'bmr_etiologia', and the chosen --cef-target are present."
        )

    use_cef_multi_two_stage = (args.cef_target == "resistente_cefalosporina_multi")

    if use_cef_multi_two_stage:
        print("  Level 3 target is resistente_cefalosporina_multi: using two-stage clinical gate.")
        print("  Stage 1: BMR gate (NEGATIVE vs non-NEGATIVE) on bmr_etiologia.")
        print("  Stage 2: among BMR-positive, cefalosporina vs other.")

        gate_train_mask = bmr_valid_train_mask
        gate_test_mask = bmr_valid_test_mask
        X3_gate_train_base = X_train.loc[gate_train_mask].copy()
        X3_gate_test_base = X_test.loc[gate_test_mask].copy()
        y3_gate_train = (y_bmr_train.loc[gate_train_mask].astype(str) != args.bmr_negative_label).astype(int)
        y3_gate_test = (y_bmr_test.loc[gate_test_mask].astype(str) != args.bmr_negative_label).astype(int)

        gate_minority = int(pd.Series(y3_gate_train).value_counts().min())
        gate_cv_splits = max(min(args.cv_splits, gate_minority), 2)
        if gate_cv_splits != args.cv_splits:
            print(f"  Reducing cv_splits from {args.cv_splits} to {gate_cv_splits} for L3 stage-1 gate.")

        rank_model_l3_gate = _build_ranking_model(args.model_type, pd.Series(y3_gate_train), args.random_state)
        if args.skip_rfecv:
            print("  Skipping RFECV for L3 stage-1 gate – using all features.")
            l3_gate_feats = X3_gate_train_base.columns.tolist()
            l3_gate_rfecv_history = {}
        else:
            print("  Selecting L3 stage-1 gate features by SHAP importance …")
            l3_gate_feats, l3_gate_rfecv_history = shap_rfecv(
                rank_model_l3_gate, X3_gate_train_base, y3_gate_train, min_features=2, max_features=args.max_features, scoring="pr_auc"
            )
        l3_gate_features = cap_features(
            l3_gate_feats, X3_gate_train_base, pd.Series(y3_gate_train),
            args.max_features, args.random_state, args.model_type, rank_model=rank_model_l3_gate
        )

        scaler_l3_gate = MinMaxScaler()
        X3_gate_train_scaled = pd.DataFrame(
            scaler_l3_gate.fit_transform(X3_gate_train_base[l3_gate_features]),
            columns=l3_gate_features, index=X3_gate_train_base.index,
        )
        X3_gate_test_scaled = pd.DataFrame(
            scaler_l3_gate.transform(X3_gate_test_base[l3_gate_features]),
            columns=l3_gate_features, index=X3_gate_test_base.index,
        )
        sw_l3_gate = compute_balanced_sample_weight(
            pd.Series(y3_gate_train, index=X3_gate_train_base.index),
            w_train.reindex(X3_gate_train_base.index),
        )
        l3_gate_params, l3_gate_threshold, l3_gate_study = optimise_binary_model(
            X3_gate_train_scaled, pd.Series(y3_gate_train, index=X3_gate_train_base.index),
            n_splits=gate_cv_splits,
            n_trials=args.binary_trials,
            random_state=args.random_state,
            sample_weight=sw_l3_gate,
            model_type=args.model_type,
            study_name=_study_name("l3_gate_bmr"),
            optuna_storage=optuna_storage,
            optuna_load_if_exists=args.optuna_load_if_exists,
        )
        l3_gate_base = build_binary_model(args.model_type, l3_gate_params.copy())
        l3_gate_model = _fit_calibrated_or_base(
            l3_gate_base, X3_gate_train_scaled, pd.Series(y3_gate_train), max_cv=5, method="isotonic"
        )
        l3_gate_test_proba = l3_gate_model.predict_proba(X3_gate_test_scaled)[:, 1]
        l3_gate_test_pred = (l3_gate_test_proba >= l3_gate_threshold).astype(int)

        stage2_train_mask = gate_train_mask & (y_bmr_train.astype(str) != args.bmr_negative_label) & y_cef_train.notna()
        stage2_test_mask = gate_test_mask & (y_bmr_test.astype(str) != args.bmr_negative_label) & y_cef_test.notna()
        X3_train_base = X_train.loc[stage2_train_mask].copy()
        X3_test_base = X_test.loc[stage2_test_mask].copy()
        y3_train_raw = y_cef_train.loc[stage2_train_mask].astype(str)
        y3_test_raw = y_cef_test.loc[stage2_test_mask].astype(str)

        y3_train_stage2 = y3_train_raw.str.lower().str.contains(args.cef_multi_positive_label.lower(), regex=False).astype(int)
        y3_test_stage2 = y3_test_raw.str.lower().str.contains(args.cef_multi_positive_label.lower(), regex=False).astype(int)
        if y3_train_stage2.nunique() < 2:
            raise ValueError(
                "Stage-2 Level-3 target has <2 classes. "
                "Adjust --cef-multi-positive-label or verify resistente_cefalosporina_multi labels."
            )

        y3_train_enc = y3_train_stage2
        y3_test_enc = y3_test_stage2
        is_l3_binary = True
        l3_target_names = ["other", args.cef_multi_positive_label]

        print(
            f"  L3 stage-2 positive token: '{args.cef_multi_positive_label}' | "
            f"train rows: {len(X3_train_base)} | test rows: {len(X3_test_base)}"
        )

        l3_minority_count = int(y3_train_stage2.value_counts().min())
        l3_small_dataset = l3_minority_count < args.l3_min_positive
        l3_cv_splits = max(min(args.cv_splits, l3_minority_count), 2)
        if l3_cv_splits != args.cv_splits:
            print(f"  Reducing cv_splits from {args.cv_splits} to {l3_cv_splits} for L3 stage-2.")

        rank_model_l3 = _build_ranking_model(args.model_type, y3_train_stage2, args.random_state)
        if args.skip_rfecv:
            print("  Skipping RFECV for Level 3 stage-2 – using all features.")
            l3_shap_feats = X3_train_base.columns.tolist()
            l3_rfecv_history = {}
        else:
            print("  Selecting Level 3 stage-2 features by SHAP importance …")
            l3_shap_feats, l3_rfecv_history = shap_rfecv(
                rank_model_l3, X3_train_base, y3_train_stage2, min_features=2, max_features=args.max_features, scoring="pr_auc"
            )
        l3_features = cap_features(
            l3_shap_feats, X3_train_base, y3_train_stage2,
            args.max_features, args.random_state, args.model_type, rank_model=rank_model_l3
        )

        scaler_l3 = MinMaxScaler()
        X3_train_scaled = pd.DataFrame(
            scaler_l3.fit_transform(X3_train_base[l3_features]),
            columns=l3_features, index=X3_train_base.index,
        )
        X3_test_scaled = pd.DataFrame(
            scaler_l3.transform(X3_test_base[l3_features]),
            columns=l3_features, index=X3_test_base.index,
        )
        sw_l3 = compute_balanced_sample_weight(y3_train_stage2, w_train.reindex(y3_train_stage2.index))
        l3_params, l3_threshold, l3_study = optimise_binary_model(
            X3_train_scaled, y3_train_stage2,
            n_splits=l3_cv_splits,
            n_trials=args.binary_trials,
            random_state=args.random_state,
            sample_weight=sw_l3,
            model_type=args.model_type,
            study_name=_study_name("l3_stage2_bin_from_multi"),
            optuna_storage=optuna_storage,
            optuna_load_if_exists=args.optuna_load_if_exists,
        )
        l3_base = build_binary_model(args.model_type, l3_params.copy())
        l3_model = _fit_calibrated_or_base(l3_base, X3_train_scaled, y3_train_stage2, max_cv=5, method="isotonic")
        l3_test_proba = l3_model.predict_proba(X3_test_scaled)[:, 1]
        l3_test_pred = (l3_test_proba >= l3_threshold).astype(int)

        l3_f1 = f1_score(y3_test_stage2, l3_test_pred, average="macro", zero_division=0)
        l3_auc: Optional[float] = None
        if y3_test_stage2.nunique() > 1:
            l3_auc = float(roc_auc_score(y3_test_stage2, l3_test_proba))

        (l3_dir / "stage1_bmr_gate_summary.json").write_text(json.dumps({
            "params": l3_gate_params,
            "threshold": l3_gate_threshold,
            "macro_f1": float(f1_score(y3_gate_test, l3_gate_test_pred, average="macro", zero_division=0)),
            "roc_auc": float(roc_auc_score(y3_gate_test, l3_gate_test_proba)) if pd.Series(y3_gate_test).nunique() > 1 else None,
            "features": l3_gate_features,
            "shaprfecv_features": l3_gate_feats,
        }, indent=2))
        l3_gate_study.trials_dataframe().to_csv(l3_dir / "stage1_bmr_gate_optuna_trials.csv", index=False)
        pd.DataFrame({
            "true_bmr": y3_gate_test,
            "pred_bmr": l3_gate_test_pred,
            "proba_bmr_positive": l3_gate_test_proba,
        }, index=X3_gate_test_scaled.index).to_csv(l3_dir / "stage1_bmr_gate_predictions.csv", index_label="row_index")

        (l3_dir / "summary.json").write_text(json.dumps({
            "mode": "two_stage_from_resistente_cefalosporina_multi",
            "stage1_gate": "BMR (NEGATIVE vs non-NEGATIVE)",
            "stage2_target": f"{args.cef_multi_positive_label} vs other (within BMR-positive)",
            "params": l3_params,
            "threshold": l3_threshold,
            "macro_f1": l3_f1,
            "roc_auc": l3_auc,
            "features": l3_features,
            "shaprfecv_features": l3_shap_feats,
            "hemo_positive_gate": True,
        }, indent=2))
        l3_study.trials_dataframe().to_csv(l3_dir / "optuna_trials.csv", index=False)
        pd.DataFrame({
            "true_stage2": y3_test_stage2.values,
            "pred_stage2": l3_test_pred,
            "proba_stage2_positive": l3_test_proba,
            "raw_label": y3_test_raw.values,
        }, index=X3_test_scaled.index).to_csv(l3_dir / "predictions.csv", index_label="row_index")

        all_summaries["level3_cefalosporina"] = {
            "mode": "two_stage_from_resistente_cefalosporina_multi",
            "stage1_gate_macro_f1": float(f1_score(y3_gate_test, l3_gate_test_pred, average="macro", zero_division=0)),
            "stage1_gate_roc_auc": float(roc_auc_score(y3_gate_test, l3_gate_test_proba)) if pd.Series(y3_gate_test).nunique() > 1 else None,
            "stage2_macro_f1": l3_f1,
            "stage2_roc_auc": l3_auc,
            "macro_f1": l3_f1,
            "roc_auc": l3_auc,
            "stage2_positive_label_token": args.cef_multi_positive_label,
        }
        all_summaries["l3_gate_rfecv_scores"] = l3_gate_rfecv_history
        all_summaries["l3_rfecv_scores"] = l3_rfecv_history
    else:
        gate_train_mask = cef_valid_train_mask
        gate_test_mask = cef_valid_test_mask
        X3_gate_train_base = X_train.loc[gate_train_mask].copy()
        X3_gate_test_base = X_test.loc[gate_test_mask].copy()
        cef_negative_tokens = {"negative", "[]"}
        y_cef_gate_train_norm = y_cef_train.loc[gate_train_mask].astype(str).str.strip().str.lower()
        y_cef_gate_test_norm = y_cef_test.loc[gate_test_mask].astype(str).str.strip().str.lower()
        y3_gate_train = (~y_cef_gate_train_norm.isin(cef_negative_tokens)).astype(int)
        y3_gate_test = (~y_cef_gate_test_norm.isin(cef_negative_tokens)).astype(int)

        gate_minority = int(pd.Series(y3_gate_train).value_counts().min())
        gate_cv_splits = max(min(args.cv_splits, gate_minority), 2)
        if gate_cv_splits != args.cv_splits:
            print(f"  Reducing cv_splits from {args.cv_splits} to {gate_cv_splits} for L3 stage-1 gate.")

        rank_model_l3_gate = _build_ranking_model(args.model_type, pd.Series(y3_gate_train), args.random_state)
        if args.skip_rfecv:
            print("  Skipping L3 stage-1 gate RFECV – using all features.")
            l3_gate_feats = X3_gate_train_base.columns.tolist()
            l3_gate_rfecv_history = {}
        else:
            print("  Selecting L3 stage-1 gate features by SHAP importance …")
            l3_gate_feats, l3_gate_rfecv_history = shap_rfecv(
                rank_model_l3_gate, X3_gate_train_base, y3_gate_train, min_features=2, max_features=args.max_features, scoring="pr_auc"
            )
        l3_gate_features = cap_features(
            l3_gate_feats, X3_gate_train_base, pd.Series(y3_gate_train),
            args.max_features, args.random_state, args.model_type, rank_model=rank_model_l3_gate
        )

        scaler_l3_gate = MinMaxScaler()
        X3_gate_train_scaled = pd.DataFrame(
            scaler_l3_gate.fit_transform(X3_gate_train_base[l3_gate_features]),
            columns=l3_gate_features, index=X3_gate_train_base.index,
        )
        X3_gate_test_scaled = pd.DataFrame(
            scaler_l3_gate.transform(X3_gate_test_base[l3_gate_features]),
            columns=l3_gate_features, index=X3_gate_test_base.index,
        )
        sw_l3_gate = compute_balanced_sample_weight(
            pd.Series(y3_gate_train, index=X3_gate_train_base.index),
            w_train.reindex(X3_gate_train_base.index),
        )
        l3_gate_params, l3_gate_threshold, l3_gate_study = optimise_binary_model(
            X3_gate_train_scaled, pd.Series(y3_gate_train, index=X3_gate_train_base.index),
            n_splits=gate_cv_splits,
            n_trials=args.binary_trials,
            random_state=args.random_state,
            sample_weight=sw_l3_gate,
            model_type=args.model_type,
            study_name=_study_name("l3_gate_cef"),
            optuna_storage=optuna_storage,
            optuna_load_if_exists=args.optuna_load_if_exists,
        )
        l3_gate_base = build_binary_model(args.model_type, l3_gate_params.copy())
        l3_gate_model = _fit_calibrated_or_base(
            l3_gate_base, X3_gate_train_scaled, pd.Series(y3_gate_train), max_cv=5, method="isotonic"
        )
        l3_gate_test_proba = l3_gate_model.predict_proba(X3_gate_test_scaled)[:, 1]
        l3_gate_test_pred = (l3_gate_test_proba >= l3_gate_threshold).astype(int)

        y_cef_train_norm = y_cef_train.astype(str).str.strip().str.lower()
        y_cef_test_norm = y_cef_test.astype(str).str.strip().str.lower()
        stage2_train_mask = gate_train_mask & (~y_cef_train_norm.isin(cef_negative_tokens))
        stage2_test_mask = gate_test_mask & (~y_cef_test_norm.isin(cef_negative_tokens))
        X3_train_base = X_train.loc[stage2_train_mask].copy()
        X3_test_base = X_test.loc[stage2_test_mask].copy()
        y3_train_raw = y_cef_train.loc[stage2_train_mask].astype(str)
        y3_test_raw = y_cef_test.loc[stage2_test_mask].astype(str)

        if y3_train_raw.empty:
            raise ValueError(
                "Stage-2 Level 3 has no BMR-positive train rows after NEGATIVE exclusion. "
                "Check that --cef-target contains raw BMR type labels."
            )

        le_cef = LabelEncoder()
        y3_train_enc = pd.Series(
            le_cef.fit_transform(y3_train_raw), index=y3_train_raw.index, name="cef_enc"
        )
        stage2_test_mask = stage2_test_mask & y3_test_raw.isin(le_cef.classes_)
        X3_test_base = X_test.loc[stage2_test_mask].copy()
        y3_test_raw = y_cef_test.loc[stage2_test_mask].astype(str)
        y3_test_enc = pd.Series(
            le_cef.transform(y3_test_raw), index=y3_test_raw.index, name="cef_enc"
        )

        l3_target_names = [str(c) for c in le_cef.classes_]
        is_l3_binary = len(le_cef.classes_) == 2

        print(f"  BMR-positive classes: {l3_target_names}")
        print(f"  L3 stage-2 train rows: {len(X3_train_base)}  |  test rows: {len(X3_test_base)}")

        rank_model_l3 = _build_ranking_model(args.model_type, y3_train_enc, args.random_state)
        if args.skip_rfecv:
            print("  Skipping RFECV for Level 3 stage-2 – using all features.")
            l3_shap_feats = X3_train_base.columns.tolist()
            l3_rfecv_history = {}
        else:
            print("  Selecting Level 3 stage-2 features by SHAP importance …")
            l3_shap_feats, l3_rfecv_history = shap_rfecv(
                rank_model_l3, X3_train_base, y3_train_enc, min_features=2, max_features=args.max_features, scoring="pr_auc"
            )
        l3_features = cap_features(
            l3_shap_feats, X3_train_base, y3_train_enc,
            args.max_features, args.random_state, args.model_type, rank_model=rank_model_l3
        )
        print(f"  Stage-2 SHAP selection kept {len(l3_features)} features.")

        scaler_l3 = MinMaxScaler()
        X3_train_scaled = pd.DataFrame(
            scaler_l3.fit_transform(X3_train_base[l3_features]),
            columns=l3_features, index=X3_train_base.index,
        )
        X3_test_scaled = pd.DataFrame(
            scaler_l3.transform(X3_test_base[l3_features]),
            columns=l3_features, index=X3_test_base.index,
        )

        sw_l3 = compute_balanced_sample_weight(y3_train_enc, w_train.reindex(y3_train_enc.index))
        if is_l3_binary:
            l3_params, l3_threshold, l3_study = optimise_binary_model(
                X3_train_scaled, y3_train_enc,
                n_splits=args.cv_splits,
                n_trials=args.binary_trials,
                random_state=args.random_state,
                sample_weight=sw_l3,
                model_type=args.model_type,
                study_name=_study_name("l3_stage2_bin"),
                optuna_storage=optuna_storage,
                optuna_load_if_exists=args.optuna_load_if_exists,
            )
            l3_base = build_binary_model(args.model_type, l3_params.copy())
        else:
            l3_params, l3_study = optimise_multiclass_model(
                X3_train_scaled, y3_train_enc,
                n_splits=args.cv_splits,
                n_trials=args.binary_trials,
                random_state=args.random_state,
                sample_weight=sw_l3,
                model_type=args.model_type,
                num_classes=len(le_cef.classes_),
                study_name=_study_name("l3_stage2_multi"),
                optuna_storage=optuna_storage,
                optuna_load_if_exists=args.optuna_load_if_exists,
            )
            l3_threshold = None
            l3_base = build_multiclass_model(args.model_type, l3_params.copy(), len(le_cef.classes_))

        l3_model = _fit_calibrated_or_base(l3_base, X3_train_scaled, y3_train_enc, max_cv=5, method="isotonic")

        l3_test_proba = l3_model.predict_proba(X3_test_scaled)
        if is_l3_binary:
            l3_test_pred = (l3_test_proba[:, 1] >= l3_threshold).astype(int)
        else:
            l3_test_pred = np.argmax(l3_test_proba, axis=1)

        l3_f1 = f1_score(y3_test_enc, l3_test_pred, average="macro", zero_division=0)
        l3_auc: Optional[float] = None
        if y3_test_enc.nunique() > 1:
            if is_l3_binary:
                l3_auc = float(roc_auc_score(y3_test_enc, l3_test_proba[:, 1]))
            else:
                l3_auc = float(roc_auc_score(y3_test_enc, l3_test_proba, multi_class="ovr", average="macro"))

        l3_report = classification_report(
            y3_test_enc, l3_test_pred,
            target_names=l3_target_names,
            zero_division=0,
        )

        (l3_dir / "stage1_bmr_gate_summary.json").write_text(json.dumps({
            "params": l3_gate_params,
            "threshold": l3_gate_threshold,
            "macro_f1": float(f1_score(y3_gate_test, l3_gate_test_pred, average="macro", zero_division=0)),
            "roc_auc": float(roc_auc_score(y3_gate_test, l3_gate_test_proba)) if pd.Series(y3_gate_test).nunique() > 1 else None,
            "features": l3_gate_features,
            "shaprfecv_features": l3_gate_feats,
        }, indent=2))
        l3_gate_study.trials_dataframe().to_csv(l3_dir / "stage1_bmr_gate_optuna_trials.csv", index=False)
        pd.DataFrame({
            "true_bmr": y3_gate_test,
            "pred_bmr": l3_gate_test_pred,
            "proba_bmr_positive": l3_gate_test_proba,
        }, index=X3_gate_test_scaled.index).to_csv(l3_dir / "stage1_bmr_gate_predictions.csv", index_label="row_index")

        stage2_summary = {
            "mode": "binary" if is_l3_binary else "multiclass",
            "classes": l3_target_names,
            "params": l3_params,
            "threshold": l3_threshold,
            "macro_f1": l3_f1,
            "roc_auc": l3_auc,
            "features": l3_features,
            "shaprfecv_features": l3_shap_feats,
        }
        if is_l3_binary:
            stage2_summary["negative_label"] = l3_target_names[0]
            stage2_summary["positive_label"] = l3_target_names[1]

        (l3_dir / "summary.json").write_text(json.dumps({
            "mode": "two_stage_binary" if is_l3_binary else "two_stage_multiclass",
            "stage1_gate": "BMR (NEGATIVE vs non-NEGATIVE)",
            "stage2_target": "BMR class among positives",
            "stage2": stage2_summary,
            "hemo_positive_gate": True,
        }, indent=2))
        l3_study.trials_dataframe().to_csv(l3_dir / "optuna_trials.csv", index=False)

        stage2_pred_df = pd.DataFrame(index=X3_test_scaled.index)
        stage2_pred_df["true"] = y3_test_enc.values
        stage2_pred_df["true_label"] = [l3_target_names[x] for x in y3_test_enc.values]
        stage2_pred_df["pred"] = l3_test_pred
        stage2_pred_df["pred_label"] = [l3_target_names[x] for x in l3_test_pred]
        if is_l3_binary:
            stage2_pred_df["proba_positive"] = l3_test_proba[:, 1]
        else:
            for idx, class_name in enumerate(l3_target_names):
                stage2_pred_df[f"proba_{class_name}"] = l3_test_proba[:, idx]
        stage2_pred_df.to_csv(l3_dir / "stage2_bmr_type_predictions.csv", index_label="row_index")

        all_summaries["level3_cefalosporina"] = {
            "mode": "two_stage_binary" if is_l3_binary else "two_stage_multiclass",
            "stage1_gate_macro_f1": float(f1_score(y3_gate_test, l3_gate_test_pred, average="macro", zero_division=0)),
            "stage1_gate_roc_auc": float(roc_auc_score(y3_gate_test, l3_gate_test_proba)) if pd.Series(y3_gate_test).nunique() > 1 else None,
            "stage2_macro_f1": l3_f1,
            "stage2_roc_auc": l3_auc,
            "macro_f1": l3_f1,
            "roc_auc": l3_auc,
            "classes": l3_target_names,
            "threshold": l3_threshold,
        }
        all_summaries["l3_gate_rfecv_scores"] = l3_gate_rfecv_history
        all_summaries["l3_rfecv_scores"] = l3_rfecv_history

    all_summaries["l1_rfecv_scores"] = l1_rfecv_history
    all_summaries["l2_rfecv_scores"] = l2_rfecv_history

    # ------------------------------------------------------------------
    # Aggregate summary
    # ------------------------------------------------------------------
    (output_dir / "aggregate_summary.json").write_text(
        json.dumps(all_summaries, indent=2)
    )
    print("\n" + "=" * 60)
    print("COMPLETED.  Results saved in:", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    home = Path.cwd()
    default_db = os.path.join(
        home, "mepram_data", "df_merged_full_multilabel_grouped.csv"
    )
    default_out = os.path.join(home, "mepram_data", "outputs", "independent_three_level")

    parser = argparse.ArgumentParser(
        description=(
            "Three-level independent (no cascade): "
            "sepsis → resultado_hemo_grouped → resistente_cefalosporina"
        )
    )
    parser.add_argument("--database-file", "-db", type=Path, default=default_db)
    parser.add_argument("--output-dir", "-o", type=Path, default=default_out)
    parser.add_argument(
        "--sepsis-target", type=str, default="sepsis",
        help="Column name for the Level-1 binary target (default: sepsis).",
    )
    parser.add_argument(
        "--hemo-target", type=str, default="resultado_hemo_grouped",
        help=(
            "Column name for the Level-2 multiclass target (default: resultado_hemo_grouped). "
            "Raw resultado_hemo values are also supported; when more than two positive ``hemo`` labels "
            "exist, Stage 2 will automatically use multiclass modeling."
        ),
    )
    parser.add_argument(
        "--hemo-negative-label",
        type=str,
        default="NEGATIVE",
        help="Label treated as hemoculture-negative for Level-2 Stage-1 gate.",
    )
    parser.add_argument(
        "--skip-l2-gate",
        action="store_true",
        help=(
            "Skip Level 2 hemoculture positive/negative gate and directly predict "
            "etiology classes (including NEGATIVE) in a single model."
        ),
    )
    parser.add_argument(
        "--hemo-coco-label",
        type=str,
        default="Coco gram+",
        help="Positive-class label for Level-2 Stage-2 subtype model.",
    )
    parser.add_argument(
        "--hemo-bacilo-label",
        type=str,
        default="Bacilo gram-",
        help="Negative-class label for Level-2 Stage-2 subtype model.",
    )
    parser.add_argument(
        "--bmr-target", type=str, default="bmr_etiologia",
        help="Column name for the Level-3 first-stage BMR gate target (default: bmr_etiologia).",
    )
    parser.add_argument(
        "--bmr-negative-label",
        type=str,
        default="NEGATIVE",
        help="Label treated as BMR-negative for Level-3 Stage-1 gate.",
    )
    parser.add_argument(
        "--cef-target", type=str, default="resistente_cefalosporina",
        help="Column name for the Level-3 binary target (default: resistente_cefalosporina).",
    )
    parser.add_argument(
        "--skip-l3-hemo-filter",
        action="store_true",
        help=(
            "Skip filtering Level 3 rows to resultado_hemo != NEGATIVE before the BMR gate. "
            "By default, Level 3 is scoped to positive blood culture rows only."
        ),
    )
    parser.add_argument(
        "--cef-multi-positive-label",
        type=str,
        default="cefalosporina",
        help=(
            "Token used to mark the positive class in resistente_cefalosporina_multi "
            "for Level-3 stage-2 (default: 'cefalosporina')."
        ),
    )
    parser.add_argument(
        "--weight-column", type=str, default="sample_weight",
        help="Column containing per-sample weights (default: sample_weight).",
    )
    parser.add_argument(
        "--model-type", type=str, choices=["xgb", "lgbm", "rf", "catb"],
        default="lgbm",
        help=(
            "Estimator used for all three levels. "
            "Note: Level 2 always uses lgbm/xgb/rf (not catb) for multiclass. "
            "Default: lgbm."
        ),
    )
    parser.add_argument(
        "--binary-trials", "-btrials", type=int, default=500,
        help="Optuna trials per level (default: 500).",
    )
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument(
        "--test-size", "-tsize", type=float, default=0.35,
        help="Hold-out fraction (default: 0.35).",
    )
    parser.add_argument("--random-state", type=int, default=99)
    parser.add_argument(
        "--optuna-storage",
        type=str,
        default="",
        help=(
            "Optuna storage URL (e.g. sqlite:////mnt/.../optuna_studies/three_level.db "
            "or postgresql://...). If empty, studies are in-memory only."
        ),
    )
    parser.add_argument(
        "--optuna-study-prefix",
        type=str,
        default="three_level",
        help="Prefix used to build per-stage Optuna study names.",
    )
    parser.add_argument(
        "--optuna-load-if-exists",
        action="store_true",
        help="Reuse existing Optuna studies with the same study name.",
    )
    parser.add_argument(
        "--na-perc-limit", "-na", type=float, default=0.20,
        help="Drop columns with more than this fraction of missing values.",
    )
    parser.add_argument(
        "--max-features", type=int, default=None,
        help=(
            "Maximum number of features to keep per level, ranked by mean |SHAP|. "
            "Default: None (keep all features selected by SHAP-RFECV)."
        ),
    )
    parser.add_argument(
        "--l3-min-positive", type=int, default=30,
        help=(
            "Minimum number of minority-class training samples for Level 3 "
            "before a warning is printed about unreliable results. "
            "CV folds are also capped to this count if it is lower than --cv-splits. "
            "Default: 30."
        ),
    )
    parser.add_argument(
        "--max-corr", type=float, default=0.90,
        help=(
            "Remove features whose absolute Spearman correlation with any "
            "previously-kept feature exceeds this threshold. "
            "Set to 1.0 to disable. Default: 0.90 (removes near-duplicate "
            "continuous/recoded vital-sign pairs and perfectly correlated "
            "one-hot complements)."
        ),
    )
    parser.add_argument(
        "--no-impute", dest="impute_missing", action="store_false",
        help="Disable missing-value imputation (default: enabled).",
    )
    parser.set_defaults(impute_missing=True)

    parser.add_argument(
    "--skip-rfecv",
    action="store_true",
    help="Skip RFECV and use all features."
    )
    return parser


def main() -> None:
    start = time.time()
    parser = build_arg_parser()
    args = parser.parse_args()
    print("Parsed args:", args)

    output_folder = Path(args.output_dir)
    try:
        run_training(args)
    except Exception:
        if output_folder.exists():
            shutil.rmtree(output_folder)
        raise
    print(f"\nElapsed: {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
