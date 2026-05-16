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


