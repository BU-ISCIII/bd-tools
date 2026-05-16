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

from .components import *

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

