"""Master three-level modelling pipeline orchestration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize

from .utils import (
    DELETE_COLUMNS,
    METADATA_COLUMNS,
    N_CPUS,
    JOB_ID,
    TODAY,
    TARGET_REMOVE,
    GATE_RECALL,
    compute_balanced_sample_weight,
    load_processed_dataframe,
    preprocess_train_test_features,
)
from .config import runtime_defaults
from .models import _find_threshold_for_recall, _fit_calibrated_or_base, build_binary_model, build_multiclass_model
from .optuna_utils import optimise_binary_model, optimise_multiclass_model
from .oof import generate_oof_probas_binary, generate_oof_probas_multiclass
from .feature_filters import shap_rfecv, remove_correlated_features, fit_iqr_bounds, apply_iqr_bounds_to_nan
from .feature_selection import cap_features, _build_ranking_model

def _map_hemo_subtype_to_gnb_binary(target_series: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(target_series.str.contains("GNB", case=False, na=False), "GNB", "non_GNB"),
        index=target_series.index,
        name=target_series.name,
    )

def run_training(args: argparse.Namespace) -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("SELECTED ARGS:", args)

    # ------------------------------------------------------------------
    # 1. Load & preprocess
    # ------------------------------------------------------------------
    all_targets = {
        args.sepsis_target,
        args.hemo_target,
        args.hemo_gate_target,
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

    available_metadata_cols = [c for c in METADATA_COLUMNS if c in working_df.columns]
    metadata_df = working_df[available_metadata_cols].copy()

    exclude_cols = all_targets.union(available_metadata_cols)
    feature_cols = [c for c in working_df.columns if c not in exclude_cols]
    if not feature_cols:
        raise ValueError("No feature columns remain after excluding targets.")

    feature_df_raw = working_df[feature_cols]

    # ------------------------------------------------------------------
    # 2. Train / test split – stratified on sepsis
    # ------------------------------------------------------------------
    y_sepsis = working_df.loc[feature_df_raw.index, args.sepsis_target]
    y_hemo = working_df.loc[feature_df_raw.index, args.hemo_target]
    y_hemo_gate = working_df.loc[feature_df_raw.index, args.hemo_gate_target]
    y_cef = working_df.loc[feature_df_raw.index, args.cef_target]
    y_bmr = working_df.loc[feature_df_raw.index, args.bmr_target]
    w = working_df.loc[feature_df_raw.index, args.weight_column]

    (
        X_train, X_test,
        y_sep_train, y_sep_test,
        y_hemo_train, y_hemo_test,
        y_hemo_gate_train, y_hemo_gate_test,
        y_cef_train, y_cef_test,
        y_bmr_train, y_bmr_test,
        w_train, w_test,
    ) = train_test_split(
        feature_df_raw, y_sepsis, y_hemo, y_hemo_gate, y_cef, y_bmr, w,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_sepsis,
    )

    # ------------------------------------------------------------------
    # 2a. Leakage-safe preprocessing (fit on train, apply on test)
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create processing folder for data processing artifacts
    processing_dir = output_dir / "processing"
    processing_dir.mkdir(parents=True, exist_ok=True)

    iqr_bounds = fit_iqr_bounds(X_train, iqr_multiplier=5.0)

    X_train = apply_iqr_bounds_to_nan(
        X_train,
        iqr_bounds,
        output_csv_path=output_dir / "processing" / "iqr_outliers_train.csv",
    )

    X_test = apply_iqr_bounds_to_nan(
        X_test,
        iqr_bounds,
        output_csv_path=output_dir / "processing" / "iqr_outliers_test.csv",
    )

    cat_cols = runtime_defaults().get("categorical_for_dummies", ["foco", "ultimo_antib"])
    X_train, X_test = preprocess_train_test_features(
        X_train, X_test,
        na_perc_limit=args.na_perc_limit,
        impute_missing=args.impute_missing,
        categorical_for_dummies=cat_cols,
        output_csv_paths={
            "nan_dropped": output_dir / "processing" / "nan_dropped_features.csv",
            "imputed": output_dir / "processing" / "imputed_features.csv",
        },
    )
    # Align targets/weights after row filtering caused by no-impute mode
    y_sep_train, y_hemo_train, y_hemo_gate_train, y_cef_train, y_bmr_train, w_train = (
        y_sep_train.loc[X_train.index],
        y_hemo_train.loc[X_train.index],
        y_hemo_gate_train.loc[X_train.index],
        y_cef_train.loc[X_train.index],
        y_bmr_train.loc[X_train.index],
        w_train.loc[X_train.index],
    )
    y_sep_test, y_hemo_test, y_hemo_gate_test, y_cef_test, y_bmr_test, w_test = (
        y_sep_test.loc[X_test.index],
        y_hemo_test.loc[X_test.index],
        y_hemo_gate_test.loc[X_test.index],
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
        kept_cols = remove_correlated_features(
            X_train, 
            threshold=args.max_corr,
            output_csv_path=output_dir / "processing" / "correlation_dropped_features.csv",
        )
        n_removed = len(X_train.columns) - len(kept_cols)
        if n_removed:
            print(f"  Dropped {n_removed} redundant features → {len(kept_cols)} remain.")
        X_train = X_train[kept_cols]
        X_test = X_test[kept_cols]


    # ------------------------------------------------------------------
    # 3. Output directory
    # ------------------------------------------------------------------

    all_summaries: Dict = {"args": {str(k): str(v) for k,v in args.__dict__.items()}}
    optuna_storage = args.optuna_storage or None
    study_prefix = args.optuna_study_prefix.strip() if args.optuna_study_prefix else "three_level"
    if optuna_storage:
        print(f"Optuna storage: {optuna_storage}")

    def _study_name(tag: str) -> str:
        return f"{study_prefix}_{tag}_rs{args.random_state}"
    
    def _format_report_with_confusion(
        *,
        title: str,
        report_label: str,
        report: str,
        confusion,
        predicted_counts: dict,
        actual_counts: dict,
        binary_labels: Optional[tuple] = None,
    ) -> str:
        text = (
            f"{title}\n\n"
            f"{report_label} classification report:\n"
            f"{report}\n"
        )

        if binary_labels is not None:
            neg, pos = binary_labels
            text += "\nConfusion matrix [TN, FP; FN, TP]:\n"
            text += f"{confusion}\n"
            text += f"predicted_true: {predicted_counts.get(pos, 0)}\n"
            text += f"predicted_false: {predicted_counts.get(neg, 0)}\n"
            text += f"real_true: {actual_counts.get(pos, 0)}\n"
            text += f"real_false: {actual_counts.get(neg, 0)}\n"
        else:
            text += "\nConfusion matrix:\n"
            text += f"{confusion}\n"
            text += f"predicted_counts: {predicted_counts}\n"
            text += f"actual_counts: {actual_counts}\n"

        return text


    class _NoOptunaStudy:
        def trials_dataframe(self):
            return pd.DataFrame()

    def _optimise_binary(*, X, y, n_splits, n_trials, random_state, sample_weight, model_type, study_name):
        if not args.skip_optuna:
            return optimise_binary_model(
                X, y, n_splits=n_splits, n_trials=n_trials, random_state=random_state,
                sample_weight=sample_weight, model_type=model_type, study_name=study_name,
                optuna_storage=optuna_storage, optuna_load_if_exists=args.optuna_load_if_exists,
            )
        return {}, 0.5, _NoOptunaStudy()

    def _optimise_multiclass(*, X, y, n_splits, n_trials, random_state, sample_weight, model_type, num_classes, study_name):
        if not args.skip_optuna:
            return optimise_multiclass_model(
                X, y, n_splits=n_splits, n_trials=n_trials, random_state=random_state,
                sample_weight=sample_weight, model_type=model_type, num_classes=num_classes,
                study_name=study_name, optuna_storage=optuna_storage,
                optuna_load_if_exists=args.optuna_load_if_exists,
            )
        return {}, _NoOptunaStudy()


    # ==================================================================
    # LEVEL 1 – sepsis
    # ==================================================================
    if args.skip_level1:
        print("\n" + "=" * 60)
        print("LEVEL 1: sepsis")
        print("=" * 60)
        print("  Skipped Level 1 by flag.")
        l1_rfecv_history = {}
        all_summaries["level1_sepsis"] = {"skipped": True}
    else:
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
                rank_model_l1, X_train, y_sep_train_enc, min_features=1, max_features=args.max_features, scoring="pr_auc",
                output_csv_path=output_dir / "processing" / "l1_rfecv_features.csv",
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
        l1_params, l1_threshold, l1_study = _optimise_binary(
            X=X_l1_train, y=y_sep_train_enc,
            n_splits=args.cv_splits,
            n_trials=args.binary_trials,
            random_state=args.random_state,
            sample_weight=sw_l1,
            model_type=args.model_type,
            study_name=_study_name("l1")
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
        l1_pr_auc = average_precision_score(y_sep_test_enc, l1_test_proba)
        l1_f1 = f1_score(y_sep_test_enc, l1_test_pred, average="macro", zero_division=0)
        l1_precision = precision_score(y_sep_test_enc, l1_test_pred, average="macro", zero_division=0)
        l1_recall = recall_score(y_sep_test_enc, l1_test_pred, average="macro", zero_division=0)
        print(
            f"  Level 1 – Macro F1: {l1_f1:.3f}  |  ROC-AUC: {l1_auc:.3f}  |  PR-AUC: {l1_pr_auc:.3f}"
            f"  |  Precision: {l1_precision:.3f}  |  Recall: {l1_recall:.3f}"
        )

        l1_confusion = confusion_matrix(y_sep_test_enc, l1_test_pred, labels=[0, 1]).tolist()
        l1_pred_counts = pd.Series(l1_test_pred).map({0: le_sep.classes_[0], 1: le_sep.classes_[1]}).value_counts().to_dict()
        l1_true_counts = pd.Series(y_sep_test_enc).map({0: le_sep.classes_[0], 1: le_sep.classes_[1]}).value_counts().to_dict()
        l1_report_text = (
            "Level 1 Sepsis Report\n\n"
            + l1_report
            + "\nConfusion matrix [TN, FP; FN, TP]:\n"
            + f"{l1_confusion}\n"
            + f"predicted_counts: {l1_pred_counts}\n"
            + f"actual_counts: {l1_true_counts}\n"
        )

        # Level 1
        (l1_dir / f"report_{args.sepsis_target}.txt").write_text(l1_report_text)
        (l1_dir / "summary.json").write_text(json.dumps({
            "target": args.sepsis_target,
            "params": l1_params,
            "threshold": l1_threshold,
            "macro_f1": l1_f1,
            "roc_auc": l1_auc,
            "pr_auc": l1_pr_auc,
            "precision": l1_precision,
            "recall": l1_recall,
            "classes": le_sep.classes_.tolist(),
            "negative_label": str(le_sep.classes_[0]),
            "positive_label": str(le_sep.classes_[1]),
            "predicted_counts": l1_pred_counts,
            "actual_counts": l1_true_counts,
            "confusion_matrix": l1_confusion,
            "shaprfecv_features": selected_features,
            "features": l1_features,
        }, indent=2))
        l1_study.trials_dataframe().to_csv(l1_dir / "optuna_trials.csv", index=False)
        l1_pred_df = pd.DataFrame({
            "true": y_sep_test_enc.values,
            "pred": l1_test_pred,
            "proba": l1_test_proba,
        }, index=X_l1_test.index)

        l1_pred_df = metadata_df.reindex(l1_pred_df.index).join(l1_pred_df)

        # Add human-readable labels and error/margin info (like stage-1 gate)
        l1_pred_df["true_label"] = np.where(l1_pred_df["true"] == 1, str(le_sep.classes_[1]), str(le_sep.classes_[0]))
        l1_pred_df["pred_label"] = np.where(l1_pred_df["pred"] == 1, str(le_sep.classes_[1]), str(le_sep.classes_[0]))

        l1_pred_df["error_type"] = np.select(
            [
                (l1_pred_df["true"] == 0) & (l1_pred_df["pred"] == 0),
                (l1_pred_df["true"] == 0) & (l1_pred_df["pred"] == 1),
                (l1_pred_df["true"] == 1) & (l1_pred_df["pred"] == 0),
                (l1_pred_df["true"] == 1) & (l1_pred_df["pred"] == 1),
            ],
            ["TN", "FP", "FN", "TP"],
            default="NA",
        )

        l1_pred_df["margin_from_threshold"] = l1_pred_df["proba"] - float(l1_threshold)
        l1_pred_df["abs_margin_from_threshold"] = l1_pred_df["margin_from_threshold"].abs()

        l1_pred_df.to_csv(l1_dir / "predictions.csv", index_label="row_index")

        # Predictions CSV contains labels, probabilities and margin fields

        all_summaries["level1_sepsis"] = {
            "target": args.sepsis_target,
            "macro_f1": l1_f1,
            "roc_auc": l1_auc,
            "pr_auc": l1_pr_auc,
            "precision": l1_precision,
            "recall": l1_recall,
            "classes": le_sep.classes_.tolist(),
            "negative_label": str(le_sep.classes_[0]),
            "positive_label": str(le_sep.classes_[1]),
            "threshold": l1_threshold,
            "predicted_counts": l1_pred_counts,
            "actual_counts": l1_true_counts,
            "confusion_matrix": l1_confusion,
        }

    # ==================================================================
    # LEVEL 2 – two-stage binary hemo model
    # ==================================================================
    if args.skip_level2:
        print("\n" + "=" * 60)
        print("LEVEL 2: skipped")
        print("=" * 60)
        print("  Skipped Level 2 by flag.")
        l2_rfecv_history = {}
        all_summaries["level2_hemo"] = {"skipped": True}
    else:
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
                    rank_model_l2, X2_train_base, y2_train, min_features=1, max_features=args.max_features, scoring="pr_auc",
                    output_csv_path=output_dir / "processing" / "l2_direct_rfecv_features.csv",
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
                l2_params, l2_threshold, l2_study = _optimise_binary(
                    X=X2_train_scaled, y=y2_train,
                    n_splits=args.cv_splits,
                    n_trials=args.binary_trials,
                    random_state=args.random_state,
                    sample_weight=sw_l2,
                    model_type=args.model_type,
                    study_name=_study_name("l2_direct_bin")
                )
                l2_base = build_binary_model(args.model_type, l2_params.copy())
            else:
                l2_params, l2_study = _optimise_multiclass(
                    X=X2_train_scaled, y=y2_train,
                    n_splits=args.cv_splits,
                    n_trials=args.binary_trials,
                    random_state=args.random_state,
                    sample_weight=sw_l2,
                    model_type=args.model_type,
                    num_classes=len(le_l2.classes_),
                    study_name=_study_name("l2_direct_multi")
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
            l2_precision = precision_score(y2_test, l2_test_pred, average="macro", zero_division=0)
            l2_recall = recall_score(y2_test, l2_test_pred, average="macro", zero_division=0)
            l2_pr_auc: Optional[float] = None
            l2_auc: Optional[float] = None
            if y2_test.nunique() > 1:
                if is_binary_subtype:
                    l2_auc = float(roc_auc_score(y2_test, l2_test_proba[:, 1]))
                    l2_pr_auc = float(average_precision_score(y2_test, l2_test_proba[:, 1]))
                else:
                    l2_auc = float(roc_auc_score(y2_test, l2_test_proba, multi_class="ovr", average="macro"))
                    l2_pr_auc = float(average_precision_score(
                        label_binarize(y2_test, classes=np.arange(len(le_l2.classes_))),
                        l2_test_proba,
                        average="macro"
                    ))

            l2_report = classification_report(
                y2_test, l2_test_pred,
                target_names=l2_target_names,
                zero_division=0,
            )
            print(
                f"  Level 2 direct etiology – Macro F1: {l2_f1:.3f}"
                + (f"  |  ROC-AUC: {l2_auc:.3f}" if l2_auc is not None else "")
                + (f"  |  PR-AUC: {l2_pr_auc:.3f}" if l2_pr_auc is not None else "")
                + f"  |  Precision: {l2_precision:.3f}  |  Recall: {l2_recall:.3f}"
            )

            l2_confusion = confusion_matrix(
                y2_test,
                l2_test_pred,
                labels=np.arange(len(l2_target_names)),
            ).tolist()
            l2_pred_counts = pd.Series(l2_test_pred).map(
                {i: le_l2.classes_[i] for i in range(len(le_l2.classes_))}
            ).value_counts().to_dict()
            l2_true_counts = pd.Series(y2_test).map(
                {i: le_l2.classes_[i] for i in range(len(le_l2.classes_))}
            ).value_counts().to_dict()
            l2_report_text = _format_report_with_confusion(
                title="Level 2 Target Report",
                report_label="Target",
                report=l2_report,
                confusion=l2_confusion,
                predicted_counts=l2_pred_counts,
                actual_counts=l2_true_counts,
                binary_labels=(l2_target_names[0], l2_target_names[1]) if is_binary_subtype else None,
            )

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

            stage_pred_counts = stage_pred_df["pred_label"].value_counts().to_dict()
            stage_true_counts = stage_pred_df["true_label"].value_counts().to_dict()

            stage2_summary = {
                "mode": "binary" if is_binary_subtype else "multiclass",
                "classes": l2_target_names,
                "params": l2_params,
                "threshold": l2_threshold,
                "macro_f1": l2_f1,
                "roc_auc": l2_auc,
                "pr_auc": l2_pr_auc,
                "precision": l2_precision,
                "recall": l2_recall,
                "predicted_counts": stage_pred_counts,
                "actual_counts": stage_true_counts,
                "confusion_matrix": l2_confusion,
                "shaprfecv_features": l2_shap_feats,
                "features": l2_features,
            }

            if is_binary_subtype:
                stage2_summary["negative_label"] = l2_target_names[0]
                stage2_summary["positive_label"] = l2_target_names[1]
            
            # Level 2
            (l2_dir / f"report_{args.hemo_target}.txt").write_text(l2_report_text)
            (l2_dir / "summary.json").write_text(json.dumps({
                "mode": "direct_binary" if is_binary_subtype else "direct_multiclass",
                "target": args.hemo_target,
                **stage2_summary,
            }, indent=2))

            l2_study.trials_dataframe().to_csv(
                l2_dir / "optuna_trials_direct_etiology.csv",
                index=False,
            )
            # Add error/margin info for direct binary subtype (similar to gate outputs)
            if is_binary_subtype:
                stage_pred_df["error_type"] = np.select(
                    [
                        (stage_pred_df["true"] == 0) & (stage_pred_df["pred"] == 0),
                        (stage_pred_df["true"] == 0) & (stage_pred_df["pred"] == 1),
                        (stage_pred_df["true"] == 1) & (stage_pred_df["pred"] == 0),
                        (stage_pred_df["true"] == 1) & (stage_pred_df["pred"] == 1),
                    ],
                    ["TN", "FP", "FN", "TP"],
                    default="NA",
                )
                stage_pred_df["margin_from_threshold"] = stage_pred_df["proba_positive"] - float(l2_threshold)
                stage_pred_df["abs_margin_from_threshold"] = stage_pred_df["margin_from_threshold"].abs()
                stage_pred_df = metadata_df.reindex(stage_pred_df.index).join(stage_pred_df)
                stage_pred_df.to_csv(l2_dir / "predictions_direct_etiology.csv", index_label="row_index")
                stage_pred_df.query("error_type in ['FP','FN']").to_csv(
                    l2_dir / "errors_direct_etiology.csv", index_label="row_index"
                )
                (
                    stage_pred_df
                    .groupby("error_type")
                    .agg(
                        n=("error_type", "size"),
                        mean_proba=("proba_positive", "mean"),
                        median_proba=("proba_positive", "median"),
                        min_proba=("proba_positive", "min"),
                        max_proba=("proba_positive", "max"),
                        mean_abs_margin=("abs_margin_from_threshold", "mean"),
                    )
                    .reset_index()
                    .to_csv(l2_dir / "error_summary_direct_etiology.csv", index=False)
                )
            else:
                stage_pred_df = metadata_df.reindex(stage_pred_df.index).join(stage_pred_df)
                stage_pred_df.to_csv(
                    l2_dir / "predictions_direct_etiology.csv",
                    index_label="row_index",
                )

            all_summaries["level2_hemo"] = {
                "mode": "direct_binary" if is_binary_subtype else "direct_multiclass",
                "target": args.hemo_target,
                "classes": l2_target_names,
                "threshold": l2_threshold,
                "macro_f1": l2_f1,
                "roc_auc": l2_auc,
                "pr_auc": l2_pr_auc,
                "precision": l2_precision,
                "recall": l2_recall,
                "predicted_counts": stage_pred_counts,
                "actual_counts": stage_true_counts,
                "confusion_matrix": l2_confusion,
            }

            all_summaries["l2_gate_rfecv_scores"] = l2_rfecv_history

        else:
            # Stage 1: binary gate using separate gate target (e.g. infected_yes_no)
            hemo_gate_valid_train = y_hemo_gate_train.notna()
            hemo_gate_valid_test = y_hemo_gate_test.notna()
            y2_gate_train = (
                y_hemo_gate_train.loc[hemo_gate_valid_train].astype(str) != args.hemo_gate_negative_label
            ).astype(int)
            gate_test_mask = hemo_gate_valid_test
            y2_gate_test = (
                y_hemo_gate_test.loc[gate_test_mask].astype(str) != args.hemo_gate_negative_label
            ).astype(int)

            X2_gate_train_base = X_train.loc[hemo_gate_valid_train].copy()
            X2_gate_test_base = X_test.loc[gate_test_mask].copy()

            rank_model_l2_gate = _build_ranking_model(args.model_type, y2_gate_train, args.random_state)
            if args.skip_rfecv:
                print("  Skipping RFECV for Level 2 Stage-1 – using all features.")
                l2_gate_shap_feats = X2_gate_train_base.columns.tolist()
                l2_gate_rfecv_history = {}
            else:
                print("  Selecting Level 2 Stage-1 features by SHAP importance …")
                l2_gate_shap_feats, l2_gate_rfecv_history = shap_rfecv(
                    rank_model_l2_gate, X2_gate_train_base, y2_gate_train, min_features=1, max_features=args.max_features, scoring="pr_auc",
                    output_csv_path=output_dir / "processing" / "l2_gate_rfecv_features.csv",
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
            l2_gate_params, l2_gate_threshold, l2_gate_study = _optimise_binary(
                X=X2_gate_train_scaled, y=y2_gate_train,
                n_splits=args.cv_splits,
                n_trials=args.binary_trials,
                random_state=args.random_state,
                sample_weight=sw_l2_gate,
                model_type=args.model_type,
                study_name=_study_name("l2_gate")
            )
            
            l2_gate_base = build_binary_model(args.model_type, l2_gate_params.copy())
            l2_gate_model = _fit_calibrated_or_base(
                l2_gate_base, X2_gate_train_scaled, y2_gate_train, max_cv=5, method="isotonic"
            )
            l2_gate_train_proba = pd.Series(
                generate_oof_probas_binary(
                    X=X2_gate_train_scaled,
                    y=y2_gate_train,
                    best_params=l2_gate_params,
                    model_type=args.model_type,
                    n_splits=args.cv_splits,
                    random_state=args.random_state,
                ),
                index=X2_gate_train_scaled.index,
                name="l2_gate_proba",
            )

            l2_gate_optuna_threshold = l2_gate_threshold

            if args.set_gate_recall is not None:
                if not 0.0 <= args.set_gate_recall <= 1.0:
                    raise ValueError("--set-gate-recall must be between 0 and 1.")
                l2_gate_threshold = _find_threshold_for_recall(
                    y2_gate_train,
                    l2_gate_train_proba,
                    target_recall=args.set_gate_recall,
                )
                threshold_strategy = f"target_recall_{args.set_gate_recall}"
            else:
                threshold_strategy = "best_threshold"

            l2_gate_test_proba = l2_gate_model.predict_proba(X2_gate_test_scaled)[:, 1]
            l2_gate_test_proba = pd.Series(
                l2_gate_test_proba,
                index=X2_gate_test_scaled.index,
                name="l2_gate_proba",
            )
            l2_gate_test_pred = (l2_gate_test_proba >= l2_gate_threshold).astype(int)
            l2_gate_test_pred = pd.Series(l2_gate_test_pred, index=X2_gate_test_scaled.index, name="l2_gate_pred")
            l2_gate_train_pred = (l2_gate_train_proba >= l2_gate_threshold).astype(int)
            l2_gate_train_pred = pd.Series(
                l2_gate_train_pred, index=X2_gate_train_scaled.index, name="l2_gate_pred"
            )
            l2_gate_f1 = f1_score(y2_gate_test, l2_gate_test_pred, average="macro", zero_division=0)
            l2_gate_precision = precision_score(y2_gate_test, l2_gate_test_pred, average="macro", zero_division=0)
            l2_gate_recall = recall_score(y2_gate_test, l2_gate_test_pred, average="macro", zero_division=0)
            l2_gate_pr_auc: Optional[float] = None
            l2_gate_auc: Optional[float] = None
            if y2_gate_test.nunique() > 1:
                l2_gate_auc = float(roc_auc_score(y2_gate_test, l2_gate_test_proba))
                l2_gate_pr_auc = float(average_precision_score(y2_gate_test, l2_gate_test_proba))


            # Stage 2: predict the full Level-2 target among rows predicted gate-positive.
            # This keeps NEGATIVE as a valid subtype label (for gate false positives).
            subtype_train_mask = hemo_valid_train & hemo_gate_valid_train & (l2_gate_train_pred == 1)
            subtype_test_mask = (
                hemo_valid_test & hemo_gate_valid_test
                & (l2_gate_test_pred == 1)
            )

            X2_sub_train_base = X_train.loc[subtype_train_mask].copy()
            X2_sub_test_base = X_test.loc[subtype_test_mask].copy()
            if args.l2_gate_proba_as_feature:
                print("  Adding Level 2 gate probability as a feature to Stage 2 subtype modelling.")
                X2_sub_train_base = X2_sub_train_base.join(
                    l2_gate_train_proba.loc[subtype_train_mask], how="left"
                )
                X2_sub_test_base = X2_sub_test_base.join(
                    l2_gate_test_proba.loc[subtype_test_mask], how="left"
                )
            y2_sub_train_raw = y_hemo_train.loc[subtype_train_mask].astype(str)
            y2_sub_test_raw = y_hemo_test.loc[subtype_test_mask].astype(str)

            stage2_mode = args.hemo_stage2_mode
            if stage2_mode == "binary":
                print("  Stage 2 mode: binary GNB vs non-GNB")
                y2_sub_train_raw = _map_hemo_subtype_to_gnb_binary(y2_sub_train_raw)
                y2_sub_test_raw = _map_hemo_subtype_to_gnb_binary(y2_sub_test_raw)
            elif stage2_mode == "multiclass":
                print("  Stage 2 mode: multiclass (all etiology classes)")
            else:
                print("  Stage 2 mode: auto (binary if 2 classes else multiclass)")

            if y2_sub_train_raw.empty:
                raise ValueError(
                    "No train rows for Level 2 Stage-2 subtype after gating. "
                    "Check that the Level-2 gate target and Level-2 etiology target are present and aligned."
                )

            if stage2_mode == "binary" and y2_sub_train_raw.nunique() < 2:
                raise ValueError(
                    "Stage 2 binary mode requires both GNB and non-GNB labels in the training subset."
                )

            le_l2 = LabelEncoder()
            y2_sub_train = pd.Series(
                le_l2.fit_transform(y2_sub_train_raw),
                index=y2_sub_train_raw.index,
                name="hemo_sub_enc",
            )

            subtype_test_mask = subtype_test_mask & y2_sub_test_raw.isin(le_l2.classes_)

            X2_sub_test_base = X2_sub_test_base.loc[subtype_test_mask].copy()

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
                    rank_model_l2_sub, X2_sub_train_base, y2_sub_train, min_features=1, max_features=args.max_features, scoring="pr_auc",
                    output_csv_path=output_dir / "processing" / "l2_sub_rfecv_features.csv",
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
                l2_params, l2_threshold, l2_study = _optimise_binary(
                    X=X2_train_scaled, y=y2_sub_train,
                    n_splits=args.cv_splits,
                    n_trials=args.binary_trials,
                    random_state=args.random_state,
                    sample_weight=sw_l2,
                    model_type=args.model_type,
                    study_name=_study_name("l2_sub_bin")
                )
                l2_base = build_binary_model(args.model_type, l2_params.copy())
            else:
                l2_params, l2_study = _optimise_multiclass(
                    X=X2_train_scaled, y=y2_sub_train,
                    n_splits=args.cv_splits,
                    n_trials=args.binary_trials,
                    random_state=args.random_state,
                    sample_weight=sw_l2,
                    model_type=args.model_type,
                    num_classes=len(le_l2.classes_),
                    study_name=_study_name("l2_sub_multi")
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
            l2_precision = precision_score(y2_sub_test, l2_test_pred, average="macro", zero_division=0)
            l2_recall = recall_score(y2_sub_test, l2_test_pred, average="macro", zero_division=0)
            l2_pr_auc: Optional[float] = None
            l2_auc: Optional[float] = None
            if y2_sub_test.nunique() > 1:
                if is_binary_subtype:
                    l2_auc = float(roc_auc_score(y2_sub_test, l2_test_proba[:, 1]))
                    l2_pr_auc = float(average_precision_score(y2_sub_test, l2_test_proba[:, 1]))
                else:
                    l2_auc = float(roc_auc_score(y2_sub_test, l2_test_proba, multi_class="ovr", average="macro"))
                    l2_pr_auc = float(average_precision_score(
                        label_binarize(y2_sub_test, classes=np.arange(len(le_l2.classes_))),
                        l2_test_proba,
                        average="macro"
                    ))

            l2_report = classification_report(
                y2_sub_test, l2_test_pred,
                target_names=l2_target_names,
                zero_division=0,
            )
            print(
                f"  Level 2 Stage-1 gate – Macro F1: {l2_gate_f1:.3f}"
                + (f"  |  ROC-AUC: {l2_gate_auc:.3f}" if l2_gate_auc is not None else "")
                + (f"  |  PR-AUC: {l2_gate_pr_auc:.3f}" if l2_gate_pr_auc is not None else "")
                + f"  |  Precision: {l2_gate_precision:.3f}  |  Recall: {l2_gate_recall:.3f}"
            )
            print(
                f"  Level 2 Stage-2 subtype – Macro F1: {l2_f1:.3f}"
                + (f"  |  ROC-AUC: {l2_auc:.3f}" if l2_auc is not None else "")
                + (f"  |  PR-AUC: {l2_pr_auc:.3f}" if l2_pr_auc is not None else "")
                + f"  |  Precision: {l2_precision:.3f}  |  Recall: {l2_recall:.3f}"
            )

            stage1_confusion = confusion_matrix(y2_gate_test, l2_gate_test_pred, labels=[0, 1]).tolist()
            stage1_pred_counts = {
                "gate_negative": int((l2_gate_test_pred == 0).sum()),
                "gate_positive": int((l2_gate_test_pred == 1).sum()),
            }
            stage1_true_counts = {
                "gate_negative": int((y2_gate_test == 0).sum()),
                "gate_positive": int((y2_gate_test == 1).sum()),
            }

            stage1_report_text = _format_report_with_confusion(
                title="Stage 1 Gate Report",
                report_label="Binary gate",
                report=classification_report(
                    y2_gate_test,
                    l2_gate_test_pred,
                    target_names=["gate_negative", "gate_positive"],
                    zero_division=0,
                ),
                confusion=stage1_confusion,
                predicted_counts=stage1_pred_counts,
                actual_counts=stage1_true_counts,
                binary_labels=("gate_negative", "gate_positive"),
            )

            (l2_dir / f"report_gate_{args.hemo_gate_target}.txt").write_text(stage1_report_text)

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

            stage2_pred_counts = stage2_pred_df["pred_label"].value_counts().to_dict()
            stage2_true_counts = stage2_pred_df["true_label"].value_counts().to_dict()

            stage2_confusion = confusion_matrix(
                y2_sub_test,
                l2_test_pred,
                labels=np.arange(len(l2_target_names)),
            ).tolist()

            stage2_report_text = _format_report_with_confusion(
                title="Stage 2 Staged Target Report",
                report_label="Staged target",
                report=l2_report,
                confusion=stage2_confusion,
                predicted_counts=stage2_pred_counts,
                actual_counts=stage2_true_counts,
                binary_labels=(l2_target_names[0], l2_target_names[1]) if is_binary_subtype else None,
            )

            (l2_dir / f"report_staged_{args.hemo_target}.txt").write_text(stage2_report_text)

            actual_stage2_mode = (
                stage2_mode
                if stage2_mode != "auto"
                else ("binary" if is_binary_subtype else "multiclass")
            )
            stage2_summary = {
                "mode": "binary" if is_binary_subtype else "multiclass",
                "stage2_mode": actual_stage2_mode,
                "classes": l2_target_names,
                "params": l2_params,
                "threshold": l2_threshold,
                "macro_f1": l2_f1,
                "roc_auc": l2_auc,
                "pr_auc": l2_pr_auc,
                "precision": l2_precision,
                "recall": l2_recall,
                "gate_proba_feature": args.l2_gate_proba_as_feature,
                "predicted_counts": stage2_pred_counts,
                "actual_counts": stage2_true_counts,
                "confusion_matrix": stage2_confusion,
                "shaprfecv_features": l2_shap_feats,
                "features": l2_features,
            }

            if is_binary_subtype:
                stage2_summary["negative_label"] = l2_target_names[0]
                stage2_summary["positive_label"] = l2_target_names[1]

            (l2_dir / "summary.json").write_text(json.dumps({
                "mode": "two_stage_binary" if is_binary_subtype else "two_stage_multiclass",
                "stage1_gate": {
                    "negative_label": args.hemo_gate_negative_label,
                    "positive_label": f"not_{args.hemo_gate_negative_label}",
                    "params": l2_gate_params,
                    "optuna_threshold": l2_gate_optuna_threshold,
                    "threshold": l2_gate_threshold,
                    "threshold_strategy": threshold_strategy,
                    "macro_f1": l2_gate_f1,
                    "roc_auc": l2_gate_auc,
                    "pr_auc": l2_gate_pr_auc,
                    "precision": l2_gate_precision,
                    "recall": l2_gate_recall,
                    "predicted_true": stage1_pred_counts["gate_positive"],
                    "predicted_false": stage1_pred_counts["gate_negative"],
                    "real_true": stage1_true_counts["gate_positive"],
                    "real_false": stage1_true_counts["gate_negative"],
                    "confusion_matrix": stage1_confusion,
                    "shaprfecv_features": l2_gate_shap_feats,
                    "features": l2_gate_features,
                },
                "stage2_subtype": stage2_summary,
            }, indent=2))

            l2_gate_study.trials_dataframe().to_csv(
                l2_dir / "optuna_trials_stage1_gate.csv",
                index=False,
            )
            l2_study.trials_dataframe().to_csv(
                l2_dir / "optuna_trials_stage2_subtype.csv",
                index=False,
            )

            l2_gate_pred_df = pd.DataFrame({
                "true": y2_gate_test.values,
                "pred": l2_gate_test_pred.values,
                "proba": l2_gate_test_proba.values,
            }, index=X2_gate_test_scaled.index)

            l2_gate_pred_df["true_label"] = np.where(
                l2_gate_pred_df["true"] == 1,
                "gate_positive",
                "gate_negative",
            )

            l2_gate_pred_df["pred_label"] = np.where(
                l2_gate_pred_df["pred"] == 1,
                "gate_positive",
                "gate_negative",
            )

            l2_gate_pred_df["error_type"] = np.select(
                [
                    (l2_gate_pred_df["true"] == 0) & (l2_gate_pred_df["pred"] == 0),
                    (l2_gate_pred_df["true"] == 0) & (l2_gate_pred_df["pred"] == 1),
                    (l2_gate_pred_df["true"] == 1) & (l2_gate_pred_df["pred"] == 0),
                    (l2_gate_pred_df["true"] == 1) & (l2_gate_pred_df["pred"] == 1),
                ],
                ["TN", "FP", "FN", "TP"],
                default="NA",
            )

            l2_gate_pred_df["margin_from_threshold"] = (
                l2_gate_pred_df["proba"] - l2_gate_threshold
            )

            l2_gate_pred_df["abs_margin_from_threshold"] = (
                l2_gate_pred_df["margin_from_threshold"].abs()
            )

            l2_gate_pred_df = metadata_df.reindex(l2_gate_pred_df.index).join(l2_gate_pred_df)

            l2_gate_pred_df.to_csv(
                l2_dir / "predictions_stage1_gate.csv",
                index_label="row_index",
            )

            l2_gate_pred_df.query("error_type in ['FP', 'FN']").to_csv(
                l2_dir / "errors_stage1_gate.csv",
                index_label="row_index",
            )

            (
                l2_gate_pred_df
                .groupby("error_type")
                .agg(
                    n=("error_type", "size"),
                    mean_proba=("proba", "mean"),
                    median_proba=("proba", "median"),
                    min_proba=("proba", "min"),
                    max_proba=("proba", "max"),
                    mean_abs_margin=("abs_margin_from_threshold", "mean"),
                )
                .reset_index()
                .to_csv(l2_dir / "error_summary_stage1_gate.csv", index=False)
            )

            stage2_pred_df = metadata_df.reindex(stage2_pred_df.index).join(stage2_pred_df)

            stage2_pred_df.to_csv(
                l2_dir / "predictions_stage2_subtype.csv",
                index_label="row_index",
            )

            # For binary subtype, add error/margin files similar to gate outputs
            if is_binary_subtype:
                stage2_pred_df["error_type"] = np.select(
                    [
                        (stage2_pred_df["true"] == 0) & (stage2_pred_df["pred"] == 0),
                        (stage2_pred_df["true"] == 0) & (stage2_pred_df["pred"] == 1),
                        (stage2_pred_df["true"] == 1) & (stage2_pred_df["pred"] == 0),
                        (stage2_pred_df["true"] == 1) & (stage2_pred_df["pred"] == 1),
                    ],
                    ["TN", "FP", "FN", "TP"],
                    default="NA",
                )
                stage2_pred_df["margin_from_threshold"] = stage2_pred_df["proba_positive"] - float(l2_threshold)
                stage2_pred_df["abs_margin_from_threshold"] = stage2_pred_df["margin_from_threshold"].abs()

                stage2_pred_df.to_csv(l2_dir / "predictions_stage2_subtype.csv", index_label="row_index")
                stage2_pred_df.query("error_type in ['FP','FN']").to_csv(
                    l2_dir / "errors_stage2_subtype.csv", index_label="row_index"
                )
                (
                    stage2_pred_df
                    .groupby("error_type")
                    .agg(
                        n=("error_type", "size"),
                        mean_proba=("proba_positive", "mean"),
                        median_proba=("proba_positive", "median"),
                        min_proba=("proba_positive", "min"),
                        max_proba=("proba_positive", "max"),
                        mean_abs_margin=("abs_margin_from_threshold", "mean"),
                    )
                    .reset_index()
                    .to_csv(l2_dir / "error_summary_stage2_subtype.csv", index=False)
                )

            actual_stage2_mode = (
                stage2_mode
                if stage2_mode != "auto"
                else ("binary" if is_binary_subtype else "multiclass")
            )
            all_summaries["level2_hemo"] = {
                "mode": "two_stage_binary" if is_binary_subtype else "two_stage_multiclass_positive_gate",
                "stage2_mode": actual_stage2_mode,
                "stage1_gate": {
                    "macro_f1": l2_gate_f1,
                    "roc_auc": l2_gate_auc,
                    "pr_auc": l2_gate_pr_auc,
                    "precision": l2_gate_precision,
                    "recall": l2_gate_recall,
                    "negative_label": args.hemo_gate_negative_label,
                    "positive_label": f"not_{args.hemo_gate_negative_label}",
                    "threshold": l2_gate_threshold,
                    "gate_proba_feature": args.l2_gate_proba_as_feature,
                    "predicted_true": stage1_pred_counts["gate_positive"],
                    "predicted_false": stage1_pred_counts["gate_negative"],
                    "real_true": stage1_true_counts["gate_positive"],
                    "real_false": stage1_true_counts["gate_negative"],
                    "confusion_matrix": stage1_confusion,
                },
                "stage2_subtype": {
                    "mode": "binary" if is_binary_subtype else "multiclass",
                    "stage2_mode": actual_stage2_mode,
                    "classes": l2_target_names,
                    "threshold": l2_threshold,
                    "predicted_counts": stage2_pred_counts,
                    "actual_counts": stage2_true_counts,
                },
            }

            all_summaries["l2_gate_rfecv_scores"] = l2_gate_rfecv_history

    # ==================================================================
    # LEVEL 3 – resistente_cefalosporina
    # ==================================================================
    print("\n" + "=" * 60)
    print("LEVEL 3: resistente_cefalosporina")

    if args.skip_level3:
        print("  Skipped Level 3 by flag.")
        all_summaries["level3_cefalosporina"] = {"skipped": True}
        (output_dir / "aggregate_summary.json").write_text(json.dumps(all_summaries, indent=2))
        print("\n" + "=" * 60)
        print("COMPLETED.  Results saved in:", output_dir)
        return

    print("=" * 60)

    l3_dir = output_dir / "level3_cefalosporina"
    l3_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Reuse Level 2 hemoculture gate
    # ------------------------------------------------------------------
    if args.skip_level2 or args.skip_l2_gate:
        raise ValueError(
            "Level 3 requires Level 2 gate predictions. "
            "Do not use --skip-level2 or --skip-l2-gate."
        )

    # TRAIN:
    # Use TRUE hemoculture-positive samples to train resistance.
    hemo_pos_train_mask = (
        y_hemo_gate_train.notna()
        & (
            y_hemo_gate_train.astype(str)
            != args.hemo_gate_negative_label
        )
    )

    # TEST / INFERENCE:
    # Use PREDICTED hemoculture-positive samples from Level 2.
    predicted_hemo_positive_idx = l2_gate_test_pred[
        l2_gate_test_pred == 1
    ].index

    hemo_pos_test_mask = (
        y_hemo_gate_test.notna()
        & X_test.index.isin(predicted_hemo_positive_idx)
    )

    print(
        f"  Level 3: training on true hemoculture-positive rows "
        f"({hemo_pos_train_mask.sum()} samples)"
    )

    print(
        f"  Level 3: predicting resistance only for Level-2 "
        f"predicted hemoculture-positive rows "
        f"({hemo_pos_test_mask.sum()} samples)"
    )

    # ------------------------------------------------------------------
    # Resistance target filtering
    # ------------------------------------------------------------------
    cef_valid_train_mask = (
        y_cef_train.notna()
        & hemo_pos_train_mask
    )

    cef_valid_test_mask = (
        y_cef_test.notna()
        & hemo_pos_test_mask
    )

    if not cef_valid_train_mask.any():
        raise ValueError(
            "No training rows with positive blood culture and valid Level-3 label. "
            "Check that 'resultado_hemo_grouped', 'bmr_etiologia', and the chosen --cef-target are present."
        )

    # Direct Level-3 resistance model (no L3 binary gate):
    # train/predict resistance target only on positive hemoculture rows.
    X3_train_base = X_train.loc[cef_valid_train_mask].copy()
    X3_test_base = X_test.loc[cef_valid_test_mask].copy()
    y3_train_raw = y_cef_train.loc[cef_valid_train_mask].astype(str)
    y3_test_raw = y_cef_test.loc[cef_valid_test_mask].astype(str)

    if y3_train_raw.empty:
        raise ValueError("No train rows available for direct Level-3 resistance modelling.")

    le_cef = LabelEncoder()
    y3_train_enc = pd.Series(le_cef.fit_transform(y3_train_raw), index=y3_train_raw.index, name="cef_enc")
    test_seen_mask = y3_test_raw.isin(le_cef.classes_)
    X3_test_base = X3_test_base.loc[test_seen_mask].copy()
    y3_test_raw = y3_test_raw.loc[test_seen_mask]
    y3_test_enc = pd.Series(le_cef.transform(y3_test_raw), index=y3_test_raw.index, name="cef_enc")

    if X3_test_base.empty:
        raise ValueError("No test rows remain for Level-3 after filtering labels unseen in training.")

    l3_target_names = [str(c) for c in le_cef.classes_]
    is_l3_binary = len(le_cef.classes_) == 2
    print(f"  Direct Level-3 classes: {l3_target_names}")
    print(f"  L3 train rows: {len(X3_train_base)}  |  test rows: {len(X3_test_base)}")

    rank_model_l3 = _build_ranking_model(args.model_type, y3_train_enc, args.random_state)
    if args.skip_rfecv:
        print("  Skipping RFECV for Level 3 – using all features.")
        l3_shap_feats = X3_train_base.columns.tolist()
        l3_rfecv_history = {}
    else:
        print("  Selecting Level 3 features by SHAP importance …")
        l3_shap_feats, l3_rfecv_history = shap_rfecv(
            rank_model_l3, X3_train_base, y3_train_enc, min_features=1, max_features=args.max_features, scoring="pr_auc",
            output_csv_path=output_dir / "processing" / "l3_rfecv_features.csv",
        )
    l3_features = cap_features(
        l3_shap_feats, X3_train_base, y3_train_enc,
        args.max_features, args.random_state, args.model_type, rank_model=rank_model_l3
    )
    print(f"  Level-3 SHAP selection kept {len(l3_features)} features.")

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
        l3_params, l3_threshold, l3_study = _optimise_binary(
            X=X3_train_scaled, y=y3_train_enc,
            n_splits=args.cv_splits,
            n_trials=args.binary_trials,
            random_state=args.random_state,
            sample_weight=sw_l3,
            model_type=args.model_type,
            study_name=_study_name("l3_direct_bin")
        )
        l3_base = build_binary_model(args.model_type, l3_params.copy())
    else:
        l3_params, l3_study = _optimise_multiclass(
            X=X3_train_scaled, y=y3_train_enc,
            n_splits=args.cv_splits,
            n_trials=args.binary_trials,
            random_state=args.random_state,
            sample_weight=sw_l3,
            model_type=args.model_type,
            num_classes=len(le_cef.classes_),
            study_name=_study_name("l3_direct_multi")
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
    l3_precision = precision_score(y3_test_enc, l3_test_pred, average="macro", zero_division=0)
    l3_recall = recall_score(y3_test_enc, l3_test_pred, average="macro", zero_division=0)
    l3_pr_auc: Optional[float] = None
    l3_auc: Optional[float] = None
    if y3_test_enc.nunique() > 1:
        if is_l3_binary:
            l3_auc = float(roc_auc_score(y3_test_enc, l3_test_proba[:, 1]))
            l3_pr_auc = float(average_precision_score(y3_test_enc, l3_test_proba[:, 1]))
        else:
            l3_auc = float(roc_auc_score(y3_test_enc, l3_test_proba, multi_class="ovr", average="macro"))
            l3_pr_auc = float(average_precision_score(
                label_binarize(y3_test_enc, classes=np.arange(len(le_cef.classes_))),
                l3_test_proba,
                average="macro"
            ))

    l3_report = classification_report(
        y3_test_enc, l3_test_pred,
        target_names=l3_target_names,
        zero_division=0,
    )
    (l3_dir / "summary.json").write_text(json.dumps({
        "mode": "direct_binary" if is_l3_binary else "direct_multiclass",
        "target": args.cef_target,
        "classes": l3_target_names,
        "params": l3_params,
        "threshold": l3_threshold,
        "macro_f1": l3_f1,
        "roc_auc": l3_auc,
        "pr_auc": l3_pr_auc,
        "precision": l3_precision,
        "recall": l3_recall,
        "features": l3_features,
        "shaprfecv_features": l3_shap_feats,
        "hemo_positive_filter": True,
    }, indent=2))
    l3_study.trials_dataframe().to_csv(l3_dir / "optuna_trials.csv", index=False)

    l3_pred_df = pd.DataFrame(index=X3_test_scaled.index)
    l3_pred_df["true"] = y3_test_enc.values
    l3_pred_df["true_label"] = [l3_target_names[x] for x in y3_test_enc.values]
    l3_pred_df["pred"] = l3_test_pred
    l3_pred_df["pred_label"] = [l3_target_names[x] for x in l3_test_pred]
    if is_l3_binary:
        l3_pred_df["proba_positive"] = l3_test_proba[:, 1]
    else:
        for idx, class_name in enumerate(l3_target_names):
            l3_pred_df[f"proba_{class_name}"] = l3_test_proba[:, idx]

    l3_pred_df = metadata_df.reindex(l3_pred_df.index).join(l3_pred_df)

    # For binary Level-3, add error/margin columns matching predictions_stage1 format
    if is_l3_binary:
        l3_pred_df["error_type"] = np.select(
            [
                (l3_pred_df["true"] == 0) & (l3_pred_df["pred"] == 0),
                (l3_pred_df["true"] == 0) & (l3_pred_df["pred"] == 1),
                (l3_pred_df["true"] == 1) & (l3_pred_df["pred"] == 0),
                (l3_pred_df["true"] == 1) & (l3_pred_df["pred"] == 1),
            ],
            ["TN", "FP", "FN", "TP"],
            default="NA",
        )
        l3_pred_df["margin_from_threshold"] = l3_pred_df["proba_positive"] - float(l3_threshold)
        l3_pred_df["abs_margin_from_threshold"] = l3_pred_df["margin_from_threshold"].abs()

    l3_pred_df.to_csv(l3_dir / "predictions.csv", index_label="row_index")

    l3_confusion = confusion_matrix(
        y3_test_enc, l3_test_pred, labels=np.arange(len(l3_target_names))
    ).tolist()
    l3_pred_counts = l3_pred_df["pred_label"].value_counts().to_dict()
    l3_true_counts = l3_pred_df["true_label"].value_counts().to_dict()
    l3_report_text = _format_report_with_confusion(
        title="Level 3 Target Report",
        report_label="Target",
        report=l3_report,
        confusion=l3_confusion,
        predicted_counts=l3_pred_counts,
        actual_counts=l3_true_counts,
        binary_labels=(l3_target_names[0], l3_target_names[1]) if is_l3_binary else None,
    )
    (l3_dir / f"report_{args.cef_target}.txt").write_text(l3_report_text)


    all_summaries["level3_cefalosporina"] = {
        "mode": "direct_binary" if is_l3_binary else "direct_multiclass",
        "target": args.cef_target,
        "macro_f1": l3_f1,
        "roc_auc": l3_auc,
        "pr_auc": l3_pr_auc,
        "precision": l3_precision,
        "recall": l3_recall,
        "classes": l3_target_names,
        "threshold": l3_threshold,
        "predicted_counts": l3_pred_counts,
        "actual_counts": l3_true_counts,
        "confusion_matrix": l3_confusion,
    }

    all_summaries["l1_rfecv_scores"] = l1_rfecv_history
    all_summaries["l2_rfecv_scores"] = l2_rfecv_history
    all_summaries["l3_rfecv_scores"] = l3_rfecv_history

    # ------------------------------------------------------------------
    # Aggregate summary
    # ------------------------------------------------------------------
    (output_dir / "aggregate_summary.json").write_text(
        json.dumps(all_summaries, indent=2)
    )

    final_report = {
        "level1_sepsis": all_summaries.get("level1_sepsis"),
        "level2_hemo": all_summaries.get("level2_hemo"),
        "level3_cefalosporina": all_summaries.get("level3_cefalosporina"),
    }
    (output_dir / "final_report.json").write_text(json.dumps(final_report, indent=2))

    print("\n" + "=" * 60)
    print("COMPLETED.  Results saved in:", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------