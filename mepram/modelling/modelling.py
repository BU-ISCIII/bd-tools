"""Master three-level modelling pipeline orchestration."""

from __future__ import annotations

import argparse
import json
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, Optional, Sequence, Any

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
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
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
from .config import (
    feature_view_definitions,
    modelling_feature_views,
    runtime_defaults,
)
from .models import _find_threshold_for_recall, build_binary_model, build_multiclass_model
from .optuna_utils import optimise_binary_model, optimise_multiclass_model
from .feature_filters import shap_rfecv, remove_correlated_features, fit_iqr_bounds, apply_iqr_bounds_to_nan
from .feature_selection import cap_features, _build_ranking_model
from .shap_utils import save_shap_artifacts

def _matching_columns(columns: Sequence[str], patterns: Sequence[str]) -> set[str]:
    return {
        column
        for column in columns
        if any(fnmatch(column, pattern) for pattern in patterns)
    }


def _alternative_columns(
    columns: Sequence[str], alternative: dict[str, Any]
) -> set[str]:
    selected = {
        column for column in alternative.get("include", []) if column in columns
    }
    selected |= _matching_columns(columns, alternative.get("include_patterns", []))
    return selected


def resolve_feature_view_columns(
    columns: Sequence[str],
    *,
    view_name: str,
    view_config: dict[str, Any],
) -> list[str]:
    """Resolve a named view against the columns available after preprocessing.

    The input CSV can be the baseline/full view. For each configured feature
    group, all representations are first removed and only the representation
    selected by the named view is added back. This prevents raw, binary and
    categorical duplicates from leaking into a supposedly restricted view.
    """
    available = list(columns)
    view_by_name = {
        str(view["name"]): view
        for view in view_config["feature_views"]
        if isinstance(view, dict) and "name" in view
    }

    if view_name == "all":
        return available
    if view_name not in view_by_name:
        raise ValueError(
            f"Unknown feature view '{view_name}'. Available views: "
            f"{sorted(view_by_name)} plus 'all'."
        )

    view = view_by_name[view_name]
    included = _matching_columns(available, view.get("include_patterns", ["*"]))
    included |= {c for c in view.get("include", []) if c in available}

    excluded = {
        c for c in view_config.get("baseline_exclude_columns", []) if c in available
    }
    excluded |= _matching_columns(
        available, view_config.get("baseline_exclude_patterns", [])
    )
    excluded |= {c for c in view.get("exclude", []) if c in available}
    excluded |= _matching_columns(available, view.get("exclude_patterns", []))

    groups = view_config["feature_groups"]
    for group_name, selected_alternative_name in view.get("groups", {}).items():
        if group_name not in groups:
            raise ValueError(
                f"View '{view_name}' references unknown feature group '{group_name}'."
            )
        alternatives = groups[group_name].get("alternatives", {})
        if selected_alternative_name not in alternatives:
            raise ValueError(
                f"View '{view_name}' selects unknown alternative "
                f"'{selected_alternative_name}' for group '{group_name}'."
            )

        # Remove every representation belonging to this group.
        for alternative in alternatives.values():
            excluded |= _alternative_columns(available, alternative)
            excluded |= {
                c for c in alternative.get("exclude", []) if c in available
            }
            excluded |= _matching_columns(
                available, alternative.get("exclude_patterns", [])
            )

        # Re-add only the representation selected by this view.
        selected_alternative = alternatives[selected_alternative_name]
        selected_columns = _alternative_columns(available, selected_alternative)
        included |= selected_columns
        excluded -= selected_columns

        # Explicit excludes belonging to the selected alternative still win.
        excluded |= {
            c for c in selected_alternative.get("exclude", []) if c in available
        }
        excluded |= _matching_columns(
            available, selected_alternative.get("exclude_patterns", [])
        )

    selected = [
        column for column in available
        if column in included and column not in excluded
    ]
    if not selected:
        raise ValueError(f"Feature view '{view_name}' selected no available columns.")
    return selected


def apply_feature_view(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    view_name: str,
    view_config: dict[str, Any],
    stage_name: str,
    output_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = resolve_feature_view_columns(
        X_train.columns,
        view_name=view_name,
        view_config=view_config,
    )
    missing_in_test = [column for column in columns if column not in X_test.columns]
    if missing_in_test:
        raise ValueError(
            f"Feature view '{view_name}' for {stage_name} has columns missing "
            f"from test data: {missing_in_test[:20]}"
        )
    print(
        f"  Feature view for {stage_name}: '{view_name}' "
        f"({len(columns)} columns before RFECV)."
    )
    if output_dir is not None:
        pd.DataFrame({"feature": columns}).to_csv(
            output_dir / f"{stage_name}_feature_view_columns.csv", index=False
        )
    return X_train.loc[:, columns].copy(), X_test.loc[:, columns].copy()


def apply_stage_correlation_filter(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    max_corr: float,
    stage_name: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply correlation filtering independently within a stage's feature view."""
    if max_corr >= 1.0:
        return X_train, X_test
    kept_cols = remove_correlated_features(
        X_train,
        threshold=max_corr,
        output_csv_path=output_dir / f"{stage_name}_correlation_dropped_features.csv"
    )
    removed = len(X_train.columns) - len(kept_cols)
    print(
        f"  {stage_name}: correlation filter removed {removed} columns; "
        f"{len(kept_cols)} remain."
    )
    return X_train.loc[:, kept_cols].copy(), X_test.loc[:, kept_cols].copy()


class CalibratedEnsemble:
    """Average an ensemble of models calibrated on stratified folds."""

    def __init__(self, estimators: Sequence[Any]):
        if not estimators:
            raise ValueError("At least one calibrated estimator is required.")
        self.estimators = list(estimators)
        self.classes_ = self.estimators[0].classes_

    def predict_proba(self, X):
        probabilities = [estimator.predict_proba(X) for estimator in self.estimators]
        return np.mean(probabilities, axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def _make_stratified_cv(y: pd.Series, n_splits: int, random_state: int):
    class_counts = pd.Series(y).value_counts()
    if class_counts.empty or len(class_counts) < 2:
        raise ValueError("At least two classes are required for stratified CV.")
    effective_splits = min(n_splits, int(class_counts.min()))
    if effective_splits < 2:
        raise ValueError(
            "At least two observations in every class are required for stratified CV."
        )
    return StratifiedKFold(
        n_splits=effective_splits,
        shuffle=True,
        random_state=random_state,
    )


def _fit_stratified_calibrated_ensemble(
    base_estimator,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int,
    random_state: int,
    method: str = "isotonic",
    sample_weight: Optional[pd.Series] = None,
):
    """Fit fold-specific base models and calibrate each on a disjoint stratified fold."""
    cv = _make_stratified_cv(y, n_splits, random_state)
    calibrated_estimators = []
    for fit_idx, calibration_idx in cv.split(X, y):
        estimator = clone(base_estimator)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = np.asarray(sample_weight.iloc[fit_idx])
        estimator.fit(X.iloc[fit_idx], y.iloc[fit_idx], **fit_kwargs)
        try:
            calibrator = CalibratedClassifierCV(estimator=estimator, cv="prefit", method=method)
        except TypeError:  # sklearn < 1.2
            calibrator = CalibratedClassifierCV(base_estimator=estimator, cv="prefit", method=method)
        calibration_kwargs = {}
        if sample_weight is not None:
            calibration_kwargs["sample_weight"] = np.asarray(sample_weight.iloc[calibration_idx])
        calibrator.fit(X.iloc[calibration_idx], y.iloc[calibration_idx], **calibration_kwargs)
        calibrated_estimators.append(calibrator)
    return CalibratedEnsemble(calibrated_estimators)


def _generate_stratified_calibrated_oof_binary(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    best_params: dict,
    model_type: str,
    n_splits: int,
    random_state: int,
    sample_weight: Optional[pd.Series] = None,
    method: str = "isotonic",
) -> np.ndarray:
    """Calibrated stratified OOF probabilities for threshold selection/stacking."""
    outer_cv = _make_stratified_cv(y, n_splits, random_state)
    oof = np.full(len(X), np.nan, dtype=float)
    for fold, (train_idx, valid_idx) in enumerate(outer_cv.split(X, y)):
        X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr = y.iloc[train_idx]
        sw_tr = sample_weight.iloc[train_idx] if sample_weight is not None else None
        model = _fit_stratified_calibrated_ensemble(
            build_binary_model(model_type, best_params.copy()),
            X_tr, y_tr,
            n_splits=max(2, min(n_splits - 1, int(y_tr.value_counts().min()))),
            random_state=random_state + fold + 1,
            method=method,
            sample_weight=sw_tr,
        )
        oof[valid_idx] = model.predict_proba(X_va)[:, 1]
    if np.isnan(oof).any():
        raise RuntimeError("Stratified calibrated OOF generation left missing predictions.")
    return oof



def _find_best_macro_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Choose a binary threshold from OOF probabilities by maximum macro F1."""
    y_array = np.asarray(y_true, dtype=int)
    proba_array = np.asarray(probabilities, dtype=float)
    candidate_thresholds = np.unique(
        np.concatenate(([0.0], proba_array, [1.0]))
    )
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in candidate_thresholds:
        predictions = (proba_array >= threshold).astype(int)
        score = f1_score(y_array, predictions, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold

def _map_hemo_subtype_to_gnb_binary(target_series: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(target_series.str.contains("GNB", case=False, na=False), "GNB", "non_GNB"),
        index=target_series.index,
        name=target_series.name,
    )

def run_training(args: argparse.Namespace) -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("SELECTED ARGS:", args)
    skip_shap_values = getattr(args, "skip_shap_values", False)

    feature_view_config = feature_view_definitions()
    stage_feature_views = modelling_feature_views()

    print("Feature views selected for modelling:")
    for stage, view_name in stage_feature_views.items():
        print(f"  {stage}: {view_name}")

    defined_views = {
        str(view["name"])
        for view in feature_view_config["feature_views"]
        if isinstance(view, dict) and "name" in view
    }

    unknown_assignments = {
        stage: view_name
        for stage, view_name in stage_feature_views.items()
        if view_name != "all" and view_name not in defined_views
    }

    if unknown_assignments:
        raise ValueError(
            "The modelling configuration references undefined feature views: "
            f"{unknown_assignments}. Available views: "
            f"{sorted(defined_views | {'all'})}"
        )
    print("STAGE FEATURE VIEWS:", stage_feature_views)

    # ------------------------------------------------------------------
    # 1. Load & preprocess
    # ------------------------------------------------------------------
    all_targets = {
        args.sepsis_target,
        args.etiology_target,
        args.etiology_gate_target,
        args.resistance_gate_target,
        args.cef_target,
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
    # 2. One-row-per-patient validation and stratified train/test split
    # ------------------------------------------------------------------
    if "person_id" not in working_df.columns:
        raise ValueError("person_id is required to verify one row per patient.")

    duplicated_mask = working_df["person_id"].duplicated(keep=False)
    if duplicated_mask.any():
        duplicated_patients = working_df.loc[duplicated_mask, "person_id"].nunique()
        duplicated_rows = int(duplicated_mask.sum())
        raise ValueError(
            "Expected exactly one row per patient, but found "
            f"{duplicated_patients} duplicated person_id values across "
            f"{duplicated_rows} rows."
        )
    print(
        f"Confirmed one row per patient: {working_df['person_id'].nunique()} "
        f"unique patients across {len(working_df)} rows."
    )

    y_sepsis = working_df.loc[feature_df_raw.index, args.sepsis_target]
    y_hemo = working_df.loc[feature_df_raw.index, args.etiology_target]
    y_hemo_gate = working_df.loc[feature_df_raw.index, args.etiology_gate_target]
    y_cef = working_df.loc[feature_df_raw.index, args.cef_target]
    w = working_df.loc[feature_df_raw.index, args.weight_column]

    (
        X_train, X_test,
        y_sep_train, y_sep_test,
        y_hemo_train, y_hemo_test,
        y_hemo_gate_train, y_hemo_gate_test,
        y_cef_train, y_cef_test,
        w_train, w_test,
    ) = train_test_split(
        feature_df_raw,
        y_sepsis,
        y_hemo,
        y_hemo_gate,
        y_cef,
        w,
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
    y_sep_train, y_hemo_train, y_hemo_gate_train, y_cef_train, w_train = (
        y_sep_train.loc[X_train.index],
        y_hemo_train.loc[X_train.index],
        y_hemo_gate_train.loc[X_train.index],
        y_cef_train.loc[X_train.index],
        w_train.loc[X_train.index],
    )
    y_sep_test, y_hemo_test, y_hemo_gate_test, y_cef_test, w_test = (
        y_sep_test.loc[X_test.index],
        y_hemo_test.loc[X_test.index],
        y_hemo_gate_test.loc[X_test.index],
        y_cef_test.loc[X_test.index],
        w_test.loc[X_test.index],
    )
    print(f"Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

    # ------------------------------------------------------------------
    # 2b. Correlation filtering is stage-specific
    # ------------------------------------------------------------------
    # Each stage may use a different feature view. Correlation filtering is
    # therefore applied after selecting that stage's view, not globally here.


    # ------------------------------------------------------------------
    # 3. Output directory
    # ------------------------------------------------------------------

    all_summaries: Dict = {
        "args": {str(k): str(v) for k, v in args.__dict__.items()},
        "feature_views": stage_feature_views,
    }
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

        X1_train_base, X1_test_base = apply_feature_view(
            X_train, X_test,
            view_name=stage_feature_views["level1_sepsis"],
            view_config=feature_view_config,
            stage_name="level1_sepsis",
            output_dir=processing_dir,
        )
        X1_train_base, X1_test_base = apply_stage_correlation_filter(
            X1_train_base, X1_test_base,
            max_corr=args.max_corr,
            stage_name="level1_sepsis",
            output_dir=processing_dir,
        )

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
            selected_features = X1_train_base.columns.tolist()
            l1_rfecv_history = {}
        else:
            print("  Selecting Level 1 features by SHAP importance …")
            selected_features, l1_rfecv_history = shap_rfecv(
                rank_model_l1, X1_train_base, y_sep_train_enc, min_features=1, max_features=args.max_features, scoring="pr_auc",
                output_csv_path=output_dir / "processing" / "l1_rfecv_features.csv",
            )
        l1_features = cap_features(
            selected_features, X1_train_base, y_sep_train_enc,
            args.max_features, args.random_state, args.model_type, rank_model=rank_model_l1
        )
        print(f"  SHAP selection kept {len(l1_features)} features.")

        scaler_l1 = MinMaxScaler()
        X_l1_train = pd.DataFrame(
            scaler_l1.fit_transform(X1_train_base[l1_features]),
            columns=l1_features, index=X_train.index,
        )
        X_l1_test = pd.DataFrame(
            scaler_l1.transform(X1_test_base[l1_features]),
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
        l1_optuna_threshold = l1_threshold
        l1_calibrated_oof = _generate_stratified_calibrated_oof_binary(
            X=X_l1_train,
            y=y_sep_train_enc,
            best_params=l1_params,
            model_type=args.model_type,
            n_splits=args.cv_splits,
            random_state=args.random_state,
            sample_weight=sw_l1,
            method="isotonic",
        )
        l1_threshold = _find_best_macro_f1_threshold(
            y_sep_train_enc, l1_calibrated_oof
        )
        print(
            f"  Optuna threshold: {l1_optuna_threshold:.3f}  |  "
            f"calibrated OOF threshold: {l1_threshold:.3f}"
        )

        # Final Level 1 model trained on all training data, then calibrated
        l1_base = build_binary_model(args.model_type, l1_params.copy())
        l1_model = _fit_stratified_calibrated_ensemble(
            l1_base, X_l1_train, y_sep_train_enc,
            n_splits=min(5, args.cv_splits), random_state=args.random_state,
            method="isotonic", sample_weight=sw_l1,
        )

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

        # save SHAP values for level 1 sepsis model if not skipped
        if not skip_shap_values:
            save_shap_artifacts(
                model=l1_model,
                X_train=X_l1_train,
                X_test=X_l1_test,
                feature_names=l1_features,
                explain_size=len(X_l1_test),
                output_dir=l1_dir / "shap",
                stage_name="level1_sepsis",
                class_names=[str(c) for c in le_sep.classes_],
                random_state=args.random_state,
            )

    # ==================================================================
    # LEVEL 2 – two-stage binary hemo model
    # ==================================================================
    if args.skip_level2:
        print("\n" + "=" * 60)
        print("LEVEL 2: skipped")
        print("=" * 60)
        print("  Skipped Level 2 by flag.")
        l2_rfecv_history = {}
        all_summaries["level2_etiology"] = {"skipped": True}
    else:
        print("\n" + "=" * 60)
        print(
            "LEVEL 2: "
            + ("direct hemo etiology" if args.skip_l2_gate else "two-stage hemo (positive gate -> subtype)")
        )
        print("=" * 60)
        l2_dir = output_dir / "level2_etiology"
        l2_dir.mkdir(exist_ok=True)

        hemo_valid_train = y_hemo_train.notna()
        hemo_valid_test = y_hemo_test.notna()
        if not hemo_valid_train.any():
            raise ValueError("No non-null hemo labels in train.")

        if args.skip_l2_gate:
            # Direct etiology prediction including NEGATIVE as a class.
            X2_direct_train_view, X2_direct_test_view = apply_feature_view(
                X_train, X_test,
                view_name=stage_feature_views["level2_etiology"],
                view_config=feature_view_config,
                stage_name="level2_direct_etiology",
                output_dir=processing_dir,
            )
            X2_train_base = X2_direct_train_view.loc[hemo_valid_train].copy()
            X2_test_base = X2_direct_test_view.loc[hemo_valid_test].copy()
            X2_train_base, X2_test_base = apply_stage_correlation_filter(
                X2_train_base, X2_test_base,
                max_corr=args.max_corr,
                stage_name="level2_direct_etiology",
                output_dir=processing_dir,
            )
            y2_train_raw = y_hemo_train.loc[hemo_valid_train].astype(str)
            y2_test_raw = y_hemo_test.loc[hemo_valid_test].astype(str)

            # Direct etiology uses the same target-mode semantics as gated
            # Stage 2:
            #   binary     -> GNB vs non_GNB
            #   multiclass -> original etiology classes
            #   auto       -> keep original classes; use binary estimator only
            #                 when exactly two classes are present
            direct_mode = args.etiology_stage2_mode
            if direct_mode == "binary":
                print("  Direct etiology mode: binary GNB vs non-GNB")
                y2_train_raw = _map_hemo_subtype_to_gnb_binary(y2_train_raw)
                y2_test_raw = _map_hemo_subtype_to_gnb_binary(y2_test_raw)
            elif direct_mode == "multiclass":
                print("  Direct etiology mode: multiclass (all etiology classes)")
            elif direct_mode == "auto":
                print("  Direct etiology mode: auto (binary if 2 classes else multiclass)")
            else:
                raise ValueError(
                    "Unknown etiology stage-2 mode "
                    f"{direct_mode!r}; expected 'auto', 'binary', or 'multiclass'."
                )

            if direct_mode == "binary" and y2_train_raw.nunique() < 2:
                raise ValueError(
                    "Direct Level 2 binary mode requires both GNB and non_GNB "
                    "labels in the training subset."
                )

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
                l2_optuna_threshold = l2_threshold
                l2_calibrated_oof = _generate_stratified_calibrated_oof_binary(
                    X=X2_train_scaled,
                    y=y2_train,
                    best_params=l2_params,
                    model_type=args.model_type,
                    n_splits=args.cv_splits,
                    random_state=args.random_state,
                    sample_weight=sw_l2,
                    method="isotonic",
                )
                l2_threshold = _find_best_macro_f1_threshold(
                    y2_train, l2_calibrated_oof
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

            l2_model = _fit_stratified_calibrated_ensemble(
                l2_base, X2_train_scaled, y2_train,
                n_splits=min(5, args.cv_splits), random_state=args.random_state,
                method="isotonic", sample_weight=sw_l2,
            )
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

            actual_direct_mode = (
                direct_mode
                if direct_mode != "auto"
                else ("binary" if is_binary_subtype else "multiclass")
            )
            stage2_summary = {
                "mode": "binary" if is_binary_subtype else "multiclass",
                "stage2_mode": actual_direct_mode,
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
            (l2_dir / f"report_{args.etiology_target}.txt").write_text(l2_report_text)
            (l2_dir / "summary.json").write_text(json.dumps({
                "mode": "direct_binary" if is_binary_subtype else "direct_multiclass",
                "stage2_mode": actual_direct_mode,
                "target": args.etiology_target,
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
            else:
                stage_pred_df = metadata_df.reindex(stage_pred_df.index).join(stage_pred_df)
                stage_pred_df.to_csv(
                    l2_dir / "predictions_direct_etiology.csv",
                    index_label="row_index",
                )

            all_summaries["level2_etiology"] = {
                "mode": "direct_binary" if is_binary_subtype else "direct_multiclass",
                "stage2_mode": actual_direct_mode,
                "target": args.etiology_target,
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

            all_summaries["l2_rfecv_scores"] = l2_rfecv_history
            
            # Save SHAP values for direct Level 2 etiology model
            if not skip_shap_values:
                save_shap_artifacts(
                    model=l2_model,
                    X_train=X2_train_scaled,
                    X_test=X2_test_scaled,
                    feature_names=l2_features,
                    explain_size=len(X2_test_scaled),
                    output_dir=l2_dir / "shap",
                    stage_name="level2_direct_etiology",
                    class_names=l2_target_names,
                    random_state=args.random_state,
                )

        else:
            # Stage 1: binary gate using separate gate target (e.g. infected_yes_no)
            hemo_gate_valid_train = y_hemo_gate_train.notna()
            hemo_gate_valid_test = y_hemo_gate_test.notna()
            y2_gate_train = (
                y_hemo_gate_train.loc[hemo_gate_valid_train].astype(str) != args.etiology_gate_negative_label
            ).astype(int)
            gate_test_mask = hemo_gate_valid_test
            y2_gate_test = (
                y_hemo_gate_test.loc[gate_test_mask].astype(str) != args.etiology_gate_negative_label
            ).astype(int)

            X2_gate_train_view, X2_gate_test_view = apply_feature_view(
                X_train, X_test,
                view_name=stage_feature_views["level2_gate"],
                view_config=feature_view_config,
                stage_name="level2_gate",
                output_dir=processing_dir,
            )
            X2_gate_train_base = X2_gate_train_view.loc[hemo_gate_valid_train].copy()
            X2_gate_test_base = X2_gate_test_view.loc[gate_test_mask].copy()
            X2_gate_train_base, X2_gate_test_base = apply_stage_correlation_filter(
                X2_gate_train_base, X2_gate_test_base,
                max_corr=args.max_corr,
                stage_name="level2_gate",
                output_dir=processing_dir,
            )

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
            l2_gate_model = _fit_stratified_calibrated_ensemble(
                l2_gate_base, X2_gate_train_scaled, y2_gate_train,
                n_splits=min(5, args.cv_splits), random_state=args.random_state,
                method="isotonic", sample_weight=sw_l2_gate,
            )

            # Save SHAP values for Level 2 Stage-1 gate model
            if not skip_shap_values:
                save_shap_artifacts(
                    model=l2_gate_model,
                    X_train=X2_gate_train_scaled,
                    X_test=X2_gate_test_scaled,
                    feature_names=l2_gate_features,
                    explain_size=len(X2_gate_test_scaled),
                    output_dir=l2_dir / "shap",
                    stage_name="level2_gate",
                    class_names=["gate_negative", "gate_positive"],
                    random_state=args.random_state,
                )
            l2_gate_train_proba = pd.Series(
                _generate_stratified_calibrated_oof_binary(
                    X=X2_gate_train_scaled,
                    y=y2_gate_train,
                    best_params=l2_gate_params,
                    model_type=args.model_type,
                    n_splits=args.cv_splits,
                    random_state=args.random_state,
                    sample_weight=sw_l2_gate,
                    method="isotonic",
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

            X2_sub_train_view, X2_sub_test_view = apply_feature_view(
                X_train, X_test,
                view_name=stage_feature_views["level2_etiology"],
                view_config=feature_view_config,
                stage_name="level2_etiology",
                output_dir=processing_dir,
            )
            X2_sub_train_base = X2_sub_train_view.loc[subtype_train_mask].copy()
            X2_sub_test_base = X2_sub_test_view.loc[subtype_test_mask].copy()
            if args.l2_gate_proba_as_feature:
                print("  Adding Level 2 gate probability as a feature to Stage 2 subtype modelling.")
                X2_sub_train_base = X2_sub_train_base.join(
                    l2_gate_train_proba.loc[subtype_train_mask], how="left"
                )
                X2_sub_test_base = X2_sub_test_base.join(
                    l2_gate_test_proba.loc[subtype_test_mask], how="left"
                )
            X2_sub_train_base, X2_sub_test_base = apply_stage_correlation_filter(
                X2_sub_train_base, X2_sub_test_base,
                max_corr=args.max_corr,
                stage_name="level2_etiology",
                output_dir=processing_dir,
            )
            y2_sub_train_raw = y_hemo_train.loc[subtype_train_mask].astype(str)
            y2_sub_test_raw = y_hemo_test.loc[subtype_test_mask].astype(str)

            stage2_mode = args.etiology_stage2_mode
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

            y2_sub_test_raw = y2_sub_test_raw.loc[subtype_test_mask].astype(str)
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
                l2_optuna_threshold = l2_threshold
                l2_calibrated_oof = _generate_stratified_calibrated_oof_binary(
                    X=X2_train_scaled,
                    y=y2_sub_train,
                    best_params=l2_params,
                    model_type=args.model_type,
                    n_splits=args.cv_splits,
                    random_state=args.random_state,
                    sample_weight=sw_l2,
                    method="isotonic",
                )
                l2_threshold = _find_best_macro_f1_threshold(
                    y2_sub_train, l2_calibrated_oof
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

            l2_model = _fit_stratified_calibrated_ensemble(
                l2_base, X2_train_scaled, y2_sub_train,
                n_splits=min(5, args.cv_splits), random_state=args.random_state,
                method="isotonic", sample_weight=sw_l2,
            )

            # Save SHAP values for Level 2 Stage-2 subtype model
            if not skip_shap_values:
                save_shap_artifacts(
                    model=l2_model,
                    X_train=X2_train_scaled,
                    X_test=X2_test_scaled,
                    feature_names=l2_features,
                    explain_size=len(X2_test_scaled),
                    output_dir=l2_dir / "shap",
                    stage_name="level2_subtype",
                    class_names=l2_target_names,
                    random_state=args.random_state,
                )
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

            (l2_dir / f"report_gate_{args.etiology_gate_target}.txt").write_text(stage1_report_text)

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

            (l2_dir / f"report_staged_{args.etiology_target}.txt").write_text(stage2_report_text)

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
                    "negative_label": args.etiology_gate_negative_label,
                    "positive_label": f"not_{args.etiology_gate_negative_label}",
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

            actual_stage2_mode = (
                stage2_mode
                if stage2_mode != "auto"
                else ("binary" if is_binary_subtype else "multiclass")
            )
            all_summaries["level2_etiology"] = {
                "mode": "two_stage_binary" if is_binary_subtype else "two_stage_multiclass_positive_gate",
                "stage2_mode": actual_stage2_mode,
                "stage1_gate": {
                    "macro_f1": l2_gate_f1,
                    "roc_auc": l2_gate_auc,
                    "pr_auc": l2_gate_pr_auc,
                    "precision": l2_gate_precision,
                    "recall": l2_gate_recall,
                    "negative_label": args.etiology_gate_negative_label,
                    "positive_label": f"not_{args.etiology_gate_negative_label}",
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
            all_summaries["l2_stage2_rfecv_scores"] = l2_rfecv_history

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
    # Level 3 resistance population mode
    # ------------------------------------------------------------------
    # Supported modes:
    #   gated:
    #       reuse the Level-2 gate. Train resistance on TRUE culture-positive
    #       rows and test only on rows predicted culture-positive by Level 2.
    #   resistance_gate:
    #       train an independent culture-positive/culture-negative gate for
    #       Level 3, then test resistance only on rows predicted positive by
    #       this Level-3 gate. This works even when Level 2 etiology is direct.
    #   direct:
    #       train/test on every row with a non-null resistance target. This is
    #       independent of Level 2 and works with --skip-l2-gate.
    #   true_positive_only:
    #       train/test only on TRUE culture-positive rows, without using gate
    #       predictions. Useful as an oracle/benchmark population.
    #
    # Backward-compatible "auto":
    #   - direct when Level 2 or its gate is skipped
    #   - gated otherwise
    requested_resistance_mode = getattr(args, "resistance_mode", "auto")
    if requested_resistance_mode == "auto":
        resistance_mode = (
            "direct"
            if args.skip_level2 or args.skip_l2_gate
            else "gated"
        )
    else:
        resistance_mode = requested_resistance_mode

    if resistance_mode not in {
        "gated",
        "resistance_gate",
        "direct",
        "true_positive_only",
    }:
        raise ValueError(
            f"Unknown resistance_mode={resistance_mode!r}. "
            "Expected 'auto', 'gated', 'resistance_gate', 'direct', "
            "or 'true_positive_only'."
        )

    print(f"  Level 3 resistance mode: {resistance_mode}")

    # These are populated for gate-based Level-3 modes and later reused by
    # end-to-end cascade accounting.
    resistance_gate_summary: Optional[dict[str, Any]] = None
    cascade_gate_test_pred: Optional[pd.Series] = None
    cascade_y_gate_test: Optional[pd.Series] = None
    cascade_gate_mode: Optional[str] = None
    cascade_gate_proba: Optional[pd.Series] = None

    if resistance_mode == "gated":
        if args.skip_level2 or args.skip_l2_gate:
            raise ValueError(
                "Level 3 resistance_mode='gated' requires Level 2 gate "
                "predictions. Use resistance_mode='direct', "
                "'true_positive_only', or 'resistance_gate' when skipping "
                "the Level-2 gate."
            )

        # TRAIN: true culture-positive rows.
        hemo_pos_train_mask = (
            y_hemo_gate_train.notna()
            & (
                y_hemo_gate_train.astype(str)
                != args.etiology_gate_negative_label
            )
        )

        # TEST: rows predicted culture-positive by Level 2 gate.
        predicted_hemo_positive_idx = l2_gate_test_pred[
            l2_gate_test_pred == 1
        ].index
        hemo_pos_test_mask = (
            y_hemo_gate_test.notna()
            & X_test.index.isin(predicted_hemo_positive_idx)
        )

        cef_valid_train_mask = y_cef_train.notna() & hemo_pos_train_mask
        cef_valid_test_mask = y_cef_test.notna() & hemo_pos_test_mask

        cascade_gate_test_pred = l2_gate_test_pred
        cascade_y_gate_test = y2_gate_test
        cascade_gate_mode = "level2_gate_reused"
        cascade_gate_proba = l2_gate_test_proba

        print(
            f"  Level 3: training on true culture-positive rows "
            f"({int(hemo_pos_train_mask.sum())} samples)"
        )
        print(
            f"  Level 3: testing on Level-2 predicted culture-positive rows "
            f"({int(hemo_pos_test_mask.sum())} samples)"
        )

    elif resistance_mode == "resistance_gate":
        # Independent Level-3 culture gate. This is intentionally separate
        # from the Level-2 etiology workflow, so direct etiology can still be
        # combined with gate-filtered resistance prediction.
        l3_gate_target_train = working_df.loc[X_train.index, args.resistance_gate_target]
        l3_gate_target_test = working_df.loc[X_test.index, args.resistance_gate_target]
        gate_valid_train = l3_gate_target_train.notna()
        gate_valid_test = l3_gate_target_test.notna()

        if not gate_valid_train.any():
            raise ValueError(
                "No non-null Level-3 resistance gate labels in train. "
                f"Check --resistance-gate-target {args.resistance_gate_target!r}."
            )
        if not gate_valid_test.any():
            raise ValueError(
                "No non-null Level-3 resistance gate labels in test. "
                f"Check --resistance-gate-target {args.resistance_gate_target!r}."
            )

        y3_gate_train = (
            l3_gate_target_train.loc[gate_valid_train].astype(str)
            != args.resistance_gate_negative_label
        ).astype(int)
        y3_gate_test = (
            l3_gate_target_test.loc[gate_valid_test].astype(str)
            != args.resistance_gate_negative_label
        ).astype(int)

        X3_gate_train_view, X3_gate_test_view = apply_feature_view(
            X_train,
            X_test,
            view_name=stage_feature_views["level2_gate"],
            view_config=feature_view_config,
            stage_name="level3_resistance_gate",
            output_dir=processing_dir,
        )
        X3_gate_train_base = X3_gate_train_view.loc[gate_valid_train].copy()
        X3_gate_test_base = X3_gate_test_view.loc[gate_valid_test].copy()
        X3_gate_train_base, X3_gate_test_base = apply_stage_correlation_filter(
            X3_gate_train_base,
            X3_gate_test_base,
            max_corr=args.max_corr,
            stage_name="level3_resistance_gate",
            output_dir=processing_dir,
        )

        rank_model_l3_gate = _build_ranking_model(
            args.model_type,
            y3_gate_train,
            args.random_state,
        )
        if args.skip_rfecv:
            print("  Skipping RFECV for Level 3 resistance gate – using all features.")
            l3_gate_shap_feats = X3_gate_train_base.columns.tolist()
            l3_gate_rfecv_history = {}
        else:
            print("  Selecting Level 3 resistance-gate features by SHAP importance …")
            l3_gate_shap_feats, l3_gate_rfecv_history = shap_rfecv(
                rank_model_l3_gate,
                X3_gate_train_base,
                y3_gate_train,
                min_features=1,
                max_features=args.max_features,
                scoring="pr_auc",
                output_csv_path=(
                    output_dir
                    / "processing"
                    / "l3_resistance_gate_rfecv_features.csv"
                ),
            )

        l3_gate_features = cap_features(
            l3_gate_shap_feats,
            X3_gate_train_base,
            y3_gate_train,
            args.max_features,
            args.random_state,
            args.model_type,
            rank_model=rank_model_l3_gate,
        )

        scaler_l3_gate = MinMaxScaler()
        X3_gate_train_scaled = pd.DataFrame(
            scaler_l3_gate.fit_transform(X3_gate_train_base[l3_gate_features]),
            columns=l3_gate_features,
            index=X3_gate_train_base.index,
        )
        X3_gate_test_scaled = pd.DataFrame(
            scaler_l3_gate.transform(X3_gate_test_base[l3_gate_features]),
            columns=l3_gate_features,
            index=X3_gate_test_base.index,
        )

        sw_l3_gate = compute_balanced_sample_weight(
            y3_gate_train,
            w_train.reindex(y3_gate_train.index),
        )
        print("  Optimising independent Level 3 resistance gate model …")
        l3_gate_params, l3_gate_threshold, l3_gate_study = _optimise_binary(
            X=X3_gate_train_scaled,
            y=y3_gate_train,
            n_splits=args.cv_splits,
            n_trials=args.binary_trials,
            random_state=args.random_state,
            sample_weight=sw_l3_gate,
            model_type=args.model_type,
            study_name=_study_name("l3_resistance_gate"),
        )

        l3_gate_base = build_binary_model(
            args.model_type,
            l3_gate_params.copy(),
        )
        l3_gate_model = _fit_stratified_calibrated_ensemble(
            l3_gate_base,
            X3_gate_train_scaled,
            y3_gate_train,
            n_splits=min(5, args.cv_splits),
            random_state=args.random_state,
            method="isotonic",
            sample_weight=sw_l3_gate,
        )

        if not skip_shap_values:
            save_shap_artifacts(
                model=l3_gate_model,
                X_train=X3_gate_train_scaled,
                X_test=X3_gate_test_scaled,
                feature_names=l3_gate_features,
                explain_size=len(X3_gate_test_scaled),
                output_dir=l3_dir / "shap",
                stage_name="level3_resistance_gate",
                class_names=["gate_negative", "gate_positive"],
                random_state=args.random_state,
            )

        l3_gate_train_proba = pd.Series(
            _generate_stratified_calibrated_oof_binary(
                X=X3_gate_train_scaled,
                y=y3_gate_train,
                best_params=l3_gate_params,
                model_type=args.model_type,
                n_splits=args.cv_splits,
                random_state=args.random_state,
                sample_weight=sw_l3_gate,
                method="isotonic",
            ),
            index=X3_gate_train_scaled.index,
            name="l3_resistance_gate_proba",
        )

        l3_gate_optuna_threshold = l3_gate_threshold
        resistance_gate_recall = getattr(
            args,
            "set_resistance_gate_recall",
            None,
        )

        if resistance_gate_recall is not None:
            if not 0.0 <= resistance_gate_recall <= 1.0:
                raise ValueError(
                    "--set-resistance-gate-recall must be between 0 and 1."
                )
            l3_gate_threshold = _find_threshold_for_recall(
                y3_gate_train,
                l3_gate_train_proba,
                target_recall=resistance_gate_recall,
            )
            l3_gate_threshold_strategy = (
                f"target_recall_{resistance_gate_recall}"
            )
        else:
            l3_gate_threshold_strategy = "best_threshold"

        l3_gate_test_proba = pd.Series(
            l3_gate_model.predict_proba(X3_gate_test_scaled)[:, 1],
            index=X3_gate_test_scaled.index,
            name="l3_resistance_gate_proba",
        )
        l3_gate_test_pred = pd.Series(
            (l3_gate_test_proba >= l3_gate_threshold).astype(int),
            index=X3_gate_test_scaled.index,
            name="l3_resistance_gate_pred",
        )
        l3_gate_train_pred = pd.Series(
            (l3_gate_train_proba >= l3_gate_threshold).astype(int),
            index=X3_gate_train_scaled.index,
            name="l3_resistance_gate_pred",
        )

        l3_gate_f1 = f1_score(
            y3_gate_test,
            l3_gate_test_pred,
            average="macro",
            zero_division=0,
        )
        l3_gate_precision = precision_score(
            y3_gate_test,
            l3_gate_test_pred,
            average="macro",
            zero_division=0,
        )
        l3_gate_recall = recall_score(
            y3_gate_test,
            l3_gate_test_pred,
            average="macro",
            zero_division=0,
        )
        l3_gate_auc: Optional[float] = None
        l3_gate_pr_auc: Optional[float] = None
        if y3_gate_test.nunique() > 1:
            l3_gate_auc = float(roc_auc_score(y3_gate_test, l3_gate_test_proba))
            l3_gate_pr_auc = float(
                average_precision_score(y3_gate_test, l3_gate_test_proba)
            )

        l3_gate_confusion = confusion_matrix(
            y3_gate_test,
            l3_gate_test_pred,
            labels=[0, 1],
        ).tolist()
        l3_gate_pred_counts = {
            "gate_negative": int((l3_gate_test_pred == 0).sum()),
            "gate_positive": int((l3_gate_test_pred == 1).sum()),
        }
        l3_gate_true_counts = {
            "gate_negative": int((y3_gate_test == 0).sum()),
            "gate_positive": int((y3_gate_test == 1).sum()),
        }

        l3_gate_report_text = _format_report_with_confusion(
            title="Level 3 Independent Resistance Gate Report",
            report_label="Resistance gate",
            report=classification_report(
                y3_gate_test,
                l3_gate_test_pred,
                target_names=["gate_negative", "gate_positive"],
                zero_division=0,
            ),
            confusion=l3_gate_confusion,
            predicted_counts=l3_gate_pred_counts,
            actual_counts=l3_gate_true_counts,
            binary_labels=("gate_negative", "gate_positive"),
        )
        (l3_dir / f"report_resistance_gate_{args.etiology_gate_target}.txt").write_text(
            l3_gate_report_text
        )
        l3_gate_study.trials_dataframe().to_csv(
            l3_dir / "optuna_trials_resistance_gate.csv",
            index=False,
        )

        l3_gate_pred_df = pd.DataFrame(
            {
                "true": y3_gate_test.values,
                "pred": l3_gate_test_pred.values,
                "proba": l3_gate_test_proba.values,
            },
            index=X3_gate_test_scaled.index,
        )
        l3_gate_pred_df["true_label"] = np.where(
            l3_gate_pred_df["true"] == 1,
            "gate_positive",
            "gate_negative",
        )
        l3_gate_pred_df["pred_label"] = np.where(
            l3_gate_pred_df["pred"] == 1,
            "gate_positive",
            "gate_negative",
        )
        l3_gate_pred_df["error_type"] = np.select(
            [
                (l3_gate_pred_df["true"] == 0)
                & (l3_gate_pred_df["pred"] == 0),
                (l3_gate_pred_df["true"] == 0)
                & (l3_gate_pred_df["pred"] == 1),
                (l3_gate_pred_df["true"] == 1)
                & (l3_gate_pred_df["pred"] == 0),
                (l3_gate_pred_df["true"] == 1)
                & (l3_gate_pred_df["pred"] == 1),
            ],
            ["TN", "FP", "FN", "TP"],
            default="NA",
        )
        l3_gate_pred_df["margin_from_threshold"] = (
            l3_gate_pred_df["proba"] - l3_gate_threshold
        )
        l3_gate_pred_df["abs_margin_from_threshold"] = (
            l3_gate_pred_df["margin_from_threshold"].abs()
        )
        l3_gate_pred_df = metadata_df.reindex(l3_gate_pred_df.index).join(
            l3_gate_pred_df
        )
        l3_gate_pred_df.to_csv(
            l3_dir / "predictions_resistance_gate.csv",
            index_label="row_index",
        )

        hemo_pos_train_mask = (
            y_hemo_gate_train.notna()
            & (
                y_hemo_gate_train.astype(str)
                != args.etiology_gate_negative_label
            )
        )
        hemo_pos_test_mask = (
            y_hemo_gate_test.notna()
            & X_test.index.isin(l3_gate_test_pred[l3_gate_test_pred == 1].index)
        )

        cef_valid_train_mask = y_cef_train.notna() & hemo_pos_train_mask
        cef_valid_test_mask = y_cef_test.notna() & hemo_pos_test_mask

        resistance_gate_summary = {
            "negative_label": args.resistance_gate_negative_label,
            "positive_label": f"not_{args.resistance_gate_negative_label}",
            "params": l3_gate_params,
            "optuna_threshold": l3_gate_optuna_threshold,
            "threshold": l3_gate_threshold,
            "threshold_strategy": l3_gate_threshold_strategy,
            "macro_f1": l3_gate_f1,
            "roc_auc": l3_gate_auc,
            "pr_auc": l3_gate_pr_auc,
            "precision": l3_gate_precision,
            "recall": l3_gate_recall,
            "predicted_true": l3_gate_pred_counts["gate_positive"],
            "predicted_false": l3_gate_pred_counts["gate_negative"],
            "real_true": l3_gate_true_counts["gate_positive"],
            "real_false": l3_gate_true_counts["gate_negative"],
            "confusion_matrix": l3_gate_confusion,
            "shaprfecv_features": l3_gate_shap_feats,
            "features": l3_gate_features,
        }

        cascade_gate_test_pred = l3_gate_test_pred
        cascade_y_gate_test = y3_gate_test
        cascade_gate_mode = "level3_resistance_gate"
        cascade_gate_proba = l3_gate_test_proba

        print(
            f"  Level 3 resistance gate – Macro F1: {l3_gate_f1:.3f}"
            + (f"  |  ROC-AUC: {l3_gate_auc:.3f}" if l3_gate_auc is not None else "")
            + (f"  |  PR-AUC: {l3_gate_pr_auc:.3f}" if l3_gate_pr_auc is not None else "")
            + f"  |  Precision: {l3_gate_precision:.3f}  |  Recall: {l3_gate_recall:.3f}"
        )
        print(
            f"  Level 3: training resistance on true culture-positive rows "
            f"({int(cef_valid_train_mask.sum())} valid resistance targets)"
        )
        print(
            f"  Level 3: testing resistance on independent-gate predicted "
            f"culture-positive rows ({int(cef_valid_test_mask.sum())} "
            "valid resistance targets)"
        )

    elif resistance_mode == "direct":
        # Fully independent direct resistance prediction.
        cef_valid_train_mask = y_cef_train.notna()
        cef_valid_test_mask = y_cef_test.notna()

        print(
            f"  Level 3: direct resistance training on all rows with valid "
            f"target ({int(cef_valid_train_mask.sum())} samples)"
        )
        print(
            f"  Level 3: direct resistance testing on all rows with valid "
            f"target ({int(cef_valid_test_mask.sum())} samples)"
        )

    else:  # true_positive_only
        hemo_pos_train_mask = (
            y_hemo_gate_train.notna()
            & (
                y_hemo_gate_train.astype(str)
                != args.etiology_gate_negative_label
            )
        )
        hemo_pos_test_mask = (
            y_hemo_gate_test.notna()
            & (
                y_hemo_gate_test.astype(str)
                != args.etiology_gate_negative_label
            )
        )

        cef_valid_train_mask = y_cef_train.notna() & hemo_pos_train_mask
        cef_valid_test_mask = y_cef_test.notna() & hemo_pos_test_mask

        print(
            f"  Level 3: training on true culture-positive rows "
            f"({int(cef_valid_train_mask.sum())} valid resistance targets)"
        )
        print(
            f"  Level 3: testing on true culture-positive rows "
            f"({int(cef_valid_test_mask.sum())} valid resistance targets)"
        )

    if not cef_valid_train_mask.any():
        raise ValueError(
            f"No Level-3 training rows remain in resistance mode "
            f"{resistance_mode!r} with a valid --cef-target."
        )

    if not cef_valid_test_mask.any():
        raise ValueError(
            f"No Level-3 test rows remain in resistance mode "
            f"{resistance_mode!r} with a valid --cef-target."
        )

    # Direct Level-3 resistance model (no L3 binary gate):
    # train/predict resistance target only on positive hemoculture rows.
    X3_train_view, X3_test_view = apply_feature_view(
        X_train, X_test,
        view_name=stage_feature_views["level3_resistance"],
        view_config=feature_view_config,
        stage_name="level3_resistance",
        output_dir=processing_dir,
    )
    X3_train_base = X3_train_view.loc[cef_valid_train_mask].copy()
    X3_test_base = X3_test_view.loc[cef_valid_test_mask].copy()
    X3_train_base, X3_test_base = apply_stage_correlation_filter(
        X3_train_base, X3_test_base,
        max_corr=args.max_corr,
        stage_name="level3_resistance",
        output_dir=processing_dir,
    )
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
        l3_optuna_threshold = l3_threshold
        l3_calibrated_oof = _generate_stratified_calibrated_oof_binary(
            X=X3_train_scaled,
            y=y3_train_enc,
            best_params=l3_params,
            model_type=args.model_type,
            n_splits=args.cv_splits,
            random_state=args.random_state,
            sample_weight=sw_l3,
            method="isotonic",
        )
        l3_threshold = _find_best_macro_f1_threshold(
            y3_train_enc, l3_calibrated_oof
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

    l3_model = _fit_stratified_calibrated_ensemble(
        l3_base, X3_train_scaled, y3_train_enc,
        n_splits=min(5, args.cv_splits), random_state=args.random_state,
        method="isotonic", sample_weight=sw_l3,
    )

    # Save SHAP values for Level 3 cefalosporin model
    if not skip_shap_values:
        save_shap_artifacts(
            model=l3_model,
            X_train=X3_train_scaled,
            X_test=X3_test_scaled,
            feature_names=l3_features,
            explain_size=len(X3_test_scaled),
            output_dir=l3_dir / "shap",
            stage_name="level3_cefalosporina",
            class_names=l3_target_names,
            random_state=args.random_state,
        )
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
    l3_summary_payload = {
        "mode": "direct_binary" if is_l3_binary else "direct_multiclass",
        "resistance_mode": resistance_mode,
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
        "hemo_positive_filter": resistance_mode != "direct",
    }
    if resistance_gate_summary is not None:
        l3_summary_payload["resistance_gate"] = resistance_gate_summary
    (l3_dir / "summary.json").write_text(json.dumps(l3_summary_payload, indent=2))
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
        "resistance_mode": resistance_mode,
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
    if resistance_gate_summary is not None:
        all_summaries["level3_cefalosporina"]["resistance_gate"] = (
            resistance_gate_summary
        )

    all_summaries["l1_rfecv_scores"] = l1_rfecv_history
    all_summaries["l2_rfecv_scores"] = l2_rfecv_history
    all_summaries["l3_rfecv_scores"] = l3_rfecv_history

    # End-to-end cascade accounting is meaningful only when Level 3 uses a
    # predicted culture-positive gate, either reused from Level 2 or trained
    # independently for resistance.
    if resistance_mode in {"gated", "resistance_gate"}:
        if cascade_gate_test_pred is None or cascade_y_gate_test is None:
            raise RuntimeError(
                "Internal error: gate-based Level 3 mode did not populate "
                "cascade gate predictions."
            )

        gate_eval_index = cascade_y_gate_test.index
        true_gate_positive = cascade_y_gate_test.reindex(gate_eval_index).eq(1)
        predicted_gate_positive = (
            cascade_gate_test_pred.reindex(gate_eval_index).eq(1)
        )
        cef_available = y_cef_test.reindex(gate_eval_index).notna()
        resistant_label = None
        resistant_detected_overall = None
        resistant_total_overall = None

        if is_l3_binary and len(l3_target_names) == 2:
            resistant_label = l3_target_names[1]
            true_resistant = (
                y_cef_test.reindex(gate_eval_index)
                .astype(str)
                .eq(resistant_label)
                & cef_available
            )
            predicted_resistant = pd.Series(False, index=gate_eval_index)
            predicted_resistant.loc[l3_pred_df.index] = (
                l3_pred_df["pred"].eq(1).values
            )
            resistant_detected_overall = int(
                (true_resistant & predicted_resistant).sum()
            )
            resistant_total_overall = int(true_resistant.sum())

        all_summaries["end_to_end_cascade"] = {
            "mode": resistance_mode,
            "gate_mode": cascade_gate_mode,
            "gate_evaluable_admissions": int(len(gate_eval_index)),
            "true_gate_positive": int(true_gate_positive.sum()),
            "predicted_gate_positive": int(predicted_gate_positive.sum()),
            "true_gate_positive_lost": int(
                (true_gate_positive & ~predicted_gate_positive).sum()
            ),
            "gate_false_positives": int(
                (~true_gate_positive & predicted_gate_positive).sum()
            ),
            "resistance_reference_available_after_gate": int(
                (predicted_gate_positive & cef_available).sum()
            ),
            "resistant_label": resistant_label,
            "resistant_cases_detected_end_to_end": resistant_detected_overall,
            "resistant_cases_total_with_reference": resistant_total_overall,
            "end_to_end_resistant_recall": (
                resistant_detected_overall / resistant_total_overall
                if resistant_total_overall not in (None, 0)
                else None
            ),
        }
    else:
        all_summaries["end_to_end_cascade"] = {
            "mode": resistance_mode,
            "skipped": True,
            "reason": (
                "End-to-end gate cascade metrics are not applicable because "
                "Level 3 resistance did not use predicted gate-positive rows."
            ),
        }

    # ------------------------------------------------------------------
    # Aggregate summary
    # ------------------------------------------------------------------
    (output_dir / "aggregate_summary.json").write_text(
        json.dumps(all_summaries, indent=2)
    )

    final_report = {
        "level1_sepsis": all_summaries.get("level1_sepsis"),
        "level2_etiology": all_summaries.get("level2_etiology"),
        "level3_cefalosporina": all_summaries.get("level3_cefalosporina"),
    }
    (output_dir / "final_report.json").write_text(json.dumps(final_report, indent=2))

    print("\n" + "=" * 60)
    print("COMPLETED.  Results saved in:", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------   