"""Benchmark runner for train-fitted model pipelines."""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from .feature_views import resolve_feature_view
from .reporting import job_output_dir, write_diagnostics_manifest, write_job_summary
from .resources import configure_resources
from .types import BenchmarkConfig, BenchmarkJob, DiagnosticResult, PipelineBuildResult


SUPPORTED_TRAINING_MODELS = {
    "dummy_uniform",
    "dummy_stratified",
    "dummy_most_frequent",
    "logistic",
    "catboost",
}


def get_output_dir(config: BenchmarkConfig) -> Path:
    return Path(config.raw.get("output_dir", "outputs/ml_benchmark"))


def build_pipeline_contract(
    config: BenchmarkConfig,
    job: BenchmarkJob,
    feature_frame: Any,
) -> PipelineBuildResult:
    from .models import get_model_registry
    from .pipeline import build_benchmark_pipeline, infer_column_types

    registry = get_model_registry()
    if job.model_name not in registry:
        raise ValueError(f"Unknown model '{job.model_name}'.")
    feature_frame = _apply_policy_drops(feature_frame, job)
    numeric_columns, categorical_columns = infer_column_types(feature_frame)
    return build_benchmark_pipeline(
        job=job,
        model_spec=registry[job.model_name],
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        random_state=config.split.random_state,
        n_jobs=configure_resources(
            int(config.raw.get("compute", {}).get("default_n_jobs", 1))
        ).n_jobs,
    )


def run_job(config: BenchmarkConfig, job: BenchmarkJob, *, dry_run: bool = False) -> Dict[str, Any]:
    resources = configure_resources(
        int(config.raw.get("compute", {}).get("default_n_jobs", 1))
    )
    output_dir = job_output_dir(get_output_dir(config), job)
    if dry_run:
        diagnostics = [
            DiagnosticResult(
                name="job_contract",
                status="planned",
                message="Dry run only; no model was fitted.",
            )
        ]
        write_diagnostics_manifest(output_dir, diagnostics)
        write_job_summary(
            output_dir,
            job=job,
            status="planned",
            resources=resources,
            diagnostic_results=diagnostics,
        )
        return {"job": job.slug, "status": "planned", "output_dir": str(output_dir)}

    if job.model_name not in SUPPORTED_TRAINING_MODELS:
        diagnostics = [
            DiagnosticResult(
                name="training",
                status="not_implemented",
                message=(
                    "Training is currently implemented for dummy, logistic, and "
                    f"CatBoost jobs only. Received model='{job.model_name}'."
                ),
            )
        ]
        write_diagnostics_manifest(output_dir, diagnostics)
        write_job_summary(
            output_dir,
            job=job,
            status="not_implemented",
            resources=resources,
            diagnostic_results=diagnostics,
        )
        return {
            "job": job.slug,
            "status": "not_implemented",
            "output_dir": str(output_dir),
        }

    prepared = _prepare_job_data(config, job)
    build_result = build_pipeline_contract(config, job, prepared["X"])
    split_data = _split_train_test(config, prepared)
    cv_results, predictions = _cross_validate_job(
        config=config,
        job=job,
        pipeline=build_result.pipeline,
        X=split_data["X_train"],
        y=split_data["y_train"],
        sample_weight=split_data["sample_weight_train"],
    )
    validation_metrics = _aggregate_metrics(
        y_true=predictions["y_true"],
        y_pred=predictions["y_pred"],
        proba=_prediction_probabilities(predictions),
        classes=split_data["classes"],
        task_type=job.target.task_type,
        positive_label=job.target.positive_label,
    )
    test_metrics, test_predictions = _fit_and_evaluate_test(
        job=job,
        pipeline=build_result.pipeline,
        X_train=split_data["X_train"],
        y_train=split_data["y_train"],
        X_test=split_data["X_test"],
        y_test=split_data["y_test"],
        sample_weight_train=split_data["sample_weight_train"],
    )
    metrics = {
        "validation_cv": validation_metrics,
        "test": test_metrics,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(output_dir / "cv_results.csv", index=False)
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    diagnostics = [
        DiagnosticResult(
            name="training",
            status="completed",
            files=[
                "cv_results.csv",
                "validation_predictions.csv",
                "test_predictions.csv",
            ],
            message=(
                f"Held out {len(split_data['X_test'])} test rows, then completed "
                f"{config.split.cv_splits}-fold cross-validation on "
                f"{len(split_data['X_train'])} training rows and "
                f"{prepared['X'].shape[1]} selected input columns."
            ),
            metadata={
                "pipeline_notes": build_result.notes,
                "numeric_columns": len(build_result.numeric_columns),
                "categorical_columns": len(build_result.categorical_columns),
                "feature_view_selected_columns": len(prepared["feature_columns"]),
                "dropped_missing_target_rows": prepared["dropped_missing_target_rows"],
                "train_rows": len(split_data["X_train"]),
                "test_rows": len(split_data["X_test"]),
            },
        )
    ]
    write_diagnostics_manifest(output_dir, diagnostics)
    write_job_summary(
        output_dir,
        job=job,
        status="completed",
        resources=resources,
        metrics=metrics,
        diagnostic_results=diagnostics,
        extra={
            "n_rows": len(prepared["X"]),
            "train_rows": len(split_data["X_train"]),
            "test_rows": len(split_data["X_test"]),
            "n_features": prepared["X"].shape[1],
            "classes": [str(value) for value in prepared["classes"]],
            "main_metric": job.target.main_metric,
            "feature_view_resolution": {
                "selected_columns": prepared["feature_columns"],
                "excluded_columns": prepared["feature_view_resolution"].excluded_columns,
                "included_by_group": prepared[
                    "feature_view_resolution"
                ].included_by_group,
                "excluded_by_group": prepared[
                    "feature_view_resolution"
                ].excluded_by_group,
            },
        },
    )
    return {
        "job": job.slug,
        "status": "completed",
        "metrics": metrics,
        "output_dir": str(output_dir),
    }


def _split_train_test(
    config: BenchmarkConfig, prepared: Dict[str, Any]
) -> Dict[str, Any]:
    X = prepared["X"]
    y = prepared["y"]
    sample_weight = prepared["sample_weight"]
    indices = np.arange(len(X))
    stratify = y if config.split.stratify and y.value_counts().min() >= 2 else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=config.split.test_size,
        random_state=config.split.random_state,
        shuffle=True,
        stratify=stratify,
    )
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)
    sample_weight_train = (
        None if sample_weight is None else sample_weight.iloc[train_idx]
    )
    sample_weight_test = None if sample_weight is None else sample_weight.iloc[test_idx]
    return {
        "X_train": X.iloc[train_idx],
        "X_test": X.iloc[test_idx],
        "y_train": y.iloc[train_idx],
        "y_test": y.iloc[test_idx],
        "sample_weight_train": sample_weight_train,
        "sample_weight_test": sample_weight_test,
        "classes": sorted(pd.Series(y.iloc[train_idx]).dropna().unique().tolist(), key=str),
    }


def _prepare_job_data(config: BenchmarkConfig, job: BenchmarkJob) -> Dict[str, Any]:
    data_path = Path(config.raw.get("data", {}).get("path", ""))
    if not data_path:
        raise ValueError("Benchmark config must define data.path.")
    if not data_path.is_absolute():
        data_path = config.path.parent / data_path
        if not data_path.exists():
            data_path = Path(config.raw.get("data", {}).get("path", ""))
    df = pd.read_csv(data_path)

    target_columns = _target_columns(config, job)
    target_column = target_columns[0]
    missing_targets = df[target_column].isna()
    dropped_missing_target_rows = int(missing_targets.sum())
    if dropped_missing_target_rows:
        df = df.loc[~missing_targets].copy()

    y = df[target_column]
    reserved_columns = set(_all_target_columns(config))
    reserved_columns.update(target_columns)
    weight_column = config.raw.get("data", {}).get("weight_column")
    sample_weight = None
    if weight_column and weight_column in df.columns:
        sample_weight = df[weight_column]
        reserved_columns.add(weight_column)

    candidate_columns = [column for column in df.columns if column not in reserved_columns]
    resolution = resolve_feature_view(
        candidate_columns,
        job.feature_view,
        config.feature_groups,
    )
    feature_columns = list(resolution.selected_columns)
    X = _apply_policy_drops(df[feature_columns], job)
    feature_columns = list(X.columns)
    if X.empty:
        raise ValueError(
            f"Feature view '{job.feature_view.name}' produced no columns for job "
            f"'{job.slug}'."
        )

    return {
        "X": X,
        "y": y,
        "sample_weight": sample_weight,
        "classes": sorted(pd.Series(y).dropna().unique().tolist(), key=str),
        "feature_columns": feature_columns,
        "feature_view_resolution": resolution,
        "dropped_missing_target_rows": dropped_missing_target_rows,
    }


def _target_columns(config: BenchmarkConfig, job: BenchmarkJob) -> list[str]:
    columns = list(job.target.columns) if job.target.columns else [job.target.name]
    missing = [column for column in columns if column not in _data_columns(config)]
    if missing:
        raise ValueError(
            f"Target '{job.target.name}' references missing columns: {missing}"
        )
    if len(columns) != 1:
        raise NotImplementedError(
            "This benchmark runner currently supports one target column per job."
        )
    return columns


def _all_target_columns(config: BenchmarkConfig) -> list[str]:
    columns: list[str] = []
    for target in config.targets:
        columns.extend(target.columns or [target.name])
    columns.extend(config.raw.get("data", {}).get("target_like_columns", []))
    return columns


def _data_columns(config: BenchmarkConfig) -> list[str]:
    data_path = Path(config.raw.get("data", {}).get("path", ""))
    if not data_path.is_absolute():
        candidate = config.path.parent / data_path
        data_path = candidate if candidate.exists() else data_path
    return pd.read_csv(data_path, nrows=0).columns.tolist()


def _apply_policy_drops(X: pd.DataFrame, job: BenchmarkJob) -> pd.DataFrame:
    drop_columns = set(job.feature_policy.drop_columns)
    for pattern in job.feature_policy.drop_patterns:
        drop_columns.update(column for column in X.columns if fnmatch(column, pattern))
    kept = [column for column in X.columns if column not in drop_columns]
    return X[kept]


def _cross_validate_job(
    *,
    config: BenchmarkConfig,
    job: BenchmarkJob,
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = _build_splitter(config, y)
    fold_rows = []
    prediction_frames = []

    for fold, (train_idx, validation_idx) in enumerate(splitter.split(X, y)):
        estimator = clone(pipeline)
        fit_params = {}
        if sample_weight is not None:
            fit_params["model__sample_weight"] = sample_weight.iloc[train_idx]
        # sklearn Pipeline.fit calls fit/fit_transform on every preprocessing
        # step using only the fold-training rows, then fits the final model.
        estimator.fit(X.iloc[train_idx], y.iloc[train_idx], **fit_params)

        y_true = y.iloc[validation_idx].reset_index(drop=True)
        # Pipeline.predict calls transform on the fitted preprocessing steps for
        # the validation rows. No imputer/qcut/scaler/category mapping is refit.
        y_pred = pd.Series(
            np.asarray(estimator.predict(X.iloc[validation_idx])).ravel()
        ).reset_index(drop=True)
        proba = _predict_proba(estimator, X.iloc[validation_idx])
        classes = _estimator_classes(estimator)
        fold_metrics = _aggregate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            proba=proba,
            classes=classes,
            task_type=job.target.task_type,
            positive_label=job.target.positive_label,
        )
        fold_metrics.update(
            {
                "fold": fold,
                "train_rows": int(len(train_idx)),
                "validation_rows": int(len(validation_idx)),
            }
        )
        fold_rows.append(fold_metrics)

        pred_frame = pd.DataFrame(
            {
                "fold": fold,
                "row_index": X.index[validation_idx],
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )
        if proba is not None:
            for class_idx, class_value in enumerate(classes):
                pred_frame[f"proba_{class_value}"] = proba[:, class_idx]
        prediction_frames.append(pred_frame)

    return pd.DataFrame(fold_rows), pd.concat(prediction_frames, ignore_index=True)


def _fit_and_evaluate_test(
    *,
    job: BenchmarkJob,
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sample_weight_train: pd.Series | None,
) -> tuple[Dict[str, Any], pd.DataFrame]:
    estimator = clone(pipeline)
    fit_params = {}
    if sample_weight_train is not None:
        fit_params["model__sample_weight"] = sample_weight_train
    # Fit the whole pipeline once on the full training subset. The held-out test
    # set is not seen while preprocessing statistics or model parameters are fit.
    estimator.fit(X_train, y_train, **fit_params)

    # This prediction path applies Pipeline.transform to X_test before the model
    # predicts, reusing preprocessing learned from X_train.
    y_pred = pd.Series(np.asarray(estimator.predict(X_test)).ravel(), index=X_test.index)
    proba = _predict_proba(estimator, X_test)
    classes = _estimator_classes(estimator)
    metrics = _aggregate_metrics(
        y_true=y_test,
        y_pred=y_pred,
        proba=proba,
        classes=classes,
        task_type=job.target.task_type,
        positive_label=job.target.positive_label,
    )
    predictions = pd.DataFrame(
        {
            "row_index": X_test.index,
            "y_true": y_test.to_numpy(),
            "y_pred": y_pred.to_numpy(),
        }
    )
    if proba is not None:
        for class_idx, class_value in enumerate(classes):
            predictions[f"proba_{class_value}"] = proba[:, class_idx]
    return metrics, predictions


def _build_splitter(config: BenchmarkConfig, y: pd.Series):
    if config.split.strategy != "cross_validation":
        raise NotImplementedError(
            "Only split.strategy='cross_validation' is implemented in this slice."
        )
    n_splits = min(config.split.cv_splits, int(y.value_counts().min()))
    if n_splits < 2:
        raise ValueError("Need at least two rows per class for stratified CV.")
    if config.split.stratify:
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=config.split.random_state,
        )
    return KFold(
        n_splits=min(config.split.cv_splits, len(y)),
        shuffle=True,
        random_state=config.split.random_state,
    )


def _aggregate_metrics(
    *,
    y_true,
    y_pred,
    proba: np.ndarray | None,
    classes: list[Any],
    task_type: str,
    positive_label: Any | None,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "accuracy": _safe_metric(accuracy_score, y_true, y_pred),
        "balanced_accuracy": _safe_metric(balanced_accuracy_score, y_true, y_pred),
        "f1_macro": _safe_metric(f1_score, y_true, y_pred, average="macro"),
        "f1_weighted": _safe_metric(f1_score, y_true, y_pred, average="weighted"),
    }
    if proba is None:
        return metrics

    if task_type == "binary" and len(classes) == 2:
        positive = positive_label if positive_label is not None else classes[-1]
        positive_idx = classes.index(positive) if positive in classes else 1
        y_score = proba[:, positive_idx]
        metrics["roc_auc"] = _safe_metric(roc_auc_score, y_true, y_score)
        metrics["average_precision"] = _safe_metric(
            average_precision_score,
            pd.Series(y_true).map(lambda value: 1 if value == positive else 0),
            y_score,
        )
        metrics["log_loss"] = _safe_metric(log_loss, y_true, proba, labels=classes)
    elif task_type == "multiclass" and len(classes) > 2:
        metrics["roc_auc_ovr"] = _safe_metric(
            roc_auc_score,
            y_true,
            proba,
            labels=classes,
            multi_class="ovr",
            average="macro",
        )
        metrics["log_loss"] = _safe_metric(log_loss, y_true, proba, labels=classes)
    return metrics


def _safe_metric(func, *args, **kwargs):
    try:
        value = func(*args, **kwargs)
    except Exception:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _predict_proba(estimator, X: pd.DataFrame) -> np.ndarray | None:
    if not hasattr(estimator, "predict_proba"):
        return None
    try:
        return estimator.predict_proba(X)
    except Exception:
        return None


def _estimator_classes(estimator) -> list[Any]:
    if hasattr(estimator, "classes_"):
        return list(estimator.classes_)
    model = getattr(estimator, "named_steps", {}).get("model")
    if model is not None and hasattr(model, "classes_"):
        return list(model.classes_)
    return []


def _prediction_probabilities(predictions: pd.DataFrame) -> np.ndarray | None:
    proba_columns = [column for column in predictions.columns if column.startswith("proba_")]
    if not proba_columns:
        return None
    return predictions[proba_columns].to_numpy()
