"""Benchmark runner for train-fitted model pipelines."""

from __future__ import annotations

import json
import shutil
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
    "dummy_prior",
    "logistic",
    "logistic_calibrated",
    "catboost",
    "catboost_calibrated",
    "lightgbm",
    "lightgbm_calibrated",
}


def get_output_dir(config: BenchmarkConfig) -> Path:
    return Path(config.raw.get("output_dir", "outputs/ml_benchmark"))


def _feature_selection_cache_dir(config: BenchmarkConfig) -> str:
    return str(get_output_dir(config) / "feature_selection_cache")


def _feature_selection_cache_key(job: BenchmarkJob) -> str:
    # Deliberately excludes feature_set.name/max_features so top-N caps reuse
    # the same train-fitted SHAP-RFECV ranking.
    return "__".join(
        [
            job.target.name,
            job.model_name,
            job.feature_policy.name,
            job.feature_view.name,
            "shap_rfecv",
        ]
    )


def _write_feature_selection_cache_index(base_output_dir: Path) -> Path | None:
    cache_dir = base_output_dir / "feature_selection_cache"
    cache_files = sorted(cache_dir.glob("*.json"))
    if not cache_files:
        return None

    rows = []
    for path in cache_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            rows.append({"cache_file": str(path), "status": "invalid_json"})
            continue
        metadata = payload.get("metadata", {})
        rows.append(
            {
                "cache_file": str(path),
                "status": "ok",
                "ranking_feature_count": len(payload.get("ranking_features", [])),
                "history_rounds": len(payload.get("history", [])),
                "best_score": payload.get("best_score"),
                "best_feature_count": payload.get("best_feature_count"),
                "input_feature_count": metadata.get("input_feature_count"),
                "selector_rows": metadata.get("selector_rows"),
                "cv_splits": metadata.get("cv_splits"),
                "step_fraction": metadata.get("step_fraction"),
                "min_features_to_select": metadata.get("min_features_to_select"),
                "max_features_cap": metadata.get("max_features_cap"),
                "max_shap_rows": metadata.get("max_shap_rows"),
                "max_selector_rows": metadata.get("max_selector_rows"),
                "selector_estimator": metadata.get("selector_estimator"),
                "selector_estimator_params": json.dumps(
                    metadata.get("selector_estimator_params", {}),
                    sort_keys=True,
                ),
            }
        )
    path = base_output_dir / "feature_selection_cache_index.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def build_pipeline_contract(
    config: BenchmarkConfig,
    job: BenchmarkJob,
    feature_frame: Any,
    model_params: dict[str, Any] | None = None,
) -> PipelineBuildResult:
    from .models import get_model_registry
    from .pipeline import build_benchmark_pipeline, infer_column_types

    registry = get_model_registry()
    if job.model_name not in registry:
        raise ValueError(f"Unknown model '{job.model_name}'.")
    feature_frame = _apply_policy_drops(feature_frame, job)
    numeric_columns, categorical_columns = infer_column_types(feature_frame)
    shap_rfecv = config.feature_selection.shap_rfecv
    return build_benchmark_pipeline(
        job=job,
        model_spec=registry[job.model_name],
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        random_state=config.split.random_state,
        n_jobs=configure_resources(
            int(config.raw.get("compute", {}).get("default_n_jobs", 1))
        ).n_jobs,
        model_params=model_params,
        feature_selection=shap_rfecv,
        feature_selection_estimator_params=_selector_params_for_model(
            shap_rfecv.selector_estimator_params,
            job.model_name,
        ),
        feature_selection_cache_dir=(
            _feature_selection_cache_dir(config)
            if config.feature_selection.cache_enabled
            else None
        ),
        feature_selection_cache_key=_feature_selection_cache_key(job),
    )


def _selector_params_for_model(
    selector_estimator_params: dict[str, dict[str, Any]],
    model_name: str,
) -> dict[str, Any]:
    if model_name in selector_estimator_params:
        return selector_estimator_params[model_name]
    base_name = model_name.removesuffix("_calibrated")
    return selector_estimator_params.get(base_name, {})


def _tune_hyperparameters(
    config: BenchmarkConfig,
    job: BenchmarkJob,
    prepared: Dict[str, Any],
    split_data: Dict[str, Any],
) -> Dict[str, Any]:
    tuning = config.tuning
    output_dir = job_output_dir(get_output_dir(config), job)
    base_result = {
        "best_params": {},
        "files": [],
        "summary": {
            "enabled": tuning.enabled,
            "status": "disabled",
            "best_params": {},
        },
    }
    if not tuning.enabled:
        return base_result
    if tuning.models and job.model_name not in set(tuning.models):
        base_result["summary"].update(
            {"status": "skipped_model", "configured_models": tuning.models}
        )
        return base_result

    from .models import get_model_registry

    model_spec = get_model_registry()[job.model_name]
    if model_spec.suggest_params is None:
        base_result["summary"].update({"status": "no_search_space"})
        return base_result

    output_dir.mkdir(parents=True, exist_ok=True)
    best_params_path = output_dir / "best_params.json"
    trials_path = output_dir / "tuning_trials.csv"
    study_path = output_dir / "optuna_study.db"
    legacy_study_path = get_output_dir(config) / "optuna_studies" / f"{job.slug}.db"
    if legacy_study_path.exists() and not study_path.exists():
        shutil.copy2(legacy_study_path, study_path)
    report_paths = _optuna_report_paths(output_dir)
    metric = _tuning_metric(config, job)
    direction = _tuning_direction(config, metric)

    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna tuning is enabled, but optuna is not installed."
        ) from exc

    if tuning.reuse_existing and best_params_path.exists():
        best_payload = json.loads(best_params_path.read_text(encoding="utf-8"))
        report_files = []
        report_status = {}
        if study_path.exists():
            study = optuna.load_study(
                study_name=job.slug,
                storage=f"sqlite:///{study_path}",
            )
            report_files, report_status = _write_optuna_html_reports(
                study,
                report_paths,
            )
        return {
            "best_params": best_payload.get("best_params", {}),
            "files": [str(best_params_path), str(trials_path), *report_files],
            "summary": {
                "enabled": True,
                "status": "cached_result",
                "metric": best_payload.get("metric", metric),
                "direction": best_payload.get("direction", direction),
                "best_value": best_payload.get("best_value"),
                "best_params": best_payload.get("best_params", {}),
                "study_file": str(study_path),
                "best_params_file": str(best_params_path),
                "trials_file": str(trials_path),
                "html_reports": report_files,
                "html_report_status": report_status,
                "validation_scope": best_payload.get(
                    "validation_scope",
                    "training_subset_only",
                ),
                "held_out_test_used": best_payload.get("held_out_test_used", False),
            },
        }

    storage = f"sqlite:///{study_path}" if tuning.storage == "sqlite" else None
    study = optuna.create_study(
        study_name=job.slug,
        direction=direction,
        storage=storage,
        load_if_exists=True,
    )
    completed_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    remaining_trials = max(0, tuning.n_trials - len(completed_trials))

    def objective(trial):
        params = model_spec.suggest_params(trial, job.target.task_type)
        pipeline = build_pipeline_contract(config, job, prepared["X"], params).pipeline
        return _tuning_cv_score(
            config=config,
            job=job,
            pipeline=pipeline,
            X=split_data["X_train"],
            y=split_data["y_train"],
            sample_weight=split_data["sample_weight_train"],
            metric=metric,
        )

    if remaining_trials:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            timeout=tuning.timeout_seconds,
            show_progress_bar=False,
        )

    trials = study.trials_dataframe(attrs=("number", "value", "state", "params"))
    trials.to_csv(trials_path, index=False)
    report_files, report_status = _write_optuna_html_reports(study, report_paths)
    best_payload = {
        "metric": metric,
        "direction": direction,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "study_name": study.study_name,
        "study_file": str(study_path),
        "n_trials_requested": tuning.n_trials,
        "n_trials_total": len(study.trials),
        "cv_splits": tuning.cv_splits,
        "validation_scope": "training_subset_only",
        "held_out_test_used": False,
        "html_reports": report_files,
        "html_report_status": report_status,
    }
    best_params_path.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
    return {
        "best_params": study.best_params,
        "files": [str(best_params_path), str(trials_path), *report_files],
        "summary": {
            "enabled": True,
            "status": "completed" if remaining_trials else "cached_study",
            **best_payload,
            "best_params_file": str(best_params_path),
            "trials_file": str(trials_path),
            "html_reports": report_files,
            "html_report_status": report_status,
        },
    }


def _optuna_report_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "optimization_history": output_dir / "optuna_optimization_history.html",
        "param_importances": output_dir / "optuna_param_importances.html",
        "slice": output_dir / "optuna_slice.html",
    }


def _write_optuna_html_reports(study, report_paths: dict[str, Path]) -> tuple[list[str], dict[str, str]]:
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_slice,
        )
    except Exception as exc:
        return [], {"status": f"visualization_unavailable: {exc}"}

    completed_trials = [
        trial
        for trial in study.trials
        if getattr(trial.state, "name", "") == "COMPLETE"
    ]
    if not completed_trials:
        return [], {"status": "no_completed_trials"}

    plotters = {
        "optimization_history": plot_optimization_history,
        "param_importances": plot_param_importances,
        "slice": plot_slice,
    }
    files = []
    status = {}
    for name, plotter in plotters.items():
        path = report_paths[name]
        try:
            fig = plotter(study)
            fig.write_html(path)
        except Exception as exc:
            status[name] = f"failed: {exc}"
            continue
        files.append(str(path))
        status[name] = "written"
    return files, status


def _tuning_metric(config: BenchmarkConfig, job: BenchmarkJob) -> str:
    if config.tuning.metric:
        return config.tuning.metric
    if job.target.main_metric:
        return job.target.main_metric
    return "roc_auc" if job.target.task_type == "binary" else "f1_macro"


def _tuning_direction(config: BenchmarkConfig, metric: str) -> str:
    if config.tuning.direction in {"maximize", "minimize"}:
        return config.tuning.direction
    return "minimize" if metric == "log_loss" else "maximize"


def _tuning_cv_score(
    *,
    config: BenchmarkConfig,
    job: BenchmarkJob,
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series | None,
    metric: str,
) -> float:
    splitter = _build_splitter(config, y, cv_splits=config.tuning.cv_splits)
    scores = []
    for train_idx, validation_idx in splitter.split(X, y):
        estimator = clone(pipeline)
        fit_params = {}
        if sample_weight is not None:
            fit_params["model__sample_weight"] = sample_weight.iloc[train_idx]
        estimator.fit(X.iloc[train_idx], y.iloc[train_idx], **fit_params)
        y_true = y.iloc[validation_idx].reset_index(drop=True)
        y_pred = pd.Series(
            np.asarray(estimator.predict(X.iloc[validation_idx])).ravel()
        ).reset_index(drop=True)
        proba = _predict_proba(estimator, X.iloc[validation_idx])
        metrics = _aggregate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            proba=proba,
            classes=_estimator_classes(estimator),
            task_type=job.target.task_type,
            positive_label=job.target.positive_label,
        )
        if metric not in metrics:
            raise ValueError(
                f"Tuning metric '{metric}' was not computed for job '{job.slug}'."
            )
        scores.append(metrics[metric])
    return float(np.mean(scores))


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
                    "Training is currently implemented for dummy, logistic, "
                    f"CatBoost, and LightGBM jobs only. Received model='{job.model_name}'."
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
    split_data = _split_train_test(config, prepared)
    tuning_result = _tune_hyperparameters(config, job, prepared, split_data)
    build_result = build_pipeline_contract(
        config,
        job,
        prepared["X"],
        model_params=tuning_result["best_params"],
    )
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
    test_metrics, test_predictions, final_estimator = _fit_and_evaluate_test(
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
    legacy_predictions = output_dir / "predictions.csv"
    if legacy_predictions.exists():
        legacy_predictions.unlink()
    cv_results.to_csv(output_dir / "cv_results.csv", index=False)
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    audit = _build_and_write_audit(
        output_dir=output_dir,
        prepared=prepared,
        split_data=split_data,
        final_estimator=final_estimator,
        build_result=build_result,
        job=job,
    )
    cache_index_path = _write_feature_selection_cache_index(get_output_dir(config))

    diagnostics = [
        DiagnosticResult(
            name="training",
            status="completed",
            files=[
                "cv_results.csv",
                "validation_predictions.csv",
                "test_predictions.csv",
                "final_features.csv",
                "imputation_report.csv",
                "correlation_matrix.csv",
                "correlation_pairs.csv",
                "shap_rfecv_history.csv",
                *tuning_result["files"],
                *([str(cache_index_path)] if cache_index_path is not None else []),
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
                "audit_files": audit["files"],
                "tuning": tuning_result["summary"],
                "feature_selection_cache_index": (
                    str(cache_index_path)
                    if cache_index_path is not None
                    else None
                ),
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
            "tuning": tuning_result["summary"],
            "feature_selection_cache_index": (
                str(cache_index_path)
                if cache_index_path is not None
                else None
            ),
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
            "benchmark_audit": audit["summary"],
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
    original_rows = len(df)
    original_columns = list(df.columns)

    target_columns = _target_columns(config, job)
    target_column = target_columns[0]
    missing_targets = df[target_column].isna()
    dropped_missing_target_rows = int(missing_targets.sum())
    if dropped_missing_target_rows:
        df = df.loc[~missing_targets].copy()

    df, cohort_filter_report = _apply_cohort_row_filters(df, config)

    y = df[target_column]
    reserved_columns = set(_all_target_columns(config))
    reserved_columns.update(target_columns)
    weight_column = config.raw.get("data", {}).get("weight_column")
    sample_weight = None
    if weight_column and weight_column in df.columns:
        sample_weight = df[weight_column]
        reserved_columns.add(weight_column)

    candidate_columns = [column for column in df.columns if column not in reserved_columns]
    target_removed_columns = [column for column in original_columns if column in reserved_columns]
    resolution = resolve_feature_view(
        candidate_columns,
        job.feature_view,
        config.feature_groups,
    )
    feature_columns = list(resolution.selected_columns)
    X = _apply_policy_drops(df[feature_columns], job)
    feature_columns = list(X.columns)
    X, feature_na_report = _apply_feature_na_row_filter(X, config)
    y = y.loc[X.index]
    if sample_weight is not None:
        sample_weight = sample_weight.loc[X.index]
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
        "original_rows": original_rows,
        "original_features": len(original_columns),
        "original_columns": original_columns,
        "rows_after_target_na_drop": original_rows - dropped_missing_target_rows,
        "rows_after_cohort_filters": cohort_filter_report["rows_after"],
        "rows_after_feature_na_filter": feature_na_report["rows_after"],
        "cohort_filter_report": cohort_filter_report,
        "feature_na_filter_report": feature_na_report,
        "target_removed_columns": target_removed_columns,
        "candidate_features_after_target_removal": len(candidate_columns),
    }


def _apply_cohort_row_filters(
    df: pd.DataFrame,
    config: BenchmarkConfig,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    rules = config.row_filters.cohort_rules
    report = {
        "status": "not_configured" if not rules else "disabled",
        "rows_before": len(df),
        "rows_after": len(df),
        "rows_dropped": 0,
        "rules": [],
    }
    if not rules:
        return df, report

    filtered = df
    applied = False
    enabled = False
    for rule in rules:
        rule_report = {
            "name": rule.name,
            "type": rule.type,
            "column": rule.column,
            "enabled": rule.enabled,
            "status": "disabled",
            "rows_before": len(filtered),
            "rows_after": len(filtered),
            "rows_dropped": 0,
            "values": rule.values,
            "min_count": rule.min_count,
            "min_fraction": rule.min_fraction,
            "drop_missing": rule.drop_missing,
        }
        if not rule.enabled:
            report["rules"].append(rule_report)
            continue
        enabled = True
        if rule.column not in filtered.columns:
            rule_report["status"] = "missing_column"
            report["rules"].append(rule_report)
            continue

        rule_input = filtered
        mask = _cohort_rule_mask(rule_input, rule)
        before = len(filtered)
        filtered = filtered.loc[mask].copy()
        rule_report.update(
            {
                "status": "applied",
                "rows_after": len(filtered),
                "rows_dropped": before - len(filtered),
            }
        )
        if rule.type == "min_frequency":
            counts = rule_input[rule.column].value_counts(dropna=False)
            kept_values = filtered[rule.column].dropna().unique().tolist()
            rule_report["kept_values"] = sorted(kept_values, key=str)
            rule_report["value_counts_before"] = {
                str(key): int(value) for key, value in counts.items()
            }
        applied = True
        report["rules"].append(rule_report)

    report.update(
        {
            "status": (
                "applied"
                if applied
                else ("configured_no_effect" if enabled else "disabled")
            ),
            "rows_after": len(filtered),
            "rows_dropped": len(df) - len(filtered),
        }
    )
    return filtered, report


def _cohort_rule_mask(df: pd.DataFrame, rule: Any) -> pd.Series:
    series = df[rule.column]
    if rule.type == "exclude_values":
        return ~series.isin(rule.values)
    if rule.type == "include_values":
        return series.isin(rule.values)
    if rule.type == "drop_missing":
        return series.notna()
    if rule.type == "min_frequency":
        counts = series.value_counts(dropna=False)
        total = len(series)
        keep_values = set(counts.index)
        if rule.min_count is not None:
            keep_values &= set(counts[counts >= rule.min_count].index)
        if rule.min_fraction is not None:
            keep_values &= set(counts[(counts / total) >= rule.min_fraction].index)
        mask = series.isin(keep_values)
        if not rule.drop_missing:
            mask = mask | series.isna()
        return mask
    raise ValueError(
        f"Unknown cohort row filter type '{rule.type}' for rule '{rule.name}'."
    )


def _apply_feature_na_row_filter(
    X: pd.DataFrame,
    config: BenchmarkConfig,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    spec = config.row_filters.feature_na
    report = {
        "status": "not_configured" if spec is None else "disabled",
        "rows_before": len(X),
        "rows_after": len(X),
        "rows_dropped": 0,
        "columns": [],
        "missing_columns": [],
        "mode": None if spec is None else spec.mode,
        "max_missing_fraction": None if spec is None else spec.max_missing_fraction,
    }
    if spec is None or not spec.enabled:
        return X, report

    columns, missing_columns = _resolve_feature_na_columns(X, spec)
    report["columns"] = columns
    report["missing_columns"] = missing_columns
    if not columns:
        report["status"] = "no_matching_columns"
        return X, report

    missing = X[columns].isna()
    if spec.mode == "any":
        drop_mask = missing.any(axis=1)
    elif spec.mode == "all":
        drop_mask = missing.all(axis=1)
    elif spec.mode == "max_fraction":
        if spec.max_missing_fraction is None:
            raise ValueError(
                "feature_na_filter.mode=max_fraction requires max_missing_fraction."
            )
        drop_mask = missing.mean(axis=1) > spec.max_missing_fraction
    else:
        raise ValueError(f"Unknown feature_na_filter mode '{spec.mode}'.")

    filtered = X.loc[~drop_mask].copy()
    report.update(
        {
            "status": "applied",
            "rows_after": len(filtered),
            "rows_dropped": int(drop_mask.sum()),
        }
    )
    return filtered, report


def _resolve_feature_na_columns(X: pd.DataFrame, spec: Any) -> tuple[list[str], list[str]]:
    selected = set()
    missing = []
    for column in spec.columns:
        if column in X.columns:
            selected.add(column)
        else:
            missing.append(column)
    patterns = spec.include_patterns or (["*"] if not spec.columns else [])
    for pattern in patterns:
        selected.update(column for column in X.columns if fnmatch(column, pattern))
    for column in spec.exclude_columns:
        selected.discard(column)
    for pattern in spec.exclude_patterns:
        selected = {column for column in selected if not fnmatch(column, pattern)}
    return sorted(selected), missing


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
    return metrics, predictions, estimator


def _build_and_write_audit(
    *,
    output_dir: Path,
    prepared: Dict[str, Any],
    split_data: Dict[str, Any],
    final_estimator,
    build_result: PipelineBuildResult,
    job: BenchmarkJob,
) -> Dict[str, Any]:
    preprocess = final_estimator.named_steps["preprocess"]
    correlation_filter = final_estimator.named_steps["correlation_filter"]
    feature_selection = final_estimator.named_steps["feature_selection"]

    transformed_train = preprocess.transform(split_data["X_train"])
    transformed_train = _as_dataframe(transformed_train, preprocess.get_feature_names_out())
    final_features = final_estimator[:-1].get_feature_names_out().tolist()

    final_features_path = output_dir / "final_features.csv"
    pd.DataFrame({"feature": final_features}).to_csv(final_features_path, index=False)

    imputation_report = _imputation_report(preprocess)
    imputation_report_path = output_dir / "imputation_report.csv"
    imputation_report.to_csv(imputation_report_path, index=False)

    corr_numeric = transformed_train.select_dtypes(include=[np.number])
    corr_matrix = corr_numeric.corr(method="spearman").fillna(0.0)
    corr_matrix_path = output_dir / "correlation_matrix.csv"
    corr_matrix.to_csv(corr_matrix_path)

    corr_pairs = _top_correlation_pairs(
        corr_matrix,
        threshold=job.feature_policy.correlation_filter.threshold,
    )
    corr_pairs_path = output_dir / "correlation_pairs.csv"
    corr_pairs.to_csv(corr_pairs_path, index=False)
    shap_history = _shap_rfecv_history(feature_selection)
    shap_history_path = output_dir / "shap_rfecv_history.csv"
    shap_history.to_csv(shap_history_path, index=False)

    numeric_builder = preprocess.numeric_pipeline_
    qcut_created = [
        f"{column}_qcut"
        for column, bins in numeric_builder.qcut_bins_.items()
        if len(bins) >= 2
    ]
    iqr_report = _iqr_report(split_data["X_train"], numeric_builder)
    feature_view = prepared["feature_view_resolution"]
    policy_dropped = sorted(
        set(feature_view.selected_columns) - set(prepared["feature_columns"])
    )
    feature_set_status = getattr(feature_selection, "status_", "unknown")

    summary = {
        "rows": {
            "original_rows": prepared["original_rows"],
            "rows_dropped_missing_target": prepared["dropped_missing_target_rows"],
            "rows_after_target_na_drop": prepared["rows_after_target_na_drop"],
            "rows_after_cohort_filters": prepared["rows_after_cohort_filters"],
            "rows_dropped_custom_filters": prepared["cohort_filter_report"][
                "rows_dropped"
            ],
            "custom_filters_status": prepared["cohort_filter_report"]["status"],
            "custom_filters": prepared["cohort_filter_report"]["rules"],
            "rows_after_feature_na_filter": prepared["rows_after_feature_na_filter"],
            "rows_dropped_feature_na": prepared["feature_na_filter_report"][
                "rows_dropped"
            ],
            "feature_na_row_filter_status": prepared["feature_na_filter_report"][
                "status"
            ],
            "feature_na_row_filter": prepared["feature_na_filter_report"],
            "train_rows": len(split_data["X_train"]),
            "validation_folds": job_feature_cv_summary(job, split_data),
            "test_rows": len(split_data["X_test"]),
        },
        "features": {
            "original_features": prepared["original_features"],
            "target_removed_features": {
                "count": len(prepared["target_removed_columns"]),
                "columns": prepared["target_removed_columns"],
            },
            "candidate_features_after_target_removal": prepared[
                "candidate_features_after_target_removal"
            ],
            "feature_view": {
                "name": feature_view.feature_view,
                "selected_count": len(feature_view.selected_columns),
                "excluded_count": len(feature_view.excluded_columns),
                "excluded_columns": feature_view.excluded_columns,
                "included_by_group": feature_view.included_by_group,
                "excluded_by_group": feature_view.excluded_by_group,
            },
            "feature_policy_dropped": {
                "count": len(policy_dropped),
                "columns": policy_dropped,
                "drop_columns": job.feature_policy.drop_columns,
                "drop_patterns": job.feature_policy.drop_patterns,
            },
            "selected_input_features": len(prepared["feature_columns"]),
            "features_dropped_too_many_na": {
                "count": 0,
                "columns": [],
                "status": "not_configured",
            },
            "recoded_variables_created": {
                "count": len(qcut_created),
                "columns": qcut_created,
                "source": "qcut_numeric",
            },
            "imputation": {
                "numeric_strategy": job.feature_policy.impute_numeric,
                "categorical_strategy": job.feature_policy.impute_categorical,
                "imputed_feature_count": int(len(imputation_report)),
                "details_file": "imputation_report.csv",
            },
            "iqr_outlier_handling": {
                "strategy": job.feature_policy.iqr_outlier_handling,
                "multiplier": job.feature_policy.iqr_multiplier,
                "features_with_bounds": len(numeric_builder.iqr_bounds_),
                "outliers_replaced_with_na_total": int(
                    iqr_report["outliers_replaced_with_na"].sum()
                ),
                "top_features": iqr_report.head(20).to_dict(orient="records"),
            },
            "qcut_numeric": {
                "strategy": job.feature_policy.qcut_numeric,
                "requested_bins": job.feature_policy.qcut_bins,
                "created_count": len(qcut_created),
                "created_columns": qcut_created,
            },
            "categorical_encoding": _categorical_encoding_summary(preprocess),
            "calibration": {
                "enabled": job.feature_policy.calibration.enabled,
                "method": job.feature_policy.calibration.method,
                "cv": job.feature_policy.calibration.cv,
                "ensemble": job.feature_policy.calibration.ensemble,
            },
            "correlation": {
                "enabled": job.feature_policy.correlation_filter.enabled,
                "mode": job.feature_policy.correlation_filter.mode,
                "threshold": job.feature_policy.correlation_filter.threshold,
                "matrix_file": "correlation_matrix.csv",
                "pairs_file": "correlation_pairs.csv",
                "pairs_above_threshold": int(len(corr_pairs)),
                "dropped_count": len(correlation_filter.dropped_features_),
                "dropped_features": correlation_filter.dropped_features_,
            },
            "feature_selection": {
                "strategy": job.feature_set.strategy,
                "max_features": job.feature_set.max_features,
                "status": feature_set_status,
                "cache_status": getattr(
                    feature_selection, "cache_status_", "not_applicable"
                ),
                "cache_file": getattr(feature_selection, "cache_path_", None),
                "ranking_feature_count": len(
                    getattr(feature_selection, "ranking_features_", [])
                ),
                "best_score": getattr(feature_selection, "best_score_", None),
                "selector_rows": getattr(feature_selection, "selector_rows_", None),
                "selector_cv_splits": getattr(feature_selection, "cv_splits", None),
                "selector_step_fraction": getattr(
                    feature_selection,
                    "step_fraction",
                    None,
                ),
                "selector_min_features_to_select": getattr(
                    feature_selection,
                    "min_features_to_select",
                    None,
                ),
                "selector_max_shap_rows": getattr(
                    feature_selection,
                    "max_shap_rows",
                    None,
                ),
                "selector_max_rows": getattr(
                    feature_selection,
                    "max_selector_rows",
                    None,
                ),
                "selector_estimator_params": getattr(
                    feature_selection,
                    "selector_estimator_params_",
                    {},
                ),
                "history_file": "shap_rfecv_history.csv",
                "history_rounds": int(len(shap_history)),
                "kept_count": len(final_features),
                "dropped_count": len(
                    getattr(feature_selection, "dropped_features_", [])
                ),
                "dropped_features": getattr(
                    feature_selection, "dropped_features_", []
                ),
                "kept_features_file": "final_features.csv",
            },
        },
        "files": {
            "final_features": "final_features.csv",
            "imputation_report": "imputation_report.csv",
            "correlation_matrix": "correlation_matrix.csv",
            "correlation_pairs": "correlation_pairs.csv",
            "shap_rfecv_history": "shap_rfecv_history.csv",
        },
    }
    return {
        "summary": summary,
        "files": list(summary["files"].values()),
    }


def job_feature_cv_summary(job: BenchmarkJob, split_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "strategy": job.feature_set.strategy,
        "cv_source": "training_subset_only",
        "training_subset_rows": len(split_data["X_train"]),
    }


def _as_dataframe(X, columns) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(np.asarray(X), columns=list(columns))


def _imputation_report(preprocess) -> pd.DataFrame:
    rows = []
    numeric_builder = preprocess.numeric_pipeline_
    if numeric_builder.imputer_ is not None:
        for column, value in zip(
            numeric_builder.feature_names_out_,
            numeric_builder.imputer_.statistics_,
        ):
            rows.append(
                {
                    "feature": column,
                    "source": "numeric",
                    "strategy": numeric_builder.impute_numeric,
                    "fill_value": value,
                }
            )
    if preprocess.categorical_imputer_ is not None:
        for column, value in zip(
            list(preprocess.categorical_columns),
            preprocess.categorical_imputer_.statistics_,
        ):
            rows.append(
                {
                    "feature": column,
                    "source": "categorical",
                    "strategy": preprocess.impute_categorical,
                    "fill_value": value,
                }
            )
    return pd.DataFrame(rows, columns=["feature", "source", "strategy", "fill_value"])


def _iqr_report(X_train: pd.DataFrame, numeric_builder) -> pd.DataFrame:
    rows = []
    numeric = X_train[list(numeric_builder.feature_names_in_)].apply(
        pd.to_numeric, errors="coerce"
    )
    for column, (lower, upper) in numeric_builder.iqr_bounds_.items():
        mask = (numeric[column] < lower) | (numeric[column] > upper)
        rows.append(
            {
                "feature": column,
                "lower_bound": lower,
                "upper_bound": upper,
                "outliers_replaced_with_na": int(mask.sum()),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "feature",
            "lower_bound",
            "upper_bound",
            "outliers_replaced_with_na",
        ],
    ).sort_values(
        "outliers_replaced_with_na", ascending=False
    )


def _top_correlation_pairs(corr: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    columns = list(corr.columns)
    for left_idx, left in enumerate(columns):
        for right in columns[left_idx + 1 :]:
            value = float(abs(corr.loc[left, right]))
            if value > threshold:
                rows.append(
                    {
                        "feature_1": left,
                        "feature_2": right,
                        "abs_spearman": value,
                    }
                )
    return pd.DataFrame(
        rows,
        columns=["feature_1", "feature_2", "abs_spearman"],
    ).sort_values("abs_spearman", ascending=False)


def _categorical_encoding_summary(preprocess) -> Dict[str, Any]:
    columns = list(preprocess.categorical_columns)
    if preprocess.one_hot_encoder_ is None:
        return {
            "handling": preprocess.categorical_handling,
            "input_columns": columns,
            "created_columns_count": 0,
            "created_columns": [],
        }
    created = preprocess.one_hot_encoder_.get_feature_names_out(columns).tolist()
    return {
        "handling": preprocess.categorical_handling,
        "input_columns": columns,
        "created_columns_count": len(created),
        "created_columns": created,
    }


def _shap_rfecv_history(feature_selection) -> pd.DataFrame:
    rows = []
    for item in getattr(feature_selection, "history_", []):
        rows.append(
            {
                "round": item.get("round"),
                "n_features": item.get("n_features"),
                "cv_score": item.get("cv_score"),
                "best_score_so_far": item.get("best_score_so_far"),
                "removed_features": "|".join(item.get("removed_features") or []),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "round",
            "n_features",
            "cv_score",
            "best_score_so_far",
            "removed_features",
        ],
    )


def _build_splitter(
    config: BenchmarkConfig,
    y: pd.Series,
    cv_splits: int | None = None,
):
    if config.split.strategy != "cross_validation":
        raise NotImplementedError(
            "Only split.strategy='cross_validation' is implemented in this slice."
        )
    requested_splits = cv_splits or config.split.cv_splits
    n_splits = min(requested_splits, int(y.value_counts().min()))
    if n_splits < 2:
        raise ValueError("Need at least two rows per class for stratified CV.")
    if config.split.stratify:
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=config.split.random_state,
        )
    return KFold(
        n_splits=min(requested_splits, len(y)),
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
