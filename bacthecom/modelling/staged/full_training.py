"""Config-driven, patient-grouped mortality training and raw-CSV inference.

Evaluation models never see holdout rows. Full-data models are separate artifacts.
Stage 2 estimates P(death <=14d | death <=30d, X), without an injected probability.
"""
from __future__ import annotations

import fnmatch
import re
import unicodedata
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from .feature_filters import fit_iqr_bounds, apply_iqr_bounds_to_nan, remove_correlated_features_train_fit
from .explanations import export_explanations
from .metrics import binary_metrics
from .models import build_model, fit_pr_model
from .optuna import get_param_space


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


class MortalityPreprocessor(BaseEstimator, TransformerMixin):
    """All decisions are fitted per training partition, including after IQR masking."""
    def __init__(self, missing_percent=80.0, iqr_multiplier=5.0, max_corr=0.99, missing_indicators=True):
        self.missing_percent = missing_percent
        self.iqr_multiplier = iqr_multiplier
        self.max_corr = max_corr
        self.missing_indicators = missing_indicators

    def _numeric(self, X):
        return X.loc[:, self.input_columns_].apply(pd.to_numeric, errors="raise").astype(float).replace([np.inf, -np.inf], np.nan)

    def fit(self, X, y=None):
        if not 0 <= self.missing_percent <= 100 or self.iqr_multiplier < 0 or not 0 < self.max_corr <= 1:
            raise ValueError("Invalid missingness, IQR or correlation settings")
        self.input_columns_ = list(X.columns)
        values = self._numeric(X)
        self.bounds_ = fit_iqr_bounds(values, self.iqr_multiplier)
        values, outliers = apply_iqr_bounds_to_nan(values, self.bounds_)
        missing = values.isna().mean() * 100
        self.kept_ = missing[(missing <= self.missing_percent) & (missing < 100)].index.tolist()
        self.indicators_ = [c for c in self.kept_ if values[c].isna().any()] if self.missing_indicators else []
        self.medians_ = values[self.kept_].median()
        encoded = self._impute(values)
        self.constant_ = encoded.columns[encoded.nunique() <= 1].tolist()
        encoded = encoded.drop(columns=self.constant_)
        filtered, _, correlation = remove_correlated_features_train_fit(encoded, encoded.iloc[:0], threshold=self.max_corr)
        self.output_columns_ = list(filtered.columns)
        if not self.output_columns_:
            raise ValueError("No predictors remain after preprocessing")
        self.audit_ = {"input_columns": self.input_columns_, "output_columns": self.output_columns_,
                       "missing_percent_after_iqr": missing.to_dict(), "iqr_bounds": self.bounds_,
                       "iqr_outliers": outliers, "constant_columns": self.constant_,
                       "correlation_dropped": correlation, "medians": self.medians_.to_dict(),
                       "missing_indicators": self.indicators_}
        return self

    def _impute(self, values):
        result = values[self.kept_].fillna(self.medians_).copy()
        for col in self.indicators_:
            result[col + "__missing_indicator"] = values[col].isna().astype(float)
        return result

    def transform(self, X):
        values, _ = apply_iqr_bounds_to_nan(self._numeric(X), self.bounds_)
        return self._impute(values).loc[:, self.output_columns_]


def grouped_folds(X, y, groups, n_splits, seed):
    """Reduce folds if necessary, but never silently allow single-class folds."""
    max_folds = min(n_splits, *(groups[y == label].nunique() for label in (0, 1)))
    for count in range(max_folds, 1, -1):
        folds = list(StratifiedGroupKFold(count, shuffle=True, random_state=seed).split(X, y, groups))
        if all(y.iloc[a].nunique() == y.iloc[b].nunique() == 2 for a, b in folds):
            return folds
    raise ValueError("Insufficient independent patient groups for binary stratified CV")


def exclude_target_like_features(predictors, targets, forbidden=(), patterns=()):
    """Name-based exclusions only: no holdout-label-dependent feature selection."""
    def normalized(name):
        return re.sub(r"[^a-z0-9]+", "_", unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()).strip("_")
    protected = {normalized(c) for c in list(targets) + list(forbidden)}
    tokens = re.compile(r"(?:^|_)(?:mortalidad|mortality|death|dead|deceased|exitus|target|outcome|label|survival|fallecimiento|defuncion)(?:_|$)")
    dropped, kept = {}, []
    for col in predictors:
        name = normalized(col)
        reason = None
        if name in protected or any(name.startswith(t + "_") for t in protected):
            reason = "target/forbidden column or derivative"
        elif tokens.search(name) or re.search(r"(?:^|_)(?:early|late)_(?:14|30)(?:_|$)", name):
            reason = "outcome-like name"
        elif name.startswith(("fecha_", "date_", "record_id", "patient_group", "episode_id")):
            reason = "identifier/date"
        elif any(fnmatch.fnmatchcase(col.lower(), pattern.lower()) for pattern in patterns):
            reason = "configured exclusion pattern"
        if reason:
            dropped[col] = reason
        else:
            kept.append(col)
    return kept, dropped


def load_dataset(cfg):
    data = pd.read_csv(cfg["data"])
    schema = json.loads(Path(cfg["schema"]).read_text())
    ids = pd.read_csv(cfg["identifiers"], dtype={"patient_group": str})
    if len(ids) != len(data) or not np.array_equal(ids.row_number, np.arange(len(data))):
        raise ValueError("Identifiers must be row-aligned with the original filtered CSV")
    if ids.patient_group.isna().any():
        raise ValueError("Missing patient group")
    predictors = schema["predictor_columns"]
    if len(set(predictors)) != len(predictors):
        raise ValueError("Duplicate predictors in schema")
    forbidden = schema.get("config", {}).get("filtering", {}).get("forbidden_predictor_columns", [])
    predictors, excluded = exclude_target_like_features(predictors, schema["target_columns"], forbidden,
                                                        cfg.get("feature_exclusion_patterns", []))
    if not predictors:
        raise ValueError("No predictors remain after target-like feature exclusion")
    data.attrs["excluded_predictors"] = excluded
    print(f"Excluded {len(excluded)} target-like/forbidden predictors", flush=True)
    missing = set(predictors) - set(data.columns)
    if missing:
        raise ValueError(f"Missing schema predictors: {sorted(missing)}")
    targets = sorted({r[k] for r in cfg["runs"] for k in ("target", "target1", "target2") if k in r})
    for target in targets:
        if target not in schema["target_columns"]:
            raise ValueError(f"Target absent from schema target list: {target}")
        values = pd.to_numeric(data[target], errors="raise")
        if not values.dropna().isin([0, 1]).all():
            raise ValueError(f"Non-binary target: {target}")
        data[target] = values
    for run in cfg["runs"]:
        if run["mode"] == "STAGED":
            if run["target1"] == run["target2"] or (data[run["target2"]] > data[run["target1"]]).any():
                raise ValueError("Stage 2 must be a nested, earlier mortality outcome")
    return data, data[predictors], ids.patient_group


def fit_stage(X, y, groups, cfg, run, directory):
    directory.mkdir(parents=True)
    training = cfg["training"]
    folds = grouped_folds(X, y, groups, training["cv_splits"], run["seed"])
    # Preprocessing is deterministic per fold, so cache it across trials.
    prepared = []
    for fold, (tr, va) in enumerate(folds):
        prep = MortalityPreprocessor(**cfg["postprocessing"]).fit(X.iloc[tr])
        prepared.append((prep.transform(X.iloc[tr]), prep.transform(X.iloc[va]), y.iloc[tr], y.iloc[va]))
        save_json(directory / f"fold_{fold}_preprocessing.json", prep.audit_)
    pd.DataFrame([{"row_index": int(X.index[i]), "fold": fold} for fold, (_, va) in enumerate(folds) for i in va]).to_csv(directory / "cv_folds.csv", index=False)
    algorithm = run["algorithm"]

    def objective(trial):
        ratio = float((y == 0).sum() / (y == 1).sum()) if training["class_weighting"] else 1.0
        params = get_param_space(algorithm, trial, ratio, run["seed"], training["n_cpus"], fast_search=training.get("fast_search", True), class_weighting=training["class_weighting"])
        if not training["class_weighting"]:
            params.pop("class_weights", None)
            params.pop("scale_pos_weight", None)
        if algorithm == "catb":
            params["eval_metric"] = "PRAUC:type=Classic"
            if any(float(w) != 1.0 for w in params.get("class_weights", [1.0, 1.0])):
                params["eval_metric"] += ";use_weights=false"
        elif algorithm == "lgbm":
            params["metric"] = "average_precision"
        else:
            params["eval_metric"] = "aucpr"
        scores, tree_counts = [], []
        for fold_number, (tr, va, yt, yv) in enumerate(prepared, start=1):
            model = build_model(algorithm, params)
            trees = fit_pr_model(model, algorithm, tr, yt, va, yv, training.get("early_stopping_rounds", 50))
            tree_counts.append(trees)
            scores.append(float(average_precision_score(yv, model.predict_proba(va)[:, 1])))
            trial.report(float(np.mean(scores)), step=fold_number)
            trial.set_user_attr("fold_average_precision", scores)
            trial.set_user_attr("fold_tree_counts", tree_counts)
            if fold_number < len(prepared) and trial.should_prune():
                raise optuna.TrialPruned(f"Pruned after {fold_number} CV folds")
        # Select the refit tree count from CV only; never early-stop on holdout.
        params["iterations" if algorithm == "catb" else "n_estimators"] = max(1, int(np.median(tree_counts)))
        trial.set_user_attr("model_params", params)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", study_name=run["name"] + "_" + directory.name,
                               storage="sqlite:///" + str(directory / "optuna.db"),
                               sampler=optuna.samplers.TPESampler(seed=run["seed"]),
                               pruner=optuna.pruners.MedianPruner(
                                   n_startup_trials=training.get("pruning_startup_trials", 5),
                                   n_warmup_steps=training.get("pruning_min_folds", 2))
                               if training.get("pruning", True) else optuna.pruners.NopPruner())
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=training["n_trials"], n_jobs=1,
                   timeout=training.get("timeout_seconds"), gc_after_trial=False)
    save_json(directory / "optimization_summary.json", {
        "objective": "mean_cv_average_precision", "pruning": training.get("pruning", True),
        "trial_states": dict(pd.Series([t.state.name for t in study.trials]).value_counts().items()),
        "best_cv_average_precision": study.best_value,
        "best_fold_tree_counts": study.best_trial.user_attrs["fold_tree_counts"]})
    params = study.best_trial.user_attrs["model_params"]
    study.trials_dataframe().to_csv(directory / "trials.csv", index=False)
    save_json(directory / "best_params.json", params)
    pipeline = Pipeline([("postprocessing", MortalityPreprocessor(**cfg["postprocessing"])),
                         ("model", build_model(algorithm, params))])
    pipeline.fit(X, y)
    joblib.dump(pipeline, directory / "evaluation_model.joblib")
    save_json(directory / "preprocessing.json", pipeline["postprocessing"].audit_)
    return pipeline, params


def metrics(y, p, threshold):
    result = binary_metrics(y, p, threshold)
    result.update(brier_score=float(brier_score_loss(y, p)), log_loss=float(log_loss(y, p, labels=[0, 1])))
    return result


def write_summary_metrics(report, path):
    summary = {}
    for name, payload in report.items():
        if not isinstance(payload, dict):
            continue
        summary[name] = {
            "pr_auc": payload.get("pr_auc"),
            "roc_auc": payload.get("roc_auc"),
            "f1": payload.get("f1"),
            "precision": payload.get("precision"),
            "recall": payload.get("recall"),
            "threshold": payload.get("threshold"),
            "n": payload.get("n"),
            "actual_positive": payload.get("actual_positive"),
            "actual_negative": payload.get("actual_negative"),
            "predicted_positive": payload.get("predicted_positive"),
            "predicted_negative": payload.get("predicted_negative"),
            "positive_rate": payload.get("positive_rate"),
            "brier_score": payload.get("brier_score"),
            "log_loss": payload.get("log_loss"),
        }
    save_json(path, summary)
    return summary


def _format_metric(value, digits=4, missing="n/a"):
    if value is None:
        return missing
    return f"{value:.{digits}f}"


def log_holdout_summary(report, run_name, out):
    lines = [f"Holdout summary for {run_name}", f"Results dir: {out}"]
    for label, payload in report.items():
        if not isinstance(payload, dict):
            continue
        pr_auc = payload.get("pr_auc")
        roc_auc = payload.get("roc_auc")
        f1 = payload.get("f1")
        prec = payload.get("precision")
        rec = payload.get("recall")
        thr = payload.get("threshold")
        lines.append(
            f"- {label}: pr_auc={_format_metric(pr_auc, 4)} | roc_auc={_format_metric(roc_auc, 4)} | "
            f"f1={_format_metric(f1, 4)} | precision={_format_metric(prec, 4)} | "
            f"recall={_format_metric(rec, 4)} | threshold={_format_metric(thr, 3)} | n={payload.get('n')}"
        )
    print("\n".join(lines), flush=True)


def refit(X, y, cfg, run, params, directory):
    pipeline = Pipeline([("postprocessing", MortalityPreprocessor(**cfg["postprocessing"])),
                         ("model", build_model(run["algorithm"], params))])
    pipeline.fit(X, y)
    joblib.dump(pipeline, directory / "full_model.joblib")
    save_json(directory / "full_preprocessing.json", pipeline["postprocessing"].audit_)
    settings = cfg.get("explanations", {})
    if settings.get("full_refit", True):
        export_explanations(pipeline, X, run["algorithm"], settings, directory / "explanations_full", run["seed"], "Full refit: eligible training population (descriptive)")


def train_run(data, X, groups, cfg, run, out, validate_only=False):
    staged = run["mode"] == "STAGED"
    targets = [run["target1"], run["target2"]] if staged else [run["target"]]
    valid = data[targets].notna().all(axis=1)
    X, groups, labels = X.loc[valid], groups.loc[valid], data.loc[valid, targets].astype(int)
    # Joint mortality strata for staged runs; same split seed for direct runs.
    strata = labels.astype(str).agg("_".join, axis=1)
    split = cfg["training"]["holdout_folds"]
    if groups.nunique() < split:
        raise ValueError("Too few patient groups for holdout")
    tr, te = next(StratifiedGroupKFold(split, shuffle=True, random_state=run["seed"]).split(X, strata, groups))
    if set(groups.iloc[tr]) & set(groups.iloc[te]):
        raise AssertionError("Patient overlap")
    y1 = labels[targets[0]]
    if y1.iloc[tr].nunique() != 2 or y1.iloc[te].nunique() != 2:
        raise ValueError("Holdout or training partition lacks a target class")
    # Fail before an expensive stage-1 search if stage 2 cannot be validated.
    grouped_folds(X.iloc[tr], y1.iloc[tr], groups.iloc[tr], cfg["training"]["cv_splits"], run["seed"])
    if staged:
        early_rows = tr[y1.iloc[tr].values == 1]
        grouped_folds(X.iloc[early_rows], labels[targets[1]].iloc[early_rows], groups.iloc[early_rows], cfg["training"]["cv_splits"], run["seed"])
    if validate_only:
        for stage, target in enumerate(targets):
            rows = tr if stage == 0 else tr[y1.iloc[tr].values == 1]
            folds = grouped_folds(X.iloc[rows], labels[target].iloc[rows], groups.iloc[rows], cfg["training"]["cv_splits"], run["seed"])
            for fit_rows, validation_rows in folds:
                prep = MortalityPreprocessor(**cfg["postprocessing"]).fit(X.iloc[rows[fit_rows]])
                prep.transform(X.iloc[rows[validation_rows]])
            print(f"Preflight {run['name']} stage {stage + 1}: {len(rows)} training rows, {len(folds)} grouped folds", flush=True)
        return
    out.mkdir(parents=True, exist_ok=False)
    provenance = {key: hashlib.sha256(Path(cfg[key]).read_bytes()).hexdigest()
                  for key in ("data", "schema", "identifiers") if key in cfg}
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "catboost", "optuna")}
    save_json(out / "run_config.json", {"configuration": cfg, "run": run, "rows": len(X),
              "input_sha256": provenance, "package_versions": versions,
              "excluded_predictors": data.attrs.get("excluded_predictors", {}),
              "excluded_missing_labels": int((~valid).sum()), "holdout_rows": len(te),
              "full_refit_has_no_independent_performance_estimate": True})
    pd.DataFrame({"row_index": X.index, "partition": np.where(np.isin(np.arange(len(X)), te), "test", "train")}).to_csv(out / "split.csv", index=False)
    model1, params1 = fit_stage(X.iloc[tr], y1.iloc[tr], groups.iloc[tr], cfg, run, out / "stage1")
    export_explanations(model1, X.iloc[te], run["algorithm"], cfg.get("explanations", {}), out / "stage1/explanations_holdout", run["seed"], "Stage 1: held-out patients")
    p1 = model1.predict_proba(X.iloc[te])[:, 1]
    threshold = cfg["training"]["threshold"]
    report = {"stage1_holdout": metrics(y1.iloc[te], p1, threshold)}
    predictions = pd.DataFrame({"row_index": X.index[te], "true_stage1": y1.iloc[te].values, "p_stage1": p1})
    if staged:
        y2 = labels[targets[1]]
        tr2 = tr[y1.iloc[tr].values == 1]
        model2, params2 = fit_stage(X.iloc[tr2], y2.iloc[tr2], groups.iloc[tr2], cfg, run, out / "stage2")
        export_explanations(model2, X.iloc[te].loc[y1.iloc[te] == 1], run["algorithm"], cfg.get("explanations", {}), out / "stage2/explanations_holdout", run["seed"], "Stage 2: held-out observed 30-day deaths")
        p2 = model2.predict_proba(X.iloc[te])[:, 1]
        joint = p1 * p2
        gated = np.where(p1 >= threshold, p2, 0.0)
        report["early_joint_all_holdout"] = metrics(y2.iloc[te], joint, threshold)
        report["early_hard_gate_all_holdout"] = metrics(y2.iloc[te], gated, threshold)
        conditional = y1.iloc[te].values == 1
        report["stage2_true_30_day_deaths_only"] = metrics(y2.iloc[te].loc[conditional], p2[conditional], threshold)
        predictions["true_stage2"] = y2.iloc[te].values
        predictions["p_14_given_30"] = p2
        predictions["p_14_joint"] = joint
        predictions["p_14_hard_gate"] = gated
        predictions["prediction_14_hard_gate"] = (gated >= threshold).astype(int)
    predictions.to_csv(out / "holdout_predictions.csv", index=False)
    save_json(out / "holdout_metrics.json", report)
    write_summary_metrics(report, out / "summary_metrics.json")
    log_holdout_summary(report, run["name"], out)
    if cfg["training"]["full_refit"]:
        refit(X, y1, cfg, run, params1, out / "stage1")
        if staged:
            mask = y1 == 1
            refit(X.loc[mask], y2.loc[mask], cfg, run, params2, out / "stage2")
    save_json(out / "COMPLETE.json", {"status": "complete", "full_refit": cfg["training"]["full_refit"]})
    print(f"Completed {run['name']}: {out}", flush=True)


def read_config(path):
    cfg = json.loads(path.read_text())
    required = {"data", "schema", "identifiers", "output_dir", "postprocessing", "training", "runs"}
    if not required <= set(cfg) or set(cfg) - required - {"explanations", "feature_exclusion_patterns"}:
        raise ValueError(f"Config keys must be {sorted(required)}")
    for key in ("data", "schema", "identifiers", "output_dir"):
        cfg[key] = str((path.parent / cfg[key]).resolve())
    settings = cfg.setdefault("explanations", {"enabled": True, "max_samples": 1000, "top_features": 30, "full_refit": True})
    if set(settings) - {"enabled", "max_samples", "top_features", "full_refit"} or settings.get("max_samples", 1000) < 1 or settings.get("top_features", 30) < 1:
        raise ValueError("Invalid explanation settings")
    names = []
    for run in cfg["runs"]:
        if run["mode"] not in ("TARGET", "STAGED") or run["algorithm"] not in ("catb", "lgbm", "xgb"):
            raise ValueError("Invalid mode or algorithm")
        if not run["name"] or Path(run["name"]).name != run["name"] or run["name"] in (".", ".."):
            raise ValueError("Run name must be a simple directory name")
        names.append(run["name"])
    if len(set(names)) != len(names):
        raise ValueError("Run names must be unique")
    t = cfg["training"]
    if t.get("early_stopping_rounds", 50) < 0 or t.get("pruning_startup_trials", 5) < 0 or not 1 <= t.get("pruning_min_folds", 2) <= t["cv_splits"]:
        raise ValueError("Invalid early-stopping/pruning settings")
    if t.get("timeout_seconds") is not None and t["timeout_seconds"] <= 0:
        raise ValueError("timeout_seconds must be positive or null")
    if t["n_trials"] < 1 or t["cv_splits"] < 2 or t["holdout_folds"] < 2 or t["n_cpus"] < 1 or not 0 < t["threshold"] < 1:
        raise ValueError("Invalid training settings")
    return cfg


def predict(run_dir, csv, output, evaluation=False):
    metadata = json.loads((run_dir / "run_config.json").read_text())
    run, cfg = metadata["run"], metadata["configuration"]
    suffix = "evaluation_model.joblib" if evaluation else "full_model.joblib"
    frame = pd.read_csv(csv)
    p1 = joblib.load(run_dir / "stage1" / suffix).predict_proba(frame)[:, 1]
    threshold = cfg["training"]["threshold"]
    result = pd.DataFrame({"row_index": frame.index, "p_stage1": p1, "prediction_stage1": (p1 >= threshold).astype(int)})
    if run["mode"] == "STAGED":
        p2 = joblib.load(run_dir / "stage2" / suffix).predict_proba(frame)[:, 1]
        result["p_14_given_30"] = p2
        result["p_14_joint"] = p1 * p2
        result["p_14_hard_gate"] = np.where(p1 >= threshold, p2, 0.0)
        result["prediction_14_hard_gate"] = (result.p_14_hard_gate >= threshold).astype(int)
    result.to_csv(output, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run", help="Run name from the configured model matrix")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--predict", type=Path, help="Trained run directory")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evaluation-model", action="store_true")
    args = parser.parse_args()
    if args.predict:
        if not args.data or not args.output:
            parser.error("Prediction requires --data and --output")
        predict(args.predict, args.data, args.output, args.evaluation_model)
        return
    if not args.config:
        parser.error("Training requires --config")
    cfg = read_config(args.config.resolve())
    data, X, groups = load_dataset(cfg)
    runs = [r for r in cfg["runs"] if args.run is None or r["name"] == args.run]
    if not runs:
        parser.error("No configured run matches --run")
    print(f"Validated {len(data)} rows, {X.shape[1]} predictors, {groups.nunique()} patient groups", flush=True)
    if args.validate_only:
        for run in runs:
            train_run(data, X, groups, cfg, run, None, validate_only=True)
        return
    for run in runs:
        out = Path(cfg["output_dir"]) / run["name"]
        try:
            train_run(data, X, groups, cfg, run, out)
        except Exception as exc:
            # Keep diagnostics; never delete prior results on failure.
            if out.exists() and not (out / "COMPLETE.json").exists():
                save_json(out / "FAILED.json", {"error": repr(exc)})
            raise


if __name__ == "__main__":
    from bacthecom.modelling.staged.full_training import main as entrypoint
    entrypoint()
