"""Optuna and training wrappers for staged mortality pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from optuna.trial import Trial
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .metrics import binary_metrics, save_predictions, visualize_results
from .models import build_model, fit_model_with_optional_early_stopping
from .utils import safe_cv_splits, save_json


def get_param_space(
    algorithm: str,
    trial: Trial,
    imbalance_ratio: float,
    seed: int,
    n_cpus: int,
    fast_search: bool = False,
) -> dict[str, Any]:
    if algorithm == "catb":
        pos_w = trial.suggest_float("pos_weight", max(0.1, imbalance_ratio * 0.75), max(0.2, imbalance_ratio * 1.25))
        if fast_search:
            iterations_low, iterations_high = 250, 700
            depth_low, depth_high = 4, 7
            leaf_low, leaf_high = 20, 120
        else:
            iterations_low, iterations_high = 400, 1200
            depth_low, depth_high = 4, 10
            leaf_low, leaf_high = 10, 150
        return {
            "iterations": trial.suggest_int("iterations", iterations_low, iterations_high),
            "learning_rate": trial.suggest_float("learning_rate", 0.01 if fast_search else 0.005, 0.1, log=True),
            "depth": trial.suggest_int("depth", depth_low, depth_high),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 80 if fast_search else 120, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", leaf_low, leaf_high),
            "random_strength": trial.suggest_float("random_strength", 0, 10 if fast_search else 20),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0 if fast_search else 1.5),
            "bootstrap_type": "Bayesian",
            "loss_function": "Logloss",
            "eval_metric": "PRAUC",
            "class_weights": [1.0, pos_w],
            "random_seed": seed,
            "thread_count": n_cpus,
            "logging_level": "Silent",
            "allow_writing_files": False,
        }
    if algorithm == "lgbm":
        pos_w = trial.suggest_float("scale_pos_weight", max(0.1, imbalance_ratio * 0.75), max(0.2, imbalance_ratio * 1.25))
        return {
            "n_estimators": trial.suggest_int("n_estimators", 250 if fast_search else 400, 700 if fast_search else 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01 if fast_search else 0.005, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8 if fast_search else 12),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128 if fast_search else 256),
            "min_child_samples": trial.suggest_int("min_child_samples", 20 if fast_search else 10, 120 if fast_search else 150),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30 if fast_search else 50, log=True),
            "subsample": trial.suggest_float("subsample", 0.7 if fast_search else 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7 if fast_search else 0.6, 1.0),
            "scale_pos_weight": pos_w,
            "n_jobs": n_cpus,
            "random_state": seed,
            "verbose": -1,
        }
    if algorithm == "xgb":
        pos_w = trial.suggest_float("scale_pos_weight", max(0.1, imbalance_ratio * 0.75), max(0.2, imbalance_ratio * 1.25))
        return {
            "n_estimators": trial.suggest_int("n_estimators", 250 if fast_search else 400, 700 if fast_search else 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01 if fast_search else 0.005, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7 if fast_search else 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 12 if fast_search else 20),
            "gamma": trial.suggest_float("gamma", 0, 3 if fast_search else 5),
            "subsample": trial.suggest_float("subsample", 0.7 if fast_search else 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7 if fast_search else 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30 if fast_search else 50, log=True),
            "scale_pos_weight": pos_w,
            "n_jobs": n_cpus,
            "random_state": seed,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        }
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def make_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int,
    algorithm: str,
    imbalance_ratio: float,
    n_cpus: int,
    seed: int,
    early_stopping_rounds: int = 50,
    fast_search: bool = False,
):
    n_splits = safe_cv_splits(y_train, n_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def objective(trial: Trial) -> float:
        pr_scores: list[float] = []
        roc_scores: list[float] = []
        params = get_param_space(algorithm, trial, imbalance_ratio, seed, n_cpus, fast_search=fast_search)
        for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train), start=1):
            X_tr = X_train.iloc[tr_idx]
            X_va = X_train.iloc[va_idx]
            y_tr = y_train.iloc[tr_idx]
            y_va = y_train.iloc[va_idx]

            model = build_model(algorithm, params)
            fit_model_with_optional_early_stopping(model, algorithm, X_tr, y_tr, X_va, y_va, early_stopping_rounds=early_stopping_rounds)

            y_proba = model.predict_proba(X_va)[:, 1]
            pr_scores.append(float(average_precision_score(y_va, y_proba)))
            roc_scores.append(float(roc_auc_score(y_va, y_proba)))

            running_mean_pr = float(np.mean(pr_scores))
            trial.report(running_mean_pr, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_pr = float(np.mean(pr_scores))
        mean_roc = float(np.mean(roc_scores))
        trial.set_user_attr("pr_auc", mean_pr)
        trial.set_user_attr("roc_auc", mean_roc)
        trial.set_user_attr("n_cv_splits", n_splits)
        trial.set_user_attr("early_stopping_rounds", early_stopping_rounds)
        trial.set_user_attr("fast_search", fast_search)
        return mean_pr

    return objective


def final_params_from_trial_params(algorithm: str, trial_params: dict[str, Any], imbalance_ratio: float, seed: int, n_cpus: int) -> dict[str, Any]:
    params = trial_params.copy()
    if algorithm == "catb":
        pos_w = params.pop("pos_weight", imbalance_ratio)
        params.setdefault("iterations", 750)
        params.setdefault("loss_function", "Logloss")
        params.setdefault("eval_metric", "PRAUC")
        params["class_weights"] = [1.0, pos_w]
        params["random_seed"] = seed
        params["thread_count"] = n_cpus
        params["logging_level"] = "Silent"
    elif algorithm == "lgbm":
        params.setdefault("n_estimators", 750)
        params["n_jobs"] = n_cpus
        params["random_state"] = seed
        params.setdefault("verbose", -1)
    elif algorithm == "xgb":
        params.setdefault("n_estimators", 750)
        params["n_jobs"] = n_cpus
        params["random_state"] = seed
        params.setdefault("objective", "binary:logistic")
        params.setdefault("eval_metric", "logloss")
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return params


def optimise_and_fit_binary_model(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target: str,
    algorithm: str,
    n_splits: int,
    n_trials: int,
    storage: str | None,
    imbalance_ratio: float,
    n_cpus: int,
    seed: int,
    array_id: str,
    job_id: str,
    run_dir: Path,
    study_name: str,
    logger=None,
    early_stopping_rounds: int = 50,
    fast_search: bool = False,
):
    startup_trials = min(5, max(1, n_trials // 4))
    study = optuna.create_study(
        study_name=f"{study_name}_{target}",
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=startup_trials, n_ei_candidates=12 if fast_search else 24, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=startup_trials, n_warmup_steps=1),
        direction="maximize",
        load_if_exists=True,
    )

    objective = make_objective(
        X_train=X_train,
        y_train=y_train,
        n_splits=n_splits,
        algorithm=algorithm,
        imbalance_ratio=imbalance_ratio,
        n_cpus=n_cpus,
        seed=seed,
        early_stopping_rounds=early_stopping_rounds,
        fast_search=fast_search,
    )

    study.optimize(objective, n_trials=n_trials, n_jobs=1, gc_after_trial=True)

    best_params = final_params_from_trial_params(algorithm=algorithm, trial_params=study.best_trial.params.copy(), imbalance_ratio=imbalance_ratio, seed=seed, n_cpus=n_cpus)

    model = build_model(algorithm, best_params)
    model.fit(X_train, y_train)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    metrics = binary_metrics(y_test, y_test_proba, threshold=0.5)
    metrics.update(
        {
            "target": target,
            "algorithm": algorithm,
            "array_task_id": array_id,
            "job_id": job_id,
            "seed": seed,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "early_stopping_rounds_cv": int(early_stopping_rounds),
            "fast_search": bool(fast_search),
        }
    )

    trials_df = study.trials_dataframe()
    trials_df.to_csv(run_dir / "optuna" / f"optuna_{target}_array_{array_id}.csv", index=False)
    trials_df.to_pickle(run_dir / "optuna" / f"optuna_{target}_array_{array_id}.pkl")

    save_json(run_dir / f"best_params_{target}_array_{array_id}.json", best_params)
    save_json(run_dir / f"metrics_{target}_array_{array_id}.json", metrics)
    save_json(
        run_dir / "optuna" / f"best_trial_{target}_array_{array_id}.json",
        {
            "number": study.best_trial.number,
            "value": study.best_trial.value,
            "params": study.best_trial.params,
            "user_attrs": study.best_trial.user_attrs,
        },
    )

    joblib.dump(model, run_dir / "models" / f"{target}_{algorithm}_array_{array_id}.pkl")

    save_predictions(run_dir / "predictions" / f"predictions_{target}_array_{array_id}.csv", X_test.index, y_test, y_test_proba)
    visualize_results(y_test, y_test_proba, target, run_dir, array_id)

    return model, y_test_proba, metrics, best_params
