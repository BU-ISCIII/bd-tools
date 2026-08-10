"""Optuna study creation and hyperparameter optimization."""

from .base import *
from .config import algorithm_params, optuna_search_spaces
from .models import (
    build_binary_model,
    build_multiclass_model,
    _fit_model,
    _find_best_threshold,
)
from sqlalchemy.pool import NullPool

_ALGO_PARAMS = algorithm_params()
_SEARCH_SPACES = optuna_search_spaces()


def _suggest_from_space(trial: optuna.Trial, name: str, spec: Dict):
    kind = spec["type"]
    if kind == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
    if kind == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
        )
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unsupported search space type '{kind}' for '{name}'.")


def _create_optuna_study(
    *,
    direction: str,
    study_name: Optional[str],
    storage: Optional[str],
    load_if_exists: bool,
    random_state: int,
):
    sampler = optuna.samplers.TPESampler(seed=random_state)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=20,
        n_warmup_steps=2,
        interval_steps=1,
    )

    kwargs = dict(
        direction=direction,
        sampler=sampler,
        pruner=pruner,
    )

    if storage:
        kwargs.update(
            study_name=study_name,
            storage=storage,
            load_if_exists=load_if_exists,
        )

    return optuna.create_study(**kwargs)


def _prepare_optuna_study(
    *,
    direction: str,
    study_name: Optional[str],
    storage: Optional[str],
    load_if_exists: bool,
    random_state: int,
    n_trials: int,
) -> Tuple[optuna.Study, int]:
    study = _create_optuna_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists,
        random_state=random_state,
    )
    existing_trials = len(study.trials)
    if load_if_exists and existing_trials >= n_trials:
        print(
            f"Loaded existing Optuna study '{study_name}' with {existing_trials} trials; "
            f"skipping optimization because {n_trials} trials were requested."
        )
        return study, 0
    if load_if_exists and existing_trials > 0:
        remaining = n_trials - existing_trials
        print(
            f"Loaded existing Optuna study '{study_name}' with {existing_trials} trials; "
            f"continuing optimization for {remaining} more trial(s)."
        )
        return study, remaining
    return study, n_trials


def _prepare_optuna_study(
    *,
    direction: str,
    study_name: Optional[str],
    storage: Optional[str],
    load_if_exists: bool,
    random_state: int,
    n_trials: int,
) -> Tuple[optuna.Study, int]:
    study = _create_optuna_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists,
        random_state=random_state,
    )
    existing_trials = len(study.trials)
    if load_if_exists and existing_trials >= n_trials:
        print(
            f"Loaded existing Optuna study '{study_name}' with {existing_trials} trials; "
            f"skipping optimization because {n_trials} trials were requested."
        )
        return study, 0
    if load_if_exists and existing_trials > 0:
        remaining = n_trials - existing_trials
        print(
            f"Loaded existing Optuna study '{study_name}' with {existing_trials} trials; "
            f"continuing optimization for {remaining} more trial(s)."
        )
        return study, remaining
    return study, n_trials


def _finalise_binary_params(
    best_trial_params: Dict,
    model_type: str,
    scale_pos_weight: float,
    random_state: int,
) -> Dict:
    params = dict(_ALGO_PARAMS.get(model_type, {}).get("binary", {}))
    params.update(best_trial_params)
    params.setdefault("random_state", random_state)

    if model_type in {"rf", "xgb", "lgbm"}:
        params.setdefault("n_jobs", N_CPUS)
    if model_type == "catb":
        params.setdefault("thread_count", N_CPUS)
    if model_type == "xgb" and "scale_pos_weight" not in params:
        params["scale_pos_weight"] = scale_pos_weight

    return params


def _binary_params_for_trial(
    trial: optuna.Trial,
    model_type: str,
    scale_pos_weight: float,
    random_state: int,
) -> Dict:
    if model_type not in _SEARCH_SPACES:
        raise ValueError(f"Unknown model_type: '{model_type}'")

    params = dict(_ALGO_PARAMS.get(model_type, {}).get("binary", {}))

    for name, spec in _SEARCH_SPACES[model_type]["binary"].items():
        params[name] = _suggest_from_space(trial, name, spec)

    params.setdefault("random_state", random_state)

    if model_type in {"rf", "xgb", "lgbm"}:
        params.setdefault("n_jobs", N_CPUS)
    if model_type == "catb":
        params.setdefault("thread_count", N_CPUS)
    if model_type == "xgb" and "scale_pos_weight" not in params:
        params["scale_pos_weight"] = scale_pos_weight

    return params


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

        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            w_tr = weight_series.iloc[tr_idx].to_numpy() if weight_series is not None else None

            model = build_binary_model(model_type, params.copy())
            _fit_model(model, model_type, X_tr, y_tr, X_va, y_va, sample_weight=w_tr)

            proba = model.predict_proba(X_va)[:, 1]
            scores.append(average_precision_score(y_va, proba))

            trial.report(float(np.mean(scores)), step=fold_idx)

            if trial.should_prune():
                raise optuna.TrialPruned()

        score = float(np.mean(scores))

        if trial.number % 100 == 0:
            print(f"Trial {trial.number}: score={score}")

        return score

    study, remaining_trials = _prepare_optuna_study(
        direction="maximize",
        study_name=study_name,
        storage=optuna_storage,
        load_if_exists=optuna_load_if_exists,
        random_state=random_state,
        n_trials=n_trials,
    )

    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials, n_jobs=1, gc_after_trial=True)

    best = _finalise_binary_params(
        study.best_trial.params.copy(),
        model_type,
        spw,
        random_state,
    )

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
        np.concatenate(oof_true_parts),
        np.concatenate(oof_proba_parts),
    )

    return best, threshold, study


def _finalise_multiclass_params(
    best_trial_params: Dict,
    model_type: str,
    num_classes: int,
    random_state: int,
) -> Dict:
    params = dict(_ALGO_PARAMS.get(model_type, {}).get("multiclass", {}))
    params.update(best_trial_params)
    params["num_class"] = num_classes
    params.setdefault("random_state", random_state)

    if model_type in {"rf", "xgb", "lgbm"}:
        params.setdefault("n_jobs", N_CPUS)
    if model_type == "catb":
        params.setdefault("thread_count", N_CPUS)

    return params


def _multiclass_params_for_trial(
    trial: optuna.Trial,
    model_type: str,
    num_classes: int,
    random_state: int,
) -> Dict:
    if model_type not in _SEARCH_SPACES:
        raise ValueError(f"Unsupported multiclass model_type: '{model_type}'")

    params = dict(_ALGO_PARAMS.get(model_type, {}).get("multiclass", {}))

    for name, spec in _SEARCH_SPACES[model_type]["multiclass"].items():
        params[name] = _suggest_from_space(trial, name, spec)

    params["num_class"] = num_classes
    params.setdefault("random_state", random_state)

    if model_type in {"rf", "xgb", "lgbm"}:
        params.setdefault("n_jobs", N_CPUS)
    if model_type == "catb":
        params.setdefault("thread_count", N_CPUS)

    return params


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

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    weight_series = (
        sample_weight.reindex(X.index)
        if isinstance(sample_weight, pd.Series)
        else (pd.Series(sample_weight, index=X.index) if sample_weight is not None else None)
    )

    def objective(trial: optuna.Trial) -> float:
        params = _multiclass_params_for_trial(trial, model_type, num_classes, random_state)
        scores = []

        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            if y_tr.nunique() < 2:
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

                trial.report(float(np.mean(scores)), step=fold_idx)

                if trial.should_prune():
                    raise optuna.TrialPruned()

            except optuna.TrialPruned:
                raise
            except Exception as e:
                print(f"Skipping fold due to error: {e}")
                continue

        if not scores:
            return 0.0

        score = float(np.mean(scores))

        if trial.number % 100 == 0:
            print(f"Trial {trial.number}: score={score}")

        return score

    study, remaining_trials = _prepare_optuna_study(
        direction="maximize",
        study_name=study_name,
        storage=optuna_storage,
        load_if_exists=optuna_load_if_exists,
        random_state=random_state,
        n_trials=n_trials,
    )

    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials, n_jobs=1, gc_after_trial=True)

    best = _finalise_multiclass_params(
        study.best_trial.params.copy(),
        model_type,
        num_classes,
        random_state,
    )

    return best, study