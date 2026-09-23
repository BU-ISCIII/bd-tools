"""Model helpers for staged mortality pipeline."""

from __future__ import annotations

from typing import Any

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from .config import algorithm_params


def default_model_params(algorithm: str) -> dict[str, Any]:
    cfg = algorithm_params()
    if algorithm not in cfg:
        raise ValueError(f"Unsupported algorithm for defaults: {algorithm}")
    return dict(cfg[algorithm])


def fit_model_with_optional_early_stopping(
    model: Any,
    algorithm: str,
    X_tr,
    y_tr,
    X_va=None,
    y_va=None,
    early_stopping_rounds: int = 0,
) -> Any:
    if early_stopping_rounds <= 0 or X_va is None or y_va is None:
        model.fit(X_tr, y_tr)
        return model

    if algorithm == "catb":
        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=early_stopping_rounds,
            use_best_model=True,
            verbose=False,
        )
        return model

    if algorithm == "lgbm":
        try:
            import lightgbm as lgb

            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="average_precision",
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
            )
            return model
        except Exception:
            model.fit(X_tr, y_tr)
            return model

    if algorithm == "xgb":
        try:
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False,
                early_stopping_rounds=early_stopping_rounds,
            )
            return model
        except TypeError:
            model.fit(X_tr, y_tr)
            return model

    model.fit(X_tr, y_tr)
    return model


def build_model(algorithm: str, params: dict[str, Any]):
    merged = default_model_params(algorithm)
    merged.update(params)
    params = merged

    if algorithm == "catb":
        params.pop("verbose", None)
        params.pop("verbose_eval", None)
        params.pop("silent", None)
        params.setdefault("logging_level", "Silent")
        return CatBoostClassifier(**params)

    if algorithm == "lgbm":
        return LGBMClassifier(**params)

    if algorithm == "xgb":
        params.pop("use_label_encoder", None)
        return XGBClassifier(**params)

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def fit_pr_model(model, algorithm, X_train, y_train, X_valid, y_valid, patience=50):
    """Early stop against PR on CV validation only; return selected tree count."""
    if algorithm == 'catb':
        weights = model.get_params().get('class_weights', [1.0, 1.0])
        weighted = any(float(w) != 1.0 for w in weights)
        model.set_params(eval_metric='PRAUC:type=Classic' + (';use_weights=false' if weighted else ''))
        kwargs = dict(eval_set=(X_valid, y_valid), use_best_model=patience > 0, verbose=False)
        if patience > 0:
            kwargs['early_stopping_rounds'] = patience
        model.fit(X_train, y_train, **kwargs)
        return int(model.tree_count_)
    if algorithm == 'lgbm':
        import lightgbm as lgb
        model.set_params(metric='average_precision')
        callbacks = [lgb.early_stopping(patience, first_metric_only=True, verbose=False)] if patience > 0 else []
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=callbacks)
        return int(model.best_iteration_ or model.n_estimators)
    if algorithm == 'xgb':
        import inspect
        model.set_params(eval_metric='aucpr')
        kwargs = dict(eval_set=[(X_valid, y_valid)], verbose=False)
        if patience > 0:
            if 'early_stopping_rounds' in inspect.signature(model.fit).parameters:
                kwargs['early_stopping_rounds'] = patience
            else:
                model.set_params(early_stopping_rounds=patience)
        model.fit(X_train, y_train, **kwargs)
        return int(model.best_iteration + 1) if patience > 0 else int(model.n_estimators)
    raise ValueError(f'Unsupported algorithm: {algorithm}')
