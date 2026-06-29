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
