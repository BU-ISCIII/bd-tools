"""Out-of-fold helpers for staged mortality pipeline."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .models import build_model
from .utils import safe_cv_splits


def generate_oof_stage1_proba(
    X: pd.DataFrame,
    y: pd.Series,
    algorithm: str,
    params: dict,
    n_splits: int,
    seed: int,
) -> pd.Series:
    n_splits = safe_cv_splits(y, n_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = pd.Series(index=X.index, dtype=float, name="stage1_proba_oof")

    for tr_idx, va_idx in cv.split(X, y):
        X_tr = X.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]

        model = build_model(algorithm, params)
        model.fit(X_tr, y_tr)
        oof.iloc[va_idx] = model.predict_proba(X_va)[:, 1]

    return oof
