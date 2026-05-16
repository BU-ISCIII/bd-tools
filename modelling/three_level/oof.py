"""Out-of-fold probability generation utilities."""

from .base import *
from .models import build_binary_model, build_multiclass_model, _fit_model

def generate_oof_probas_binary(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: Dict,
    model_type: str,
    n_splits: int,
    random_state: int,
) -> np.ndarray:
    """Return calibrated OOF positive-class probabilities (shape [n,]) for a binary classifier.

    Within each outer fold the base model is wrapped with CalibratedClassifierCV
    (isotonic, cv=3) so that the probabilities fed to the next cascade level are
    well-calibrated and consistent with what the final calibrated model will produce.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(X))
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        base = build_binary_model(model_type, best_params.copy())
        # cv=3 calibration within the training fold (never touches the OOF validation slice)
        model = _fit_calibrated_or_base(base, X_tr, y_tr, max_cv=3, method="isotonic")
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
    return oof


def generate_oof_probas_multiclass(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: Dict,
    model_type: str,
    num_classes: int,
    n_splits: int,
    random_state: int,
) -> np.ndarray:
    """Return calibrated OOF per-class probabilities (shape [n, num_classes]).

    Same calibration strategy as generate_oof_probas_binary.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros((len(X), num_classes))
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        base = build_multiclass_model(model_type, best_params.copy(), num_classes)
        model = _fit_calibrated_or_base(base, X_tr, y_tr, max_cv=3, method="isotonic")
        oof[va_idx] = model.predict_proba(X_va)
    return oof


