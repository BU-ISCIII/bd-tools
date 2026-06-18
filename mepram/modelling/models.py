"""Model builders and fitting helpers."""

import contextlib
import io

from .base import *
from .config import algorithm_params
from sklearn.metrics import precision_recall_curve, roc_curve

_ALGO_PARAMS = algorithm_params()


def _with_fixed_params(model_type: str, task: str, params: Dict) -> Dict:
    fixed = _ALGO_PARAMS.get(model_type, {}).get(task, {})
    merged = dict(fixed)
    merged.update(params)
    return merged


def build_binary_model(model_type: str, params: Dict) -> object:
    if model_type == "xgb":
        return XGBClassifier(
            **_with_fixed_params(model_type, "binary", params)
        )

    if model_type == "lgbm":
        return LGBMClassifier(
            **_with_fixed_params(model_type, "binary", params)
        )

    if model_type == "rf":
        return RandomForestClassifier(
            **_with_fixed_params(model_type, "binary", params)
        )

    if model_type == "catb":
        params = _with_fixed_params(model_type, "binary", params)

        params.setdefault("logging_level", "Silent")
        params.setdefault("thread_count", N_CPUS)
        params.setdefault("random_state", 42)

        return CatBoostClassifier(**params)

    raise ValueError(f"Unknown model_type '{model_type}'.")


def build_multiclass_model(model_type: str, params: Dict, num_classes: int) -> object:
    params = _with_fixed_params(model_type, "multiclass", params)

    if model_type == "xgb":
        params.setdefault("objective", "multi:softprob")
        params["num_class"] = num_classes
        return XGBClassifier(**params)

    if model_type == "lgbm":
        params.setdefault("objective", "multiclass")
        params["num_class"] = num_classes
        return LGBMClassifier(**params)

    if model_type == "rf":
        params.setdefault("class_weight", "balanced")
        return RandomForestClassifier(**params)

    if model_type == "catb":
        # CatBoost infers class count from y; num_class is not a valid CatBoost parameter.
        params.pop("num_class", None)
        params.setdefault("loss_function", "MultiClass")
        params.setdefault("auto_class_weights", "Balanced")
        params.setdefault("thread_count", N_CPUS)
        params.setdefault("random_state", 42)
        params.setdefault("logging_level", "Silent")

        params.pop("custom_metric", None)

        return CatBoostClassifier(**params)

    raise ValueError(f"Unsupported model type for multiclass: '{model_type}'.")


def _fit_model(model, model_type: str, X_tr, y_tr, X_va=None, y_va=None, sample_weight=None) -> None:
    """Fit a model; CatBoost uses early-stopping with the validation fold."""
    sw = sample_weight

    if model_type == "catb" and X_va is not None:
        n_iter = getattr(model, "iterations", 500)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model.fit(
                X_tr,
                y_tr,
                eval_set=(X_va, y_va),
                early_stopping_rounds=max(20, int(0.05 * n_iter)),
                verbose=False,
                sample_weight=sw,
            )
    else:
        if sw is not None:
            model.fit(X_tr, y_tr, sample_weight=sw)
        else:
            model.fit(X_tr, y_tr)


def _fit_calibrated_or_base(
    base_model,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    *,
    max_cv: int = 5,
    method: str = "isotonic",
):
    """Fit calibrated model when feasible; otherwise fit and return base model."""
    class_counts = pd.Series(y_tr).value_counts()

    if class_counts.empty or int(class_counts.min()) < 2:
        print(
            "  Warning: skipping calibration because at least one class has <2 samples "
            f"(counts={class_counts.to_dict()})."
        )
        base_model.fit(X_tr, y_tr)
        return base_model

    cal_cv = min(max_cv, int(class_counts.min()))
    model = CalibratedClassifierCV(base_model, cv=cal_cv, method=method)
    model.fit(X_tr, y_tr)
    return model


def _find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Find the probability threshold that maximises Youden's J."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx])

def _find_threshold_for_recall(y_true, proba, target_recall=0.85):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)

    valid = np.where(recall[:-1] >= target_recall)[0]
    if len(valid) == 0:
        return 0.5

    # Among thresholds achieving target recall, maximize precision
    best_idx = valid[np.argmax(precision[valid])]
    return float(thresholds[best_idx])