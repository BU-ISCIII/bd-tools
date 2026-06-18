"""Feature-selection helpers (SHAP ranking and feature capping)."""

from .base import *

def _build_ranking_model(model_type: str, y: pd.Series, random_state: int) -> object:
    """Return a fast, fixed-param model of *model_type* used only for importance ranking.

    Parameters are intentionally conservative (shallow, few estimators) so the
    ranking step finishes quickly.  The result is never used for prediction.
    """
    n_classes = int(y.nunique())
    is_multi = n_classes > 2

    if model_type == "lgbm":
        params: Dict = {
            "n_estimators": 200, "num_leaves": 31, "max_depth": 6,
            "class_weight": "balanced", "n_jobs": N_CPUS,
            "random_state": random_state, "verbosity": -1,
        }
        if is_multi:
            params["objective"] = "multiclass"
            params["num_class"] = n_classes
        else:
            params["objective"] = "binary"
        return LGBMClassifier(**params)

    elif model_type == "xgb":
        if is_multi:
            return XGBClassifier(
                objective="multi:softprob", num_class=n_classes,
                n_estimators=200, max_depth=6,
                random_state=random_state, n_jobs=N_CPUS, verbosity=0,
            )
        n_pos = float((y == 1).sum())
        spw = float(len(y) - n_pos) / max(n_pos, 1.0)
        return XGBClassifier(
            objective="binary:logistic", n_estimators=200, max_depth=6,
            scale_pos_weight=spw,
            random_state=random_state, n_jobs=N_CPUS, verbosity=0,
        )

    elif model_type == "catb":
        return CatBoostClassifier(
            iterations=200, depth=6,
            auto_class_weights="Balanced", verbose=False, random_state=42, thread_count=N_CPUS
        )

    else:  # "rf" or anything else
        return RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight="balanced_subsample",
            random_state=random_state, n_jobs=N_CPUS,
        )


def _shap_importance(model, X: pd.DataFrame) -> np.ndarray:
    """Return mean absolute SHAP values (shape [n_features,]) using TreeExplainer.

    Handles binary (2-D array), multiclass list-of-arrays (XGB/RF style), and
    multiclass 3-D array (LGBM style) transparently.

    Raises ImportError if shap is not installed (caller handles the fallback).
    """
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)

    if isinstance(sv, list):
        # XGB / RF multiclass: list of (n_samples, n_features), one per class
        return np.mean([np.abs(a).mean(axis=0) for a in sv], axis=0)
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        # LGBM multiclass: (n_samples, n_features, n_classes)
        return np.abs(sv).mean(axis=(0, 2))
    # Binary: (n_samples, n_features)
    return np.abs(sv).mean(axis=0)


def cap_features(
    features: List[str],
    X: pd.DataFrame,
    y: pd.Series,
    max_features: Optional[int],
    random_state: int,
    model_type: Optional[str] = "rf",
    rank_model: Optional[BaseEstimator] = None,
) -> List[str]:
    """Trim *features* to the top *max_features* by mean |SHAP| value.

    A lightweight version of *model_type* (200 estimators, fixed depth) is
    fitted on *X[features]* solely to rank candidates – no hyperparameter
    tuning is performed.  Mean absolute SHAP values are used for ranking when
    the ``shap`` package is available; the model's own ``feature_importances_``
    are used as a fallback otherwise.

    Using the same model family as the one being trained avoids the MDI bias
    of Random Forest importance (bias toward high-cardinality / continuous
    features) and makes the selected subset consistent with what the final
    model will actually rely on.

    If *max_features* is None, or the feature list is already within budget,
    the original list is returned unchanged.
    """
    if max_features is None or len(features) <= max_features:
        return features
    if rank_model is None:
        model = _build_ranking_model(model_type, y, random_state)
    else:
        model = copy.deepcopy(rank_model)
    model.fit(X[features], y)

    try:
        raw = _shap_importance(model, X[features])
        method = "mean |SHAP|"
    except ImportError:
        raw = model.feature_importances_
        method = "feature_importances_ (install shap for SHAP-based ranking)"
    except Exception as exc:
        # TreeExplainer can fail for some model configurations; degrade gracefully
        print(f"  SHAP ranking failed ({exc}); falling back to feature_importances_.")
        raw = model.feature_importances_
        method = "feature_importances_"

    importance = pd.Series(raw, index=features)
    top_features = importance.nlargest(max_features).index.tolist()
    print(
        f"  Feature cap applied: {len(features)} → {max_features} features"
        f" (ranked by {method})."
    )
    return top_features

