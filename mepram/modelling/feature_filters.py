"""RFECV and correlation-based feature filtering utilities."""

from .base import *
from .models import build_binary_model, build_multiclass_model, _fit_model
from .feature_selection import _build_ranking_model, _shap_importance

def shap_rfecv(
        model: BaseEstimator,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        cv: int = 5,
        min_features: int = 5,
        max_features: int = 50,
        scoring: str = "roc_auc",
        random_state: int = 42,
        output_csv_path: Optional[Path] = None,
    ) -> Tuple[List[str], Dict[str, Dict[str, object]]]:
    """
    SHAP-based Recursive Feature Elimination with Cross Validation.

    At each step the CV score is recorded for the *current* feature set,
    then the globally least-important feature (by mean |SHAP| on the full
    training data) is removed.  After all iterations the feature set whose
    CV score was highest is returned together with the full score history.
    """
    remaining_features = list(X.columns)
    y = np.asarray(y)
    n_classes = len(np.unique(y))
    is_multiclass = n_classes > 2

    if n_classes < 2:
        print(
            "  Warning: shap_rfecv received a single-class target; "
            "skipping RFECV and returning all features."
        )
        return remaining_features, {}

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    history: Dict[str, Dict[str, object]] = {}

    def _cv_metrics(features: List[str]) -> Dict[str, float]:
        """Return mean CV metrics for *features* using the current skf splits."""
        if scoring not in ["roc_auc", "pr_auc"]:
            raise ValueError("Only roc_auc and pr_auc are supported for shap_rfecv")
        roc_auc_scores: List[float] = []
        pr_auc_scores: List[float] = []
        f1_scores: List[float] = []
        precision_scores: List[float] = []
        recall_scores: List[float] = []
        for train_idx, val_idx in skf.split(X[features], y):
            X_tr = X.iloc[train_idx][features]
            X_va = X.iloc[val_idx][features]
            y_tr, y_va = y[train_idx], y[val_idx]
            
            # Skip folds where training set has only one unique class (common in imbalanced data)
            if len(np.unique(y_tr)) < 2:
                print(f"    Skipping fold with single class in training set: {np.unique(y_tr)}")
                continue
            
            est = copy.deepcopy(model)
            try:
                est.fit(X_tr, y_tr)
                if is_multiclass:
                    proba = est.predict_proba(X_va)
                    # For multiclass, use macro one-vs-rest averages.
                    pr_auc = float(np.mean([
                        average_precision_score((y_va == i).astype(int), proba[:, i])
                        for i in range(n_classes)
                    ]))
                    roc_auc = float(roc_auc_score(y_va, proba, multi_class="ovr", average="macro"))
                    y_pred = np.argmax(proba, axis=1)
                else:
                    proba = est.predict_proba(X_va)[:, 1]
                    pr_auc = float(average_precision_score(y_va, proba))
                    roc_auc = float(roc_auc_score(y_va, proba))
                    y_pred = (proba >= 0.5).astype(int)
                f1 = f1_score(y_va, y_pred, average="macro", zero_division=0)
                precision = precision_score(y_va, y_pred, average="macro", zero_division=0)
                recall = recall_score(y_va, y_pred, average="macro", zero_division=0)
            except Exception as e:
                print(f"    Exception during fold fit/eval: {type(e).__name__}: {str(e)[:100]}")
                roc_auc = 0.0
                pr_auc = 0.0
                f1 = 0.0
                precision = 0.0
                recall = 0.0
            roc_auc_scores.append(float(roc_auc))
            pr_auc_scores.append(float(pr_auc))
            f1_scores.append(float(f1))
            precision_scores.append(float(precision))
            recall_scores.append(float(recall))
        if not roc_auc_scores:
            print("    Warning: no valid CV folds were available for shap_rfecv; returning zeroed metrics.")
            return {
                "roc_auc": 0.0,
                "pr_auc": 0.0,
                "f1_macro": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
            }
        return {
            "roc_auc": float(np.mean(roc_auc_scores)),
            "pr_auc": float(np.mean(pr_auc_scores)),
            "f1_macro": float(np.mean(f1_scores)),
            "precision_macro": float(np.mean(precision_scores)),
            "recall_macro": float(np.mean(recall_scores)),
        }

    def _global_worst_feature(features: List[str]) -> str:
        """Fit on all training data; return the least-important feature by mean |SHAP|."""
        est = copy.deepcopy(model)
        est.fit(X[features], y)
        try:
            explainer = shap.TreeExplainer(est)
        except Exception:
            explainer = shap.Explainer(est, X[features])
        sv = explainer.shap_values(X[features])
        if isinstance(sv, list):
            # XGB / RF multiclass: list of (n_samples, n_features), one per class
            shap_matrix = np.mean([np.abs(a) for a in sv], axis=0)
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            # LGBM multiclass: (n_samples, n_features, n_classes)
            shap_matrix = np.abs(sv).mean(axis=2)
        else:
            shap_matrix = np.abs(sv)
        importance = np.mean(shap_matrix, axis=0)
        return features[int(np.argmin(importance))]

    while len(remaining_features) > min_features:
        # Score the CURRENT feature set — label and score are in sync
        metrics = _cv_metrics(remaining_features)
        entry = history.get(str(len(remaining_features)), {})
        entry.update({
            "metrics": metrics,
            "features": remaining_features.copy(),
        })
        history[str(len(remaining_features))] = entry
        primary_metric = "pr_auc" if scoring == "pr_auc" else "roc_auc"
        print(
            f"Features: {len(remaining_features)} | "
            f"ROC_AUC: {metrics['roc_auc']:.4f} | PR_AUC: {metrics['pr_auc']:.4f} | "
            f"F1: {metrics['f1_macro']:.4f} | "
            f"Precision: {metrics['precision_macro']:.4f} | Recall: {metrics['recall_macro']:.4f}"
        )

        # Remove the globally least important feature
        removed = _global_worst_feature(remaining_features)
        remaining_features.remove(removed)
        # Track the removed feature in history for CSV export
        n_after_removal = len(remaining_features)
        entry = history.get(str(n_after_removal), {})
        entry["removed_feature"] = removed
        history[str(n_after_removal)] = entry
        print(f"Removed feature: {removed}")

    # Score and record the final minimal feature set
    final_metrics = _cv_metrics(remaining_features)
    entry = history.get(str(len(remaining_features)), {})
    entry.update({
        "metrics": final_metrics,
        "features": remaining_features.copy(),
    })
    history[str(len(remaining_features))] = entry
    primary_metric = "pr_auc" if scoring == "pr_auc" else "roc_auc"
    print(
        f"Features: {len(remaining_features)} | "
        f"ROC_AUC: {final_metrics['roc_auc']:.4f} | PR_AUC: {final_metrics['pr_auc']:.4f} | "
        f"F1: {final_metrics['f1_macro']:.4f} | "
        f"Precision: {final_metrics['precision_macro']:.4f} | Recall: {final_metrics['recall_macro']:.4f}"
    )

    # Return the feature set whose CV score was highest (true RFECV selection),
    # constrained to at most max_features.  Keys are strings; cast before comparing.
    primary_metric = "pr_auc" if scoring == "pr_auc" else "roc_auc"
    best_n = max(
        history,
        key=lambda k: history[k]["metrics"][primary_metric] if int(k) <= max_features else 0.0,
    )
    best_score = history[best_n]["metrics"][primary_metric]
    best_features = history[best_n]["features"]
    print(
        f"Best CV score ({primary_metric}): {best_score:.4f} at {best_n} features → "
        f"Selected {len(best_features)} features."
    )
    
    # Export RFECV history to CSV if output path provided
    if output_csv_path:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        rfecv_records = []
        for n_feat_str in sorted(history.keys(), key=lambda x: -int(x)):
            entry = history[n_feat_str]
            if "metrics" in entry:
                metrics = entry["metrics"]
                removed = entry.get("removed_feature", None)
                features = entry.get("features")
                rfecv_records.append({
                    "n_features": n_feat_str,
                    "removed_feature": removed,
                    "features": ",".join(features) if features is not None else None,
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "f1_macro": metrics["f1_macro"],
                    "precision_macro": metrics["precision_macro"],
                    "recall_macro": metrics["recall_macro"],
                })
        df_rfecv = pd.DataFrame(rfecv_records)
        df_rfecv.to_csv(output_csv_path, index=False)
        print(f"  Saved RFECV history to {output_csv_path}")
    
    return best_features, history

def remove_correlated_features(
    X: pd.DataFrame,
    threshold: float = 0.99,
    method: str = "spearman",
    output_csv_path: Optional[Path] = None,
) -> List[str]:
    """
    Remove highly correlated numeric features.

    Strategy:
    - Non-numeric columns are ignored for correlation but kept in the returned feature list.
    - For each pair with abs(correlation) > threshold, drop the feature with lower variance.
    - Save an audit CSV if output_csv_path is provided.
    - Return list of kept columns.
    """

    if X.empty or X.shape[1] <= 1:
        kept_cols = X.columns.tolist()

        if output_csv_path is not None:
            output_csv_path = Path(output_csv_path)
            output_csv_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=[
                "kept_feature",
                "dropped_feature",
                "abs_correlation",
                "variance_kept",
                "variance_dropped",
                "threshold",
                "method",
            ]).to_csv(output_csv_path, index=False)

        return kept_cols

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    non_numeric_cols = [
        col for col in X.columns
        if col not in numeric_cols
    ]

    if non_numeric_cols:
        print(
            f"  Removing {len(non_numeric_cols)} non-numeric columns before correlation: "
            f"{non_numeric_cols[:5]}..."
        )

    if len(numeric_cols) <= 1:
        kept_cols = X.columns.tolist()

        if output_csv_path is not None:
            output_csv_path = Path(output_csv_path)
            output_csv_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=[
                "kept_feature",
                "dropped_feature",
                "abs_correlation",
                "variance_kept",
                "variance_dropped",
                "threshold",
                "method",
            ]).to_csv(output_csv_path, index=False)

        return kept_cols

    X_num = X[numeric_cols].copy()

    corr = X_num.corr(method=method).abs()

    upper = corr.where(
        np.triu(np.ones(corr.shape), k=1).astype(bool)
    )

    variances = X_num.var(axis=0, skipna=True)

    dropped = set()
    audit_records = []

    for col in upper.columns:
        if col in dropped:
            continue

        high_corr_features = upper.index[
            upper[col] > threshold
        ].tolist()

        for other_col in high_corr_features:
            if other_col in dropped or col in dropped:
                continue

            corr_value = float(upper.loc[other_col, col])

            var_col = float(variances[col])
            var_other = float(variances[other_col])

            if var_col >= var_other:
                kept_feature = col
                dropped_feature = other_col
                variance_kept = var_col
                variance_dropped = var_other
            else:
                kept_feature = other_col
                dropped_feature = col
                variance_kept = var_other
                variance_dropped = var_col

            dropped.add(dropped_feature)

            audit_records.append({
                "kept_feature": kept_feature,
                "dropped_feature": dropped_feature,
                "abs_correlation": corr_value,
                "variance_kept": variance_kept,
                "variance_dropped": variance_dropped,
                "threshold": threshold,
                "method": method,
            })

            print(
                f"    {kept_feature} -> {dropped_feature} | "
                f"abs_{method}={corr_value:.4f} | "
                f"var_kept={variance_kept:.5g} | "
                f"var_dropped={variance_dropped:.5g}"
            )

    kept_cols = [
        col for col in X.columns
        if col not in dropped
    ]

    if output_csv_path is not None:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        audit_df = pd.DataFrame(
            audit_records,
            columns=[
                "kept_feature",
                "dropped_feature",
                "abs_correlation",
                "variance_kept",
                "variance_dropped",
                "threshold",
                "method",
            ],
        )

        audit_df.to_csv(output_csv_path, index=False)

        print(
            f"  Saved correlation-drop audit to {output_csv_path}"
        )

    print(
        f"  Dropped {len(dropped)} redundant features → {len(kept_cols)} remain."
    )

    return kept_cols

def fit_iqr_bounds(
    X: pd.DataFrame,
    columns: Optional[List[str]] = None,
    iqr_multiplier: float = 3.0,
) -> Dict[str, Dict[str, float]]:
    """
    Fit IQR outlier bounds on training data.

    Bounds are computed as:
        lower = Q1 - iqr_multiplier * IQR
        upper = Q3 + iqr_multiplier * IQR

    Returns a dictionary that can be reused on validation/test data.
    """

    if columns is None:
        columns = X.select_dtypes(include=[np.number]).columns.tolist()

    bounds = {}

    for column in columns:
        if column not in X.columns:
            continue

        values = pd.to_numeric(
            X[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr

        bounds[column] = {
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower": float(lower),
            "upper": float(upper),
        }

    return bounds


def apply_iqr_bounds_to_nan(
    X: pd.DataFrame,
    bounds: Dict[str, Dict[str, float]],
    output_csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Apply precomputed IQR bounds to a dataframe.

    Values below lower or above upper are replaced with NaN.

    If output_csv_path is provided, save a CSV audit table with:
    - row_index
    - feature
    - value
    - lower
    - upper
    """

    X_clean = X.copy()
    outlier_records = []

    for column, info in bounds.items():
        if column not in X_clean.columns:
            continue

        lower = info["lower"]
        upper = info["upper"]

        values = pd.to_numeric(
            X_clean[column],
            errors="coerce",
        )

        outlier_mask = (
            (values < lower)
            | (values > upper)
        )

        if outlier_mask.any():
            outlier_rows = X_clean.loc[outlier_mask, [column]].copy()

            for row_index, value in outlier_rows[column].items():
                outlier_records.append({
                    "row_index": row_index,
                    "feature": column,
                    "value": value,
                    "lower": lower,
                    "upper": upper,
                })

            X_clean.loc[outlier_mask, column] = np.nan

    if output_csv_path is not None:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pd.DataFrame(outlier_records).to_csv(
            output_csv_path,
            index=False,
        )

    return X_clean

# ---------------------------------------------------------------------------
# Main training orchestration
# ---------------------------------------------------------------------------