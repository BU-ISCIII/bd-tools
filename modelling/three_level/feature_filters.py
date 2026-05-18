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
        history[str(len(remaining_features))] = {
            "metrics": metrics,
            "features": remaining_features.copy(),
        }
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
        if str(n_after_removal) not in history:
            history[str(n_after_removal)] = {"removed_feature": removed}
        else:
            history[str(n_after_removal)]["removed_feature"] = removed
        print(f"Removed feature: {removed}")

    # Score and record the final minimal feature set
    final_metrics = _cv_metrics(remaining_features)
    history[str(len(remaining_features))] = {
        "metrics": final_metrics,
        "features": remaining_features.copy(),
    }
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
                rfecv_records.append({
                    "n_features": n_feat_str,
                    "removed_feature": removed,
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

# ---------------------------------------------------------------------------
# Main training orchestration
# ---------------------------------------------------------------------------


def remove_correlated_features(
    X: pd.DataFrame,
    threshold: float = 0.90,
    output_csv_path: Optional[Path] = None,
) -> List[str]:
    """Return the subset of *X.columns* that survives a Spearman correlation filter.

    All feature columns are numeric after ``pd.get_dummies``, so Spearman rank
    correlation is a single valid measure for every column type:

    * **Binary 0/1** (one-hot dummies, presence/absence flags):
      Spearman = phi coefficient for binary pairs, which equals Pearson.
    * **Ordinal** (recoded vital signs 0-3, clinical scores 0-4):
      Spearman is designed for ordered discrete data.
    * **Continuous** (raw vital signs, labs, counts):
      Spearman is valid and more robust to outliers than Pearson.

    No Cramér's V is needed because string/categorical columns have already
    been one-hot encoded before this function is called.

    Algorithm
    ---------
    1. Sort columns by **descending variance** so the more informative
       representation wins when a pair must be pruned.  For example,
       ``temperatura`` (continuous, higher variance) beats
       ``temperatura_recoded`` (0-3 quartile bin, lower variance).
    2. Walk the sorted list; keep each column unless it is already
       scheduled for removal because it is too similar to an earlier-kept
       column.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix built from **training rows only**.  Test rows must
        not be included to avoid test-set leakage into the correlation
        estimates.
    threshold : float
        Absolute Spearman correlation above which a pair is considered
        redundant.  Default 0.90.

    Returns
    -------
    List[str]
        Column names to keep, preserved in their original order.
    """
    if threshold >= 1.0:
        return X.columns.tolist()

    # Filter to numeric columns only (exclude any object/string columns)
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) < len(X.columns):
        dropped_non_numeric = set(X.columns) - set(numeric_cols)
        print(f"  Removing {len(dropped_non_numeric)} non-numeric columns before correlation: {list(dropped_non_numeric)[:5]}...")
    X_numeric = X[numeric_cols]

    # Pairwise Spearman on training data.  Constant columns yield NaN
    # correlations; fill with 0 so they are treated as uncorrelated and
    # kept (SHAP can remove them later if they carry no information).
    corr = X_numeric.corr(method="spearman").abs().fillna(0.0)

    # Higher-variance columns are preferred. For equal variance, prefer
    # *_binary over *_counts so *_counts is dropped first.
    variances = X_numeric.var()
    def _suffix_priority(col: str) -> int:
        if col.endswith("_counts"):
            return 0
        if col.endswith("_binary"):
            return 2
        return 1
    col_order = sorted(
        variances.index.tolist(),
        key=lambda c: (-float(variances[c]), _suffix_priority(c), c),
    )

    dropped: set = set()
    kept_ordered: List[str] = []
    drop_audit: List[Dict[str, object]] = []
    for col in col_order:
        if col in dropped:
            continue
        kept_ordered.append(col)
        # Schedule every column that is too similar to *col* for removal.
        redundant = corr.index[corr[col] > threshold].tolist()
        for partner in redundant:
            if partner != col:
                if partner not in dropped:
                    dropped.add(partner)
                    drop_audit.append(
                        {
                            "kept_col": col,
                            "dropped_col": partner,
                            "spearman_abs_corr": float(corr.loc[col, partner]),
                            "kept_variance": float(variances[col]),
                            "dropped_variance": float(variances[partner]),
                        }
                    )

    if drop_audit:
        print("  Correlation-drop audit (kept -> dropped):")
        drop_audit_sorted = sorted(
            drop_audit,
            key=lambda d: (d["spearman_abs_corr"], d["kept_variance"]),
            reverse=True,
        )
        for item in drop_audit_sorted:
            print(
                "    "
                f"{item['kept_col']} -> {item['dropped_col']} | "
                f"abs_spearman={item['spearman_abs_corr']:.4f} | "
                f"var_kept={item['kept_variance']:.6g} | "
                f"var_dropped={item['dropped_variance']:.6g}"
            )
        
        # Save to CSV if output path provided
        if output_csv_path:
            output_csv_path.parent.mkdir(parents=True, exist_ok=True)
            df_audit = pd.DataFrame([
                {
                    "feature_dropped": item["dropped_col"],
                    "correlated_feature": item["kept_col"],
                    "abs_spearman": item["spearman_abs_corr"],
                    "var_kept": item["kept_variance"],
                    "var_dropped": item["dropped_variance"],
                }
                for item in drop_audit_sorted
            ])
            df_audit.to_csv(output_csv_path, index=False)
            print(f"  Saved correlation-drop audit to {output_csv_path}")

    # Return names in the original DataFrame column order.
    # Keep all numeric columns that survived correlation filter, plus all non-numeric columns
    kept_set = set(kept_ordered)
    non_numeric_cols = set(X.columns) - set(numeric_cols)
    kept_set.update(non_numeric_cols)
    return [c for c in X.columns if c in kept_set]


