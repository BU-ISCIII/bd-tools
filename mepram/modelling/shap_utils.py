"""SHAP artifact generation utilities.

This module computes and saves numeric SHAP outputs.
It should not create plots. Plotting is handled by plot_results.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap


def extract_tree_estimator(model):
    """
    Extract one fitted base estimator from a calibrated ensemble.

    SHAP is computed on the underlying fitted tree model, not directly on the
    calibrated ensemble probability output.
    """

    if hasattr(model, "estimators") and len(model.estimators) > 0:
        calibrated = model.estimators[0]
    else:
        calibrated = model

    if hasattr(calibrated, "estimator"):
        return calibrated.estimator

    if hasattr(calibrated, "base_estimator"):
        return calibrated.base_estimator

    return calibrated


def shap_values_to_dict(
    shap_values,
    *,
    feature_names: list[str],
    class_names: Optional[list[str]] = None,
) -> dict[str, np.ndarray]:
    """
    Convert SHAP output into a dictionary of 2D arrays.

    Returns
    -------
    dict
        Binary/single-output:
            {"output": array[n_samples, n_features]}

        Multiclass:
            {"class_<label>": array[n_samples, n_features], ...}
    """

    if hasattr(shap_values, "values"):
        values = shap_values.values
    else:
        values = shap_values

    arrays: dict[str, np.ndarray] = {}

    if isinstance(values, list):
        for i, arr in enumerate(values):
            label = (
                str(class_names[i])
                if class_names is not None and i < len(class_names)
                else str(i)
            )
            arrays[f"class_{label}"] = np.asarray(arr)
        return arrays

    values = np.asarray(values)

    if values.ndim == 2:
        arrays["output"] = values
        return arrays

    if values.ndim == 3:
        n_features = len(feature_names)

        # Common format: samples x features x classes
        if values.shape[1] == n_features:
            for i in range(values.shape[2]):
                label = (
                    str(class_names[i])
                    if class_names is not None and i < len(class_names)
                    else str(i)
                )
                arrays[f"class_{label}"] = values[:, :, i]
            return arrays

        # Alternative format: samples x classes x features
        if values.shape[2] == n_features:
            for i in range(values.shape[1]):
                label = (
                    str(class_names[i])
                    if class_names is not None and i < len(class_names)
                    else str(i)
                )
                arrays[f"class_{label}"] = values[:, i, :]
            return arrays

    raise ValueError(f"Unsupported SHAP values shape: {values.shape}")


def save_shap_artifacts(
    *,
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: list[str],
    output_dir: Path,
    stage_name: str,
    class_names: Optional[list[str]] = None,
    background_size: int = 200,
    explain_size: int = 500,
    random_state: int = 42,
) -> None:
    """
    Compute and save SHAP values for later plotting.

    Saved files
    -----------
    <stage_name>_shap_values.npz
    <stage_name>_shap_X.csv
    <stage_name>_shap_importance.csv
    <stage_name>_shap_metadata.json
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train = X_train.loc[:, feature_names].copy()
    X_test = X_test.loc[:, feature_names].copy()

    if X_train.empty or X_test.empty:
        print(f"  Skipping SHAP artifacts for {stage_name}: empty train or test matrix.")
        return

    X_background = X_train.sample(
        n=min(background_size, len(X_train)),
        random_state=random_state,
    )

    X_explain = X_test.sample(
        n=min(explain_size, len(X_test)),
        random_state=random_state,
    )

    estimator = extract_tree_estimator(model)

    print(
        f"  Computing SHAP artifacts for {stage_name}: "
        f"{len(X_explain)} rows, {len(feature_names)} features."
    )

    try:
        explainer = shap.TreeExplainer(estimator)
        raw_shap_values = explainer.shap_values(X_explain)
        explainer_type = "TreeExplainer"
    except Exception as exc:
        print(
            f"  TreeExplainer failed for {stage_name}: "
            f"{type(exc).__name__}: {str(exc)[:120]}. Falling back to shap.Explainer."
        )
        explainer = shap.Explainer(estimator, X_background)
        raw_shap_values = explainer(X_explain)
        explainer_type = "Explainer"

    shap_arrays = shap_values_to_dict(
        raw_shap_values,
        feature_names=feature_names,
        class_names=class_names,
    )

    np.savez_compressed(
        output_dir / f"{stage_name}_shap_values.npz",
        **shap_arrays,
    )

    X_explain.to_csv(
        output_dir / f"{stage_name}_shap_X.csv",
        index=True,
        index_label="row_index",
    )

    importance_records = []

    for output_name, values in shap_arrays.items():
        if values.ndim != 2:
            raise ValueError(
                f"Expected 2D SHAP array for {output_name}, got {values.shape}"
            )

        mean_abs = np.mean(np.abs(values), axis=0)

        for feature, value in zip(feature_names, mean_abs):
            importance_records.append({
                "stage": stage_name,
                "output": output_name,
                "feature": feature,
                "mean_abs_shap": float(value),
            })

    importance_df = (
        pd.DataFrame(importance_records)
        .sort_values(["output", "mean_abs_shap"], ascending=[True, False])
        .reset_index(drop=True)
    )

    importance_df.to_csv(
        output_dir / f"{stage_name}_shap_importance.csv",
        index=False,
    )

    metadata = {
        "stage_name": stage_name,
        "feature_names": feature_names,
        "class_names": class_names,
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "n_explained_rows": int(len(X_explain)),
        "background_size": int(len(X_background)),
        "explainer_type": explainer_type,
        "outputs": list(shap_arrays.keys()),
        "note": (
            "SHAP values are computed on one fitted base estimator extracted "
            "from the calibrated ensemble. Final predictions are produced by "
            "the calibrated ensemble."
        ),
    }

    (output_dir / f"{stage_name}_shap_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"  Saved SHAP artifacts for {stage_name} to {output_dir}")