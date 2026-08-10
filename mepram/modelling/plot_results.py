#!/usr/bin/env python3
"""Plot model results, evaluation curves, confusion matrices and processing summaries.

This template is intended for an analysis directory containing one or more
model-training runs like:

  01-training_5756602/
    level1_sepsis/
    level2_etiology/
    level3_cefalosporina/
    processing/

  01-training_5756741/
    ...

When the supplied path is a parent directory, every immediate child directory
whose name begins with ``01-`` is processed independently. Plots are written
under each ``<run_dir>/plots/`` directory.
"""

from __future__ import annotations

import argparse
import json
import re

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required: pip install pyyaml") from exc
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib as mpl
import shap
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelBinarizer


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def safe_str(value: Any) -> str:
    return str(value).replace(" ", "_").replace("/", "_").replace("\\", "_")


def display_labels_for_context(labels: list[str], context_name: str) -> list[str]:
    """Return plot-facing class labels without changing metric labels.

    - Level 1 binary sepsis: no sepsis / sepsis
    - Level 2 Stage 1 gate: cultivo - / cultivo +
    - Level 2 Stage 2 binary subtype: nonGNB / GNB
    """
    context = context_name.lower()
    labels_as_str = [str(label) for label in labels]

    # Level 1: sepsis
    if len(labels) == 2 and ("level1" in context or "sepsis" in context):
        if set(labels_as_str) == {"0", "1"}:
            mapping = {"0": "no sepsis", "1": "sepsis"}
            return [mapping[label] for label in labels_as_str]

    # Level 2 Stage 2 subtype must be checked BEFORE the gate branch.
    if len(labels) == 2 and any(
        token in context
        for token in ("stage2_subtype", "stage2", "subtype", "gnb")
    ):
        normalised = {
            label.lower().replace("_", "").replace("-", "").replace(" ", "")
            for label in labels_as_str
        }

        # Preserve semantic order when labels already contain subtype names.
        if normalised <= {"gnb", "nongnb"}:
            mapped = []
            for label in labels_as_str:
                key = label.lower().replace("_", "").replace("-", "").replace(" ", "")
                mapped.append("GNB" if key == "gnb" else "nonGNB")
            return mapped

        # Fallback for encoded binary subtype predictions.
        if set(labels_as_str) == {"0", "1"}:
            mapping = {"0": "nonGNB", "1": "GNB"}
            return [mapping[label] for label in labels_as_str]

    # Level 2 Stage 1 culture gate only.
    # Preserve the original confusion-matrix class order; only replace
    # plot-facing labels with clinically readable names.
    if len(labels) == 2 and any(
        token in context
        for token in ("stage1_gate", "gate")
    ):
        mapped = []
        for label in labels_as_str:
            key = (
                label.lower()
                .replace("_", "")
                .replace("-", "")
                .replace(" ", "")
            )

            if key in {"0", "gatenegative", "negative", "false"}:
                mapped.append("cultivo -")
            elif key in {"1", "gatepositive", "positive", "true"}:
                mapped.append("cultivo +")
            else:
                mapped.append(label)

        return mapped

    return labels_as_str

def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_binary_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_prefix: Path,
    positive_label: str | int = 1,
) -> None:
    y_true_binary = np.array(y_true)
    if y_true_binary.dtype == object:
        y_true_binary = np.array([1 if y == positive_label else 0 for y in y_true_binary])

    if len(np.unique(y_true_binary)) < 2:
        print(f"Skipping binary curves because only one class is present in y_true for {output_prefix}")
        return

    fpr, tpr, _ = roc_curve(y_true_binary, y_score)
    precision, recall, _ = precision_recall_curve(y_true_binary, y_score)

    baseline = float(np.mean(y_true_binary))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, label=f"PR (AUC={auc(recall, precision):.3f})")
    ax.axhline(
        baseline,
        linestyle="--",
        color="gray",
        linewidth=1,
        label=f"Baseline prevalence={baseline:.3f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    save_figure(fig, output_prefix.with_name(output_prefix.name + "_pr_curve.png"))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc(fpr, tpr):.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="No-skill line")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    save_figure(fig, output_prefix.with_name(output_prefix.name + "_roc_curve.png"))

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        fpr,
        tpr,
        label=f"ROC curve (AUC={auc(fpr, tpr):.3f})",
        linewidth=2,
    )

    ax.plot(
        recall,
        precision,
        label=f"PR curve (AUC={auc(recall, precision):.3f})",
        linewidth=2,
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1,
        label="ROC no-skill line",
    )

    ax.axhline(
        baseline,
        linestyle=":",
        color="gray",
        linewidth=1,
        label=f"PR baseline prevalence={baseline:.3f}",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate / Recall")
    ax.set_ylabel("True positive rate / Precision")
    ax.set_title("ROC and Precision-Recall Curves")
    ax.legend(loc="lower left", fontsize="small")
    ax.grid(alpha=0.25)

    save_figure(
        fig,
        output_prefix.with_name(output_prefix.name + "_roc_pr_overlay.png"),
    )


def plot_confusion(cm: np.ndarray, labels: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.tight_layout()
    save_figure(fig, output_path)


def plot_multiclass_curves(
    y_true: np.ndarray,
    proba_df: pd.DataFrame,
    class_labels: list[str],
    output_prefix: Path,
) -> None:
    labels = list(class_labels)
    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(y_true)
    if y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        labels = lb.classes_.tolist()

    if y_true_bin.shape[1] != proba_df.shape[1]:
        raise ValueError("Mismatch between number of classes and probability columns")

    fig, ax = plt.subplots(figsize=(8, 6))
    for index, class_name in enumerate(labels):
        if np.unique(y_true_bin[:, index]).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, index], proba_df.iloc[:, index])
        ax.plot(fpr, tpr, label=f"{class_name} (AUC={auc(fpr, tpr):.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multiclass ROC Curves")
    ax.legend(loc="lower right", fontsize="small")
    save_figure(fig, output_prefix.with_name(output_prefix.name + "_roc_curve.png"))

    fig, ax = plt.subplots(figsize=(8, 6))
    for index, class_name in enumerate(labels):
        if np.unique(y_true_bin[:, index]).size < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true_bin[:, index], proba_df.iloc[:, index])
        ax.plot(recall, precision, label=f"{class_name} (AUC={auc(recall, precision):.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Multiclass Precision-Recall Curves")
    ax.legend(loc="lower left", fontsize="small")
    save_figure(fig, output_prefix.with_name(output_prefix.name + "_pr_curve.png"))

    # Combined one-vs-rest ROC + PR overlay for multiclass targets (e.g. level-2 etiology).
    fig, ax = plt.subplots(figsize=(9, 7))
    plotted = False
    for index, class_name in enumerate(labels):
        if np.unique(y_true_bin[:, index]).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, index], proba_df.iloc[:, index])
        precision, recall, _ = precision_recall_curve(y_true_bin[:, index], proba_df.iloc[:, index])
        ax.plot(fpr, tpr, linewidth=2, label=f"ROC {class_name} (AUC={auc(fpr, tpr):.3f})")
        ax.plot(recall, precision, linestyle="--", linewidth=2, label=f"PR {class_name} (AUC={auc(recall, precision):.3f})")
        plotted = True

    if plotted:
        ax.plot([0, 1], [0, 1], linestyle=":", color="gray", linewidth=1, label="ROC no-skill line")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("False Positive Rate / Recall")
        ax.set_ylabel("True Positive Rate / Precision")
        ax.set_title("Multiclass ROC and Precision-Recall Curves")
        ax.legend(loc="best", fontsize="small", ncol=2)
        ax.grid(alpha=0.25)
        save_figure(fig, output_prefix.with_name(output_prefix.name + "_roc_pr_overlay.png"))
    else:
        plt.close(fig)


def plot_level_metrics_from_summary(summary_path: Path, out_dir: Path) -> None:
    data = load_json(summary_path)
    level_name = summary_path.parent.name
    targets = []
    if "target" in data:
        targets.append((level_name, data))
    else:
        for key, value in data.items():
            if isinstance(value, dict) and ("roc_auc" in value or "pr_auc" in value):
                targets.append((f"{level_name}_{key}", value))

    for name, summary in targets:
        metrics = {
            k: summary[k]
            for k in ["roc_auc", "pr_auc", "macro_f1", "precision", "recall"]
            if k in summary
        }
        if metrics:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.bar(metrics.keys(), metrics.values(), color=["#3b78b8", "#d1495b", "#edae49", "#76c893", "#4a4e69"][: len(metrics)])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Score")
            ax.set_title(f"Summary metrics for {name}")
            for i, value in enumerate(metrics.values()):
                ax.text(i, value + 0.02, f"{value:.3f}", ha="center")
            save_figure(fig, out_dir / f"{safe_str(name)}_summary_metrics.png")

        if "confusion_matrix" in summary:
            labels = []
            if "classes" in summary:
                labels = [str(x) for x in summary["classes"]]
            elif all(label in summary for label in ("negative_label", "positive_label")):
                labels = [str(summary["negative_label"]), str(summary["positive_label"])]
            else:
                labels = ["0", "1"]

            cm = np.array(summary["confusion_matrix"], dtype=int)

            # Prediction files are plotted before summaries. If a prediction-based
            # confusion matrix already exists for this level, keep that one and
            # skip the summary-derived duplicate. This applies to level 1, level 2, etc.
            summary_cm_path = out_dir / f"{safe_str(name)}_summary_confusion_matrix.png"
            existing_prediction_cm = [
                p for p in out_dir.glob("*_confusion_matrix.png")
                if not p.name.endswith("_summary_confusion_matrix.png")
            ]
            if existing_prediction_cm:
                continue

            plot_confusion(
                cm,
                display_labels_for_context(labels, name),
                summary_cm_path,
            )


def summarize_processing(
    run_dir: Path,
    out_dir: Path,
    plots_dir: Path,
    manifest: list[dict[str, Any]],
) -> None:
    processing_dir = run_dir / "processing"
    if not processing_dir.exists():
        return

    files = {
        "imputed_features.csv": plot_imputed_features,
        "nan_dropped_features.csv": plot_nan_dropped_features,
        "correlation_dropped_features.csv": plot_correlation_dropped_features,
        "iqr_outliers_train.csv": plot_iqr_outliers,
        "iqr_outliers_test.csv": plot_iqr_outliers,
        "l1_rfecv_features.csv": plot_rfecv_feature_list,
        "l2_gate_rfecv_features.csv": plot_rfecv_feature_list,
        "l2_sub_rfecv_features.csv": plot_rfecv_feature_list,
        "l2_direct_rfecv_features.csv": plot_rfecv_feature_list,
        "l3_rfecv_features.csv": plot_rfecv_feature_list,
    }

    for filename, plot_func in files.items():
        path = processing_dir / filename
        if not path.exists():
            continue
        before = _png_snapshot(out_dir)
        plot_func(path, out_dir)
        df = pd.read_csv(path)
        metrics: dict[str, Any] = {
            "rows": int(len(df)),
            "columns": [str(c) for c in df.columns],
        }
        if {"n_features", "roc_auc", "pr_auc"}.issubset(df.columns):
            best_pr_idx = df["pr_auc"].idxmax()
            best_roc_idx = df["roc_auc"].idxmax()
            metrics.update({
                "best_pr_auc": float(df.loc[best_pr_idx, "pr_auc"]),
                "best_pr_auc_n_features": int(df.loc[best_pr_idx, "n_features"]),
                "best_roc_auc": float(df.loc[best_roc_idx, "roc_auc"]),
                "best_roc_auc_n_features": int(df.loc[best_roc_idx, "n_features"]),
            })
        for col in ("number_of_imputation", "nan_percentage", "abs_spearman", "n_outliers"):
            if col in df.columns and not df.empty:
                metrics[f"max_{col}"] = float(df[col].max())
        _register_new_plots(
            manifest,
            before,
            out_dir,
            plots_dir,
            path.resolve(),
            metrics,
        )

def plot_imputed_features(path: Path, out_dir: Path) -> None:
    df = pd.read_csv(path)
    df = df.sort_values("number_of_imputation", ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.25 * len(df))))
    ax.barh(df["feature"], df["number_of_imputation"], color="#3b78b8")
    ax.invert_yaxis()
    ax.set_xlabel("Number of imputed values")
    ax.set_title("Top imputed features")
    save_figure(fig, out_dir / "processing_imputed_features.png")


def plot_nan_dropped_features(path: Path, out_dir: Path) -> None:
    df = pd.read_csv(path)
    df = df.sort_values("nan_percentage", ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.25 * len(df))))
    ax.barh(df["feature"], df["nan_percentage"], color="#d1495b")
    ax.invert_yaxis()
    ax.set_xlabel("Missing percentage")
    ax.set_title("Top features dropped by missingness")
    save_figure(fig, out_dir / "processing_nan_dropped_features.png")


def plot_correlation_dropped_features(path: Path, out_dir: Path) -> None:
    df = pd.read_csv(path)

    if "abs_spearman" not in df.columns:
        return

    top = df.sort_values("abs_spearman", ascending=False).head(50)

    fig, ax = plt.subplots(figsize=(10, max(5, 0.15 * len(top))))

    norm = Normalize(vmin=0.98, vmax=1.00)
    cmap = mpl.colormaps["Reds"]

    colors = cmap(norm(top["abs_spearman"]))

    ax.barh(
        top["feature_dropped"],
        top["abs_spearman"],
        color=colors,
    )

    ax.invert_yaxis()

    ax.set_xlim(0.98, 1.00)

    ax.set_xlabel("Absolute Spearman correlation")
    ax.set_title("Dropped correlated features")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("|Spearman|")
    
    plt.tight_layout()
    save_figure(fig, out_dir / "processing_correlation_dropped_features.png")


def plot_iqr_outliers(path: Path, out_dir: Path) -> None:
    df = pd.read_csv(path)

    if df.empty:
        print(f"Skipping IQR outlier plot for {path}: empty input file")
        return

    count_candidates = [
        "n_outliers",
        "n_outlier",
        "outlier_count",
        "outliers",
        "n_iqr_outliers",
        "iqr_outliers",
        "count",
    ]
    count_col = next((col for col in count_candidates if col in df.columns), None)

    if count_col is None:
        numeric_cols = [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col])
        ]
        count_col = numeric_cols[0] if numeric_cols else None

    if count_col is None:
        print(
            f"Skipping IQR outlier plot for {path}: no outlier-count column found. "
            f"Available columns: {list(df.columns)}"
        )
        return

    feature_col = "feature" if "feature" in df.columns else None
    if feature_col is None:
        feature_candidates = [col for col in df.columns if col != count_col]
        feature_col = feature_candidates[0] if feature_candidates else None

    if feature_col is None:
        df = df.copy()
        df["feature"] = df.index.astype(str)
        feature_col = "feature"

    df = df.sort_values(count_col, ascending=False).head(30)

    fig, ax = plt.subplots(figsize=(10, max(5, 0.25 * len(df))))
    ax.barh(df[feature_col].astype(str), df[count_col], color="#76c893")
    ax.invert_yaxis()
    ax.set_xlabel("Outlier count")
    ax.set_title(f"IQR outliers: {path.name}")
    plt.tight_layout()
    save_figure(fig, out_dir / f"processing_{path.stem}.png")


def get_run_max_features(path: Path, default: int = 170) -> int:
    run_dir = path.parents[1]  # processing/file.csv -> run_dir
    summary_path = run_dir / "aggregate_summary.json"

    if not summary_path.exists():
        return default

    try:
        data = load_json(summary_path)
        value = data.get("args", {}).get("max_features", default)
        return int(value)
    except Exception:
        return default


def plot_rfecv_feature_list(path: Path, out_dir: Path) -> None:
    df = pd.read_csv(path)

    required = {"n_features", "roc_auc", "pr_auc"}
    if not required.issubset(df.columns):
        print(f"Skipping RFECV performance plot for {path}: missing {required - set(df.columns)}")
        return

    df = df.sort_values("n_features")
    max_features = get_run_max_features(path, default=170)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(df["n_features"], df["roc_auc"], marker="o", linewidth=2, markersize=3, label="ROC-AUC")
    ax.plot(df["n_features"], df["pr_auc"], marker="o", linewidth=2, markersize=3, label="PR-AUC")

    # Best PR-AUC across all tested feature counts
    best_pr_idx = df["pr_auc"].idxmax()
    best_pr_n = int(df.loc[best_pr_idx, "n_features"])
    best_pr = float(df.loc[best_pr_idx, "pr_auc"])

    # Best PR-AUC within the feature cap
    capped_df = df[df["n_features"] <= max_features]
    if capped_df.empty:
        capped_df = df.copy()

    best_cap_idx = capped_df["pr_auc"].idxmax()
    best_cap_n = int(capped_df.loc[best_cap_idx, "n_features"])
    best_cap_pr = float(capped_df.loc[best_cap_idx, "pr_auc"])

    ax.axvline(
        max_features,
        linestyle="--",
        color="black",
        linewidth=1.5,
        alpha=0.8,
        label=f"Feature cap ({max_features})",
    )

    ax.scatter(
        best_pr_n,
        best_pr,
        marker="*",
        s=260,
        color="gold",
        edgecolor="black",
        linewidth=0.6,
        zorder=20,
        label=f"Best PR-AUC = {best_pr:.3f}",
    )

    same_best = best_pr_n == best_cap_n

    if not same_best:
        ax.scatter(
            best_cap_n,
            best_cap_pr,
            marker="o",
            color="red",
            edgecolor="black",
            linewidth=0.5,
            s=90,
            zorder=18,
            label=f"Best capped PR-AUC = {best_cap_pr:.3f}",
        )

    if same_best:
        ax.annotate(
            f"{best_pr_n} features",
            xy=(best_pr_n, best_pr),
            xytext=(8, 12),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
            arrowprops=dict(arrowstyle="->", color="gray", linewidth=0.8),
        )
    else:
        ax.annotate(
            f"{best_pr_n} features",
            xy=(best_pr_n, best_pr),
            xytext=(8, 12),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
            arrowprops=dict(arrowstyle="->", color="gray", linewidth=0.8),
        )

        ax.annotate(
            f"{best_cap_n} features",
            xy=(best_cap_n, best_cap_pr),
            xytext=(8, -36),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
            arrowprops=dict(arrowstyle="->", color="gray", linewidth=0.8),
        )

    ax.set_xlim(1, 170)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Number of features")
    ax.set_ylabel("Score")
    ax.set_title(f"RFECV performance: {path.stem}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)

    plt.tight_layout()

    save_figure(fig, out_dir / f"processing_{path.stem}_performance.png")


def infer_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if "true_label" in df.columns and "pred_label" in df.columns:
        y_true = df["true_label"].astype(str).to_numpy()
        y_pred = df["pred_label"].astype(str).to_numpy()
        raw_labels = list(dict.fromkeys(np.concatenate([y_true, y_pred]).tolist()))
        if set(raw_labels) == {"GNB", "non_GNB"}:
            labels = ["non_GNB", "GNB"]
        else:
            labels = raw_labels
    else:
        y_true = df["true"].to_numpy().astype(int)
        y_pred = df["pred"].to_numpy().astype(int)
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    return y_true, y_pred, labels


def infer_positive_label(labels: list[str], context_name: str) -> str | int:
    """Infer the clinically meaningful positive class for a binary target."""
    if len(labels) != 2:
        return labels[-1]

    context = context_name.lower()
    normalised = {
        str(label).lower().replace("_", "").replace("-", "").replace(" ", ""): label
        for label in labels
    }

    # Numeric / boolean encodings: class 1 is positive.
    for key in ("1", "true", "yes", "positive"):
        if key in normalised:
            return normalised[key]

    # Level 1: sepsis is the positive outcome.
    if "sepsis" in context or "level1" in context:
        for key, original in normalised.items():
            if key in {"sepsis", "withsepsis"} or ("sepsis" in key and "no" not in key and "non" not in key):
                return original

    # Level 3: antimicrobial resistance is the positive outcome.
    if any(token in context for token in ("level3", "resist", "cefalospor")):
        for key, original in normalised.items():
            if (
                key in {"r", "resistant", "resistente", "resistance", "resistencia"}
                or "resist" in key
            ) and not any(neg in key for neg in ("nonresist", "noresist", "suscept", "sensible")):
                return original

    # Level 2 gate/subtype defaults.
    if any(token in context for token in ("stage1", "gate")):
        for key, original in normalised.items():
            if key in {"cultivo+", "cultivopositivo", "gatepositive", "positive"} or "positive" in key:
                return original
    if any(token in context for token in ("stage2", "subtype", "gnb")):
        for key, original in normalised.items():
            if key == "gnb":
                return original

    return labels[-1]


def infer_scores(df: pd.DataFrame, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if "proba_positive" in df.columns:
        # Explicitly exported positive-class probability; never invert it.
        return df["proba_positive"].to_numpy().astype(float), np.array(labels)

    if "proba" in df.columns:
        # The generic ``proba`` column is ambiguous: some exports contain the
        # probability of class 0, while the plotting code evaluates class 1.
        # Its orientation is resolved later from the saved predicted labels.
        return df["proba"].to_numpy().astype(float), np.array(labels)

    prob_cols = [
        c for c in df.columns
        if c.startswith("proba_") and c != "proba_positive"
    ]

    if prob_cols:
        class_names = [c.replace("proba_", "", 1) for c in prob_cols]
        return df[prob_cols].to_numpy().astype(float), np.array(class_names)

    return np.array([]), np.array(labels)


def orient_binary_score_to_positive_class(
    y_true: np.ndarray,
    positive_label: str | int,
    y_score: np.ndarray,
    source_name: str,
    score_source: str,
) -> np.ndarray:
    """Ensure a one-dimensional binary score represents P(positive class).

    Binary prediction exports are not consistent: columns named ``proba`` or
    even ``proba_positive`` may contain the probability of the first estimator
    class rather than the clinically positive outcome. The orientation is
    resolved against ``y_true`` for the explicitly selected positive label.

    When the score is the complement of the positive-class probability, its
    ROC AUC is below 0.5. In that case ``1 - score`` is used. This changes only
    probability orientation; predicted labels and confusion matrices are not
    modified.
    """
    if y_score.ndim != 1 or y_score.size == 0:
        return y_score

    y_binary = (np.asarray(y_true).astype(str) == str(positive_label)).astype(int)
    if np.unique(y_binary).size < 2:
        return y_score

    finite = np.isfinite(y_score)
    if finite.sum() == 0 or np.unique(y_binary[finite]).size < 2:
        return y_score

    score_auc = float(roc_auc_score(y_binary[finite], y_score[finite]))
    if score_auc < 0.5:
        print(
            f"Inverting {score_source} for {source_name}: raw ROC-AUC={score_auc:.4f} "
            f"for positive class {positive_label!r}; using 1 - score"
        )
        return 1.0 - y_score

    print(
        f"Keeping {score_source} for {source_name}: ROC-AUC={score_auc:.4f} "
        f"for positive class {positive_label!r}"
    )
    return y_score


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return value


def _png_snapshot(directory: Path) -> set[Path]:
    if not directory.exists():
        return set()
    return {p.resolve() for p in directory.rglob("*.png")}


def _plot_type_from_name(path: Path) -> str:
    name = path.stem
    if name.endswith("_confusion_matrix"):
        return "confusion_matrix"
    if name.endswith("_roc_pr_overlay"):
        return "roc_pr_overlay"
    if name.endswith("_roc_curve"):
        return "roc_curve"
    if name.endswith("_pr_curve"):
        return "precision_recall_curve"
    if name.endswith("_summary_metrics"):
        return "summary_metrics"
    if "rfecv" in name and name.endswith("_performance"):
        return "rfecv_performance"
    if "imputed_features" in name:
        return "imputed_features"
    if "nan_dropped_features" in name:
        return "missingness_dropped_features"
    if "correlation_dropped_features" in name:
        return "correlation_dropped_features"
    if "iqr_outliers" in name:
        return "iqr_outliers"
    if "shap" in name and name.endswith("_bar"):
        return "shap_importance_bar"
    if "shap" in name and name.endswith("_beeswarm"):
        return "shap_beeswarm"
    return "plot"


def _register_new_plots(
    manifest: list[dict[str, Any]],
    before: set[Path],
    output_dir: Path,
    plots_dir: Path,
    source_file: Path,
    metrics: dict[str, Any],
) -> None:
    after = _png_snapshot(output_dir)
    for plot_path in sorted(after - before):
        manifest.append({
            "plot_file": str(plot_path.relative_to(plots_dir)),
            "plot_type": _plot_type_from_name(plot_path),
            "input_file": str(source_file),
            "metrics": _json_safe(metrics),
        })


def compute_prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    y_score: np.ndarray,
    class_names: np.ndarray,
    positive_label: str | int | None = None,
) -> dict[str, Any]:
    cmatrix = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
        output_dict=True,
    )
    metrics: dict[str, Any] = {
        "rows": int(len(y_true)),
        "labels": [str(x) for x in labels],
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": cmatrix.tolist(),
        "classification_report": report,
    }

    if y_score.size and y_score.ndim == 1 and len(labels) == 2:
        if positive_label is None:
            positive_label = labels[-1]
        y_binary = np.array([1 if y == positive_label else 0 for y in y_true])
        if np.unique(y_binary).size == 2:
            fpr, tpr, _ = roc_curve(y_binary, y_score)
            precision, recall, _ = precision_recall_curve(y_binary, y_score)
            metrics.update({
                "positive_label": str(positive_label),
                "roc_auc": float(auc(fpr, tpr)),
                "pr_auc": float(auc(recall, precision)),
                "positive_prevalence": float(np.mean(y_binary)),
            })
    elif y_score.size and y_score.ndim == 2 and y_score.shape[1] > 1:
        label_to_index = {str(label): i for i, label in enumerate(class_names)}
        per_class: dict[str, Any] = {}
        for class_name in class_names:
            class_name_str = str(class_name)
            idx = label_to_index[class_name_str]
            y_binary = (np.asarray(y_true).astype(str) == class_name_str).astype(int)
            if np.unique(y_binary).size < 2:
                continue
            fpr, tpr, _ = roc_curve(y_binary, y_score[:, idx])
            precision, recall, _ = precision_recall_curve(y_binary, y_score[:, idx])
            per_class[class_name_str] = {
                "roc_auc": float(auc(fpr, tpr)),
                "pr_auc": float(auc(recall, precision)),
                "prevalence": float(np.mean(y_binary)),
            }
        metrics["one_vs_rest"] = per_class

    return metrics


def plot_predictions(prediction_path: Path, out_dir: Path) -> dict[str, Any]:
    df = pd.read_csv(prediction_path)
    if df.empty:
        return {"rows": 0, "skipped": "empty input file"}

    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefix every prediction-derived plot with the prediction subfolder/level.
    # Examples:
    #   level1_sepsis_predictions_confusion_matrix.png
    #   level2_etiology_predictions_stage1_gate_roc_curve.png
    #   level3_cefalosporina_predictions_pr_curve.png
    level_name = safe_str(prediction_path.parent.name)

    # Remove the leading "predictions" token from plot filenames while
    # preserving any informative suffix, e.g.:
    #   predictions.csv -> level1_sepsis
    #   predictions_stage1_gate.csv -> level2_etiology_stage1_gate
    #   predictions_stage2_subtype.csv -> level2_etiology_stage2_subtype
    prediction_stem = re.sub(
        r"^predictions(?:[_-]+)?",
        "",
        prediction_path.stem,
        flags=re.IGNORECASE,
    )
    base_name = safe_str(prediction_stem).strip("_")
    plot_prefix = level_name if not base_name else f"{level_name}_{base_name}"

    y_true, y_pred, labels = infer_labels(df)
    context_name = f"{prediction_path.parent.name}_{base_name}"
    positive_label = infer_positive_label(labels, context_name)
    y_score, class_names = infer_scores(df, labels)
    if y_score.size and y_score.ndim == 1:
        if "proba_positive" in df.columns:
            score_source = "proba_positive"
        elif "proba" in df.columns:
            score_source = "proba"
        else:
            score_source = "binary probability score"
        y_score = orient_binary_score_to_positive_class(
            y_true=y_true,
            positive_label=positive_label,
            y_score=y_score,
            source_name=str(prediction_path),
            score_source=score_source,
        )

    cmatrix = confusion_matrix(y_true, y_pred, labels=labels)
    plot_confusion(
        cmatrix,
        display_labels_for_context(labels, context_name),
        out_dir / f"{plot_prefix}_confusion_matrix.png",
    )

    if y_score.size and y_score.ndim == 1:
        plot_binary_curves(
            y_true=y_true,
            y_score=y_score,
            output_prefix=out_dir / Path(plot_prefix),
            positive_label=positive_label,
        )
    elif y_score.size and y_score.ndim == 2:
        if y_score.shape[1] == 1:
            plot_binary_curves(
                y_true=y_true,
                y_score=y_score[:, 0],
                output_prefix=out_dir / Path(plot_prefix),
                positive_label=positive_label,
            )
        else:
            plot_multiclass_curves(
                y_true=y_true,
                proba_df=pd.DataFrame(y_score, columns=class_names),
                class_labels=class_names.tolist(),
                output_prefix=out_dir / Path(plot_prefix),
            )

    metrics = compute_prediction_metrics(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        y_score=y_score,
        class_names=class_names,
        positive_label=positive_label,
    )
    true_col = "true_label" if "true_label" in df.columns else "true"
    metrics["label_distribution"] = {
        str(k): int(v)
        for k, v in df[true_col].value_counts(dropna=False).items()
    }
    return metrics



def _first_npz_array(npz: np.lib.npyio.NpzFile, preferred_keys: tuple[str, ...] = ("shap_values", "values", "arr_0")) -> np.ndarray:
    """Return the first useful ndarray from an NPZ SHAP artifact."""
    for key in preferred_keys:
        if key in npz.files:
            return np.asarray(npz[key])
    for key in npz.files:
        array = np.asarray(npz[key])
        if array.size:
            return array
    raise ValueError("NPZ file does not contain any non-empty arrays")



def _infer_shap_feature_count(shap_values: np.ndarray, n_rows: int) -> int | None:
    """Infer the feature dimension in a SHAP array for a known row count."""
    values = np.asarray(shap_values)

    if values.ndim == 2:
        if values.shape[0] == n_rows:
            return int(values.shape[1])
        if values.shape[1] == n_rows:
            return int(values.shape[0])
        return None

    if values.ndim == 3:
        # Common layouts:
        #   samples x features x classes
        #   classes x samples x features
        #   samples x classes x features
        if values.shape[0] == n_rows:
            return int(max(values.shape[1], values.shape[2])) if 1 in values.shape[1:] else int(values.shape[1])
        if values.shape[1] == n_rows:
            return int(values.shape[2])
        return None

    return None


def _read_importance_feature_names(importance_path: Path) -> list[str]:
    """Read feature names from a SHAP importance CSV when possible."""
    if not importance_path.exists():
        return []

    try:
        importance_df = pd.read_csv(importance_path)
    except Exception as exc:
        print(f"Could not read SHAP importance feature names from {importance_path}: {exc}")
        return []

    feature_col_candidates = [
        "feature",
        "Feature",
        "feature_name",
        "feature_names",
        "variable",
        "Variable",
        "name",
    ]
    feature_col = next((col for col in feature_col_candidates if col in importance_df.columns), None)
    if feature_col is None:
        return []

    return [str(x) for x in importance_df[feature_col].dropna().tolist()]


def align_shap_X_to_values(
    x_df: pd.DataFrame,
    shap_values: np.ndarray,
    importance_path: Path,
    source_name: str,
) -> pd.DataFrame:
    """Align SHAP X columns to the SHAP value matrix.

    Some training exports write an identifier, target, or prediction column into
    *_shap_X.csv. In that case X has one or more extra columns compared with the
    SHAP value matrix. This function keeps the feature columns that correspond
    to the SHAP values so plotting can continue safely.
    """
    expected_n_features = _infer_shap_feature_count(shap_values, n_rows=x_df.shape[0])
    if expected_n_features is None:
        return x_df

    if x_df.shape[1] == expected_n_features:
        return x_df

    original_columns = [str(c) for c in x_df.columns]

    # Prefer the importance CSV because it is the most explicit feature list.
    importance_features = _read_importance_feature_names(importance_path)
    if importance_features:
        unique_importance_features = []
        seen = set()
        for feature in importance_features:
            if feature in original_columns and feature not in seen:
                unique_importance_features.append(feature)
                seen.add(feature)

        if len(unique_importance_features) == expected_n_features:
            print(
                f"Aligned {source_name}: selected {expected_n_features} X columns "
                "using the SHAP importance feature list"
            )
            return x_df.loc[:, unique_importance_features]

    # Drop common non-feature columns if doing so gives the right shape.
    non_feature_names = {
        "id", "ID", "index", "Index", "sample", "sample_id", "row", "row_id",
        "true", "true_label", "target", "y", "label", "pred", "pred_label",
        "proba", "proba_positive", "prediction", "probability",
    }
    kept_columns = [col for col in x_df.columns if str(col) not in non_feature_names]
    if len(kept_columns) == expected_n_features:
        dropped = [str(col) for col in x_df.columns if col not in kept_columns]
        print(
            f"Aligned {source_name}: dropped non-feature column(s) {dropped} "
            f"from SHAP X"
        )
        return x_df.loc[:, kept_columns]

    # Very common case: first column is an exported CSV index or identifier.
    if x_df.shape[1] == expected_n_features + 1:
        dropped = str(x_df.columns[0])
        print(
            f"Aligned {source_name}: SHAP values have {expected_n_features} features "
            f"but X has {x_df.shape[1]} columns; dropping first X column '{dropped}'"
        )
        return x_df.iloc[:, 1:].copy()

    # Last-resort deterministic fallback: keep the first expected feature columns.
    if x_df.shape[1] > expected_n_features:
        print(
            f"Aligned {source_name}: SHAP values have {expected_n_features} features "
            f"but X has {x_df.shape[1]} columns; keeping the first {expected_n_features} columns. "
            "Check *_shap_X.csv if these are not the model feature columns."
        )
        return x_df.iloc[:, :expected_n_features].copy()

    return x_df


def _normalise_shap_values(shap_values: np.ndarray, n_rows: int, n_features: int) -> dict[str, np.ndarray]:
    """Convert SHAP arrays into one or more 2D sample x feature matrices.

    Handles common layouts:
      - binary/regression: (n_samples, n_features)
      - multiclass: (n_samples, n_features, n_classes)
      - multiclass: (n_classes, n_samples, n_features)
    """
    values = np.asarray(shap_values)

    if values.ndim == 2:
        if values.shape == (n_rows, n_features):
            return {"overall": values}
        if values.shape == (n_features, n_rows):
            return {"overall": values.T}
        raise ValueError(f"Unsupported 2D SHAP shape {values.shape}; expected ({n_rows}, {n_features})")

    if values.ndim == 3:
        matrices: dict[str, np.ndarray] = {}
        if values.shape[0] == n_rows and values.shape[1] == n_features:
            # sample x feature x class
            for class_index in range(values.shape[2]):
                matrices[f"class_{class_index}"] = values[:, :, class_index]
            matrices["overall"] = np.mean(np.abs(values), axis=2)
            return matrices

        if values.shape[1] == n_rows and values.shape[2] == n_features:
            # class x sample x feature
            for class_index in range(values.shape[0]):
                matrices[f"class_{class_index}"] = values[class_index, :, :]
            matrices["overall"] = np.mean(np.abs(values), axis=0)
            return matrices

        if values.shape[0] == n_rows and values.shape[2] == n_features:
            # sample x class x feature
            for class_index in range(values.shape[1]):
                matrices[f"class_{class_index}"] = values[:, class_index, :]
            matrices["overall"] = np.mean(np.abs(values), axis=1)
            return matrices

    raise ValueError(f"Unsupported SHAP array shape {values.shape}")


def _top_shap_features(shap_matrix: np.ndarray, feature_names: list[str], max_display: int = 20) -> list[str]:
    mean_abs = np.nanmean(np.abs(shap_matrix), axis=0)
    order = np.argsort(mean_abs)[::-1][:max_display]
    return [feature_names[i] for i in order]


def plot_shap_importance_bar_from_csv(
    importance_path: Path,
    output_path: Path,
    title: str,
    max_display: int = 20,
) -> dict[str, Any]:
    """Plot absolute SHAP importance directly from *_shap_importance.csv.

    Expected columns include:
      - feature
      - mean_abs_shap

    The plot matches the saved SHAP importance table exactly and avoids
    cancellation from signed mean SHAP values.
    """
    if not importance_path.exists():
        raise FileNotFoundError(f"SHAP importance CSV not found: {importance_path}")

    importance_df = pd.read_csv(importance_path)
    if importance_df.empty:
        raise ValueError(f"SHAP importance CSV is empty: {importance_path}")

    feature_candidates = [
        "feature",
        "Feature",
        "feature_name",
        "feature_names",
        "variable",
        "Variable",
        "name",
    ]
    feature_col = next(
        (col for col in feature_candidates if col in importance_df.columns),
        None,
    )
    if feature_col is None:
        raise ValueError(
            f"No feature-name column found in {importance_path}. "
            f"Available columns: {list(importance_df.columns)}"
        )

    importance_candidates = [
        "mean_abs_shap",
        "mean_absolute_shap",
        "mean_abs_shap_value",
        "importance",
    ]
    importance_col = next(
        (col for col in importance_candidates if col in importance_df.columns),
        None,
    )
    if importance_col is None:
        raise ValueError(
            f"No absolute SHAP importance column found in {importance_path}. "
            f"Available columns: {list(importance_df.columns)}"
        )

    plot_df = importance_df[[feature_col, importance_col]].copy()
    plot_df[feature_col] = plot_df[feature_col].astype(str)
    plot_df[importance_col] = pd.to_numeric(
        plot_df[importance_col],
        errors="coerce",
    )
    plot_df = plot_df.dropna(subset=[feature_col, importance_col])

    # SHAP importance is absolute by definition here.
    plot_df[importance_col] = plot_df[importance_col].abs()

    top = (
        plot_df.sort_values(importance_col, ascending=False)
        .head(max_display)
        .iloc[::-1]
    )

    fig, ax = plt.subplots(figsize=(10, max(5, 0.30 * len(top))))
    ax.barh(
        top[feature_col],
        top[importance_col],
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    save_figure(fig, output_path)

    ranked = plot_df.sort_values(importance_col, ascending=False)
    return {
        "rows": int(len(importance_df)),
        "features": int(plot_df[feature_col].nunique()),
        "top_features": ranked[feature_col].head(10).astype(str).tolist(),
        "top_mean_abs_shap": [
            float(v) for v in ranked[importance_col].head(10).tolist()
        ],
        "importance_source": str(importance_path),
        "importance_column": importance_col,
    }

def plot_shap_beeswarm(
    shap_matrix: np.ndarray,
    x_df: pd.DataFrame,
    output_path: Path,
    title: str,
    max_display: int = 20,
    max_points_per_feature: int = 800,
) -> dict[str, Any]:
    feature_names = [str(c) for c in x_df.columns]
    if shap_matrix.shape != x_df.shape:
        raise ValueError(f"SHAP/X shape mismatch: SHAP {shap_matrix.shape}, X {x_df.shape}")

    mean_abs = np.nanmean(np.abs(shap_matrix), axis=0)
    order = np.argsort(mean_abs)[::-1][:max_display]
    ordered_indices = list(order)[::-1]
    ordered_features = [feature_names[i] for i in ordered_indices]

    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.34 * len(ordered_features))))

    for y_pos, feature_index in enumerate(ordered_indices):
        shap_vals = np.asarray(shap_matrix[:, feature_index], dtype=float)
        feature_vals = pd.to_numeric(x_df.iloc[:, feature_index], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(shap_vals)
        shap_vals = shap_vals[valid]
        feature_vals = feature_vals[valid]

        if shap_vals.size > max_points_per_feature:
            sample_idx = rng.choice(shap_vals.size, size=max_points_per_feature, replace=False)
            shap_vals = shap_vals[sample_idx]
            feature_vals = feature_vals[sample_idx]

        jitter = rng.normal(loc=0.0, scale=0.08, size=shap_vals.size)
        colors = feature_vals.copy()
        if not np.isfinite(colors).any():
            colors = np.zeros_like(shap_vals)

        ax.scatter(
            shap_vals,
            np.full(shap_vals.shape, y_pos) + jitter,
            c=colors,
            cmap=shap.plots.colors.red_blue,
            s=10,
            alpha=0.65,
            linewidths=0,
        )

    ax.axvline(0, color="gray", linewidth=1)
    ax.set_yticks(range(len(ordered_features)))
    ax.set_yticklabels(ordered_features)
    ax.set_xlabel("SHAP value")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)

    sm = cm.ScalarMappable(cmap=shap.plots.colors.red_blue)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Feature value")

    plt.tight_layout()
    save_figure(fig, output_path)

    return {
        "rows": int(shap_matrix.shape[0]),
        "features": int(shap_matrix.shape[1]),
        "top_features": [str(feature_names[i]) for i in order[:10]],
        "top_mean_abs_shap": [float(mean_abs[i]) for i in order[:10]],
    }


def plot_shap_artifact(shap_values_path: Path, plots_dir: Path) -> dict[str, Any]:
    """Create SHAP importance and beeswarm plots for one SHAP artifact prefix."""
    prefix = shap_values_path.name.replace("_shap_values.npz", "")
    shap_dir = shap_values_path.parent
    x_path = shap_dir / f"{prefix}_shap_X.csv"
    importance_path = shap_dir / f"{prefix}_shap_importance.csv"
    metadata_path = shap_dir / f"{prefix}_shap_metadata.json"

    if not x_path.exists():
        print(f"Skipping SHAP plots for {shap_values_path}: missing {x_path.name}")
        return {"skipped": "missing shap_X.csv"}

    x_df = pd.read_csv(x_path)
    if x_df.empty:
        print(f"Skipping SHAP plots for {shap_values_path}: empty X matrix")
        return {"skipped": "empty shap_X.csv"}

    with np.load(shap_values_path, allow_pickle=False) as npz:
        raw_values = _first_npz_array(npz)

    x_df = align_shap_X_to_values(
        x_df=x_df,
        shap_values=raw_values,
        importance_path=importance_path,
        source_name=str(shap_values_path),
    )

    matrices = _normalise_shap_values(
        raw_values,
        n_rows=x_df.shape[0],
        n_features=x_df.shape[1],
    )

    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            metadata = load_json(metadata_path)
        except Exception as exc:
            print(f"Could not read SHAP metadata {metadata_path}: {exc}")

    output_dir = plots_dir / shap_values_path.parent.parent.name / "shap"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = [str(c) for c in x_df.columns]
    metrics: dict[str, Any] = {
        "prefix": prefix,
        "rows": int(x_df.shape[0]),
        "features": int(x_df.shape[1]),
        "npz_shape": list(raw_values.shape),
        "metadata": metadata,
    }

    # Beeswarm uses the SHAP value matrix. The bar importance plot is read
    # directly from *_shap_importance.csv so it exactly reflects mean_abs_shap.
    matrix = matrices.get("overall", next(iter(matrices.values())))
    title_prefix = prefix.replace("_", " ")

    bar_metrics = plot_shap_importance_bar_from_csv(
        importance_path=importance_path,
        output_path=output_dir / f"{safe_str(prefix)}_shap_bar.png",
        title=f"SHAP importance: {title_prefix}",
    )
    beeswarm_metrics = plot_shap_beeswarm(
        matrix,
        x_df,
        output_dir / f"{safe_str(prefix)}_shap_beeswarm.png",
        title=f"SHAP values: {title_prefix}",
    )

    metrics.update({
        "bar": bar_metrics,
        "beeswarm": beeswarm_metrics,
    })

    # Keep a compact CSV-derived reference in the manifest when available.
    if importance_path.exists():
        try:
            importance_df = pd.read_csv(importance_path)
            metrics["importance_file_rows"] = int(len(importance_df))
            metrics["importance_file_columns"] = [str(c) for c in importance_df.columns]
        except Exception as exc:
            print(f"Could not read SHAP importance {importance_path}: {exc}")

    return metrics


def plot_shap_outputs(run_dir: Path, plots_dir: Path, manifest: list[dict[str, Any]]) -> None:
    shap_value_files = sorted(run_dir.glob("level*/shap/*_shap_values.npz"))
    if not shap_value_files:
        return

    print(f"Plotting SHAP artifacts from {run_dir}")
    for shap_values_path in shap_value_files:
        level_plot_dir = plots_dir / shap_values_path.parent.parent.name / "shap"
        before = _png_snapshot(level_plot_dir)
        try:
            metrics = plot_shap_artifact(shap_values_path, plots_dir)
        except Exception as exc:
            print(f"Skipping SHAP plots for {shap_values_path}: {exc}")
            metrics = {"skipped": str(exc)}
        _register_new_plots(
            manifest,
            before,
            level_plot_dir,
            plots_dir,
            shap_values_path.resolve(),
            metrics,
        )

def collect_prediction_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.rglob("predictions*.csv"))


def discover_run_directories(root_dir: Path, run_prefix: str = "01-") -> list[Path]:
    """Return model-training run directories below ``root_dir``.

    If ``root_dir`` itself starts with the requested prefix, it is treated as
    one run. Otherwise, only immediate child directories starting with the
    prefix are returned. Restricting discovery to immediate children prevents
    nested result folders from being processed more than once.
    """
    root_dir = root_dir.expanduser().resolve()

    if not root_dir.exists():
        raise FileNotFoundError(f"Run root not found: {root_dir}")
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Run root is not a directory: {root_dir}")

    if root_dir.name.startswith(run_prefix):
        return [root_dir]

    return sorted(
        (
            path.resolve()
            for path in root_dir.iterdir()
            if path.is_dir() and path.name.startswith(run_prefix)
        ),
        key=lambda path: path.name,
    )


def run_plotting(run_dir: Path, plots_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    plots_dir = plots_dir.resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    for prediction_path in collect_prediction_files(run_dir):
        level_plot_dir = plots_dir / prediction_path.parent.name
        print(f"Plotting {prediction_path} -> {level_plot_dir}")
        before = _png_snapshot(level_plot_dir)
        metrics = plot_predictions(prediction_path, level_plot_dir)
        _register_new_plots(
            manifest,
            before,
            level_plot_dir,
            plots_dir,
            prediction_path.resolve(),
            metrics,
        )

    summary_files = sorted(run_dir.rglob("summary.json"))
    for summary_path in summary_files:
        level_plot_dir = plots_dir / summary_path.parent.name
        print(f"Plotting summary {summary_path} -> {level_plot_dir}")
        before = _png_snapshot(level_plot_dir)
        plot_level_metrics_from_summary(summary_path, level_plot_dir)
        summary_data = load_json(summary_path)
        _register_new_plots(
            manifest,
            before,
            level_plot_dir,
            plots_dir,
            summary_path.resolve(),
            {"summary_metrics": summary_data},
        )

    print(f"Plotting processing tables from {run_dir / 'processing'}")
    summarize_processing(
        run_dir,
        plots_dir / "processing",
        plots_dir,
        manifest,
    )

    plot_shap_outputs(
        run_dir,
        plots_dir,
        manifest,
    )

    manifest_path = plots_dir / "plot_manifest.yaml"
    payload = {
        "run_directory": str(run_dir),
        "plots_directory": str(plots_dir),
        "n_plots": len(manifest),
        "plots": manifest,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            _json_safe(payload),
            handle,
            sort_keys=False,
            allow_unicode=True,
        )
    return manifest_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot model evaluation results and processing reports for every "
            "model-training folder matching a run prefix."
        )
    )
    parser.add_argument(
        "--run-dir",
        default=".",
        help=(
            "Parent directory containing run folders, or one individual run "
            "directory. Default: current directory."
        ),
    )
    parser.add_argument(
        "--run-prefix",
        default="01-",
        help="Prefix used to discover run directories. Default: 01-.",
    )
    parser.add_argument(
        "--plots-dir",
        default=None,
        help=(
            "Optional plot output path. For one run, this is the exact output "
            "directory. For multiple runs, one subdirectory per run is "
            "created below it. Default: <run_dir>/plots for every run."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(args.run_dir).expanduser().resolve()
    run_dirs = discover_run_directories(root_dir, args.run_prefix)

    if not run_dirs:
        raise FileNotFoundError(
            f"No immediate child directories starting with "
            f"{args.run_prefix!r} were found in {root_dir}"
        )

    shared_plots_root = (
        Path(args.plots_dir).expanduser().resolve()
        if args.plots_dir
        else None
    )
    multiple_runs = len(run_dirs) > 1
    completed: list[tuple[Path, Path, Path]] = []
    failed: list[tuple[Path, str]] = []

    print(
        f"Found {len(run_dirs)} model-training run(s) matching "
        f"{args.run_prefix!r} in {root_dir}"
    )
    for run_dir in run_dirs:
        print(f"\n{'=' * 72}\nProcessing run: {run_dir.name}\n{'=' * 72}")

        if shared_plots_root is None:
            plots_dir = run_dir / "plots"
        elif multiple_runs:
            plots_dir = shared_plots_root / run_dir.name
        else:
            plots_dir = shared_plots_root

        try:
            manifest_path = run_plotting(run_dir, plots_dir)
        except Exception as exc:
            failed.append((run_dir, str(exc)))
            print(f"Failed to plot {run_dir}: {exc}")
            continue

        completed.append((run_dir, plots_dir, manifest_path))
        print(f"Saved plots to {plots_dir}")
        print(f"Saved plot manifest to {manifest_path}")

    print(f"\n{'=' * 72}\nPLOTTING SUMMARY\n{'=' * 72}")
    print(f"Runs discovered : {len(run_dirs)}")
    print(f"Runs completed  : {len(completed)}")
    print(f"Runs failed     : {len(failed)}")

    for run_dir, plots_dir, manifest_path in completed:
        print(f"  OK     {run_dir.name}")
        print(f"         plots:    {plots_dir}")
        print(f"         manifest: {manifest_path}")

    for run_dir, error in failed:
        print(f"  FAILED {run_dir.name}: {error}")

    if failed:
        raise RuntimeError(
            f"Plotting failed for {len(failed)} of {len(run_dirs)} runs."
        )


if __name__ == "__main__":
    main()