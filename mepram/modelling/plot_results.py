#!/usr/bin/env python3
"""Plot model results, evaluation curves, confusion matrices and processing summaries.

This template is intended for analysis runs like:
  01-training_5756602/
    level1_sepsis/
    level2_hemo/
    level3_cefalosporina/
    processing/

It collects prediction files and summary JSON files, then writes plots under
<run_dir>/plots/.
"""

from __future__ import annotations

import argparse
import json

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
            plot_confusion(cm, labels, out_dir / f"{safe_str(name)}_summary_confusion_matrix.png")


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
    df = df.sort_values("n_outliers", ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.25 * len(df))))
    ax.barh(df["feature"], df["n_outliers"], color="#76c893")
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
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    else:
        y_true = df["true"].to_numpy().astype(int)
        y_pred = df["pred"].to_numpy().astype(int)
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    return y_true, y_pred, labels


def infer_scores(df: pd.DataFrame, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if "proba" in df.columns:
        return df["proba"].to_numpy().astype(float), np.array(labels)

    if "proba_positive" in df.columns:
        return df["proba_positive"].to_numpy().astype(float), np.array(labels)

    prob_cols = [
        c for c in df.columns
        if c.startswith("proba_") and c != "proba_positive"
    ]

    if prob_cols:
        class_names = [c.replace("proba_", "", 1) for c in prob_cols]
        return df[prob_cols].to_numpy().astype(float), np.array(class_names)

    return np.array([]), np.array(labels)


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
    base_name = prediction_path.stem

    y_true, y_pred, labels = infer_labels(df)
    y_score, class_names = infer_scores(df, labels)

    cmatrix = confusion_matrix(y_true, y_pred, labels=labels)
    plot_confusion(
        cmatrix,
        labels,
        out_dir / f"{safe_str(base_name)}_confusion_matrix.png",
    )

    if y_score.size and y_score.ndim == 1:
        plot_binary_curves(
            y_true=y_true,
            y_score=y_score,
            output_prefix=out_dir / Path(safe_str(base_name)),
            positive_label=labels[-1],
        )
    elif y_score.size and y_score.ndim == 2:
        if y_score.shape[1] == 1:
            plot_binary_curves(
                y_true=y_true,
                y_score=y_score[:, 0],
                output_prefix=out_dir / Path(safe_str(base_name)),
                positive_label=labels[-1],
            )
        else:
            plot_multiclass_curves(
                y_true=y_true,
                proba_df=pd.DataFrame(y_score, columns=class_names),
                class_labels=class_names.tolist(),
                output_prefix=out_dir / Path(safe_str(base_name)),
            )

    metrics = compute_prediction_metrics(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        y_score=y_score,
        class_names=class_names,
    )
    true_col = "true_label" if "true_label" in df.columns else "true"
    metrics["label_distribution"] = {
        str(k): int(v)
        for k, v in df[true_col].value_counts(dropna=False).items()
    }
    return metrics

def collect_prediction_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.rglob("predictions*.csv"))


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
    parser = argparse.ArgumentParser(description="Plot model evaluation results and processing reports.")
    parser.add_argument("--run-dir", nargs="?", default=None, help="Path to the run directory containing level*/ and processing/ subfolders.")
    parser.add_argument("--plots-dir", default=None, help="Output directory for plots. Default is <run_dir>/plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    plots_dir = Path(args.plots_dir).expanduser().resolve() if args.plots_dir else run_dir / "plots"
    manifest_path = run_plotting(run_dir, plots_dir)
    print(f"Saved plots to {plots_dir}")
    print(f"Saved plot manifest to {manifest_path}")


if __name__ == "__main__":
    main()
