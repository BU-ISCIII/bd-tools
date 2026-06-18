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
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    save_figure(fig, output_prefix.with_name(output_prefix.name + "_roc_curve.png"))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, label=f"PR (AUC={auc(recall, precision):.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    save_figure(fig, output_prefix.with_name(output_prefix.name + "_pr_curve.png"))


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


def summarize_processing(run_dir: Path, out_dir: Path) -> None:
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
    }
    for filename, plot_func in files.items():
        path = processing_dir / filename
        if path.exists():
            plot_func(path, out_dir)


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
    if "abs_spearman" in df.columns:
        top = df.sort_values("abs_spearman", ascending=False).head(50)
        fig, ax = plt.subplots(figsize=(10, max(5, 0.15 * len(top))))
        ax.barh(top["feature_dropped"], top["abs_spearman"], color="#edae49")
        ax.invert_yaxis()
        ax.set_xlabel("Absolute Spearman correlation")
        ax.set_title("Dropped correlated features")
        save_figure(fig, out_dir / "processing_correlation_dropped_features.png")


def plot_iqr_outliers(path: Path, out_dir: Path) -> None:
    df = pd.read_csv(path)
    df = df.sort_values("n_outliers", ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.25 * len(df))))
    ax.barh(df["feature"], df["n_outliers"], color="#76c893")
    ax.invert_yaxis()
    ax.set_xlabel("Outlier count")
    ax.set_title(f"IQR outliers: {path.name}")
    save_figure(fig, out_dir / f"processing_{path.stem}.png")


def plot_rfecv_feature_list(path: Path, out_dir: Path) -> None:
    df = pd.read_csv(path, header=None, names=["feature"])
    fig, ax = plt.subplots(figsize=(10, max(4, 0.15 * len(df))))
    ax.barh(df["feature"], [1] * len(df), color="#4a4e69")
    ax.invert_yaxis()
    ax.set_xlabel("Selected feature")
    ax.set_title("RFECV selected features")
    ax.set_xticks([])
    save_figure(fig, out_dir / "processing_l1_rfecv_features.png")


def infer_binary_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if "true_label" in df.columns and "pred_label" in df.columns:
        y_true = df["true_label"].astype(str).to_numpy()
        y_pred = df["pred_label"].astype(str).to_numpy()
        labels = sorted(np.unique(np.concatenate([y_true, y_pred]).astype(str)).tolist())
    else:
        y_true = df["true"].to_numpy().astype(int)
        y_pred = df["pred"].to_numpy().astype(int)
        labels = ["0", "1"]
    return y_true, y_pred, labels


def infer_scores(df: pd.DataFrame, labels: list[str]) -> tuple[np.ndarray, np.ndarray | None]:
    if "proba" in df.columns:
        return df["proba"].to_numpy().astype(float), np.array(labels)
    prob_cols = [c for c in df.columns if c.startswith("proba_")]
    if prob_cols:
        proba_df = df[prob_cols].astype(float)
        classes = [c.replace("proba_", "") for c in prob_cols]
        return proba_df.to_numpy(), np.array(classes)
    return np.array([]), np.array(labels)


def write_classification_report(path: Path, y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> None:
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def plot_predictions(prediction_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(prediction_path)
    if df.empty:
        return

    level_name = prediction_path.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = prediction_path.stem
    summary_path = out_dir / f"{safe_str(base_name)}_dataset_summary.txt"

    y_true, y_pred, labels = infer_binary_labels(df)
    y_score, class_names = infer_scores(df, labels)
    write_classification_report(out_dir / f"{safe_str(base_name)}_classification_report.txt", y_true, y_pred, labels)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plot_confusion(cm, labels, out_dir / f"{safe_str(base_name)}_confusion_matrix.png")

    if y_score.size and y_score.ndim == 1:
        plot_binary_curves(y_true, y_score, out_dir / Path(f"{safe_str(base_name)}"), positive_label=labels[-1])
    elif y_score.size and y_score.ndim == 2:
        plot_multiclass_curves(y_true, pd.DataFrame(y_score, columns=class_names), class_names, out_dir / Path(f"{safe_str(base_name)}"))

    summary = {
        "file": str(prediction_path),
        "rows": len(df),
        "label_distribution": df["true_label"].value_counts(dropna=False).to_dict() if "true_label" in df.columns else df["true"].value_counts(dropna=False).to_dict(),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def collect_prediction_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.rglob("predictions*.csv"))


def run_plotting(run_dir: Path, plots_dir: Path) -> None:
    run_dir = run_dir.resolve()
    plots_dir = plots_dir.resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)

    for prediction_path in collect_prediction_files(run_dir):
        level_plot_dir = plots_dir / prediction_path.parent.name
        print(f"Plotting {prediction_path} -> {level_plot_dir}")
        plot_predictions(prediction_path, level_plot_dir)

    summary_files = sorted(run_dir.rglob("summary.json"))
    for summary_path in summary_files:
        level_plot_dir = plots_dir / summary_path.parent.name
        print(f"Plotting summary {summary_path} -> {level_plot_dir}")
        plot_level_metrics_from_summary(summary_path, level_plot_dir)

    print(f"Plotting processing tables from {run_dir / 'processing'}")
    summarize_processing(run_dir, plots_dir / "processing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot model evaluation results and processing reports.")
    parser.add_argument("run_dir", nargs="?", default="01-training_5756602", help="Path to the run directory containing level*/ and processing/ subfolders.")
    parser.add_argument("--plots-dir", default=None, help="Output directory for plots. Default is <run_dir>/plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    plots_dir = Path(args.plots_dir).expanduser().resolve() if args.plots_dir else run_dir / "plots"
    run_plotting(run_dir, plots_dir)
    print(f"Saved plots to {plots_dir}")


if __name__ == "__main__":
    main()
