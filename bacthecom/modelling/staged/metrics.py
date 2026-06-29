"""Metrics and plotting helpers for staged mortality pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def binary_metrics(y_true: pd.Series, y_proba: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if y_true.nunique() > 1 else None,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
        "n": int(len(y_true)),
        "positive_rate": float(pd.Series(y_true).mean()),
        "predicted_positive": int(y_pred.sum()),
        "predicted_negative": int((y_pred == 0).sum()),
        "actual_positive": int(pd.Series(y_true).sum()),
        "actual_negative": int((pd.Series(y_true) == 0).sum()),
    }


def save_predictions(
    path: Path,
    index: pd.Index,
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> None:
    df = pd.DataFrame({"true": y_true.values, "proba": y_proba, "pred": (y_proba >= threshold).astype(int)}, index=index)
    df.to_csv(path, index_label="row_index")


def prettify_target_name(target: str) -> str:
    labels = {
        "mortalidad": "Mortality",
        "mortalidad_14_dias": "14-day mortality",
        "mortalidad_30_dias": "30-day mortality",
    }
    return labels.get(target, target.replace("_", " ").capitalize())


def apply_publication_axes_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=10, width=0.8, length=4)
    ax.grid(True, which="major", axis="both", alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, path_base: Path) -> None:
    fig.savefig(f"{path_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{path_base}.pdf", bbox_inches="tight")


def visualize_results(y_true: pd.Series, y_proba: np.ndarray, target: str, run_dir: Path, array_id: str) -> None:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    title_target = prettify_target_name(target)
    blue = "#1f77b4"
    dark_blue = "#0b3c5d"
    grey = "#6b7280"

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    baseline = float(pd.Series(y_true).mean())

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot(recall, precision, color=blue, linewidth=2.4, label=f"PR-AUC = {pr_auc:.3f}")
    ax.axhline(baseline, color=grey, linestyle="--", linewidth=1.1, label=f"Baseline = {baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-recall curve: {title_target}")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    apply_publication_axes_style(ax)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    save_figure(fig, plots_dir / f"pr_curve_{target}_array_{array_id}")
    plt.close(fig)

    if y_true.nunique() > 1:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        ax.plot(fpr, tpr, color=dark_blue, linewidth=2.4, label=f"ROC-AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], color=grey, linestyle="--", linewidth=1.1, label="Chance")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC curve: {title_target}")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        apply_publication_axes_style(ax)
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        save_figure(fig, plots_dir / f"roc_curve_{target}_array_{array_id}")
        plt.close(fig)

    y_pred = (y_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9, width=0.8, length=3)
    cbar.set_label("Count", rotation=270, labelpad=13)

    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion matrix: {title_target}")

    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm[i, j] > threshold else "#111827"
            ax.text(j, i, f"{cm[i, j]:,}\n({cm_percent[i, j]:.1f}%)", ha="center", va="center", color=text_color, fontsize=11, fontweight="bold")

    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", labelsize=10, width=0.8, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    save_figure(fig, plots_dir / f"confusion_matrix_{target}_array_{array_id}")
    plt.close(fig)
