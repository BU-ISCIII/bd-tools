"""Homogeneous output contracts and aggregate benchmark reporting."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .types import BenchmarkJob, DiagnosticResult, ResourceLimits


def job_output_dir(base_output_dir: Path, job: BenchmarkJob) -> Path:
    return base_output_dir / "jobs" / job.slug


def write_diagnostics_manifest(
    output_dir: Path, diagnostics: Iterable[DiagnosticResult]
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "diagnostics_manifest.json"
    path.write_text(
        json.dumps([asdict(item) for item in diagnostics], indent=2),
        encoding="utf-8",
    )
    return path


def write_job_summary(
    output_dir: Path,
    *,
    job: BenchmarkJob,
    status: str,
    resources: ResourceLimits,
    metrics: Dict[str, Any] | None = None,
    diagnostic_results: List[DiagnosticResult] | None = None,
    extra: Dict[str, Any] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "status": status,
        "target": asdict(job.target),
        "model": job.model_name,
        "feature_policy": asdict(job.feature_policy),
        "feature_view": asdict(job.feature_view),
        "feature_set": asdict(job.feature_set),
        "job_index": job.index,
        "resources": asdict(resources),
        "metrics": metrics or {},
        "diagnostics": [asdict(item) for item in diagnostic_results or []],
    }
    if extra:
        payload.update(extra)
    path = output_dir / "summary.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def build_aggregate_report(base_output_dir: Path) -> Dict[str, Any]:
    """Aggregate completed benchmark job outputs into comparison artifacts."""
    reports_dir = base_output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    summaries = _load_job_summaries(base_output_dir)
    comparison = _comparison_table(summaries)
    comparison_path = reports_dir / "job_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    rankings = _ranking_table(comparison)
    rankings_path = reports_dir / "model_rankings.csv"
    rankings.to_csv(rankings_path, index=False)

    plot_files: list[str] = []
    plot_files.extend(_write_validation_test_plots(comparison, reports_dir))

    class_rows = []
    diagnostic_rows = []
    for item in summaries:
        summary = item["summary"]
        job_dir = item["job_dir"]
        for split in ["validation", "test"]:
            predictions_path = job_dir / (
                "validation_predictions.csv"
                if split == "validation"
                else "test_predictions.csv"
            )
            if not predictions_path.exists():
                continue
            predictions = pd.read_csv(predictions_path)
            class_rows.extend(_class_level_rows(summary, split, predictions))
            diagnostic_rows.extend(
                _write_prediction_diagnostics(summary, split, predictions, reports_dir)
            )

    class_metrics = pd.DataFrame(class_rows)
    class_metrics_path = reports_dir / "class_level_metrics.csv"
    class_metrics.to_csv(class_metrics_path, index=False)

    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics_path = reports_dir / "diagnostic_artifacts.csv"
    diagnostics.to_csv(diagnostics_path, index=False)

    manifest = {
        "base_output_dir": str(base_output_dir),
        "reports_dir": str(reports_dir),
        "n_jobs": int(len(summaries)),
        "files": {
            "job_comparison": str(comparison_path),
            "model_rankings": str(rankings_path),
            "class_level_metrics": str(class_metrics_path),
            "diagnostic_artifacts": str(diagnostics_path),
            "plots": plot_files,
        },
    }
    manifest_path = reports_dir / "report_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["files"]["manifest"] = str(manifest_path)
    return manifest


def _load_job_summaries(base_output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for summary_path in sorted((base_output_dir / "jobs").glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "completed":
            continue
        rows.append({"summary": summary, "job_dir": summary_path.parent})
    return rows


def _comparison_table(items: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in items:
        summary = item["summary"]
        target = summary.get("target", {})
        feature_view = summary.get("feature_view", {})
        feature_set = summary.get("feature_set", {})
        main_metric = summary.get("main_metric") or target.get("main_metric")
        row = {
            "job_slug": item["job_dir"].name,
            "target": target.get("name"),
            "task_type": target.get("task_type"),
            "main_metric": main_metric,
            "model": summary.get("model"),
            "feature_policy": summary.get("feature_policy", {}).get("name"),
            "feature_view": feature_view.get("name"),
            "feature_set": feature_set.get("name"),
            "n_rows": summary.get("n_rows"),
            "train_rows": summary.get("train_rows"),
            "test_rows": summary.get("test_rows"),
            "n_features": summary.get("n_features"),
        }
        for split_name, split_metrics in summary.get("metrics", {}).items():
            prefix = "validation" if split_name == "validation_cv" else split_name
            for metric, value in split_metrics.items():
                row[f"{prefix}_{metric}"] = value
        if main_metric:
            row["validation_main_metric"] = row.get(f"validation_{main_metric}")
            row["test_main_metric"] = row.get(f"test_{main_metric}")
            row["test_minus_validation_main_metric"] = (
                row["test_main_metric"] - row["validation_main_metric"]
                if pd.notna(row.get("test_main_metric"))
                and pd.notna(row.get("validation_main_metric"))
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _ranking_table(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return comparison
    ranked_groups = []
    for target, target_df in comparison.groupby("target", dropna=False):
        ranked = target_df.copy()
        ranked["validation_rank"] = ranked["validation_main_metric"].rank(
            ascending=False,
            method="min",
            na_option="bottom",
        )
        ranked["test_rank"] = ranked["test_main_metric"].rank(
            ascending=False,
            method="min",
            na_option="bottom",
        )
        ranked_groups.append(ranked)
    ranking = pd.concat(ranked_groups, ignore_index=True)
    return ranking.sort_values(["target", "validation_rank", "test_rank"])


def _write_validation_test_plots(
    comparison: pd.DataFrame,
    reports_dir: Path,
) -> list[str]:
    if comparison.empty:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    for target, target_df in comparison.groupby("target", dropna=False):
        metric = target_df["main_metric"].dropna().iloc[0]
        plot_df = target_df.dropna(
            subset=["validation_main_metric", "test_main_metric"]
        ).copy()
        if plot_df.empty:
            continue
        safe_target = _safe_name(str(target))
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(plot_df["validation_main_metric"], plot_df["test_main_metric"])
        for _, row in plot_df.iterrows():
            ax.annotate(
                f"{row['model']}|{row['feature_set']}",
                (row["validation_main_metric"], row["test_main_metric"]),
                fontsize=7,
                alpha=0.75,
            )
        min_value = min(
            plot_df["validation_main_metric"].min(),
            plot_df["test_main_metric"].min(),
        )
        max_value = max(
            plot_df["validation_main_metric"].max(),
            plot_df["test_main_metric"].max(),
        )
        ax.plot([min_value, max_value], [min_value, max_value], "--", color="gray")
        ax.set_xlabel(f"Validation {metric}")
        ax.set_ylabel(f"Test {metric}")
        ax.set_title(f"Validation vs Test: {target}")
        fig.tight_layout()
        path = reports_dir / f"validation_vs_test__{safe_target}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        files.append(str(path))

        ordered = plot_df.sort_values("validation_main_metric", ascending=False)
        fig, ax = plt.subplots(figsize=(max(8, len(ordered) * 0.7), 5))
        x = np.arange(len(ordered))
        width = 0.38
        ax.bar(x - width / 2, ordered["validation_main_metric"], width, label="validation")
        ax.bar(x + width / 2, ordered["test_main_metric"], width, label="test")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                f"{row.model}\n{row.feature_set}"
                for row in ordered.itertuples(index=False)
            ],
            rotation=45,
            ha="right",
        )
        ax.set_ylabel(metric)
        ax.set_title(f"Validation/Test {metric}: {target}")
        ax.legend()
        fig.tight_layout()
        path = reports_dir / f"validation_test_bars__{safe_target}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        files.append(str(path))
    return files


def _write_prediction_diagnostics(
    summary: dict[str, Any],
    split: str,
    predictions: pd.DataFrame,
    reports_dir: Path,
) -> list[dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_name = summary.get("target", {}).get("name", "target")
    task_type = summary.get("target", {}).get("task_type")
    job_slug = _job_slug(summary)
    job_report_dir = reports_dir / "jobs" / job_slug
    job_report_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    y_true = predictions["y_true"]
    y_pred = predictions["y_pred"]
    labels = sorted(pd.concat([y_true, y_pred]).dropna().unique().tolist(), key=str)

    cm_path = job_report_dir / f"{split}_confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"{split.title()} Confusion Matrix\n{job_slug}")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=160)
    plt.close(fig)
    rows.append(_artifact_row(summary, split, "confusion_matrix", cm_path))

    proba_columns = [column for column in predictions.columns if column.startswith("proba_")]
    if task_type == "binary" and proba_columns:
        proba_col = _positive_probability_column(summary, proba_columns)
        y_score = predictions[proba_col]
        positive_label = _positive_label(summary, y_true)
        y_binary = (y_true.astype(str) == str(positive_label)).astype(int)

        roc_path = job_report_dir / f"{split}_roc_curve.png"
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_binary, y_score, ax=ax)
        ax.set_title(f"{split.title()} ROC\n{job_slug}")
        fig.tight_layout()
        fig.savefig(roc_path, dpi=160)
        plt.close(fig)
        rows.append(_artifact_row(summary, split, "roc_curve", roc_path))

        pr_path = job_report_dir / f"{split}_pr_curve.png"
        fig, ax = plt.subplots(figsize=(6, 5))
        PrecisionRecallDisplay.from_predictions(y_binary, y_score, ax=ax)
        ax.set_title(f"{split.title()} Precision-Recall\n{job_slug}")
        fig.tight_layout()
        fig.savefig(pr_path, dpi=160)
        plt.close(fig)
        rows.append(_artifact_row(summary, split, "pr_curve", pr_path))

        calibration_path = job_report_dir / f"{split}_calibration_curve.png"
        prob_true, prob_pred = calibration_curve(
            y_binary,
            y_score,
            n_bins=10,
            strategy="uniform",
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(prob_pred, prob_true, marker="o", label="model")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed fraction positive")
        ax.set_title(f"{split.title()} Calibration\n{job_slug}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(calibration_path, dpi=160)
        plt.close(fig)
        rows.append(_artifact_row(summary, split, "calibration_curve", calibration_path))
    elif proba_columns:
        roc_path = job_report_dir / f"{split}_roc_curve.png"
        pr_path = job_report_dir / f"{split}_pr_curve.png"
        y_true_str = y_true.astype(str)

        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
        plotted = False
        for proba_col in sorted(proba_columns):
            class_label = proba_col.removeprefix("proba_")
            y_binary = (y_true_str == class_label).astype(int)
            if y_binary.nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(y_binary, predictions[proba_col])
            precision, recall, _ = precision_recall_curve(
                y_binary,
                predictions[proba_col],
            )
            ax_roc.plot(fpr, tpr, label=str(class_label))
            ax_pr.plot(recall, precision, label=str(class_label))
            plotted = True
        if plotted:
            ax_roc.plot([0, 1], [0, 1], "--", color="gray")
            ax_roc.set_xlabel("False positive rate")
            ax_roc.set_ylabel("True positive rate")
            ax_roc.set_title(f"{split.title()} OvR ROC\n{job_slug}")
            ax_roc.legend(fontsize=7)
            fig_roc.tight_layout()
            fig_roc.savefig(roc_path, dpi=160)
            rows.append(_artifact_row(summary, split, "roc_curve_ovr", roc_path))

            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            ax_pr.set_title(f"{split.title()} OvR Precision-Recall\n{job_slug}")
            ax_pr.legend(fontsize=7)
            fig_pr.tight_layout()
            fig_pr.savefig(pr_path, dpi=160)
            rows.append(_artifact_row(summary, split, "pr_curve_ovr", pr_path))
        plt.close(fig_roc)
        plt.close(fig_pr)
    return rows


def _class_level_rows(
    summary: dict[str, Any],
    split: str,
    predictions: pd.DataFrame,
) -> list[dict[str, Any]]:
    report = classification_report(
        predictions["y_true"],
        predictions["y_pred"],
        output_dict=True,
        zero_division=0,
    )
    auc_by_class = _class_auc_metrics(predictions)
    rows = []
    for label, metrics in report.items():
        if not isinstance(metrics, dict):
            continue
        auc_metrics = auc_by_class.get(str(label), {})
        rows.append(
            {
                "job_slug": _job_slug(summary),
                "target": summary.get("target", {}).get("name"),
                "task_type": summary.get("target", {}).get("task_type"),
                "model": summary.get("model"),
                "feature_view": summary.get("feature_view", {}).get("name"),
                "feature_set": summary.get("feature_set", {}).get("name"),
                "split": split,
                "class": label,
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1_score": metrics.get("f1-score"),
                "support": metrics.get("support"),
                "roc_auc": auc_metrics.get("roc_auc"),
                "pr_auc": auc_metrics.get("pr_auc"),
            }
        )
    return rows


def _class_auc_metrics(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    proba_columns = [
        column for column in predictions.columns if column.startswith("proba_")
    ]
    if not proba_columns:
        return {}
    y_true = predictions["y_true"].astype(str)
    metrics = {}
    for proba_column in proba_columns:
        label = proba_column.removeprefix("proba_")
        y_binary = (y_true == label).astype(int)
        if y_binary.nunique() < 2:
            continue
        y_score = predictions[proba_column]
        metrics[label] = {
            "roc_auc": _safe_auc(roc_auc_score, y_binary, y_score),
            "pr_auc": _safe_auc(average_precision_score, y_binary, y_score),
        }
    return metrics


def _safe_auc(metric_func, y_true, y_score) -> float | None:
    try:
        return float(metric_func(y_true, y_score))
    except ValueError:
        return None


def _positive_probability_column(summary: dict[str, Any], proba_columns: list[str]) -> str:
    positive = summary.get("target", {}).get("positive_label")
    if positive is not None and f"proba_{positive}" in proba_columns:
        return f"proba_{positive}"
    if "proba_1" in proba_columns:
        return "proba_1"
    return sorted(proba_columns)[-1]


def _positive_label(summary: dict[str, Any], y_true: pd.Series):
    positive = summary.get("target", {}).get("positive_label")
    if positive is not None:
        return positive
    values = sorted(y_true.dropna().unique().tolist(), key=str)
    return values[-1]


def _artifact_row(
    summary: dict[str, Any],
    split: str,
    artifact_type: str,
    path: Path,
) -> dict[str, str]:
    return {
        "job_slug": _job_slug(summary),
        "target": summary.get("target", {}).get("name"),
        "model": summary.get("model"),
        "feature_view": summary.get("feature_view", {}).get("name"),
        "feature_set": summary.get("feature_set", {}).get("name"),
        "split": split,
        "artifact_type": artifact_type,
        "path": str(path),
    }


def _job_slug(summary: dict[str, Any]) -> str:
    return "__".join(
        [
            str(summary.get("target", {}).get("name")),
            str(summary.get("model")),
            str(summary.get("feature_view", {}).get("name")),
            str(summary.get("feature_set", {}).get("name")),
        ]
    )


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
