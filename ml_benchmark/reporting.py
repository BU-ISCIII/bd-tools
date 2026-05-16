"""Homogeneous output contracts and aggregate benchmark reporting."""

from __future__ import annotations

import json
import shutil
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
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .types import BenchmarkJob, DiagnosticResult, ResourceLimits


REPORT_VALIDATION_COLOR = "#0072B2"
REPORT_TEST_COLOR = "#E69F00"
REPORT_REFERENCE_COLOR = "#6F6F6F"
REPORT_GRID_COLOR = "#D0D7DE"
REPORT_BOX_EDGE_COLOR = "#8C959F"
REPORT_CONFUSION_CMAP = "YlGnBu"
REPORT_LINE_COLORS = [
    REPORT_VALIDATION_COLOR,
    REPORT_TEST_COLOR,
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#000000",
]


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
    legacy_job_reports_dir = reports_dir / "jobs"
    if legacy_job_reports_dir.exists():
        shutil.rmtree(legacy_job_reports_dir)

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
    calibration_metric_rows = []
    calibration_curve_rows = []
    threshold_metric_rows = []
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
            calibration = _calibration_report_rows(summary, split, predictions)
            calibration_metric_rows.extend(calibration["metrics"])
            calibration_curve_rows.extend(calibration["curves"])
            threshold_metric_rows.extend(_threshold_metric_rows(summary, split, predictions))
            diagnostic_rows.extend(
                _write_prediction_diagnostics(
                    summary,
                    split,
                    predictions,
                    job_report_dir=job_dir / "reports",
                )
            )

    class_metrics = pd.DataFrame(class_rows)
    class_metrics_path = reports_dir / "class_level_metrics.csv"
    class_metrics.to_csv(class_metrics_path, index=False)

    calibration_metrics = pd.DataFrame(calibration_metric_rows)
    calibration_metrics_path = reports_dir / "calibration_metrics.csv"
    calibration_metrics.to_csv(calibration_metrics_path, index=False)

    calibration_curves = pd.DataFrame(calibration_curve_rows)
    calibration_curves_path = reports_dir / "calibration_curves.csv"
    calibration_curves.to_csv(calibration_curves_path, index=False)

    threshold_metrics = pd.DataFrame(threshold_metric_rows)
    threshold_metrics_path = reports_dir / "threshold_metrics.csv"
    threshold_metrics.to_csv(threshold_metrics_path, index=False)

    threshold_summary = _threshold_summary_table(threshold_metrics)
    threshold_summary_path = reports_dir / "threshold_summary.csv"
    threshold_summary.to_csv(threshold_summary_path, index=False)

    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics_path = reports_dir / "diagnostic_artifacts.csv"
    diagnostics.to_csv(diagnostics_path, index=False)

    target_dirs = _write_target_report_tables(
        reports_dir,
        {
            "job_comparison": comparison,
            "model_rankings": rankings,
            "class_level_metrics": class_metrics,
            "calibration_metrics": calibration_metrics,
            "calibration_curves": calibration_curves,
            "threshold_metrics": threshold_metrics,
            "threshold_summary": threshold_summary,
            "diagnostic_artifacts": diagnostics,
        },
    )

    manifest = {
        "base_output_dir": str(base_output_dir),
        "reports_dir": str(reports_dir),
        "n_jobs": int(len(summaries)),
        "target_dirs": target_dirs,
        "files": {
            "job_comparison": str(comparison_path),
            "model_rankings": str(rankings_path),
            "class_level_metrics": str(class_metrics_path),
            "calibration_metrics": str(calibration_metrics_path),
            "calibration_curves": str(calibration_curves_path),
            "threshold_metrics": str(threshold_metrics_path),
            "threshold_summary": str(threshold_summary_path),
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
        target_dir = _target_report_dir(reports_dir, str(target))
        target_dir.mkdir(parents=True, exist_ok=True)
        metric = target_df["main_metric"].dropna().iloc[0]
        plot_df = target_df.dropna(
            subset=["validation_main_metric", "test_main_metric"]
        ).copy()
        if plot_df.empty:
            continue
        safe_target = _safe_name(str(target))
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        ax.scatter(
            plot_df["validation_main_metric"],
            plot_df["test_main_metric"],
            color=REPORT_VALIDATION_COLOR,
            edgecolor="white",
            linewidth=0.8,
            s=46,
        )
        for _, row in plot_df.iterrows():
            ax.annotate(
                _compact_run_label(row),
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
        ax.plot([min_value, max_value], [min_value, max_value], "--", color=REPORT_REFERENCE_COLOR)
        ax.margins(x=0.12, y=0.12)
        ax.set_xlabel(f"Validation {metric}")
        ax.set_ylabel(f"Test {metric}")
        ax.set_title(f"Validation vs Test: {target}")
        fig.tight_layout()
        path = target_dir / f"validation_vs_test__{safe_target}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        files.append(str(path))

        metric_panels = _available_bar_metrics(target_df)
        if not metric_panels:
            continue
        ordered = target_df.sort_values(
            f"validation_{metric_panels[0][0]}",
            ascending=False,
            na_position="last",
        )
        panel_height = max(3.6, len(ordered) * 0.48 + 1.8)
        fig, axes = plt.subplots(
            len(metric_panels),
            1,
            figsize=(14, panel_height * len(metric_panels)),
            squeeze=False,
        )
        y = np.arange(len(ordered))
        height = 0.36
        labels = [_compact_run_label(row) for _, row in ordered.iterrows()]
        for ax, (metric_name, metric_label) in zip(axes[:, 0], metric_panels):
            validation_col = f"validation_{metric_name}"
            test_col = f"test_{metric_name}"
            validation_values = pd.to_numeric(ordered[validation_col], errors="coerce")
            test_values = pd.to_numeric(ordered[test_col], errors="coerce")
            validation_bars = ax.barh(
                y - height / 2,
                validation_values,
                height,
                label="validation",
                color=REPORT_VALIDATION_COLOR,
            )
            test_bars = ax.barh(
                y + height / 2,
                test_values,
                height,
                label="test",
                color=REPORT_TEST_COLOR,
            )
            _add_horizontal_bar_value_labels(ax, validation_bars)
            _add_horizontal_bar_value_labels(ax, test_bars)
            ax.set_xlabel(metric_label)
            ax.set_title(f"Validation/Test {metric_label}: {target}")
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlim(
                left=0,
                right=_metric_axis_top(validation_values, test_values),
            )
            ax.invert_yaxis()
            ax.grid(axis="x", color=REPORT_GRID_COLOR, alpha=0.6)
            ax.legend(loc="lower right")
        fig.subplots_adjust(left=0.32, right=0.96, top=0.96, bottom=0.04, hspace=0.42)
        path = target_dir / f"validation_test_metric_bars__{safe_target}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        files.append(str(path))
    return files


def _write_prediction_diagnostics(
    summary: dict[str, Any],
    split: str,
    predictions: pd.DataFrame,
    job_report_dir: Path,
) -> list[dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    task_type = summary.get("target", {}).get("task_type")
    job_slug = _job_slug(summary)
    job_report_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    y_true = predictions["y_true"]
    y_pred = predictions["y_pred"]
    labels = sorted(pd.concat([y_true, y_pred]).dropna().unique().tolist(), key=str)

    cm_path = job_report_dir / f"{split}_confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        ax=ax,
        colorbar=False,
        cmap=REPORT_CONFUSION_CMAP,
    )
    ax.set_title(f"{split.title()} Confusion Matrix\n{job_slug}")
    metric_text = _confusion_matrix_metric_text(summary, predictions)
    if metric_text:
        ax.text(
            1.04,
            0.5,
            metric_text,
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "white",
                "edgecolor": REPORT_BOX_EDGE_COLOR,
                "alpha": 0.95,
            },
        )
    fig.subplots_adjust(left=0.10, right=0.72, top=0.86, bottom=0.14)
    fig.savefig(cm_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    rows.append(_artifact_row(summary, split, "confusion_matrix", cm_path))

    proba_columns = [column for column in predictions.columns if column.startswith("proba_")]
    if task_type == "binary" and proba_columns:
        proba_col = _positive_probability_column(summary, proba_columns)
        y_score = predictions[proba_col]
        positive_label = _positive_label(summary, y_true)
        y_binary = (y_true.astype(str) == str(positive_label)).astype(int)

        roc_path = job_report_dir / f"{split}_roc_curve.png"
        fig, ax = plt.subplots(figsize=(7.2, 5.8))
        RocCurveDisplay.from_predictions(
            y_binary,
            y_score,
            ax=ax,
            color=REPORT_VALIDATION_COLOR,
        )
        ax.set_title(f"{split.title()} ROC\n{job_slug}")
        fig.tight_layout()
        fig.savefig(roc_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        rows.append(_artifact_row(summary, split, "roc_curve", roc_path))

        pr_path = job_report_dir / f"{split}_pr_curve.png"
        fig, ax = plt.subplots(figsize=(7.2, 5.8))
        PrecisionRecallDisplay.from_predictions(
            y_binary,
            y_score,
            ax=ax,
            color=REPORT_TEST_COLOR,
        )
        ax.set_title(f"{split.title()} Precision-Recall\n{job_slug}")
        fig.tight_layout()
        fig.savefig(pr_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        rows.append(_artifact_row(summary, split, "pr_curve", pr_path))

        calibration_path = job_report_dir / f"{split}_calibration_curve.png"
        prob_true, prob_pred = calibration_curve(
            y_binary,
            y_score,
            n_bins=10,
            strategy="uniform",
        )
        fig, ax = plt.subplots(figsize=(7.2, 5.8))
        ax.plot(
            prob_pred,
            prob_true,
            marker="o",
            label="model",
            color=REPORT_VALIDATION_COLOR,
        )
        ax.plot([0, 1], [0, 1], "--", color=REPORT_REFERENCE_COLOR, label="perfect")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed fraction positive")
        ax.set_title(f"{split.title()} Calibration\n{job_slug}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(calibration_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        rows.append(_artifact_row(summary, split, "calibration_curve", calibration_path))
    elif proba_columns:
        roc_path = job_report_dir / f"{split}_roc_curve.png"
        pr_path = job_report_dir / f"{split}_pr_curve.png"
        y_true_str = y_true.astype(str)

        fig_roc, ax_roc = plt.subplots(figsize=(7.2, 5.8))
        fig_pr, ax_pr = plt.subplots(figsize=(7.2, 5.8))
        plotted = False
        for color_idx, proba_col in enumerate(sorted(proba_columns)):
            class_label = proba_col.removeprefix("proba_")
            y_binary = (y_true_str == class_label).astype(int)
            if y_binary.nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(y_binary, predictions[proba_col])
            precision, recall, _ = precision_recall_curve(
                y_binary,
                predictions[proba_col],
            )
            color = REPORT_LINE_COLORS[color_idx % len(REPORT_LINE_COLORS)]
            ax_roc.plot(fpr, tpr, label=str(class_label), color=color)
            ax_pr.plot(recall, precision, label=str(class_label), color=color)
            plotted = True
        if plotted:
            ax_roc.plot([0, 1], [0, 1], "--", color=REPORT_REFERENCE_COLOR)
            ax_roc.set_xlabel("False positive rate")
            ax_roc.set_ylabel("True positive rate")
            ax_roc.set_title(f"{split.title()} OvR ROC\n{job_slug}")
            ax_roc.legend(fontsize=7)
            fig_roc.tight_layout()
            fig_roc.savefig(roc_path, dpi=160, bbox_inches="tight")
            rows.append(_artifact_row(summary, split, "roc_curve_ovr", roc_path))

            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            ax_pr.set_title(f"{split.title()} OvR Precision-Recall\n{job_slug}")
            ax_pr.legend(fontsize=7)
            fig_pr.tight_layout()
            fig_pr.savefig(pr_path, dpi=160, bbox_inches="tight")
            rows.append(_artifact_row(summary, split, "pr_curve_ovr", pr_path))
        plt.close(fig_roc)
        plt.close(fig_pr)

        calibration_path = job_report_dir / f"{split}_calibration_curve.png"
        fig, ax = plt.subplots(figsize=(7.2, 5.8))
        calibration_plotted = False
        for color_idx, proba_col in enumerate(sorted(proba_columns)):
            class_label = proba_col.removeprefix("proba_")
            y_binary = (y_true_str == class_label).astype(int)
            if y_binary.nunique() < 2:
                continue
            prob_true, prob_pred = calibration_curve(
                y_binary,
                predictions[proba_col],
                n_bins=10,
                strategy="uniform",
            )
            color = REPORT_LINE_COLORS[color_idx % len(REPORT_LINE_COLORS)]
            ax.plot(prob_pred, prob_true, marker="o", label=str(class_label), color=color)
            calibration_plotted = True
        if calibration_plotted:
            ax.plot([0, 1], [0, 1], "--", color=REPORT_REFERENCE_COLOR, label="perfect")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Observed fraction positive")
            ax.set_title(f"{split.title()} OvR Calibration\n{job_slug}")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(calibration_path, dpi=160, bbox_inches="tight")
            rows.append(
                _artifact_row(summary, split, "calibration_curve_ovr", calibration_path)
            )
        plt.close(fig)
    return rows


def _available_bar_metrics(comparison: pd.DataFrame) -> list[tuple[str, str]]:
    requested = [
        ("average_precision", "PR-AUC / average precision"),
        ("f1_macro", "F1 macro"),
        ("recall_macro", "Recall macro"),
        ("precision_macro", "Precision macro"),
        ("accuracy", "Accuracy"),
        ("roc_auc", "ROC-AUC"),
        ("roc_auc_ovr", "ROC-AUC OvR"),
        ("log_loss", "Log loss"),
    ]
    available = []
    for metric, label in requested:
        columns = [f"validation_{metric}", f"test_{metric}"]
        if all(column in comparison.columns for column in columns):
            values = comparison[columns].apply(pd.to_numeric, errors="coerce")
            if values.notna().any().any():
                available.append((metric, label))
    return available


def _compact_run_label(row: pd.Series) -> str:
    return "\n".join(
        [
            str(row.get("model", "")),
            str(row.get("feature_view", "")),
            str(row.get("feature_set", "")),
        ]
    )


def _add_bar_value_labels(ax, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=7,
        )


def _add_horizontal_bar_value_labels(ax, bars) -> None:
    for bar in bars:
        width = bar.get_width()
        if not np.isfinite(width):
            continue
        ax.annotate(
            f"{width:.3f}",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7,
        )


def _metric_axis_top(*series: pd.Series) -> float:
    values = pd.concat([item.dropna() for item in series])
    if values.empty:
        return 1.0
    max_value = float(values.max())
    if max_value <= 1.0:
        return 1.08
    return max_value * 1.12


def _confusion_matrix_metric_text(
    summary: dict[str, Any],
    predictions: pd.DataFrame,
) -> str:
    y_true = predictions["y_true"].astype(str)
    y_pred = predictions["y_pred"].astype(str)
    labels = sorted(pd.concat([y_true, y_pred]).dropna().unique().tolist(), key=str)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    accuracy = _safe_auc(accuracy_score, y_true, y_pred)
    f1_macro = _safe_auc(f1_score, y_true, y_pred, average="macro")
    f1_weighted = _safe_auc(f1_score, y_true, y_pred, average="weighted")
    lines = [
        f"Accuracy: {_format_metric_value(accuracy)}",
        f"F1 macro: {_format_metric_value(f1_macro)}",
        f"F1 weighted: {_format_metric_value(f1_weighted)}",
        "Per-class:",
    ]
    for label in labels:
        class_metrics = report.get(str(label), {})
        if not isinstance(class_metrics, dict):
            continue
        lines.append(
            "{label}: P {precision} R {recall} F1 {f1} n {support}".format(
                label=label,
                precision=_format_metric_value(class_metrics.get("precision")),
                recall=_format_metric_value(class_metrics.get("recall")),
                f1=_format_metric_value(class_metrics.get("f1-score")),
                support=_format_support_value(class_metrics.get("support")),
            )
        )

    auc_metrics = _prediction_auc_summary(summary, predictions)
    lines.extend(
        f"{name}: {_format_metric_value(value)}"
        for name, value in auc_metrics.items()
    )
    return "\n".join(lines)


def _prediction_auc_summary(
    summary: dict[str, Any],
    predictions: pd.DataFrame,
) -> dict[str, float | None]:
    proba_columns = [
        column for column in predictions.columns if column.startswith("proba_")
    ]
    if not proba_columns:
        return {"ROC-AUC": None, "PR-AUC": None}
    task_type = summary.get("target", {}).get("task_type")
    y_true = predictions["y_true"].astype(str)
    if task_type == "binary":
        proba_column = _positive_probability_column(summary, proba_columns)
        y_score = predictions[proba_column].astype(float)
        positive_label = str(_positive_label(summary, y_true))
        y_binary = (y_true == positive_label).astype(int)
        if y_binary.nunique() < 2:
            return {"ROC-AUC": None, "PR-AUC": None}
        return {
            "ROC-AUC": _safe_auc(roc_auc_score, y_binary, y_score),
            "PR-AUC": _safe_auc(average_precision_score, y_binary, y_score),
        }

    class_labels = [column.removeprefix("proba_") for column in proba_columns]
    y_binary = pd.DataFrame(
        {
            label: (y_true == label).astype(int)
            for label in class_labels
        }
    )
    valid_labels = [
        label
        for label in class_labels
        if y_binary[label].nunique() == 2
    ]
    if not valid_labels:
        return {"ROC-AUC OvR": None, "PR-AUC OvR": None}
    y_binary_valid = y_binary[valid_labels]
    proba_valid = predictions[[f"proba_{label}" for label in valid_labels]].astype(float)
    return {
        "ROC-AUC OvR": _safe_auc(
            roc_auc_score,
            y_binary_valid,
            proba_valid,
            average="macro",
        ),
        "PR-AUC OvR": _safe_auc(
            average_precision_score,
            y_binary_valid,
            proba_valid,
            average="macro",
        ),
    }


def _format_metric_value(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"


def _format_support_value(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return str(int(value))


def _calibration_report_rows(
    summary: dict[str, Any],
    split: str,
    predictions: pd.DataFrame,
    *,
    n_bins: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    proba_columns = [
        column for column in predictions.columns if column.startswith("proba_")
    ]
    if not proba_columns:
        return {"metrics": [], "curves": []}

    metric_rows = []
    curve_rows = []
    y_true = predictions["y_true"].astype(str)
    for proba_column in sorted(proba_columns):
        class_label = proba_column.removeprefix("proba_")
        y_binary = (y_true == class_label).astype(int)
        if y_binary.nunique() < 2:
            continue
        y_score = predictions[proba_column].astype(float)
        ece, bins = _expected_calibration_error(
            y_binary.to_numpy(),
            y_score.to_numpy(),
            n_bins=n_bins,
        )
        metric_rows.append(
            {
                **_calibration_base_row(summary, split),
                "calibration_type": "one_vs_rest",
                "class": class_label,
                "brier_score": float(brier_score_loss(y_binary, y_score)),
                "ece": ece,
                "n_bins": n_bins,
                "support": int(y_binary.sum()),
                "prevalence": float(y_binary.mean()),
            }
        )
        for bin_row in bins:
            curve_rows.append(
                {
                    **_calibration_base_row(summary, split),
                    "calibration_type": "one_vs_rest",
                    "class": class_label,
                    **bin_row,
                }
            )

    top_label = _top_label_calibration(summary, split, predictions, n_bins=n_bins)
    metric_rows.extend(top_label["metrics"])
    curve_rows.extend(top_label["curves"])
    return {"metrics": metric_rows, "curves": curve_rows}


def _threshold_metric_rows(
    summary: dict[str, Any],
    split: str,
    predictions: pd.DataFrame,
    thresholds: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if summary.get("target", {}).get("task_type") != "binary":
        return []
    proba_columns = [
        column for column in predictions.columns if column.startswith("proba_")
    ]
    if not proba_columns:
        return []
    thresholds = thresholds if thresholds is not None else np.linspace(0.0, 1.0, 101)
    positive_label = _positive_label(summary, predictions["y_true"])
    proba_column = _positive_probability_column(summary, proba_columns)
    y_true = (predictions["y_true"].astype(str) == str(positive_label)).astype(int)
    y_score = predictions[proba_column].astype(float)

    rows = []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        specificity = _safe_divide(tn, tn + fp)
        npv = _safe_divide(tn, tn + fn)
        f1 = _fbeta(precision, recall, beta=1.0)
        f2 = _fbeta(precision, recall, beta=2.0)
        f05 = _fbeta(precision, recall, beta=0.5)
        rows.append(
            {
                "job_slug": _job_slug(summary),
                "target": summary.get("target", {}).get("name"),
                "model": summary.get("model"),
                "feature_view": summary.get("feature_view", {}).get("name"),
                "feature_set": summary.get("feature_set", {}).get("name"),
                "split": split,
                "positive_label": positive_label,
                "threshold": float(threshold),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "sensitivity": recall,
                "recall": recall,
                "specificity": specificity,
                "precision": precision,
                "npv": npv,
                "f1": f1,
                "f2": f2,
                "f0_5": f05,
                "balanced_accuracy": (
                    (recall + specificity) / 2
                    if pd.notna(recall) and pd.notna(specificity)
                    else np.nan
                ),
                "predicted_positive_rate": float(y_pred.mean()),
                "true_prevalence": float(y_true.mean()),
            }
        )
    return rows


def _threshold_summary_table(threshold_metrics: pd.DataFrame) -> pd.DataFrame:
    if threshold_metrics.empty:
        return threshold_metrics
    rows = []
    criteria = {
        "best_f1": "f1",
        "best_f2": "f2",
        "best_f0_5": "f0_5",
        "best_balanced_accuracy": "balanced_accuracy",
        "youden_j": "youden_j",
    }
    metrics = threshold_metrics.copy()
    metrics["youden_j"] = metrics["sensitivity"] + metrics["specificity"] - 1
    group_columns = ["job_slug", "target", "model", "feature_view", "feature_set", "split"]
    for group_key, group in metrics.groupby(group_columns, dropna=False):
        group_values = dict(zip(group_columns, group_key))
        for criterion, metric_column in criteria.items():
            ranked = group.dropna(subset=[metric_column])
            if ranked.empty:
                continue
            best = ranked.sort_values(
                [metric_column, "threshold"],
                ascending=[False, True],
            ).iloc[0]
            rows.append(
                {
                    **group_values,
                    "criterion": criterion,
                    "optimized_metric": metric_column,
                    "optimized_value": best[metric_column],
                    "threshold": best["threshold"],
                    "sensitivity": best["sensitivity"],
                    "specificity": best["specificity"],
                    "precision": best["precision"],
                    "npv": best["npv"],
                    "f1": best["f1"],
                    "f2": best["f2"],
                    "f0_5": best["f0_5"],
                    "balanced_accuracy": best["balanced_accuracy"],
                    "tp": int(best["tp"]),
                    "fp": int(best["fp"]),
                    "tn": int(best["tn"]),
                    "fn": int(best["fn"]),
                }
            )
    return pd.DataFrame(rows)


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def _fbeta(precision: float, recall: float, beta: float) -> float:
    if pd.isna(precision) or pd.isna(recall):
        return np.nan
    if precision == 0 and recall == 0:
        return 0.0
    beta_squared = beta**2
    denominator = beta_squared * precision + recall
    return (
        float((1 + beta_squared) * precision * recall / denominator)
        if denominator
        else np.nan
    )


def _top_label_calibration(
    summary: dict[str, Any],
    split: str,
    predictions: pd.DataFrame,
    *,
    n_bins: int,
) -> dict[str, list[dict[str, Any]]]:
    proba_columns = [
        column for column in predictions.columns if column.startswith("proba_")
    ]
    if len(proba_columns) < 2:
        return {"metrics": [], "curves": []}
    proba = predictions[proba_columns].astype(float)
    class_labels = [column.removeprefix("proba_") for column in proba_columns]
    top_indices = proba.to_numpy().argmax(axis=1)
    top_scores = proba.to_numpy()[np.arange(len(proba)), top_indices]
    predicted_labels = pd.Series([class_labels[idx] for idx in top_indices])
    correct = (
        predicted_labels.astype(str).to_numpy()
        == predictions["y_true"].astype(str).to_numpy()
    ).astype(int)
    ece, bins = _expected_calibration_error(correct, top_scores, n_bins=n_bins)
    metric_rows = [
        {
            **_calibration_base_row(summary, split),
            "calibration_type": "top_label",
            "class": "__top_label__",
            "brier_score": float(np.mean((top_scores - correct) ** 2)),
            "ece": ece,
            "n_bins": n_bins,
            "support": int(correct.sum()),
            "prevalence": float(correct.mean()),
        }
    ]
    curve_rows = [
        {
            **_calibration_base_row(summary, split),
            "calibration_type": "top_label",
            "class": "__top_label__",
            **bin_row,
        }
        for bin_row in bins
    ]
    return {"metrics": metric_rows, "curves": curve_rows}


def _expected_calibration_error(
    y_true_binary: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bins: int,
) -> tuple[float, list[dict[str, Any]]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    total = len(y_score)
    for bin_idx in range(n_bins):
        lower = edges[bin_idx]
        upper = edges[bin_idx + 1]
        if bin_idx == n_bins - 1:
            mask = (y_score >= lower) & (y_score <= upper)
        else:
            mask = (y_score >= lower) & (y_score < upper)
        count = int(mask.sum())
        if count:
            mean_predicted = float(y_score[mask].mean())
            observed_fraction = float(y_true_binary[mask].mean())
            bin_error = abs(observed_fraction - mean_predicted)
            ece += (count / total) * bin_error
        else:
            mean_predicted = np.nan
            observed_fraction = np.nan
            bin_error = np.nan
        rows.append(
            {
                "bin": bin_idx,
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "n": count,
                "mean_predicted": mean_predicted,
                "observed_fraction": observed_fraction,
                "absolute_error": bin_error,
            }
        )
    return float(ece), rows


def _calibration_base_row(summary: dict[str, Any], split: str) -> dict[str, Any]:
    return {
        "job_slug": _job_slug(summary),
        "target": summary.get("target", {}).get("name"),
        "task_type": summary.get("target", {}).get("task_type"),
        "model": summary.get("model"),
        "feature_view": summary.get("feature_view", {}).get("name"),
        "feature_set": summary.get("feature_set", {}).get("name"),
        "split": split,
    }


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


def _safe_auc(metric_func, y_true, y_score, **kwargs) -> float | None:
    try:
        return float(metric_func(y_true, y_score, **kwargs))
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


def _write_target_report_tables(
    reports_dir: Path,
    tables: dict[str, pd.DataFrame],
) -> dict[str, str]:
    targets = sorted(
        {
            str(target)
            for table in tables.values()
            if "target" in table.columns
            for target in table["target"].dropna().unique().tolist()
        }
    )
    target_dirs = {}
    for target in targets:
        target_dir = _target_report_dir(reports_dir, target)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_dirs[target] = str(target_dir)
        for name, table in tables.items():
            if "target" not in table.columns:
                continue
            target_table = table.loc[table["target"].astype(str) == target].copy()
            target_table.to_csv(target_dir / f"{name}.csv", index=False)
    return target_dirs


def _job_slug(summary: dict[str, Any]) -> str:
    return "__".join(
        [
            str(summary.get("target", {}).get("name")),
            str(summary.get("model")),
            str(summary.get("feature_view", {}).get("name")),
            str(summary.get("feature_set", {}).get("name")),
        ]
    )


def _target_report_dir(reports_dir: Path, target: str) -> Path:
    return reports_dir / _safe_name(target)


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
