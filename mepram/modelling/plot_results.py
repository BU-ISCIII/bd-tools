#!/usr/bin/env python3
"""
Create publication-ready evaluation plots from a MEPRAM modelling run.

This module is plotting-only: it does NOT retrain models.

Supported layouts
-----------------
Single/direct model:
    01-*/
        summary.json
        predictions*.csv
        report_*.txt

Staged model:
    01-*/
        level1_sepsis/
        level2_etiology/
        level3_cefalosporina/

A scope directory (e.g. 03-SEPSIS, 04-ETIOLOGY or 05-RESISTANCE)
containing multiple 01-* runs is also supported.

Plots generated
---------------
For every readable prediction target:
    plots/performance/
        <target>_confusion_matrix.png
        <target>_roc_curve.png
        <target>_pr_curve.png
        <target>_<report>_metrics.png

Only the requested evaluation plots are produced:
confusion matrices, ROC curves, precision-recall curves, and
per-category Precision/Recall/F1 plots.

Class-count plots, merged staged plots and redundant overall/stage-comparison
barplots are intentionally not generated. Available processing diagnostics,
including RFECV histories, are plotted separately.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "savefig.bbox": "tight",
})

# Blue is deliberately used for ROC/PR curves, as requested.
CURVE_COLOR = "#1565C0"
CHANCE_COLOR = "0.55"
CMAP_CONFUSION = "Blues"

METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create publication-ready evaluation plots from a MEPRAM run."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help=(
            "A single 01-* modelling run directory, or a scope directory "
            "(e.g. 03-SEPSIS, 04-ETIOLOGY, 05-RESISTANCE) containing 01-* runs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Plot output directory. Defaults to <run-dir>/plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution. Default: 300.",
    )
    parser.add_argument(
        "--format",
        dest="fmt",
        choices=("png", "pdf", "both"),
        default="png",
        help="Output format. Default: png.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value))
    return value.strip("._") or "plot"


def _save(fig: plt.Figure, path: Path, dpi: int, fmt: str) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    if fmt in {"png", "both"}:
        png = path.with_suffix(".png")
        fig.savefig(png, dpi=dpi)
        written.append(str(png))

    if fmt in {"pdf", "both"}:
        pdf = path.with_suffix(".pdf")
        fig.savefig(pdf)
        written.append(str(pdf))

    plt.close(fig)
    return written


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


# ---------------------------------------------------------------------------
# Target / class semantics
# ---------------------------------------------------------------------------

def _scope_name(run_dir: Path) -> str:
    """Return the nearest scope name, e.g. 04-ETIOLOGY or 05-RESISTANCE."""
    for parent in (run_dir, *run_dir.parents):
        name = parent.name.lower()
        if name.startswith(("03-", "04-", "05-")):
            return name
    return ""


def _target_kind(stage_key: str, prediction_path: Path, run_dir: Path) -> str:
    """
    Determine the biological target represented by a prediction file.

    Returned values:
        sepsis
        culture_gate
        resistance_gate
        gnb
        cef_resistance
    """
    stem = prediction_path.stem.lower()

    if stage_key == "level1_sepsis":
        return "sepsis"

    if stage_key == "level2_etiology":
        # A Level-2 file is an etiology output unless its own filename
        # explicitly identifies the Stage-1 gate. In particular,
        # predictions_direct_etiology.csv must never be labelled as a gate.
        if "gate" in stem or "stage1" in stem:
            return "culture_gate"
        return "gnb"

    if stage_key == "level3_cefalosporina":
        if "gate" in stem or "stage1" in stem:
            return "resistance_gate"
        return "cef_resistance"

    # Direct models: infer the target from the scope/run name and filenames.
    combined = " ".join([
        _scope_name(run_dir),
        run_dir.name.lower(),
        stem,
    ])

    # For direct runs inside a scope, prefer the scope's canonical target
    # rather than inferring from filenames which may mention 'cultivo' etc.
    if stage_key == "direct":
        scope = _scope_name(run_dir)
        if scope.startswith("04-"):
            return "gnb"
        if scope.startswith("05-"):
            return "cef_resistance"
        if scope.startswith("03-"):
            return "sepsis"

    if "sepsis" in combined:
        return "sepsis"

    if any(x in combined for x in (
        "resistance",
        "resistente",
        "cefalospor",
        "cef_res",
    )):
        return "cef_resistance"

    # For a direct etiology model, a gate is the safest default if the
    # prediction/report naming indicates a gate; otherwise GNB is inferred.
    if "gate" in combined or "cultivo" in combined:
        return "culture_gate"

    if any(x in combined for x in ("gnb", "etiology", "etiologia", "gnball")):
        return "gnb"

    # The direct resistance target is particularly important in 05-RESISTANCE.
    if _scope_name(run_dir).startswith("05-"):
        return "cef_resistance"

    # The direct etiology target is most commonly the GNB/nonGNB endpoint.
    if _scope_name(run_dir).startswith("04-"):
        return "gnb"

    return "culture_gate"


TARGET_LABELS = {
    "sepsis": ("no_sepsis", "sepsis"),
    "culture_gate": ("cultivo-", "cultivo+"),
    "resistance_gate": ("gate_negative", "gate_positive"),
    "gnb": ("nonGNB", "GNB"),
    "cef_resistance": ("non_Cef_resistant", "Cef_resistant"),
}


def _labels_from_classes(
    target_kind: str,
    values: np.ndarray,
    classes: list[str],
) -> np.ndarray:
    """Map encoded predictions to the authoritative exported class names.

    ``summary.json["classes"]`` (or classes inferred from prediction data) is
    authoritative. Static endpoint labels are used only when no usable class
    metadata exists. This supports binary and multiclass direct models without
    incorrectly forcing etiology outputs to cultivo-/cultivo+.
    """
    class_names = [str(value) for value in classes]
    if not class_names:
        return _canonicalise_array(target_kind, values)

    normalised_classes = {
        _normalise_raw_label(class_name): class_name
        for class_name in class_names
    }

    mapped: list[str] = []
    for value in np.asarray(values):
        normalised = _normalise_raw_label(value)

        if normalised in normalised_classes:
            mapped.append(normalised_classes[normalised])
            continue

        # LabelEncoder outputs are commonly stored as integer indices while
        # summary.json stores the corresponding biological class names.
        try:
            numeric = float(value)
            index = int(numeric)
            if numeric == index and 0 <= index < len(class_names):
                mapped.append(class_names[index])
                continue
        except (TypeError, ValueError):
            pass

        mapped.append(str(value).strip())

    return np.asarray(mapped, dtype=str)


def _normalise_raw_label(value: Any) -> str:
    text = str(value).strip()
    return re.sub(r"[\s_-]+", "_", text.lower())


def _canonical_label(target_kind: str, value: Any) -> str:
    """
    Convert modelling labels (0/1, yes/no, textual labels, etc.) to the
    publication labels requested for each endpoint.
    """
    negative, positive = TARGET_LABELS[target_kind]
    norm = _normalise_raw_label(value)

    # Numeric / boolean convention used by the modelling stages.
    if norm in {"0", "0.0", "false", "no"}:
        return negative
    if norm in {"1", "1.0", "true", "yes"}:
        return positive

    if target_kind == "sepsis":
        if norm in {
            "no_sepsis", "nosepsis", "negative", "no_sepsis_",
        }:
            return negative
        if norm in {"sepsis", "positive"}:
            return positive

    elif target_kind in {"culture_gate", "resistance_gate"}:
        if norm in {
            "cultivo_-", "cultivo_neg", "cultivo_negative",
            "gate_negative", "gate_neg", "negative", "neg",
        }:
            return negative
        if norm in {
            "cultivo_+", "cultivo_pos", "cultivo_positive",
            "gate_positive", "gate_pos", "positive", "pos",
        }:
            return positive

    elif target_kind == "gnb":
        if norm in {
            "gnb", "gram_negative", "gramnegative", "gram_negative_bacteria",
        }:
            return positive
        if norm in {
            "nongnb", "non_gnb", "gram_positive", "gram_positive_bacteria",
            "non_gram_negative",
        }:
            return negative

    elif target_kind == "cef_resistance":
        if norm in {
            "cef_resistant", "resistant_cef", "resistant",
            "resist_cef", "resist_cef_3a_4a", "resist_cef_3a4a",
            "resist_cef_3a_4a", "resistance", "resist",
        }:
            return positive
        if norm in {
            "non_cef_resistant", "nonresistant_cef", "non_resistant_cef",
            "non_resistant", "no_resistance", "negative", "neg",
        }:
            return negative

    # Keep unknown multiclass labels readable rather than silently changing them.
    return str(value).strip()


def _canonicalise_array(target_kind: str, values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [_canonical_label(target_kind, value) for value in values],
        dtype=str,
    )


# ---------------------------------------------------------------------------
# Stage discovery
# ---------------------------------------------------------------------------

STAGE_ORDER = (
    ("direct", "Direct"),
    ("level1_sepsis", "Level 1 — Sepsis"),
    ("level2_etiology", "Level 2 — Etiology"),
    ("level3_cefalosporina", "Level 3 — Cefalosporin resistance"),
)


def _prediction_files(stage_dir: Path) -> list[Path]:
    return sorted(stage_dir.glob("predictions*.csv"))


def discover_stage_results(run_dir: Path) -> dict[str, dict[str, Any]]:
    """
    Discover direct and staged results.

    A stage is retained when summary.json exists OR when predictions*.csv
    exists, so plotting remains useful even if a summary file is missing.
    """
    results: dict[str, dict[str, Any]] = {}

    if (run_dir / "summary.json").is_file() or _prediction_files(run_dir):
        results["direct"] = {
            "directory": run_dir,
            "summary": _load_json(run_dir / "summary.json"),
        }

    for stage_key, _ in STAGE_ORDER:
        if stage_key == "direct":
            continue

        stage_dir = run_dir / stage_key
        if not stage_dir.is_dir():
            continue

        summary = _load_json(stage_dir / "summary.json")
        if summary or _prediction_files(stage_dir):
            results[stage_key] = {
                "directory": stage_dir,
                "summary": summary,
            }

    return results


def discover_run_directories(root: Path) -> list[Path]:
    root = root.resolve()

    if not root.is_dir():
        return []

    if root.name.startswith("01-"):
        return [root]

    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and p.name.startswith("01-")
    )


# ---------------------------------------------------------------------------
# Classification reports
# ---------------------------------------------------------------------------

def _extract_classification_report(text: str) -> pd.DataFrame | None:
    """
    Parse a sklearn classification_report table.

    Expected rows:
        class_name precision recall f1-score support
    """
    rows: list[dict[str, Any]] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        parts = re.split(r"\s+", stripped)
        if len(parts) < 5:
            continue

        numeric = parts[-4:]
        if not all(
            re.fullmatch(r"-?(?:\d+(?:\.\d*)?|\.\d+)", x)
            for x in numeric
        ):
            continue

        label = " ".join(parts[:-4]).strip()
        if not label:
            continue

        if label.lower() in {
            "accuracy",
            "macro avg",
            "weighted avg",
            "micro avg",
            "samples avg",
        }:
            continue

        try:
            precision, recall, f1, support = map(float, numeric)
        except ValueError:
            continue

        rows.append({
            "class": label,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        })

    if not rows:
        return None

    return pd.DataFrame(rows)


def discover_reports(stage_dir: Path) -> list[tuple[Path, pd.DataFrame]]:
    reports: list[tuple[Path, pd.DataFrame]] = []

    for path in sorted(stage_dir.glob("report_*.txt")):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue

        report = _extract_classification_report(text)
        if report is not None and not report.empty:
            reports.append((path, report))

    return reports


def _report_target_kind(
    stage_key: str,
    report_path: Path,
    run_dir: Path,
) -> str:
    """Use report filename semantics when they are more specific."""
    stem = report_path.stem.lower()

    if "gate" in stem or "infected_yes_no" in stem:
        return (
            "resistance_gate"
            if stage_key == "level3_cefalosporina"
            else "culture_gate"
        )

    if any(x in stem for x in (
        "gnball",
        "resultado_inf",
        "subtype",
    )):
        return "gnb"

    if any(x in stem for x in (
        "resistente_cefalosporina",
        "resistance",
        "cef",
    )):
        return "cef_resistance"

    # Fall back to the prediction-stage semantics.
    dummy_prediction = Path(stem + ".csv")
    return _target_kind(stage_key, dummy_prediction, run_dir)


# ---------------------------------------------------------------------------
# Prediction CSV parsing
# ---------------------------------------------------------------------------

_ACTUAL_COLUMNS = (
    "y_true",
    "true",
    "actual",
    "target",
    "label",
    "y",
    "true_label",
    "ground_truth",
    "groundtruth",
    "observed",
)

_PRED_COLUMNS = (
    "y_pred",
    "predicted",
    "prediction",
    "pred",
    "prediction_label",
    "predicted_label",
    "class_pred",
    "yhat",
)

_SCORE_COLUMNS = (
    "y_proba",
    "probability",
    "probabilities",
    "proba",
    "score",
    "y_score",
    "decision",
    "confidence",
    "positive_probability",
    "prob_positive",
    "prob_1",
    "probability_1",
    "class_1_probability",
    "p1",
)


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _coerce_probability_mapping(value: Any) -> dict[str, float] | None:
    mapping: Any = value

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            mapping = json.loads(text)
        except Exception:
            try:
                mapping = ast.literal_eval(text)
            except Exception:
                return None

    if not isinstance(mapping, dict):
        return None

    # Unwrap {"multiclass": {...}}-style containers.
    while len(mapping) == 1:
        only_value = next(iter(mapping.values()))
        if isinstance(only_value, dict):
            mapping = only_value
        else:
            break

    if not isinstance(mapping, dict):
        return None

    parsed: dict[str, float] = {}
    for key, raw_value in mapping.items():
        try:
            parsed[str(key)] = float(raw_value)
        except (TypeError, ValueError):
            continue

    return parsed or None


def _find_probability_column(df: pd.DataFrame, class_name: str) -> str | None:
    normalized = str(class_name).strip().lower()

    candidates = (
        normalized,
        f"proba_{normalized}",
        f"probability_{normalized}",
        f"prob_{normalized}",
        f"score_{normalized}",
        f"p_{normalized}",
    )

    lowered = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]

    return None


def _summary_class_names(summary: dict[str, Any]) -> list[str]:
    for key in (
        "classes",
        "class_names",
        "label_encoder_classes",
        "target_names",
    ):
        value = summary.get(key)
        if isinstance(value, (list, tuple)) and value:
            return [str(x) for x in value]

    negative = summary.get("negative_label")
    positive = summary.get("positive_label")
    if negative is not None and positive is not None:
        return [str(negative), str(positive)]

    return []


def _probability_matrix(
    df: pd.DataFrame,
    valid: pd.Series,
    classes: list[str],
) -> tuple[np.ndarray, np.ndarray] | None:
    if not classes:
        return None

    # Case 1: one probability column per class.
    per_class_cols: list[str] = []
    for class_name in classes:
        col = _find_probability_column(df, class_name)
        if col is None:
            per_class_cols = []
            break
        per_class_cols.append(col)

    if per_class_cols:
        scores = (
            df.loc[valid, per_class_cols]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )
        good = np.isfinite(scores).all(axis=1)
        if good.any():
            return scores[good], good

    # Case 2: serialized probability dictionary.
    lower_columns = {str(c).strip().lower(): c for c in df.columns}
    for candidate in ("proba", "probability", "probabilities"):
        column = lower_columns.get(candidate)
        if column is None:
            continue

        parsed_rows: list[list[float]] = []
        keep: list[bool] = []

        for raw_value in df.loc[valid, column]:
            mapping = _coerce_probability_mapping(raw_value)
            if mapping is None:
                keep.append(False)
                parsed_rows.append([np.nan] * len(classes))
                continue

            row: list[float] = []
            ok = True

            for class_name in classes:
                value = None
                for key in (
                    class_name,
                    class_name.strip(),
                    class_name.strip().lower(),
                ):
                    if key in mapping:
                        value = mapping[key]
                        break

                if value is None:
                    ok = False
                    row.append(np.nan)
                else:
                    row.append(float(value))

            keep.append(ok)
            parsed_rows.append(row)

        if parsed_rows:
            scores = np.asarray(parsed_rows, dtype=float)
            good = np.asarray(keep, dtype=bool)
            if good.any():
                return scores[good], good

    # Case 3: one or more scalar score columns.
    score_cols: list[str] = []
    score_names = {x.lower() for x in _SCORE_COLUMNS}

    for col in df.columns:
        lc = str(col).strip().lower()
        if lc in score_names:
            score_cols.append(col)

    if not score_cols:
        for col in df.columns:
            lc = str(col).strip().lower()
            if re.search(r"(?:proba|probability|score|prob)[_.-]?", lc):
                score_cols.append(col)

    if not score_cols:
        return None

    scores = (
        df.loc[valid, score_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )

    if scores.ndim == 1:
        scores = scores[:, None]

    good = np.isfinite(scores).all(axis=1)
    if not good.any():
        return None

    return scores[good], good


def _prediction_arrays(
    df: pd.DataFrame,
    summary: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, list[str]] | None:
    actual_col = _find_column(df, _ACTUAL_COLUMNS)
    pred_col = _find_column(df, _PRED_COLUMNS)

    if actual_col is None:
        for col in df.columns:
            lc = str(col).lower()
            if any(x in lc for x in ("true", "actual", "target")) and "prob" not in lc:
                actual_col = col
                break

    if pred_col is None:
        for col in df.columns:
            lc = str(col).lower()
            if any(x in lc for x in ("pred", "prediction")) and "prob" not in lc:
                pred_col = col
                break

    if actual_col is None:
        return None

    y_true_raw = df[actual_col]
    valid = y_true_raw.notna()
    y_true_raw = y_true_raw[valid]

    if y_true_raw.empty:
        return None

    y_pred = None
    if pred_col is not None:
        y_pred = df.loc[valid, pred_col].to_numpy()

    classes = _summary_class_names(summary)

    if not classes:
        classes = list(pd.unique(y_true_raw))
        if y_pred is not None:
            classes = list(dict.fromkeys([*classes, *pd.unique(y_pred)]))

    classes = [str(x) for x in classes]

    score_payload = _probability_matrix(df, valid, classes)
    scores = None

    if score_payload is not None:
        scores, good = score_payload
        y_true_raw = y_true_raw.iloc[np.flatnonzero(good)]

        if y_pred is not None:
            y_pred = y_pred[good]

    return (
        y_true_raw.to_numpy(),
        None if y_pred is None else np.asarray(y_pred),
        scores,
        classes,
    )


# ---------------------------------------------------------------------------
# Curves / confusion matrix
# ---------------------------------------------------------------------------

def _positive_score(
    scores: np.ndarray | None,
    classes: list[str],
) -> np.ndarray | None:
    if scores is None or scores.size == 0:
        return None

    if scores.ndim != 2 or scores.shape[0] == 0:
        return None

    if scores.shape[1] == 1:
        return scores[:, 0]

    if len(classes) >= 2:
        # The second class is the positive class under the modelling convention.
        return scores[:, 1]

    return None


def _curve_data(
    target_kind: str,
    y_true_raw: np.ndarray,
    scores: np.ndarray | None,
    classes: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, str] | None:
    if scores is None or scores.size == 0:
        return None

    y_true = _labels_from_classes(target_kind, y_true_raw, classes)

    # All requested MEPRAM endpoints are binary. Keep a safe multiclass fallback
    # for unexpected direct models rather than failing.
    if len(classes) == 2 or scores.shape[1] in {1, 2}:
        positive_score = _positive_score(scores, classes)
        if positive_score is None:
            return None

        # The modelling convention uses the second exported class as the
        # positive class. Preserve its biological label in the plots.
        positive_label = (
            str(classes[1])
            if len(classes) >= 2
            else TARGET_LABELS[target_kind][1]
        )
        y_bin = (y_true == positive_label).astype(int)

        if len(np.unique(y_bin)) != 2:
            # If canonicalisation did not recognise the source labels, use
            # the modelling class order.
            y_bin = (y_true_raw.astype(str) == str(classes[1])).astype(int)

        if len(np.unique(y_bin)) != 2:
            return None

        fpr, tpr, _ = roc_curve(y_bin, positive_score)
        precision, recall, _ = precision_recall_curve(y_bin, positive_score)

        roc_auc = roc_auc_score(y_bin, positive_score)
        pr_auc = average_precision_score(y_bin, positive_score)
        positive_rate = float(np.mean(y_bin))

        return (
            fpr,
            tpr,
            precision,
            recall,
            float(roc_auc),
            float(pr_auc),
            positive_rate,
            positive_label,
        )

    return None


def _make_confusion_matrix(
    target_kind: str,
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    classes: list[str],
) -> tuple[np.ndarray, list[str]]:
    y_true = _labels_from_classes(target_kind, y_true_raw, classes)
    y_pred = _labels_from_classes(target_kind, y_pred_raw, classes)

    labels = list(dict.fromkeys([
        *[str(value) for value in classes],
        *pd.unique(y_true),
        *pd.unique(y_pred),
    ]))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm, labels


def _plot_confusion(
    target_kind: str,
    target_title: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    output_dir: Path,
    dpi: int,
    fmt: str,
) -> list[str]:
    cm, labels = _make_confusion_matrix(
        target_kind, y_true, y_pred, classes
    )

    size = 5.4 if len(labels) <= 2 else max(5.4, 4.6 + 0.65 * len(labels))
    fig, ax = plt.subplots(figsize=(size, size * 0.9))

    image = ax.imshow(cm, cmap=CMAP_CONFUSION)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=0 if len(labels) <= 2 else 30, ha="right" if len(labels) > 2 else "center")
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{target_title}\nConfusion matrix", pad=10)

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            ax.text(
                j,
                i,
                f"{value:,}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=10,
            )

    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    return _save(
        fig,
        output_dir / "performance" / f"{_safe_name(target_kind)}_confusion_matrix",
        dpi,
        fmt,
    )


def _plot_curves(
    target_kind: str,
    target_title: str,
    curve: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, str],
    output_dir: Path,
    dpi: int,
    fmt: str,
) -> list[str]:
    fpr, tpr, precision, recall, roc_auc, pr_auc, positive_rate, positive_label = curve
    written: list[str] = []

    # ROC
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot(
        fpr,
        tpr,
        color=CURVE_COLOR,
        lw=2.4,
        label=f"ROC-AUC = {roc_auc:.3f}",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.0,
        color=CHANCE_COLOR,
        label="Chance",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{target_title}\nROC curve", pad=10)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.16, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    written.extend(_save(
        fig,
        output_dir / "performance" / f"{_safe_name(target_kind)}_roc_curve",
        dpi,
        fmt,
    ))

    # Precision-recall
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot(
        recall,
        precision,
        color=CURVE_COLOR,
        lw=2.4,
        label=f"PR-AUC = {pr_auc:.3f}",
    )

    # Baseline: precision expected for a random classifier equals positive prevalence
    ax.hlines(
        y=positive_rate,
        xmin=0,
        xmax=1,
        linestyle="--",
        linewidth=1.0,
        color=CHANCE_COLOR,
        label=f"Baseline = {positive_rate:.3f}",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{target_title}\nPrecision–recall curve", pad=10)
    ax.legend(frameon=False, loc="lower left")
    ax.grid(alpha=0.16, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    written.extend(_save(
        fig,
        output_dir / "performance" / f"{_safe_name(target_kind)}_pr_curve",
        dpi,
        fmt,
    ))

    return written


# ---------------------------------------------------------------------------
# Per-category Precision / Recall / F1
# ---------------------------------------------------------------------------

def _plot_class_metrics(
    target_kind: str,
    target_title: str,
    report_path: Path,
    report: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    fmt: str,
) -> list[str]:
    # The classification report already contains the model's biological class
    # labels. Keep them verbatim, including every multiclass category.
    classes = [str(value).strip() for value in report["class"].tolist()]

    # Remove duplicate class labels while preserving report order.
    keep = ~pd.Series(classes).duplicated()
    report = report.loc[keep.to_numpy()].copy()
    classes = [classes[i] for i, flag in enumerate(keep.to_numpy()) if flag]

    x = np.arange(len(classes))
    width = 0.23

    fig_width = max(6.8, 1.35 * len(classes) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.4))

    for j, metric in enumerate(("precision", "recall", "f1")):
        values = pd.to_numeric(
            report[metric],
            errors="coerce",
        ).to_numpy(dtype=float)

        bars = ax.bar(
            x + (j - 1) * width,
            values,
            width,
            label=METRIC_LABELS[metric],
            edgecolor="black",
            linewidth=0.4,
        )

        for bar, value in zip(bars, values):
            if not np.isfinite(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(1.045, value + 0.025),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Score")
    ax.set_xlabel("Predicted category")
    ax.set_xticks(x)
    ax.set_xticklabels(
        classes,
        rotation=0 if len(classes) <= 2 else 25,
        ha="center" if len(classes) <= 2 else "right",
    )

    ax.set_title(
        f"{target_title}\nPrecision, Recall and F1 by category",
        pad=10,
    )

    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
    )
    ax.grid(axis="y", alpha=0.16, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    fig.subplots_adjust(
        top=0.82,
        bottom=0.20 if len(classes) > 2 else 0.14,
        left=0.11,
        right=0.98,
    )

    return _save(
        fig,
        output_dir / "performance" / (
            f"{_safe_name(target_kind)}_"
            f"{_safe_name(report_path.stem)}_metrics"
        ),
        dpi,
        fmt,
    )


# ---------------------------------------------------------------------------
# Processing plots (imputed / nan / correlation / rfecv / iqr)
# ---------------------------------------------------------------------------


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _plot_rfecv_history(csv_path: Path, out_path: Path, dpi: int, fmt: str) -> list[str]:
    df = _safe_read_csv(csv_path)
    if df is None or df.empty:
        return []

    # Ensure n_features numeric
    if "n_features" not in df.columns:
        return []
    df = df.copy()
    df["n_features"] = pd.to_numeric(df["n_features"], errors="coerce")
    df = df.dropna(subset=["n_features"]).sort_values("n_features")
    if df.empty:
        return []

    # Only plot PR_AUC and ROC_AUC for publication clarity (omit F1 curve)
    metrics = [c for c in ("pr_auc", "roc_auc") if c in df.columns]
    if not metrics:
        return []

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for m in metrics:
        # PR curve should be orange for visibility; ROC uses CURVE_COLOR
        color = "#E69F00" if m == "pr_auc" else CURVE_COLOR
        ax.plot(
            df["n_features"],
            pd.to_numeric(df[m], errors="coerce"),
            marker="o",
            markersize=4,
            linewidth=1.2,
            label=m.replace("_", " ").upper(),
            color=color,
        )

    # Primary metric preference: PR_AUC if present, else ROC_AUC
    primary = "pr_auc" if "pr_auc" in df.columns else "roc_auc"
    scores = pd.to_numeric(df[primary], errors="coerce")
    if scores.empty or scores.isna().all():
        return []

    # Best scoring point (max primary metric)
    # Mark maximum features (initial full set) first as a subtle grey cap line.
    try:
        max_n = int(pd.to_numeric(df["n_features"], errors="coerce").max())
        ax.axvline(max_n, linestyle=":", color=CHANCE_COLOR, linewidth=1.0, label=f"Max: {max_n}")
    except Exception:
        max_n = None

    # Best scoring point (max primary metric) — highlight in light green.
    best_pos = scores.idxmax()
    try:
        best_n = int(df.loc[best_pos, "n_features"])
        best_score = float(df.loc[best_pos, primary])
        # vertical dashed line at best_n and a star marker at the best point
        best_color = "#90EE90"  # lightgreen
        ax.axvline(best_n, linestyle="--", color=best_color, linewidth=1.2, label=f"Best: {best_n}")
        ax.scatter([best_n], [best_score], marker="*", s=140, color=best_color, zorder=6, edgecolor="black", linewidth=0.6)
    except Exception:
        best_n = None

    ax.set_xlabel("Number of features")
    ax.set_ylabel("CV metric")
    ax.set_title(f"RFECV history: {csv_path.stem}")
    ax.legend(frameon=False)
    ax.grid(alpha=0.12)

    return _save(fig, out_path, dpi, fmt)


def _plot_top_counts(csv_path: Path, out_path: Path, dpi: int, fmt: str, top_n: int = 40, title: str | None = None) -> list[str]:
    df = _safe_read_csv(csv_path)
    if df is None or df.empty:
        return []

    # Try to find a label column and a numeric count column
    label_col = None
    count_col = None
    for c in df.columns:
        lc = str(c).lower()
        if any(x in lc for x in ("table", "feature", "name")):
            label_col = c
            break
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        # prefer explicit count-like names
        for c in numeric_cols:
            if any(x in str(c).lower() for x in ("count", "n_", "n", "imput", "nan", "outlier")):
                count_col = c
                break
        if count_col is None:
            count_col = numeric_cols[0]

    if label_col is None and len(numeric_cols) == 1:
        # single column of features: count entries
        ser = df.iloc[:, 0].dropna().astype(str)
        counts = ser.value_counts().nlargest(top_n)
        labels = counts.index.tolist()
        values = counts.values
    elif label_col is not None and count_col is not None:
        # If the numeric column is an index (row_index) or otherwise not a
        # true pre-aggregated count, compute counts per label instead of
        # plotting the numeric values directly. This avoids repeated
        # features caused by per-row listings (e.g. IQR outliers).
        lc_name = str(count_col).strip().lower()
        if lc_name in {"row_index", "index"} or lc_name.startswith("unnamed"):
            counts = df[label_col].dropna().astype(str).value_counts().nlargest(top_n)
            labels = counts.index.tolist()
            values = counts.values
        else:
            plot_df = df[[label_col, count_col]].dropna()
            plot_df[count_col] = pd.to_numeric(plot_df[count_col], errors="coerce").fillna(0)
            plot_df = plot_df.sort_values(count_col, ascending=False).head(top_n)
            labels = plot_df[label_col].astype(str).tolist()
            values = plot_df[count_col].to_list()
    else:
        return []

    fig, ax = plt.subplots(figsize=(max(6.4, 0.3 * len(labels) + 4), 6))
    y = list(range(len(labels)))
    ax.barh(y, values, color=CURVE_COLOR)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    # Adjust x-axis label for percentage-like columns (e.g. nan percentage).
    xlabel = "Count"
    try:
        # If the original CSV had a percentage-like numeric column name, prefer percent label.
        if count_col is not None and ("percent" in str(count_col).lower() or "percentage" in str(count_col).lower() or "nan" in str(count_col).lower() and max(values) <= 100):
            xlabel = "Percent (%)"
    except Exception:
        pass
    ax.set_xlabel(xlabel)
    ax.set_title(title or f"{csv_path.stem}")
    # Annotate values: format as percentage when the xlabel indicates percent.
    for i, v in enumerate(values):
        if xlabel.startswith("Percent"):
            try:
                txt = f"{float(v):.1f}%"
            except Exception:
                txt = str(v)
        else:
            try:
                txt = f"{int(v):,}"
            except Exception:
                txt = str(v)
        ax.text(v if v >= 0 else 0, i, f" {txt}", va="center", fontsize=8)
    fig.tight_layout()

    return _save(fig, out_path, dpi, fmt)


def _plot_processing_artifacts(run_dir: Path, output_dir: Path, dpi: int, fmt: str) -> list[str]:
    written: list[str] = []
    processing_dir = run_dir / "processing"
    plots_dir = output_dir / "processing"

    if not processing_dir.exists():
        return []

    # Discover every stage-specific RFECV history. A fixed filename list used
    # to omit, among others, l2_direct_rfecv_features.csv and
    # l3_resistance_gate_rfecv_features.csv.
    for path in sorted(processing_dir.glob("*_rfecv_features.csv")):
        out_name = f"processing_{_safe_name(path.stem)}_performance"
        written.extend(
            _plot_rfecv_history(
                path,
                plots_dir / out_name,
                dpi,
                fmt,
            )
        )

    # Top-count style plots
    count_files = [
        ("imputed_features.csv", "processing_imputed_features", "Imputed features"),
        ("nan_dropped_features.csv", "processing_nan_dropped_features", "NaN-dropped features"),
        ("correlation_dropped_features.csv", "processing_correlation_dropped_features", "Correlation-dropped features"),
        ("iqr_outliers_train.csv", "processing_iqr_outliers_train", "IQR outliers (train)"),
        ("iqr_outliers_test.csv", "processing_iqr_outliers_test", "IQR outliers (test)"),
    ]

    for fname, out_name, title in count_files:
        path = processing_dir / fname
        if path.is_file():
            written.extend(_plot_top_counts(path, plots_dir / out_name, dpi, fmt, title=title))

    return written


def _safe_load_npz(path: Path) -> dict[str, np.ndarray] | None:
    try:
        with np.load(path) as data:
            return {k: data[k] for k in data.files}
    except Exception:
        return None


def _plot_shap_importance(csv_path: Path, out_path: Path, dpi: int, fmt: str, top_n: int = 30) -> list[str]:
    df = _safe_read_csv(csv_path)
    if df is None or df.empty:
        return []
    # df expected columns: stage, output, feature, mean_abs_shap
    df = df.copy()
    if "mean_abs_shap" not in df.columns or "feature" not in df.columns:
        return []
    df = df.groupby("feature")["mean_abs_shap"].mean().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(6.4, max(4, 0.25 * len(df) + 2)))
    y = list(range(len(df)))
    ax.barh(y, df.values, color="#4C72B0")
    ax.set_yticks(y)
    ax.set_yticklabels(df.index.tolist(), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(f"SHAP importance: {csv_path.stem}")
    fig.tight_layout()
    return _save(fig, out_path, dpi, fmt)


def _plot_shap_beeswarm(npz_path: Path, X_csv: Path, out_path: Path, dpi: int, fmt: str, max_features: int = 50) -> list[str]:
    arrays = _safe_load_npz(npz_path)
    if arrays is None:
        return []
    # pick first output key
    key = next(iter(arrays))
    values = arrays[key]
    if values.ndim != 2:
        return []
    dfX = _safe_read_csv(X_csv)
    if dfX is None:
        return []
    # Exclude index-like columns such as 'row_index' from feature names.
    index_like = {"row_index", "index"}
    feature_names = [c for c in dfX.columns if str(c).strip().lower() not in index_like]

    # Align SHAP values columns with filtered feature names.
    # Typical cases:
    #  - values.shape[1] == len(feature_names): already aligned
    #  - values.shape[1] == dfX.shape[1]: values include index column(s) -> remove them
    #  - otherwise: try best-effort alignment by trimming or warning.
    if values.shape[1] == len(feature_names):
        pass
    elif values.shape[1] == dfX.shape[1]:
        # find index-like positions in dfX and remove corresponding cols from values
        idxs = [i for i, c in enumerate(dfX.columns) if str(c).strip().lower() in index_like]
        if idxs:
            try:
                values = np.delete(values, idxs, axis=1)
            except Exception:
                pass
    else:
        # best-effort: if values has fewer columns than names, trim names; if more, trim values
        if values.shape[1] < len(feature_names):
            feature_names = feature_names[: values.shape[1]]
        elif values.shape[1] > len(feature_names):
            values = values[:, : len(feature_names)]
    n_features = min(values.shape[1], max_features)

    # Compute mean absolute effect and order
    mean_abs = np.mean(np.abs(values), axis=0)
    order = np.argsort(mean_abs)[::-1][:n_features]
    selected_features = [feature_names[i] for i in order if i < len(feature_names)]

    # create a compact beeswarm-like horizontal strip for the top features
    fig, axes = plt.subplots(n_features, 1, figsize=(6.4, 0.28 * n_features + 2), sharex=True)
    if n_features == 1:
        axes = [axes]
    for ax, idx, fname in zip(axes, order, selected_features):
        vals = values[:, idx]
        # jitter along x for visibility
        y = np.random.normal(0, 0.04, size=len(vals))
        ax.scatter(vals, y, s=6, alpha=0.65, cmap=plt.get_cmap("coolwarm"), c=vals, edgecolors="none")
        ax.set_yticks([])
        ax.set_ylabel(fname, rotation=0, labelpad=80, va="center", fontsize=8)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    axes[-1].set_xlabel("SHAP value")
    fig.subplots_adjust(hspace=0.1, left=0.3)
    return _save(fig, out_path, dpi, fmt)


def _plot_shap_artifacts(run_dir: Path, output_dir: Path, dpi: int, fmt: str) -> list[str]:
    written: list[str] = []
    out_shap_dir = output_dir / "shap"
    out_shap_dir.mkdir(parents=True, exist_ok=True)

    # Search recursively for shap artifact files under the run directory.
    # Artifacts are commonly stored in stage subfolders like
    # <run>/levelX_stage/shap/*.npz / *_shap_importance.csv / *_shap_X.csv
    importance_files = list(run_dir.rglob("*_shap_importance.csv"))
    value_files = list(run_dir.rglob("*_shap_values.npz"))

    for csv in sorted(importance_files):
        # derive a stable base name from the containing stage folder + file stem
        stage = csv.parent.name
        base = csv.stem.replace("_shap_importance", "")
        out_name = f"{stage}_{base}_importance"
        written.extend(_plot_shap_importance(csv, out_shap_dir / out_name, dpi, fmt))

    for npz in sorted(value_files):
        stage = npz.parent.name
        base = npz.stem.replace("_shap_values", "")
        xcsv = npz.parent / f"{base}_shap_X.csv"
        out_name = f"{stage}_{base}_beeswarm"
        if xcsv.is_file():
            written.extend(_plot_shap_beeswarm(npz, xcsv, out_shap_dir / out_name, dpi, fmt))

    return written


# ---------------------------------------------------------------------------
# Main plotting logic
# ---------------------------------------------------------------------------

def _prediction_title(
    target_kind: str,
    stage_key: str,
    prediction_path: Path,
) -> str:
    target_name = {
        "sepsis": "Sepsis",
        "culture_gate": "Blood-culture gate",
        "resistance_gate": "Resistance gate",
        "gnb": "Etiology",
        "cef_resistance": "Cefalosporin resistance",
    }[target_kind]

    stage_name = {
        "direct": "Direct model",
        "level1_sepsis": "Level 1",
        "level2_etiology": "Level 2",
        "level3_cefalosporina": "Level 3",
    }.get(stage_key, stage_key)

    stem = prediction_path.stem
    suffix = re.sub(r"^predictions?_?", "", stem, flags=re.IGNORECASE)
    suffix = re.sub(r"[_-]+", " ", suffix).strip()

    if suffix:
        return f"{stage_name} — {target_name} ({suffix})"

    return f"{stage_name} — {target_name}"


def process_prediction_file(
    stage_key: str,
    stage_dir: Path,
    prediction_path: Path,
    summary: dict[str, Any],
    run_dir: Path,
    output_dir: Path,
    dpi: int,
    fmt: str,
) -> list[str]:
    try:
        df = pd.read_csv(prediction_path)
    except Exception as exc:
        print(f"WARNING: could not read {prediction_path}: {exc}", file=sys.stderr)
        return []

    if df.empty:
        return []

    parsed = _prediction_arrays(df, summary)
    if parsed is None:
        print(
            f"WARNING: no usable y_true/y_pred data in {prediction_path}",
            file=sys.stderr,
        )
        return []

    y_true, y_pred, scores, classes = parsed
    target_kind = _target_kind(stage_key, prediction_path, run_dir)
    target_title = _prediction_title(target_kind, stage_key, prediction_path)

    written: list[str] = []

    if y_pred is not None:
        written.extend(
            _plot_confusion(
                target_kind,
                target_title,
                y_true,
                y_pred,
                classes,
                output_dir,
                dpi,
                fmt,
            )
        )

    curve = _curve_data(
        target_kind,
        y_true,
        scores,
        classes,
    )

    if curve is not None:
        written.extend(
            _plot_curves(
                target_kind,
                target_title,
                curve,
                output_dir,
                dpi,
                fmt,
            )
        )
    else:
        print(
            f"WARNING: no usable probability scores for ROC/PR: "
            f"{prediction_path}",
            file=sys.stderr,
        )

    return written


def process_run(
    run_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_results = discover_stage_results(run_dir)

    print("=" * 72)
    print("MEPRAM RESULT PLOTTING")
    print("=" * 72)
    print(f"Run directory : {run_dir}")
    print(f"Output        : {output_dir}")
    print(
        "Stages found  : "
        + (", ".join(stage_results) if stage_results else "none")
    )

    written: list[str] = []

    for stage_key, _stage_label in STAGE_ORDER:
        if stage_key not in stage_results:
            continue

        stage_data = stage_results[stage_key]
        summary = stage_data["summary"]
        stage_dir = stage_data["directory"]

        if summary.get("skipped", False):
            print(f"Skipping stage: {stage_key}")
            continue

        prediction_files = _prediction_files(stage_dir)

        for prediction_path in prediction_files:
            print(f"Plotting: {prediction_path.name}")
            written.extend(
                process_prediction_file(
                    stage_key=stage_key,
                    stage_dir=stage_dir,
                    prediction_path=prediction_path,
                    summary=summary,
                    run_dir=run_dir,
                    output_dir=output_dir,
                    dpi=args.dpi,
                    fmt=args.fmt,
                )
            )

        # Classification reports are the authoritative source for
        # per-category Precision/Recall/F1.
        for report_path, report in discover_reports(stage_dir):
            target_kind = _report_target_kind(
                stage_key,
                report_path,
                run_dir,
            )
            target_title = _prediction_title(
                target_kind,
                stage_key,
                Path(report_path.stem.replace("report_", "predictions_") + ".csv"),
            )

            print(f"Plotting report: {report_path.name}")
            written.extend(
                _plot_class_metrics(
                    target_kind,
                    target_title,
                    report_path,
                    report,
                    output_dir,
                    args.dpi,
                    args.fmt,
                )
            )

    # Manifest: deliberately compact and reproducible.
    manifest = {
        "run_directory": str(run_dir.resolve()),
        "plots_directory": str(output_dir.resolve()),
        "n_plots": len(written),
        "plots": sorted(written),
        "stages_found": list(stage_results),
        "plot_types": [
            "confusion_matrix",
            "roc_curve",
            "pr_curve",
            "per_category_precision_recall_f1",
        ],
        "class_labels": TARGET_LABELS,
    }

    (output_dir / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    yaml_lines = [
        f"run_directory: {json.dumps(manifest['run_directory'])}",
        f"plots_directory: {json.dumps(manifest['plots_directory'])}",
        f"n_plots: {manifest['n_plots']}",
        "stages_found:",
    ]
    yaml_lines.extend(f"  - {stage}" for stage in manifest["stages_found"])
    yaml_lines.append("plot_types:")
    yaml_lines.extend(f"  - {kind}" for kind in manifest["plot_types"])
    yaml_lines.append("plots:")
    yaml_lines.extend(f"  - {json.dumps(plot)}" for plot in manifest["plots"])

    (output_dir / "plot_manifest.yaml").write_text(
        "\n".join(yaml_lines) + "\n"
    )

    # Processing plots: attempt to generate processing charts if processing
    # artifacts are present under <run_dir>/processing. These are optional
    # and best-effort: missing or unexpected CSVs are skipped with a warning.
    try:
        written_proc = _plot_processing_artifacts(run_dir, output_dir, args.dpi, args.fmt)
        written.extend(written_proc)
    except Exception as exc:
        print(f"WARNING: could not produce processing plots: {exc}", file=sys.stderr)
    try:
        written_shap = _plot_shap_artifacts(run_dir, output_dir, args.dpi, args.fmt)
        written.extend(written_shap)
    except Exception as exc:
        print(f"WARNING: could not produce SHAP plots: {exc}", file=sys.stderr)

    print("-" * 72)
    print(f"Created {len(written)} plot file(s).")
    print(f"Manifest: {output_dir / 'plot_manifest.yaml'}")
    print(f"Finished: {run_dir.name}")
    print()

    return len(written)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    root = args.run_dir.resolve()

    if not root.is_dir():
        print(f"ERROR: directory does not exist: {root}", file=sys.stderr)
        return 2

    run_dirs = discover_run_directories(root)

    print(f"Discovered {len(run_dirs)} modelling run(s) under: {root}")

    if not run_dirs:
        print(
            "ERROR: no 01-* result directories were found.\n"
            "Expected a scope directory such as 03-SEPSIS containing "
            "01-* folders, or a single 01-* run directory.",
            file=sys.stderr,
        )
        return 2

    total_plots = 0

    for run_dir in run_dirs:
        if args.output_dir is not None:
            if len(run_dirs) == 1:
                output_dir = args.output_dir.resolve()
            else:
                output_dir = args.output_dir.resolve() / run_dir.name
        else:
            output_dir = run_dir / "plots"

        total_plots += process_run(
            run_dir,
            output_dir,
            args,
        )

    print("=" * 72)
    print("ALL RUNS COMPLETED")
    print(f"Scope directory : {root}")
    print(f"Runs processed  : {len(run_dirs)}")
    print(f"Total plots     : {total_plots}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
