from __future__ import annotations

import argparse
import ast
import fnmatch
import math
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from build_summary_excel_file import (
    DEFAULT_DOMAIN_BY_STAGE,
    build_summary_excel_file,
)


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
DEFAULT_SUMMARY_LOG_PATH = Path("preprocess_test_log_summary.csv")
DEFAULT_DETAILED_LOG_PATH = Path("preprocess_test_log_detailed.json")
DEFAULT_FULL_DATASET_PATH = Path("preprocess_test.csv")
DEFAULT_FILTERED_DATASET_PATH = Path("preprocess_test_filtered.csv")
DEFAULT_OUTPUT_DIR = Path("preprocessing_report")
DEFAULT_REPORT_CONFIG_PATH = Path("preprocess_bacthecom_report.yml")

DEFAULT_COLORS = {
    "input": "#0072B2",
    "output": "#009E73",
    "created": "#009E73",
    "recoded": "#56B4E9",
    "transformed": "#CC79A7",
    "dropped": "#D55E00",
}

CHART_FONT_SIZES = {
    "title": 18,
    "axis_label": 15,
    "tick": 12,
    "legend": 12,
    "value": 11,
}
EXCLUDE_FROM_PLOTS = {"_pipeline_run"}

DEFAULT_REPORT_CONFIG = {
    "report_title": "BACTHECOM Mortality ML Preprocessing Report",
    "report_description": (
        "Preprocessing report for the BACTHECOM mortality prediction workflow. "
        "The pipeline creates one row per admission and selected first hemoculture, "
        "identified by record_id, fecha_ingreso, and fecha_hemocultivo. "
        "The modelling workflow is staged: first predicting any mortality, then "
        "predicting early mortality using either 14-day or 30-day mortality, "
        "referenced to the admission date."
    ),
    "exclude_plot_stages": ["_pipeline_run"],
    "domain_by_stage": {
        "_pipeline_run": "Pipeline",
        "paciente": "Patient",
        "episodio_ingreso": "Admission",
        "comorbilidad": "Comorbidity",
        "factores_riesgo_infeccion_bmr": "BMR risk factors",
        "signos_sintomas": "Symptoms and severity",
        "laboratorio": "Laboratory",
        "episodio_uci": "ICU",
        "episodio_infeccion": "Infection episodes",
        "merged_dataset": "Merged dataset",
        "filtered_dataset": "Model dataset",
    },
    "predictive_targets": [
        {
            "model": "Stage 1 - Any mortality",
            "target_column": "mortalidad_any",
            "notes": "Binary mortality prediction over the full selected hemoculture cohort. It is derived from fecha_mortalidad and/or the raw SQLite mortalidad column.",
            "class_labels": {0: "Alive", 1: "Dead", "0": "Alive", "1": "Dead"},
        },
        {
            "model": "Early mortality - 14 days from admission",
            "target_column": "mortalidad_14_dias",
            "notes": "Admission-referenced 14-day mortality.",
            "class_labels": {
                0: "No 14-day mortality",
                1: "14-day mortality",
                "0": "No 14-day mortality",
                "1": "14-day mortality",
            },
        },
        {
            "model": "Early mortality - 30 days from admission",
            "target_column": "mortalidad_30_dias",
            "notes": "Admission-referenced 30-day mortality.",
            "class_labels": {
                0: "No 30-day mortality",
                1: "30-day mortality",
                "0": "No 30-day mortality",
                "1": "30-day mortality",
            },
        },
    ],
    "target_evolution": [
        {
            "domain": "Mortality",
            "target_version": "Raw SQLite mortality",
            "target_columns": ["mortalidad"],
            "count_mode": "single",
            "notes": "Raw mortality flag from the SQLite source, when retained.",
        },
        {
            "domain": "Mortality",
            "target_version": "Stage 1 any mortality",
            "target_columns": ["mortalidad_any"],
            "count_mode": "single",
            "notes": "Main Stage-1 target for the full cohort.",
        },
        {
            "domain": "Early mortality",
            "target_version": "14 days from admission",
            "target_columns": ["mortalidad_14_dias"],
            "count_mode": "single",
            "notes": "Admission-referenced 14-day mortality target.",
        },
        {
            "domain": "Early mortality",
            "target_version": "30 days from admission",
            "target_columns": ["mortalidad_30_dias"],
            "count_mode": "single",
            "notes": "Admission-referenced 30-day mortality target.",
        },
    ],
    "prediction_detail_sections": [
        {
            "section": "Stage 1 - Any mortality",
            "description": "Binary mortality prediction target over the full selected hemoculture cohort.",
            "target_column": "mortalidad_any",
            "class_labels": {0: "Alive", 1: "Dead", "0": "Alive", "1": "Dead"},
        },
        {
            "section": "Early mortality - 14 days from admission",
            "description": "Admission-referenced 14-day mortality target.",
            "target_column": "mortalidad_14_dias",
            "class_labels": {
                0: "No 14-day mortality",
                1: "14-day mortality",
                "0": "No 14-day mortality",
                "1": "14-day mortality",
            },
        },
        {
            "section": "Early mortality - 30 days from admission",
            "description": "Admission-referenced 30-day mortality target.",
            "target_column": "mortalidad_30_dias",
            "class_labels": {
                0: "No 30-day mortality",
                1: "30-day mortality",
                "0": "No 30-day mortality",
                "1": "30-day mortality",
            },
        },
    ],
}



def merge_config(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_report_config(config_path: Path) -> dict[str, object]:
    if not config_path.exists():
        return dict(DEFAULT_REPORT_CONFIG)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return merge_config(DEFAULT_REPORT_CONFIG, loaded)


def infer_source_domain(stage_name: str | None, domain_by_stage: dict[str, str]) -> str:
    if not stage_name:
        return "Unmapped"
    return domain_by_stage.get(stage_name, stage_name.replace("_", " "))


def ordered_group_labels(domain_by_stage: dict[str, str]) -> list[str]:
    return list(dict.fromkeys(domain_by_stage.values()))


def resolve_subset(df: pd.DataFrame, subset_config: dict[str, object] | None) -> pd.DataFrame:
    if not subset_config:
        return df
    column = str(subset_config.get("column", "")).strip()
    if not column:
        return df
    subset = df
    if column not in subset.columns:
        return subset.iloc[0:0].copy()
    if "include_values" in subset_config:
        subset = subset.loc[subset[column].isin(list(subset_config["include_values"]))]
    if "exclude_values" in subset_config:
        subset = subset.loc[~subset[column].isin(list(subset_config["exclude_values"]))]
    return subset


def first_available_column(df: pd.DataFrame, candidates: list[object]) -> str | None:
    for candidate in candidates:
        name = str(candidate)
        if name in df.columns:
            return name
    return None


def filter_plot_stages(df: pd.DataFrame) -> pd.DataFrame:
    if "table_name" not in df.columns:
        return df
    return df.loc[~df["table_name"].isin(EXCLUDE_FROM_PLOTS)].copy()


def write_before_after_chart_png(
    df: pd.DataFrame,
    *,
    label_column: str,
    before_column: str,
    after_column: str,
    output_path: Path,
    title: str,
    x_label: str,
    colors: dict[str, str],
) -> None:
    plot_df = (
        filter_plot_stages(df)[[label_column, before_column, after_column]]
        .copy()
        .reset_index(drop=True)
    )
    plot_df[before_column] = pd.to_numeric(plot_df[before_column], errors="coerce").fillna(0)
    plot_df[after_column] = pd.to_numeric(plot_df[after_column], errors="coerce").fillna(0)
    height = max(6, 0.55 * len(plot_df))
    fig, ax = plt.subplots(figsize=(15, height))
    y_positions = list(range(len(plot_df)))
    bar_height = 0.38
    ax.barh(
        [position - bar_height / 2 for position in y_positions],
        plot_df[before_column],
        height=bar_height,
        label="Before",
        color=colors["input"],
    )
    ax.barh(
        [position + bar_height / 2 for position in y_positions],
        plot_df[after_column],
        height=bar_height,
        label="After",
        color=colors["output"],
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df[label_column], fontsize=CHART_FONT_SIZES["tick"])
    ax.invert_yaxis()
    ax.set_title(title, fontsize=CHART_FONT_SIZES["title"])
    ax.set_xlabel(x_label, fontsize=CHART_FONT_SIZES["axis_label"])
    ax.tick_params(axis="x", labelsize=CHART_FONT_SIZES["tick"])
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="upper right", fontsize=CHART_FONT_SIZES["legend"])
    max_value = max(plot_df[[before_column, after_column]].max().max(), 1)
    label_offset = max_value * 0.01
    ax.set_xlim(0, max_value * 1.14)
    for index, row in plot_df.iterrows():
        ax.text(
            row[before_column] + label_offset,
            index - bar_height / 2,
            f" {int(row[before_column]):,}",
            va="center",
            fontsize=CHART_FONT_SIZES["value"],
        )
        ax.text(
            row[after_column] + label_offset,
            index + bar_height / 2,
            f" {int(row[after_column]):,}",
            va="center",
            fontsize=CHART_FONT_SIZES["value"],
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_stacked_change_chart_png(
    change_counts: pd.DataFrame,
    *,
    output_path: Path,
    colors: dict[str, str],
) -> None:
    columns = ["created", "recoded", "transformed", "dropped"]
    plot_df = filter_plot_stages(change_counts)[["table_name"] + columns].copy()
    height = max(6, 0.55 * len(plot_df))
    fig, ax = plt.subplots(figsize=(15, height))
    left = pd.Series(0, index=plot_df.index, dtype=float)
    for column in columns:
        values = pd.to_numeric(plot_df[column], errors="coerce").fillna(0)
        ax.barh(
            plot_df["table_name"],
            values,
            left=left,
            label=column.capitalize(),
            color=colors[column],
        )
        left = left + values
    ax.invert_yaxis()
    ax.set_title(
        "Preprocessing Variables by Stage",
        fontsize=CHART_FONT_SIZES["title"],
    )
    ax.set_xlabel(
        "Number of variables",
        fontsize=CHART_FONT_SIZES["axis_label"],
    )
    ax.tick_params(axis="x", labelsize=CHART_FONT_SIZES["tick"])
    ax.tick_params(axis="y", labelsize=CHART_FONT_SIZES["tick"])
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=CHART_FONT_SIZES["legend"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_clinical_figure(fig, output_path):
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def feature_categories(columns, rules, detailed_logs, domain_by_stage):
    stages = first_stage_by_column(columns, detailed_logs)
    result = {}
    for column in columns:
        matched = next((label for label, rule in rules.items()
                        if any(fnmatch.fnmatchcase(column.lower(), pattern.lower())
                               for pattern in rule.get("patterns", []))), None)
        result[column] = matched or infer_source_domain(stages.get(column), domain_by_stage)
    return result


def write_missingness_chart_png(df, *, output_path, colors, limit=25):
    missing = df.isna().mean().mul(100).sort_values(ascending=False)
    plot = missing[missing.gt(0)].head(limit).sort_values()
    fig, ax = plt.subplots(figsize=(12, max(4, 0.31 * len(plot) + 1.5)), layout="constrained")
    if len(plot):
        bars = ax.barh(plot.index, plot.values, color=colors["input"], height=0.7)
        ax.bar_label(bars, labels=["<0.1%" if 0 < v < 0.1 else f"{v:.1f}%" for v in plot], padding=4, fontsize=9)
    else:
        ax.text(0.5, 0.5, "No missing predictor values", transform=ax.transAxes, ha="center")
    ax.set(title=f"Predictors with most missing data · top {len(plot)} of {len(df.columns)}",
           xlabel="Missing episodes (%)", xlim=(0, 112))
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    save_clinical_figure(fig, output_path)


def write_category_missingness_panels(df, mapping, labels, output_path, limit=6):
    groups = [(label, [c for c in df if mapping[c] == label]) for label in labels]
    groups = [(label, cols) for label, cols in groups if cols]
    if not groups:
        return
    fig, axes = plt.subplots(math.ceil(len(groups) / 2), 2, squeeze=False,
                             figsize=(18, 3.0 * math.ceil(len(groups) / 2)), layout="constrained")
    for ax, (label, columns) in zip(axes.flat, groups):
        missing = df[columns].isna().mean().mul(100).sort_values(ascending=False)
        plot = missing[missing.gt(0)].head(limit).sort_values(ascending=False)
        if len(plot):
            bars = ax.barh(plot.index, plot.values, color="#3B7EA1", height=0.65)
            ax.bar_label(bars, labels=["<0.1%" if 0 < v < 0.1 else f"{v:.1f}%" for v in plot], padding=3, fontsize=8)
        else:
            ax.text(0.5, 0.5, "All features complete", transform=ax.transAxes, ha="center", color="#26734D")
            ax.set_yticks([])
        ax.set(title=f"{label} · {len(columns)} predictors", xlim=(0, 112), xlabel="Missing episodes (%)")
        if len(plot):
            ax.set_ylim(max(3, len(plot)) - 0.4, -0.6)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(groups):]:
        ax.set_visible(False)
    fig.suptitle(f"Missingness within clinical categories · up to {limit} features per category", fontsize=16)
    save_clinical_figure(fig, output_path)


def first_stage_by_column(
    columns: list[str],
    detailed_logs: list[dict[str, object]],
) -> dict[str, str]:
    available_columns = set(columns)
    first_stage: dict[str, str] = {}
    for stage in detailed_logs:
        stage_name = str(stage.get("table_name", ""))
        if stage_name in {"_pipeline_run", "merged_dataset", "filtered_dataset"}:
            continue
        for column in stage.get("output_columns", []):
            column = str(column)
            if column in available_columns:
                first_stage.setdefault(column, stage_name)
    return first_stage


def grouped_slide_missingness(
    df: pd.DataFrame,
    detailed_logs: list[dict[str, object]],
    domain_by_stage: dict[str, str],
    variable_group_labels: list[str],
) -> pd.DataFrame:
    missing_percent = df.isna().mean().mul(100)
    first_stage = first_stage_by_column(df.columns.tolist(), detailed_logs)
    groups: dict[str, list[str]] = defaultdict(list)
    for column in df.columns:
        groups[infer_source_domain(first_stage.get(column), domain_by_stage)].append(column)

    bands = [
        ("0%", lambda values: values == 0),
        (">0-10%", lambda values: (values > 0) & (values <= 10)),
        ("10-40%", lambda values: (values > 10) & (values <= 40)),
        ("40-80%", lambda values: (values > 40) & (values <= 80)),
        ("80-100%", lambda values: values > 80),
    ]
    rows = []
    ordered_groups = [
        group for group in variable_group_labels
        if groups.get(group)
    ]
    ordered_groups.extend(
        sorted(group for group in groups if group not in set(ordered_groups))
    )
    for label in ordered_groups:
        columns = groups[label]
        if not columns:
            continue
        group_missing = missing_percent[columns]
        row = {"group": label, "variables": len(columns)}
        for band_label, mask_fn in bands:
            row[band_label] = int(mask_fn(group_missing).sum())
        row["max_missing_percent"] = round(float(group_missing.max()), 1)
        rows.append(
            row
        )

    return pd.DataFrame.from_records(rows, columns=["group", "variables", *[label for label, _ in bands], "max_missing_percent"])


def write_grouped_missingness_chart_png(
    df: pd.DataFrame,
    *,
    detailed_logs: list[dict[str, object]],
    output_path: Path,
    domain_by_stage: dict[str, str],
    variable_group_labels: list[str],
) -> pd.DataFrame:
    plot_df = grouped_slide_missingness(
        df,
        detailed_logs,
        domain_by_stage,
        variable_group_labels,
    )
    bands = ["0%", ">0-10%", "10-40%", "40-80%", "80-100%"]
    colors = ["#009E73", "#56B4E9", "#E6AB02", "#D55E00", "#8C3B6A"]
    fig, ax = plt.subplots(figsize=(12, max(4, 0.48 * len(plot_df) + 2)), layout="constrained")
    left = pd.Series(0.0, index=plot_df.index)
    for band, color in zip(bands, colors):
        values = plot_df[band].div(plot_df["variables"]).mul(100)
        bars = ax.barh(plot_df["group"], values, left=left, label=band, color=color, height=0.72)
        for bar, count, width in zip(bars, plot_df[band], values):
            if width >= 8:
                ax.text(bar.get_x() + width / 2, bar.get_y() + bar.get_height() / 2,
                        str(count), ha="center", va="center", fontsize=9, color="white" if band in ["0%", "40-80%", "80-100%"] else "#222222")
        left += values
    ax.set_yticks(range(len(plot_df)), [f"{r.group} (n={r.variables})" for r in plot_df.itertuples()])
    ax.invert_yaxis()
    ax.set(title="Predictor completeness by clinical category",
           xlabel="Share of predictors (%) · segment labels are feature counts", xlim=(0, 100))
    ax.legend(title="Missing values per feature", loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=5, frameon=False)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    save_clinical_figure(fig, output_path)
    return plot_df


def variable_distribution_by_group(
    columns: list[str],
    detailed_logs: list[dict[str, object]],
    domain_by_stage: dict[str, str],
    variable_group_labels: list[str],
) -> pd.DataFrame:
    available_columns = set(columns)
    first_stage_by_column: dict[str, str] = {}
    for stage in detailed_logs:
        stage_name = str(stage.get("table_name", ""))
        if stage_name in {"_pipeline_run", "merged_dataset", "filtered_dataset"}:
            continue
        for column in stage.get("output_columns", []):
            column = str(column)
            if column in available_columns:
                first_stage_by_column.setdefault(column, stage_name)

    counts = Counter(
        infer_source_domain(first_stage_by_column.get(column), domain_by_stage)
        for column in columns
    )

    records = []
    ordered_groups = [
        group for group in variable_group_labels
        if counts.get(group, 0) > 0
    ]
    ordered_groups.extend(
        sorted(group for group in counts if group not in set(ordered_groups))
    )
    for group in ordered_groups:
        records.append({"group": group, "variables": int(counts[group])})

    return pd.DataFrame.from_records(records)


def variable_distribution_filtered_vs_unfiltered(
    *,
    full_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    detailed_logs: list[dict[str, object]],
    domain_by_stage: dict[str, str],
    variable_group_labels: list[str],
) -> pd.DataFrame:
    full_distribution = variable_distribution_by_group(
        full_df.columns.tolist(),
        detailed_logs,
        domain_by_stage,
        variable_group_labels,
    )
    filtered_distribution = variable_distribution_by_group(
        filtered_df.columns.tolist(),
        detailed_logs,
        domain_by_stage,
        variable_group_labels,
    )
    result = full_distribution.rename(columns={"variables": "unfiltered_variables"}).merge(
        filtered_distribution.rename(columns={"variables": "filtered_variables"}),
        on="group",
        how="outer",
    )
    result["unfiltered_variables"] = result["unfiltered_variables"].fillna(0).astype(int)
    result["filtered_variables"] = result["filtered_variables"].fillna(0).astype(int)
    result["removed_by_filter"] = (
        result["unfiltered_variables"] - result["filtered_variables"]
    )
    return result


def write_variable_distribution_chart_png(
    distribution: pd.DataFrame,
    *,
    output_path: Path,
    colors: dict[str, str],
) -> None:
    fig, ax = plt.subplots(figsize=(12, max(4, 0.55 * len(distribution) + 2)), layout="constrained")
    positions = list(range(len(distribution)))
    for shift, column, label, color in [(-0.19, "unfiltered_variables", "Audit table", colors["input"]),
                                       (0.19, "filtered_variables", "Filtered table", colors["output"])]:
        bars = ax.barh([p + shift for p in positions], distribution[column], height=0.34,
                       color=color, label=f"{label} (n={int(distribution[column].sum())})")
        ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_yticks(positions, distribution["group"])
    ax.invert_yaxis()
    ax.set(title="Feature counts by clinical category", xlabel="Number of columns")
    ax.margins(x=0.15)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
    save_clinical_figure(fig, output_path)


def predictive_target_summary(
    filtered_df: pd.DataFrame,
    target_specs: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_records = []
    class_records = []
    for spec in target_specs:
        target_column = str(spec["target_column"])
        subset = resolve_subset(filtered_df, spec.get("subset"))
        if target_column not in subset.columns:
            continue
        target = subset[target_column].dropna()
        counts = target.value_counts(dropna=False)
        sample_count = int(target.shape[0])
        majority_count = int(counts.max()) if sample_count else 0
        majority_raw = counts.idxmax() if sample_count else ""
        class_labels = dict(spec.get("class_labels", {}))
        majority_label = class_labels.get(majority_raw, str(majority_raw))
        class_count = int(target.nunique(dropna=True))
        majority_percent = round((majority_count / sample_count) * 100, 1) if sample_count else 0

        summary_records.append(
            {
                "model": str(spec["model"]),
                "target_column": target_column,
                "samples": sample_count,
                "classes": class_count,
                "majority_class": majority_label,
                "majority_class_percent": majority_percent,
                "exclusions_or_grouping": str(spec.get("notes", "")),
            }
        )

        for raw_value, count in counts.sort_index().items():
            class_records.append(
                {
                    "model": str(spec["model"]),
                    "target_column": target_column,
                    "class": class_labels.get(raw_value, str(raw_value)),
                    "raw_value": raw_value,
                    "samples": int(count),
                    "percent": round((int(count) / sample_count) * 100, 1)
                    if sample_count
                    else 0,
                }
            )

    return pd.DataFrame.from_records(summary_records), pd.DataFrame.from_records(class_records)


def normalize_list_target(value: object) -> str:
    labels = parse_list_target(value)
    if not labels:
        return "NEGATIVE"
    return " + ".join(labels)


def parse_list_target(value: object) -> list[str]:
    if isinstance(value, list):
        labels = value
    elif pd.isna(value):
        labels = []
    else:
        try:
            parsed = ast.literal_eval(str(value))
            labels = parsed if isinstance(parsed, list) else [parsed]
        except (SyntaxError, ValueError):
            labels = []
    labels = [str(label) for label in labels if str(label)]
    return labels


def multilabel_balance_records(
    *,
    domain: str,
    target_version: str,
    target_column: str,
    values: pd.Series,
    notes: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    labels_by_row = values.map(parse_list_target)
    sample_count = int(labels_by_row.shape[0])
    exploded = labels_by_row.map(lambda labels: labels if labels else ["NEGATIVE"]).explode()
    counts = exploded.value_counts(dropna=False)
    classes = int(counts.shape[0])
    majority_class = str(counts.index[0]) if classes else ""
    majority_count = int(counts.iloc[0]) if classes else 0
    majority_percent = round((majority_count / sample_count) * 100, 1) if sample_count else 0
    top_classes = "; ".join(
        f"{label}: {int(count):,}" for label, count in counts.head(4).items()
    )
    summary = {
        "domain": domain,
        "target_version": target_version,
        "target_column": target_column,
        "samples": sample_count,
        "classes": classes,
        "majority_class": majority_class,
        "majority_class_count": majority_count,
        "majority_class_percent": majority_percent,
        "top_classes": top_classes,
        "notes": notes,
    }
    records = [
        {
            "domain": domain,
            "target_version": target_version,
            "target_column": target_column,
            "class": str(label),
            "samples": int(count),
            "percent": round((int(count) / sample_count) * 100, 1)
            if sample_count
            else 0,
        }
        for label, count in counts.items()
    ]
    return summary, records


def series_balance_records(
    *,
    domain: str,
    target_version: str,
    target_column: str,
    values: pd.Series,
    notes: str,
    top_n: int = 4,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    target = values.dropna()
    counts = target.value_counts(dropna=False)
    sample_count = int(target.shape[0])
    class_count = int(target.nunique(dropna=True))
    majority_count = int(counts.max()) if sample_count else 0
    majority_class = str(counts.idxmax()) if sample_count else ""
    majority_percent = round((majority_count / sample_count) * 100, 1) if sample_count else 0
    top_classes = "; ".join(
        f"{class_name}: {int(count):,}"
        for class_name, count in counts.head(top_n).items()
    )
    summary = {
        "domain": domain,
        "target_version": target_version,
        "target_column": target_column,
        "samples": sample_count,
        "classes": class_count,
        "majority_class": majority_class,
        "majority_class_count": majority_count,
        "majority_class_percent": majority_percent,
        "top_classes": top_classes,
        "notes": notes,
    }
    class_records = [
        {
            "domain": domain,
            "target_version": target_version,
            "target_column": target_column,
            "class": str(class_name),
            "samples": int(count),
            "percent": round((int(count) / sample_count) * 100, 1)
            if sample_count
            else 0,
        }
        for class_name, count in counts.items()
    ]
    return summary, class_records


def target_evolution_summary(
    filtered_df: pd.DataFrame,
    target_specs: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    class_records: list[dict[str, object]] = []
    for spec in target_specs:
        candidates = spec.get("target_columns") or [spec.get("target_column")]
        target_column = first_available_column(filtered_df, list(candidates))
        if target_column is None:
            continue
        count_mode = str(spec.get("count_mode", "single"))
        if count_mode == "multilabel":
            summary, records = multilabel_balance_records(
                domain=str(spec["domain"]),
                target_version=str(spec["target_version"]),
                target_column=target_column,
                values=filtered_df[target_column],
                notes=str(spec.get("notes", "")),
            )
        else:
            summary, records = series_balance_records(
                domain=str(spec["domain"]),
                target_version=str(spec["target_version"]),
                target_column=target_column,
                values=filtered_df[target_column],
                notes=str(spec.get("notes", "")),
            )
        summaries.append(summary)
        class_records.extend(records)

    return pd.DataFrame.from_records(summaries), pd.DataFrame.from_records(class_records)


def class_count_records(
    *,
    section: str,
    target_column: str,
    values: pd.Series,
    class_labels: dict[object, str] | None = None,
    top_n: int | None = None,
    other_label: str = "Other",
) -> list[dict[str, object]]:
    target = values.dropna()
    counts = target.value_counts(dropna=False)
    sample_count = int(target.shape[0])

    rows: list[dict[str, object]] = []
    count_items = list(counts.items())
    if top_n is not None and len(count_items) > top_n:
        displayed_items = count_items[:top_n]
        other_count = sum(int(count) for _, count in count_items[top_n:])
        displayed_items.append((other_label, other_count))
    else:
        displayed_items = count_items

    for class_value, count in displayed_items:
        class_name = (
            class_labels.get(class_value, str(class_value))
            if class_labels is not None
            else str(class_value)
        )
        rows.append(
            {
                "section": section,
                "target_column": target_column,
                "class": class_name,
                "rows": int(count),
                "percent": round((int(count) / sample_count) * 100, 1)
                if sample_count
                else 0,
            }
        )
    rows.append(
        {
            "section": section,
            "target_column": target_column,
            "class": "Total",
            "rows": sample_count,
            "percent": 100.0 if sample_count else 0,
        }
    )
    return rows


def multilabel_class_count_records(
    *,
    section: str,
    target_column: str,
    values: pd.Series,
) -> list[dict[str, object]]:
    labels_by_row = values.dropna().map(parse_list_target)
    sample_count = int(labels_by_row.shape[0])
    exploded = labels_by_row.map(lambda labels: labels if labels else ["NEGATIVE"]).explode()
    counts = exploded.value_counts(dropna=False)

    rows = [
        {
            "section": section,
            "target_column": target_column,
            "class": str(label),
            "rows": int(count),
            "percent": round((int(count) / sample_count) * 100, 1)
            if sample_count
            else 0,
        }
        for label, count in counts.items()
    ]
    rows.append(
        {
            "section": section,
            "target_column": target_column,
            "class": "Total label assignments",
            "rows": int(counts.sum()),
            "percent": round((int(counts.sum()) / sample_count) * 100, 1)
            if sample_count
            else 0,
        }
    )
    rows.append(
        {
            "section": section,
            "target_column": target_column,
            "class": "Total patients",
            "rows": sample_count,
            "percent": 100.0 if sample_count else 0,
        }
    )
    return rows


def prediction_detail_tables(
    filtered_df: pd.DataFrame,
    section_specs: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_count = len(filtered_df)
    column_count = len(filtered_df.columns)

    summary_records = []
    class_records = []
    for spec in section_specs:
        target_column = str(spec["target_column"])
        if target_column not in filtered_df.columns:
            continue
        values = filtered_df[target_column].dropna()
        if spec.get("count_mode") == "multilabel":
            labels = values.map(parse_list_target)
            class_count = int(
                labels.map(lambda row_labels: row_labels if row_labels else ["NEGATIVE"])
                .explode()
                .nunique(dropna=True)
            )
        else:
            class_count = int(values.nunique(dropna=True))
        summary_records.append(
            {
                "section": str(spec["section"]),
                "description": str(spec["description"]),
                "target_column": target_column,
                "rows": row_count,
                "columns": column_count,
                "classes": class_count,
            }
        )
        if spec.get("count_mode") == "multilabel":
            class_records.extend(
                multilabel_class_count_records(
                    section=str(spec["section"]),
                    target_column=target_column,
                    values=values,
                )
            )
        else:
            class_records.extend(
                class_count_records(
                    section=str(spec["section"]),
                    target_column=target_column,
                    values=values,
                    class_labels=dict(spec.get("class_labels", {})) or None,
                    top_n=spec.get("top_n"),
                )
            )

    return pd.DataFrame.from_records(summary_records), pd.DataFrame.from_records(class_records)


def count_change_variables(changes: list[dict[str, object]], field: str) -> int:
    variables: set[str] = set()
    non_column_targets = {"row_filter"}
    for change in changes:
        value = change.get(field, [])
        values = value if isinstance(value, list) else [value]
        for item in values:
            if not item:
                continue
            variable = str(item)
            if variable in non_column_targets:
                continue
            variables.add(variable)
    return len(variables)


def summarize_change_counts(detailed_logs: list[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for log in detailed_logs:
        created_variables = log.get("created_variables", [])
        recoded_variables = log.get("recoded_variables", [])
        transformed_variables = log.get("transformed_variables", [])
        dropped_variables = log.get("dropped_variables", [])
        records.append(
            {
                "table_name": log["table_name"],
                "created": count_change_variables(created_variables, "target"),
                "recoded": count_change_variables(recoded_variables, "target"),
                "transformed": count_change_variables(transformed_variables, "target"),
                "dropped": count_change_variables(dropped_variables, "source"),
                "warnings": len(log.get("warnings", [])),
                "notes": len(log.get("notes", [])),
            }
        )
    return pd.DataFrame.from_records(records)


def top_missingness(df: pd.DataFrame, *, top_n: int = 15) -> pd.DataFrame:
    missing = df.isna().mean().sort_values(ascending=False).head(top_n)
    return (
        missing.reset_index()
        .rename(columns={"index": "column", 0: "missing_fraction"})
        .assign(missing_percent=lambda data: (data["missing_fraction"] * 100).round(1))
    )


def write_markdown_report(
    *,
    output_path: Path,
    report_title: str,
    report_description: str,
    summary: pd.DataFrame,
    change_counts: pd.DataFrame,
    detailed_logs: list[dict[str, Any]],
    full_df: pd.DataFrame | None,
    filtered_df: pd.DataFrame | None,
    target_summary: pd.DataFrame | None,
    target_class_counts: pd.DataFrame | None,
    evolution_summary: pd.DataFrame | None,
    evolution_class_counts: pd.DataFrame | None,
    prediction_detail_summary: pd.DataFrame | None,
    prediction_detail_class_counts: pd.DataFrame | None,
    chart_paths: dict[str, Path],
) -> None:
    def report_link(path: Path) -> str:
        return path.relative_to(output_path.parent).as_posix()

    filtered_row = summary.loc[summary["table_name"] == "filtered_dataset"]
    final_row = summary.loc[summary["table_name"] == "target_building"]
    full_shape = (
        f"{len(full_df):,} rows x {len(full_df.columns):,} columns"
        if full_df is not None
        else "not provided"
    )
    filtered_shape = (
        f"{len(filtered_df):,} rows x {len(filtered_df.columns):,} columns"
        if filtered_df is not None
        else "not provided"
    )

    warnings = [
        (log["table_name"], warning)
        for log in detailed_logs
        for warning in log.get("warnings", [])
    ]
    top_created = (
        change_counts.sort_values("created", ascending=False)
        [["table_name", "created"]]
        .head(8)
        .to_dict(orient="records")
    )

    lines = [
        f"# {report_title}",
        "",
        report_description,
        "",
        "## Dataset Shapes",
        "",
        f"- Full dataset: `{full_shape}`",
        f"- Filtered/model-compatible dataset: `{filtered_shape}`",
    ]
    if not final_row.empty:
        row = final_row.iloc[0]
        lines.append(
            f"- Final pipeline stage before filtering: `{int(row['output_rows']):,}` rows and `{int(row['output_column_count']):,}` columns"
        )
    if not filtered_row.empty:
        row = filtered_row.iloc[0]
        lines.append(
            f"- Filtered stage removed `{int(row['columns_dropped_count']):,}` columns and kept `{int(row['output_column_count']):,}` columns"
        )

    lines.extend(
        [
            "",
            "## Figures",
            "",
            f"![Rows before and after preprocessing stage]({report_link(chart_paths['rows'])})",
            "",
            f"![Columns before and after preprocessing stage]({report_link(chart_paths['columns'])})",
            "",
            f"![Preprocessing variables by stage]({report_link(chart_paths['changes'])})",
            "",
            f"![Missingness in filtered variables]({report_link(chart_paths['missingness'])})",
            "",
            f"![Grouped missingness for slides]({report_link(chart_paths['grouped_missingness'])})",
            "",
            f"![Variable distribution by group]({report_link(chart_paths['variable_distribution'])})",
            "",
            "## Most Feature-Creating Stages",
            "",
        ]
    )
    for item in top_created:
        lines.append(f"- `{item['table_name']}`: `{item['created']}` created variables")

    lines.extend(["", "## Predictive Targets", ""])
    if target_summary is not None and target_class_counts is not None:
        lines.extend(
            [
                "| Model | Target column | Samples | Classes | Majority class | Majority class % |",
                "|---|---|---:|---:|---|---:|",
            ]
        )
        for _, row in target_summary.iterrows():
            lines.append(
                f"| {row['model']} | `{row['target_column']}` | "
                f"{int(row['samples']):,} | {int(row['classes'])} | "
                f"{row['majority_class']} | {row['majority_class_percent']}% |"
            )

        lines.extend(
            [
                "",
                "| Model | Class | Samples | % within target |",
                "|---|---|---:|---:|",
            ]
        )
        for _, row in target_class_counts.iterrows():
            lines.append(
                f"| {row['model']} | {row['class']} | "
                f"{int(row['samples']):,} | {row['percent']}% |"
            )

        lines.extend(
            [
                "",
                "| Model | Exclusions or grouping |",
                "|---|---|",
            ]
        )
        for _, row in target_summary.iterrows():
            lines.append(f"| {row['model']} | {row['exclusions_or_grouping']} |")
    else:
        lines.append("- Filtered dataset was not provided, so target summaries were not computed.")

    lines.extend(["", "## Target Evolution", ""])
    if evolution_summary is not None and evolution_class_counts is not None:
        lines.extend(
            [
                "| Domain | Target version | Target column | Samples | Classes | Majority class | Majority class % | Top classes |",
                "|---|---|---|---:|---:|---|---:|---|",
            ]
        )
        for _, row in evolution_summary.iterrows():
            lines.append(
                f"| {row['domain']} | {row['target_version']} | "
                f"`{row['target_column']}` | {int(row['samples']):,} | "
                f"{int(row['classes'])} | {row['majority_class']} | "
                f"{row['majority_class_percent']}% | {row['top_classes']} |"
            )

        lines.extend(["", "| Domain | Target version | Notes |", "|---|---|---|"])
        for _, row in evolution_summary.iterrows():
            lines.append(f"| {row['domain']} | {row['target_version']} | {row['notes']} |")
    else:
        lines.append("- Filtered dataset was not provided, so target evolution was not computed.")

    lines.extend(["", "## Prediction Target Details", ""])
    if (
        prediction_detail_summary is not None
        and prediction_detail_class_counts is not None
    ):
        for _, section_row in prediction_detail_summary.iterrows():
            section = section_row["section"]
            lines.extend(
                [
                    f"### {section}",
                    "",
                    section_row["description"],
                    "",
                    f"- Dataset shape: `{int(section_row['rows']):,}` rows x `{int(section_row['columns']):,}` columns",
                    f"- Target column: `{section_row['target_column']}`",
                    f"- Classes: `{int(section_row['classes']):,}`",
                    "",
                    "| Class | Rows | % within target |",
                    "|---|---:|---:|",
                ]
            )
            class_rows = prediction_detail_class_counts.loc[
                prediction_detail_class_counts["section"] == section
            ]
            for _, class_row in class_rows.iterrows():
                lines.append(
                    f"| {class_row['class']} | {int(class_row['rows']):,} | "
                    f"{class_row['percent']}% |"
                )
            lines.append("")
    else:
        lines.append("- Filtered dataset was not provided, so prediction detail tables were not computed.")

    lines.extend(["", "## Warnings", ""])
    if warnings:
        for table_name, warning in warnings:
            lines.append(f"- `{table_name}`: {warning}")
    else:
        lines.append("- No warnings were recorded in the preprocessing log.")

    lines.extend(["", "## Missingness Snapshot", ""])
    if filtered_df is not None:
        missing = top_missingness(filtered_df)
        lines.append("Top missing columns in the filtered dataset:")
        lines.append("")
        for _, row in missing.iterrows():
            lines.append(f"- `{row['column']}`: `{row['missing_percent']}%` missing")
    else:
        lines.append("- Filtered dataset was not provided, so missingness was not computed.")

    lines.extend(["", "## Table Summary", ""])
    for _, row in summary.iterrows():
        lines.append(
            f"- `{row['table_name']}`: rows `{int(row['input_rows']):,}` -> `{int(row['output_rows']):,}`, "
            f"columns `{int(row['input_column_count']):,}` -> `{int(row['output_column_count']):,}`, "
            f"dropped `{int(row['columns_dropped_count']):,}`, warnings `{int(row['warning_count']):,}`"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(
    *,
    summary_log_path: Path,
    detailed_log_path: Path,
    full_dataset_path: Path | None,
    filtered_dataset_path: Path | None,
    output_dir: Path,
    report_config_path: Path = DEFAULT_REPORT_CONFIG_PATH,
    build_summary_excel: bool = True,
) -> None:
    report_config = load_report_config(report_config_path)
    domain_by_stage = dict(report_config["domain_by_stage"])
    variable_group_labels = ordered_group_labels(domain_by_stage)
    colors = merge_config(DEFAULT_COLORS, dict(report_config.get("colors", {})))
    global EXCLUDE_FROM_PLOTS
    EXCLUDE_FROM_PLOTS = set(report_config.get("exclude_plot_stages", ["_pipeline_run"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = output_dir / "graphs"
    tables_dir = output_dir / "tables"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_log_path, low_memory=False)
    detailed_logs = json.loads(detailed_log_path.read_text(encoding="utf-8"))
    change_counts = summarize_change_counts(detailed_logs)

    full_df = (
        pd.read_csv(full_dataset_path, low_memory=False)
        if full_dataset_path and full_dataset_path.exists()
        else None
    )
    filtered_df = (
        pd.read_csv(filtered_dataset_path, low_memory=False)
        if filtered_dataset_path and filtered_dataset_path.exists()
        else None
    )

    all_columns = list(dict.fromkeys([*(full_df.columns if full_df is not None else []),
                                     *(filtered_df.columns if filtered_df is not None else [])]))
    mapping = feature_categories(all_columns, report_config.get("feature_categories", {}), detailed_logs, domain_by_stage)
    variable_group_labels = list(dict.fromkeys([*report_config.get("feature_categories", {}), *mapping.values()]))
    clinical_logs = [{"table_name": label, "output_columns": [c for c in all_columns if mapping[c] == label]}
                     for label in variable_group_labels]
    clinical_domains = {label: label for label in variable_group_labels}
    plot_options = report_config.get("clinical_plots", {})
    predictors = filtered_df
    predictor_columns = []
    if filtered_df is not None:
        schema_path = filtered_dataset_path.with_name(filtered_dataset_path.stem.removesuffix("_filtered") + "_schema.json")
        if schema_path.exists():
            predictor_columns = json.loads(schema_path.read_text())["predictor_columns"]
            missing = set(predictor_columns) - set(filtered_df)
            if missing:
                raise ValueError(f"Schema predictors missing from filtered table: {sorted(missing)}")
        else:
            targets = {t["target_column"] for t in report_config.get("predictive_targets", [])}
            excluded = set(plot_options.get("exclude_predictor_categories", []))
            predictor_columns = [c for c in filtered_df if c not in targets and mapping[c] not in excluded]
        predictors = filtered_df[predictor_columns]
    membership = pd.DataFrame([{"feature": c, "category": mapping[c],
        "in_filtered": filtered_df is not None and c in filtered_df,
        "is_predictor": c in predictor_columns} for c in all_columns])
    membership.to_csv(tables_dir / "feature_categories.csv", index=False)
    if predictors is not None:
        pd.DataFrame({"feature": predictors.columns,
                      "category": [mapping[c] for c in predictors],
                      "missing_percent": predictors.isna().mean().mul(100).values}).to_csv(
            tables_dir / "predictor_missingness.csv", index=False)

    rows_chart = graphs_dir / "rows_by_stage.png"
    columns_chart = graphs_dir / "columns_by_stage.png"
    changes_chart = graphs_dir / "logged_operations_by_stage.png"
    missingness_chart = graphs_dir / "filtered_missingness.png"
    grouped_missingness_chart = graphs_dir / "filtered_missingness_grouped_for_slides.png"
    variable_distribution_chart = (
        graphs_dir / "variable_distribution_filtered_vs_unfiltered.png"
    )

    write_before_after_chart_png(
        summary,
        label_column="table_name",
        before_column="input_rows",
        after_column="output_rows",
        output_path=rows_chart,
        title="Rows Before and After Each Preprocessing Stage",
        x_label="Rows",
        colors=colors,
    )
    write_before_after_chart_png(
        summary,
        label_column="table_name",
        before_column="input_column_count",
        after_column="output_column_count",
        output_path=columns_chart,
        title="Columns Before and After Each Preprocessing Stage",
        x_label="Columns",
        colors=colors,
    )
    write_stacked_change_chart_png(
        change_counts,
        output_path=changes_chart,
        colors=colors,
    )
    if filtered_df is not None:
        write_missingness_chart_png(
            predictors,
            output_path=missingness_chart,
            colors=colors,
            limit=int(plot_options.get("top_missing_features", 25)),
        )
        grouped_missingness = write_grouped_missingness_chart_png(
            predictors,
            detailed_logs=clinical_logs,
            output_path=grouped_missingness_chart,
            domain_by_stage=clinical_domains,
            variable_group_labels=variable_group_labels,
        )
        grouped_missingness.to_csv(
            tables_dir / "grouped_missingness_for_slides.csv",
            index=False,
        )
    if full_df is not None and filtered_df is not None:
        variable_distribution = variable_distribution_filtered_vs_unfiltered(
            full_df=full_df,
            filtered_df=filtered_df,
            detailed_logs=clinical_logs,
            domain_by_stage=clinical_domains,
            variable_group_labels=variable_group_labels,
        )
        write_variable_distribution_chart_png(
            variable_distribution,
            output_path=variable_distribution_chart,
            colors=colors,
        )
        variable_distribution.to_csv(
            tables_dir / "variable_distribution_filtered_vs_unfiltered.csv",
            index=False,
        )
    if predictors is not None and len(predictors.columns):
        write_category_missingness_panels(predictors, mapping, variable_group_labels,
            graphs_dir / "missingness_by_clinical_category.png",
            limit=int(plot_options.get("top_missing_per_category", 6)))
    target_summary = None
    target_class_counts = None
    evolution_summary = None
    evolution_class_counts = None
    prediction_detail_summary = None
    prediction_detail_class_counts = None
    if filtered_df is not None:
        target_summary, target_class_counts = predictive_target_summary(
            filtered_df,
            list(report_config.get("predictive_targets", [])),
        )
        target_summary.to_csv(tables_dir / "predictive_targets_summary.csv", index=False)
        target_class_counts.to_csv(
            tables_dir / "predictive_targets_class_counts.csv",
            index=False,
        )
        evolution_summary, evolution_class_counts = target_evolution_summary(
            filtered_df,
            list(report_config.get("target_evolution", [])),
        )
        evolution_summary.to_csv(tables_dir / "target_evolution_summary.csv", index=False)
        evolution_class_counts.to_csv(
            tables_dir / "target_evolution_class_counts.csv",
            index=False,
        )
        prediction_detail_summary, prediction_detail_class_counts = (
            prediction_detail_tables(
                filtered_df,
                list(report_config.get("prediction_detail_sections", [])),
            )
        )
        prediction_detail_summary.to_csv(
            tables_dir / "prediction_detail_summary.csv",
            index=False,
        )
        prediction_detail_class_counts.to_csv(
            tables_dir / "prediction_detail_class_counts.csv",
            index=False,
        )

    change_counts.to_csv(tables_dir / "change_counts_by_stage.csv", index=False)
    write_markdown_report(
        output_path=output_dir / "preprocessing_report.md",
        report_title=str(report_config.get("report_title", "Preprocessing Report")),
        report_description=str(report_config.get("report_description", "")),
        summary=summary,
        change_counts=change_counts,
        detailed_logs=detailed_logs,
        full_df=full_df,
        filtered_df=filtered_df,
        target_summary=target_summary,
        target_class_counts=target_class_counts,
        evolution_summary=evolution_summary,
        evolution_class_counts=evolution_class_counts,
        prediction_detail_summary=prediction_detail_summary,
        prediction_detail_class_counts=prediction_detail_class_counts,
        chart_paths={
            "rows": rows_chart,
            "columns": columns_chart,
            "changes": changes_chart,
            "missingness": missingness_chart,
            "grouped_missingness": grouped_missingness_chart,
            "variable_distribution": variable_distribution_chart,
        },
    )
    with (output_dir / "preprocessing_report.md").open("a") as handle:
        handle.write("\n## Clinical feature categories\n\n"
                     "Categories follow the first matching YAML rule. Predictor missingness plots exclude targets and metadata using the schema when available.\n\n"
                     "[Feature-to-category mapping](tables/feature_categories.csv) · [Complete predictor missingness table](tables/predictor_missingness.csv)\n\n"
                     "![Missingness by clinical category](graphs/missingness_by_clinical_category.png)\n\n"
                     "Clinical plots are also exported as SVG for resizing. A zero missingness rate does not establish confirmed absence for history flags.\n")
    if build_summary_excel and full_dataset_path and filtered_dataset_path:
        build_summary_excel_file(
            full_dataset_path=full_dataset_path,
            filtered_dataset_path=filtered_dataset_path,
            detailed_log_path=detailed_log_path,
            output_path=tables_dir / "summary_excel_file.xlsx",
            domain_by_stage=domain_by_stage,
            feature_domain_map=mapping,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate preprocessing summary figures, tables, markdown, and Excel output from preprocessing logs."
    )
    parser.add_argument("--summary-log-path", type=Path, default=DEFAULT_SUMMARY_LOG_PATH)
    parser.add_argument("--detailed-log-path", type=Path, default=DEFAULT_DETAILED_LOG_PATH)
    parser.add_argument("--full-dataset-path", type=Path, default=DEFAULT_FULL_DATASET_PATH)
    parser.add_argument("--filtered-dataset-path", type=Path, default=DEFAULT_FILTERED_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--report-config-path",
        type=Path,
        default=DEFAULT_REPORT_CONFIG_PATH,
    )
    parser.add_argument(
        "--skip-summary-excel",
        action="store_true",
        help="Do not generate tables/summary_excel_file.xlsx.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_report(
        summary_log_path=args.summary_log_path,
        detailed_log_path=args.detailed_log_path,
        full_dataset_path=args.full_dataset_path,
        filtered_dataset_path=args.filtered_dataset_path,
        output_dir=args.output_dir,
        report_config_path=args.report_config_path,
        build_summary_excel=not args.skip_summary_excel,
    )


if __name__ == "__main__":
    main()