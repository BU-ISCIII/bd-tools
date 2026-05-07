from __future__ import annotations

import argparse
import ast
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from build_clinician_variable_dictionary import (
    DOMAIN_BY_STAGE,
    build_variable_dictionary,
    infer_source_domain,
)


DEFAULT_SUMMARY_LOG_PATH = Path("preprocess_test_log_summary.csv")
DEFAULT_DETAILED_LOG_PATH = Path("preprocess_test_log_detailed.json")
DEFAULT_FULL_DATASET_PATH = Path("preprocess_test.csv")
DEFAULT_FILTERED_DATASET_PATH = Path("preprocess_test_filtered.csv")
DEFAULT_SQLITE_PATH = Path("db_mepram_sepsis_vf.sqlite3")
DEFAULT_OUTPUT_DIR = Path("report")

COLORS = {
    "input": "#0072B2",
    "output": "#009E73",
    "created": "#009E73",
    "recoded": "#56B4E9",
    "transformed": "#CC79A7",
    "dropped": "#D55E00",
}
VARIABLE_GROUP_LABELS = list(dict.fromkeys(DOMAIN_BY_STAGE.values()))

CHART_FONT_SIZES = {
    "title": 18,
    "axis_label": 15,
    "tick": 12,
    "legend": 12,
    "value": 11,
}
EXCLUDE_FROM_PLOTS = {"_pipeline_run"}


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
        color=COLORS["input"],
    )
    ax.barh(
        [position + bar_height / 2 for position in y_positions],
        plot_df[after_column],
        height=bar_height,
        label="After",
        color=COLORS["output"],
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
            color=COLORS[column],
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


def write_missingness_chart_png(
    df: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    missing_percent = df.isna().mean().mul(100).sort_values(ascending=False)
    plot_df = missing_percent.reset_index()
    plot_df.columns = ["column", "missing_percent"]
    width = max(22, 0.13 * len(plot_df))
    fig, ax = plt.subplots(figsize=(width, 9.5))
    ax.bar(
        range(len(plot_df)),
        plot_df["missing_percent"],
        color=COLORS["input"],
        width=0.85,
    )
    ax.set_title("Missing Values in Filtered Dataset", fontsize=24, pad=18)
    ax.set_ylabel("Missing values (%)", fontsize=20)
    ax.set_xlabel(
        f"Filtered variables (n={len(plot_df):,})",
        fontsize=20,
        labelpad=16,
    )
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, len(plot_df) - 0.5)
    ax.margins(x=0)
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df["column"], rotation=90, ha="center", fontsize=11)
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


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
) -> pd.DataFrame:
    missing_percent = df.isna().mean().mul(100)
    first_stage = first_stage_by_column(df.columns.tolist(), detailed_logs)
    groups: dict[str, list[str]] = defaultdict(list)
    for column in df.columns:
        groups[infer_source_domain(first_stage.get(column))].append(column)

    bands = [
        ("0%", lambda values: values == 0),
        (">0-10%", lambda values: (values > 0) & (values <= 10)),
        ("10-40%", lambda values: (values > 10) & (values <= 40)),
        ("40-80%", lambda values: (values > 40) & (values <= 80)),
        ("80-100%", lambda values: values > 80),
    ]
    rows = []
    ordered_groups = [
        group for group in VARIABLE_GROUP_LABELS
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

    return pd.DataFrame.from_records(rows)


def write_grouped_missingness_chart_png(
    df: pd.DataFrame,
    *,
    detailed_logs: list[dict[str, object]],
    output_path: Path,
) -> pd.DataFrame:
    plot_df = grouped_slide_missingness(df, detailed_logs)
    band_columns = ["0%", ">0-10%", "10-40%", "40-80%", "80-100%"]
    band_colors = {
        "0%": "#009E73",
        ">0-10%": "#56B4E9",
        "10-40%": "#E69F00",
        "40-80%": "#D55E00",
        "80-100%": "#CC79A7",
    }
    fig, ax = plt.subplots(figsize=(16, 9))
    bottom = pd.Series(0, index=plot_df.index, dtype=float)
    x_positions = range(len(plot_df))
    for band in band_columns:
        ax.bar(
            x_positions,
            plot_df[band],
            bottom=bottom,
            color=band_colors[band],
            label=band,
            width=0.78,
        )
        bottom = bottom + plot_df[band]
    ax.set_title("Missingness Bands by Variable Group", fontsize=26, pad=18)
    ax.set_ylabel("Number of variables", fontsize=20)
    ax.set_xlabel(
        f"Filtered variables grouped for slides (n={len(df.columns):,})",
        fontsize=20,
        labelpad=16,
    )
    ax.set_ylim(0, max(plot_df["variables"].max() * 1.12, 1))
    ax.set_xlim(-0.5, len(plot_df) - 0.5)
    ax.margins(x=0)
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(
        [f"{row.group}\n(n={row.variables})" for row in plot_df.itertuples()],
        rotation=45,
        ha="right",
        fontsize=13,
    )
    ax.tick_params(axis="y", labelsize=16)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Missingness band", title_fontsize=13, fontsize=12, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return plot_df


def variable_distribution_by_group(
    columns: list[str],
    detailed_logs: list[dict[str, object]],
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
        infer_source_domain(first_stage_by_column.get(column))
        for column in columns
    )

    records = []
    ordered_groups = [
        group for group in VARIABLE_GROUP_LABELS
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
) -> pd.DataFrame:
    full_distribution = variable_distribution_by_group(full_df.columns.tolist(), detailed_logs)
    filtered_distribution = variable_distribution_by_group(
        filtered_df.columns.tolist(),
        detailed_logs,
    )
    result = full_distribution.rename(columns={"variables": "unfiltered_variables"}).merge(
        filtered_distribution.rename(columns={"variables": "filtered_variables"}),
        on="group",
        how="left",
    )
    result["filtered_variables"] = result["filtered_variables"].fillna(0).astype(int)
    result["removed_by_filter"] = (
        result["unfiltered_variables"] - result["filtered_variables"]
    )
    return result


def write_variable_distribution_chart_png(
    distribution: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    plot_df = distribution.copy()
    x_positions = list(range(len(plot_df)))
    bar_width = 0.38

    fig, ax = plt.subplots(figsize=(15, 8.5))
    ax.bar(
        [position - bar_width / 2 for position in x_positions],
        plot_df["unfiltered_variables"],
        width=bar_width,
        label=f"Unfiltered (n={int(plot_df['unfiltered_variables'].sum()):,})",
        color=COLORS["input"],
    )
    ax.bar(
        [position + bar_width / 2 for position in x_positions],
        plot_df["filtered_variables"],
        width=bar_width,
        label=f"Filtered (n={int(plot_df['filtered_variables'].sum()):,})",
        color=COLORS["output"],
    )
    ax.set_title("Variable Distribution by Group", fontsize=24, pad=18)
    ax.set_ylabel("Number of variables", fontsize=18)
    ax.set_xlabel("Variable group", fontsize=18, labelpad=16)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(plot_df["group"], rotation=35, ha="right", fontsize=13)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=13, loc="upper right")
    max_value = max(
        plot_df["unfiltered_variables"].max(),
        plot_df["filtered_variables"].max(),
        1,
    )
    ax.set_ylim(0, max_value * 1.18)
    label_offset = max_value * 0.015
    for position, row in zip(x_positions, plot_df.itertuples(index=False)):
        ax.text(
            position - bar_width / 2,
            row.unfiltered_variables + label_offset,
            f"{int(row.unfiltered_variables)}",
            ha="center",
            va="bottom",
            fontsize=12,
        )
        ax.text(
            position + bar_width / 2,
            row.filtered_variables + label_offset,
            f"{int(row.filtered_variables)}",
            ha="center",
            va="bottom",
            fontsize=12,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def predictive_target_summary(filtered_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_specs = [
        {
            "model": "Model 1 - Sepsis",
            "target_column": "sepsis",
            "subset": filtered_df,
            "class_labels": {
                0: "Sepsis (-)",
                1: "Sepsis (+)",
                "0": "Sepsis (-)",
                "1": "Sepsis (+)",
            },
            "notes": "No exclusions; binary target over the full cohort.",
        },
        {
            "model": "Model 2 - Etiology",
            "target_column": "resultado_hemo_grouped",
            "subset": filtered_df,
            "class_labels": {
                "NEGATIVE": "Negative blood culture",
                "Bacilo gram-": "Gram-negative bacillus",
                "Coco gram+": "Gram-positive coccus",
            },
            "notes": (
                "Grouped hemoculture target. NEGATIVE includes negative blood "
                "cultures and non-target or unmapped detected organisms."
            ),
        },
        {
            "model": "Model 3 - Resistance",
            "target_column": "resistente_cefalosporina",
            "subset": filtered_df.loc[
                filtered_df["resultado_hemo_grouped"] != "NEGATIVE"
            ],
            "class_labels": {
                "NEGATIVE": "Not resistant",
                "RESIST_CEFALOSPORINAS_3a_4a": "Resistant to cephalosporins 3a/4a",
            },
            "notes": (
                "Restricted to grouped positive blood cultures "
                "(resultado_hemo_grouped != NEGATIVE); grouped as binary "
                "cephalosporin 3a/4a resistance."
            ),
        },
    ]

    summary_records = []
    class_records = []
    for spec in target_specs:
        target_column = spec["target_column"]
        subset = spec["subset"]
        if target_column not in subset.columns:
            raise ValueError(f"Missing target column: {target_column}")
        target = subset[target_column].dropna()
        counts = target.value_counts(dropna=False)
        sample_count = int(target.shape[0])
        majority_count = int(counts.max()) if sample_count else 0
        majority_raw = counts.idxmax() if sample_count else ""
        majority_label = spec["class_labels"].get(majority_raw, str(majority_raw))
        class_count = int(target.nunique(dropna=True))
        majority_percent = round((majority_count / sample_count) * 100, 1) if sample_count else 0

        summary_records.append(
            {
                "model": spec["model"],
                "target_column": target_column,
                "samples": sample_count,
                "classes": class_count,
                "majority_class": majority_label,
                "majority_class_percent": majority_percent,
                "exclusions_or_grouping": spec["notes"],
            }
        )

        for raw_value, count in counts.sort_index().items():
            class_records.append(
                {
                    "model": spec["model"],
                    "target_column": target_column,
                    "class": spec["class_labels"].get(raw_value, str(raw_value)),
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


def raw_phenotype_target_from_sqlite(
    filtered_df: pd.DataFrame,
    sqlite_path: Path | None,
) -> pd.Series | None:
    if sqlite_path is None or not sqlite_path.exists():
        return None

    with sqlite3.connect(sqlite_path) as connection:
        raw = pd.read_sql_query(
            """
            SELECT person_id, fecha_ingreso_urgencias, fenotipo_resistencia
            FROM tbl_hemocultivo_de_urgencias
            """,
            connection,
        )
        code_names = pd.read_sql_query(
            """
            SELECT value, name
            FROM tbl_codes2names
            WHERE variable = 'fenotipo_resistencia'
            """,
            connection,
        )

    phenotype_name_map = {
        str(row.value): str(row.name)
        for row in code_names.itertuples(index=False)
        if pd.notna(row.value) and pd.notna(row.name)
    }

    def decode_codes(values: pd.Series) -> str:
        labels = []
        for value in values.dropna():
            try:
                key = str(int(float(value)))
            except (TypeError, ValueError):
                key = str(value)
            if key in phenotype_name_map:
                labels.append(phenotype_name_map[key])
        labels = sorted(set(labels))
        if not labels:
            return "NEGATIVE"
        return " + ".join(labels)

    grouped = (
        raw.groupby(["person_id", "fecha_ingreso_urgencias"], dropna=False)[
            "fenotipo_resistencia"
        ]
        .apply(decode_codes)
        .reset_index(name="raw_fenotipo_resistencia")
    )
    merged = filtered_df[["person_id", "fecha_ingreso_urgencias"]].merge(
        grouped,
        on=["person_id", "fecha_ingreso_urgencias"],
        how="left",
    )
    return merged["raw_fenotipo_resistencia"].fillna("NEGATIVE")


def target_evolution_summary(
    filtered_df: pd.DataFrame,
    *,
    sqlite_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_columns = {
        "resultado_hemo",
        "resultado_hemo_grouped",
        "fenotipo_resistencia",
        "resistente_cefalosporina",
    }
    missing_columns = sorted(required_columns - set(filtered_df.columns))
    if missing_columns:
        raise ValueError(
            "Missing columns for target evolution summary: "
            + ", ".join(missing_columns)
        )

    summaries: list[dict[str, object]] = []
    class_records: list[dict[str, object]] = []

    microorganism_column = (
        "resultado_hemo_mo"
        if "resultado_hemo_mo" in filtered_df.columns
        else "resultado_hemo"
    )
    target_series = [
        (
            "Etiology",
            "Microorganism target",
            microorganism_column,
            filtered_df[microorganism_column],
            "Hemoculture microorganism target after clinician coinfection resolution.",
        ),
        (
            "Etiology",
            "Clinical organism grouping",
            "resultado_hemo",
            filtered_df["resultado_hemo"],
            "Configured clinical organism grouping from the preprocessing pipeline.",
        ),
        (
            "Etiology",
            "Gram grouped",
            "resultado_hemo_grouped",
            filtered_df["resultado_hemo_grouped"],
            "Pipeline target grouped into negative, Gram-negative bacillus, and Gram-positive coccus.",
        ),
    ]

    for domain, version, column, series, notes in target_series:
        summary, records = series_balance_records(
            domain=domain,
            target_version=version,
            target_column=column,
            values=series,
            notes=notes,
        )
        summaries.append(summary)
        class_records.extend(records)

    if "fenotipo_resistencia_individual" in filtered_df.columns:
        summary, records = multilabel_balance_records(
            domain="Resistance",
            target_version="Individual phenotype labels",
            target_column="fenotipo_resistencia_individual",
            values=filtered_df["fenotipo_resistencia_individual"],
            notes=(
                "Individual resistance phenotype labels before antibiotic-family "
                "grouping; multilabel rows are counted once for each phenotype present."
            ),
        )
        summaries.append(summary)
        class_records.extend(records)
    else:
        raw_phenotype = raw_phenotype_target_from_sqlite(filtered_df, sqlite_path)
        if raw_phenotype is not None:
            summary, records = series_balance_records(
                domain="Resistance",
                target_version="Individual phenotype combinations",
                target_column="raw_fenotipo_resistencia",
                values=raw_phenotype,
                notes="Derived from SQLite hemoculture phenotype codes and decoded with tbl_codes2names.",
            )
            summaries.append(summary)
            class_records.extend(records)

    summary, records = multilabel_balance_records(
        domain="Resistance",
        target_version="Antibiotic family labels",
        target_column="fenotipo_resistencia",
        values=filtered_df["fenotipo_resistencia"],
        notes=(
            "Pipeline target after mapping raw resistance phenotypes to antibiotic "
            "families; multilabel rows are counted once for each family present."
        ),
    )
    summaries.append(summary)
    class_records.extend(records)

    summary, records = series_balance_records(
        domain="Resistance",
        target_version="Cephalosporin yes/no",
        target_column="resistente_cefalosporina",
        values=filtered_df["resistente_cefalosporina"],
        notes="Final binary target: any cephalosporin 3a/4a resistance versus negative.",
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_count = len(filtered_df)
    column_count = len(filtered_df.columns)
    required_columns = {
        "sepsis",
        "resultado_hemo_mo",
        "resultado_hemo",
        "fenotipo_resistencia_individual",
        "fenotipo_resistencia",
        "resistente_cefalosporina",
    }
    missing_columns = sorted(required_columns - set(filtered_df.columns))
    if missing_columns:
        raise ValueError(
            "Missing columns for prediction detail tables: "
            + ", ".join(missing_columns)
        )

    specs = [
        {
            "section": "Sepsis",
            "description": "Binary sepsis prediction target.",
            "target_column": "sepsis",
            "values": filtered_df["sepsis"],
            "class_labels": {
                0: "Sepsis (-)",
                1: "Sepsis (+)",
                "0": "Sepsis (-)",
                "1": "Sepsis (+)",
            },
            "top_n": None,
        },
        {
            "section": "Etiology - microorganisms",
            "description": "Original microorganism target before clinical grouping; shown as top 50 classes plus Other.",
            "target_column": "resultado_hemo_mo",
            "values": filtered_df["resultado_hemo_mo"],
            "class_labels": None,
            "top_n": 50,
        },
        {
            "section": "Etiology - clinical grouping",
            "description": "Clinician-configured microorganism grouping used by the etiology model.",
            "target_column": "resultado_hemo",
            "values": filtered_df["resultado_hemo"],
            "class_labels": None,
            "top_n": None,
        },
        {
            "section": "Resistance - individual phenotypes",
            "description": "Individual resistance phenotype labels before antibiotic-family grouping; multilabel rows are counted once for each phenotype present.",
            "target_column": "fenotipo_resistencia_individual",
            "values": filtered_df["fenotipo_resistencia_individual"],
            "class_labels": None,
            "top_n": None,
            "count_mode": "multilabel",
        },
        {
            "section": "Resistance - antibiotic families",
            "description": "Resistance phenotypes grouped into antibiotic-family labels; multilabel rows are counted once for each family present.",
            "target_column": "fenotipo_resistencia",
            "values": filtered_df["fenotipo_resistencia"],
            "class_labels": None,
            "top_n": None,
            "count_mode": "multilabel",
        },
        {
            "section": "Resistance - cephalosporins",
            "description": "Final binary cephalosporin resistance target.",
            "target_column": "resistente_cefalosporina",
            "values": filtered_df["resistente_cefalosporina"],
            "class_labels": {
                "NEGATIVE": "Not resistant",
                "RESIST_CEFALOSPORINAS_3a_4a": "Resistant to cephalosporins 3a/4a",
            },
            "top_n": None,
        },
    ]

    summary_records = []
    class_records = []
    for spec in specs:
        values = spec["values"].dropna()
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
                "section": spec["section"],
                "description": spec["description"],
                "target_column": spec["target_column"],
                "rows": row_count,
                "columns": column_count,
                "classes": class_count,
            }
        )
        if spec.get("count_mode") == "multilabel":
            class_records.extend(
                multilabel_class_count_records(
                    section=spec["section"],
                    target_column=spec["target_column"],
                    values=values,
                )
            )
        else:
            class_records.extend(
                class_count_records(
                    section=spec["section"],
                    target_column=spec["target_column"],
                    values=values,
                    class_labels=spec["class_labels"],
                    top_n=spec["top_n"],
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
        "# Preprocessing Report",
        "",
        "This report summarizes the preprocessing audit logs. It is intended for clinician review and pipeline QA.",
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
    sqlite_path: Path | None,
    output_dir: Path,
    build_dictionary: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = output_dir / "graphs"
    tables_dir = output_dir / "tables"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_log_path)
    detailed_logs = json.loads(detailed_log_path.read_text(encoding="utf-8"))
    change_counts = summarize_change_counts(detailed_logs)

    full_df = pd.read_csv(full_dataset_path) if full_dataset_path and full_dataset_path.exists() else None
    filtered_df = (
        pd.read_csv(filtered_dataset_path)
        if filtered_dataset_path and filtered_dataset_path.exists()
        else None
    )

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
    )
    write_before_after_chart_png(
        summary,
        label_column="table_name",
        before_column="input_column_count",
        after_column="output_column_count",
        output_path=columns_chart,
        title="Columns Before and After Each Preprocessing Stage",
        x_label="Columns",
    )
    write_stacked_change_chart_png(change_counts, output_path=changes_chart)
    if filtered_df is not None:
        write_missingness_chart_png(filtered_df, output_path=missingness_chart)
        grouped_missingness = write_grouped_missingness_chart_png(
            filtered_df,
            detailed_logs=detailed_logs,
            output_path=grouped_missingness_chart,
        )
        grouped_missingness.to_csv(
            tables_dir / "grouped_missingness_for_slides.csv",
            index=False,
        )
    if full_df is not None and filtered_df is not None:
        variable_distribution = variable_distribution_filtered_vs_unfiltered(
            full_df=full_df,
            filtered_df=filtered_df,
            detailed_logs=detailed_logs,
        )
        write_variable_distribution_chart_png(
            variable_distribution,
            output_path=variable_distribution_chart,
        )
        variable_distribution.to_csv(
            tables_dir / "variable_distribution_filtered_vs_unfiltered.csv",
            index=False,
        )
    target_summary = None
    target_class_counts = None
    evolution_summary = None
    evolution_class_counts = None
    prediction_detail_summary = None
    prediction_detail_class_counts = None
    if filtered_df is not None:
        target_summary, target_class_counts = predictive_target_summary(filtered_df)
        target_summary.to_csv(tables_dir / "predictive_targets_summary.csv", index=False)
        target_class_counts.to_csv(
            tables_dir / "predictive_targets_class_counts.csv",
            index=False,
        )
        evolution_summary, evolution_class_counts = target_evolution_summary(
            filtered_df,
            sqlite_path=sqlite_path,
        )
        evolution_summary.to_csv(tables_dir / "target_evolution_summary.csv", index=False)
        evolution_class_counts.to_csv(
            tables_dir / "target_evolution_class_counts.csv",
            index=False,
        )
        prediction_detail_summary, prediction_detail_class_counts = (
            prediction_detail_tables(filtered_df)
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
    if build_dictionary and full_dataset_path and filtered_dataset_path:
        build_variable_dictionary(
            full_dataset_path=full_dataset_path,
            filtered_dataset_path=filtered_dataset_path,
            detailed_log_path=detailed_log_path,
            output_path=tables_dir / "clinician_variable_dictionary.xlsx",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate clinician-facing preprocessing summary figures and markdown from preprocessing logs."
    )
    parser.add_argument("--summary-log-path", type=Path, default=DEFAULT_SUMMARY_LOG_PATH)
    parser.add_argument("--detailed-log-path", type=Path, default=DEFAULT_DETAILED_LOG_PATH)
    parser.add_argument("--full-dataset-path", type=Path, default=DEFAULT_FULL_DATASET_PATH)
    parser.add_argument("--filtered-dataset-path", type=Path, default=DEFAULT_FILTERED_DATASET_PATH)
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-variable-dictionary",
        action="store_true",
        help="Do not generate tables/clinician_variable_dictionary.xlsx.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_report(
        summary_log_path=args.summary_log_path,
        detailed_log_path=args.detailed_log_path,
        full_dataset_path=args.full_dataset_path,
        filtered_dataset_path=args.filtered_dataset_path,
        sqlite_path=args.sqlite_path,
        output_dir=args.output_dir,
        build_dictionary=not args.skip_variable_dictionary,
    )


if __name__ == "__main__":
    main()
