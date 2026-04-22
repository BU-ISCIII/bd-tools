from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_SUMMARY_LOG_PATH = Path("preprocess_test_log_summary.csv")
DEFAULT_DETAILED_LOG_PATH = Path("preprocess_test_log_detailed.json")
DEFAULT_FULL_DATASET_PATH = Path("preprocess_test.csv")
DEFAULT_FILTERED_DATASET_PATH = Path("preprocess_test_filtered.csv")
DEFAULT_OUTPUT_DIR = Path("preprocess_report")

COLORS = {
    "input": "#0072B2",
    "output": "#009E73",
    "created": "#009E73",
    "recoded": "#56B4E9",
    "transformed": "#CC79A7",
    "dropped": "#D55E00",
}


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
    plot_df = df[[label_column, before_column, after_column]].copy()
    plot_df[before_column] = pd.to_numeric(plot_df[before_column], errors="coerce").fillna(0)
    plot_df[after_column] = pd.to_numeric(plot_df[after_column], errors="coerce").fillna(0)
    height = max(4.5, 0.38 * len(plot_df))
    fig, ax = plt.subplots(figsize=(12, height))
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
    ax.set_yticklabels(plot_df[label_column])
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="upper right")
    for index, row in plot_df.iterrows():
        ax.text(row[before_column], index - bar_height / 2, f" {int(row[before_column]):,}", va="center", fontsize=8)
        ax.text(row[after_column], index + bar_height / 2, f" {int(row[after_column]):,}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_stacked_change_chart_png(
    change_counts: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    columns = ["created", "recoded", "transformed", "dropped"]
    plot_df = change_counts[["table_name"] + columns].copy()
    height = max(4.5, 0.38 * len(plot_df))
    fig, ax = plt.subplots(figsize=(12, height))
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
    ax.set_title("Logged Preprocessing Operations by Stage")
    ax.set_xlabel("Number of logged operations")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
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
    width = max(18, 0.12 * len(plot_df))
    fig, ax = plt.subplots(figsize=(width, 7))
    ax.bar(
        range(len(plot_df)),
        plot_df["missing_percent"],
        color=COLORS["input"],
        width=0.85,
    )
    ax.set_title("Missing Values in Filtered Dataset")
    ax.set_ylabel("Missing values (%)")
    ax.set_xlabel("Filtered variables, sorted from most to least missing")
    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df["column"], rotation=90, ha="center", fontsize=6)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_change_counts(detailed_logs: list[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for log in detailed_logs:
        records.append(
            {
                "table_name": log["table_name"],
                "created": len(log.get("created_variables", [])),
                "recoded": len(log.get("recoded_variables", [])),
                "transformed": len(log.get("transformed_variables", [])),
                "dropped": len(log.get("dropped_variables", [])),
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
    chart_paths: dict[str, Path],
) -> None:
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
            f"![Rows before and after preprocessing stage]({chart_paths['rows'].name})",
            "",
            f"![Columns before and after preprocessing stage]({chart_paths['columns'].name})",
            "",
            f"![Logged preprocessing operations by stage]({chart_paths['changes'].name})",
            "",
            f"![Missingness in filtered variables]({chart_paths['missingness'].name})",
            "",
            "## Most Feature-Creating Steps",
            "",
        ]
    )
    for item in top_created:
        lines.append(f"- `{item['table_name']}`: `{item['created']}` created-variable log entries")

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
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_log_path)
    detailed_logs = json.loads(detailed_log_path.read_text(encoding="utf-8"))
    change_counts = summarize_change_counts(detailed_logs)

    full_df = pd.read_csv(full_dataset_path) if full_dataset_path and full_dataset_path.exists() else None
    filtered_df = (
        pd.read_csv(filtered_dataset_path)
        if filtered_dataset_path and filtered_dataset_path.exists()
        else None
    )

    rows_chart = output_dir / "rows_by_stage.png"
    columns_chart = output_dir / "columns_by_stage.png"
    changes_chart = output_dir / "logged_operations_by_stage.png"
    missingness_chart = output_dir / "filtered_missingness.png"

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

    change_counts.to_csv(output_dir / "change_counts_by_stage.csv", index=False)
    write_markdown_report(
        output_path=output_dir / "preprocessing_report.md",
        summary=summary,
        change_counts=change_counts,
        detailed_logs=detailed_logs,
        full_df=full_df,
        filtered_df=filtered_df,
        chart_paths={
            "rows": rows_chart,
            "columns": columns_chart,
            "changes": changes_chart,
            "missingness": missingness_chart,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate clinician-facing preprocessing summary figures and markdown from preprocessing logs."
    )
    parser.add_argument("--summary-log-path", type=Path, default=DEFAULT_SUMMARY_LOG_PATH)
    parser.add_argument("--detailed-log-path", type=Path, default=DEFAULT_DETAILED_LOG_PATH)
    parser.add_argument("--full-dataset-path", type=Path, default=DEFAULT_FULL_DATASET_PATH)
    parser.add_argument("--filtered-dataset-path", type=Path, default=DEFAULT_FILTERED_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_report(
        summary_log_path=args.summary_log_path,
        detailed_log_path=args.detailed_log_path,
        full_dataset_path=args.full_dataset_path,
        filtered_dataset_path=args.filtered_dataset_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
