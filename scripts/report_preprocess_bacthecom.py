from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_DETAILED_LOG_PATH = Path("00-data/preprocessed_db_log_detailed.json")
DEFAULT_FULL_DATASET_PATH = Path("00-data/preprocessed_db.csv")
DEFAULT_OUTPUT_DIR = Path("report")


CHANGE_SECTIONS = {
    "created_variables": "created",
    "recoded_variables": "recoded",
    "transformed_variables": "transformed",
    "dropped_variables": "dropped",
}


def normalize_to_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def load_detailed_logs(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    if not isinstance(data, list):
        raise ValueError(f"Detailed log must be a list: {path}")
    return data


def flatten_variable_changes(detailed_logs: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stage in detailed_logs:
        stage_name = str(stage.get("table_name", ""))
        for section_name, short_name in CHANGE_SECTIONS.items():
            changes = stage.get(section_name, [])
            if not isinstance(changes, list):
                continue
            for idx, change in enumerate(changes, start=1):
                source_vals = normalize_to_list(change.get("source"))
                target_vals = normalize_to_list(change.get("target"))
                rows.append(
                    {
                        "table_name": stage_name,
                        "change_type": short_name,
                        "change_idx": idx,
                        "source": "; ".join(source_vals),
                        "target": "; ".join(target_vals),
                        "how": str(change.get("how", "")),
                        "source_n": len(source_vals),
                        "target_n": len(target_vals),
                    }
                )
    return pd.DataFrame(rows)


def compute_change_counts(detailed_logs: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stage in detailed_logs:
        row = {"table_name": str(stage.get("table_name", ""))}
        for section_name, short_name in CHANGE_SECTIONS.items():
            section = stage.get(section_name, [])
            row[short_name] = len(section) if isinstance(section, list) else 0
        rows.append(row)
    return pd.DataFrame(rows)


def compute_dropped_variables(detailed_logs: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stage in detailed_logs:
        table_name = str(stage.get("table_name", ""))
        dropped = stage.get("dropped_variables", [])
        if not isinstance(dropped, list):
            continue
        for change in dropped:
            source_vals = normalize_to_list(change.get("source"))
            for variable in source_vals:
                rows.append(
                    {
                        "table_name": table_name,
                        "dropped_variable": variable,
                        "how": str(change.get("how", "")),
                    }
                )
    return pd.DataFrame(rows).drop_duplicates()


def compute_mortality_by_center(df: pd.DataFrame, center_column: str, mortality_columns: list[str]) -> pd.DataFrame:
    available_mortality = [col for col in mortality_columns if col in df.columns]
    if center_column not in df.columns:
        raise ValueError(f"Center column not found in dataset: {center_column}")
    if not available_mortality:
        raise ValueError(f"None of mortality columns found: {mortality_columns}")

    grouped = df.groupby(center_column, dropna=False)
    rows: list[dict[str, Any]] = []
    for center, chunk in grouped:
        row: dict[str, Any] = {
            "centro": center if pd.notna(center) else "MISSING_CENTER",
            "n_episodes": len(chunk),
        }
        for col in available_mortality:
            numeric_col = pd.to_numeric(chunk[col], errors="coerce")
            row[f"{col}_rate_pct"] = round(float(numeric_col.mean() * 100), 2)
            row[f"{col}_n_positive"] = int((numeric_col == 1).sum())
            row[f"{col}_n_non_missing"] = int(numeric_col.notna().sum())
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("n_episodes", ascending=False)
    return out


def compute_completeness_by_center(df: pd.DataFrame, center_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if center_column not in df.columns:
        raise ValueError(f"Center column not found in dataset: {center_column}")

    per_center_rows: list[dict[str, Any]] = []
    per_variable_rows: list[dict[str, Any]] = []
    all_columns = [col for col in df.columns if col != center_column]

    for center, chunk in df.groupby(center_column, dropna=False):
        center_name = center if pd.notna(center) else "MISSING_CENTER"
        total_cells = len(chunk) * len(all_columns)
        non_missing_cells = int(chunk[all_columns].notna().sum().sum()) if total_cells > 0 else 0
        completeness_pct = round((non_missing_cells / total_cells * 100), 2) if total_cells > 0 else 0.0
        per_center_rows.append(
            {
                "centro": center_name,
                "n_episodes": len(chunk),
                "n_variables": len(all_columns),
                "non_missing_cells": non_missing_cells,
                "total_cells": total_cells,
                "overall_completeness_pct": completeness_pct,
            }
        )

        non_missing_per_col = chunk[all_columns].notna().mean().mul(100)
        for col, pct in non_missing_per_col.items():
            per_variable_rows.append(
                {
                    "centro": center_name,
                    "variable": col,
                    "non_missing_pct": round(float(pct), 2),
                    "missing_pct": round(float(100 - pct), 2),
                    "n_non_missing": int(chunk[col].notna().sum()),
                    "n_total": len(chunk),
                }
            )

    per_center_df = pd.DataFrame(per_center_rows).sort_values("overall_completeness_pct", ascending=False)
    per_variable_df = pd.DataFrame(per_variable_rows)
    return per_center_df, per_variable_df


def write_markdown_summary(
    output_path: Path,
    *,
    full_df: pd.DataFrame,
    change_counts: pd.DataFrame,
    dropped_variables: pd.DataFrame,
    mortality_by_center: pd.DataFrame,
    completeness_by_center: pd.DataFrame,
) -> None:
    created_total = int(change_counts["created"].sum()) if "created" in change_counts.columns else 0
    recoded_total = int(change_counts["recoded"].sum()) if "recoded" in change_counts.columns else 0
    transformed_total = int(change_counts["transformed"].sum()) if "transformed" in change_counts.columns else 0
    dropped_total = int(dropped_variables["dropped_variable"].nunique()) if not dropped_variables.empty else 0

    with output_path.open("w", encoding="utf-8") as file_handle:
        file_handle.write("# Preprocessing Evaluation Report\n\n")
        file_handle.write("## Dataset\n")
        file_handle.write(f"- Rows: {len(full_df)}\n")
        file_handle.write(f"- Columns: {full_df.shape[1]}\n\n")

        file_handle.write("## Variable Changes\n")
        file_handle.write(f"- Created changes: {created_total}\n")
        file_handle.write(f"- Recoded changes: {recoded_total}\n")
        file_handle.write(f"- Transformed changes: {transformed_total}\n")
        file_handle.write(f"- Unique dropped variables: {dropped_total}\n\n")

        file_handle.write("## Mortality by Center\n")
        file_handle.write(f"- Number of centers: {mortality_by_center['centro'].nunique()}\n")
        file_handle.write("Top 10 centers by episode count:\n\n")
        file_handle.write("```\n")
        file_handle.write(mortality_by_center.head(10).to_string(index=False))
        file_handle.write("\n```\n")
        file_handle.write("\n\n")

        file_handle.write("## Completeness by Center\n")
        file_handle.write("Top 10 centers by overall completeness:\n\n")
        file_handle.write("```\n")
        file_handle.write(completeness_by_center.head(10).to_string(index=False))
        file_handle.write("\n```\n")
        file_handle.write("\n")


def plot_change_counts(change_counts: pd.DataFrame, output_dir: Path) -> None:
    if change_counts.empty:
        return
    plot_df = change_counts.copy()
    cols = ["created", "recoded", "transformed", "dropped"]
    for col in cols:
        if col not in plot_df.columns:
            plot_df[col] = 0
    plot_df["total"] = plot_df[cols].sum(axis=1)
    plot_df = plot_df.sort_values("total", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(4, 0.7 * len(plot_df))))
    left = pd.Series(0, index=plot_df.index, dtype=float)
    colors = {"created": "#009E73", "recoded": "#56B4E9", "transformed": "#CC79A7", "dropped": "#D55E00"}
    for col in cols:
        ax.barh(plot_df["table_name"], plot_df[col], left=left, label=col, color=colors[col])
        left = left + plot_df[col]
    ax.set_title("Variable Changes by Preprocessing Stage")
    ax.set_xlabel("Count")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "plot_variable_change_counts_by_stage.png", dpi=160)
    plt.close(fig)


def plot_change_type_counts(change_long: pd.DataFrame, output_dir: Path) -> None:
    if change_long.empty or "change_type" not in change_long.columns:
        return
    counts = change_long["change_type"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(counts.index.astype(str), counts.values, color=["#009E73", "#56B4E9", "#CC79A7", "#D55E00"])
    ax.set_title("Preprocessing Change Types across All Stages")
    ax.set_xlabel("Change Type")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "plot_change_type_counts.png", dpi=160)
    plt.close(fig)


def plot_variable_missingness(full_df: pd.DataFrame, output_dir: Path, top_n: int = 40) -> None:
    missing_pct = full_df.isna().mean().mul(100).sort_values(ascending=False)
    if missing_pct.empty:
        return
    plot_df = missing_pct.head(top_n).reset_index()
    plot_df.columns = ["variable", "missing_pct"]
    fig, ax = plt.subplots(figsize=(max(10, 0.3 * len(plot_df)), 8))
    ax.barh(plot_df["variable"], plot_df["missing_pct"], color="#0072B2")
    ax.set_title(f"Top {len(plot_df)} Variables by Missingness Percentage")
    ax.set_xlabel("Missingness (%)")
    ax.set_ylabel("Variable")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "plot_top_missing_variables.png", dpi=160)
    plt.close(fig)


def plot_dropped_variables_by_reason(dropped_variables: pd.DataFrame, output_dir: Path) -> None:
    if dropped_variables.empty:
        return
    counts = (
        dropped_variables.assign(how=dropped_variables["how"].fillna("unspecified").astype(str))
        .groupby("how")["dropped_variable"]
        .nunique()
        .sort_values(ascending=False)
    )
    if counts.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts.index.astype(str), counts.values, color="#D55E00")
    ax.set_title("Number of Dropped Variables by Drop Reason")
    ax.set_xlabel("Drop Reason")
    ax.set_ylabel("Variables Dropped")
    ax.set_xticklabels(counts.index.astype(str), rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "plot_dropped_variables_by_reason.png", dpi=160)
    plt.close(fig)


def plot_mortality_by_center(mortality_by_center: pd.DataFrame, output_dir: Path) -> None:
    if mortality_by_center.empty:
        return
    rate_cols = [col for col in mortality_by_center.columns if col.endswith("_rate_pct")]
    if not rate_cols:
        return
    plot_df = mortality_by_center.copy().sort_values("n_episodes", ascending=False).head(20)
    x = range(len(plot_df))
    width = 0.8 / max(1, len(rate_cols))

    fig, ax = plt.subplots(figsize=(14, 6))
    for idx, col in enumerate(rate_cols):
        positions = [val - 0.4 + width / 2 + idx * width for val in x]
        ax.bar(positions, plot_df[col], width=width, label=col.replace("_rate_pct", ""))
    ax.set_title("Mortality Rates by Center (Top 20 by episodes)")
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["centro"].astype(str), rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "plot_mortality_by_center.png", dpi=160)
    plt.close(fig)


def plot_completeness_by_center(completeness_by_center: pd.DataFrame, output_dir: Path) -> None:
    if completeness_by_center.empty:
        return
    plot_df = completeness_by_center.sort_values("overall_completeness_pct", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(plot_df))
    ax.bar(x, plot_df["overall_completeness_pct"], color="#0072B2")
    ax.set_title("Overall Completeness by Center")
    ax.set_ylabel("Completeness (%)")
    ax.set_xlabel("Center")
    ax.set_ylim(0, 100)
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["centro"].astype(str), rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "plot_completeness_by_center.png", dpi=160)
    plt.close(fig)


def plot_completeness_heatmap(completeness_by_center_variable: pd.DataFrame, output_dir: Path) -> None:
    if completeness_by_center_variable.empty:
        return
    pivot = completeness_by_center_variable.pivot(index="variable", columns="centro", values="non_missing_pct")
    if pivot.empty:
        return
    top_variables = pivot.mean(axis=1).sort_values(ascending=False).head(80).index
    top_centers = pivot.mean(axis=0).sort_values(ascending=False).head(30).index
    pivot = pivot.loc[top_variables, top_centers]
    fig, ax = plt.subplots(figsize=(max(12, 0.35 * len(pivot.columns)), max(6, 0.25 * len(pivot))))
    image = ax.imshow(pivot.values, aspect="auto", interpolation="nearest", vmin=0, vmax=100, cmap=sns.color_palette("rocket", as_cmap=True))
    ax.set_title("Non-missing (%) by Variable and Center")
    ax.set_xlabel("Center")
    ax.set_ylabel("Variable")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(v) for v in pivot.index], fontsize=8)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Non-missing %")
    fig.tight_layout()
    fig.savefig(output_dir / "plot_completeness_heatmap.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate preprocessing outputs: variable changes, mortality by center, completeness by center.")
    parser.add_argument("--detailed-log-path", type=Path, default=DEFAULT_DETAILED_LOG_PATH)
    parser.add_argument("--full-dataset-path", type=Path, default=DEFAULT_FULL_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--center-column", default="centro")
    parser.add_argument(
        "--mortality-columns",
        default="mortalidad,mortalidad_14_dias,mortalidad_30_dias",
        help="Comma-separated mortality columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detailed_logs = load_detailed_logs(args.detailed_log_path)
    full_df = pd.read_csv(args.full_dataset_path)
    mortality_columns = [col.strip() for col in args.mortality_columns.split(",") if col.strip()]

    change_long = flatten_variable_changes(detailed_logs)
    change_counts = compute_change_counts(detailed_logs)
    dropped_variables = compute_dropped_variables(detailed_logs)
    mortality_by_center = compute_mortality_by_center(full_df, args.center_column, mortality_columns)
    completeness_by_center, completeness_by_center_variable = compute_completeness_by_center(full_df, args.center_column)

    change_long.to_csv(args.output_dir / "variable_changes_long.csv", index=False)
    change_counts.to_csv(args.output_dir / "variable_change_counts_by_stage.csv", index=False)
    dropped_variables.to_csv(args.output_dir / "dropped_or_filtered_variables.csv", index=False)
    mortality_by_center.to_csv(args.output_dir / "mortality_by_center.csv", index=False)
    completeness_by_center.to_csv(args.output_dir / "completeness_by_center.csv", index=False)
    completeness_by_center_variable.to_csv(args.output_dir / "completeness_by_center_variable.csv", index=False)
    plot_change_counts(change_counts, args.output_dir)
    plot_change_type_counts(change_long, args.output_dir)
    plot_variable_missingness(full_df, args.output_dir)
    plot_dropped_variables_by_reason(dropped_variables, args.output_dir)
    plot_mortality_by_center(mortality_by_center, args.output_dir)
    plot_completeness_by_center(completeness_by_center, args.output_dir)
    plot_completeness_heatmap(completeness_by_center_variable, args.output_dir)
    write_markdown_summary(
        args.output_dir / "preprocessing_evaluation_report.md",
        full_df=full_df,
        change_counts=change_counts,
        dropped_variables=dropped_variables,
        mortality_by_center=mortality_by_center,
        completeness_by_center=completeness_by_center,
    )

    print(f"Report files written in: {args.output_dir}")


if __name__ == "__main__":
    main()
