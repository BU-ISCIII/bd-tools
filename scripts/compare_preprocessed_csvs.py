from __future__ import annotations

import argparse
import csv
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_CSV = Path(
    "/data/ucct/bi/research/20260330_MEPRAM-RESULTS_CNM_C/ANALYSIS/"
    "00-mepram-data/20260413_df_merged_full_multilabel_grouped.csv"
)
DEFAULT_REPO_CSV = Path("preprocess_test_filtered.csv")
DEFAULT_OUTPUT_DIR = Path("csv_comparison_report")

SCORE_RE = re.compile(r"^=\s*([-+]?\d+(?:\.\d+)?)\s*score$", re.I)


def strip_accents(value: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(char)
    )


def feature_name(value: str) -> str:
    return strip_accents(value).strip().replace(" ", "_")


def canonical_column_name(value: str) -> str:
    normalized = feature_name(value).lower()
    normalized = normalized.replace("-", "_")
    normalized = re.sub(r"[^a-z0-9_]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def normalized_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return ""
    score_match = SCORE_RE.match(text)
    if score_match:
        text = score_match.group(1)
    if text.lower() == "false":
        return "0"
    if text.lower() == "true":
        return "1"
    try:
        if text:
            number = float(text)
            if math.isfinite(number):
                return str(int(number)) if number.is_integer() else f"{number:.12g}"
    except ValueError:
        pass
    return strip_accents(text)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def write_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def explicit_source_to_repo_renames() -> dict[str, str]:
    renames: dict[str, str] = {}

    symptom_renames = {
        "sintoma_dificultad para respirar": "sintoma_respirar",
        "sintoma_dolor abdominal": "sintoma_abdominal",
        "sintoma_dolor articular": "sintoma_articular",
        "sintoma_dolor costal": "sintoma_costal",
        "sintoma_dolor en el ángulo renal": "sintoma_renal",
        "sintoma_lesión de la piel": "sintoma_piel",
        "sintoma_lesión de mucosa": "sintoma_mucosa",
        "sintoma_síndrome de disuria - polaquiuria": "sintoma_polaquiuria",
        "sintoma_tenesmo de ano y/o recto": "sintoma_recto",
    }
    for source, repo in symptom_renames.items():
        renames[source] = repo
        renames[f"{source}_categorico"] = f"{repo}_categorico"

    organisms = [
        "Enterococcus",
        "_Other bacteria",
        "_Fungi",
        "_Virus",
        "Escherichia coli",
        "Klebsiella pneumoniae",
        "Streptococcus pneumoniae",
        "_Enterobacteria",
        "Pseudomonas aeruginosa",
        "Staphylococcus aureus",
    ]
    for organism in organisms:
        organism_feature = feature_name(organism)
        renames[f"{organism}_binary"] = f"infprev_{organism_feature}_binary"
        renames[f"colo_{organism}_binary"] = f"colo_{organism_feature}_binary"

    antibiotic_classes = {
        "Aminoglucósidos": "Aminoglucosidos",
        "AntiTuberculoso": "AntiTuberculoso",
        "Azoles": "Azoles",
        "Carbapenemas": "Carbapenemas",
        "Cefalosporinas 1 gen": "Cefalosporinas 1 gen",
        "Cefalosporinas 2 gen": "Cefalosporinas 2 gen",
        "Cefalosporinas 3 gen": "Cefalosporinas 3 gen",
        "Cefalosporinas 4 gen": "Cefalosporinas 4 gen",
        "Equinocandinas": "Equinocandinas",
        "Fosfomicina": "Fosfomicina",
        "Glicopéptidos": "Glicopéptidos",
        "Lincosamidas": "Lincosamidas",
        "Lipopéptidos": "Lipopeptidos",
        "Macrólidos": "Macrolidos",
        "Metronidazol": "Metronidazol",
        "Monobactámicos": "Monobactamicos",
        "Nitrofurantoína": "Nitrofurantoina",
        "Penicilinas": "Penicilinas",
        "Quinolonas": "Quinolonas",
        "Sulfonamidas": "Sulfonamidas",
        "Tetraciclinas": "Tetraciclinas",
        "Tigeciclina": "Tigeciclina",
    }
    for source_class, repo_class in antibiotic_classes.items():
        renames[source_class] = f"antib_previo_{feature_name(repo_class)}_counts"
        renames[f"{source_class}_binary"] = (
            f"antib_previo_{feature_name(repo_class)}_binary"
        )

    organism_totals = [
        "Pseudomonas aeruginosa",
        "_Other bacteria",
        "Klebsiella pneumoniae",
        "Streptococcus pneumoniae",
        "Staphylococcus aureus",
        "Escherichia coli",
    ]
    for organism in organism_totals:
        renames[f"{organism}_total"] = f"{feature_name(organism)}_total"

    renames["all_cult_org"] = "dominant_all_cult_org"
    renames["colonizacion_total"] = "colonizacion_total_grouped"
    renames["colonizacion_total_cat"] = "colonizacion_total_grouped"
    return renames


def column_key(column: str, explicit_renames: dict[str, str], *, source_side: bool) -> str:
    if source_side and column in explicit_renames:
        return canonical_column_name(explicit_renames[column])
    return canonical_column_name(column)


def pair_columns(
    source_headers: list[str],
    repo_headers: list[str],
    explicit_renames: dict[str, str],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    source_by_key: dict[str, list[str]] = defaultdict(list)
    repo_by_key: dict[str, list[str]] = defaultdict(list)
    for column in source_headers:
        source_by_key[column_key(column, explicit_renames, source_side=True)].append(column)
    for column in repo_headers:
        repo_by_key[column_key(column, explicit_renames, source_side=False)].append(column)

    pairs: list[dict[str, str]] = []
    source_only: list[dict[str, str]] = []
    repo_only: list[dict[str, str]] = []

    for key in sorted(set(source_by_key) | set(repo_by_key)):
        source_columns = source_by_key.get(key, []).copy()
        repo_columns = repo_by_key.get(key, []).copy()

        for source_column in source_columns.copy():
            if source_column in repo_columns:
                pairs.append(
                    {
                        "source_column": source_column,
                        "repo_column": source_column,
                        "normalized_name": key,
                        "match_type": "same name",
                    }
                )
                source_columns.remove(source_column)
                repo_columns.remove(source_column)

        for source_column in source_columns.copy():
            expected_repo_column = explicit_renames.get(source_column)
            if expected_repo_column in repo_columns:
                pairs.append(
                    {
                        "source_column": source_column,
                        "repo_column": expected_repo_column,
                        "normalized_name": key,
                        "match_type": "explicit rename",
                    }
                )
                source_columns.remove(source_column)
                repo_columns.remove(expected_repo_column)

        while source_columns and repo_columns:
            source_column = source_columns.pop(0)
            repo_column = repo_columns.pop(0)
            pairs.append(
                {
                    "source_column": source_column,
                    "repo_column": repo_column,
                    "normalized_name": key,
                    "match_type": "normalized name",
                }
            )

        source_only.extend(
            {
                "source_column": column,
                "normalized_name": key,
                "expected_repo_column": explicit_renames.get(column, ""),
            }
            for column in source_columns
        )
        repo_only.extend(
            {
                "repo_column": column,
                "normalized_name": key,
            }
            for column in repo_columns
        )

    return pairs, source_only, repo_only


def sort_key(value: str) -> tuple[int, str]:
    return (0, f"{int(value):012d}") if value.isdigit() else (1, value)


def compare_values(
    source_rows: list[dict[str, str]],
    repo_rows: list[dict[str, str]],
    pairs: list[dict[str, str]],
    key_column: str,
) -> list[dict[str, Any]]:
    source_ids = [row.get(key_column, "") for row in source_rows]
    repo_ids = [row.get(key_column, "") for row in repo_rows]
    source_counts = {value: source_ids.count(value) for value in set(source_ids)}
    repo_counts = {value: repo_ids.count(value) for value in set(repo_ids)}
    source_by_id = {
        row[key_column]: row
        for row in source_rows
        if row.get(key_column, "") and source_counts[row[key_column]] == 1
    }
    repo_by_id = {
        row[key_column]: row
        for row in repo_rows
        if row.get(key_column, "") and repo_counts[row[key_column]] == 1
    }
    common_ids = sorted(set(source_by_id) & set(repo_by_id), key=sort_key)

    value_differences: list[dict[str, Any]] = []
    for pair in pairs:
        source_column = pair["source_column"]
        repo_column = pair["repo_column"]
        difference_count = 0
        examples: list[str] = []
        for row_id in common_ids:
            source_value = source_by_id[row_id].get(source_column, "")
            repo_value = repo_by_id[row_id].get(repo_column, "")
            if normalized_value(source_value) != normalized_value(repo_value):
                difference_count += 1
                if len(examples) < 2:
                    examples.append(
                        f"{key_column}={row_id}: "
                        f"source={source_value!r}; repo={repo_value!r}"
                    )
        if difference_count:
            value_differences.append(
                {
                    "source_column": source_column,
                    "repo_column": repo_column,
                    "normalized_name": pair["normalized_name"],
                    "match_type": pair["match_type"],
                    "normalized_differences": difference_count,
                    "examples": " | ".join(examples),
                }
            )

    value_differences.sort(
        key=lambda row: int(row["normalized_differences"]),
        reverse=True,
    )
    return value_differences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare source and repo preprocessed CSVs after column-name normalization."
    )
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--repo-csv", type=Path, default=DEFAULT_REPO_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--key-column", default="person_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    explicit_renames = explicit_source_to_repo_renames()
    source_headers, source_rows = read_csv(args.source_csv)
    repo_headers, repo_rows = read_csv(args.repo_csv)
    pairs, source_only, repo_only = pair_columns(
        source_headers,
        repo_headers,
        explicit_renames,
    )
    value_differences = compare_values(
        source_rows,
        repo_rows,
        pairs,
        args.key_column,
    )

    write_csv(
        args.output_dir / "columns_in_repo_not_in_source_after_name_normalization.csv",
        ["repo_column", "normalized_name"],
        repo_only,
    )
    write_csv(
        args.output_dir / "columns_in_source_not_in_repo_after_name_normalization.csv",
        ["source_column", "normalized_name", "expected_repo_column"],
        source_only,
    )
    write_csv(
        args.output_dir / "differences_after_name_normalization.csv",
        [
            "source_column",
            "repo_column",
            "normalized_name",
            "match_type",
            "normalized_differences",
            "examples",
        ],
        value_differences,
    )

    print(f"Wrote reports to {args.output_dir}")
    print(f"Matched column pairs: {len(pairs)}")
    print(f"Repo-only columns after name normalization: {len(repo_only)}")
    print(f"Source-only columns after name normalization: {len(source_only)}")
    print(f"Columns with normalized value differences: {len(value_differences)}")


if __name__ == "__main__":
    main()
