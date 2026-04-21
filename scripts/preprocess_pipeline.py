from __future__ import annotations

import argparse
import ast
import fnmatch
import importlib.util
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
DEFAULT_DB_PATH = ROOT_DIR / "database" / "db_mepram_sepsis.sqlite3"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "outputs" / "preprocessed_output.csv"
DEFAULT_CONFIG_PATH = ROOT_DIR / "preprocess_config.py"
DEFAULT_DROP_COLUMNS_PATH = PROJECT_ROOT / "data" / "preprocess_columns_to_drop.txt"


# Dataclasses define the structured objects passed through the pipeline.
# They keep preprocessing outputs explicit and consistent:
# - VariableChange: one transformation entry in the audit log
# - TableLog: full preprocessing log for one table
# - PreprocessResult: processed dataframe + its log
# - PipelineArtifacts: final pipeline outputs
@dataclass
class VariableChange:
    kind: str
    source: str | list[str]
    target: str | list[str]
    how: str


@dataclass
class TableLog:
    table_name: str
    input_rows: int
    output_rows: int
    input_columns: list[str]
    output_columns: list[str]
    merge_keys: list[str]
    # Tracking of column loss and explicit drops
    columns_lost: list[str] = field(default_factory=list)
    columns_dropped_count: int = 0
    # default_factory=list gives each TableLog its own list instance.
    created_variables: list[VariableChange] = field(default_factory=list)
    recoded_variables: list[VariableChange] = field(default_factory=list)
    transformed_variables: list[VariableChange] = field(default_factory=list)
    dropped_variables: list[VariableChange] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    validation_checks: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# Standard return type for every preprocessing function.
# This keeps the dataframe and its audit log together.
@dataclass
class PreprocessResult:
    df: pd.DataFrame
    log: TableLog


@dataclass
class PipelineArtifacts:
    tables: dict[str, pd.DataFrame]
    maps: dict[str, dict[Any, Any]]
    logs: list[TableLog] = field(default_factory=list)


def add_change(
    log: TableLog,
    section: str,
    *,
    source: str | list[str],
    target: str | list[str],
    how: str,
) -> None:
    change = VariableChange(kind=section, source=source, target=target, how=how)
    getattr(log, section).append(change)


def finalize_log(
    *,
    table_name: str,
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    merge_keys: Iterable[str],
) -> TableLog:
    input_cols = set(input_df.columns)
    output_cols = set(output_df.columns)
    # Columns present in input but not in output (lost during processing)
    columns_lost = sorted(input_cols - output_cols)
    return TableLog(
        table_name=table_name,
        input_rows=len(input_df),
        output_rows=len(output_df),
        input_columns=input_df.columns.tolist(),
        output_columns=output_df.columns.tolist(),
        merge_keys=list(merge_keys),
        columns_lost=columns_lost,
        columns_dropped_count=0,  # Will be updated by drop_columns_with_log
    )


def export_logs(
    logs: list[TableLog],
    *,
    summary_output_path: Path,
    detailed_output_path: Path,
) -> pd.DataFrame:
    summary_records: list[dict[str, Any]] = []
    detailed_records: list[dict[str, Any]] = []
    for log in logs:
        columns_lost = sorted(set(log.input_columns) - set(log.output_columns))
        columns_dropped = [change.source for change in log.dropped_variables]
        columns_dropped_count = sum(
            len(source) if isinstance(source, list) else 1
            for source in columns_dropped
        )
        columns_dropped_text = "; ".join(
            ", ".join(source) if isinstance(source, list) else str(source)
            for source in columns_dropped
        )
        summary = {
            "table_name": log.table_name,
            "input_rows": log.input_rows,
            "output_rows": log.output_rows,
            "input_column_count": len(log.input_columns),
            "output_column_count": len(log.output_columns),
            "columns_dropped_count": columns_dropped_count,
            "columns_dropped": columns_dropped_text,
            "columns_lost_count": len(columns_lost),
            "columns_lost": "; ".join(columns_lost) if columns_lost else "",
            "merge_keys": ",".join(log.merge_keys),
            "warning_count": len(log.warnings),
            "note_count": len(log.notes),
            "validation_count": len(log.validation_checks),
            "config_name": log.metadata.get("config_name", ""),
            "config_version": log.metadata.get("config_version", ""),
        }
        log_dict = asdict(log)
        detail = {
            "table_name": log_dict["table_name"],
            "input_rows": log_dict["input_rows"],
            "output_rows": log_dict["output_rows"],
            "input_columns": log_dict["input_columns"],
            "output_columns": log_dict["output_columns"],
            "input_column_number": len(log.input_columns),
            "output_column_number": len(log.output_columns),
            "merge_keys": log_dict["merge_keys"],
            "created_variables": log_dict["created_variables"],
            "recoded_variables": log_dict["recoded_variables"],
            "transformed_variables": log_dict["transformed_variables"],
            "dropped_variables": log_dict["dropped_variables"],
            "columns_dropped_count": columns_dropped_count,
            "columns_dropped": columns_dropped,
            "columns_lost_count": len(columns_lost),
            "columns_lost": columns_lost,
            "warnings": log_dict["warnings"],
            "notes": log_dict["notes"],
            "validation_checks": log_dict["validation_checks"],
        }
        summary_records.append(summary)
        detailed_records.append(detail)

    df = pd.DataFrame.from_records(summary_records)
    df.to_csv(summary_output_path, index=False)
    with detailed_output_path.open("w", encoding="utf-8") as fh:
        json.dump(detailed_records, fh, indent=2, ensure_ascii=False)
    return df


def read_drop_columns(drop_columns_path: Path, columns: list[str]) -> list[str]:
    if not drop_columns_path.exists():
        raise FileNotFoundError(f"Drop-columns file not found: {drop_columns_path}")

    selected_columns: list[str] = []
    for raw_line in drop_columns_path.read_text(encoding="utf-8").splitlines():
        entry = raw_line.strip()
        if not entry or entry.startswith("#"):
            continue
        matches = (
            sorted(fnmatch.filter(columns, entry))
            if any(char in entry for char in "*?[")
            else [entry]
        )
        selected_columns.extend(column for column in matches if column in columns)
    return list(dict.fromkeys(selected_columns))


def export_dataset_outputs(
    df: pd.DataFrame,
    output_file_path: Path,
    *,
    drop_columns_path: Path,
) -> None:
    df.to_csv(output_file_path, index=False)
    drop_columns = read_drop_columns(drop_columns_path, df.columns.tolist())
    filtered_output_path = output_file_path.with_name(
        f"{output_file_path.stem}_filtered.csv"
    )
    df.drop(columns=drop_columns, errors="ignore").to_csv(filtered_output_path, index=False)


def load_config_module(config_file_path: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("preprocess_runtime_config", config_file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config module from {config_file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "as_dict"):
        raise AttributeError(f"Config module {config_file_path} must define as_dict()")
    config = module.as_dict()
    config["CONFIG_PATH"] = str(config_file_path)
    return config


def load_tables(db_file_path: Path) -> dict[str, pd.DataFrame]:
    with sqlite3.connect(db_file_path) as conn:
        table_names = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'tbl_%'",
            conn,
        )["name"].tolist()
        return {name: pd.read_sql_query(f"SELECT * FROM {name}", conn) for name in table_names}


def extract_numeric(series: pd.Series) -> pd.Series:
    extracted = series.astype(str).str.extract(r"(\d+\.\d+|\d+)")[0]
    return pd.to_numeric(extracted, errors="coerce")


def ensure_columns_present(df: pd.DataFrame, required_columns: list[str], table_name: str) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def duplicate_key_rows(df: pd.DataFrame, keys: list[str]) -> int:
    if not keys:
        return 0
    return int(df.duplicated(subset=keys, keep=False).sum())


def duplicate_key_groups(df: pd.DataFrame, keys: list[str]) -> int:
    if not keys:
        return 0
    duplicate_mask = df.duplicated(subset=keys, keep=False)
    if not duplicate_mask.any():
        return 0
    return int(df.loc[duplicate_mask, keys].drop_duplicates().shape[0])


def admission_stats(df: pd.DataFrame) -> dict[str, int | None]:
    stats: dict[str, int | None] = {
        "n_rows": int(len(df)),
        "n_unique_person_id": None,
        "n_unique_person_fecha": None,
        "n_person_id_with_multiple_fechas": None,
    }
    if "person_id" in df.columns:
        stats["n_unique_person_id"] = int(df["person_id"].nunique(dropna=True))
    if {"person_id", "fecha_ingreso_urgencias"}.issubset(df.columns):
        person_fecha = df[["person_id", "fecha_ingreso_urgencias"]].drop_duplicates()
        stats["n_unique_person_fecha"] = int(len(person_fecha))
        multiple_fechas = person_fecha.groupby("person_id")["fecha_ingreso_urgencias"].nunique()
        stats["n_person_id_with_multiple_fechas"] = int((multiple_fechas > 1).sum())
    return stats


def format_admission_stats(name: str, stats: dict[str, int | None]) -> str:
    return (
        f"{name}: rows={stats['n_rows']}, "
        f"unique_person_id={stats['n_unique_person_id']}, "
        f"unique_person_fecha={stats['n_unique_person_fecha']}, "
        f"person_id_with_multiple_fechas={stats['n_person_id_with_multiple_fechas']}"
    )


def format_duplicate_key_stats(
    name: str,
    duplicate_rows: int,
    duplicate_groups: int,
    keys: list[str],
) -> str:
    return (
        f"{name}: duplicated_rows_on_{keys}={duplicate_rows}, "
        f"unique_duplicated_key_groups_on_{keys}={duplicate_groups}"
    )


def overlapping_non_key_columns(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    keys: list[str],
) -> list[str]:
    key_set = set(keys)
    left_only = set(left_df.columns) - key_set
    right_only = set(right_df.columns) - key_set
    return sorted(left_only & right_only)


def validate_result(
    result: PreprocessResult,
    *,
    required_columns: list[str],
    allow_empty: bool = False,
) -> PreprocessResult:
    ensure_columns_present(result.df, required_columns, result.log.table_name)
    result.log.validation_checks.append(
        f"required_columns_present:{','.join(required_columns)}"
    )
    if not allow_empty and result.df.empty:
        raise ValueError(f"{result.log.table_name} produced an empty dataframe")
    result.log.validation_checks.append(f"row_count:{len(result.df)}")
    duplicate_rows = duplicate_key_rows(result.df, result.log.merge_keys)
    duplicate_groups = duplicate_key_groups(result.df, result.log.merge_keys)
    result.log.validation_checks.append(
        f"duplicate_merge_key_rows:{duplicate_rows}"
    )
    result.log.validation_checks.append(
        f"duplicate_merge_key_groups:{duplicate_groups}"
    )
    if duplicate_rows > 0:
        result.log.warnings.append(
            f"{duplicate_rows} rows across {duplicate_groups} duplicated key groups for merge keys {result.log.merge_keys}"
        )
    return result


def attach_run_metadata(log: TableLog, config: dict[str, Any]) -> None:
    log.metadata["config_name"] = config.get("CONFIG_NAME", "")
    log.metadata["config_version"] = config.get("CONFIG_VERSION", "")
    log.metadata["config_path"] = config.get("CONFIG_PATH", "")


def drop_columns_with_log(
    df: pd.DataFrame,
    columns: list[str],
    *,
    log: TableLog,
    reason: str,
    target_description: str | list[str] | None = None,
) -> pd.DataFrame:
    existing_columns = [col for col in columns if col in df.columns]
    if not existing_columns:
        return df
    add_change(
        log,
        "dropped_variables",
        source=existing_columns,
        target=target_description or existing_columns,
        how=reason,
    )
    log.columns_dropped_count += len(existing_columns)
    return df.drop(columns=existing_columns)


def feature_name(value: Any) -> str:
    return str(value).strip().replace(" ", "_")


def as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if pd.isna(value):
        return tuple()
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("(") or stripped.startswith("["):
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                return (value,)
            return as_tuple(parsed)
        return (value,)
    return (value,)


def strip_resistance_prefix(value: str) -> str:
    return value.split(" resistente a ")[-1]


def clean_outliers_iqr(
    df: pd.DataFrame,
    variables: list[str],
    *,
    iqr_multiplier: float = 3.0,
    replace_with: float | None = np.nan,
) -> pd.DataFrame:
    cleaned = df.copy()
    for variable in variables:
        q1 = cleaned[variable].quantile(0.25)
        q3 = cleaned[variable].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        mask = cleaned[variable].between(lower, upper) | cleaned[variable].isna()
        cleaned.loc[~mask, variable] = replace_with
    return cleaned


def clean_outliers_iqr_and_recode_quartiles(
    df: pd.DataFrame,
    variables: list[str],
    *,
    iqr_multiplier: float = 3.0,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    cleaned = df.copy()
    recode_info: dict[str, dict[str, Any]] = {}
    for variable in variables:
        if variable not in cleaned.columns:
            continue

        q1 = cleaned[variable].quantile(0.25)
        q3 = cleaned[variable].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        outliers_mask = (cleaned[variable] < lower) | (cleaned[variable] > upper)
        outliers_count = int(outliers_mask.sum())
        cleaned.loc[outliers_mask, variable] = np.nan

        new_column = f"{variable}_recoded"
        if cleaned[variable].dropna().empty:
            cleaned[new_column] = pd.Series(pd.NA, index=cleaned.index, dtype="Int64")
            bins: list[float] = []
        else:
            recoded, bins_array = pd.qcut(
                cleaned[variable],
                q=4,
                labels=False,
                duplicates="drop",
                retbins=True,
            )
            cleaned[new_column] = recoded.astype("Int64")
            bins = [float(edge) for edge in bins_array]

        recode_info[variable] = {
            "new_column": new_column,
            "q1": float(q1) if pd.notna(q1) else None,
            "q3": float(q3) if pd.notna(q3) else None,
            "iqr": float(iqr) if pd.notna(iqr) else None,
            "lower_limit": float(lower) if pd.notna(lower) else None,
            "upper_limit": float(upper) if pd.notna(upper) else None,
            "outliers_replaced": outliers_count,
            "quartile_bins": bins,
        }
    return cleaned, recode_info


def format_quartile_recode_info(info: dict[str, Any]) -> str:
    bins = info["quartile_bins"]
    if len(bins) < 2:
        categories = "no categories created because all values are missing after IQR cleaning"
    else:
        categories = "; ".join(
            f"{index}: {bins[index]} to {bins[index + 1]}"
            for index in range(len(bins) - 1)
        )
    return (
        "create ordinal quartile category after IQR outlier removal; "
        f"IQR limits lower={info['lower_limit']}, upper={info['upper_limit']}; "
        f"outliers replaced={info['outliers_replaced']}; "
        f"pd.qcut(q=4, labels 0-3, duplicates='drop'); categories: {categories}"
    )


def build_reference_maps(
    tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> dict[str, dict[Any, Any]]:
    codes = tables["tbl_codes2names"].copy()
    microorganisms = tables["tbl_microorganismos"].copy()
    microorganisms["label"] = microorganisms["label"].map(config["MICROORGANISM_LABEL_MAP"])
    microorganisms = (
        microorganisms.sort_values(by=["snomed_code", "label"])
        .drop_duplicates(subset="snomed_code", keep="first")
    )

    foco_map = (
        codes[codes["variable"].astype(str).str.contains("foco", na=False)][["value", "name"]]
        .assign(name=lambda df: df["name"].str.replace("foco | ", "", regex=False))
        .set_index("value")["name"]
        .to_dict()
    )
    phenomap = (
        codes[codes["variable"] == "fenotipo_resistencia"][["value", "name"]]
        .set_index("value")["name"]
        .to_dict()
    )
    symptom_map_df = (
        codes[codes["variable"] == "sintoma"][["value", "name"]]
        .assign(name=lambda df: df["name"].str.split(" | ").str[-1])
    )
    symptom_map = {str(float(row["value"])): row["name"] for _, row in symptom_map_df.iterrows()}
    antibiotic_code_name_map = (
        codes[codes["variable"] == "antimicrobiano_previo"][["value", "name"]]
        .assign(
            name=lambda df: (
                df["name"].str.split(" | ", regex=False).str[-1].str.split("; ", regex=False).str[0]
            )
        )
        .set_index("value")["name"]
        .to_dict()
    )
    antibiotic_name_family_map = {
        drug_name: family
        for _, family, drug_name in config["ANTIMICROBIAL_GROUPS"]
        if drug_name is not None
    }

    return {
        "organism_codes_map": dict(zip(microorganisms["snomed_code"], microorganisms["label"])),
        "foco_map": foco_map,
        "phenomap": phenomap,
        "symptom_map": symptom_map,
        "antibiotic_code_name_map": antibiotic_code_name_map,
        "antibiotic_name_family_map": antibiotic_name_family_map,
        "hemoculture_coinfection_resolution_map": config[
            "HEMOCULTIVO_COINFECTION_RESOLUTION_GROUPS"
        ],
    }


def preprocess_tbl_paciente(
    tables: dict[str, pd.DataFrame],
    maps: dict[str, dict[Any, Any]],
    config: dict[str, Any],
) -> PreprocessResult:
    source = tables["tbl_paciente"].copy()
    previous_infections = tables["tbl_infecciones_previas"].copy()
    centers = tables["tbl_personid2center"].copy()

    df = source.merge(centers, how="left")
    log = finalize_log(
        table_name="tbl_paciente",
        input_df=source,
        output_df=df,
        merge_keys=["person_id", "fecha_ingreso_urgencias"],
    )

    df["inf_previa_sino"] = np.where(
        df["person_id"].isin(previous_infections["person_id"]),
        1,
        0,
    )
    add_change(
        log,
        "created_variables",
        source="person_id",
        target="inf_previa_sino",
        how="1 if person_id appears in tbl_infecciones_previas, else 0",
    )

    bmr_previa = previous_infections.groupby("person_id")["bmr_infec_previa"].apply(
        lambda values: int((values > 0).any())
    )
    df = df.merge(bmr_previa.rename("bmr_infec_previa"), on="person_id", how="left")
    df["bmr_infec_previa"] = df["bmr_infec_previa"].fillna(0).astype(int)
    add_change(
        log,
        "created_variables",
        source="bmr_infec_previa",
        target="bmr_infec_previa",
        how="patient-level flag built from any previous infection with bmr_infec_previa > 0",
    )

    log.output_rows = len(df)
    log.output_columns = df.columns.tolist()
    result = PreprocessResult(df=df, log=log)
    attach_run_metadata(result.log, config)
    return validate_result(result, required_columns=["person_id", "fecha_ingreso_urgencias"])


def preprocess_tbl_comorbilidad(
    tables: dict[str, pd.DataFrame],
    maps: dict[str, dict[Any, Any]],
    config: dict[str, Any],
) -> PreprocessResult:
    source = tables["tbl_comorbilidad"].copy()
    df = pd.get_dummies(source, columns=["tipo_cancer", "tipo_hepatopatia"])
    cancer_dummy_columns = [
        column for column in df.columns if column.startswith("tipo_cancer_")
    ]
    hepatopathy_dummy_columns = [
        column for column in df.columns if column.startswith("tipo_hepatopatia_")
    ]
    df["canceres_si_no"] = (
        df[cancer_dummy_columns].sum(axis=1).gt(0).astype(int)
        if cancer_dummy_columns
        else 0
    )
    df["hepatopatias_si_no"] = (
        df[hepatopathy_dummy_columns].sum(axis=1).gt(0).astype(int)
        if hepatopathy_dummy_columns
        else 0
    )
    log = finalize_log(
        table_name="tbl_comorbilidad",
        input_df=source,
        output_df=df,
        merge_keys=["person_id", "fecha_ingreso_urgencias"],
    )

    add_change(
        log,
        "transformed_variables",
        source=["tipo_cancer", "tipo_hepatopatia"],
        target=[col for col in df.columns if "cancer" in col or "hepatopatia" in col],
        how="one-hot encode categorical comorbidity variables",
    )
    add_change(
        log,
        "created_variables",
        source=cancer_dummy_columns,
        target="canceres_si_no",
        how="collapsed cancer dummy columns to binary flag: 1 if any tipo_cancer_* column is positive, else 0",
    )
    add_change(
        log,
        "created_variables",
        source=hepatopathy_dummy_columns,
        target="hepatopatias_si_no",
        how="collapsed hepatopathy dummy columns to binary flag: 1 if any tipo_hepatopatia_* column is positive, else 0",
    )
    add_change(
        log,
        "dropped_variables",
        source=["tipo_cancer", "tipo_hepatopatia"],
        target=["tipo_cancer_*", "tipo_hepatopatia_*"],
        how="original categoricals replaced by one-hot encoded columns",
    )
    result = PreprocessResult(df=df, log=log)
    attach_run_metadata(result.log, config)
    return validate_result(result, required_columns=["person_id", "fecha_ingreso_urgencias"])


def preprocess_tbl_factores_riesgo_bmr(
    tables: dict[str, pd.DataFrame],
    maps: dict[str, dict[Any, Any]],
    config: dict[str, Any],
) -> PreprocessResult:
    source = tables["tbl_factores_riesgo_bmr"].copy()
    log = finalize_log(
        table_name="tbl_factores_riesgo_bmr",
        input_df=source,
        output_df=source,
        merge_keys=["person_id", "fecha_ingreso_urgencias"],
    )
    log.notes.append("Pass-through table in current notebook. Add table-local feature engineering here if needed.")
    result = PreprocessResult(df=source, log=log)
    attach_run_metadata(result.log, config)
    return validate_result(result, required_columns=["person_id", "fecha_ingreso_urgencias"])


def preprocess_tbl_sintomas(
    tables: dict[str, pd.DataFrame],
    maps: dict[str, dict[Any, Any]],
    config: dict[str, Any],
) -> PreprocessResult:
    source = tables["tbl_sintomas"].copy()
    log = finalize_log(
        table_name="tbl_sintomas",
        input_df=source,
        output_df=source,
        merge_keys=["person_id", "fecha_ingreso_urgencias"],
    )

    df = source.copy()
    df["sintoma"] = df["sintoma"].astype(str).map(maps["symptom_map"])
    df["sintoma"] = "sintoma_" + df["sintoma"]
    add_change(
        log,
        "recoded_variables",
        source="sintoma",
        target="sintoma",
        how="map coded symptom values to readable names using tbl_codes2names",
    )

    pivoted = df.pivot_table(
        index=["person_id", "fecha_ingreso_urgencias"],
        columns="sintoma",
        values="duracion_sintoma",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    
    add_change(
        log,
        "dropped_variables",
        source=["sintoma", "duracion_sintoma"],
        target=["sintoma_binary", "sintoma_categorico"],
        how="original categoricals replaced by one-hot encoded columns",
    )

    symptom_columns = [col for col in pivoted.columns if col not in ["person_id", "fecha_ingreso_urgencias"]]
    presence = pivoted.copy()
    presence[symptom_columns] = (presence[symptom_columns] > 0).astype(int)

    duration = pivoted.copy()
    duration[symptom_columns] = duration[symptom_columns].apply(
        lambda column: column.map(
            lambda value: 0 if value == 0 else (1 if value <= config["SINTOMA_SHORT_DURATION_MAX_DAYS"] else 2)
        )
    )
    duration = duration.rename(columns={col: f"{col}_categorico" for col in symptom_columns})
    add_change(
        log,
        "created_variables",
        source="duracion_sintoma",
        target=[col for col in presence.columns if col.startswith("sintoma_")] + [col for col in duration.columns if col.endswith("_categorico")],
        how=f"pivot symptoms to wide columns; binary presence is 1 if summed duration > 0, and categorical duration is 0 if absent, 1 if <= {config['SINTOMA_SHORT_DURATION_MAX_DAYS']} days, 2 if > {config['SINTOMA_SHORT_DURATION_MAX_DAYS']} days",
    )

    result = presence.merge(duration, on=["person_id", "fecha_ingreso_urgencias"], how="left")
    log.output_rows = len(result)
    log.output_columns = result.columns.tolist()
    result_obj = PreprocessResult(df=result, log=log)
    attach_run_metadata(result_obj.log, config)
    return validate_result(result_obj, required_columns=["person_id", "fecha_ingreso_urgencias"])


def preprocess_tbl_signos(
    tables: dict[str, pd.DataFrame],
    maps: dict[str, dict[Any, Any]],
    config: dict[str, Any],
) -> PreprocessResult:
    source = tables["tbl_signos"].copy()
    df = source.copy()
    log = finalize_log(
        table_name="tbl_signos",
        input_df=source,
        output_df=source,
        merge_keys=["person_id", "fecha_ingreso_urgencias"],
    )

    numeric_columns = [col for col in config["SIGNOS_NUMERIC_COLUMNS"] if col in df.columns]
    for column in numeric_columns:
        df[column] = extract_numeric(df[column])
    add_change(
        log,
        "transformed_variables",
        source=list(numeric_columns),
        target=list(numeric_columns),
        how="extract numeric values from mixed text fields",
    )

    df["hipotermia_hipertermia"] = np.where(
        df["temperatura"] >= config["SIGNOS_THRESHOLDS"]["temperatura_fever_min"],
        2,
        np.where(df["temperatura"] >= config["SIGNOS_THRESHOLDS"]["temperatura_normal_min"], 0, 1),
    )
    df["hipotermia_hipertermia"] = df["hipotermia_hipertermia"].where(df["temperatura"].notna())
    add_change(
        log,
        "recoded_variables",
        source="temperatura",
        target="hipotermia_hipertermia",
        how=(
            "categorical temperature flag created after numeric cleaning: "
            f"2 if temperatura >= {config['SIGNOS_THRESHOLDS']['temperatura_fever_min']} (hyperthermia), "
            f"0 if {config['SIGNOS_THRESHOLDS']['temperatura_normal_min']} <= temperatura < {config['SIGNOS_THRESHOLDS']['temperatura_fever_min']} (normal), "
            f"1 if temperatura < {config['SIGNOS_THRESHOLDS']['temperatura_normal_min']} (hypothermia), "
            "and NaN if temperatura is missing"
        ),
    )

    df["hipotension"] = np.where(
        df["tension_arterial"].isna(),
        df["hipotension"],
        np.where(df["tension_arterial"] <= config["SIGNOS_THRESHOLDS"]["tension_arterial_hypotension_max"], 1, 0),
    )
    add_change(
        log,
        "recoded_variables",
        source="tension_arterial",
        target="hipotension",
        how=(
            "binary hypotension flag recomputed after numeric cleaning: "
            f"1 if tension_arterial <= {config['SIGNOS_THRESHOLDS']['tension_arterial_hypotension_max']}, "
            "0 otherwise, and preserve original value if tension_arterial is missing"
        ),
    )

    df["taquipnea"] = np.where(
        df["frec_respiratoria"].isna(),
        df["taquipnea"],
        np.where(df["frec_respiratoria"] > config["SIGNOS_THRESHOLDS"]["frec_respiratoria_taquipnea_min"], 1, 0),
    )
    add_change(
        log,
        "recoded_variables",
        source="frec_respiratoria",
        target="taquipnea",
        how=(
            "binary tachypnea flag recomputed after numeric cleaning: "
            f"1 if frec_respiratoria > {config['SIGNOS_THRESHOLDS']['frec_respiratoria_taquipnea_min']}, "
            "0 otherwise, and preserve original value if frec_respiratoria is missing"
        ),
    )

    df["taquicardia"] = np.where(
        df["frec_cardiaca"].isna(),
        df["taquicardia"],
        np.where(df["frec_cardiaca"] > config["SIGNOS_THRESHOLDS"]["frec_cardiaca_taquicardia_min"], 1, 0),
    )
    add_change(
        log,
        "recoded_variables",
        source="frec_cardiaca",
        target="taquicardia",
        how=(
            "binary tachycardia flag recomputed after numeric cleaning: "
            f"1 if frec_cardiaca > {config['SIGNOS_THRESHOLDS']['frec_cardiaca_taquicardia_min']}, "
            "0 otherwise, and preserve original value if frec_cardiaca is missing"
        ),
    )

    df["hipoxemia"] = np.where(
        df["saturacion_o2"].isna(),
        df["hipoxemia"],
        np.where(df["saturacion_o2"] > config["SIGNOS_THRESHOLDS"]["saturacion_o2_hipoxemia_max"], 0, 1),
    )
    add_change(
        log,
        "recoded_variables",
        source="saturacion_o2",
        target="hipoxemia",
        how=(
            "binary hypoxemia flag recomputed after numeric cleaning: "
            f"1 if saturacion_o2 <= {config['SIGNOS_THRESHOLDS']['saturacion_o2_hipoxemia_max']}, "
            "0 otherwise, and preserve original value if saturacion_o2 is missing"
        ),
    )

    iqr_columns = [col for col in config["SIGNOS_OUTLIER_COLUMNS"] if col in df.columns]
    quartile_columns = [
        col
        for col in config.get("SIGNOS_QUARTILE_RECODE_COLUMNS", config["SIGNOS_OUTLIER_COLUMNS"])
        if col in df.columns
    ]
    df, quartile_info = clean_outliers_iqr_and_recode_quartiles(
        df,
        quartile_columns,
        iqr_multiplier=config["IQR_DEFAULT_MULTIPLIER"],
    )
    iqr_only_columns = [col for col in iqr_columns if col not in quartile_columns]
    if iqr_only_columns:
        df = clean_outliers_iqr(
            df,
            iqr_only_columns,
            iqr_multiplier=config["IQR_DEFAULT_MULTIPLIER"],
        )
    add_change(
        log,
        "transformed_variables",
        source=iqr_columns,
        target=iqr_columns,
        how=f"replace IQR outliers with NaN using multiplier {config['IQR_DEFAULT_MULTIPLIER']}",
    )
    for variable, info in quartile_info.items():
        add_change(
            log,
            "created_variables",
            source=variable,
            target=info["new_column"],
            how=format_quartile_recode_info(info),
        )
    log.output_rows = len(df)
    log.output_columns = df.columns.tolist()
    result = PreprocessResult(df=df, log=log)
    attach_run_metadata(result.log, config)
    return validate_result(result, required_columns=["person_id", "fecha_ingreso_urgencias"])


def preprocess_tbl_sepsis(
    tables: dict[str, pd.DataFrame],
    maps: dict[str, dict[Any, Any]],
    config: dict[str, Any],
) -> PreprocessResult:
    source = tables["tbl_sepsis"].copy()
    df = source.copy()
    log = finalize_log(
        table_name="tbl_sepsis",
        input_df=source,
        output_df=source,
        merge_keys=["person_id", "fecha_ingreso_urgencias"],
    )

    df["lactato_serico"] = np.where(df["lactato_serico"] == "<= 2 millimole per liter", 0, 1)
    add_change(
        log,
        "recoded_variables",
        source="lactato_serico",
        target="lactato_serico",
        how="recode '<= 2 millimole per liter' to 0 and all other non-null values to 1",
    )

    numeric_columns = [col for col in config["SEPSIS_NUMERIC_COLUMNS"] if col in df.columns and col != "lactato_serico"]
    for column in numeric_columns:
        df[column] = extract_numeric(df[column])

    if "foco" in df.columns:
        df["foco"] = pd.to_numeric(df["foco"], errors="coerce").map(maps["foco_map"])
        add_change(
            log,
            "recoded_variables",
            source="foco",
            target="foco",
            how="map foco codes to readable labels using foco_map",
        )

    sepsis_outlier_columns = [col for col in config["SEPSIS_OUTLIER_COLUMNS"] if col in df.columns]
    sepsis_quartile_columns = [
        col
        for col in config.get("SEPSIS_QUARTILE_RECODE_COLUMNS", config["SEPSIS_OUTLIER_COLUMNS"])
        if col in df.columns
    ]
    df, quartile_info = clean_outliers_iqr_and_recode_quartiles(
        df,
        sepsis_quartile_columns,
        iqr_multiplier=config["IQR_DEFAULT_MULTIPLIER"],
    )
    iqr_only_columns = [col for col in sepsis_outlier_columns if col not in sepsis_quartile_columns]
    if iqr_only_columns:
        df = clean_outliers_iqr(
            df,
            iqr_only_columns,
            iqr_multiplier=config["IQR_DEFAULT_MULTIPLIER"],
        )
    add_change(
        log,
        "transformed_variables",
        source=sepsis_outlier_columns,
        target=sepsis_outlier_columns,
        how=f"replace IQR outliers with NaN using multiplier {config['IQR_DEFAULT_MULTIPLIER']}",
    )
    for variable, info in quartile_info.items():
        add_change(
            log,
            "created_variables",
            source=variable,
            target=info["new_column"],
            how=format_quartile_recode_info(info),
        )
    log.output_rows = len(df)
    log.output_columns = df.columns.tolist()
    result = PreprocessResult(df=df, log=log)
    attach_run_metadata(result.log, config)
    return validate_result(result, required_columns=["person_id", "fecha_ingreso_urgencias"])


def preprocess_tbl_infecciones_previas(
    tables: dict[str, pd.DataFrame],
    maps: dict[str, dict[Any, Any]],
    config: dict[str, Any],
) -> PreprocessResult:
    source = tables["tbl_infecciones_previas"].copy()
    df = source.copy()
    log = finalize_log(
        table_name="tbl_infecciones_previas",
        input_df=source,
        output_df=source,
        merge_keys=["person_id", "fecha_ingreso_urgencias"],
    )
    df["grupo_microorganismo"] = df["microorganism_infec_prev"].astype(str).map(maps["organism_codes_map"])
    add_change(
        log,
        "recoded_variables",
        source="microorganism_infec_prev",
        target="grupo_microorganismo",
        how="map infection organism codes to grouped organism labels",
    )

    for bad_value, good_value in config["FECHA_INFECCION_CORRECTIONS"].items():
        df.loc[df["fecha_infeccion"] == bad_value, "fecha_infeccion"] = good_value

    organisms = df["grupo_microorganismo"].dropna().unique().tolist()
    pivot_source = drop_columns_with_log(
        df,
        ["microorganism_infec_prev", "bmr_infec_previa", "feno_resist_infec_prev", "sindrome_infeccioso", "fecha_infeccion"],
        log=log,
        reason="replace raw detailed previous-infection columns with grouped history features",
        target_description=["grupo_microorganismo", "*_binary", "num_inf_previas", "tiempo_ultima"],
    ).copy()
    pivot_source["dummy"] = 1
    pivoted = pivot_source.pivot_table(
        index=["person_id", "fecha_ingreso_urgencias"],
        columns="grupo_microorganismo",
        values="dummy",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    for organism in organisms:
        pivoted[f"infprev_{feature_name(organism)}_binary"] = np.where(pivoted[organism] >= 1, 1, 0)
        pivoted = pivoted.drop(columns=organism)

    infection_broad_group_columns: list[str] = []
    for broad_group in sorted(set(config["MICROORGANISM_BROAD_GROUP_MAP"].values())):
        target_column = f"Inf_{feature_name(broad_group)}"
        source_columns = [
            f"infprev_{feature_name(organism)}_binary"
            for organism, group in config["MICROORGANISM_BROAD_GROUP_MAP"].items()
            if group == broad_group
        ]
        existing_source_columns = [
            column for column in source_columns if column in pivoted.columns
        ]
        pivoted[target_column] = (
            pivoted[existing_source_columns].fillna(0).sum(axis=1)
            if existing_source_columns
            else 0
        )
        infection_broad_group_columns.append(target_column)

    visits = df[["person_id", "fecha_ingreso_urgencias", "fecha_infeccion"]].copy()
    visits["fecha_infeccion"] = pd.to_datetime(visits["fecha_infeccion"], errors="coerce")
    visits["fecha_ingreso_urgencias"] = pd.to_datetime(visits["fecha_ingreso_urgencias"], errors="coerce")
    num_visitas = visits.groupby("person_id")["fecha_infeccion"].nunique().reset_index(name="num_inf_previas")
    ultima = visits.groupby("person_id")["fecha_infeccion"].max().reset_index(name="ultima_fecha")
    ingreso = visits[["person_id", "fecha_ingreso_urgencias"]].drop_duplicates()
    ultima = ultima.merge(ingreso, on="person_id", how="left")
    ultima["tiempo_ultima"] = (ultima["fecha_ingreso_urgencias"] - ultima["ultima_fecha"]).dt.days

    result = pivoted.merge(num_visitas, on="person_id", how="left")
    result = result.merge(ultima[["person_id", "tiempo_ultima"]], on="person_id", how="left")
    add_change(
        log,
        "created_variables",
        source="grupo_microorganismo",
        target=[col for col in result.columns if col.startswith("infprev_") and col.endswith("_binary")],
        how="pivot grouped organisms and binarize per patient/admission",
    )
    add_change(
        log,
        "created_variables",
        source="infprev_*_binary",
        target=infection_broad_group_columns,
        how="sum previous-infection organism binaries into broad Gram-stain groups using MICROORGANISM_BROAD_GROUP_MAP",
    )
    add_change(
        log,
        "created_variables",
        source="fecha_infeccion",
        target=["num_inf_previas", "tiempo_ultima"],
        how="derive previous infection count and days since last infection",
    )
    log.output_rows = len(result)
    log.output_columns = result.columns.tolist()
    result_obj = PreprocessResult(df=result, log=log)
    attach_run_metadata(result_obj.log, config)
    return validate_result(result_obj, required_columns=["person_id", "fecha_ingreso_urgencias"])


def preprocess_tbl_tratamiento_antibiotico_previo(
    tables: dict[str, pd.DataFrame],
    maps: dict[str, dict[Any, Any]],
    config: dict[str, Any],
) -> PreprocessResult:
    source = tables["tbl_tratamiento_antibiotico_previo"].copy()
    df = source.copy()
    log = finalize_log(
        table_name="tbl_tratamiento_antibiotico_previo",
        input_df=source,
        output_df=source,
        merge_keys=["person_id", "fecha_ingreso_urgencias"],
    )
    merge_keys = ["person_id", "fecha_ingreso_urgencias"]

    df["antib_previo_si_no"] = np.where(df["dias_trat_antimicrobiano"] > 0, 1, 0)
    rows_before_prior_filter = len(df)
    df = df[df["antib_previo_si_no"] == 1].copy()
    removed_without_prior_antibiotic = rows_before_prior_filter - len(df)
    log.validation_checks.append(
        f"rows_removed_without_prior_antibiotic:{removed_without_prior_antibiotic}"
    )
    log.notes.append(
        f"Removed {removed_without_prior_antibiotic} rows because antib_previo_si_no == 0 "
        "(dias_trat_antimicrobiano <= 0 or missing)."
    )
    add_change(
        log,
        "created_variables",
        source="dias_trat_antimicrobiano",
        target="antib_previo_si_no",
        how="1 if dias_trat_antimicrobiano > 0, then keep only rows with previous antibiotic exposure",
    )

    df["antimicrobiano_previo_nombre"] = df["antimicrobiano_previo"].map(
        maps["antibiotic_code_name_map"]
    )
    add_change(
        log,
        "recoded_variables",
        source="antimicrobiano_previo",
        target="antimicrobiano_previo_nombre",
        how="map prior antibiotic codes to decoded drug names using antibiotic_code_name_map",
    )

    df["prev_betalactamase_inhib"] = (
        df["antimicrobiano_previo_nombre"]
        .str.contains("beta-lactamase inhibitor", na=False)
        .groupby([df[col] for col in merge_keys])
        .transform("max")
        .astype(int)
    )
    add_change(
        log,
        "created_variables",
        source="antimicrobiano_previo_nombre",
        target="prev_betalactamase_inhib",
        how="per patient/admission flag: 1 if any decoded previous antibiotic contains 'beta-lactamase inhibitor'",
    )

    df["fecha_ingreso_urgencias_dt"] = pd.to_datetime(
        df["fecha_ingreso_urgencias"], errors="coerce"
    )
    df["fecha_administracion_antib_dt"] = pd.to_datetime(
        df["fecha_administracion_antib"], errors="coerce"
    )
    add_change(
        log,
        "transformed_variables",
        source=["fecha_ingreso_urgencias", "fecha_administracion_antib"],
        target=["fecha_ingreso_urgencias_dt", "fecha_administracion_antib_dt"],
        how="parse dates for pre-admission window filtering and days-since-last-antibiotic calculation",
    )

    days_since_administration = (
        df["fecha_ingreso_urgencias_dt"] - df["fecha_administracion_antib_dt"]
    ).dt.total_seconds() / 86400.0
    window_days = config["PRIOR_ANTIBIOTIC_WINDOW_DAYS"]
    within_window = days_since_administration < window_days
    df["antimicrobiano_previo_90d_nombre"] = (
        df["antimicrobiano_previo_nombre"].where(within_window).fillna("NEGATIVE")
    )
    df["antimicrobiano_previo_familia"] = df["antimicrobiano_previo_90d_nombre"].map(
        maps["antibiotic_name_family_map"]
    )
    log.validation_checks.append(
        f"rows_outside_{window_days}_day_antibiotic_window:{int((~within_window).sum())}"
    )
    add_change(
        log,
        "recoded_variables",
        source="antimicrobiano_previo_nombre",
        target=["antimicrobiano_previo_90d_nombre", "antimicrobiano_previo_familia"],
        how=(
            f"keep decoded drug names only when admission minus administration date is < {window_days} days; "
            "set older or missing-window rows to NEGATIVE, then map drug names to configured antibiotic families"
        ),
    )

    rows_before_future_filter = len(df)
    valid_pre_admission_date = (
        df["fecha_administracion_antib_dt"] <= df["fecha_ingreso_urgencias_dt"]
    )
    df = df[valid_pre_admission_date].copy()
    removed_date_mismatch = rows_before_future_filter - len(df)
    log.validation_checks.append(
        f"rows_removed_with_antibiotic_date_after_admission:{removed_date_mismatch}"
    )
    log.notes.append(
        f"Removed {removed_date_mismatch} rows because fecha_administracion_antib was after "
        "fecha_ingreso_urgencias."
    )
    add_change(
        log,
        "transformed_variables",
        source=["fecha_ingreso_urgencias_dt", "fecha_administracion_antib_dt"],
        target="row_filter",
        how="remove rows where fecha_administracion_antib is after fecha_ingreso_urgencias",
    )

    if df.empty:
        result_df = pd.DataFrame(
            columns=[
                "person_id",
                "fecha_ingreso_urgencias",
                "antib_previo_si_no",
                "ultimo_antib",
                "dias_ultimo_antib",
                "prev_betalactamase_inhib",
                "antib_previo_total_veces",
            ]
        )
    else:
        base = (
            df.groupby(merge_keys, dropna=False)
            .agg(
                antib_previo_si_no=("antib_previo_si_no", "max"),
                prev_betalactamase_inhib=("prev_betalactamase_inhib", "max"),
            )
            .reset_index()
        )

        last_antibiotic = (
            df.sort_values(merge_keys + ["fecha_administracion_antib_dt"])
            .groupby(merge_keys, dropna=False)
            .tail(1)[
                merge_keys
                + [
                    "antimicrobiano_previo_familia",
                    "fecha_ingreso_urgencias_dt",
                    "fecha_administracion_antib_dt",
                ]
            ]
            .copy()
        )
        last_antibiotic["dias_ultimo_antib"] = (
            last_antibiotic["fecha_ingreso_urgencias_dt"]
            - last_antibiotic["fecha_administracion_antib_dt"]
        ).dt.total_seconds() / 86400.0
        last_antibiotic = last_antibiotic.rename(
            columns={"antimicrobiano_previo_familia": "ultimo_antib"}
        )[merge_keys + ["ultimo_antib", "dias_ultimo_antib"]]

        family_rows = df[df["antimicrobiano_previo_familia"].notna()].copy()
        if family_rows.empty:
            family_binary_columns: list[str] = []
            family_pivot = base[merge_keys].copy()
        else:
            family_pivot = (
                family_rows.assign(presence=1)
                .pivot_table(
                    index=merge_keys,
                    columns="antimicrobiano_previo_familia",
                    values="presence",
                    aggfunc="sum",
                    fill_value=0,
                )
                .reset_index()
            )
            family_columns = [
                col for col in family_pivot.columns if col not in merge_keys
            ]
            family_binary_columns = []
            for family in family_columns:
                binary_column = f"{feature_name(family)}_binary"
                family_pivot[binary_column] = np.where(family_pivot[family] >= 1, 1, 0)
                family_binary_columns.append(binary_column)
            family_pivot = family_pivot.drop(columns=family_columns)

        result_df = base.merge(last_antibiotic, on=merge_keys, how="left")
        result_df = result_df.merge(family_pivot, on=merge_keys, how="left")
        for column in family_binary_columns:
            result_df[column] = result_df[column].fillna(0).astype(int)
        result_df["antib_previo_total_veces"] = (
            result_df[family_binary_columns].sum(axis=1).astype(int)
            if family_binary_columns
            else 0
        )

        add_change(
            log,
            "created_variables",
            source=["fecha_ingreso_urgencias", "fecha_administracion_antib", "antimicrobiano_previo_familia"],
            target=["ultimo_antib", "dias_ultimo_antib"],
            how="per patient/admission, keep the family of the latest previous antibiotic administration and compute days to admission",
        )
        add_change(
            log,
            "created_variables",
            source="antimicrobiano_previo_familia",
            target=family_binary_columns,
            how="pivot configured antibiotic families to binary exposure flags per patient/admission",
        )
        add_change(
            log,
            "created_variables",
            source=family_binary_columns,
            target="antib_previo_total_veces",
            how="sum explicit antibiotic-family binary columns; does not use positional column indexes",
        )

    add_change(
        log,
        "dropped_variables",
        source=[
            "fecha_administracion_antib",
            "antimicrobiano_previo",
            "via_administ_antib_prev",
            "dias_trat_antimicrobiano",
        ],
        target=[
            "antib_previo_si_no",
            "prev_betalactamase_inhib",
            "ultimo_antib",
            "dias_ultimo_antib",
            "*_binary",
            "antib_previo_total_veces",
        ],
        how="aggregate raw previous-antibiotic treatment rows into one patient/admission feature row",
    )
    log.output_rows = len(result_df)
    log.output_columns = result_df.columns.tolist()
    result = PreprocessResult(df=result_df, log=log)
    attach_run_metadata(result.log, config)
    return validate_result(result, required_columns=["person_id", "fecha_ingreso_urgencias"])


def preprocess_tbl_hemocultivo_de_urgencias(
    tables: dict[str, pd.DataFrame],
    maps: dict[str, dict[Any, Any]],
    config: dict[str, Any],
) -> PreprocessResult:
    source = tables["tbl_hemocultivo_de_urgencias"].copy()
    df = source.copy()
    log = finalize_log(
        table_name="tbl_hemocultivo_de_urgencias",
        input_df=source,
        output_df=source,
        merge_keys=["person_id", "fecha_ingreso_urgencias"],
    )
    merge_keys = ["person_id", "fecha_ingreso_urgencias"]

    microorganism_codes = (
        pd.to_numeric(df["microorganismo"], errors="coerce")
        .astype("Int64")
        .astype(str)
        .replace("<NA>", "0")
    )
    df["microorganismo"] = microorganism_codes.map(maps["organism_codes_map"]).fillna("NEGATIVE")
    df["microorganismo_pre_correccion_clinica"] = df["microorganismo"]
    add_change(
        log,
        "recoded_variables",
        source="microorganismo",
        target="microorganismo",
        how="map hemoculture microorganism SNOMED codes to grouped organism labels; missing and unmapped codes become NEGATIVE",
    )

    conflicts = (
        df.groupby(["person_id", "id_hemocultivo"], dropna=False)["bmr_etiologia"]
        .nunique(dropna=True)
        .reset_index(name="distinct_bmr_etiologia")
        .query("distinct_bmr_etiologia > 1")
    )
    log.validation_checks.append(
        f"hemoculture_bmr_conflict_groups:{len(conflicts)}"
    )
    if not conflicts.empty:
        log.warnings.append(
            f"{len(conflicts)} person_id/id_hemocultivo groups have conflicting bmr_etiologia values; aggregation uses max."
    )

    coinfection_map = maps["hemoculture_coinfection_resolution_map"]
    pre_correction_grouped = (
        df.groupby(merge_keys + ["microorganismo_pre_correccion_clinica"], as_index=False, dropna=False)
        .agg(hemo_positivo_si_no=("hemo_positivo_si_no", "max"))
    )
    pre_correction_pivot = (
        pre_correction_grouped.pivot_table(
            index=merge_keys,
            columns="microorganismo_pre_correccion_clinica",
            values="hemo_positivo_si_no",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    pre_correction_organism_columns = [
        column
        for column in pre_correction_pivot.columns
        if column not in merge_keys and column != "NEGATIVE"
    ]

    def build_multilabel_resultado_hemo(row: pd.Series) -> tuple[str, ...]:
        organisms = [
            column
            for column in pre_correction_organism_columns
            if row[column] >= 1
        ]
        return tuple(organisms) if organisms else ("NEGATIVE",)

    pre_correction_pivot["resultado_hemo_multilabel"] = pre_correction_pivot.apply(
        build_multilabel_resultado_hemo,
        axis=1,
    )
    pre_correction_result = pre_correction_pivot[
        merge_keys + ["resultado_hemo_multilabel"]
    ]

    override_mask = df["person_id"].isin(coinfection_map)
    override_person_count = int(df.loc[override_mask, "person_id"].nunique())
    override_row_count = int(override_mask.sum())
    df.loc[override_mask, "microorganismo"] = df.loc[override_mask, "person_id"].map(
        coinfection_map
    )
    log.validation_checks.append(
        f"coinfection_override_persons:{override_person_count}"
    )
    log.validation_checks.append(
        f"coinfection_override_rows:{override_row_count}"
    )
    log.notes.append(
        f"Applied clinician-reviewed co-infection organism overrides to {override_row_count} rows from {override_person_count} patients."
    )
    add_change(
        log,
        "recoded_variables",
        source="person_id",
        target="microorganismo",
        how="override grouped organism label with clinician-reviewed dominant organism map for known co-infection patients",
    )

    df["bmr_etiologia"] = df["bmr_etiologia"].fillna(0.0)

    def phenotype_tuple(values: pd.Series) -> tuple[float, ...]:
        return tuple(0.0 if pd.isna(value) else float(value) for value in values)

    grouped = (
        df.groupby(merge_keys + ["microorganismo"], as_index=False, dropna=False)
        .agg(
            id_hemocultivo=("id_hemocultivo", "first"),
            fecha_hemocultivo=("fecha_hemocultivo", "first"),
            hemo_positivo_si_no=("hemo_positivo_si_no", "max"),
            bmr_etiologia=("bmr_etiologia", "max"),
            fenotipo_resistencia=("fenotipo_resistencia", phenotype_tuple),
        )
    )
    add_change(
        log,
        "transformed_variables",
        source=[
            "id_hemocultivo",
            "fecha_hemocultivo",
            "hemo_positivo_si_no",
            "bmr_etiologia",
            "fenotipo_resistencia",
        ],
        target=[
            "hemo_positivo_si_no",
            "bmr_etiologia",
            "fenotipo_resistencia",
        ],
        how="deduplicate per patient/admission/organism; use max hemo_positivo_si_no and bmr_etiologia, collect resistance phenotype codes as tuples",
    )

    base = (
        grouped.groupby(merge_keys, as_index=False, dropna=False)
        .agg(
            bmr_etiologia=("bmr_etiologia", "max"),
            fenotipo_resistencia=(
                "fenotipo_resistencia",
                lambda tuples: tuple(value for phenotype in tuples for value in phenotype),
            ),
        )
    )

    organism_pivot = (
        grouped.pivot_table(
            index=merge_keys,
            columns="microorganismo",
            values="hemo_positivo_si_no",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    organism_columns = [
        column
        for column in organism_pivot.columns
        if column not in merge_keys
    ]
    organism_column_rename = {
        column: f"hemo_{feature_name(column)}_binary"
        for column in organism_columns
    }
    for column in organism_columns:
        organism_pivot[column] = np.where(organism_pivot[column] >= 1, 1, 0)
    organism_pivot = organism_pivot.rename(columns=organism_column_rename)
    hemo_organism_columns = list(organism_column_rename.values())

    result_df = base.merge(organism_pivot, on=merge_keys, how="left")
    result_df = result_df.merge(pre_correction_result, on=merge_keys, how="left")

    dominant_organism_columns = {
        organism_column_rename[column]: column
        for column in organism_columns
        if column != "NEGATIVE"
    }

    def build_resultado_hemo(row: pd.Series) -> str:
        organisms = [
            organism
            for column, organism in dominant_organism_columns.items()
            if row[column] == 1
        ]
        return organisms[0] if organisms else "NEGATIVE"

    result_df["resultado_hemo"] = result_df.apply(build_resultado_hemo, axis=1)
    remaining_multiorganism_rows = int(
        result_df[list(dominant_organism_columns)].sum(axis=1).gt(1).sum()
    ) if dominant_organism_columns else 0
    log.validation_checks.append(
        f"post_clinical_correction_multiorganism_rows:{remaining_multiorganism_rows}"
    )
    if remaining_multiorganism_rows > 0:
        log.warnings.append(
            f"{remaining_multiorganism_rows} rows still have multiple positive organisms after clinician correction; resultado_hemo uses the first organism column."
        )
    add_change(
        log,
        "created_variables",
        source="microorganismo",
        target=hemo_organism_columns,
        how="pivot grouped hemoculture organisms to binary columns per patient/admission",
    )
    add_change(
        log,
        "created_variables",
        source=list(dominant_organism_columns),
        target="resultado_hemo",
        how="scalar dominant organism after clinician co-infection correction; NEGATIVE if no positive non-NEGATIVE organism column exists",
    )
    add_change(
        log,
        "created_variables",
        source="microorganismo_pre_correccion_clinica",
        target="resultado_hemo_multilabel",
        how="pre-clinician-correction tuple of all positive grouped organisms; NEGATIVE tuple if no positive organism exists",
    )
    add_change(
        log,
        "dropped_variables",
        source=[
            "id_hemocultivo",
            "fecha_hemocultivo",
            "hemo_positivo_si_no",
            "microorganismo_pre_correccion_clinica",
        ],
        target=[
            "bmr_etiologia",
            "fenotipo_resistencia",
            "*organism_binary_columns",
            "resultado_hemo",
            "resultado_hemo_multilabel",
        ],
        how="aggregate raw hemoculture rows into one patient/admission feature and target row",
    )

    log.output_rows = len(result_df)
    log.output_columns = result_df.columns.tolist()
    result = PreprocessResult(df=result_df, log=log)
    attach_run_metadata(result.log, config)
    return validate_result(result, required_columns=["person_id", "fecha_ingreso_urgencias"])


def preprocess_tbl_colonizaciones_previas(
    tables: dict[str, pd.DataFrame],
    maps: dict[str, dict[Any, Any]],
    config: dict[str, Any],
) -> PreprocessResult:
    source = tables["tbl_colonizaciones_previas"].copy()
    df = source.copy()
    log = finalize_log(
        table_name="tbl_colonizaciones_previas",
        input_df=source,
        output_df=source,
        merge_keys=["person_id", "fecha_ingreso_urgencias"],
    )
    merge_keys = ["person_id", "fecha_ingreso_urgencias"]

    df["microorganism_colonizador_grupo"] = (
        pd.to_numeric(df["microorganism_colonizador"], errors="coerce")
        .astype("Int64")
        .astype(str)
        .replace("<NA>", "0")
        .map(maps["organism_codes_map"])
    )
    unmapped_rows = int(df["microorganism_colonizador_grupo"].isna().sum())
    log.validation_checks.append(f"unmapped_colonization_organism_rows:{unmapped_rows}")
    if unmapped_rows:
        log.warnings.append(
            f"{unmapped_rows} colonization rows have microorganism_colonizador values that do not map to organism groups; these rows do not create organism binary columns."
        )
    add_change(
        log,
        "recoded_variables",
        source="microorganism_colonizador",
        target="microorganism_colonizador_grupo",
        how="map colonizing microorganism SNOMED codes to grouped organism labels using organism_codes_map",
    )

    organisms = df["microorganism_colonizador_grupo"].dropna().unique().tolist()
    pivot_source = df[merge_keys + ["microorganism_colonizador_grupo"]].copy()
    pivot_source["dummy"] = 1
    pivoted = (
        pivot_source.pivot_table(
            index=merge_keys,
            columns="microorganism_colonizador_grupo",
            values="dummy",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    colonization_binary_columns: list[str] = []
    for organism in organisms:
        binary_column = f"colo_{feature_name(organism)}_binary"
        pivoted[binary_column] = np.where(pivoted[organism] >= 1, 1, 0)
        colonization_binary_columns.append(binary_column)
        pivoted = pivoted.drop(columns=organism)

    pivoted["colonizacion_total_grouped"] = (
        pivoted[colonization_binary_columns].sum(axis=1).astype(int)
        if colonization_binary_columns
        else 0
    )
    colonization_broad_group_columns: list[str] = []
    for broad_group in sorted(set(config["MICROORGANISM_BROAD_GROUP_MAP"].values())):
        target_column = f"Colo_{feature_name(broad_group)}"
        source_columns = [
            f"colo_{feature_name(organism)}_binary"
            for organism, group in config["MICROORGANISM_BROAD_GROUP_MAP"].items()
            if group == broad_group
        ]
        existing_source_columns = [
            column for column in source_columns if column in pivoted.columns
        ]
        pivoted[target_column] = (
            pivoted[existing_source_columns].fillna(0).sum(axis=1)
            if existing_source_columns
            else 0
        )
        colonization_broad_group_columns.append(target_column)
    add_change(
        log,
        "created_variables",
        source="microorganism_colonizador_grupo",
        target=colonization_binary_columns,
        how="pivot grouped colonizing organisms to binary columns per patient/admission",
    )
    add_change(
        log,
        "created_variables",
        source=colonization_binary_columns,
        target="colonizacion_total_grouped",
        how="sum explicit colonization organism binary columns; moved from notebook post-merge synthetic features into table-local preprocessing",
    )
    add_change(
        log,
        "created_variables",
        source="colo_*_binary",
        target=colonization_broad_group_columns,
        how="sum colonization organism binaries into broad Gram-stain groups using MICROORGANISM_BROAD_GROUP_MAP",
    )
    add_change(
        log,
        "dropped_variables",
        source=[
            "fecha_colonizacion",
            "microorganism_colonizador",
            "bmr_colonizador",
            "feno_resist_colo",
        ],
        target=colonization_binary_columns + ["colonizacion_total_grouped"],
        how="replace raw colonization rows with grouped organism binary features; notebook did not preserve colonization BMR or phenotype detail",
    )

    log.output_rows = len(pivoted)
    log.output_columns = pivoted.columns.tolist()
    result = PreprocessResult(df=pivoted, log=log)
    attach_run_metadata(result.log, config)
    return validate_result(result, required_columns=["person_id", "fecha_ingreso_urgencias"])


def preprocess_tbl_otros_cultivos_en_urgencias(
    tables: dict[str, pd.DataFrame],
    maps: dict[str, dict[Any, Any]],
    config: dict[str, Any],
) -> PreprocessResult:
    source = tables["tbl_otros_cultivos_en_urgencias"].copy()
    df = source.copy()
    log = finalize_log(
        table_name="tbl_otros_cultivos_en_urgencias",
        input_df=source,
        output_df=source,
        merge_keys=["person_id", "fecha_ingreso_urgencias"],
    )
    merge_keys = ["person_id", "fecha_ingreso_urgencias"]

    organism_codes = (
        pd.to_numeric(df["microorganismo_otros_cult"], errors="coerce")
        .astype("Int64")
        .astype(str)
        .replace("<NA>", "0")
    )
    df["otro_cult_microorganismo"] = (
        organism_codes.map(maps["organism_codes_map"]).fillna("NEGATIVE")
    )
    add_change(
        log,
        "recoded_variables",
        source="microorganismo_otros_cult",
        target="otro_cult_microorganismo",
        how="map other-emergency-culture microorganism SNOMED codes to grouped organism labels; missing and unmapped codes become NEGATIVE",
    )

    df["tipo_cultivo"] = df["tipo_cultivo"].fillna(0)
    add_change(
        log,
        "transformed_variables",
        source="tipo_cultivo",
        target="tipo_cultivo",
        how="fill missing culture type with 0 before duplicate removal, matching notebook behavior",
    )

    columns_before_dropna = len(df)
    df = df.dropna(subset=merge_keys + ["otro_cult_microorganismo"]).drop_duplicates().copy()
    rows_removed = columns_before_dropna - len(df)
    log.validation_checks.append(f"rows_removed_missing_key_or_organism_after_mapping:{rows_removed}")
    if rows_removed:
        log.notes.append(
            f"Removed {rows_removed} rows with missing merge keys or mapped other-culture organism after mapping."
        )

    pivot_source = df[merge_keys + ["otro_cult_microorganismo"]].copy()
    pivot_source["dummy"] = 1
    pivoted = (
        pivot_source.pivot_table(
            index=merge_keys,
            columns="otro_cult_microorganismo",
            values="dummy",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    organism_columns = [
        column for column in pivoted.columns if column not in merge_keys
    ]
    rename_columns = {
        column: f"otros_cult_{feature_name(column)}_count"
        for column in organism_columns
    }
    result_df = pivoted.rename(columns=rename_columns)
    other_culture_columns = list(rename_columns.values())

    add_change(
        log,
        "created_variables",
        source="otro_cult_microorganismo",
        target=other_culture_columns,
        how="pivot grouped other emergency culture organisms to count columns per patient/admission",
    )
    add_change(
        log,
        "dropped_variables",
        source=[
            "tipo_cultivo",
            "id_otros_cultivos",
            "fecha_otros_cultivos",
            "microorganismo_otros_cult",
            "bmr_etiologia_otros",
            "fenotipo_resistencia_otros",
        ],
        target=other_culture_columns,
        how="replace raw other-culture rows with grouped organism count columns; notebook did not preserve culture type, BMR, or phenotype detail",
    )

    log.output_rows = len(result_df)
    log.output_columns = result_df.columns.tolist()
    result = PreprocessResult(df=result_df, log=log)
    attach_run_metadata(result.log, config)
    return validate_result(result, required_columns=["person_id", "fecha_ingreso_urgencias"])


def merge_with_validation(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    *,
    left_name: str,
    right_name: str,
    keys: list[str],
    how: str,
    log: TableLog,
) -> pd.DataFrame:
    ensure_columns_present(left_df, keys, left_name)
    ensure_columns_present(right_df, keys, right_name)
    left_dupes = duplicate_key_rows(left_df, keys)
    left_dup_groups = duplicate_key_groups(left_df, keys)
    right_dupes = duplicate_key_rows(right_df, keys)
    right_dup_groups = duplicate_key_groups(right_df, keys)
    left_admission_stats = admission_stats(left_df)
    right_admission_stats = admission_stats(right_df)
    overlaps = overlapping_non_key_columns(left_df, right_df, keys)
    before_rows = len(left_df)
    log.validation_checks.append(
        f"merge:{left_name}+{right_name}:before_rows={before_rows}:left_dupes={left_dupes}:left_dup_groups={left_dup_groups}:right_dupes={right_dupes}:right_dup_groups={right_dup_groups}:keys={keys}"
    )
    log.validation_checks.append(format_duplicate_key_stats(left_name, left_dupes, left_dup_groups, keys))
    log.validation_checks.append(format_duplicate_key_stats(right_name, right_dupes, right_dup_groups, keys))
    log.validation_checks.append(format_admission_stats(left_name, left_admission_stats))
    log.validation_checks.append(format_admission_stats(right_name, right_admission_stats))
    if left_dupes > 0:
        log.warnings.append(
            format_duplicate_key_stats(left_name, left_dupes, left_dup_groups, keys)
        )
    if right_dupes > 0:
        log.warnings.append(
            format_duplicate_key_stats(right_name, right_dupes, right_dup_groups, keys)
        )
    if overlaps:
        overlap_text = ", ".join(overlaps)
        raise ValueError(
            f"Cannot merge {right_name} into {left_name} on keys {keys}. "
            f"Duplicate-key stats on the chosen merge keys: "
            f"{format_duplicate_key_stats(left_name, left_dupes, left_dup_groups, keys)}; "
            f"{format_duplicate_key_stats(right_name, right_dupes, right_dup_groups, keys)}. "
            f"Admission stats: {format_admission_stats(left_name, left_admission_stats)}; "
            f"{format_admission_stats(right_name, right_admission_stats)}. "
            f"Overlapping non-key columns would be duplicated/suffixed: {overlap_text}. "
            "Resolve this in the table preprocessor or adjust the merge inputs explicitly."
        )
    merged = left_df.merge(right_df, on=keys, how=how)
    after_rows = len(merged)
    row_delta = after_rows - before_rows
    log.notes.append(
        f"Merged {right_name} into {left_name} with how={how}, keys={keys}, rows {before_rows}->{after_rows} (delta={row_delta})"
    )
    if row_delta > 0:
        log.warnings.append(
            f"Merge {left_name} <- {right_name} increased rows by {row_delta}. This usually indicates one-to-many or many-to-many matches on keys {keys}."
        )
    elif row_delta < 0:
        log.warnings.append(
            f"Merge {left_name} <- {right_name} reduced rows by {-row_delta}. Review merge type and key completeness on {keys}."
        )
    return merged


def merge_preprocessed_tables(
    results: dict[str, PreprocessResult],
    config: dict[str, Any],
) -> PreprocessResult:
    master = results["tbl_paciente"].df.copy()
    input_df = master.copy()
    log = finalize_log(
        table_name="merged_dataset",
        input_df=input_df,
        output_df=input_df,
        merge_keys=["person_id", "fecha_ingreso_urgencias"],
    )

    merge_plan = [
        ("tbl_comorbilidad", ["person_id", "fecha_ingreso_urgencias"], "left"),
        ("tbl_factores_riesgo_bmr", ["person_id", "fecha_ingreso_urgencias"], "left"),
        ("tbl_sepsis", ["person_id", "fecha_ingreso_urgencias"], "left"),
        ("tbl_signos", ["person_id", "fecha_ingreso_urgencias"], "left"),
        ("tbl_sintomas", ["person_id", "fecha_ingreso_urgencias"], "left"),
        ("tbl_infecciones_previas", ["person_id", "fecha_ingreso_urgencias"], "left"),
        ("tbl_tratamiento_antibiotico_previo", ["person_id", "fecha_ingreso_urgencias"], "left"),
        ("tbl_colonizaciones_previas", ["person_id", "fecha_ingreso_urgencias"], "left"),
        ("tbl_hemocultivo_de_urgencias", ["person_id", "fecha_ingreso_urgencias"], "left"),
        ("tbl_otros_cultivos_en_urgencias", ["person_id", "fecha_ingreso_urgencias"], "left"),
    ]

    for name, keys, how in merge_plan:
        master = merge_with_validation(
            master,
            results[name].df,
            left_name="merged_dataset",
            right_name=name,
            keys=keys,
            how=how,
            log=log,
        )
    log.output_rows = len(master)
    log.output_columns = master.columns.tolist()
    log.notes.append("Keep this stage thin. Most engineering should happen inside the table preprocessors.")
    attach_run_metadata(log, config)
    return validate_result(
        PreprocessResult(df=master, log=log),
        required_columns=["person_id", "fecha_ingreso_urgencias"],
    )


def build_cross_table_features(df: pd.DataFrame, config: dict[str, Any]) -> PreprocessResult:
    source = df.copy()
    result = df.copy()
    log = finalize_log(
        table_name="cross_table_features",
        input_df=source,
        output_df=result,
        merge_keys=["person_id", "fecha_ingreso_urgencias"],
    )
    organism_groups = sorted(
        {
            feature_name(label)
            for label in config["MICROORGANISM_LABEL_MAP"].values()
            if pd.notna(label)
        }
    )
    organism_groups_with_negative = organism_groups + ["NEGATIVE"]
    all_culture_columns: list[str] = []
    missing_source_pairs: list[str] = []

    for organism in organism_groups_with_negative:
        hemo_column = f"hemo_{organism}_binary"
        other_column = f"otros_cult_{organism}_count"
        combined_column = f"all_cult_{organism}_count"
        hemo_values = (
            result[hemo_column].fillna(0)
            if hemo_column in result.columns
            else pd.Series(0, index=result.index)
        )
        other_values = (
            result[other_column].fillna(0)
            if other_column in result.columns
            else pd.Series(0, index=result.index)
        )
        if hemo_column not in result.columns or other_column not in result.columns:
            missing_source_pairs.append(f"{organism}:hemo={hemo_column in result.columns},otros={other_column in result.columns}")
        result[combined_column] = hemo_values + other_values
        all_culture_columns.append(combined_column)

    non_negative_all_culture_columns = [
        column for column in all_culture_columns if column != "all_cult_NEGATIVE_count"
    ]
    column_to_organism = {
        f"all_cult_{feature_name(label)}_count": label
        for label in set(config["MICROORGANISM_LABEL_MAP"].values())
        if pd.notna(label)
    }

    def pick_dominant_urgent_culture(row: pd.Series) -> str:
        if (row >= 2).any():
            return column_to_organism[row.idxmax()]
        if (row == 1).sum() == 1:
            return column_to_organism[row[row == 1].index[0]]
        return "NEGATIVE"

    result["dominant_all_cult_org"] = result[non_negative_all_culture_columns].apply(
        pick_dominant_urgent_culture,
        axis=1,
    )
    log.validation_checks.append(
        f"dominant_all_cult_org_non_negative_rows:{int(result['dominant_all_cult_org'].ne('NEGATIVE').sum())}"
    )
    if missing_source_pairs:
        log.notes.append(
            "Some culture source columns were absent and treated as 0: "
            + "; ".join(missing_source_pairs)
        )
    add_change(
        log,
        "created_variables",
        source=[
            "hemo_*_binary",
            "otros_cult_*",
        ],
        target=all_culture_columns,
        how="sum hemoculture organism binaries and other-emergency-culture organism counts into combined urgent-culture count columns",
    )
    add_change(
        log,
        "created_variables",
        source=non_negative_all_culture_columns,
        target="dominant_all_cult_org",
        how=(
            "dominant urgent-culture organism: if any organism has combined count >= 2 choose the max; "
            "if exactly one organism has count == 1 choose it; otherwise NEGATIVE"
        ),
    )

    synthetic_feature_sources = {
        "recurrencia_precoz": ["tiempo_ultima"],
        "densidad_inf": ["num_inf_previas", "tiempo_ultima"],
        "residencia_dialisis": ["paciente_residencia", "hemodialisis_permanente"],
        "carga_dispositivos": [
            "cateter_venoso",
            "sonda_urinaria",
            "sonda_nasogastrica",
            "derivacion_ventriculoper",
            "valvula_prot_cardiaca",
            "portador_otros_disposit",
        ],
        "total_inmunoriesgo_cat": [
            "inmunosupresion",
            "sida",
            "linfoma",
            "leucemia",
            "neoplasia",
        ],
    }
    missing_synthetic_sources = sorted(
        {
            column
            for columns in synthetic_feature_sources.values()
            for column in columns
            if column not in result.columns
        }
    )
    if missing_synthetic_sources:
        log.warnings.append(
            "Synthetic feature source columns missing and treated as 0/NaN where needed: "
            + ", ".join(missing_synthetic_sources)
        )

    tiempo_ultima = (
        result["tiempo_ultima"]
        if "tiempo_ultima" in result.columns
        else pd.Series(np.nan, index=result.index)
    )
    num_inf_previas = (
        result["num_inf_previas"]
        if "num_inf_previas" in result.columns
        else pd.Series(np.nan, index=result.index)
    )
    result["recurrencia_precoz"] = (tiempo_ultima < 30).astype(int)
    result["densidad_inf"] = num_inf_previas / (tiempo_ultima + 1)
    result["residencia_dialisis"] = (
        result.get("paciente_residencia", pd.Series(0, index=result.index)).fillna(0).eq(1)
        | result.get("hemodialisis_permanente", pd.Series(0, index=result.index)).fillna(0).eq(1)
    ).astype(int)

    device_columns = synthetic_feature_sources["carga_dispositivos"]
    result["carga_dispositivos"] = pd.DataFrame(
        {
            column: result[column].fillna(0)
            if column in result.columns
            else pd.Series(0, index=result.index)
            for column in device_columns
        }
    ).sum(axis=1)

    immuno_columns = synthetic_feature_sources["total_inmunoriesgo_cat"]
    result["total_inmunoriesgo_cat"] = (
        pd.DataFrame(
            {
                column: result[column].fillna(0)
                if column in result.columns
                else pd.Series(0, index=result.index)
                for column in immuno_columns
            }
        )
        .gt(0)
        .sum(axis=1)
        .astype(int)
    )
    synthetic_targets = list(synthetic_feature_sources)
    add_change(
        log,
        "created_variables",
        source=sorted(
            {
                column
                for columns in synthetic_feature_sources.values()
                for column in columns
            }
        ),
        target=synthetic_targets,
        how="create synthetic features after table merge, because they combine patient, infection-history, and risk-factor columns",
    )

    organism_total_columns: list[str] = []
    for organism_label in sorted(
        {
            label
            for label in config["MICROORGANISM_LABEL_MAP"].values()
            if pd.notna(label)
        }
    ):
        organism_feature = feature_name(organism_label)
        target_column = f"{organism_feature}_total"
        source_columns = [
            f"infprev_{organism_feature}_binary",
            f"colo_{organism_feature}_binary",
            f"hemo_{organism_feature}_binary",
            f"otros_cult_{organism_feature}_count",
        ]
        existing_source_columns = [
            column for column in source_columns if column in result.columns
        ]
        result[target_column] = (
            result[existing_source_columns].fillna(0).sum(axis=1)
            if existing_source_columns
            else 0
        )
        organism_total_columns.append(target_column)
    add_change(
        log,
        "created_variables",
        source=[
            "infprev_*_binary",
            "colo_*_binary",
            "hemo_*_binary",
            "otros_cult_*_count",
        ],
        target=organism_total_columns,
        how=(
            "create per-organism total columns using explicit source-specific organism features; "
            "does not include all_cult_*_count to avoid double-counting hemoculture and other-culture evidence"
        ),
    )

    log.notes.append(
        "Original source columns not dropped in the log are retained in the full dataset; "
        "use the drop-columns file to remove them from filtered outputs if desired."
    )
    log.output_rows = len(result)
    log.output_columns = result.columns.tolist()
    attach_run_metadata(log, config)
    return validate_result(
        PreprocessResult(df=result, log=log),
        required_columns=["person_id", "fecha_ingreso_urgencias"],
    )


def build_targets(
    df: pd.DataFrame,
    maps: dict[str, dict[Any, Any]],
    config: dict[str, Any],
) -> PreprocessResult:
    source = df.copy()
    result = df.copy()
    log = finalize_log(
        table_name="target_building",
        input_df=source,
        output_df=result,
        merge_keys=["person_id", "fecha_ingreso_urgencias"],
    )
    result["resultado_hemo_grouped"] = (
        result["resultado_hemo"]
        .map(config["MICROORGANISM_BROAD_GROUP_MAP"])
        .fillna(result["resultado_hemo"])
    )
    add_change(
        log,
        "created_variables",
        source="resultado_hemo",
        target="resultado_hemo_grouped",
        how="map selected organisms to broad Gram-stain groups using MICROORGANISM_BROAD_GROUP_MAP; unmapped labels keep their original value",
    )

    result["infected_yes_no"] = np.where(
        result["resultado_hemo"] == "NEGATIVE",
        "NEGATIVE",
        "POSITIVE",
    )
    add_change(
        log,
        "created_variables",
        source="resultado_hemo",
        target="infected_yes_no",
        how="POSITIVE if resultado_hemo is not NEGATIVE, else NEGATIVE",
    )

    antibiotic_name_family_map = {
        drug_name: family
        for _, family, drug_name in config["ANTIMICROBIAL_GROUPS"]
        if drug_name is not None
    }
    phenotype_drug_family_map = {
        spanish_name: antibiotic_name_family_map[english_name]
        for spanish_name, english_name in config["FENOTYPE_DRUG_TRANSLATIONS"].items()
        if english_name in antibiotic_name_family_map
    }

    def phenotype_code_to_name(code: Any) -> str | None:
        if pd.isna(code):
            return None
        lookup_candidates: list[Any] = [code]
        try:
            lookup_candidates.append(float(code))
        except (TypeError, ValueError):
            pass
        try:
            lookup_candidates.append(int(float(code)))
            lookup_candidates.append(str(int(float(code))))
        except (TypeError, ValueError):
            pass
        lookup_candidates.append(str(code))
        for candidate in lookup_candidates:
            if candidate in maps["phenomap"]:
                return str(maps["phenomap"][candidate])
        return None

    def phenotype_codes_to_families(value: Any) -> list[str]:
        families: list[str] = []
        for code in as_tuple(value):
            phenotype_name = phenotype_code_to_name(code)
            if phenotype_name is None:
                continue
            drug_name = strip_resistance_prefix(phenotype_name)
            family = phenotype_drug_family_map.get(drug_name)
            if family is not None:
                families.append(family)
        deduplicated = sorted(set(families))
        return deduplicated if deduplicated else ["NEGATIVE"]

    result["fenotipo_resistencia"] = result["fenotipo_resistencia"].apply(
        phenotype_codes_to_families
    )
    add_change(
        log,
        "recoded_variables",
        source="fenotipo_resistencia",
        target="fenotipo_resistencia",
        how="decode phenotype resistance codes to drug names, map drugs to antimicrobial families, deduplicate, and set NEGATIVE when no mapped phenotype remains",
    )

    result["resistente_cefalosporina"] = result["fenotipo_resistencia"].apply(
        lambda labels: (
            "RESIST_CEFALOSPORINAS_3a_4a"
            if any("Cefalosporina" in label for label in labels)
            else "NEGATIVE"
        )
    )
    result["resistente_cefalosporina_multi"] = result["fenotipo_resistencia"].apply(
        lambda labels: (
            "RESIST_CEFALOSPORINAS_3a_4a"
            if any("Cefalosporina" in label for label in labels)
            else ("OTHER" if not all(label == "NEGATIVE" for label in labels) else "NEGATIVE")
        )
    )
    add_change(
        log,
        "created_variables",
        source="fenotipo_resistencia",
        target=["resistente_cefalosporina", "resistente_cefalosporina_multi"],
        how=(
            "derive cephalosporin resistance targets from mapped phenotype families: "
            "binary target is resistant if any family contains 'Cefalosporina'; "
            "multi target distinguishes cephalosporin resistance, other resistance, and NEGATIVE"
        ),
    )

    result["sample_weight"] = 1
    add_change(
        log,
        "created_variables",
        source="row",
        target="sample_weight",
        how="set uniform sample weight to 1, matching final notebook output",
    )
    log.output_rows = len(result)
    log.output_columns = result.columns.tolist()
    attach_run_metadata(log, config)
    return validate_result(
        PreprocessResult(df=result, log=log),
        required_columns=["person_id", "fecha_ingreso_urgencias"],
    )


def build_run_log(
    *,
    input_file_path: Path,
    output_file_path: Path,
    drop_columns_path: Path,
    config: dict[str, Any],
) -> TableLog:
    log = TableLog(
        table_name="_pipeline_run",
        input_rows=0,
        output_rows=0,
        input_columns=[],
        output_columns=[],
        merge_keys=[],
    )
    attach_run_metadata(log, config)
    log.notes.extend(
        [
            f"input_file_path={input_file_path}",
            f"output_file_path={output_file_path}",
            f"drop_columns_path={drop_columns_path}",
            f"config_file_path={config['CONFIG_PATH']}",
        ]
    )
    log.metadata["config_values"] = config
    return log


def run_pipeline(
    input_file_path: Path = DEFAULT_DB_PATH,
    output_file_path: Path = DEFAULT_OUTPUT_PATH,
    config_file_path: Path = DEFAULT_CONFIG_PATH,
    drop_columns_path: Path = DEFAULT_DROP_COLUMNS_PATH,
) -> PipelineArtifacts:
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    tables = load_tables(input_file_path)
    config = load_config_module(config_file_path)
    maps = build_reference_maps(tables, config)

    results: dict[str, PreprocessResult] = {}
    results["tbl_paciente"] = preprocess_tbl_paciente(tables, maps, config)
    results["tbl_comorbilidad"] = preprocess_tbl_comorbilidad(tables, maps, config)
    results["tbl_factores_riesgo_bmr"] = preprocess_tbl_factores_riesgo_bmr(tables, maps, config)
    results["tbl_sintomas"] = preprocess_tbl_sintomas(tables, maps, config)
    results["tbl_signos"] = preprocess_tbl_signos(tables, maps, config)
    results["tbl_sepsis"] = preprocess_tbl_sepsis(tables, maps, config)
    results["tbl_infecciones_previas"] = preprocess_tbl_infecciones_previas(tables, maps, config)
    results["tbl_tratamiento_antibiotico_previo"] = preprocess_tbl_tratamiento_antibiotico_previo(tables, maps, config)
    results["tbl_hemocultivo_de_urgencias"] = preprocess_tbl_hemocultivo_de_urgencias(tables, maps, config)
    results["tbl_colonizaciones_previas"] = preprocess_tbl_colonizaciones_previas(tables, maps, config)
    results["tbl_otros_cultivos_en_urgencias"] = preprocess_tbl_otros_cultivos_en_urgencias(tables, maps, config)

    merged = merge_preprocessed_tables(results, config)
    cross_features = build_cross_table_features(merged.df, config)
    targets = build_targets(cross_features.df, maps, config)

    logs = [
        build_run_log(
            input_file_path=input_file_path,
            output_file_path=output_file_path,
            drop_columns_path=drop_columns_path,
            config=config,
        )
    ]
    logs.extend([result.log for result in results.values()])
    logs.extend([merged.log, cross_features.log, targets.log])

    export_dataset_outputs(
        targets.df,
        output_file_path,
        drop_columns_path=drop_columns_path,
    )
    targets.log.notes.append(
        f"Full dataset keeps all columns. Filtered dataset uses drop-columns file: {drop_columns_path}"
    )
    export_logs(
        logs,
        summary_output_path=output_file_path.with_name(
            f"{output_file_path.stem}_log_summary.csv"
        ),
        detailed_output_path=output_file_path.with_name(
            f"{output_file_path.stem}_log_detailed.json"
        ),
    )

    return PipelineArtifacts(
        tables={name: result.df for name, result in results.items()} | {"final": targets.df},
        maps=maps,
        logs=logs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess MePRAM tables into a modelling-ready dataset.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Full path to the SQLite input file, including filename. Default: {DEFAULT_DB_PATH}",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Full path to the output CSV file, including filename. Log files are written alongside it. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Full path to the preprocessing config file, including filename. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--drop-columns-path",
        type=Path,
        default=DEFAULT_DROP_COLUMNS_PATH,
        help=(
            "Full path to a text file listing columns or glob patterns to drop "
            f"from the filtered output CSV. Default: {DEFAULT_DROP_COLUMNS_PATH}"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        input_file_path=args.input_path,
        output_file_path=args.output_path,
        config_file_path=args.config_path,
        drop_columns_path=args.drop_columns_path,
    )


if __name__ == "__main__":
    main()
