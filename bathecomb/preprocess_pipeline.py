from __future__ import annotations

import argparse
import fnmatch
import json
import sqlite3
import subprocess
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "playground" / "db_bacthecom.db"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "playground" / "preprocess_bathecomb.csv"
DEFAULT_DROP_COLUMNS_PATH = ROOT_DIR / "config" / "preprocess_columns_to_drop.txt"
ADMISSION_KEYS = ["record_id", "fecha_ingreso"]


@dataclass
class VariableChange:
    kind: str
    source: str | list[str]
    target: str | list[str]
    how: str
    variable_type: str = ""
    n_classes: int | dict[str, int] | None = None
    descriptions: dict[str, str] = field(default_factory=dict)


@dataclass
class TableLog:
    table_name: str
    input_rows: int
    output_rows: int
    input_columns: list[str]
    output_columns: list[str]
    merge_keys: list[str]
    columns_lost: list[str] = field(default_factory=list)
    columns_dropped_count: int = 0
    created_variables: list[VariableChange] = field(default_factory=list)
    recoded_variables: list[VariableChange] = field(default_factory=list)
    transformed_variables: list[VariableChange] = field(default_factory=list)
    dropped_variables: list[VariableChange] = field(default_factory=list)
    role_variables: list[VariableChange] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    validation_checks: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessResult:
    df: pd.DataFrame
    log: TableLog


@dataclass
class PipelineArtifacts:
    tables: dict[str, pd.DataFrame]
    logs: list[TableLog]


ANTIMICROBIAL_FAMILY_BY_NORMALIZED_NAME = {
    "amikacina": "Aminoglucosidos",
    "gentamicina": "Aminoglucosidos",
    "kanamicina": "Aminoglucosidos",
    "netilmicina": "Aminoglucosidos",
    "tobramicina": "Aminoglucosidos",
    "amoxicilina": "Penicilinas",
    "amoxicilina clavulanico": "Penicilinas",
    "amoxicillin clavulanic acid": "Penicilinas",
    "ampicilina": "Penicilinas",
    "ampicilina sulbactam": "Penicilinas",
    "cloxacilina": "Penicilinas",
    "cloxacillin": "Penicilinas",
    "mecillinam": "Penicilinas",
    "oxacilina": "Penicilinas",
    "penicilina": "Penicilinas",
    "piperacilina": "Penicilinas",
    "piperacilina tazobactam": "Penicilinas",
    "piperacillin tazobactam": "Penicilinas",
    "ticarcilina": "Penicilinas",
    "ticarcilina clavulanico": "Penicilinas",
    "aztreonam": "Monobactamicos",
    "cefalexina": "Cefalosporinas 1 gen",
    "cefalotina": "Cefalosporinas 1 gen",
    "cefazolina": "Cefalosporinas 1 gen",
    "cefoxitina": "Cefalosporinas 2 gen",
    "cefuroxima": "Cefalosporinas 2 gen",
    "cefepime": "Cefalosporinas 3/4 gen",
    "cefepima": "Cefalosporinas 3/4 gen",
    "cefiderocol": "Cefalosporinas 3/4 gen",
    "cefixime": "Cefalosporinas 3/4 gen",
    "cefixima": "Cefalosporinas 3/4 gen",
    "cefotaxima": "Cefalosporinas 3/4 gen",
    "ceftriaxona": "Cefalosporinas 3/4 gen",
    "ceftriaxone": "Cefalosporinas 3/4 gen",
    "ceftazidima": "Cefalosporinas 3/4 gen",
    "ceftazidima avibactam": "Cefalosporinas 3/4 gen",
    "ceftazidima clavulanico": "Cefalosporinas 3/4 gen",
    "ceftolozano tazobactam": "Cefalosporinas 3/4 gen",
    "ceftarolina": "Cefalosporinas 2 gen",
    "ceftarolina fosamilo": "Cefalosporinas 2 gen",
    "doripenem": "Carbapenemas",
    "ertapenem": "Carbapenemas",
    "imipenem": "Carbapenemas",
    "imipenem cilastatina": "Carbapenemas",
    "imipenen cilastatina": "Carbapenemas",
    "imipenem cilastatin": "Carbapenemas",
    "meropenem": "Carbapenemas",
    "meropenem vaborbactam": "Carbapenemas",
    "ciprofloxacino": "Quinolonas",
    "ciprofloxacin": "Quinolonas",
    "delafloxacina": "Quinolonas",
    "levofloxacino": "Quinolonas",
    "levofloxacina": "Quinolonas",
    "levofloxacin": "Quinolonas",
    "moxifloxacino": "Quinolonas",
    "norfloxacino": "Quinolonas",
    "ofloxacino": "Quinolonas",
    "acido nalidixico": "Quinolonas",
    "vancomicina": "Glicopeptidos",
    "teicoplanina": "Glicopeptidos",
    "dalbavancina": "Glicopeptidos",
    "daptomicina": "Lipopeptidos",
    "daptomycin": "Lipopeptidos",
    "linezolid": "Oxazolidinonas",
    "azitromicina": "Macrolidos",
    "eritromicina": "Macrolidos",
    "clindamicina": "Lincosamidas",
    "clindamycin": "Lincosamidas",
    "colistina": "Polimixinas",
    "colistimetato de sodio": "Polimixinas",
    "fosfomicina": "Fosfomicinas",
    "fosfomicina trometamol": "Fosfomicinas",
    "nitrofurantoina": "Nitrofuranos",
    "tetraciclina": "Tetraciclinas",
    "doxiciclina": "Tetraciclinas",
    "minociclina": "Tetraciclinas",
    "tigeciclina": "Tetraciclinas",
    "trimetoprim": "Sulfamidas",
    "trimetroprim sulfametoxazol": "Sulfamidas",
    "sulfametoxazol trimetoprima": "Sulfamidas",
    "sulfametoxazol trimetoprim": "Sulfamidas",
    "rifampicina": "Rifamicinas",
    "metronidazol": "Nitroimidazoles",
    "cloranfenicol": "Anfenicoles",
    "acido fusidico": "Otros antibacterianos",
    "mupirocina": "Otros antibacterianos",
    "fluconazol": "Antifungicos",
    "voriconazol": "Antifungicos",
    "isavuconazol": "Antifungicos",
    "posaconazol": "Antifungicos",
    "caspofungin": "Antifungicos",
    "caspofungina": "Antifungicos",
    "micafungina sodica": "Antifungicos",
    "anfotericina b": "Antifungicos",
    "anfotericina b liposomas": "Antifungicos",
}


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    for token in ["/", "+", "(", ")", ",", "-", "_"]:
        text = text.replace(token, " ")
    return " ".join(text.lower().strip().split())


def antimicrobial_family(value: Any) -> str:
    normalized = normalize_text(value)
    return ANTIMICROBIAL_FAMILY_BY_NORMALIZED_NAME.get(normalized, "Other/Unmapped")


def add_change(
    log: TableLog,
    section: str,
    *,
    source: str | list[str],
    target: str | list[str],
    how: str,
    variable_type: str = "",
    n_classes: int | dict[str, int] | None = None,
    descriptions: dict[str, str] | None = None,
) -> None:
    getattr(log, section).append(
        VariableChange(
            kind=section,
            source=source,
            target=target,
            how=how,
            variable_type=variable_type,
            n_classes=n_classes,
            descriptions=descriptions or {},
        )
    )


def finalize_log(
    *,
    table_name: str,
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    merge_keys: Iterable[str],
) -> TableLog:
    return TableLog(
        table_name=table_name,
        input_rows=len(input_df),
        output_rows=len(output_df),
        input_columns=input_df.columns.tolist(),
        output_columns=output_df.columns.tolist(),
        merge_keys=list(merge_keys),
        columns_lost=sorted(set(input_df.columns) - set(output_df.columns)),
    )


def git_output(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


def run_metadata(run_label: str) -> dict[str, str]:
    tracked = [
        "bathecomb/preprocess_pipeline.py",
        "bathecomb/config/preprocess_columns_to_drop.txt",
        "bathecomb/config/preprocess_report.yml",
    ]
    dirty = git_output(["status", "--porcelain", "--", *tracked])
    return {
        "config_name": "bathecomb_first_pass",
        "config_version": "0.1.0",
        "config_path": "",
        "git_head_commit": git_output(["rev-parse", "HEAD"]),
        "preprocess_script_commit": git_output(
            ["log", "-1", "--format=%H", "--", "bathecomb/preprocess_pipeline.py"]
        ),
        "preprocess_script_blob": git_output(["hash-object", "bathecomb/preprocess_pipeline.py"]),
        "preprocess_code_dirty": "yes" if dirty else "no",
        "run_label": run_label,
    }


def attach_metadata(log: TableLog, metadata: dict[str, str]) -> None:
    log.metadata.update(metadata)


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
        dropped_sources = [change.source for change in log.dropped_variables]
        dropped_count = sum(
            len(source) if isinstance(source, list) else 1 for source in dropped_sources
        )
        summary_records.append(
            {
                "table_name": log.table_name,
                "input_rows": log.input_rows,
                "output_rows": log.output_rows,
                "input_column_count": len(log.input_columns),
                "output_column_count": len(log.output_columns),
                "columns_dropped_count": dropped_count,
                "columns_dropped": "; ".join(
                    ", ".join(source) if isinstance(source, list) else str(source)
                    for source in dropped_sources
                ),
                "columns_lost_count": len(columns_lost),
                "columns_lost": "; ".join(columns_lost),
                "merge_keys": ",".join(log.merge_keys),
                "warning_count": len(log.warnings),
                "note_count": len(log.notes),
                "validation_count": len(log.validation_checks),
                "config_name": log.metadata.get("config_name", ""),
                "config_version": log.metadata.get("config_version", ""),
                "git_head_commit": log.metadata.get("git_head_commit", ""),
                "preprocess_script_commit": log.metadata.get("preprocess_script_commit", ""),
                "preprocess_script_blob": log.metadata.get("preprocess_script_blob", ""),
                "preprocess_code_dirty": log.metadata.get("preprocess_code_dirty", ""),
                "run_label": log.metadata.get("run_label", ""),
            }
        )
        detail = asdict(log)
        detail.update(
            {
                "input_column_number": len(log.input_columns),
                "output_column_number": len(log.output_columns),
                "columns_dropped_count": dropped_count,
                "columns_dropped": dropped_sources,
                "columns_lost_count": len(columns_lost),
                "columns_lost": columns_lost,
            }
        )
        detailed_records.append(detail)
    summary = pd.DataFrame.from_records(summary_records)
    summary.to_csv(summary_output_path, index=False)
    detailed_output_path.write_text(
        json.dumps(detailed_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def read_drop_columns(path: Path, columns: list[str]) -> list[str]:
    if not path.exists():
        return []
    selected: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        entry = raw_line.strip()
        if not entry or entry.startswith("#"):
            continue
        matches = (
            sorted(fnmatch.filter(columns, entry))
            if any(char in entry for char in "*?[")
            else [entry]
        )
        selected.extend(column for column in matches if column in columns)
    return list(dict.fromkeys(selected))


def load_tables(db_path: Path) -> dict[str, pd.DataFrame]:
    with sqlite3.connect(db_path) as conn:
        table_names = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
            conn,
        )["name"].tolist()
        return {name: pd.read_sql_query(f'SELECT * FROM "{name}"', conn) for name in table_names}


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    text_columns = result.select_dtypes(include=["object"]).columns
    result[text_columns] = result[text_columns].replace(r"^\s*$", np.nan, regex=True)
    return result


def normalize_sex(value: Any) -> str | float:
    normalized = normalize_text(value)
    if normalized in {"hombre", "m", "male"}:
        return "M"
    if normalized in {"mujer", "f", "female"}:
        return "F"
    return np.nan


def first_notna(values: pd.Series) -> Any:
    valid = values.dropna()
    return valid.iloc[0] if not valid.empty else np.nan


def sorted_unique_list(values: pd.Series) -> list[str]:
    labels = sorted({str(value) for value in values.dropna() if str(value).strip()})
    return labels


def list_literal(values: Iterable[str]) -> str:
    return repr(sorted({str(value) for value in values if str(value).strip()}))


def dominant_label(values: pd.Series, negative: str = "NEGATIVE") -> str:
    labels = [str(value) for value in values.dropna() if str(value).strip()]
    if not labels:
        return negative
    return Counter(labels).most_common(1)[0][0]


def duplicate_key_groups(df: pd.DataFrame, keys: list[str]) -> int:
    if df.empty:
        return 0
    duplicated = df.duplicated(subset=keys, keep=False)
    return int(df.loc[duplicated, keys].drop_duplicates().shape[0])


def validate_result(result: PreprocessResult) -> PreprocessResult:
    missing = [key for key in result.log.merge_keys if key not in result.df.columns]
    if missing:
        raise ValueError(f"{result.log.table_name} missing merge keys: {missing}")
    duplicate_groups = duplicate_key_groups(result.df, result.log.merge_keys)
    result.log.validation_checks.append(f"duplicate_merge_key_groups:{duplicate_groups}")
    if duplicate_groups:
        result.log.warnings.append(
            f"{duplicate_groups} duplicated groups for merge keys {result.log.merge_keys}"
        )
    result.log.validation_checks.append(f"row_count:{len(result.df)}")
    return result


def aggregate_binary_max(df: pd.DataFrame, keys: list[str], columns: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in df.columns]
    if not available:
        return df[keys].drop_duplicates()
    values = df[keys + available].copy()
    for column in available:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    return values.groupby(keys, as_index=False).max()


def preprocess_paciente(tables: dict[str, pd.DataFrame], metadata: dict[str, str]) -> PreprocessResult:
    source = clean_missing_values(tables["paciente"])
    df = source.copy()
    df["sexo"] = df["sexo"].map(normalize_sex)
    df["fecha_nacimiento"] = pd.to_datetime(df["fecha_nacimiento"], errors="coerce")
    log = finalize_log(table_name="paciente", input_df=source, output_df=df, merge_keys=["record_id"])
    attach_metadata(log, metadata)
    add_change(
        log,
        "recoded_variables",
        source="sexo",
        target="sexo",
        how="normalize Hombre/Mujer/M/F values to M/F",
    )
    return validate_result(PreprocessResult(df=df, log=log))


def preprocess_episodio_ingreso(
    tables: dict[str, pd.DataFrame], metadata: dict[str, str]
) -> PreprocessResult:
    source = clean_missing_values(tables["episodio_ingreso"])
    df = source.copy()
    for column in ["fecha_ingreso", "fecha_alta", "fecha_hemocultivo", "fecha_mortalidad"]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    binary_columns = [
        "foco_controlable",
        "foco_controlado",
        "mortalidad",
        "mortalidad_30_dias",
        "uci_por_el_episodio",
        "en_uci_antes_del_hemocultivo",
        "mujer_gestante",
        "paciente_residencia",
    ]
    numeric_columns = [
        "dias_hemocultivo_mortalidad",
        "duracion_UCI",
        "dias_hemocultivo_ingresoUCI",
        "dias_hemocultivo_salidaUCI",
    ]
    categorical_columns = [
        "fecha_alta",
        "fecha_hemocultivo",
        "organo_aparato",
        "control_foco",
        "IRAs_nosocomial",
        "fecha_mortalidad",
        "codigo_postal",
    ]
    agg: dict[str, str] = {}
    agg.update({column: "max" for column in binary_columns if column in df.columns})
    agg.update({column: "max" for column in numeric_columns if column in df.columns})
    agg.update({column: first_notna for column in categorical_columns if column in df.columns})
    result = df.groupby(ADMISSION_KEYS, as_index=False, dropna=False).agg(agg)
    result["admission_id"] = (
        result["record_id"].astype("Int64").astype(str)
        + "_"
        + pd.to_datetime(result["fecha_ingreso"]).dt.strftime("%Y%m%d")
    )
    log = finalize_log(
        table_name="episodio_ingreso",
        input_df=source,
        output_df=result,
        merge_keys=ADMISSION_KEYS,
    )
    attach_metadata(log, metadata)
    add_change(
        log,
        "transformed_variables",
        source=source.columns.tolist(),
        target=result.columns.tolist(),
        how="aggregate duplicate admission rows to one row per record_id and fecha_ingreso",
    )
    log.notes.append("Admission level is defined as record_id + fecha_ingreso.")
    return validate_result(PreprocessResult(df=result, log=log))


def preprocess_simple_admission_table(
    tables: dict[str, pd.DataFrame],
    table_name: str,
    metadata: dict[str, str],
    *,
    binary_columns: list[str] | None = None,
    numeric_max_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
) -> PreprocessResult:
    source = clean_missing_values(tables[table_name])
    df = source.copy()
    df["fecha_ingreso"] = pd.to_datetime(df["fecha_ingreso"], errors="coerce")
    if "fecha_hemocultivo" in df.columns:
        df["fecha_hemocultivo"] = pd.to_datetime(df["fecha_hemocultivo"], errors="coerce")
    agg: dict[str, Any] = {}
    for column in binary_columns or []:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            agg[column] = "max"
    for column in numeric_max_columns or []:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            agg[column] = "max"
    for column in categorical_columns or []:
        if column in df.columns:
            agg[column] = first_notna
    result = df.groupby(ADMISSION_KEYS, as_index=False, dropna=False).agg(agg)
    log = finalize_log(table_name=table_name, input_df=source, output_df=result, merge_keys=ADMISSION_KEYS)
    attach_metadata(log, metadata)
    add_change(
        log,
        "transformed_variables",
        source=source.columns.tolist(),
        target=result.columns.tolist(),
        how="aggregate to admission level using max for flags/numeric severity and first non-missing for categorical fields",
    )
    return validate_result(PreprocessResult(df=result, log=log))


def preprocess_laboratorio(tables: dict[str, pd.DataFrame], metadata: dict[str, str]) -> PreprocessResult:
    source = clean_missing_values(tables["laboratorio"])
    df = source.copy()
    df["fecha_ingreso"] = pd.to_datetime(df["fecha_ingreso"], errors="coerce")
    lab_columns = [
        column
        for column in df.columns
        if column not in {"record_id", "fecha_ingreso", "fecha_hemocultivo"}
    ]
    for column in lab_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    grouped = df.groupby(ADMISSION_KEYS, as_index=False, dropna=False)
    result = grouped[lab_columns].agg(["mean", "max"])
    result.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else str(column)
        for column in result.columns
    ]
    result = result.reset_index()
    result["laboratorio_n_rows"] = grouped.size()["size"].to_numpy()
    log = finalize_log(table_name="laboratorio", input_df=source, output_df=result, merge_keys=ADMISSION_KEYS)
    attach_metadata(log, metadata)
    add_change(
        log,
        "transformed_variables",
        source=lab_columns,
        target=[column for column in result.columns if column not in ADMISSION_KEYS],
        how="aggregate repeated laboratory rows per admission using mean and max",
    )
    return validate_result(PreprocessResult(df=result, log=log))


def preprocess_episodio_uci(tables: dict[str, pd.DataFrame], metadata: dict[str, str]) -> PreprocessResult:
    source = clean_missing_values(tables["episodio_uci"])
    df = source.copy()
    for column in ["fecha_ingreso", "fecha_ingreso_UCI", "fecha_hemocultivo"]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    grouped = df.groupby(ADMISSION_KEYS, as_index=False, dropna=False)
    result = grouped.agg(
        fecha_ingreso_UCI=("fecha_ingreso_UCI", "min"),
        episodio_uci_n_rows=("record_id", "size"),
    )
    result["has_uci_record"] = 1
    log = finalize_log(table_name="episodio_uci", input_df=source, output_df=result, merge_keys=ADMISSION_KEYS)
    attach_metadata(log, metadata)
    add_change(
        log,
        "created_variables",
        source=["fecha_ingreso_UCI"],
        target=["fecha_ingreso_UCI", "episodio_uci_n_rows", "has_uci_record"],
        how="collapse ICU rows to admission level and flag admissions with ICU records",
    )
    return validate_result(PreprocessResult(df=result, log=log))


def preprocess_episodio_infeccion(
    tables: dict[str, pd.DataFrame], metadata: dict[str, str]
) -> PreprocessResult:
    source = clean_missing_values(tables["episodio_infeccion"])
    df = source.copy()
    df["fecha_ingreso"] = pd.to_datetime(df["fecha_ingreso"], errors="coerce")
    df["fecha_cultivo"] = pd.to_datetime(df["fecha_cultivo"], errors="coerce")
    df["days_culture_from_admission"] = (
        df["fecha_cultivo"] - df["fecha_ingreso"]
    ).dt.days
    df["infection_before_admission"] = (df["days_culture_from_admission"] < 0).astype(int)
    df["infection_same_day_admission"] = (df["days_culture_from_admission"] == 0).astype(int)
    df["infection_after_admission"] = (df["days_culture_from_admission"] > 0).astype(int)
    df["blood_culture"] = (df["especimen"].fillna("") == "Sangre").astype(int)
    df["resistance_mechanism"] = df["fenotipo_resistencia"].fillna("NEGATIVE")
    df.loc[df["resistance_mechanism"].eq(""), "resistance_mechanism"] = "NEGATIVE"

    def mechanism_target(values: pd.Series) -> str:
        labels = sorted({str(value) for value in values.dropna() if str(value) != "NEGATIVE"})
        if not labels:
            return "NEGATIVE"
        carbapenemase = {"KPC", "VIM", "OXA-48", "IMP"}
        if any(label in carbapenemase for label in labels):
            return "CARBAPENEMASE"
        if "CARB" in labels:
            return "CARB"
        if "MRSA" in labels:
            return "MRSA"
        if "BLEE" in labels:
            return "BLEE"
        if "MR" in labels:
            return "MR"
        return "OTHER_RESISTANCE"

    grouped = df.groupby(ADMISSION_KEYS, as_index=False, dropna=False)
    result = grouped.agg(
        infection_episode_count=("episode_id", "nunique"),
        organism_count=("microorganismo", lambda values: len(set(values.dropna()))),
        organism_list=("microorganismo", lambda values: list_literal(sorted_unique_list(values))),
        dominant_microorganism=("microorganismo", dominant_label),
        blood_culture_episode_count=("blood_culture", "sum"),
        resistance_mechanism_labels=(
            "resistance_mechanism",
            lambda values: list_literal(label for label in values if label != "NEGATIVE"),
        ),
        resistance_mechanism_target=("resistance_mechanism", mechanism_target),
        fecha_cultivo_min=("fecha_cultivo", "min"),
        fecha_cultivo_max=("fecha_cultivo", "max"),
        days_first_culture_from_admission=("days_culture_from_admission", "min"),
        days_last_culture_from_admission=("days_culture_from_admission", "max"),
        infection_before_admission_count=("infection_before_admission", "sum"),
        infection_same_day_admission_count=("infection_same_day_admission", "sum"),
        infection_after_admission_count=("infection_after_admission", "sum"),
    )
    log = finalize_log(
        table_name="episodio_infeccion",
        input_df=source,
        output_df=result,
        merge_keys=ADMISSION_KEYS,
    )
    attach_metadata(log, metadata)
    add_change(
        log,
        "created_variables",
        source=["fecha_cultivo", "microorganismo", "fenotipo_resistencia", "especimen"],
        target=[
            "dominant_microorganism",
            "resistance_mechanism_target",
            "infection_*_admission_count",
            "blood_culture_episode_count",
        ],
        how="aggregate infection episodes to admission level and derive timing, organism, and coarse resistance mechanism targets",
        variable_type="target",
    )
    return validate_result(PreprocessResult(df=result, log=log))


def preprocess_antibiograma(
    tables: dict[str, pd.DataFrame], metadata: dict[str, str]
) -> PreprocessResult:
    source = clean_missing_values(tables["antibiograma"])
    infections = tables["episodio_infeccion"][["episode_id", "record_id", "fecha_ingreso"]].copy()
    infections["fecha_ingreso"] = pd.to_datetime(infections["fecha_ingreso"], errors="coerce")
    df = source.merge(infections, on="episode_id", how="left")
    df["family"] = df["antimicrobiano"].map(antimicrobial_family)
    df["is_resistant"] = df["interpretacion"].eq("R").astype(int)
    df["is_intermediate"] = df["interpretacion"].eq("I").astype(int)
    df["is_susceptible"] = df["interpretacion"].eq("S").astype(int)
    grouped = df.groupby(ADMISSION_KEYS, as_index=False, dropna=False)
    result = grouped.agg(
        antibiogram_rows=("episode_id", "size"),
        antibiogram_episode_count=("episode_id", "nunique"),
        antibiogram_resistant_rows=("is_resistant", "sum"),
        antibiogram_intermediate_rows=("is_intermediate", "sum"),
        antibiogram_susceptible_rows=("is_susceptible", "sum"),
        antibiogram_resistant_drugs=(
            "antimicrobiano",
            lambda values: list_literal(values[df.loc[values.index, "is_resistant"].eq(1)]),
        ),
        antibiogram_resistant_families=(
            "family",
            lambda values: list_literal(values[df.loc[values.index, "is_resistant"].eq(1)]),
        ),
    )
    result["resistente_cefalosporina"] = result["antibiogram_resistant_families"].map(
        lambda value: (
            "RESIST_CEFALOSPORINAS_3a_4a"
            if "Cefalosporinas 3/4 gen" in value
            else "NEGATIVE"
        )
    )
    log = finalize_log(table_name="antibiograma", input_df=source, output_df=result, merge_keys=ADMISSION_KEYS)
    attach_metadata(log, metadata)
    add_change(
        log,
        "created_variables",
        source=["interpretacion", "antimicrobiano"],
        target=["antibiogram_resistant_families", "resistente_cefalosporina"],
        how="join antibiogram rows to infection admissions, map resistant antimicrobials to families, and derive cephalosporin resistance target",
        variable_type="target",
    )
    missing_admission = int(df["record_id"].isna().sum())
    log.validation_checks.append(f"antibiogram_rows_missing_infection_episode:{missing_admission}")
    return validate_result(PreprocessResult(df=result, log=log))


def preprocess_tto_antimicrobiano(
    tables: dict[str, pd.DataFrame], metadata: dict[str, str]
) -> PreprocessResult:
    source = clean_missing_values(tables["tto_antimicrobiano"])
    infections = tables["episodio_infeccion"][["episode_id", "record_id", "fecha_ingreso"]].copy()
    infections["fecha_ingreso"] = pd.to_datetime(infections["fecha_ingreso"], errors="coerce")
    df = source.merge(infections, on="episode_id", how="left")
    df["dias_tratamiento"] = pd.to_numeric(df["dias_tratamiento"], errors="coerce")
    df["treatment_family"] = df["antimicrobiano"].map(antimicrobial_family)
    df["tratamiento_apropiado"] = pd.to_numeric(df["tratamiento_apropiado"], errors="coerce")
    grouped = df.groupby(ADMISSION_KEYS, as_index=False, dropna=False)
    result = grouped.agg(
        treatment_rows=("episode_id", "size"),
        treatment_episode_count=("episode_id", "nunique"),
        treatment_drug_list=("antimicrobiano", lambda values: list_literal(sorted_unique_list(values))),
        treatment_family_count=("treatment_family", lambda values: len(set(values.dropna()))),
        treatment_days_max=("dias_tratamiento", "max"),
        treatment_days_sum=("dias_tratamiento", "sum"),
        treatment_appropriate_any=("tratamiento_apropiado", "max"),
    )
    log = finalize_log(
        table_name="tto_antimicrobiano",
        input_df=source,
        output_df=result,
        merge_keys=ADMISSION_KEYS,
    )
    attach_metadata(log, metadata)
    add_change(
        log,
        "transformed_variables",
        source=["antimicrobiano", "dias_tratamiento", "tratamiento_apropiado"],
        target=[column for column in result.columns if column not in ADMISSION_KEYS],
        how="join treatment rows to infection admissions and aggregate treatment exposure to admission level",
    )
    negative_days = int((df["dias_tratamiento"] < 0).sum())
    long_days = int((df["dias_tratamiento"] > 365).sum())
    log.warnings.append(
        f"dias_tratamiento requires review: negative_rows={negative_days}, rows_over_365_days={long_days}"
    )
    return validate_result(PreprocessResult(df=result, log=log))


def merge_results(results: dict[str, PreprocessResult], metadata: dict[str, str]) -> PreprocessResult:
    base = results["episodio_ingreso"].df.copy()
    input_df = base.copy()
    for name, result in results.items():
        if name == "episodio_ingreso":
            continue
        if result.log.merge_keys == ["record_id"]:
            base = base.merge(result.df, on="record_id", how="left")
        else:
            base = base.merge(result.df, on=ADMISSION_KEYS, how="left")
    base["fecha_ingreso"] = pd.to_datetime(base["fecha_ingreso"], errors="coerce")
    if "fecha_nacimiento" in base.columns:
        base["age_at_admission"] = (
            (base["fecha_ingreso"] - base["fecha_nacimiento"]).dt.days / 365.25
        ).round(1)
    count_defaults = [
        column
        for column in base.columns
        if column.endswith("_count") or column.endswith("_rows") or column.endswith("_n_rows")
    ]
    base[count_defaults] = base[count_defaults].fillna(0)
    for column in ["has_uci_record"]:
        if column in base.columns:
            base[column] = base[column].fillna(0)
    for column in [
        "dominant_microorganism",
        "resistance_mechanism_target",
        "resistente_cefalosporina",
    ]:
        if column in base.columns:
            base[column] = base[column].fillna("NEGATIVE")
    for column in [
        "organism_list",
        "resistance_mechanism_labels",
        "antibiogram_resistant_drugs",
        "antibiogram_resistant_families",
        "treatment_drug_list",
    ]:
        if column in base.columns:
            base[column] = base[column].fillna("[]")
    log = finalize_log(
        table_name="merged_dataset",
        input_df=input_df,
        output_df=base,
        merge_keys=ADMISSION_KEYS,
    )
    attach_metadata(log, metadata)
    add_change(
        log,
        "created_variables",
        source=["fecha_ingreso", "fecha_nacimiento"],
        target="age_at_admission",
        how="calculate patient age at each admission date",
    )
    add_change(
        log,
        "transformed_variables",
        source=list(results.keys()),
        target=base.columns.tolist(),
        how="left-join all admission-level tables to episodio_ingreso base",
    )
    return validate_result(PreprocessResult(df=base, log=log))


def export_dataset_outputs(
    df: pd.DataFrame, output_path: Path, drop_columns_path: Path
) -> tuple[pd.DataFrame, list[str], Path]:
    serializable = df.copy()
    for column in serializable.select_dtypes(include=["datetime64[ns]"]).columns:
        serializable[column] = serializable[column].dt.strftime("%Y-%m-%d")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable.to_csv(output_path, index=False)
    drop_columns = read_drop_columns(drop_columns_path, serializable.columns.tolist())
    filtered = serializable.drop(columns=drop_columns, errors="ignore")
    filtered_path = output_path.with_name(f"{output_path.stem}_filtered.csv")
    filtered.to_csv(filtered_path, index=False)
    return filtered, drop_columns, filtered_path


def build_run_log(
    input_path: Path,
    output_path: Path,
    drop_columns_path: Path,
    metadata: dict[str, str],
) -> TableLog:
    log = TableLog(
        table_name="_pipeline_run",
        input_rows=0,
        output_rows=0,
        input_columns=[],
        output_columns=[],
        merge_keys=[],
    )
    attach_metadata(log, metadata)
    log.notes.extend(
        [
            f"input_file_path={input_path}",
            f"output_file_path={output_path}",
            f"drop_columns_path={drop_columns_path}",
            "aggregation_level=record_id + fecha_ingreso",
        ]
    )
    return log


def build_filtered_dataset_log(
    full_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    dropped_columns: list[str],
    output_path: Path,
    filtered_path: Path,
    drop_columns_path: Path,
    metadata: dict[str, str],
) -> TableLog:
    log = finalize_log(
        table_name="filtered_dataset",
        input_df=full_df,
        output_df=filtered_df,
        merge_keys=[key for key in ADMISSION_KEYS if key in filtered_df.columns],
    )
    attach_metadata(log, metadata)
    add_change(
        log,
        "dropped_variables",
        source=dropped_columns,
        target=filtered_df.columns.tolist(),
        how=f"drop columns matching explicit names or glob patterns from {drop_columns_path}",
    )
    log.notes.extend([f"full_output_path={output_path}", f"filtered_output_path={filtered_path}"])
    log.validation_checks.append(f"filtered_columns_dropped:{len(dropped_columns)}")
    return log


def run_pipeline(
    input_path: Path = DEFAULT_DB_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    drop_columns_path: Path = DEFAULT_DROP_COLUMNS_PATH,
    run_label: str = "",
) -> PipelineArtifacts:
    metadata = run_metadata(run_label)
    tables = load_tables(input_path)
    results: dict[str, PreprocessResult] = {}
    results["episodio_ingreso"] = preprocess_episodio_ingreso(tables, metadata)
    results["paciente"] = preprocess_paciente(tables, metadata)
    results["comorbilidad"] = preprocess_simple_admission_table(
        tables,
        "comorbilidad",
        metadata,
        binary_columns=[
            "infarto",
            "insuficiencia_cardiaca",
            "evp",
            "e_cerebrovascular",
            "demencia",
            "e_pulmonar_cronica",
            "ulcera_peptica",
            "colagenopatia",
            "hemiplejia",
            "erc",
            "neoplasia_tratamiento_activo",
            "neoplasia_solida_metastasica",
            "neoplasia_solida_no_metastasica",
            "linfoma",
            "leucemia",
            "sida",
            "hepatopatia_ligera",
            "hepatopatia_moderada_o_grave",
            "diabetes",
            "diabetes_sin_lesion_organo_diana",
            "diabetes_con_lesion_organo_diana",
            "inmunosupresion",
            "TOS",
            "TPH",
            "gran_quemado",
        ],
        categorical_columns=[
            "tipo_cancer",
            "puntaje_child_pugh",
            "tipo_hepatopatia",
            "causa_inmunosupresion",
            "clasificacion_quemadura",
        ],
    )
    results["factores_riesgo_infeccion_bmr"] = preprocess_simple_admission_table(
        tables,
        "factores_riesgo_infeccion_bmr",
        metadata,
        binary_columns=[
            "hospit_ano_previo",
            "hospit_mes_previo",
            "hospit_ano_previo_uci",
            "cirugia_previa_sin_implante",
            "cirugia_previa_con_implante",
            "asistencia_sanitaria_prev",
            "hemodialisis_permanente",
            "dialisis_peritoneal",
            "cateter_venoso",
            "sonda_urinaria",
            "sonda_nasogastrica",
            "derivacion_ventriculoper",
            "valvula_prot_cardiaca",
            "portador_otros_disposit",
        ],
    )
    results["signos_sintomas"] = preprocess_simple_admission_table(
        tables,
        "signos_sintomas",
        metadata,
        binary_columns=[
            "sepsis",
            "shock_septico",
            "fiebre",
            "tos",
            "dificultad_respirar",
            "dolor_costal",
            "disuria",
            "polaquiuria",
            "tenesmo_vejiga",
            "tenesmo_ano_recto",
            "dolor_fosa_renal",
            "nauseas",
            "vomitos",
            "dolor_abdominal",
            "diarrea",
            "lesiones_piel",
            "lesiones_mucosas",
            "cefalea",
            "dolores_articulares",
        ],
        numeric_max_columns=[
            "qsofa",
            "duracion_sintoma",
            "indice_de_charlson",
            "escala_karnofsky",
            "barthel_inf_90",
            "temperatura",
            "frec_cardiaca",
            "frecuencia_respiratoria",
            "tension_arterial_sist",
            "tension_arterial_diast",
            "saturacion_pO2",
        ],
        categorical_columns=["somnolencia_estupor_coma", "situacion_funcional_basal"],
    )
    results["laboratorio"] = preprocess_laboratorio(tables, metadata)
    results["episodio_uci"] = preprocess_episodio_uci(tables, metadata)
    results["episodio_infeccion"] = preprocess_episodio_infeccion(tables, metadata)
    results["antibiograma"] = preprocess_antibiograma(tables, metadata)
    results["tto_antimicrobiano"] = preprocess_tto_antimicrobiano(tables, metadata)
    merged = merge_results(results, metadata)

    filtered_df, dropped_columns, filtered_path = export_dataset_outputs(
        merged.df,
        output_path,
        drop_columns_path,
    )
    logs = [
        build_run_log(input_path, output_path, drop_columns_path, metadata),
        *[result.log for result in results.values()],
        merged.log,
        build_filtered_dataset_log(
            merged.df,
            filtered_df,
            dropped_columns,
            output_path,
            filtered_path,
            drop_columns_path,
            metadata,
        ),
    ]
    export_logs(
        logs,
        summary_output_path=output_path.with_name(f"{output_path.stem}_log_summary.csv"),
        detailed_output_path=output_path.with_name(f"{output_path.stem}_log_detailed.json"),
    )
    return PipelineArtifacts(
        tables={name: result.df for name, result in results.items()}
        | {"final": merged.df, "filtered": filtered_df},
        logs=logs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess BATHECOMB SQLite data into an admission-level dataset."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--drop-columns-path", type=Path, default=DEFAULT_DROP_COLUMNS_PATH)
    parser.add_argument("--run-label", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        input_path=args.input_path,
        output_path=args.output_path,
        drop_columns_path=args.drop_columns_path,
        run_label=args.run_label,
    )


if __name__ == "__main__":
    main()
