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
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "playground" / "preprocess_bacthecom_mortality.csv"
DEFAULT_DROP_COLUMNS_PATH = ROOT_DIR / "config" / "preprocess_columns_to_drop.txt"
# Final ML row definition: one row per patient admission and the selected blood-culture date.
# The selected blood culture is the earliest fecha_hemocultivo on/after admission,
# allowing a 2-day pre-admission buffer.
BASE_ADMISSION_KEYS = ["record_id", "fecha_ingreso"]
EPISODE_KEYS = ["record_id", "fecha_ingreso", "fecha_hemocultivo"]
ADMISSION_KEYS = EPISODE_KEYS
HEMOCULTURE_PRE_ADMISSION_BUFFER_DAYS = 2


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



def classify_microorganism(value: Any) -> str:
    """Classify raw microorganism names into the BactHeCom modelling groups.

    This mirrors the original exploratory preprocessing but keeps it local to this
    script so the pipeline does not depend on notebook-only utils.py.
    """
    text = normalize_text(value)
    if not text:
        return "Other"
    if "escherichia coli" in text or text.startswith("e coli"):
        return "E_coli"
    if "klebsiella pneumoniae" in text or text.startswith("k pneumoniae"):
        return "K_pneumoniae"
    if "pseudomonas aeruginosa" in text or text.startswith("p aeruginosa"):
        return "P_aeruginosa"
    if "staphylococcus aureus" in text or text.startswith("s aureus"):
        return "S_aureus"
    if "enterococcus" in text:
        return "Enterococcus"
    enterobacterales_tokens = [
        "enterobacter", "serratia", "proteus", "citrobacter", "morganella",
        "providencia", "raoultella", "hafnia", "kluyvera", "salmonella",
        "shigella", "yersinia", "klebsiella", "escherichia",
    ]
    if any(token in text for token in enterobacterales_tokens):
        return "Enterobacterias"
    return "Other"


def microorganism_species_label(value: Any) -> str:
    """Return first two tokens joined by underscore, as in the original notebook."""
    text = str(value).strip() if not pd.isna(value) else ""
    if not text:
        return ""
    return "_".join(text.split()[:2])


def safe_set_literal(values: Iterable[Any]) -> str:
    return repr(sorted({str(v) for v in values if not pd.isna(v) and str(v).strip()}))


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
        "bacthecom/preprocess_pipeline.py",
        "bacthecom/config/preprocess_columns_to_drop.txt",
        "bacthecom/config/preprocess_report.yml",
    ]
    dirty = git_output(["status", "--porcelain", "--", *tracked])
    return {
        "config_name": "bacthecom_mortality_episode_level",
        "config_version": "0.1.0",
        "config_path": "",
        "git_head_commit": git_output(["rev-parse", "HEAD"]),
        "preprocess_script_commit": git_output(
            ["log", "-1", "--format=%H", "--", "bacthecom/preprocess_pipeline.py"]
        ),
        "preprocess_script_blob": git_output(["hash-object", "bacthecom/preprocess_pipeline.py"]),
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


def normalize_sex(value: Any) -> int | float:
    """Encode sex for modelling: M=1, F=0."""
    normalized = normalize_text(value)
    if normalized in {"hombre", "m", "male"}:
        return 1
    if normalized in {"mujer", "f", "female"}:
        return 0
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


def keep_first_hemoculture_with_buffer(
    df: pd.DataFrame,
    *,
    buffer_days: int = HEMOCULTURE_PRE_ADMISSION_BUFFER_DAYS,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return one selected hemoculture row per record_id + fecha_ingreso.

    Selection rule:
    - eligible hemocultures satisfy fecha_hemocultivo >= fecha_ingreso - buffer_days
    - among eligible hemocultures, keep the earliest fecha_hemocultivo

    Rows without fecha_ingreso or fecha_hemocultivo cannot define the requested ML
    episode key and are excluded from the selected base cohort.
    """
    if df.empty:
        return df.copy(), {
            "input_rows": 0,
            "rows_missing_required_dates": 0,
            "rows_before_buffer": 0,
            "selected_rows": 0,
            "selected_admissions": 0,
        }

    work = df.copy()
    work["fecha_ingreso"] = pd.to_datetime(work["fecha_ingreso"], errors="coerce")
    work["fecha_hemocultivo"] = pd.to_datetime(work["fecha_hemocultivo"], errors="coerce")

    required_ok = work["record_id"].notna() & work["fecha_ingreso"].notna() & work["fecha_hemocultivo"].notna()
    rows_missing_required_dates = int((~required_ok).sum())
    work = work.loc[required_ok].copy()

    lower_bound = work["fecha_ingreso"] - pd.to_timedelta(buffer_days, unit="D")
    eligible = work["fecha_hemocultivo"] >= lower_bound
    rows_before_buffer = int((~eligible).sum())
    work = work.loc[eligible].copy()

    work["days_hemoculture_from_admission"] = (
        work["fecha_hemocultivo"] - work["fecha_ingreso"]
    ).dt.days

    # Stable deterministic choice: earliest hemoculture; ties resolved by original row order.
    work["__original_order"] = np.arange(len(work))
    work = work.sort_values(
        ["record_id", "fecha_ingreso", "fecha_hemocultivo", "__original_order"]
    )
    selected = work.drop_duplicates(subset=BASE_ADMISSION_KEYS, keep="first").drop(columns="__original_order")

    stats = {
        "input_rows": int(len(df)),
        "rows_missing_required_dates": rows_missing_required_dates,
        "rows_before_buffer": rows_before_buffer,
        "selected_rows": int(len(selected)),
        "selected_admissions": int(selected[BASE_ADMISSION_KEYS].drop_duplicates().shape[0]),
    }
    return selected, stats


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


def normalize_merge_key_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure merge keys have identical dtypes in all preprocessed tables."""
    df = df.copy()
    if "record_id" in df.columns:
        df["record_id"] = pd.to_numeric(df["record_id"], errors="coerce").astype("Int64")
    for column in ["fecha_ingreso", "fecha_hemocultivo"]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce").dt.normalize()
    return df


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
        how="normalize Hombre/Mujer/M/F values to binary sexo: M=1, F=0",
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

    # Original notebook logic: when duplicate admission rows exist, keep the row
    # with the largest observed days-to-mortality or days-to-ICU signal.
    if {"dias_hemocultivo_mortalidad", "dias_hemocultivo_ingresoUCI"}.issubset(df.columns):
        df["priority_days"] = np.where(
            df["dias_hemocultivo_mortalidad"].notna(),
            df["dias_hemocultivo_mortalidad"],
            df["dias_hemocultivo_ingresoUCI"],
        )
        df = (
            df.sort_values(["record_id", "fecha_ingreso", "priority_days"], ascending=[True, True, False])
            .drop_duplicates(subset=BASE_ADMISSION_KEYS, keep="first")
            .drop(columns="priority_days")
        )

    # Enforce one ML episode per admission: choose the first hemoculture since admission,
    # allowing a 2-day pre-admission buffer.
    df, selection_stats = keep_first_hemoculture_with_buffer(
        df,
        buffer_days=HEMOCULTURE_PRE_ADMISSION_BUFFER_DAYS,
    )

    # Original notebook target recode: mortality within 14 days from admission.
    if {"fecha_mortalidad", "fecha_ingreso"}.issubset(df.columns):
        delta = df["fecha_mortalidad"] - df["fecha_ingreso"]
        df["mortalidad_14_dias"] = np.where(delta <= pd.Timedelta(days=14), 1, 0)
        df.loc[df["fecha_mortalidad"].isna(), "mortalidad_14_dias"] = 0

    if "en_uci_antes_del_hemocultivo" in df.columns:
        df["en_uci_antes_del_hemocultivo"] = df["en_uci_antes_del_hemocultivo"].fillna(0)
    if "IRAs_nosocomial" in df.columns:
        df["IRAs_nosocomial"] = df["IRAs_nosocomial"].map({"No": 0, "Si": 1, "NO": 0, "SI": 1, "no": 0, "si": 1})

    # Keep infection focus as model-ready dummy variables. The raw text column is
    # renamed to foco, one-hot encoded after aggregation, and then dropped.
    if "organo_aparato" in df.columns:
        df["foco"] = df["organo_aparato"].map(lambda v: normalize_text(v).replace(" ", "_") if not pd.isna(v) else np.nan)

    binary_columns = [
        "foco_controlable",
        "foco_controlado",
        "mortalidad",
        "mortalidad_30_dias",
        "mortalidad_14_dias",
        "uci_por_el_episodio",
        "en_uci_antes_del_hemocultivo",
        "mujer_gestante",
        "paciente_residencia",
        "IRAs_nosocomial",
    ]
    numeric_columns = [
        "dias_hemocultivo_mortalidad",
        "duracion_UCI",
        "dias_hemocultivo_ingresoUCI",
        "dias_hemocultivo_salidaUCI",
        "days_hemoculture_from_admission",
    ]
    categorical_columns = [
        "fecha_alta",
        "control_foco",
        "fecha_mortalidad",
        "foco",
    ]
    agg: dict[str, Any] = {}
    agg.update({column: "max" for column in binary_columns if column in df.columns})
    agg.update({column: "max" for column in numeric_columns if column in df.columns})
    agg.update({column: first_notna for column in categorical_columns if column in df.columns})

    result = df.groupby(ADMISSION_KEYS, as_index=False, dropna=False).agg(agg)
    if "foco" in result.columns:
        foco_dummies = pd.get_dummies(result["foco"], prefix="foco", dtype=int)
        result = pd.concat([result.drop(columns=["foco"]), foco_dummies], axis=1)

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
        how=(
            "deduplicate admissions using priority_days as in original notebook; select earliest "
            f"fecha_hemocultivo per record_id + fecha_ingreso using >= fecha_ingreso - {HEMOCULTURE_PRE_ADMISSION_BUFFER_DAYS} days buffer; "
            "derive mortalidad_14_dias and recode IRAs_nosocomial"
        ),
    )
    add_change(
        log,
        "dropped_variables",
        source=["codigo_postal", "organo_aparato", "foco", "admission_id", "episode_key"],
        target=result.columns.tolist(),
        how="drop non-modelling identifiers/location and organ-system text fields from episode table",
    )
    log.metadata.update({f"hemoculture_selection_{k}": v for k, v in selection_stats.items()})
    log.notes.append(
        "ML unit is one row per record_id + fecha_ingreso + selected fecha_hemocultivo; "
        f"selection allows {HEMOCULTURE_PRE_ADMISSION_BUFFER_DAYS} pre-admission days."
    )
    log.validation_checks.append(
        "selected_hemoculture_duplicate_admissions:"
        f"{duplicate_key_groups(result, BASE_ADMISSION_KEYS)}"
    )
    return validate_result(PreprocessResult(df=result, log=log))


def preprocess_comorbilidad(tables: dict[str, pd.DataFrame], metadata: dict[str, str]) -> PreprocessResult:
    """Process comorbidity table using the original notebook recodes.

    Adds has_cancer and num_comorbilidades, fills hepatopathy indicators with 0,
    and drops free-text/detail variables not used for modelling.
    """
    source = clean_missing_values(tables["comorbilidad"])
    df = source.copy()
    df = normalize_merge_key_dtypes(df)

    dropped_detail_columns = [
        "tipo_hepatopatia",
        "causa_inmunosupresion",
        "tipo_cancer",
        "fecha_TOS",
        "fecha_TPH",
        "clasificacion_quemadura",
        "puntaje_child_pugh",
    ]

    for col in ["hepatopatia_ligera", "hepatopatia_moderada_o_grave"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    cancer_cols = [
        "neoplasia_tratamiento_activo",
        "neoplasia_solida_metastasica",
        "neoplasia_solida_no_metastasica",
    ]
    for col in cancer_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    available_cancer_cols = [c for c in cancer_cols if c in df.columns]
    if available_cancer_cols:
        df["has_cancer"] = (df[available_cancer_cols].sum(axis=1) > 0).astype(int)

    df = df.drop(columns=[c for c in dropped_detail_columns if c in df.columns], errors="ignore")

    binary_cols = [
        c for c in df.columns
        if c not in ADMISSION_KEYS and c not in {"fecha_hemocultivo"}
        and c not in {"has_cancer"}
        and not c.startswith("fecha_")
    ]
    for col in binary_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    comorbidity_cols = [
        c for c in binary_cols
        if c not in {"record_id", "fecha_ingreso", "fecha_hemocultivo"}
    ]
    if comorbidity_cols:
        df["num_comorbilidades"] = df[comorbidity_cols].fillna(0).sum(axis=1)

    agg: dict[str, Any] = {}
    for col in df.columns:
        if col in ADMISSION_KEYS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            agg[col] = "max"
        else:
            agg[col] = first_notna

    result = df.groupby(ADMISSION_KEYS, as_index=False, dropna=False).agg(agg)
    log = finalize_log(table_name="comorbilidad", input_df=source, output_df=result, merge_keys=ADMISSION_KEYS)
    attach_metadata(log, metadata)
    add_change(
        log,
        "created_variables",
        source=cancer_cols,
        target=["has_cancer", "num_comorbilidades"],
        how="create has_cancer and count total comorbidities as in original notebook",
    )
    add_change(
        log,
        "dropped_variables",
        source=dropped_detail_columns,
        target=result.columns.tolist(),
        how="drop detailed/free-text comorbidity descriptors not used by mortality model",
    )
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



def preprocess_signos_sintomas(tables: dict[str, pd.DataFrame], metadata: dict[str, str]) -> PreprocessResult:
    """Process symptoms and vital signs using the original notebook recodes."""
    source = clean_missing_values(tables["signos_sintomas"])
    df = source.copy()
    df = normalize_merge_key_dtypes(df)

    # Keep first hemoculture row per admission/table, then aggregate to selected key.
    df = df.sort_values(["record_id", "fecha_ingreso", "fecha_hemocultivo"])
    df = df.drop_duplicates(subset=ADMISSION_KEYS, keep="first")

    if "situacion_funcional_basal" in df.columns:
        df = df.drop(columns=["situacion_funcional_basal"])
    if "somnolencia_estupor_coma" in df.columns:
        df["somnolencia_estupor_coma"] = np.where(
            df["somnolencia_estupor_coma"].astype(str).str.lower().eq("normal"),
            0,
            np.where(df["somnolencia_estupor_coma"].isna(), np.nan, 1),
        )

    symptom_cols = [
        "somnolencia_estupor_coma", "fiebre", "tos", "dificultad_respirar",
        "dolor_costal", "disuria", "polaquiuria", "tenesmo_vejiga", "tenesmo_ano_recto",
        "dolor_fosa_renal", "nauseas", "vomitos", "dolor_abdominal", "diarrea",
        "lesiones_piel", "lesiones_mucosas", "cefalea", "dolores_articulares",
    ]
    available_symptom_cols = [c for c in symptom_cols if c in df.columns]
    if available_symptom_cols:
        df["missing_symptoms"] = df[available_symptom_cols].isna().all(axis=1).astype(int)
        df[available_symptom_cols] = df[available_symptom_cols].fillna(0).astype(int)

    # Vital sign abnormality indicators from original notebook.
    if "temperatura" in df.columns:
        df["temperatura"] = pd.to_numeric(df["temperatura"], errors="coerce")
        df["hipertermia"] = np.where(df["temperatura"].isna(), np.nan, np.where(df["temperatura"] >= 38, 1, 0))
        df["hipotermia"] = np.where(df["temperatura"].isna(), np.nan, np.where(df["temperatura"] < 36, 1, 0))
    if {"tension_arterial_sist", "tension_arterial_diast"}.issubset(df.columns):
        df["tension_arterial_sist"] = pd.to_numeric(df["tension_arterial_sist"], errors="coerce")
        df["tension_arterial_diast"] = pd.to_numeric(df["tension_arterial_diast"], errors="coerce")
        missing_bp = df["tension_arterial_sist"].isna() | df["tension_arterial_diast"].isna()
        df["hipotension"] = np.where(
            missing_bp,
            np.nan,
            np.where((df["tension_arterial_sist"] <= 90) & (df["tension_arterial_diast"] <= 60), 1, 0),
        )
        df["hipertension"] = np.where(
            missing_bp,
            np.nan,
            np.where((df["tension_arterial_sist"] >= 140) & (df["tension_arterial_diast"] >= 90), 1, 0),
        )
    if "frecuencia_respiratoria" in df.columns:
        df["frecuencia_respiratoria"] = pd.to_numeric(df["frecuencia_respiratoria"], errors="coerce")
        df["taquipnea"] = np.where(df["frecuencia_respiratoria"].isna(), np.nan, np.where(df["frecuencia_respiratoria"] > 20, 1, 0))
    if "frec_cardiaca" in df.columns:
        df["frec_cardiaca"] = pd.to_numeric(df["frec_cardiaca"], errors="coerce")
        df["taquicardia"] = np.where(df["frec_cardiaca"].isna(), np.nan, np.where(df["frec_cardiaca"] > 90, 1, 0))
    if "saturacion_pO2" in df.columns:
        df["saturacion_pO2"] = pd.to_numeric(df["saturacion_pO2"], errors="coerce")
        # Preserve the original threshold. If values are stored as 90-100 instead of 0.90-1.00,
        # this will almost always be 0 and should be reviewed upstream.
        df["hipoxemia"] = np.where(df["saturacion_pO2"].isna(), np.nan, np.where(df["saturacion_pO2"] < 0.90, 1, 0))

    signs_cols = ["hipertermia", "hipotermia", "hipotension", "hipertension", "taquipnea", "taquicardia", "hipoxemia"]
    available_signs_cols = [c for c in signs_cols if c in df.columns]
    if available_signs_cols:
        df["missing_signs"] = df[available_signs_cols].isna().all(axis=1).astype(int)
        # Keep abnormality indicators, but fill missing as 0 after creating the missingness flag.
        df[available_signs_cols] = df[available_signs_cols].fillna(0).astype(int)

    drop_raw_vitals = [
        "barthel_inf_90", "temperatura", "tension_arterial_sist", "tension_arterial_diast",
        "frec_cardiaca", "saturacion_pO2", "frecuencia_respiratoria", "duracion_sintoma",
    ]
    df = df.drop(columns=[c for c in drop_raw_vitals if c in df.columns], errors="ignore")

    agg: dict[str, Any] = {}
    for col in df.columns:
        if col in ADMISSION_KEYS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            agg[col] = "max"
        else:
            agg[col] = first_notna
    result = df.groupby(ADMISSION_KEYS, as_index=False, dropna=False).agg(agg)

    log = finalize_log(table_name="signos_sintomas", input_df=source, output_df=result, merge_keys=ADMISSION_KEYS)
    attach_metadata(log, metadata)
    add_change(
        log,
        "created_variables",
        source=["symptoms", "vital_signs"],
        target=["missing_symptoms", "hipertermia", "hipotermia", "hipotension", "hipertension", "taquipnea", "taquicardia", "hipoxemia", "missing_signs"],
        how="reproduce original notebook symptom missingness and vital-sign abnormality recodes",
    )
    add_change(
        log,
        "dropped_variables",
        source=drop_raw_vitals + ["situacion_funcional_basal"],
        target=result.columns.tolist(),
        how="drop raw vital signs after deriving clinical abnormality flags",
    )
    return validate_result(PreprocessResult(df=result, log=log))


def preprocess_laboratorio(tables: dict[str, pd.DataFrame], metadata: dict[str, str]) -> PreprocessResult:
    source = clean_missing_values(tables["laboratorio"])
    df = source.copy()
    df["fecha_ingreso"] = pd.to_datetime(df["fecha_ingreso"], errors="coerce")
    if "fecha_hemocultivo" in df.columns:
        df["fecha_hemocultivo"] = pd.to_datetime(df["fecha_hemocultivo"], errors="coerce")
    lab_columns = [
        column
        for column in df.columns
        if column not in {"record_id", "fecha_ingreso", "fecha_hemocultivo"}
    ]
    for column in lab_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Normalize laboratory values with log1p in-place. This reduces right-skew
    # without fitting distributional parameters on the full dataset. Negative
    # values, if any, are treated as invalid and set to NaN before log1p.
    for column in lab_columns:
        df.loc[df[column] < 0, column] = np.nan
        df[column] = np.log1p(df[column])

    # Keep one normalized value per lab feature. Do not create *_mean / *_max
    # features and do not add laboratorio_n_rows; those summaries are not
    # meaningful for this mortality model.
    result = df.groupby(ADMISSION_KEYS, as_index=False, dropna=False).agg(
        {column: first_notna for column in lab_columns}
    )
    log = finalize_log(table_name="laboratorio", input_df=source, output_df=result, merge_keys=ADMISSION_KEYS)
    attach_metadata(log, metadata)
    add_change(
        log,
        "transformed_variables",
        source=lab_columns,
        target=[column for column in result.columns if column not in ADMISSION_KEYS],
        how="log1p-normalize laboratory values and collapse to one value per selected hemoculture using first non-missing value; no *_mean, *_max, or row-count features",
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
    """Process infection episodes using the original BactHeCom microorganism logic.

    Keeps only the first blood-culture date per admission for the current episode,
    derives microorganism groups and previous-episode summaries, and returns one
    row per record_id + fecha_ingreso + selected fecha_hemocultivo.
    """
    source = clean_missing_values(tables["episodio_infeccion"])
    df = source.copy()
    df["fecha_ingreso"] = pd.to_datetime(df["fecha_ingreso"], errors="coerce")
    df["fecha_cultivo"] = pd.to_datetime(df["fecha_cultivo"], errors="coerce")
    df["fecha_hemocultivo"] = df["fecha_cultivo"]

    df = df.sort_values(["record_id", "fecha_ingreso", "fecha_cultivo", "episode_id"])
    df["microorganismo_recoded"] = df["microorganismo"].map(microorganism_species_label)
    df["microorganismo_group"] = df["microorganismo"].map(classify_microorganism)
    df["resistance_mechanism"] = df["fenotipo_resistencia"].fillna("NEGATIVE")
    df.loc[df["resistance_mechanism"].astype(str).str.strip().eq(""), "resistance_mechanism"] = "NEGATIVE"
    df["blood_culture"] = df["especimen"].fillna("").eq("Sangre").astype(int)
    df["days_culture_from_admission"] = (df["fecha_cultivo"] - df["fecha_ingreso"]).dt.days
    df["infection_before_admission"] = (df["days_culture_from_admission"] < 0).astype(int)
    df["infection_same_day_admission"] = (df["days_culture_from_admission"] == 0).astype(int)
    df["infection_after_admission"] = (df["days_culture_from_admission"] > 0).astype(int)

    blood = df.loc[df["blood_culture"].eq(1)].copy()
    current, selection_stats = keep_first_hemoculture_with_buffer(
        blood,
        buffer_days=HEMOCULTURE_PRE_ADMISSION_BUFFER_DAYS,
    )
    selected_keys = current[ADMISSION_KEYS].drop_duplicates()
    current_rows = blood.merge(selected_keys, on=ADMISSION_KEYS, how="inner")

    grouped = current_rows.groupby(ADMISSION_KEYS, as_index=False, dropna=False)
    result = grouped.agg(
        episode_id=("episode_id", lambda x: ", ".join(map(str, pd.Series(x).dropna().unique()))),
        area_hosp=("area_hosp", first_notna),
        id_cultivo=("id_cultivo", lambda x: ", ".join(sorted_unique_list(x))),
        especimen=("especimen", first_notna),
        microorganismo_group=("microorganismo_group", lambda x: ", ".join(sorted(set(filter(None, map(str, x.dropna())))))),
        fenotipo_resistencia=("resistance_mechanism", lambda x: ", ".join(sorted(set(filter(None, map(str, x.dropna()))))) or "NEGATIVE"),
        microorganismo_recoded=("microorganismo_recoded", lambda x: ", ".join(sorted(set(filter(None, map(str, x.dropna())))))),
        organism_count=("microorganismo", lambda values: len(set(values.dropna()))),
        organism_list=("microorganismo", lambda values: list_literal(sorted_unique_list(values))),
        dominant_microorganism=("microorganismo", dominant_label),
        blood_culture_episode_count=("blood_culture", "sum"),
        resistance_mechanism_labels=("resistance_mechanism", lambda values: list_literal(label for label in values if label != "NEGATIVE")),
        fecha_cultivo_min=("fecha_cultivo", "min"),
        fecha_cultivo_max=("fecha_cultivo", "max"),
        days_first_culture_from_admission=("days_culture_from_admission", "min"),
        days_last_culture_from_admission=("days_culture_from_admission", "max"),
    )

    result["n_microorganismos"] = result["microorganismo_group"].map(lambda x: len([p for p in str(x).split(", ") if p]))
    result["n_microorganismos_unique"] = result["microorganismo_group"].map(lambda x: len(set([p for p in str(x).split(", ") if p])))
    dummies = result["microorganismo_group"].str.get_dummies(sep=", ").add_prefix("microorganismo_")
    result = pd.concat([result, dummies], axis=1)

    def mechanism_target(values: pd.Series) -> str:
        labels = sorted({str(value) for value in values.dropna() if str(value) != "NEGATIVE" and str(value).strip()})
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

    resistance_target = current_rows.groupby(ADMISSION_KEYS, as_index=False, dropna=False).agg(
        resistance_mechanism_target=("resistance_mechanism", mechanism_target)
    )
    result = result.merge(resistance_target, on=ADMISSION_KEYS, how="left")

    # Previous episode summaries: previous hemocultures before fecha_ingreso - 2 days.
    prev_records: list[dict[str, Any]] = []
    all_blood = blood.copy()
    for _, row in selected_keys.iterrows():
        rid = row["record_id"]
        ingreso = pd.to_datetime(row["fecha_ingreso"])
        hc = pd.to_datetime(row["fecha_hemocultivo"])
        cutoff = ingreso - pd.Timedelta(days=HEMOCULTURE_PRE_ADMISSION_BUFFER_DAYS)
        prev = all_blood.loc[(all_blood["record_id"].eq(rid)) & (all_blood["fecha_cultivo"] < cutoff)].copy()
        prev = prev.sort_values(["fecha_cultivo", "episode_id"], ascending=[False, True]).head(5)
        days_since = (hc - prev["fecha_cultivo"]).dt.days if not prev.empty else pd.Series(dtype=float)
        groups = sorted(set(prev["microorganismo_group"].dropna().astype(str))) if not prev.empty else []
        species = sorted(set(prev["microorganismo_recoded"].dropna().astype(str))) if not prev.empty else []
        last_group = prev["microorganismo_group"].dropna().astype(str).iloc[0] if not prev.empty and prev["microorganismo_group"].notna().any() else ""
        had_res = int(prev["resistance_mechanism"].fillna("NEGATIVE").ne("NEGATIVE").any()) if not prev.empty else 0
        rec = {
            "record_id": rid,
            "fecha_ingreso": ingreso,
            "fecha_hemocultivo": hc,
            "prev_episode_count": int(prev["fecha_cultivo"].nunique()) if not prev.empty else 0,
            "n_microorganismos_specie_prev": int(len(species)),
            "n_microorganismos_group_prev": int(len(groups)),
            "n_microorganismos_unique_specie_prev": int(len(species)),
            "n_microorganismos_unique_group_prev": int(len(groups)),
            "had_resistance_prev": had_res,
            "prev_within_30day": int((days_since <= 30).any()) if not days_since.empty else 0,
            "prev_within_90day": int((days_since <= 90).any()) if not days_since.empty else 0,
            "last_microorganismo_group_prev": last_group,
        }
        prev_records.append(rec)
    if prev_records:
        prev_df = pd.DataFrame(prev_records)
        last_dummies = pd.get_dummies(prev_df["last_microorganismo_group_prev"], dtype=int).add_prefix("last_prev_")
        prev_df = pd.concat([prev_df.drop(columns=["last_microorganismo_group_prev"]), last_dummies], axis=1)
        result = result.merge(prev_df, on=ADMISSION_KEYS, how="left")

    prev_cols = [c for c in result.columns if "prev" in c]
    if prev_cols:
        result[prev_cols] = result[prev_cols].fillna(0)
    result["has_had_resistance"] = (
        result.get("had_resistance_prev", pd.Series(0, index=result.index)).fillna(0).astype(int).eq(1)
        | result["fenotipo_resistencia"].fillna("NEGATIVE").astype(str).str.strip().ne("NEGATIVE")
    ).astype(int)

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
        source=["microorganismo", "fenotipo_resistencia", "especimen", "fecha_cultivo"],
        target=[
            "microorganismo_recoded", "microorganismo_group", "microorganismo_*",
            "prev_*", "has_had_resistance", "resistance_mechanism_target",
        ],
        how="reproduce original notebook microorganism grouping, first blood culture selection, and previous episode summary features",
    )
    log.metadata.update({f"infection_hemoculture_selection_{k}": v for k, v in selection_stats.items()})
    return validate_result(PreprocessResult(df=result, log=log))

def preprocess_antibiograma(
    tables: dict[str, pd.DataFrame], metadata: dict[str, str]
) -> PreprocessResult:
    source = clean_missing_values(tables["antibiograma"])
    infections = tables["episodio_infeccion"][["episode_id", "record_id", "fecha_ingreso", "fecha_cultivo"]].copy()
    infections["fecha_ingreso"] = pd.to_datetime(infections["fecha_ingreso"], errors="coerce")
    infections["fecha_hemocultivo"] = pd.to_datetime(infections["fecha_cultivo"], errors="coerce")
    infections = infections.drop(columns=["fecha_cultivo"])
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
    infections = tables["episodio_infeccion"][["episode_id", "record_id", "fecha_ingreso", "fecha_cultivo"]].copy()
    infections["fecha_ingreso"] = pd.to_datetime(infections["fecha_ingreso"], errors="coerce")
    infections["fecha_hemocultivo"] = pd.to_datetime(infections["fecha_cultivo"], errors="coerce")
    infections = infections.drop(columns=["fecha_cultivo"])
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
    base = normalize_merge_key_dtypes(results["episodio_ingreso"].df.copy())
    input_df = base.copy()
    for name, result in results.items():
        if name == "episodio_ingreso":
            continue
        right = normalize_merge_key_dtypes(result.df)
        if result.log.merge_keys == ["record_id"]:
            base = base.merge(right, on="record_id", how="left")
        else:
            base = base.merge(right, on=ADMISSION_KEYS, how="left")

    base["fecha_ingreso"] = pd.to_datetime(base["fecha_ingreso"], errors="coerce")
    if "fecha_nacimiento" in base.columns:
        base["age"] = ((base["fecha_ingreso"] - base["fecha_nacimiento"]).dt.days / 365.25).round(1)

    # Fill count-like variables created before the selected hemoculture.
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
        "fenotipo_resistencia",
    ]:
        if column in base.columns:
            base[column] = base[column].fillna("NEGATIVE")
    for column in [
        "organism_list",
        "resistance_mechanism_labels",
    ]:
        if column in base.columns:
            base[column] = base[column].fillna("[]")

    columns_to_drop = [
        "index",
        "episode_key",
        "fecha_nacimiento",
        "age_at_admission",
        "age_at_hemoculture",
        "tipo_cancer",
        "tipo_hepatopatia",
        "causa_inmunosupresion",
        "clasificacion_quemadura",
        "duracion_sintoma",
        "laboratorio_n_rows",
    ]
    columns_to_drop.extend([c for c in base.columns if c.endswith("_mean") or c.endswith("_max")])
    columns_to_drop.extend([c for c in base.columns if c.startswith("treatment_")])
    columns_to_drop.extend([c for c in base.columns if c.startswith("antibiogram_")])
    columns_to_drop.extend([c for c in base.columns if c.startswith("infection_") and c.endswith("_count")])
    columns_to_drop.extend([c for c in base.columns if c == "resistente_cefalosporina"])
    base = base.drop(columns=[c for c in columns_to_drop if c in base.columns], errors="ignore")

    duplicate_selected_admissions = duplicate_key_groups(base, BASE_ADMISSION_KEYS)
    if duplicate_selected_admissions:
        raise ValueError(
            "Final dataset is not one row per record_id + fecha_ingreso after hemoculture selection: "
            f"{duplicate_selected_admissions} duplicated admission groups."
        )

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
        target="age",
        how="calculate patient age at admission date only; no age_at_hemoculture variable is created",
    )
    add_change(
        log,
        "dropped_variables",
        source=columns_to_drop,
        target=base.columns.tolist(),
        how="drop leakage/non-modelling columns and non-informative text or aggregation artifacts",
    )
    add_change(
        log,
        "transformed_variables",
        source=list(results.keys()),
        target=base.columns.tolist(),
        how=(
            "left-join all episode-level tables to selected episodio_ingreso base; "
            "final base contains one selected hemoculture per admission"
        ),
    )
    log.validation_checks.append(f"duplicate_selected_admissions:{duplicate_selected_admissions}")
    return validate_result(PreprocessResult(df=base, log=log))


def export_dataset_outputs(
    df: pd.DataFrame, output_path: Path, drop_columns_path: Path
) -> tuple[pd.DataFrame, list[str], Path]:
    serializable = df.copy()
    automatic_drop_columns = [
        "index",
        "episode_key",
        "fecha_nacimiento",
        "age_at_admission",
        "age_at_hemoculture",
        "tipo_cancer",
        "tipo_hepatopatia",
        "causa_inmunosupresion",
        "clasificacion_quemadura",
        "duracion_sintoma",
        "laboratorio_n_rows",
    ]
    automatic_drop_columns.extend([c for c in serializable.columns if c.endswith("_mean") or c.endswith("_max")])
    automatic_drop_columns.extend([c for c in serializable.columns if c.startswith("treatment_")])
    automatic_drop_columns.extend([c for c in serializable.columns if c.startswith("antibiogram_")])
    automatic_drop_columns.extend([c for c in serializable.columns if c.startswith("infection_") and c.endswith("_count")])
    automatic_drop_columns.extend([c for c in serializable.columns if c == "resistente_cefalosporina"])
    serializable = serializable.drop(columns=[c for c in automatic_drop_columns if c in serializable.columns], errors="ignore")
    for column in serializable.select_dtypes(include=["datetime64[ns]"]).columns:
        serializable[column] = serializable[column].dt.strftime("%Y-%m-%d")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable.to_csv(output_path, index=False)
    drop_columns = read_drop_columns(drop_columns_path, serializable.columns.tolist())
    filtered = serializable.drop(columns=drop_columns, errors="ignore")
    filtered_path = output_path.with_name(f"{output_path.stem}_filtered.csv")
    filtered.to_csv(filtered_path, index=False)
    return filtered, sorted(set(drop_columns + [c for c in automatic_drop_columns if c in df.columns])), filtered_path


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
            "aggregation_level=one selected fecha_hemocultivo per record_id + fecha_ingreso",
            "antimicrobial_treatment_exposure=excluded_to_avoid_post_hemoculture_leakage",
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
    results["comorbilidad"] = preprocess_comorbilidad(tables, metadata)
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
    results["signos_sintomas"] = preprocess_signos_sintomas(tables, metadata)
    results["laboratorio"] = preprocess_laboratorio(tables, metadata)
    results["episodio_uci"] = preprocess_episodio_uci(tables, metadata)
    results["episodio_infeccion"] = preprocess_episodio_infeccion(tables, metadata)
    # Do not include antibiogram-derived features or cefalosporin-resistance targets
    # for this mortality model. They are post-culture microbiology outputs and can
    # introduce leakage or answer a different prediction task.
    # Do not include antimicrobial treatment exposures: antibiotics given after the
    # hemoculture are post-index information and can leak outcome/severity.
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
        description="Preprocess BAcTHECOM SQLite data into a blood-culture episode-level mortality modelling dataset."
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