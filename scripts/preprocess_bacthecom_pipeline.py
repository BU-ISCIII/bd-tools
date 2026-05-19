from __future__ import annotations

import argparse
import importlib.util
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
import json

import numpy as np
import pandas as pd


DEFAULT_DB_PATH = Path("RAW/20260507_db_bacthecom.db")
DEFAULT_CONFIG_PATH = Path("ANALYSIS/01-PREPROCESS/preprocess_config.py")
DEFAULT_OUTPUT_PATH = Path("ANALYSIS/01-PREPROCESS/00-data/preprocessed_db.csv")


@dataclass
class VariableChange:
    kind: str
    source: str | list[str]
    target: str | list[str]
    how: str
    variable_type: str = ""
    n_classes: int | dict[str, int] | None = None


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
    maps: dict[str, dict[Any, Any]]
    logs: list[TableLog] = field(default_factory=list)


ENTEROBACTERIACEAE = {
    "Escherichia", "Klebsiella", "Enterobacter", "Citrobacter", "Serratia",
    "Proteus", "Morganella", "Salmonella", "Raoultella", "Pantoea",
}


def add_change(
    log: TableLog,
    section: str,
    *,
    source: str | list[str],
    target: str | list[str],
    how: str,
    variable_type: str = "",
    n_classes: int | dict[str, int] | None = None,
) -> None:
    getattr(log, section).append(
        VariableChange(
            kind=section,
            source=source,
            target=target,
            how=how,
            variable_type=variable_type,
            n_classes=n_classes,
        )
    )


def finalize_log(table_name: str, input_df: pd.DataFrame, output_df: pd.DataFrame, merge_keys: Iterable[str]) -> TableLog:
    return TableLog(
        table_name=table_name,
        input_rows=len(input_df),
        output_rows=len(output_df),
        input_columns=input_df.columns.tolist(),
        output_columns=output_df.columns.tolist(),
        merge_keys=list(merge_keys),
        columns_lost=sorted(set(input_df.columns) - set(output_df.columns)),
    )


def attach_run_metadata(log: TableLog, run_label: str) -> None:
    log.metadata["run_label"] = run_label


def validate_result(result: PreprocessResult, required_columns: list[str]) -> PreprocessResult:
    missing = [col for col in required_columns if col not in result.df.columns]
    if missing:
        raise ValueError(f"{result.log.table_name} missing required columns: {missing}")
    result.log.validation_checks.append(f"required_columns_present:{','.join(required_columns)}")
    result.log.validation_checks.append(f"row_count:{len(result.df)}")
    return result


def export_logs(logs: list[TableLog], summary_output_path: Path, detailed_output_path: Path) -> None:
    summary_records: list[dict[str, Any]] = []
    detail_records: list[dict[str, Any]] = []
    for log in logs:
        summary_records.append(
            {
                "table_name": log.table_name,
                "input_rows": log.input_rows,
                "output_rows": log.output_rows,
                "input_column_count": len(log.input_columns),
                "output_column_count": len(log.output_columns),
                "columns_dropped_count": log.columns_dropped_count,
                "columns_lost_count": len(log.columns_lost),
                "warning_count": len(log.warnings),
                "note_count": len(log.notes),
                "validation_count": len(log.validation_checks),
                "run_label": log.metadata.get("run_label", ""),
            }
        )
        detail_records.append(asdict(log))

    pd.DataFrame(summary_records).to_csv(summary_output_path, index=False)
    with detailed_output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(detail_records, file_handle, indent=2, ensure_ascii=False)


def load_tables(db_path: Path) -> dict[str, pd.DataFrame]:
    if not db_path.exists():
        raise FileNotFoundError(f"DB file not found: {db_path}")
    excluded = {"semantic_mapping", "sqlite_sequence", "tbl_codes2names", "episodio_uci"}
    tables: dict[str, pd.DataFrame] = {}
    with sqlite3.connect(db_path) as conn:
        names = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist()
        for table in names:
            if table in excluded:
                continue
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            for col in [column for column in df.columns if "fecha" in column.lower()]:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            tables[table] = df
    return tables


def load_config_module(config_path: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("bacthecom_preprocess_runtime_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config module from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "as_dict"):
        raise AttributeError(f"Config module {config_path} must define as_dict()")
    return module.as_dict()


def build_antimicrobial_family_map(config: dict[str, Any]) -> dict[str, str]:
    family_map: dict[str, str] = {}
    for raw_label, family, normalized_name in config.get("ANTIMICROBIAL_GROUPS", []):
        if raw_label and raw_label != "A" and raw_label != "B":
            family_map[str(raw_label)] = family
            family_map[str(raw_label).upper()] = family
        if normalized_name:
            family_map[str(normalized_name)] = family
            family_map[str(normalized_name).upper()] = family
    return family_map


def classify_microorganism(name: Any) -> str:
    if pd.isna(name):
        return "Other"
    species = str(name).replace(".", "").strip()
    if not species:
        return "Other"
    exact = {
        "Escherichia_coli",
        "Staphylococcus_aureus",
        "Pseudomonas_aeruginosa",
        "Klebsiella_pneumoniae",
        "Streptococcus_pneumoniae",
    }
    if species in exact:
        return species
    if species.startswith("Enterococcus"):
        return "Enterococcus"
    genus = species.split("_")[0]
    return "Enterobacteria" if genus in ENTEROBACTERIACEAE else "Other"


def preprocess_paciente(tables: dict[str, pd.DataFrame], run_label: str) -> PreprocessResult:
    source = tables["paciente"].copy()
    df = source.copy()
    log = finalize_log("paciente", source, df, ["record_id"])

    df["sexo"] = df["sexo"].map({"Hombre": 0, "Mujer": 1})
    add_change(log, "recoded_variables", source="sexo", target="sexo", how="map Hombre->0, Mujer->1")

    attach_run_metadata(log, run_label)
    log.output_rows = len(df)
    log.output_columns = df.columns.tolist()
    return validate_result(PreprocessResult(df, log), ["record_id"])


def preprocess_epi_ingreso(tables: dict[str, pd.DataFrame], run_label: str) -> PreprocessResult:
    source = tables["episodio_ingreso"].copy()
    df = source.copy()
    log = finalize_log("episodio_ingreso", source, df, ["record_id", "fecha_ingreso"])

    df["priority_days"] = np.where(
        df["dias_hemocultivo_mortalidad"].notna(),
        df["dias_hemocultivo_mortalidad"],
        df["dias_hemocultivo_ingresoUCI"],
    )
    add_change(log, "created_variables", source=["dias_hemocultivo_mortalidad", "dias_hemocultivo_ingresoUCI"], target="priority_days", how="use mortality days; fallback ICU admission days")

    before = len(df)
    df = df.sort_values(["record_id", "fecha_ingreso", "priority_days"], ascending=[True, True, False])
    df = df.drop_duplicates(["record_id", "fecha_ingreso"], keep="first").drop(columns=["priority_days"])
    add_change(log, "transformed_variables", source=["record_id", "fecha_ingreso", "priority_days"], target=["record_id", "fecha_ingreso"], how=f"deduplicate episodes by max priority_days ({before} -> {len(df)} rows)")

    df["mortalidad_14_dias"] = ((df["fecha_mortalidad"] - df["fecha_ingreso"]) <= pd.Timedelta(days=14)).astype(int)
    add_change(log, "created_variables", source=["fecha_mortalidad", "fecha_ingreso"], target="mortalidad_14_dias", how="1 if death within 14 days")

    df["en_uci_antes_del_hemocultivo"] = df["en_uci_antes_del_hemocultivo"].fillna(0)
    df["IRAs_nosocomial"] = df["IRAs_nosocomial"].map({"No": 0, "Si": 1})
    add_change(log, "recoded_variables", source=["en_uci_antes_del_hemocultivo", "IRAs_nosocomial"], target=["en_uci_antes_del_hemocultivo", "IRAs_nosocomial"], how="fill NaN with 0 and map No/Si to 0/1")

    dropped = [col for col in ["codigo_postal", "organo_aparato"] if col in df.columns]
    if dropped:
        df = df.drop(columns=dropped)
        log.columns_dropped_count += len(dropped)
        add_change(log, "dropped_variables", source=dropped, target=dropped, how="drop non-modeling columns")

    attach_run_metadata(log, run_label)
    log.output_rows = len(df)
    log.output_columns = df.columns.tolist()
    return validate_result(PreprocessResult(df, log), ["record_id", "fecha_ingreso"])


def preprocess_epi_infeccion_and_antibiograma(tables: dict[str, pd.DataFrame], antimicrobial_family_map: dict[str, Any], run_label: str) -> PreprocessResult:
    source_epi = tables["episodio_infeccion"].copy()
    source_anti = tables["antibiograma"].copy()
    epi = source_epi.sort_values(["record_id", "fecha_ingreso", "fecha_cultivo"]).copy()
    log = finalize_log("episodio_infeccion+antibiograma", source_epi, epi, ["record_id", "fecha_ingreso"])

    epi["microorganismo_recoded"] = epi["microorganismo"].fillna("").apply(lambda value: "_".join(str(value).split(" ")[:2]))
    epi["microorganismo_group"] = epi["microorganismo_recoded"].map(classify_microorganism)
    add_change(log, "created_variables", source="microorganismo", target=["microorganismo_recoded", "microorganismo_group"], how="normalize species and map to groups")

    prior = epi[["episode_id", "record_id", "fecha_cultivo", "microorganismo_recoded", "microorganismo_group", "fenotipo_resistencia"]].copy()
    prior = prior.rename(columns={c: f"{c}_prev" for c in prior.columns if c != "record_id"})
    joined = epi[["episode_id", "record_id", "fecha_ingreso"]].merge(prior, on="record_id", how="left")
    joined = joined[(joined["fecha_cultivo_prev"].notna()) & (joined["fecha_ingreso"].notna()) & (joined["fecha_cultivo_prev"] < joined["fecha_ingreso"])].copy()
    joined["days_since_episode"] = (joined["fecha_ingreso"] - joined["fecha_cultivo_prev"]).dt.days
    joined = joined[joined["days_since_episode"] <= 90]
    joined = joined.sort_values(["episode_id", "days_since_episode"]).groupby("episode_id", as_index=False).head(5)

    prev_features = joined.groupby("episode_id").agg(
        prev_episode_count=("episode_id_prev", "count"),
        n_microorganismos_unique_specie_prev=("microorganismo_recoded_prev", lambda x: x.nunique(dropna=True)),
        n_microorganismos_unique_group_prev=("microorganismo_group_prev", lambda x: x.nunique(dropna=True)),
        had_resistance_prev=("fenotipo_resistencia_prev", lambda x: int(x.fillna("").astype(str).str.len().gt(0).any())),
    ).reset_index()
    add_change(log, "created_variables", source=["record_id", "fecha_ingreso", "fecha_cultivo"], target=["prev_episode_count", "n_microorganismos_unique_specie_prev", "n_microorganismos_unique_group_prev", "had_resistance_prev"], how="build previous-episode features from up to 5 prior cultures")

    hemo = epi[epi["especimen"] == "Sangre"].copy()
    min_culture = hemo.groupby(["record_id", "fecha_ingreso"])["fecha_cultivo"].transform("min")
    hemo = hemo[hemo["fecha_cultivo"] == min_culture]
    micro = hemo.groupby(["record_id", "fecha_ingreso", "fecha_cultivo"], dropna=False).agg(
        episode_id=("episode_id", "first"),
        area_hosp=("area_hosp", "first"),
        id_cultivo=("id_cultivo", lambda x: ", ".join(x.dropna().astype(str).unique())),
        especimen=("especimen", "first"),
        microorganismo_group=("microorganismo_group", lambda x: ", ".join(sorted(set(filter(None, x))))),
        fenotipo_resistencia=("fenotipo_resistencia", lambda x: ", ".join(sorted(set(filter(None, x))))),
    ).reset_index()
    add_change(log, "transformed_variables", source=["especimen", "fecha_cultivo"], target=["record_id", "fecha_ingreso", "fecha_cultivo"], how="keep blood-culture rows from earliest culture date per admission and aggregate")

    micro["n_microorganismos"] = micro["microorganismo_group"].apply(lambda x: len(x.split(", ")) if x else 0)
    micro_dummies = micro["microorganismo_group"].str.get_dummies(sep=", ").add_prefix("microorganismo_")
    micro = pd.concat([micro, micro_dummies], axis=1)
    add_change(log, "created_variables", source="microorganismo_group", target=["n_microorganismos"] + micro_dummies.columns.tolist(), how="create count and dummies of microorganism groups")

    micro = micro.merge(prev_features, on="episode_id", how="left")
    for col in ["prev_episode_count", "n_microorganismos_unique_specie_prev", "n_microorganismos_unique_group_prev", "had_resistance_prev"]:
        if col in micro.columns:
            micro[col] = micro[col].fillna(0)
    micro["has_had_resistance"] = ((micro["had_resistance_prev"] == 1) | (micro["fenotipo_resistencia"].fillna("").str.len() > 0)).astype(int)
    add_change(log, "created_variables", source=["had_resistance_prev", "fenotipo_resistencia"], target="has_had_resistance", how="combine previous and current resistance")

    anti = source_anti.copy()
    anti["antimicrobiano_family"] = anti["antimicrobiano"].map(
        lambda value: antimicrobial_family_map.get(value, antimicrobial_family_map.get(str(value).upper()))
    )
    anti_grouped = anti.groupby("episode_id", dropna=False).agg(
        antimicrobiano_list=("antimicrobiano", list),
        cmi_list=("cmi", list),
        interpretacion_list=("interpretacion", list),
        antimicrobiano_family_list=("antimicrobiano_family", list),
    ).reset_index()
    micro["episode_id"] = micro["episode_id"].astype(str)
    anti_grouped["episode_id"] = anti_grouped["episode_id"].astype(str)
    df = micro.merge(anti_grouped, on="episode_id", how="left")
    add_change(log, "transformed_variables", source="antibiograma", target=["antimicrobiano_list", "cmi_list", "interpretacion_list", "antimicrobiano_family_list"], how="group antibiogram by episode_id and merge")

    attach_run_metadata(log, run_label)
    log.output_rows = len(df)
    log.output_columns = df.columns.tolist()
    return validate_result(PreprocessResult(df, log), ["record_id", "fecha_ingreso"])


def preprocess_comorbilidad(tables: dict[str, pd.DataFrame], run_label: str) -> PreprocessResult:
    source = tables["comorbilidad"].copy()
    df = source.drop(columns=["fecha_hemocultivo"], errors="ignore").copy()
    log = finalize_log("comorbilidad", source, df, ["record_id", "fecha_ingreso"])

    df[["hepatopatia_ligera", "hepatopatia_moderada_o_grave"]] = df[["hepatopatia_ligera", "hepatopatia_moderada_o_grave"]].fillna(0)
    add_change(log, "recoded_variables", source=["hepatopatia_ligera", "hepatopatia_moderada_o_grave"], target=["hepatopatia_ligera", "hepatopatia_moderada_o_grave"], how="fill NaN with 0")

    df["has_cancer"] = df[["neoplasia_tratamiento_activo", "neoplasia_solida_metastasica", "neoplasia_solida_no_metastasica"]].fillna(0).sum(axis=1).gt(0).astype(int)
    add_change(log, "created_variables", source=["neoplasia_tratamiento_activo", "neoplasia_solida_metastasica", "neoplasia_solida_no_metastasica"], target="has_cancer", how="1 if any cancer field > 0")

    drop_cols = [
        "tipo_hepatopatia", "causa_inmunosupresion", "tipo_cancer", "fecha_TOS", "fecha_TPH", "clasificacion_quemadura", "puntaje_child_pugh",
    ]
    existing = [c for c in drop_cols if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
        log.columns_dropped_count += len(existing)
        add_change(log, "dropped_variables", source=existing, target=existing, how="drop non-modeling columns")

    numeric_cols = [c for c in df.columns if c not in {"record_id", "fecha_ingreso"} and pd.api.types.is_numeric_dtype(df[c])]
    df["num_comorbilidades"] = df[numeric_cols].fillna(0).sum(axis=1)
    add_change(log, "created_variables", source=numeric_cols, target="num_comorbilidades", how="sum numeric comorbidity features")

    attach_run_metadata(log, run_label)
    log.output_rows = len(df)
    log.output_columns = df.columns.tolist()
    return validate_result(PreprocessResult(df, log), ["record_id", "fecha_ingreso"])


def preprocess_signos_sintomas(tables: dict[str, pd.DataFrame], run_label: str) -> PreprocessResult:
    source = tables["signos_sintomas"].copy()
    df = source.sort_values(["record_id", "fecha_ingreso", "fecha_hemocultivo"]).groupby(["record_id", "fecha_ingreso"], as_index=False).first()
    log = finalize_log("signos_sintomas", source, df, ["record_id", "fecha_ingreso"])

    drop_initial = [c for c in ["situacion_funcional_basal", "fecha_hemocultivo", "fecha_inicio_sintoma"] if c in df.columns]
    if drop_initial:
        df = df.drop(columns=drop_initial)
        log.columns_dropped_count += len(drop_initial)
        add_change(log, "dropped_variables", source=drop_initial, target=drop_initial, how="drop metadata columns")

    df["somnolencia_estupor_coma"] = np.where(df["somnolencia_estupor_coma"] == "normal", 0, np.where(df["somnolencia_estupor_coma"].isna(), np.nan, 1))
    add_change(log, "recoded_variables", source="somnolencia_estupor_coma", target="somnolencia_estupor_coma", how="normal->0, other non-null->1")

    symptom_cols = [
        "somnolencia_estupor_coma", "fiebre", "tos", "dificultad_respirar", "dolor_costal", "disuria", "polaquiuria", "tenesmo_vejiga",
        "tenesmo_ano_recto", "dolor_fosa_renal", "nauseas", "vomitos", "dolor_abdominal", "diarrea", "lesiones_piel", "lesiones_mucosas", "cefalea", "dolores_articulares",
    ]
    df["missing_symptoms"] = df[symptom_cols].isna().all(axis=1).astype(int)
    df[symptom_cols] = df[symptom_cols].fillna(0).astype(int)
    add_change(log, "created_variables", source=symptom_cols, target="missing_symptoms", how="1 if all symptom columns are missing")

    df["hipertermia"] = np.where(df["temperatura"].isna(), np.nan, (df["temperatura"] >= 38).astype(int))
    df["hipotermia"] = np.where(df["temperatura"].isna(), np.nan, (df["temperatura"] < 36).astype(int))
    df["hipotension"] = np.where(df["tension_arterial_sist"].isna() | df["tension_arterial_diast"].isna(), np.nan, ((df["tension_arterial_sist"] <= 90) & (df["tension_arterial_diast"] <= 60)).astype(int))
    df["hipertension"] = np.where(df["tension_arterial_sist"].isna() | df["tension_arterial_diast"].isna(), np.nan, ((df["tension_arterial_sist"] >= 140) & (df["tension_arterial_diast"] >= 90)).astype(int))
    df["taquipnea"] = np.where(df["frecuencia_respiratoria"].isna(), np.nan, (df["frecuencia_respiratoria"] > 20).astype(int))
    df["taquicardia"] = np.where(df["frec_cardiaca"].isna(), np.nan, (df["frec_cardiaca"] > 90).astype(int))
    df["hipoxemia"] = np.where(df["saturacion_pO2"].isna(), np.nan, (df["saturacion_pO2"] < 90).astype(int))
    add_change(log, "created_variables", source=["temperatura", "tension_arterial_sist", "tension_arterial_diast", "frecuencia_respiratoria", "frec_cardiaca", "saturacion_pO2"], target=["hipertermia", "hipotermia", "hipotension", "hipertension", "taquipnea", "taquicardia", "hipoxemia"], how="derive signs thresholds")

    sign_cols = ["hipertermia", "hipotermia", "hipotension", "hipertension", "taquipnea", "taquicardia", "hipoxemia"]
    df["missing_signs"] = df[sign_cols].isna().all(axis=1).astype(int)
    add_change(log, "created_variables", source=sign_cols, target="missing_signs", how="1 if all derived signs are missing")

    drop_later = [c for c in ["barthel_inf_90", "temperatura", "tension_arterial_sist", "tension_arterial_diast", "frec_cardiaca", "saturacion_pO2", "frecuencia_respiratoria"] if c in df.columns]
    if drop_later:
        df = df.drop(columns=drop_later)
        log.columns_dropped_count += len(drop_later)
        add_change(log, "dropped_variables", source=drop_later, target=drop_later, how="drop raw sign numeric fields after deriving features")

    attach_run_metadata(log, run_label)
    log.output_rows = len(df)
    log.output_columns = df.columns.tolist()
    return validate_result(PreprocessResult(df, log), ["record_id", "fecha_ingreso"])


def preprocess_factores_bmr(tables: dict[str, pd.DataFrame], run_label: str) -> PreprocessResult:
    source = tables["factores_riesgo_infeccion_bmr"].copy()
    df = source.sort_values(["record_id", "fecha_ingreso"]).drop_duplicates(["record_id", "fecha_ingreso"], keep="first")
    df = df.drop(columns=["fecha_hemocultivo"], errors="ignore")
    log = finalize_log("factores_riesgo_infeccion_bmr", source, df, ["record_id", "fecha_ingreso"])
    add_change(log, "transformed_variables", source=["record_id", "fecha_ingreso"], target=["record_id", "fecha_ingreso"], how=f"deduplicate key rows ({len(source)} -> {len(df)} rows)")

    attach_run_metadata(log, run_label)
    log.output_rows = len(df)
    log.output_columns = df.columns.tolist()
    return validate_result(PreprocessResult(df, log), ["record_id", "fecha_ingreso"])


def preprocess_laboratorio(tables: dict[str, pd.DataFrame], run_label: str) -> PreprocessResult:
    source = tables["laboratorio"].copy()
    df = source.copy()
    log = finalize_log("laboratorio", source, df, ["record_id", "fecha_ingreso", "fecha_hemocultivo"])

    key_cols = ["record_id", "fecha_ingreso", "fecha_hemocultivo"]
    numeric_cols = [column for column in df.columns if column not in key_cols and pd.api.types.is_numeric_dtype(df[column])]
    before = len(df)
    df = (
        df.groupby(key_cols, dropna=False, as_index=False)[numeric_cols]
        .mean()
    )
    add_change(
        log,
        "transformed_variables",
        source=key_cols + numeric_cols,
        target=key_cols + numeric_cols,
        how=f"aggregate duplicated lab rows by mean on keys {key_cols} ({before} -> {len(df)} rows)",
    )

    attach_run_metadata(log, run_label)
    log.output_rows = len(df)
    log.output_columns = df.columns.tolist()
    return validate_result(PreprocessResult(df, log), key_cols)


def merge_preprocessed_tables(
    paciente: PreprocessResult,
    episodios: PreprocessResult,
    micro: PreprocessResult,
    comorb: PreprocessResult,
    signos: PreprocessResult,
    factores: PreprocessResult,
    laboratorio: PreprocessResult,
    run_label: str,
) -> PreprocessResult:
    source = paciente.df.copy()
    df = source.merge(episodios.df, on="record_id", how="left")
    df["edad"] = ((df["fecha_ingreso"] - df["fecha_nacimiento"]).dt.days // 365)
    df = df.drop(columns=["fecha_nacimiento"], errors="ignore")
    df = df.merge(micro.df, on=["record_id", "fecha_ingreso"], how="inner")
    df = df.merge(comorb.df, on=["record_id", "fecha_ingreso"], how="left")
    df = df.merge(signos.df, on=["record_id", "fecha_ingreso"], how="left")
    df = df.merge(factores.df, on=["record_id", "fecha_ingreso"], how="left")
    df = df.merge(
        laboratorio.df,
        left_on=["record_id", "fecha_ingreso", "fecha_cultivo"],
        right_on=["record_id", "fecha_ingreso", "fecha_hemocultivo"],
        how="left",
    )
    df = df.drop(columns=["fecha_hemocultivo"], errors="ignore")

    log = finalize_log("merged_dataset", source, df, ["record_id", "fecha_ingreso"])
    add_change(log, "created_variables", source=["fecha_ingreso", "fecha_nacimiento"], target="edad", how="age in years at admission")
    add_change(log, "transformed_variables", source=["paciente", "episodio_ingreso", "episodio_infeccion+antibiograma", "comorbilidad", "signos_sintomas", "factores_riesgo_infeccion_bmr", "laboratorio"], target="merged_dataset", how="left/inner joins on record_id and fecha_ingreso; laboratorio additionally matched by fecha_cultivo=fecha_hemocultivo")

    na_pct = (df.isna().sum() / len(df) * 100)
    drop_cols = na_pct[na_pct > 30].index.tolist()
    if drop_cols:
        df = df.drop(columns=drop_cols)
        log.columns_dropped_count += len(drop_cols)
        add_change(log, "dropped_variables", source=drop_cols, target=drop_cols, how="drop columns with >30% missing")

    asistencia_cols = ["paciente_residencia", "cirugia_previa_con_implante", "cirugia_previa_sin_implante", "asistencia_sanitaria_prev"]
    for col in asistencia_cols:
        if col not in df.columns:
            df[col] = 0
    df["asistencia_prev_indicator"] = df[asistencia_cols].isna().all(axis=1).astype(int)
    df[asistencia_cols] = df[asistencia_cols].fillna(0)
    add_change(log, "created_variables", source=asistencia_cols, target="asistencia_prev_indicator", how="1 if all assistance-related columns are missing")

    attach_run_metadata(log, run_label)
    log.output_rows = len(df)
    log.output_columns = df.columns.tolist()
    return validate_result(PreprocessResult(df, log), ["record_id", "fecha_ingreso"])


def run_pipeline(db_path: Path, config_path: Path, output_path: Path, run_label: str = "") -> PipelineArtifacts:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tables = load_tables(db_path)
    config = load_config_module(config_path)
    antimicrobial_family_map = build_antimicrobial_family_map(config)

    logs: list[TableLog] = []
    paciente = preprocess_paciente(tables, run_label)
    episodios = preprocess_epi_ingreso(tables, run_label)
    micro = preprocess_epi_infeccion_and_antibiograma(tables, antimicrobial_family_map, run_label)
    comorb = preprocess_comorbilidad(tables, run_label)
    signos = preprocess_signos_sintomas(tables, run_label)
    factores = preprocess_factores_bmr(tables, run_label)
    laboratorio = preprocess_laboratorio(tables, run_label)
    merged = merge_preprocessed_tables(paciente, episodios, micro, comorb, signos, factores, laboratorio, run_label)

    merged.df.to_csv(output_path, index=False)

    logs.extend([paciente.log, episodios.log, micro.log, comorb.log, signos.log, factores.log, laboratorio.log, merged.log])
    export_logs(
        logs,
        summary_output_path=output_path.with_name(f"{output_path.stem}_log_summary.csv"),
        detailed_output_path=output_path.with_name(f"{output_path.stem}_log_detailed.json"),
    )
    return PipelineArtifacts(
        tables={
            "paciente": paciente.df,
            "episodio_ingreso": episodios.df,
            "episodio_infeccion_antibiograma": micro.df,
            "comorbilidad": comorb.df,
            "signos_sintomas": signos.df,
            "factores_riesgo_infeccion_bmr": factores.df,
            "laboratorio": laboratorio.df,
            "final": merged.df,
        },
        maps={"antimicrobial_family_map": antimicrobial_family_map},
        logs=logs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess BACTHECOM SQLite DB with table-level logging.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--run-label", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_pipeline(args.input_path, args.config_path, args.output_path, args.run_label)
    print(f"Output written: {args.output_path}")
    print(f"Rows: {len(artifacts.tables['final'])} | Columns: {artifacts.tables['final'].shape[1]}")
    print(f"Log summary: {args.output_path.with_name(f'{args.output_path.stem}_log_summary.csv')}")
    print(f"Log detailed: {args.output_path.with_name(f'{args.output_path.stem}_log_detailed.json')}")


if __name__ == "__main__":
    main()
