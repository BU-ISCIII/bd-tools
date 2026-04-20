from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = ROOT_DIR / "database" / "db_mepram_sepsis.sqlite3"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs"


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
    # default_factory=list gives each TableLog its own list instance.
    created_variables: list[VariableChange] = field(default_factory=list)
    recoded_variables: list[VariableChange] = field(default_factory=list)
    transformed_variables: list[VariableChange] = field(default_factory=list)
    dropped_variables: list[VariableChange] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


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
    return TableLog(
        table_name=table_name,
        input_rows=len(input_df),
        output_rows=len(output_df),
        input_columns=input_df.columns.tolist(),
        output_columns=output_df.columns.tolist(),
        merge_keys=list(merge_keys),
    )


def export_logs(logs: list[TableLog], output_path: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for log in logs:
        base = {
            "table_name": log.table_name,
            "input_rows": log.input_rows,
            "output_rows": log.output_rows,
            "merge_keys": ",".join(log.merge_keys),
            "warning_count": len(log.warnings),
            "note_count": len(log.notes),
        }
        records.append(base)
    df = pd.DataFrame.from_records(records)
    df.to_csv(output_path, index=False)
    return df


def load_tables(db_path: Path) -> dict[str, pd.DataFrame]:
    with sqlite3.connect(db_path) as conn:
        table_names = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'tbl_%'",
            conn,
        )["name"].tolist()
        return {name: pd.read_sql_query(f"SELECT * FROM {name}", conn) for name in table_names}


def extract_numeric(series: pd.Series) -> pd.Series:
    extracted = series.astype(str).str.extract(r"(\d+\.\d+|\d+)")[0]
    return pd.to_numeric(extracted, errors="coerce")


def clean_outliers_iqr(
    df: pd.DataFrame,
    variables: list[str],
    *,
    replace_with: float | None = np.nan,
) -> pd.DataFrame:
    cleaned = df.copy()
    for variable in variables:
        q1 = cleaned[variable].quantile(0.25)
        q3 = cleaned[variable].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = cleaned[variable].between(lower, upper) | cleaned[variable].isna()
        cleaned.loc[~mask, variable] = replace_with
    return cleaned


def build_reference_maps(tables: dict[str, pd.DataFrame]) -> dict[str, dict[Any, Any]]:
    codes = tables["tbl_codes2names"].copy()
    microorganisms = tables["tbl_microorganismos"].copy()

    organism_label_map = {
        "NOEB": "_Other bacteria",
        "VIRUS": "_Virus",
        "FUNGUS": "_Fungi",
        "OEB": "_Enterobacteria",
        "ECOLI": "Escherichia coli",
        "SA": "Staphylococcus aureus",
        "PSA": "Pseudomonas aeruginosa",
        "KP": "Klebsiella pneumoniae",
        "SP": "Streptococcus pneumoniae",
        "EC": "Enterococcus",
    }
    microorganisms["label"] = microorganisms["label"].map(organism_label_map)
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

    return {
        "organism_codes_map": dict(zip(microorganisms["snomed_code"], microorganisms["label"])),
        "foco_map": foco_map,
        "phenomap": phenomap,
    }


def preprocess_tbl_paciente(
    tables: dict[str, pd.DataFrame],
    maps: dict[str, dict[Any, Any]],
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
    return PreprocessResult(df=df, log=log)


def preprocess_tbl_comorbilidad(
    tables: dict[str, pd.DataFrame],
    maps: dict[str, dict[Any, Any]],
) -> PreprocessResult:
    source = tables["tbl_comorbilidad"].copy()
    df = pd.get_dummies(source, columns=["tipo_cancer", "tipo_hepatopatia"])
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
        "dropped_variables",
        source=["tipo_cancer", "tipo_hepatopatia"],
        target=["tipo_cancer_*", "tipo_hepatopatia_*"],
        how="original categoricals replaced by one-hot encoded columns",
    )
    return PreprocessResult(df=df, log=log)


def preprocess_tbl_factores_riesgo_bmr(
    tables: dict[str, pd.DataFrame],
    maps: dict[str, dict[Any, Any]],
) -> PreprocessResult:
    source = tables["tbl_factores_riesgo_bmr"].copy()
    log = finalize_log(
        table_name="tbl_factores_riesgo_bmr",
        input_df=source,
        output_df=source,
        merge_keys=["person_id", "fecha_ingreso_urgencias"],
    )
    log.notes.append("Pass-through table in current notebook. Add table-local feature engineering here if needed.")
    return PreprocessResult(df=source, log=log)


def preprocess_tbl_sintomas(
    tables: dict[str, pd.DataFrame],
    maps: dict[str, dict[Any, Any]],
) -> PreprocessResult:
    source = tables["tbl_sintomas"].copy()
    codes = tables["tbl_codes2names"].copy()
    symptom_map = (
        codes[codes["variable"] == "sintoma"][["value", "name"]]
        .assign(name=lambda df: df["name"].str.split(" | ").str[-1])
    )
    symptom_map = {str(float(row["value"])): row["name"] for _, row in symptom_map.iterrows()}

    df = source.copy()
    df["sintoma"] = df["sintoma"].astype(str).map(symptom_map)
    df["sintoma"] = "sintoma_" + df["sintoma"]
    pivoted = df.pivot_table(
        index=["person_id", "fecha_ingreso_urgencias"],
        columns="sintoma",
        values="duracion_sintoma",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    presence = pivoted.copy()
    presence.iloc[:, 2:] = (presence.iloc[:, 2:] > 0).astype(int)

    duration = pivoted.copy()
    duration.iloc[:, 2:] = duration.iloc[:, 2:].applymap(
        lambda value: 0 if value == 0 else (1 if value <= 7 else 2)
    )
    duration = duration.rename(columns={col: f"{col}_categorico" for col in duration.columns[2:]})

    result = presence.merge(duration, on=["person_id", "fecha_ingreso_urgencias"], how="left")
    log = finalize_log(
        table_name="tbl_sintomas",
        input_df=source,
        output_df=result,
        merge_keys=["person_id", "fecha_ingreso_urgencias"],
    )
    add_change(
        log,
        "recoded_variables",
        source="sintoma",
        target="sintoma",
        how="map coded symptom values to readable names using tbl_codes2names",
    )
    add_change(
        log,
        "created_variables",
        source="duracion_sintoma",
        target=[col for col in result.columns if col.startswith("sintoma_")],
        how="pivot symptoms to wide binary and ordinal duration columns",
    )
    return PreprocessResult(df=result, log=log)


def preprocess_tbl_signos(
    tables: dict[str, pd.DataFrame],
    maps: dict[str, dict[Any, Any]],
) -> PreprocessResult:
    source = tables["tbl_signos"].copy()
    df = source.copy()

    numeric_columns = df.columns[2:]
    for column in numeric_columns:
        df[column] = extract_numeric(df[column])

    df["hipotermia_hipertermia"] = np.where(
        df["temperatura"] >= 38,
        2,
        np.where(df["temperatura"] >= 36, 0, 1),
    )
    df["hipotermia_hipertermia"] = df["hipotermia_hipertermia"].where(df["temperatura"].notna())
    df["hipotension"] = np.where(
        df["tension_arterial"].isna(),
        df["hipotension"],
        np.where(df["tension_arterial"] <= 100, 1, 0),
    )
    df["taquipnea"] = np.where(
        df["frec_respiratoria"].isna(),
        df["taquipnea"],
        np.where(df["frec_respiratoria"] > 20, 1, 0),
    )
    df["taquicardia"] = np.where(
        df["frec_cardiaca"].isna(),
        df["taquicardia"],
        np.where(df["frec_cardiaca"] > 90, 1, 0),
    )
    df["hipoxemia"] = np.where(
        df["saturacion_o2"].isna(),
        df["hipoxemia"],
        np.where(df["saturacion_o2"] > 90, 0, 1),
    )

    iqr_columns = [
        "temperatura",
        "frec_respiratoria",
        "frec_cardiaca",
        "tension_arterial",
        "saturacion_o2",
    ]
    df = clean_outliers_iqr(df, iqr_columns)

    log = finalize_log(
        table_name="tbl_signos",
        input_df=source,
        output_df=df,
        merge_keys=["person_id", "fecha_ingreso_urgencias"],
    )
    add_change(
        log,
        "transformed_variables",
        source=list(numeric_columns),
        target=list(numeric_columns),
        how="extract numeric values from mixed text fields",
    )
    add_change(
        log,
        "created_variables",
        source=["temperatura", "tension_arterial", "frec_respiratoria", "frec_cardiaca", "saturacion_o2"],
        target=["hipotermia_hipertermia", "hipotension", "taquipnea", "taquicardia", "hipoxemia"],
        how="recompute threshold-derived clinical flags after numeric cleaning",
    )
    add_change(
        log,
        "transformed_variables",
        source=iqr_columns,
        target=iqr_columns,
        how="replace IQR outliers with NaN",
    )
    return PreprocessResult(df=df, log=log)


def preprocess_tbl_sepsis(
    tables: dict[str, pd.DataFrame],
    maps: dict[str, dict[Any, Any]],
) -> PreprocessResult:
    source = tables["tbl_sepsis"].copy()
    df = source.copy()
    df["lactato_serico"] = np.where(df["lactato_serico"] == "<= 2 millimole per liter", 0, 1)

    for column in df.columns[2:]:
        df[column] = extract_numeric(df[column])

    if "foco" in df.columns:
        df["foco"] = pd.to_numeric(df["foco"], errors="coerce").map(maps["foco_map"])

    df = clean_outliers_iqr(df, ["proteina_c_reactiva"])

    log = finalize_log(
        table_name="tbl_sepsis",
        input_df=source,
        output_df=df,
        merge_keys=["person_id", "fecha_ingreso_urgencias"],
    )
    add_change(
        log,
        "recoded_variables",
        source="lactato_serico",
        target="lactato_serico",
        how="recode '<= 2 millimole per liter' to 0 and all other non-null values to 1",
    )
    add_change(
        log,
        "recoded_variables",
        source="foco",
        target="foco",
        how="map foco codes to readable labels using foco_map",
    )
    add_change(
        log,
        "transformed_variables",
        source="proteina_c_reactiva",
        target="proteina_c_reactiva",
        how="replace IQR outliers with NaN",
    )
    return PreprocessResult(df=df, log=log)


def preprocess_tbl_infecciones_previas(
    tables: dict[str, pd.DataFrame],
    maps: dict[str, dict[Any, Any]],
) -> PreprocessResult:
    source = tables["tbl_infecciones_previas"].copy()
    df = source.copy()
    df["grupo_microorganismo"] = df["microorganism_infec_prev"].astype(str).map(maps["organism_codes_map"])
    df.loc[df["fecha_infeccion"] == "323-05-01", "fecha_infeccion"] = "2023-05-01"

    organisms = df["grupo_microorganismo"].dropna().unique().tolist()
    pivot_source = df.drop(columns=["microorganism_infec_prev", "bmr_infec_previa", "feno_resist_infec_prev"]).copy()
    pivot_source["dummy"] = 1
    pivoted = pivot_source.pivot_table(
        index=["person_id", "fecha_ingreso_urgencias"],
        columns="grupo_microorganismo",
        values="dummy",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    for organism in organisms:
        pivoted[f"{organism}_binary"] = np.where(pivoted[organism] >= 1, 1, 0)
        pivoted = pivoted.drop(columns=organism)

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

    log = finalize_log(
        table_name="tbl_infecciones_previas",
        input_df=source,
        output_df=result,
        merge_keys=["person_id"],
    )
    add_change(
        log,
        "recoded_variables",
        source="microorganism_infec_prev",
        target="grupo_microorganismo",
        how="map infection organism codes to grouped organism labels",
    )
    add_change(
        log,
        "created_variables",
        source="grupo_microorganismo",
        target=[col for col in result.columns if col.endswith("_binary")],
        how="pivot grouped organisms and binarize per patient/admission",
    )
    add_change(
        log,
        "created_variables",
        source="fecha_infeccion",
        target=["num_inf_previas", "tiempo_ultima"],
        how="derive previous infection count and days since last infection",
    )
    add_change(
        log,
        "dropped_variables",
        source=["microorganism_infec_prev", "bmr_infec_previa", "feno_resist_infec_prev"],
        target=["grupo_microorganismo", "*_binary", "num_inf_previas", "tiempo_ultima"],
        how="replace raw detailed previous-infection records with grouped history features",
    )
    return PreprocessResult(df=result, log=log)


def preprocess_tbl_tratamiento_antibiotico_previo(
    tables: dict[str, pd.DataFrame],
    maps: dict[str, dict[Any, Any]],
) -> PreprocessResult:
    source = tables["tbl_tratamiento_antibiotico_previo"].copy()
    df = source.copy()
    log = finalize_log(
        table_name="tbl_tratamiento_antibiotico_previo",
        input_df=source,
        output_df=source,
        merge_keys=["person_id", "fecha_ingreso_urgencias"],
    )
    log.notes.append("Template only: move 90-day filtering, drug standardization, ultimo_antib, dias_ultimo_antib, family binaries, and antib_previo_total_veces here.")
    log.warnings.append("Not implemented yet in this template.")
    return PreprocessResult(df=df, log=log)


def preprocess_tbl_hemocultivo_de_urgencias(
    tables: dict[str, pd.DataFrame],
    maps: dict[str, dict[Any, Any]],
) -> PreprocessResult:
    source = tables["tbl_hemocultivo_de_urgencias"].copy()
    df = source.copy()
    log = finalize_log(
        table_name="tbl_hemocultivo_de_urgencias",
        input_df=source,
        output_df=source,
        merge_keys=["person_id", "fecha_ingreso_urgencias"],
    )
    log.notes.append("Template only: add microorganism harmonization, co-infection resolution, pivoting, resultado_hemo, bmr_etiologia aggregation, and phenotype tuple cleanup.")
    log.warnings.append("Not implemented yet in this template.")
    return PreprocessResult(df=df, log=log)


def preprocess_tbl_colonizaciones_previas(
    tables: dict[str, pd.DataFrame],
    maps: dict[str, dict[Any, Any]],
) -> PreprocessResult:
    source = tables["tbl_colonizaciones_previas"].copy()
    df = source.copy()
    log = finalize_log(
        table_name="tbl_colonizaciones_previas",
        input_df=source,
        output_df=source,
        merge_keys=["person_id"],
    )
    log.notes.append("Template only: add organism mapping, patient-level binary pivot, grouped colonization features, and colonization burden here.")
    log.warnings.append("Not implemented yet in this template.")
    return PreprocessResult(df=df, log=log)


def preprocess_tbl_otros_cultivos_en_urgencias(
    tables: dict[str, pd.DataFrame],
    maps: dict[str, dict[Any, Any]],
) -> PreprocessResult:
    source = tables["tbl_otros_cultivos_en_urgencias"].copy()
    df = source.copy()
    log = finalize_log(
        table_name="tbl_otros_cultivos_en_urgencias",
        input_df=source,
        output_df=source,
        merge_keys=["person_id", "fecha_ingreso_urgencias"],
    )
    log.notes.append("Template only: add microorganism mapping, column drops, pivoting, and helper output for combined urgent-culture dominance.")
    log.warnings.append("Not implemented yet in this template.")
    return PreprocessResult(df=df, log=log)


def merge_preprocessed_tables(results: dict[str, PreprocessResult]) -> PreprocessResult:
    master = results["tbl_paciente"].df.copy()
    input_df = master.copy()

    merge_plan = [
        ("tbl_comorbilidad", ["person_id", "fecha_ingreso_urgencias"], "left"),
        ("tbl_factores_riesgo_bmr", ["person_id", "fecha_ingreso_urgencias"], "left"),
        ("tbl_sepsis", ["person_id", "fecha_ingreso_urgencias"], "left"),
        ("tbl_signos", ["person_id", "fecha_ingreso_urgencias"], "left"),
        ("tbl_sintomas", ["person_id", "fecha_ingreso_urgencias"], "left"),
        ("tbl_infecciones_previas", ["person_id"], "left"),
        ("tbl_tratamiento_antibiotico_previo", ["person_id", "fecha_ingreso_urgencias"], "left"),
        ("tbl_colonizaciones_previas", ["person_id"], "left"),
        ("tbl_hemocultivo_de_urgencias", ["person_id", "fecha_ingreso_urgencias"], "left"),
        ("tbl_otros_cultivos_en_urgencias", ["person_id", "fecha_ingreso_urgencias"], "left"),
    ]

    for name, keys, how in merge_plan:
        master = master.merge(results[name].df, on=keys, how=how)

    log = finalize_log(
        table_name="merged_dataset",
        input_df=input_df,
        output_df=master,
        merge_keys=["person_id", "fecha_ingreso_urgencias"],
    )
    log.notes.append("Keep this stage thin. Most engineering should happen inside the table preprocessors.")
    return PreprocessResult(df=master, log=log)


def build_cross_table_features(df: pd.DataFrame) -> PreprocessResult:
    source = df.copy()
    result = df.copy()
    log = finalize_log(
        table_name="cross_table_features",
        input_df=source,
        output_df=result,
        merge_keys=["person_id", "fecha_ingreso_urgencias"],
    )
    log.notes.append("Reserve this step for features that truly depend on multiple already-preprocessed tables.")
    return PreprocessResult(df=result, log=log)


def build_targets(df: pd.DataFrame, maps: dict[str, dict[Any, Any]]) -> PreprocessResult:
    source = df.copy()
    result = df.copy()
    log = finalize_log(
        table_name="target_building",
        input_df=source,
        output_df=result,
        merge_keys=["person_id", "fecha_ingreso_urgencias"],
    )
    log.notes.append("Add final heads here: infected_yes_no, resultado_hemo_grouped, resistant_cefalosporina, df_resist, df_cefalosporinas, etc.")
    return PreprocessResult(df=result, log=log)


def run_pipeline(
    db_path: Path = DEFAULT_DB_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> PipelineArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = load_tables(db_path)
    maps = build_reference_maps(tables)

    results: dict[str, PreprocessResult] = {}
    results["tbl_paciente"] = preprocess_tbl_paciente(tables, maps)
    results["tbl_comorbilidad"] = preprocess_tbl_comorbilidad(tables, maps)
    results["tbl_factores_riesgo_bmr"] = preprocess_tbl_factores_riesgo_bmr(tables, maps)
    results["tbl_sintomas"] = preprocess_tbl_sintomas(tables, maps)
    results["tbl_signos"] = preprocess_tbl_signos(tables, maps)
    results["tbl_sepsis"] = preprocess_tbl_sepsis(tables, maps)
    results["tbl_infecciones_previas"] = preprocess_tbl_infecciones_previas(tables, maps)
    results["tbl_tratamiento_antibiotico_previo"] = preprocess_tbl_tratamiento_antibiotico_previo(tables, maps)
    results["tbl_hemocultivo_de_urgencias"] = preprocess_tbl_hemocultivo_de_urgencias(tables, maps)
    results["tbl_colonizaciones_previas"] = preprocess_tbl_colonizaciones_previas(tables, maps)
    results["tbl_otros_cultivos_en_urgencias"] = preprocess_tbl_otros_cultivos_en_urgencias(tables, maps)

    merged = merge_preprocessed_tables(results)
    cross_features = build_cross_table_features(merged.df)
    targets = build_targets(cross_features.df, maps)

    logs = [result.log for result in results.values()]
    logs.extend([merged.log, cross_features.log, targets.log])

    targets.df.to_csv(output_dir / "preprocessed_template_output.csv", index=False)
    export_logs(logs, output_dir / "preprocessing_log_summary.csv")

    with (output_dir / "preprocessing_log_detailed.json").open("w", encoding="utf-8") as fh:
        import json

        json.dump([asdict(log) for log in logs], fh, indent=2, ensure_ascii=False)

    return PipelineArtifacts(
        tables={name: result.df for name, result in results.items()} | {"final": targets.df},
        maps=maps,
        logs=logs,
    )


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
