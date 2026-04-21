from __future__ import annotations

# Versioned clinical/domain configuration for preprocessing.
# Keep merge rules, drop decisions, and pipeline control flow in preprocess_pipeline.py.

CONFIG_NAME = "default_preprocess_config"
CONFIG_VERSION = "0.1.0"

IQR_DEFAULT_MULTIPLIER = 3.0

SIGNOS_NUMERIC_COLUMNS = [
    "temperatura",
    "frec_respiratoria",
    "frec_cardiaca",
    "tension_arterial",
    "saturacion_o2",
]

SEPSIS_NUMERIC_COLUMNS = [
    "lactato_serico",
    "proteina_c_reactiva",
]

SIGNOS_THRESHOLDS = {
    "temperatura_fever_min": 38.0,
    "temperatura_normal_min": 36.0,
    "tension_arterial_hypotension_max": 100.0,
    "frec_respiratoria_taquipnea_min": 20.0,
    "frec_cardiaca_taquicardia_min": 90.0,
    "saturacion_o2_hipoxemia_max": 90.0,
}

SINTOMA_SHORT_DURATION_MAX_DAYS = 7

SEPSIS_OUTLIER_COLUMNS = ["proteina_c_reactiva"]
SIGNOS_OUTLIER_COLUMNS = [
    "temperatura",
    "frec_respiratoria",
    "frec_cardiaca",
    "tension_arterial",
    "saturacion_o2",
]
SEPSIS_QUARTILE_RECODE_COLUMNS = SEPSIS_OUTLIER_COLUMNS
SIGNOS_QUARTILE_RECODE_COLUMNS = SIGNOS_OUTLIER_COLUMNS

FECHA_INFECCION_CORRECTIONS = {
    "323-05-01": "2023-05-01",
}

MICROORGANISM_LABEL_MAP = {
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


def as_dict() -> dict:
    return {
        "CONFIG_NAME": CONFIG_NAME,
        "CONFIG_VERSION": CONFIG_VERSION,
        "IQR_DEFAULT_MULTIPLIER": IQR_DEFAULT_MULTIPLIER,
        "SIGNOS_NUMERIC_COLUMNS": SIGNOS_NUMERIC_COLUMNS,
        "SEPSIS_NUMERIC_COLUMNS": SEPSIS_NUMERIC_COLUMNS,
        "SIGNOS_THRESHOLDS": SIGNOS_THRESHOLDS,
        "SINTOMA_SHORT_DURATION_MAX_DAYS": SINTOMA_SHORT_DURATION_MAX_DAYS,
        "SEPSIS_OUTLIER_COLUMNS": SEPSIS_OUTLIER_COLUMNS,
        "SIGNOS_OUTLIER_COLUMNS": SIGNOS_OUTLIER_COLUMNS,
        "SEPSIS_QUARTILE_RECODE_COLUMNS": SEPSIS_QUARTILE_RECODE_COLUMNS,
        "SIGNOS_QUARTILE_RECODE_COLUMNS": SIGNOS_QUARTILE_RECODE_COLUMNS,
        "FECHA_INFECCION_CORRECTIONS": FECHA_INFECCION_CORRECTIONS,
        "MICROORGANISM_LABEL_MAP": MICROORGANISM_LABEL_MAP,
    }
