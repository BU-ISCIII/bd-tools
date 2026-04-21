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
PRIOR_ANTIBIOTIC_WINDOW_DAYS = 90

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

ANTIMICROBIAL_GROUPS = [
    ("AMIKACINA", "Aminoglucosidos", "amikacin"),
    ("AMOXICILINA", "Penicilinas", "amoxicillin"),
    ("AMOXICILINA / CLAVULANICO", "Penicilinas", "amoxicillin and beta-lactamase inhibitor"),
    ("AMPICILINA", "Penicilinas", "ampicillin"),
    ("AZITROMICINA", "Macrolidos", "azithromycin"),
    ("AZTREONAM", "Monobactamicos", "aztreonam"),
    ("BENCILPENICILINA", "Penicilinas", None),
    ("BENCILPENICILINA-BENZATINA", "Penicilinas", None),
    ("CEFADROXILO MONOHIDRATO", "Cefalosporinas 1 gen", "cefadroxil"),
    ("CEFAZOLINA", "Cefalosporinas 1 gen", "cefazolin"),
    ("CEFEPIMA", "Cefalosporinas 4 gen", "cefepime"),
    ("CEFIDEROCOL", "Cefalosporinas 4 gen", None),
    ("CEFIXIMA", "Cefalosporinas 3 gen", "cefixime"),
    ("CEFOTAXIMA", "Cefalosporinas 3 gen", "cefotaxime"),
    ("CEFTAROLINA FOSAMILO", "Cefalosporinas 2 gen", "ceftaroline fosamil"),
    ("CEFTAZIDIMA", "Cefalosporinas 3 gen", "ceftazidime"),
    ("CEFTAZIDIMA / AVIBACTAM", "Cefalosporinas 3 gen", "ceftazidime and beta-lactamase inhibitor"),
    ("CEFTOLOZANO / TAZOBACTAM", "Cefalosporinas 3 gen", "ceftolozane and beta-lactamase inhibitor"),
    ("CEFTRIAXONA", "Cefalosporinas 3 gen", "ceftriaxone"),
    ("CEFUROXIMA", "Cefalosporinas 2 gen", "cefuroxime"),
    ("CIPROFLOXACINO", "Quinolonas", "ciprofloxacin"),
    ("CLARITROMICINA", "Macrolidos", "clarithromycin"),
    ("CLINDAMICINA", "Lincosamidas", "clindamycin"),
    ("CLOXACILINA", "Penicilinas", "cloxacillin"),
    ("COLISTIMETATO DE SODIO", "Polimixinas", None),
    ("DALBAVANCINA", "Glicopeptidos", "dalbavancin"),
    ("DAPTOMICINA", "Lipopeptidos", "daptomycin"),
    ("DOXICICLINA", "Tetraciclinas", "doxycycline"),
    ("ERITROMICINA", "Macrolidos", "erythromycin"),
    ("ERTAPENEM", "Carbapenemas", "ertapenem"),
    ("FIDAXOMICINA", "Macrolidos", None),
    ("FOSFOMICINA", "Fosfomicina", "fosfomycin"),
    ("FOSFOMICINA-TROMETAMOL", "Fosfomicina", "fosfomycin"),
    ("GENTAMICINA", "Aminoglucosidos", "gentamicin"),
    ("IMIPENEM", "Carbapenemas", "imipenem and cilastatin"),
    ("IMIPENEM/RELEBACTAM", "Carbapenemas", None),
    ("LEVOFLOXACINO", "Quinolonas", "levofloxacin"),
    ("LINEZOLID", "Lincosamidas", "linezolid"),
    ("MEROPENEM", "Carbapenemas", "meropenem"),
    ("METRONIDAZOL", "Metronidazol", "metronidazole"),
    ("MINOCICLINA", "Minociclina", None),
    ("MOXIFLOXACINO", "Quinolonas", "moxifloxacin"),
    ("NITROFURANTOINA", "Nitrofurantoina", "nitrofurantoin"),
    ("NORFLOXACINO", "Quinolonas", "norfloxacin"),
    ("PIPERACILINA / TAZOBACTAM", "Penicilinas", "piperacillin and beta-lactamase inhibitor"),
    ("POSACONAZOL", "Azoles", "posaconazole"),
    ("SULFADIAZINA", "Sulfonamidas", "sulfadiazine"),
    ("SULFAMETOXAZOL / TRIMETOPRIMA", "Sulfonamidas", "sulfamethoxazole and trimethoprim"),
    ("TRIMETOPRIMA", "Sulfonamidas", "trimethoprim"),
    ("TEICOPLANINA", "Glicopeptidos", "teicoplanin"),
    ("TIGECICLINA", "Tigeciclina", "tigecycline"),
    ("TOBRAMICINA", "Aminoglucosidos", "tobramycin"),
    ("VANCOMICINA", "Glicopeptidos", "vancomycin"),
    ("A", "Tetraciclinas", "tetracycline"),
    ("A", "Cefalosporinas 3 gen", "cefditoren"),
    ("A", "Carbapenemas", "meropenem and vaborbactam"),
    ("A", "Cefalosporinas 1 gen", "cefalexin"),
    ("A", "Cefalosporinas 2 gen", "cefoxitin"),
    ("A", "AntiTuberculoso", "rifampicin"),
    ("A", "AntiTuberculoso", "rifabutin"),
    ("A", "AntiTuberculoso", "isoniazid"),
    ("A", "AntiTuberculoso", "pyrazinamide"),
    ("A", "AntiTuberculoso", "ethambutol"),
    ("B", "Azoles", "isavuconazole"),
    ("B", "Azoles", "fluconazole"),
    ("B", "Azoles", "itraconazole"),
    ("B", "Azoles", "voriconazole"),
    ("B", "Equinocandinas", "caspofungin"),
    ("B", "Equinocandinas", "micafungin"),
    ("B", "Equinocandinas", "anidulafungin"),
]


def as_dict() -> dict:
    return {
        "CONFIG_NAME": CONFIG_NAME,
        "CONFIG_VERSION": CONFIG_VERSION,
        "IQR_DEFAULT_MULTIPLIER": IQR_DEFAULT_MULTIPLIER,
        "SIGNOS_NUMERIC_COLUMNS": SIGNOS_NUMERIC_COLUMNS,
        "SEPSIS_NUMERIC_COLUMNS": SEPSIS_NUMERIC_COLUMNS,
        "SIGNOS_THRESHOLDS": SIGNOS_THRESHOLDS,
        "SINTOMA_SHORT_DURATION_MAX_DAYS": SINTOMA_SHORT_DURATION_MAX_DAYS,
        "PRIOR_ANTIBIOTIC_WINDOW_DAYS": PRIOR_ANTIBIOTIC_WINDOW_DAYS,
        "SEPSIS_OUTLIER_COLUMNS": SEPSIS_OUTLIER_COLUMNS,
        "SIGNOS_OUTLIER_COLUMNS": SIGNOS_OUTLIER_COLUMNS,
        "SEPSIS_QUARTILE_RECODE_COLUMNS": SEPSIS_QUARTILE_RECODE_COLUMNS,
        "SIGNOS_QUARTILE_RECODE_COLUMNS": SIGNOS_QUARTILE_RECODE_COLUMNS,
        "FECHA_INFECCION_CORRECTIONS": FECHA_INFECCION_CORRECTIONS,
        "MICROORGANISM_LABEL_MAP": MICROORGANISM_LABEL_MAP,
        "ANTIMICROBIAL_GROUPS": ANTIMICROBIAL_GROUPS,
    }
