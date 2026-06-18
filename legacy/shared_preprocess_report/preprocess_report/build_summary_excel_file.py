from __future__ import annotations

import argparse
import csv
import ast
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape


FULL_DATASET = Path("preprocess_test.csv")
FILTERED_DATASET = Path("preprocess_test_filtered.csv")
DETAILED_LOG = Path("preprocess_test_log_detailed.json")
OUTPUT_XLSX = Path("preprocessing_report") / "tables" / "summary_excel_file.xlsx"


HEADERS = [
    "variable name",
    "source domain",
    "source table/stage",
    "type",
    "number of classes",
    "data type",
    "missing %",
    "min",
    "median",
    "max",
    "number of options",
    "dominant option",
    "dominant option %",
    "top 3 options",
    "description. how it's been created/transformed/recoded...",
    "dropped (yes/no)",
    "notes",
]


DEFAULT_DOMAIN_BY_STAGE = {
    "tbl_paciente": "Patient/admission",
    "tbl_comorbilidad": "Comorbidities",
    "tbl_factores_riesgo_bmr": "BMR risk factors",
    "tbl_sintomas": "Symptoms",
    "tbl_signos": "Vital signs",
    "tbl_sepsis": "Sepsis and infection focus",
    "tbl_infecciones_previas": "Previous infections",
    "tbl_tratamiento_antibiotico_previo": "Previous antibiotic treatment",
    "tbl_hemocultivo_de_urgencias": "Emergency blood cultures",
    "tbl_colonizaciones_previas": "Previous colonizations",
    "tbl_otros_cultivos_en_urgencias": "Other emergency cultures",
    "cross_table_features": "Cross-table culture features",
    "target_building": "Predictive targets",
}

# Backward-compatible alias for older imports.
DOMAIN_BY_STAGE = DEFAULT_DOMAIN_BY_STAGE


def read_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def read_csv_columns(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return [], {}
        columns = list(reader.fieldnames)
        values = {column: [] for column in columns}
        for row in reader:
            for column in columns:
                values[column].append(row.get(column, ""))
    return columns, values


def as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def source_text(value) -> str:
    values = as_list(value)
    if not values:
        return "not recorded"
    if len(values) <= 5:
        return ", ".join(values)
    return ", ".join(values[:5]) + f", ... ({len(values)} source variables)"


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def is_missing(value: str) -> bool:
    return value == "" or value.lower() in {"nan", "none", "null", "na", "<na>"}


def parse_number(value: str) -> float | None:
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def format_number(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def median(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[middle]
    return (sorted_values[middle - 1] + sorted_values[middle]) / 2


def is_tuple_like(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    if not (
        (stripped.startswith("(") and stripped.endswith(")"))
        or (stripped.startswith("[") and stripped.endswith("]"))
    ):
        return False
    try:
        parsed = ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return False
    return isinstance(parsed, (tuple, list))


def normalize_bool_token(value: str) -> str | None:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "1.0"}:
        return "true"
    if lowered in {"false", "0", "0.0"}:
        return "false"
    return None


def parse_date(value: str) -> datetime | None:
    stripped = value.strip()
    for date_format in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(stripped, date_format)
        except ValueError:
            continue
    return None


def infer_data_type(non_missing_values: list[str], numeric_values: list[float]) -> str:
    if not non_missing_values:
        return "empty"
    bool_tokens = [normalize_bool_token(value) for value in non_missing_values]
    if all(token is not None for token in bool_tokens):
        return "bool"
    if len(numeric_values) == len(non_missing_values):
        if all(value.is_integer() for value in numeric_values):
            return "int"
        return "float"
    if all(parse_date(value) is not None for value in non_missing_values):
        return "date"
    if all(is_tuple_like(value) for value in non_missing_values):
        return "tuple"
    if numeric_values:
        return "mixed"
    return "string"


def profile_values(column: str, values: list[str]) -> dict[str, str]:
    total = len(values)
    non_missing = [value for value in values if not is_missing(value)]
    missing_count = total - len(non_missing)
    missing_percent = (missing_count / total * 100) if total else 0
    numeric_values = [
        parsed
        for value in non_missing
        if (parsed := parse_number(value)) is not None
    ]
    data_type = infer_data_type(non_missing, numeric_values)

    profile = {
        "data_type": data_type,
        "missing_percent": f"{missing_percent:.1f}",
        "min": "",
        "median": "",
        "max": "",
        "number_options": "",
        "dominant_option": "",
        "dominant_option_percent": "",
        "top_3_options": "",
    }
    if data_type in {"int", "float"} and numeric_values:
        profile["min"] = format_number(min(numeric_values))
        profile["median"] = format_number(median(numeric_values))
        profile["max"] = format_number(max(numeric_values))

    counter = Counter(non_missing)
    if counter and data_type in {"bool", "string", "tuple", "mixed"}:
        dominant, count = counter.most_common(1)[0]
        profile["number_options"] = str(len(counter))
        profile["dominant_option"] = dominant
        profile["dominant_option_percent"] = f"{(count / len(non_missing) * 100):.1f}"
        profile["top_3_options"] = "; ".join(
            f"{option}: {(option_count / len(non_missing) * 100):.1f}%"
            for option, option_count in counter.most_common(3)
        )
    return profile


def operation_sentence(stage_name: str, operation: dict) -> str:
    kind = str(operation.get("kind", "")).replace("_variables", "").replace("_", " ")
    how = clean_text(operation.get("how", ""))
    source = source_text(operation.get("source"))
    if kind:
        prefix = f"{kind.capitalize()} in {stage_name}"
    else:
        prefix = f"Processed in {stage_name}"
    if how:
        return f"{prefix} from {source}: {how}."
    return f"{prefix} from {source}."


def class_count_for_target(operation: dict, target: str) -> str:
    n_classes = operation.get("n_classes")
    if isinstance(n_classes, dict):
        value = n_classes.get(target)
    else:
        value = n_classes
    if value is None or value == "":
        return ""
    return str(value)


def format_class_count_display(
    *,
    class_count: str,
    variable_type: str,
    profile: dict[str, str],
) -> str:
    if not class_count:
        return ""
    if variable_type == "target" and profile["data_type"] == "tuple":
        option_count = profile["number_options"]
        if option_count:
            return f"{class_count} exploded labels; {option_count} row-level combinations"
        return f"{class_count} exploded labels"
    return class_count


def retained_variable_description(stage_name: str | None) -> str:
    if stage_name:
        return f"Original or retained variable from {stage_name}; no additional variable-level transformation was recorded in the preprocessing log."
    return "Variable present in the full preprocessed dataset; no variable-level transformation was recorded in the preprocessing log."


def infer_source_domain(
    stage_name: str | None,
    domain_by_stage: dict[str, str] | None = None,
) -> str:
    if not stage_name:
        return "Unmapped"
    mapping = domain_by_stage or DEFAULT_DOMAIN_BY_STAGE
    return mapping.get(stage_name, stage_name.replace("_", " "))


def build_rows(
    *,
    full_dataset_path: Path = FULL_DATASET,
    filtered_dataset_path: Path = FILTERED_DATASET,
    detailed_log_path: Path = DETAILED_LOG,
    domain_by_stage: dict[str, str] | None = None,
) -> list[list[str]]:
    full_columns, full_values = read_csv_columns(full_dataset_path)
    filtered_columns = set(read_header(filtered_dataset_path))
    log = json.loads(detailed_log_path.read_text(encoding="utf-8"))

    descriptions: dict[str, list[str]] = defaultdict(list)
    variable_types: dict[str, str] = {}
    class_counts: dict[str, str] = {}
    first_stage_by_column: dict[str, str] = {}

    for stage in log:
        stage_name = stage.get("table_name", "")
        if stage_name in {"_pipeline_run", "merged_dataset", "filtered_dataset"}:
            continue
        for column in stage.get("output_columns", []):
            first_stage_by_column.setdefault(column, stage_name)

        for key in ("created_variables", "recoded_variables", "transformed_variables", "role_variables"):
            for operation in stage.get(key, []):
                sentence = operation_sentence(stage_name, operation)
                logged_descriptions = operation.get("descriptions", {}) or {}
                for target in as_list(operation.get("target")):
                    if "*" in target:
                        continue
                    description = logged_descriptions.get(target)
                    descriptions[target].append(description or sentence)
                    if operation.get("variable_type"):
                        variable_types[target] = str(operation["variable_type"])
                        class_count = class_count_for_target(operation, target)
                        if class_count:
                            class_counts[target] = class_count

    final_dropped: set[str] = set()
    for stage in log:
        if stage.get("table_name") != "filtered_dataset":
            continue
        for group in stage.get("columns_dropped", []):
            final_dropped.update(as_list(group))

    for column in final_dropped:
        descriptions[column].append(
            "Removed from the clinician/model-compatible filtered dataset by the final drop list in */config/preprocess_columns_to_drop.txt."
        )

    column_positions = {column: index for index, column in enumerate(full_columns)}
    ordered_columns = sorted(
        full_columns,
        key=lambda column: (
            variable_types.get(column, "feature") == "target",
            column_positions[column],
        ),
    )

    rows = [HEADERS]
    for column in ordered_columns:
        desc_parts = descriptions.get(column) or [
            retained_variable_description(first_stage_by_column.get(column))
        ]
        description = " ".join(dict.fromkeys(clean_text(part) for part in desc_parts if clean_text(part)))
        dropped = "yes" if column not in filtered_columns else "no"
        variable_type = variable_types.get(column, "feature")
        profile = profile_values(column, full_values[column])
        class_count = format_class_count_display(
            class_count=class_counts.get(column, ""),
            variable_type=variable_type,
            profile=profile,
        )
        rows.append([
            column,
            infer_source_domain(
                first_stage_by_column.get(column),
                domain_by_stage=domain_by_stage,
            ),
            first_stage_by_column.get(column, ""),
            variable_type,
            class_count,
            profile["data_type"],
            profile["missing_percent"],
            profile["min"],
            profile["median"],
            profile["max"],
            profile["number_options"],
            profile["dominant_option"],
            profile["dominant_option_percent"],
            profile["top_3_options"],
            description,
            dropped,
            "",
        ])
    return rows


def column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def cell_xml(row_index: int, column_index: int, value: str, style: int = 0) -> str:
    ref = f"{column_name(column_index)}{row_index}"
    escaped = escape(str(value), {'"': "&quot;"})
    style_attr = f' s="{style}"' if style else ""
    return f'<c r="{ref}" t="inlineStr"{style_attr}><is><t>{escaped}</t></is></c>'


def sheet_xml(rows: list[list[str]]) -> str:
    row_xml = []
    for row_index, row in enumerate(rows, start=1):
        style = 1 if row_index == 1 else 0
        cells = "".join(cell_xml(row_index, column_index, value, style) for column_index, value in enumerate(row, start=1))
        row_xml.append(f'<row r="{row_index}">{cells}</row>')

    last_row = len(rows)
    last_col = column_name(len(HEADERS))
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <dimension ref="A1:{last_col}{last_row}"/>
  <sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/><selection pane="bottomLeft" activeCell="A2" sqref="A2"/></sheetView></sheetViews>
  <sheetFormatPr defaultRowHeight="15"/>
  <cols>
    <col min="1" max="1" width="38" customWidth="1"/>
    <col min="2" max="2" width="30" customWidth="1"/>
    <col min="3" max="3" width="34" customWidth="1"/>
    <col min="4" max="4" width="14" customWidth="1"/>
    <col min="5" max="5" width="18" customWidth="1"/>
    <col min="6" max="6" width="14" customWidth="1"/>
    <col min="7" max="7" width="12" customWidth="1"/>
    <col min="8" max="10" width="14" customWidth="1"/>
    <col min="11" max="11" width="18" customWidth="1"/>
    <col min="12" max="12" width="35" customWidth="1"/>
    <col min="13" max="13" width="18" customWidth="1"/>
    <col min="14" max="14" width="70" customWidth="1"/>
    <col min="15" max="15" width="110" customWidth="1"/>
    <col min="16" max="16" width="16" customWidth="1"/>
    <col min="17" max="17" width="45" customWidth="1"/>
  </cols>
  <sheetData>{''.join(row_xml)}</sheetData>
  <autoFilter ref="A1:{last_col}{last_row}"/>
</worksheet>'''


def write_xlsx(rows: list[list[str]], output_path: Path) -> None:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>""")
        zf.writestr("_rels/.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>""")
        zf.writestr("xl/workbook.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="Variables" sheetId="1" r:id="rId1"/></sheets>
</workbook>""")
        zf.writestr("xl/_rels/workbook.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>""")
        zf.writestr("xl/styles.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts>
  <fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1" applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf></cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>""")
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml(rows))
        zf.writestr("docProps/core.xml", f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Preprocessing summary Excel file</dc:title><dc:creator>preprocessing pipeline</dc:creator><cp:lastModifiedBy>preprocessing pipeline</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>""")
        zf.writestr("docProps/app.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Python</Application>
</Properties>""")


def build_summary_excel_file(
    *,
    full_dataset_path: Path = FULL_DATASET,
    filtered_dataset_path: Path = FILTERED_DATASET,
    detailed_log_path: Path = DETAILED_LOG,
    output_path: Path = OUTPUT_XLSX,
    domain_by_stage: dict[str, str] | None = None,
) -> tuple[int, int, int]:
    rows = build_rows(
        full_dataset_path=full_dataset_path,
        filtered_dataset_path=filtered_dataset_path,
        detailed_log_path=detailed_log_path,
        domain_by_stage=domain_by_stage,
    )
    write_xlsx(rows, output_path)
    dropped_col = HEADERS.index("dropped (yes/no)")
    kept = sum(1 for row in rows[1:] if row[dropped_col] == "no")
    dropped = sum(1 for row in rows[1:] if row[dropped_col] == "yes")
    return len(rows) - 1, kept, dropped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build preprocessing summary Excel file from preprocessing outputs."
    )
    parser.add_argument("--full-dataset-path", type=Path, default=FULL_DATASET)
    parser.add_argument("--filtered-dataset-path", type=Path, default=FILTERED_DATASET)
    parser.add_argument("--detailed-log-path", type=Path, default=DETAILED_LOG)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_XLSX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total, kept, dropped = build_summary_excel_file(
        full_dataset_path=args.full_dataset_path,
        filtered_dataset_path=args.filtered_dataset_path,
        detailed_log_path=args.detailed_log_path,
        output_path=args.output_path,
    )
    print(f"Wrote {args.output_path}")
    print(f"Variables: {total}; kept: {kept}; dropped: {dropped}")


if __name__ == "__main__":
    main()
