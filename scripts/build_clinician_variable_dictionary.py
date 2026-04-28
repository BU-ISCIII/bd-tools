from __future__ import annotations

import csv
import json
import re
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
FULL_DATASET = ROOT / "preprocess_test.csv"
FILTERED_DATASET = ROOT / "preprocess_test_filtered.csv"
DETAILED_LOG = ROOT / "preprocess_test_log_detailed.json"
OUTPUT_XLSX = ROOT / "report" / "tables" / "clinician_variable_dictionary.xlsx"


HEADERS = ["variable name", "description. how it's been created/transformed/recoded...", "dropped (yes/no)", "notes"]


def read_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


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


def infer_description(column: str, stage_name: str | None) -> str:
    if column == "person_id":
        return "Patient identifier retained to link all preprocessed tables."
    if column.startswith("tipo_cancer_"):
        return "Cancer ICD/category dummy created from tipo_cancer one-hot encoding."
    if column.startswith("tipo_hepatopatia_"):
        return "Liver disease ICD/category dummy created from tipo_hepatopatia one-hot encoding."
    if column.startswith("sintoma_") and column.endswith("_categorico"):
        return "Symptom duration category after pivoting symptoms to patient-level columns: 0 absent, 1 up to 7 days, 2 more than 7 days."
    if column.startswith("sintoma_"):
        return "Symptom presence flag after pivoting symptoms to patient-level columns: 1 if symptom duration was recorded, else 0."
    if column.startswith("antib_previo_") and column.endswith("_counts"):
        return "Count of previous antibiotic exposures in this antimicrobial family within the configured prior-antibiotic window."
    if column.startswith("antib_previo_") and column.endswith("_binary"):
        return "Binary flag for any previous antibiotic exposure in this antimicrobial family within the configured prior-antibiotic window."
    if column.startswith(("infprev_", "hemo_", "colo_", "otros_cult_", "all_cult_")):
        if column.endswith("_binary"):
            return "Organism or resistance binary indicator derived from culture/infection records after microorganism grouping."
        if column.endswith("_count"):
            return "Organism count derived from culture records after microorganism grouping."
        if column.endswith("_tuple"):
            return "Tuple of resistance phenotype labels derived from culture/infection records."
    if column.startswith("foco_") and column.endswith("_binary"):
        return "Infection focus dummy variable generated from the focus/foco category."
    if column.endswith("_recoded"):
        return "Ordinal recoded version of the corresponding numeric variable after outlier handling and quartile binning."
    if column.endswith("_total"):
        return "Cross-table total count/summary feature for this grouped organism across available culture tables."
    if column in {
        "resultado_hemo_mo",
        "resultado_hemo",
        "resultado_hemo_grouped",
        "resultado_hemo_multilabel",
        "fenotipo_resistencia",
        "fenotipo_resistencia_individual",
        "resistente_cefalosporina",
        "resistente_cefalosporina_multi",
        "sepsis",
        "bmr_etiologia",
    }:
        return "Prediction target or target-support variable derived during target building."
    if stage_name:
        return f"Original or retained variable from {stage_name}; no additional variable-level transformation was recorded in the preprocessing log."
    return "Variable present in the full preprocessed dataset; no variable-level transformation was recorded in the preprocessing log."


def build_rows() -> list[list[str]]:
    full_columns = read_header(FULL_DATASET)
    filtered_columns = set(read_header(FILTERED_DATASET))
    log = json.loads(DETAILED_LOG.read_text(encoding="utf-8"))

    descriptions: dict[str, list[str]] = defaultdict(list)
    first_stage_by_column: dict[str, str] = {}

    for stage in log:
        stage_name = stage.get("table_name", "")
        if stage_name in {"_pipeline_run", "merged_dataset", "filtered_dataset"}:
            continue
        for column in stage.get("output_columns", []):
            first_stage_by_column.setdefault(column, stage_name)

        for key in ("created_variables", "recoded_variables", "transformed_variables"):
            for operation in stage.get(key, []):
                sentence = operation_sentence(stage_name, operation)
                for target in as_list(operation.get("target")):
                    if "*" in target:
                        continue
                    descriptions[target].append(sentence)

    final_dropped: set[str] = set()
    for stage in log:
        if stage.get("table_name") != "filtered_dataset":
            continue
        for group in stage.get("columns_dropped", []):
            final_dropped.update(as_list(group))

    for column in final_dropped:
        descriptions[column].append(
            "Removed from the clinician/model-compatible filtered dataset by the final drop list in data/preprocess_columns_to_drop.txt."
        )

    rows = [HEADERS]
    for column in full_columns:
        desc_parts = descriptions.get(column) or [infer_description(column, first_stage_by_column.get(column))]
        description = " ".join(dict.fromkeys(clean_text(part) for part in desc_parts if clean_text(part)))
        dropped = "yes" if column not in filtered_columns else "no"
        rows.append([column, description, dropped, ""])
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
    <col min="2" max="2" width="110" customWidth="1"/>
    <col min="3" max="3" width="16" customWidth="1"/>
    <col min="4" max="4" width="45" customWidth="1"/>
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
  <dc:title>Clinician variable dictionary</dc:title><dc:creator>preprocessing pipeline</dc:creator><cp:lastModifiedBy>preprocessing pipeline</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>""")
        zf.writestr("docProps/app.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Python</Application>
</Properties>""")


def main() -> None:
    rows = build_rows()
    write_xlsx(rows, OUTPUT_XLSX)
    kept = sum(1 for row in rows[1:] if row[2] == "no")
    dropped = sum(1 for row in rows[1:] if row[2] == "yes")
    print(f"Wrote {OUTPUT_XLSX}")
    print(f"Variables: {len(rows) - 1}; kept: {kept}; dropped: {dropped}")


if __name__ == "__main__":
    main()
