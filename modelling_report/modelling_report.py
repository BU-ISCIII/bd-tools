#!/usr/bin/env python3
"""
Auto-generate markdown reports from three-level modelling result folders.

Example:
    python make_modelling_reports.py .
    python make_modelling_reports.py 01-training_5660567
"""

from pathlib import Path
import json
import sys


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def read_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def read_text(path: Path) -> str:
    if path and path.exists():
        try:
            return path.read_text()
        except Exception as e:
            return f"ERROR READING FILE: {e}"
    return "FILE NOT FOUND"


def find_one(folder: Path, pattern: str):
    if folder is None or not folder.exists():
        return None

    matches = sorted(folder.glob(pattern))
    return matches[0] if matches else None


def metric(d: dict, key: str) -> str:
    value = d.get(key)

    if value is None:
        return "NA"

    if isinstance(value, float):
        return f"{value:.3f}"

    return str(value)


def format_feature_block(features):
    if not features:
        return "No features found."

    return "\n".join([f"- {x}" for x in features])


# ---------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------

def make_report(training_dir: Path):

    training_dir = Path(training_dir)

    aggregate = read_json(training_dir / "aggregate_summary.json")
    final = read_json(training_dir / "final_report.json")

    l1_dir = find_one(training_dir, "level1_*")
    l2_dir = find_one(training_dir, "level2_*")
    l3_dir = find_one(training_dir, "level3_*")

    l1_summary = read_json(l1_dir / "summary.json") if l1_dir else {}
    l2_summary = read_json(l2_dir / "summary.json") if l2_dir else {}
    l3_summary = read_json(l3_dir / "summary.json") if l3_dir else {}

    # -------------------------------------------------------------
    # Reports
    # -------------------------------------------------------------

    l1_report = read_text(find_one(l1_dir, "report_*.txt"))

    l2_gate_report = read_text(
        find_one(l2_dir, "report_gate_*.txt")
    )

    l2_staged_report = read_text(
        find_one(l2_dir, "report_staged_*.txt")
    )

    l2_direct_report = read_text(
        find_one(l2_dir, "report_*.txt")
    )

    l3_report = read_text(
        find_one(l3_dir, "report_*.txt")
    )

    # -------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------

    l1 = aggregate.get("level1_sepsis", {})
    l2 = aggregate.get("level2_hemo", {})
    l3 = aggregate.get("level3_cefalosporina", {})

    l2_stage1 = l2.get("stage1_gate", {})
    l2_stage2 = l2.get("stage2_subtype", l2)

    # -------------------------------------------------------------
    # Markdown report
    # -------------------------------------------------------------

    md = f"""# Three-Level Modelling Report

## Run Information

| Field | Value |
|---|---|
| Training Folder | `{training_dir.name}` |
| Absolute Path | `{training_dir.resolve()}` |

---

# Folder Structure

```text
{training_dir.name}/
├── aggregate_summary.json
├── final_report.json
├── level1_*/
├── level2_*/
├── level3_*/
├── processing/
├── *.out
└── *.err
```
# Global Summary

| Level   | Mode                   | Macro F1                        | ROC-AUC                        | PR-AUC                        |
| ------- | ---------------------- | ------------------------------- | ------------------------------ | ----------------------------- |
| Level 1 | binary                 | {metric(l1, "macro_f1")}        | {metric(l1, "roc_auc")}        | {metric(l1, "pr_auc")}        |
| Level 2 | {l2.get("mode", "NA")} | {metric(l2_stage2, "macro_f1")} | {metric(l2_stage2, "roc_auc")} | {metric(l2_stage2, "pr_auc")} |
| Level 3 | {l3.get("mode", "NA")} | {metric(l3, "macro_f1")}        | {metric(l3, "roc_auc")}        | {metric(l3, "pr_auc")}        |

## Level 1
### Summary
| Field    | Value                                   |
| -------- | --------------------------------------- |
| Target   | `{l1_summary.get("target", "sepsis")}`  |
| Features | `{len(l1_summary.get("features", []))}` |
| Model    | `{l1_summary.get("model", "NA")}`       |

### Report
{l1_report}

### Selected Features
{format_feature_block(l1_summary.get("features", []))}

## Level 2
{l2.get("mode", "NA")}

### Stage 1: Gate
#### Summary
| Metric    | Value                            |
| --------- | -------------------------------- |
| Macro F1  | {metric(l2_stage1, "macro_f1")}  |
| ROC-AUC   | {metric(l2_stage1, "roc_auc")}   |
| PR-AUC    | {metric(l2_stage1, "pr_auc")}    |
| Precision | {metric(l2_stage1, "precision")} |
| Recall    | {metric(l2_stage1, "recall")}    |

#### Report
{l2_gate_report}

### Stage 2: Etiology
| Metric    | Value                            |
| --------- | -------------------------------- |
| Macro F1  | {metric(l2_stage2, "macro_f1")}  |
| ROC-AUC   | {metric(l2_stage2, "roc_auc")}   |
| PR-AUC    | {metric(l2_stage2, "pr_auc")}    |
| Precision | {metric(l2_stage2, "precision")} |
| Recall    | {metric(l2_stage2, "recall")}    |

#### Report
{l2_staged_report}

### Selected Features
{format_feature_block(
    l2_summary.get("stage2_subtype", l2_summary).get("features", [])
)}

## Level 3
### Summary
| Metric    | Value                     |
| --------- | ------------------------- |
| Macro F1  | {metric(l3, "macro_f1")}  |
| ROC-AUC   | {metric(l3, "roc_auc")}   |
| PR-AUC    | {metric(l3, "pr_auc")}    |
| Precision | {metric(l3, "precision")} |
| Recall    | {metric(l3, "recall")}    |

### Report
{l3_report}

### Selected Features
{format_feature_block(l3_summary.get("features", []))}

## Processing Files
| File                             | Exists                                                                        |
| -------------------------------- | ----------------------------------------------------------------------------- |
| nan_dropped_features.csv         | {(training_dir / "processing" / "nan_dropped_features.csv").exists()}         |
| imputed_features.csv             | {(training_dir / "processing" / "imputed_features.csv").exists()}             |
| correlation_dropped_features.csv | {(training_dir / "processing" / "correlation_dropped_features.csv").exists()} |

## Output Logs

### STDOUT
{read_text(find_one(training_dir, "*.out"))[:100]}

### STDERR
{read_text(find_one(training_dir, "*.err"))[:100]}
"""
    
out_file = training_dir / "modelling_report.md"
out_file.write_text(md)

print(f"[OK] Wrote report: {out_file}")

# ENTRY POINT
if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

    if root.name.startswith("01-training_"):
        make_report(root)

    else:
        training_dirs = sorted(root.glob("01-training_*"))

        if not training_dirs:
            print("No training folders found.")
            sys.exit(1)

        for td in training_dirs:
            make_report(td)