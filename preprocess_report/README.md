# Preprocessing Report

Reusable report generator for preprocessing outputs that follow the project log
contract:

- stage summary CSV;
- detailed JSON transformation log;
- full preprocessed CSV;
- filtered preprocessed CSV;
- YAML report configuration.

The report code is project-agnostic. Project-specific domain labels, target
sections, colors, and plot exclusions belong in a project YAML file, for
example `mepram/config/preprocess_report.yml`.

## Run

From the repository root:

```bash
./.venv/bin/python preprocess_report/preprocess_report.py \
  --summary-log-path preprocess_test_log_summary.csv \
  --detailed-log-path preprocess_test_log_detailed.json \
  --full-dataset-path preprocess_test.csv \
  --filtered-dataset-path preprocess_test_filtered.csv \
  --report-config-path mepram/config/preprocess_report.yml \
  --output-dir mepram/preprocessing_report
```

Use `--skip-summary-excel` to regenerate markdown, CSV tables, and figures
without rebuilding `tables/summary_excel_file.xlsx`.

## Outputs

The script writes:

- `preprocessing_report.md`;
- `graphs/*.png`;
- `tables/*.csv`;
- `tables/summary_excel_file.xlsx`.

The Excel summary uses explicit variable descriptions recorded in the detailed
preprocessing log. It does not contain project-specific fallback description
rules.
