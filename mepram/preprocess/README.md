# MEPRAM Project Code

This folder contains MEPRAM-specific scripts, configuration, and generated
preprocessing report artifacts.

Project-specific code belongs here when it encodes MEPRAM clinical assumptions,
MEPRAM source-table structure, or MEPRAM reporting needs. Reusable modelling
infrastructure should live in `ml_benchmark/` instead.

## Main Files

- `preprocess_pipeline.py`: builds the full MEPRAM CSV and one filtered/view CSV selected by `--feature-approach`.
- `compare_preprocessed_csvs.py`: local comparison helper for preprocessing QA.
- `config/preprocess_config.py`: clinical/domain preprocessing configuration.
- `config/preprocess_report.yml`: preprocessing report configuration for MEPRAM-specific domains and target sections.
- `config/preprocess_columns_to_drop.txt`: legacy compatibility drop list mirrored in `config/preprocess_feature_approach.yml`.
- `config/preprocess_feature_approach.yml`: feature-approach selection configuration used by preprocessing.
- `config/model_filters.yml`: MEPRAM cohort/model row filters.
- `config/benchmark_mepram.yml`: MEPRAM benchmark matrix configuration.
- `preprocessing_report/`: checked-in preprocessing report artifacts.

Reusable preprocessing report code lives in `../preprocess_report/`. This
folder only keeps the MEPRAM report config and generated MEPRAM report outputs.

## Typical Workflow

From the repository root:

```bash
./.venv/bin/python mepram/preprocess_pipeline.py \
  --input-path db_mepram_sepsis_vf.sqlite3 \
  --output-path preprocess_test.csv \
  --config-path mepram/config/preprocess_config.py \
  --drop-columns-path mepram/config/preprocess_columns_to_drop.txt \
  --feature-approach clinical \
  --feature-approach-config-path mepram/config/preprocess_feature_approach.yml \
  --run-label current_run
```

```bash
./.venv/bin/python preprocess_report/preprocess_report.py \
  --summary-log-path preprocess_test_log_summary.csv \
  --detailed-log-path preprocess_test_log_detailed.json \
  --full-dataset-path preprocess_test.csv \
  --filtered-dataset-path preprocess_test_filtered.csv \
  --report-config-path mepram/config/preprocess_report.yml \
  --output-dir mepram/preprocessing_report
```


## Pipeline Outputs

A single preprocessing run writes these artifacts next to the configured
`--output-path`:

- `<name>.csv`: full preprocessed table.
- `<name>_filtered.csv`: one filtered/view CSV selected by `--feature-approach`.

The output depends on `--feature-approach`:

- `clinical`: the default cleaned filtered output using the YAML feature view.
- `none`: the legacy drop-list output from `preprocess_columns_to_drop.txt` plus the YAML legacy-drop section.
- `raw`, `binary`, `categorical`: the other feature views defined in `config/preprocess_feature_approach.yml`.


