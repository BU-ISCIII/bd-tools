# MEPRAM Project Code

This folder contains MEPRAM-specific scripts, configuration, and generated
preprocessing report artifacts.

Project-specific code belongs here when it encodes MEPRAM clinical assumptions,
MEPRAM source-table structure, or MEPRAM reporting needs. Reusable modelling
infrastructure should live in `ml_benchmark/` instead.

## Main Files

- `preprocess_pipeline.py`: builds the full and filtered MEPRAM analytical CSVs.
- `preprocess_report.py`: builds clinician-facing preprocessing report outputs.
- `build_clinician_variable_dictionary.py`: helper used by the report script.
- `compare_preprocessed_csvs.py`: local comparison helper for preprocessing QA.
- `config/preprocess_config.py`: clinical/domain preprocessing configuration.
- `config/preprocess_columns_to_drop.txt`: final filtered-output drop list.
- `config/model_filters.yml`: MEPRAM cohort/model row filters.
- `config/benchmark_mepram.yml`: MEPRAM benchmark matrix configuration.
- `preprocessing_report/`: checked-in preprocessing report artifacts.

## Typical Workflow

From the repository root:

```bash
./.venv/bin/python mepram/preprocess_pipeline.py \
  --input-path db_mepram_sepsis_vf.sqlite3 \
  --output-path preprocess_test.csv \
  --config-path mepram/config/preprocess_config.py \
  --drop-columns-path mepram/config/preprocess_columns_to_drop.txt \
  --run-label current_run
```

```bash
./.venv/bin/python mepram/preprocess_report.py \
  --summary-log-path preprocess_test_log_summary.csv \
  --detailed-log-path preprocess_test_log_detailed.json \
  --full-dataset-path preprocess_test.csv \
  --filtered-dataset-path preprocess_test_filtered.csv \
  --output-dir mepram/preprocessing_report
```
