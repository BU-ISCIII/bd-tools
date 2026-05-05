# Predictive ML Benchmark

Reusable benchmark package for comparing predictive models across targets,
model families, and feature-selection strategies.

The package is intentionally outside `scripts/hpc_processing` so the existing
HPC scripts can remain as historical/project-specific workflows while this
package becomes the reusable implementation.

## Current Scope

This first slice establishes:

- YAML config loading.
- `target x model x feature_view x feature_set` job-matrix expansion.
- Manual feature groups and feature views for comparing alternative clinical
  representations, such as raw, binary, or categorical vital-sign encodings.
- Model-specific feature policies for scaling, categorical handling, and
  correlation-filter behavior.
- Slurm array helper output.
- Slurm script rendering.
- Resource limits based on `SLURM_CPUS_PER_TASK`.
- Model registry stubs for dummy, logistic regression, random forest, CatBoost,
  LightGBM, and XGBoost.
- Homogeneous job output contract with `summary.json` and
  `diagnostics_manifest.json`.

Training, feature-selection caching, tuning, metrics, diagnostics, and aggregate
reporting are the next implementation slices.

## Helper Commands

Print the job matrix:

```bash
python scripts/run_benchmark.py \
  --config scripts/ml_benchmark/configs/benchmark_mepram.yml \
  --print-job-matrix
```

Print the Slurm array line:

```bash
python scripts/run_benchmark.py \
  --config scripts/ml_benchmark/configs/benchmark_mepram.yml \
  --print-slurm-array
```

Print a full Slurm script:

```bash
python scripts/run_benchmark.py \
  --config scripts/ml_benchmark/configs/benchmark_mepram.yml \
  --print-slurm-script
```

Run one matrix entry in dry-run mode:

```bash
python scripts/run_benchmark.py \
  --config scripts/ml_benchmark/configs/benchmark_mepram.yml \
  --array-index 15 \
  --dry-run
```

## Feature Views

Feature views are different representations of the same analytical table.
They are intentionally separate from feature selection:

- `feature_view`: which candidate columns are allowed into the model.
- `feature_set`: whether all candidates are used or a selector/cap is applied.

Manual feature groups define mutually meaningful alternatives:

```yaml
feature_groups:
  temperature:
    alternatives:
      raw:
        include: [temperatura]
        exclude: [temperatura_recoded, sintoma_fiebre, sintoma_fiebre_categorico]
      category:
        include: [temperatura_recoded, sintoma_fiebre_categorico]
        exclude: [temperatura, sintoma_fiebre]

feature_views:
  - name: raw_vitals
    include_patterns: ["*"]
    groups:
      temperature: raw
```

## Feature Policies

Feature policies define model-specific preprocessing rules. For example,
logistic regression can report/drop correlated features more aggressively while
tree models can keep correlation filtering as report-only or disabled:

```yaml
feature_policies:
  logistic_default:
    categorical_handling: one_hot
    scale: standard
    correlation_filter:
      enabled: true
      threshold: 0.80
      mode: report_then_drop
      manual_groups_first: true

  tree_default:
    categorical_handling: preprocessed
    scale: none
    correlation_filter:
      enabled: false
      mode: report_only

models:
  - name: logistic
    feature_policy: logistic_default
  - name: lightgbm
    feature_policy: tree_default
```
