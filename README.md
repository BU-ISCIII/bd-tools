# Repository Structure

This repository contains the current MEPRAM modelling and preprocessing code, benchmarking utilities, exploratory analyses, and older legacy implementations.

```text
bd-tools/
├── env/
├── mepram/
├── bacthecom/
└── legacy/
```

## `mepram/`

Main package containing the current MEPRAM implementation.

```text
mepram/
├── modelling/
├── ml_benchmark/
├── preprocess/
├── playground/
└── tests/
```

### `mepram/modelling/`

Core modelling pipeline.

Contains:

* model training and evaluation
* CLI entry points
* configuration handling
* feature filtering and selection
* Optuna optimization
* out-of-fold predictions
* SHAP utilities
* plotting and reporting

Main entry points include:

```text
cli.py
__main__.py
modelling.py
plot_results.py
report_results.py
```

### `mepram/ml_benchmark/`

Reusable framework for benchmarking modelling configurations.

Contains:

* benchmark configuration
* job matrix generation
* feature views
* model definitions
* pipeline execution
* result aggregation/reporting
* SLURM job generation

See `mepram/ml_benchmark/README.md` for details.

### `mepram/preprocess/`

Current preprocessing pipeline and preprocessing reports.

Contains utilities for:

* dataset preprocessing
* preprocessing reports
* comparison of preprocessed CSV files
* generation of summary Excel files

See `mepram/preprocess/README.md`.

### `mepram/playground/`

Exploratory notebooks and temporary analyses.

Examples include:

* dimensionality-reduction analysis
* hemoculture gate exploration
* prediction-error analysis
* SHAP-based staged-model analysis

Code in this directory should generally not be considered part of the stable pipeline.

### `mepram/tests/`

Tests for the current MEPRAM package.

---

## `bacthecom/`

Parallel BACTHECOM-specific implementations and utilities.

```text
bacthecom/
├── ml_benchmark/
├── modelling/
├── preprocess/
├── playground/
└── tests/
```

`bacthecom/ml_benchmark/` follows approximately the same reusable benchmark structure as the MEPRAM implementation.

---

## `env/`

Environment configuration files:

```text
env/
├── modelling.yml
├── preprocessing.yml
└── web.yml
```

These define the software environments required by the different repository components.

---

## `legacy/`

Older code retained for reference and reproducibility.

```text
legacy/
├── hpc_processing/
├── jupyter_notebooks/
├── shared_modelling/
├── shared_preprocess_report/
└── web/
```

### `legacy/hpc_processing/`

Previous HPC modelling workflows, including binary, multiclass, multilabel, hierarchical and cascade modelling approaches.

### `legacy/jupyter_notebooks/`

Historical preprocessing, modelling, optimization and feature-reduction notebooks.

### Other legacy directories

Older shared modelling, preprocessing-report and web application implementations.

New development should normally target `mepram/` or `bacthecom/` rather than `legacy/`.

---

## General guideline

```text
mepram/       → current MEPRAM production/development code
bacthecom/    → BACTHECOM-specific implementation
env/          → software environment definitions
legacy/       → previous implementations kept for reference
playground/   → exploratory analyses and notebooks
tests/        → automated tests
```
