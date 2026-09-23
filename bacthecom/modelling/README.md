# Configured mortality training

Run `bash ANALYSIS/04-TRAINING/lablog` from the project root (or run the lablog
by absolute path from any directory). Activate the modelling environment first,
or set `PYTHON_BIN` to its Python interpreter. Setup does not train or submit jobs.

The lablog selects the highest-version completed preprocessing run and creates
`ANALYSIS/04-TRAINING/YYYYMMDD_ANALYSIS_NN`. To pin input data or custom settings:

```bash
bash ANALYSIS/04-TRAINING/lablog \
  --preprocess-dir ANALYSIS/01-PREPROCESS/20260922_bacthecom_v5 \
  --training-config DOC/bd-tools/bacthecom/modelling/config/training_config.json
```

`--hpc-config` supplies an optional resource/executor configuration. Defaults live
in `config/`; each analysis receives editable copies and links to the filtered
CSV, schema and aligned patient identifiers. Existing analyses remain untouched.

The default run matrix uses CatBoost with seed 42:

| Run | Mode | Target |
|---|---|---|
| target_30_catb_seed42 | TARGET | mortalidad_30_dias |
| target_14_catb_seed42 | TARGET | mortalidad_14_dias |
| staged_30_14_catb_seed42 | STAGED | target1: mortalidad_30_dias; target2: mortalidad_14_dias |

Remove unwanted runs or add uniquely named runs with `catb`, `lgbm` or `xgb`.
Targets retain preprocessing's admission-date reference. Stage 2 is fitted only
among observed 30-day deaths. The deployed 14-day probability is the product of
the stage-1 probability and stage-2 conditional probability.

Inside the new analysis:

```bash
# Regenerate after editing training_config.json or hpc_config.json:
bash lablog
bash _00_validate.sh
# Submit the configured training runs as a Slurm array:
bash _01_run_model.sh
# Equivalent submission alias:
bash _00_submit.sh
```

Slurm resources and optional Singularity execution are set in `hpc_config.json`.
Check the partition, interpreter/container and CPU/memory settings for your
cluster. Each Slurm task validates its selected run before fitting; changed
training configs require regenerating launchers before submission.

The training CLI uses schema predictors and patient-grouped holdout/CV splits,
fits missingness/outlier/imputation/correlation decisions within training folds,
and tunes by Optuna. Defaults enable 100 trials, five-fold CV, a grouped holdout,
SHAP explanations and separate full-data refits. Evaluation artifacts and metrics
remain separate from `full_model.joblib` deployment artifacts. These are produced
by training itself; no separate legacy plotting or report scripts are needed.

HPC defaults: shared Python `/data/ucct/bi/pipelines/micromamba/envs/bacthecom_env/bin/python`, partition `long_idx`, 4 CPUs and 16 GB per task, 48 hours, at most 3 simultaneous runs. Validation also runs on compute nodes. Job IDs are appended to `submitted_jobs.txt`; logs are under `logs/`. Both config hashes are checked before submission and at job start. No jobs are submitted by lablog.
