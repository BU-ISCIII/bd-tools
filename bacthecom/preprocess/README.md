# BACTHECOM preprocessing

Prepare `RAW/merged_cohort.csv` with pandas 2+, numpy and PyYAML:

```bash
python DOC/bd-tools/bacthecom/preprocess/preprocess_pipeline.py \
  --input-path ANALYSIS/01-PREPROCESS/00-data/merged_cohort.csv \
  --output-path ANALYSIS/01-PREPROCESS/merged_cohort_preprocessed.csv \
  --config-path DOC/bd-tools/bacthecom/preprocess/config/preprocess_bacthecom.yml \
  --run-label merged_cohort
```

The project interpreter is `/home/ppascual/micromamba/envs/bacthecom_env/bin/python`.
The pipeline dispatches CSV input to `preprocess_merged_csv.py`. SQLite processing
remains supported by the existing entry point; the feature contract below applies
to merged CSV input.

## Step 0: merge the raw hospital workbooks

Use the original XLSX exports for this workflow. They are the inputs used by
`playground/notebooks/cohort_preprocessing.ipynb`, and their reviewed culture
classifications are explicit companion inputs. The normalized SQLite database
has a different relational representation; equivalence of its ingestion,
selection rules and mappings to this notebook has not been established. The
merge script intentionally reads XLSX, not SQLite.

```bash
# Setup only: symlinks and both launchers; no data processing.
bash ANALYSIS/01-PREPROCESS/lablog

# Activate bacthecom_env, or set PYTHON_BIN=/path/to/python.
# Optional read-only validation: no CSV or audit files are written.
bash ANALYSIS/01-PREPROCESS/_00_merge_cohort.sh --check-inputs

# Build 00-data/merged_cohort.csv and its merge audit inside 01-PREPROCESS.
bash ANALYSIS/01-PREPROCESS/_00_merge_cohort.sh

# Run feature engineering and reporting in the next dated version folder.
bash ANALYSIS/01-PREPROCESS/_01_preprocess.sh
```

`lablog` links the four source XLSX files and four reviewed XLSX files into
`ANALYSIS/01-PREPROCESS/00-data`, retaining the review subdirectory structure.
The original files remain in `RAW`; an existing local database is left untouched.
Both stages default to `00-data/merged_cohort.csv`. Its merge audit is saved
alongside it. Existing outputs require explicit `--overwrite` to replace.
To use a different output/input path for both stages:

```bash
export MERGED_COHORT_PATH=/path/to/new/merged_cohort.csv
bash ANALYSIS/01-PREPROCESS/_00_merge_cohort.sh
bash ANALYSIS/01-PREPROCESS/_01_preprocess.sh
```

`merging_cohort.py` requires pandas, PyYAML and openpyxl. Its CLI accepts
`--raw-dir`, `--config-path`, `--output-path`, `--check-inputs` and `--overwrite`.
The merge launcher also accepts `RAW_DIR` (default: local `00-data`),
`PYTHON_BIN` and `MERGED_COHORT_PATH`. Setup reads the configured filenames
using Python/PyYAML; activate the environment or set `PYTHON_BIN` before running
`lablog`. `RAW_SOURCE_DIR` overrides the original `RAW` location used to create
symlinks; ordinary files at link destinations are never replaced.
`INPUT_PATH` still overrides the preprocessing stage's input when explicitly set.
The setup does not automatically merge data on every preprocessing run.

Edit `config/merging_cohort.yml` (linked as
`ANALYSIS/01-PREPROCESS/merging_cohort.yml`) to select workbook versions. It lists
exactly VH, RyC, RS and VR, their episode/treatment sheets, and four reviewed
`hemocultivos_clasificados_*.xlsx` files. HUSC is not included because it was not
one of the notebook's four merged cohorts. Reviewed classifications are required;
missing files or decisions fail instead of silently changing the cohort.

The merge implementation follows the notebook's same-day aggregation and
30-day gap rule: within the original admission, gaps of 1–30 days from the
immediately preceding retained, reviewed culture are follow-ups; gaps >30 days
create a new sparse episode with its culture date as episode admission date.
This is an adjacent-culture gap rule, not a window measured from the first
culture. `apply_gap_rule_after_review: false` can retain every reviewed culture
day when the review alone should define episode eligibility.

Several notebook weaknesses are corrected explicitly:

- Review decisions match culture identity. Where `_id_fila` disambiguates
  duplicate isolates, the source-row ID must also match patient, admission,
  culture date, culture ID where available, and organism. Reordering a review
  file is safe; mismatched source versions fail.
- Treatment joins use patient, admission, culture date and culture ID, with
  validated join cardinality. Organism/discharge discrepancies are audited as
  descriptive mismatches, not used to discard an identifiable treatment.
- RyC long treatments are pivoted per culture, never attached to every episode
  of the patient. Each slot retains one drug, ATC, route, start/end and duration;
  dates/numbers are not combined into unparseable semicolon strings.
- `fecha_antib_previo_*` becomes `fecha_inicio_antib_previo_*`, preserving VH
  antibiotic dates. Complete duplicate courses across isolates are collapsed.
- The first TOS/tos occurrence is `trasplante_organo_solido`, the second is
  cough `tos`, per the documented header order in all four sheets. Other
  duplicate/alias columns coalesce only when values do not conflict.
- Episode IDs include the hospital, so reused patient IDs cannot collide.
- VR defines the base schema. Extra antibiotic slots from other hospitals are
  retained by default, rather than truncated to VR's observed slot count.
  Missing and excluded columns are listed in the audit.

VH count columns use the notebook's explicit `_x1000` replacements. Sex/mental
status and booleans are normalized; unknown values remain missing. Configured
numeric fields support decimal commas and the notebook's `<LOD / sqrt(2)` rule.
Clinical bounds, engineered predictors and mortality targets remain the
responsibility of downstream preprocessing. No cohort-fitted IQR trimming or
imputation is performed. Existing raw derived mortality/duration fields are not
new targets; downstream preprocessing recomputes mortality labels from dates.

This is a reviewed adaptation of the notebook, not a promise of byte-identical
output: corrected treatment linkage, preserved transplant/cough fields, missing
value handling and hospital-qualified IDs can change the result. Same-day
clinical conflicts use the first nonmissing value and are counted by column,
as in the notebook; choose `clinical_conflict_policy: error` for strict review.
Conflicting numbered history slots fail rather than combine incompatible records.

The merge audit includes SHA-256 hashes of all eight input files, script and
configuration hashes, effective configuration, hospital row counts, reviews,
join diagnostics, duplicate handling, conflicts, invalid dates/numbers and
schema alignment. It accompanies the merged CSV; later preprocessing writes
its separate audit and schema.

```bash
python -m unittest discover -s DOC/bd-tools/bacthecom/preprocess -p 'test_*.py' -v
```

## Output contract

Each output has exactly one row per `episode_id`, in input order. Missing or
duplicate IDs fail validation. If the input lacks this column, an unambiguous ID
is generated from hospital, record, admission date and culture date. Duplicate
composite episode keys also fail; conflicting episode rows are never silently
merged or discarded.

| Output suffix | Contents |
| --- | --- |
| `.csv` | Full audit dataset, including source fields and results; **do not train on this file** |
| `_filtered.csv` | Identifiers first (`hospital`, `fecha_ingreso`, `record_id`, `fecha_hemocultivo`), then numeric predictors and three mortality targets; no raw repeated slots |
| `_identifiers.csv` | `episode_id`, `record_id`, hospital, episode keys, row number and `patient_group` |
| `_outcomes.csv` | Episode ID, mortality targets and current microbiology/antibiogram results for target selection |
| `_schema.json` | Exact predictor, target, outcome and metadata lists; mappings and effective configuration |
| `_audit.json` | Validation, excluded columns, missingness, unknown mappings, row indices and episode IDs |
| `_binary_prevalence.csv` | Positive rows, observed rows and prevalence for every generated binary predictor |
| `_log_summary.csv`, `_log_detailed.json` | Inputs to the existing report generator |

Merged inputs may supply `hospital` or the older `hospital_cohort` name.
Preprocessing uses `hospital` as the canonical identifier in all new outputs.

Use `schema['predictor_columns']` for X, never every column in `_filtered.csv`.
The first four columns are retained for case tracing and listed in
`schema['identifier_columns']`; exclude them when training. Record IDs retain
leading zeros in the export; read them as strings when loading the CSV.
Choose one mortality target there, or join `_outcomes.csv` by `episode_id` to
select/construct an etiology or resistance target. Raw outcome strings require
an explicit target encoding for the intended task. Never include other outcomes
as predictors. `record_id`, `episode_id` and hospital are grouping metadata.
Split by `patient_group` to keep a patient's episodes in the same fold; hospital
can be used for held-out evaluation.

## Previous infections

All numbered history slots are merged into organism, resistance and source
indicators. Exact organism names include `infec_prev_Escherichia_coli`,
`infec_prev_Klebsiella_pneumoniae`, `infec_prev_Pseudomonas_aeruginosa`,
`infec_prev_Staphylococcus_aureus`, `infec_prev_Enterococcus_faecium`,
`infec_prev_Acinetobacter_baumannii`, `infec_prev_Enterobacter` and `infec_prev_Other`. Mixed organisms may activate
multiple columns. `res_prev_BLEE`, `res_prev_AMPC`, `res_prev_CARB`,
`res_prev_MRSA`, `res_prev_MDR`, `res_prev_Other` and `res_prev_any` summarize
resistance. MDR requires an explicit source label; it is not inferred by counting
resistant drugs. Missing organism and phenotype inputs remain NaN in their
encoded features; no coverage indicators are created.

`foco_prev_*` covers urinary, respiratory, bloodstream, abdominal/biliary,
catheter, skin/soft tissue and other sources. Source patterns are configurable.
`num_inf_previas` counts occupied eligible history slots, including source-only,
resistance-only or date-only slots; it cannot deduplicate undocumented clinical
events. `infec_previa_si_no` indicates at least one such slot.

`dias_desde_ultima_infec` uses the latest valid date strictly before culture.
Known same-day/future historical records are excluded. Undated records explicitly
labelled previous are retained for presence/count summaries and counted in the
audit; they cannot contribute to recency.

## Previous antibiotics

Drug names and ATC-only records from all slots, including slot 27, map first to
the detailed family mappings and BACTHECOM extensions in the `domain` section of
`config/preprocess_bacthecom.yml`, then through `antibiotic_family_groups`.
The original MEPRAM mappings are embedded in this YAML, with no runtime import
from MEPRAM. Each run snapshots the mappings together with its other settings.
Existing project family assignments are preserved. Names, aliases and exact ATC
codes are normalized; semicolon lists are supported. Ambiguously aligned ATC and
name lists are not paired by guesswork.

Every run emits these 12 binary columns, even if no exposure occurs:

- `antib_prev_Penicilinas`
- `antib_prev_Cefalosporinas`
- `antib_prev_Carbapenemas`
- `antib_prev_Aminoglucosidos`
- `antib_prev_Quinolonas`
- `antib_prev_Glicopeptidos_lipopeptidos`
- `antib_prev_Macrolidos_lincosamidas`
- `antib_prev_Tetraciclinas`
- `antib_prev_Antifungicos`
- `antib_prev_Antituberculosos`
- `antib_prev_Otros_antibacterianos`
- `antib_prev_Otros_no_clasificados`

One qualifying exposure sets its family to 1; otherwise it is 0. This means
recorded evidence, not confirmed absence. Unknown drugs/families use
`Otros_no_clasificados` and are audited. Medications outside this taxonomy may
also appear there; consult the exported drug-to-family map.

`antib_previo_si_no`, `antib_previo_num_exposiciones` and
`antib_previo_num_familias` summarize exposure. Repeated courses count separately;
a semicolon-separated drug list counts each entry. A record with dates/duration
but no drug identity counts as one unclassified exposure. Route-only fields do
not establish an identifiable exposure and raw route slots are removed.

`antib_previo_total_dias`, `antib_previo_max_dias`, and
`antib_previo_mean_dias` aggregate available nonnegative durations. Missing
history and histories with no usable durations stay missing.
Partial duration coverage is reported in the audit; totals use available values.
Courses ending on/after culture do not contribute their eventual total duration.

`dias_desde_ultimo_antib` uses each course's prior end date, falling back to a
valid prior start, then selects the latest date. Ends preceding starts are not
used. Same-day/future starts are excluded from history. Undated previous records
are retained and counted in the audit; recency stays missing
if no valid prior date exists. All raw antibiotic dates, names, ATC, durations
and routes are absent from the modelling table.

## Clinical summaries and leakage controls

`carga_dispositivos` sums the six configured binary device indicators. Missing
or invalid indicators leave the total missing rather than understating burden.
Continuous vital signs remain available alongside the following configurable
research indicators:

| Feature | Default rule |
| --- | --- |
| `taquicardia` | Heart rate >100 |
| `taquipnea` | Respiratory rate >=22 |
| `hipotension` | Systolic pressure <=100 |
| `hipoxemia` | Oxygen saturation <90% |
| `hipotermia_hipertermia` | Temperature <36 or >38 |

These cutoffs are editable in `merged_csv.clinical_features`, not diagnoses.
Missing vital signs produce missing binary indicators. Fractional saturation is
converted to percent first. Invalid age/saturation and configured negative lab
values become missing. Decimal commas and Spanish/English binary values are
normalized. Unconfigured text predictors fail validation. Child-Pugh is encoded
as A/B/C; configured categorical variables use one-hot encoding. Missing or
unresolved categories produce NaN across their encoded columns, with no
`__missing` or `__unknown` column. Entirely missing categorical features are
retained as an all-NaN column.

With `merged_csv.microbiology.include_current_results: true`, current hemoculture
results are assumed available at prediction. The filtered table includes mapped
organism indicators (`infec_Ecoli`, `infec_Kpneumoniae`, `infec_Paeruginosa`,
`infec_Saureus`, `infec_Enterococcus_faecium`, `infec_Acinetobacter_baumannii`,
`infec_Enterobacter`, and `infec_Other`), resistance indicators
(`resistant_BLEE`, `resistant_AMPC`, `resistant_CARB`, `resistant_MRSA`,
`multiresistant`, `resistant_Other`, `resistant_any`) and result-coverage summaries.
Raw organism/phenotype strings remain in audit/outcomes; numeric mapped columns
are the predictors. Mappings are in `merged_csv.microbiology.organism_patterns`
and `phenotype_patterns`. Set `include_current_results: false` when predicting
those microbiology outcomes themselves or predicting before their availability.
Other-culture results remain audit-only in either mode.

Historical source groups use `merged_csv.previous_sources` and emit
`foco_prev_*` from `especimen_previo_*`; these are specimen-derived focus proxies.
Edit the source YAML in `DOC/bd-tools/bacthecom/preprocess/config` (also linked from
`ANALYSIS/01-PREPROCESS`), not a past run's configuration snapshot. Changes apply
to the next lablog run and do not rewrite existing datasets.

Mortality, discharge information, ICU outcomes/duration, focus-control outcomes
and raw dates remain excluded from predictors. Baseline clinical fields are
assumed available because collection timestamps are not provided. Fit imputation,
scaling and feature selection inside training folds; continuous missing values
are retained.

Mortality targets retain the existing any-death and inclusive day 0–14/0–30
windows from admission (`mortalidad_any`, `mortalidad_14_dias`, and
`mortalidad_30_dias`). A documented death date overrides a negative
flag; unknown death status stays missing. Invalid death dates or death before
culture fail validation.

## Validation and reporting

The audit reports generated feature and removed source-column counts, prevalence
of every generated binary predictor, unique episode counts, and rows with unknown
antibiotic mappings or unclassified infection/antibiotic history. Row indices are
zero-based; matching episode IDs are exported alongside them.

```bash
python -m unittest discover -s DOC/bd-tools/bacthecom/preprocess -p test_feature_engineering.py -v
```

Prepare the symlinks and combined launcher, then run it when ready:

```bash
bash ANALYSIS/01-PREPROCESS/lablog       # Setup only; no processing.
# Activate bacthecom_env first, or set PYTHON_BIN.
bash ANALYSIS/01-PREPROCESS/_01_preprocess.sh
```

`lablog` creates relative symlinks to the configured XLSX inputs in `00-data`,
and to preprocessing/reporting scripts and their
configs in `DOC/bd-tools/bacthecom/preprocess`, and generates executable
`_00_merge_cohort.sh` and `_01_preprocess.sh`. Existing ordinary files at symlink locations are protected.
The `_00_merge_cohort.sh` launcher builds the merged CSV first.
The `_01_preprocess.sh` launcher runs preprocessing followed by reporting; there is no
separate `_02_report.sh`.

Each launch creates `YYYYMMDD_bacthecom_vN`, using today's date and the next
version across all existing run dates. For example, after v5 the next folder is
`20260922_bacthecom_v6` on 22 September 2026. Pass `bacthecom_vN` explicitly to
choose a version. Existing directories are never overwritten. Configuration
snapshots, execution logs, datasets, validation files and the report are stored
inside the run folder. `COMPLETE` marks success; `FAILED` records a failed run.

Environment overrides: `PYTHON_BIN`, `RUN_DATE=YYYYMMDD`, and `INPUT_PATH`.
Both scripts support `--help`. Check syntax without processing data using
`bash -n ANALYSIS/01-PREPROCESS/lablog ANALYSIS/01-PREPROCESS/_01_preprocess.sh`.

`preprocess_report.py` and its imported `build_summary_excel_file.py` remain in
use. The unused `compare_preprocessed_csvs.py` cross-project comparison helper
was removed. Predictor exclusions now live in YAML; the deleted external drop
list is no longer required.

## Code layout

| File | Role |
| --- | --- |
| `merging_cohort.py` | Raw XLSX merge, reviewed episodes and merge audit |
| `test_merging_cohort.py` | Synthetic workbook, join and integration checks |
| `preprocess_pipeline.py` | CLI, configuration, logging, CSV dispatch and supported SQLite processing |
| `preprocess_merged_csv.py` | Merged-cohort feature engineering, validation and exports |
| `preprocess_report.py` | Report tables, plots and Markdown |
| `build_summary_excel_file.py` | Excel export and profiling helpers imported by the report |
| `test_feature_engineering.py` | Synthetic regression tests; not executed during cohort processing |
| `config/preprocess_bacthecom.yml` | Pipeline settings, detailed antibiotic family mappings and domain constants |

All remaining Python modules have active callers or provide regression coverage.
Unused legacy antibiotic/antibiogram transformations and helpers were removed
from the pipeline; current CSV transformations and supported SQLite processing
remain separate. Keeping report and Excel helpers separate avoids coupling
feature engineering to plotting/export dependencies.

## Clinical report plots

Edit `feature_categories` in `config/preprocess_bacthecom_report.yml` to map
features using ordered wildcard patterns (first match wins). Categories cover
demographics, comorbidities, functional status, symptoms/signs, focus/severity,
laboratory, healthcare exposure, devices, previous infections, previous
antibiotics, current microbiology, other cultures, outcomes and identifiers.
Unknown names retain their stage category when available, otherwise Unmapped.

Reports now export horizontal category counts, 100% stacked completeness bands,
the most-missing predictors, and small missingness panels per clinical category.
Clinical plots are saved as PNG and editable SVG. `clinical_plots` controls how
many features are labelled. The full mapping is in `tables/feature_categories.csv`;
complete predictor missingness is in `tables/predictor_missingness.csv`. The same
categories populate the Excel summary. Counts compare audit and filtered tables;
missingness plots use schema predictors only, with targets shown separately.

Updates apply to future lablog runs. Existing run configuration snapshots remain
unchanged; to regenerate an older report with the new categories, pass the
current source report YAML explicitly and choose a new report output directory.

## Organism vocabulary, history windows and selected absence defaults (v0.8)

The fixed organism vocabulary is E. coli plus the six ESKAPE groups:
Enterococcus faecium, Staphylococcus aureus, Klebsiella pneumoniae,
Acinetobacter baumannii, Pseudomonas aeruginosa and Enterobacter spp.
See [ESKAPE definition](https://pubmed.ncbi.nlm.nih.gov/23458769/).
All remaining positive organism labels map to Other, including E. faecalis,
Candida and non-ESKAPE Enterobacterales. Missing/negative organism fields do not
activate Other. Each positive isolate activates a group; mixed cultures activate
multiple groups. This is multi-hot encoding, with every configured group emitted
even when absent from a run. Groups are not learned from cohort frequencies.
Edit `merged_csv.microbiology.organism_patterns` to change the vocabulary.

`history_windows_days: [30, 90]` produces `infec_previa_30d`, `infec_previa_90d`,
`antib_previo_30d` and `antib_previo_90d`. Windows are inclusive days 1–30/1–90
before culture. A dated recent record gives 1. No recorded eligible history
leaves the window missing. A history whose known dates are outside the window gives 0 only if there are
no undated records; otherwise the window is unknown. Continuous recency remains
available; antibiotic duration summaries are separate exposure measures.

No missingness, unknown-date, coverage or absent-history indicators are
generated. Existing `dias_desde_*` fields stay missing when no valid date
exists; zero is never used as a fictitious date. The audit reports absent-history
and undated-history counts separately. Counts of recorded events remain counts.

Hospital admission is distinct from infection or antibiotic exposure. There is
no mean previous-admission variable. `hospit_previa_30d` uses the reported
previous-month flag (an approximation to 30 days), or zero when the year flag is
negative and the month flag missing. `hospit_previa_90d` is 1 when the month flag
is positive and 0 when the year flag is negative; otherwise it is unknown.
Month-positive/year-negative conflicts produce unknown windows and are audited.
Exact 90-day admission indicators require previous-admission dates or a source
90-day question; they cannot be reconstructed from a positive yearly flag.

Preprocessing preserves per-feature missingness, including `hepatopatia_ligera`,
`hepatopatia_moderada_o_grave` and `paciente_residencia`. No missing-to-zero
imputation is applied. CSV blanks represent NaN. Missingness summaries remain
in audit/report files; imputation belongs in training folds.
