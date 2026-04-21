# Preprocessing Assumptions for Clinical Review

This note records current preprocessing assumptions in the rewritten pipeline so they can be reviewed with clinicians before they become fixed behavior.

## Current Assumptions

### Symptoms (`tbl_sintomas`)

- Rows missing from `tbl_sintomas` after the left merge are currently interpreted as **unknown / not recorded**, not as symptom absence.
- As a consequence, symptom-derived columns remain `NaN` for patient/admission rows that do not have a matching symptom record.
- We are **not** filling these symptom columns with `0` by default.

Reason:
- The original notebook merges `tbl_sintomas_complete` with a left join and does not globally fill symptom-derived columns with `0` after that merge.
- A missing row in the symptom table may mean:
  - no symptoms were present, or
  - symptom information was not captured
- Until clinicians confirm the intended interpretation, keeping these values as `NaN` is the safer choice.

Questions for clinicians:
- If a patient/admission has no row in `tbl_sintomas`, should that mean:
  - symptom absent (`0`), or
  - symptom unknown / not recorded (`NaN`)?
- Is this interpretation the same for all symptom variables, or only for some of them?

### Missing values in modelling

- Missing values are **not always removed immediately** before modelling.
- In most current modelling scripts, the default behavior is:
  - drop rows missing the target column or weight column
  - then impute feature missingness by default
- If imputation is disabled with `--no-impute`, several scripts then drop rows with missing feature values.
- Some older scripts / variants still use stricter `dropna()` behavior.

Practical implication:
- Keeping a variable as `NaN` in preprocessing does **not** automatically mean the row will be removed in downstream models.
- It usually means the feature will be imputed later, unless the specific training script disables imputation or uses a stricter `dropna()` path.

### Prior antibiotics (`tbl_tratamiento_antibiotico_previo`)

- Rows with `dias_trat_antimicrobiano <= 0` or missing are currently interpreted as **no previous antibiotic exposure recorded for that row**.
- The pipeline creates `antib_previo_si_no = 1` only when `dias_trat_antimicrobiano > 0`.
- Rows with `antib_previo_si_no == 0` are then removed before building prior-antibiotic family features.

Observed in the current test run:

- `2739` rows were removed because `antib_previo_si_no == 0` (`dias_trat_antimicrobiano <= 0` or missing).

Reason:

- The original notebook applies the same effective filter:
  `tbl_antib_prev = tbl_antib_prev[tbl_antib_prev["antib_previo_si_no"] == 1]`
- This means only rows with a positive treatment duration contribute to `ultimo_antib`, `dias_ultimo_antib`, antibiotic-family binaries, and `antib_previo_total_veces`.

Question for clinicians:

- Does missing or zero `dias_trat_antimicrobiano` reliably mean no prior antibiotic exposure?
- Or can it mean prior antibiotic exposure was present but duration was not recorded?
- If duration is missing but `antimicrobiano_previo` or `fecha_administracion_antib` is present, should that row still contribute to prior-antibiotic features?

### Previous Infection Detail (`tbl_infecciones_previas`)

The current rewrite preserves previous infection history mainly as:

- grouped previous microorganism binary columns
- `num_inf_previas`
- `tiempo_ultima`

But the original raw table also contains:

- `bmr_infec_previa`
- `feno_resist_infec_prev`
- `sindrome_infeccioso`

These are currently not preserved in the aggregated output.

#### Clinical concern

If we aggregate previous infections only at the patient/admission level, we can lose the association between:

- the previous microorganism
- whether that previous infection was BMR
- which previous resistance phenotype was observed
- which infectious syndrome was associated with that microorganism

Example:
- previous *Escherichia coli* infection was BMR
- previous *Enterococcus* infection was not BMR

If we keep only a global flag such as `bmr_infec_previa_any = 1`, then we lose which microorganism was actually associated with the resistant infection.

#### Question for clinicians

Should previous infection detail be kept as:

1. Global patient/admission-level summaries only
2. Microorganism-specific summaries, for example one BMR history variable per microorganism group

Possible microorganism-specific representations:

- `prev_<organism>_bmr_any`
- `prev_<organism>_feno_<x>_binary`
- `prev_<organism>_sindrome_<x>_binary`

This would preserve the link between microorganism and BMR / phenotype / syndrome, at the cost of more columns and more sparsity.

#### Current grouped-microorganism counts in `tbl_infecciones_previas`

Using the current grouped microorganism mapping in the preprocessing pipeline, the previous-infections table contains the following grouped counts:

- `Escherichia coli`: `1000`
- `Klebsiella pneumoniae`: `680`
- `_Enterobacteria`: `474`
- `_Other bacteria`: `365`
- `_Virus`: `287`
- `Pseudomonas aeruginosa`: `246`
- `Enterococcus`: `212`
- `Staphylococcus aureus`: `117`
- `_Fungi`: `111`
- `Streptococcus pneumoniae`: `31`
- unmapped / not grouped: `58`

So the current grouped microorganism space is effectively around `10` categories, not the full raw microorganism code space.

#### Current grouped-microorganism × BMR counts

- `Escherichia coli`: total `1000`, `bmr_1=739`, `bmr_0=261`
- `Klebsiella pneumoniae`: total `680`, `bmr_1=550`, `bmr_0=130`
- `_Enterobacteria`: total `474`, `bmr_1=364`, `bmr_0=110`
- `_Other bacteria`: total `365`, `bmr_1=128`, `bmr_0=237`
- `_Virus`: total `287`, `bmr_1=1`, `bmr_0=286`
- `Pseudomonas aeruginosa`: total `246`, `bmr_1=138`, `bmr_0=108`
- `Enterococcus`: total `212`, `bmr_1=64`, `bmr_0=148`
- `Staphylococcus aureus`: total `117`, `bmr_1=78`, `bmr_0=39`
- `_Fungi`: total `111`, `bmr_1=1`, `bmr_0=110`
- `Streptococcus pneumoniae`: total `31`, `bmr_1=4`, `bmr_0=27`
- unmapped / not grouped: total `58`, `bmr_1=0`, `bmr_0=58` -> infections without known pathogen involved.

Practical interpretation:

- keeping grouped microorganism-specific BMR features looks feasible
- this would be on the order of `10` meaningful grouped features, not hundreds

Question to review:

- should these `58` rows be investigated and reassigned to one of the grouped categories?
- or should they be kept explicitly as an `unknown / unmapped microorganism history` category?

### Previous Colonisation Detail (`tbl_colonizaciones_previas`)

The current rewrite preserves previous colonisation history mainly as:

- grouped colonising microorganism binary columns prefixed with `colo_`
- `colonizacion_total_grouped`

But the original raw table also contains:

- `bmr_colonizador`
- `feno_resist_colo`
- `fecha_colonizacion`

These are currently not preserved in the aggregated output, matching the original notebook.

#### Clinical concern

As with previous infections, aggregating colonisations only as grouped organism binaries can lose the association between:

- the colonising microorganism
- whether that colonisation was BMR
- which resistance phenotype was observed
- when the colonisation occurred

Example:

- previous *Escherichia coli* colonisation had a resistance phenotype
- previous *Klebsiella pneumoniae* colonisation had a different resistance phenotype

If we only keep `colo_Escherichia_coli_binary = 1` and `colo_Klebsiella_pneumoniae_binary = 1`, then the organism-phenotype link is lost.

#### Current grouped-microorganism counts in `tbl_colonizaciones_previas`

Using the current grouped microorganism mapping in the preprocessing pipeline, the colonisation table contains:

- `Escherichia coli`: `251`
- `Klebsiella pneumoniae`: `188`
- `_Enterobacteria`: `116`
- `_Other bacteria`: `102`
- `Pseudomonas aeruginosa`: `81`
- `Staphylococcus aureus`: `41`
- `_Fungi`: `37`
- `Enterococcus`: `15`
- `_Virus`: `5`
- `Streptococcus pneumoniae`: `1`

So the current grouped microorganism space is `10` categories.

#### Current grouped-microorganism × BMR counts

In this table, `bmr_colonizador` is only populated as `1` or missing; there are no explicit `0` values in the current data extract.

- `Escherichia coli`: total `251`, `bmr_1=232`, `bmr_missing=19`
- `Klebsiella pneumoniae`: total `188`, `bmr_1=186`, `bmr_missing=2`
- `_Enterobacteria`: total `116`, `bmr_1=104`, `bmr_missing=12`
- `_Other bacteria`: total `102`, `bmr_1=76`, `bmr_missing=26`
- `Pseudomonas aeruginosa`: total `81`, `bmr_1=66`, `bmr_missing=15`
- `Staphylococcus aureus`: total `41`, `bmr_1=34`, `bmr_missing=7`
- `_Fungi`: total `37`, `bmr_1=0`, `bmr_missing=37`
- `Enterococcus`: total `15`, `bmr_1=5`, `bmr_missing=10`
- `_Virus`: total `5`, `bmr_1=0`, `bmr_missing=5`
- `Streptococcus pneumoniae`: total `1`, `bmr_1=0`, `bmr_missing=1`

Practical implication:

- Adding one BMR feature per grouped organism would add around `10` columns.
- However, clinicians should confirm whether missing `bmr_colonizador` means non-BMR or unknown.

#### Current resistance phenotype counts

`feno_resist_colo` has `11` non-null phenotype codes and `138` missing rows.

- code `6`: `123` rows, `Bacilo Gram negativo resistente a ceftrixona o cefotaxima`
- code `4`: `120` rows, `Bacilo Gram negativo resistente a amoxicilina/clavulánico`
- code `8`: `108` rows, `Bacilo Gram negativo resistente a ceftazidima`
- code `9`: `88` rows, `Bacilo Gram negativo resistente a ciprofloxacino`
- code `5`: `75` rows, `Bacilo Gram negativo resistente a piperacilina/tazobactam`
- code `7`: `74` rows, `Bacilo Gram negativo resistente a cefepima`
- code `10`: `36` rows, `Bacilo Gram negativo resistente a meropenem`
- code `1`: `32` rows, `Coco Gram positivo resistente a ampicilina o penicilina`
- code `2`: `22` rows, `Coco Gram positivo resistente a meticilina`
- code `11`: `12` rows, `Bacilo Gram negativo resistente a ceftolozano/tazobactam`
- code `12`: `9` rows, `Bacilo Gram negativo resistente a ceftazidima/avibactam`

#### Potential column increase if phenotype detail is preserved

- Global phenotype binaries would add up to `11` columns.
- Organism-specific BMR features would add up to `10` columns.
- Organism-specific phenotype features would add `49` observed organism-phenotype pair columns in the current extract.
- A full grouped organism × phenotype matrix would be up to `10 × 11 = 110` columns, but many combinations are not observed.

Observed organism-phenotype pair counts by group:

- `Enterococcus`: `1` phenotype code
- `Escherichia coli`: `10` phenotype codes
- `Klebsiella pneumoniae`: `8` phenotype codes
- `Pseudomonas aeruginosa`: `7` phenotype codes
- `Staphylococcus aureus`: `2` phenotype codes
- `_Enterobacteria`: `10` phenotype codes
- `_Other bacteria`: `11` phenotype codes

Questions for clinicians:

- Should `bmr_colonizador` be preserved as global summary features or organism-specific features?
- Does missing `bmr_colonizador` mean non-BMR or unknown?
- Should `feno_resist_colo` be preserved globally, organism-specifically, or not used?
- Is `fecha_colonizacion` clinically useful for a “time since last colonisation” feature?
