# Preprocessing Report

This report summarizes the preprocessing audit logs. It is intended for clinician review and pipeline QA.

## Dataset Shapes

- Full dataset: `3,913 rows x 447 columns`
- Filtered/model-compatible dataset: `3,913 rows x 215 columns`
- Final pipeline stage before filtering: `3,913` rows and `447` columns
- Filtered stage removed `232` columns and kept `215` columns

## Figures

![Rows before and after preprocessing stage](rows_by_stage.png)

![Columns before and after preprocessing stage](columns_by_stage.png)

![Preprocessing variables by stage](logged_operations_by_stage.png)

![Missingness in filtered variables](filtered_missingness.png)

![Grouped missingness for slides](filtered_missingness_grouped_for_slides.png)

![Variable distribution by group](variable_distribution_filtered_vs_unfiltered.png)

## Most Feature-Creating Stages

- `tbl_tratamiento_antibiotico_previo`: `50` created variables
- `cross_table_features`: `47` created variables
- `tbl_infecciones_previas`: `34` created variables
- `tbl_hemocultivo_de_urgencias`: `34` created variables
- `tbl_colonizaciones_previas`: `33` created variables
- `tbl_otros_cultivos_en_urgencias`: `33` created variables
- `tbl_sintomas`: `32` created variables
- `tbl_sepsis`: `13` created variables

## Predictive Targets

| Model | Target column | Samples | Classes | Majority class | Majority class % |
|---|---|---:|---:|---|---:|
| Model 1 - Sepsis | `sepsis` | 3,913 | 2 | Sepsis (-) | 50.8% |
| Model 2 - Etiology | `resultado_hemo_grouped` | 3,913 | 3 | Negative blood culture | 58.2% |
| Model 3 - Resistance | `resistente_cefalosporina` | 1,636 | 2 | Not resistant | 85.8% |

| Model | Class | Samples | % within target |
|---|---|---:|---:|
| Model 1 - Sepsis | Sepsis (-) | 1,989 | 50.8% |
| Model 1 - Sepsis | Sepsis (+) | 1,924 | 49.2% |
| Model 2 - Etiology | Gram-negative bacillus | 1,320 | 33.7% |
| Model 2 - Etiology | Gram-positive coccus | 316 | 8.1% |
| Model 2 - Etiology | Negative blood culture | 2,277 | 58.2% |
| Model 3 - Resistance | Not resistant | 1,404 | 85.8% |
| Model 3 - Resistance | Resistant to cephalosporins 3a/4a | 232 | 14.2% |

| Model | Exclusions or grouping |
|---|---|
| Model 1 - Sepsis | No exclusions; binary target over the full cohort. |
| Model 2 - Etiology | Grouped hemoculture target. NEGATIVE includes negative blood cultures and non-target or unmapped detected organisms. |
| Model 3 - Resistance | Restricted to grouped positive blood cultures (resultado_hemo_grouped != NEGATIVE); grouped as binary cephalosporin 3a/4a resistance. |

## Target Evolution

| Domain | Target version | Target column | Samples | Classes | Majority class | Majority class % | Top classes |
|---|---|---|---:|---:|---|---:|---|
| Etiology | Microorganism target | `resultado_hemo_mo` | 3,913 | 110 | NEGATIVE | 52.8% | NEGATIVE: 2,067; Escherichia coli: 836; Klebsiella pneumoniae: 201; Staphylococcus aureus: 168 |
| Etiology | Clinical organism grouping | `resultado_hemo` | 3,913 | 10 | NEGATIVE | 53.0% | NEGATIVE: 2,072; Escherichia coli: 850; Klebsiella pneumoniae: 210; _Other bacteria: 198 |
| Etiology | Gram grouped | `resultado_hemo_grouped` | 3,913 | 3 | NEGATIVE | 58.2% | NEGATIVE: 2,277; Bacilo gram-: 1,320; Coco gram+: 316 |
| Resistance | Individual phenotype combinations | `fenotipo_resistencia_individual` | 3,913 | 75 | NEGATIVE | 80.6% | NEGATIVE: 3,152; Bacilo Gram negativo resistente a amoxicilina/clavulánico: 153; Coco Gram positivo resistente a ampicilina o penicilina: 98; Bacilo Gram negativo resistente a ciprofloxacino: 67 |
| Resistance | Antibiotic family combinations | `fenotipo_resistencia` | 3,913 | 24 | NEGATIVE | 80.6% | NEGATIVE: 3,152; Penicilinas: 377; Cefalosporinas 3 gen + Cefalosporinas 4 gen + Penicilinas + Quinolonas: 88; Penicilinas + Quinolonas: 75 |
| Resistance | Cephalosporin yes/no | `resistente_cefalosporina` | 3,913 | 2 | NEGATIVE | 94.0% | NEGATIVE: 3,678; RESIST_CEFALOSPORINAS_3a_4a: 235 |

| Domain | Target version | Notes |
|---|---|---|
| Etiology | Microorganism target | Hemoculture microorganism target after clinician coinfection resolution. |
| Etiology | Clinical organism grouping | Configured clinical organism grouping from the preprocessing pipeline. |
| Etiology | Gram grouped | Pipeline target grouped into negative, Gram-negative bacillus, and Gram-positive coccus. |
| Resistance | Individual phenotype combinations | Individual resistance phenotype labels before antibiotic-family grouping. |
| Resistance | Antibiotic family combinations | Pipeline target after mapping raw resistance phenotypes to antibiotic families. |
| Resistance | Cephalosporin yes/no | Final binary target: any cephalosporin 3a/4a resistance versus negative. |

## Warnings

- `tbl_hemocultivo_de_urgencias`: 43 person_id/id_hemocultivo groups have conflicting bmr_etiologia values; aggregation uses max.

## Missingness Snapshot

Top missing columns in the filtered dataset:

- `cirugia_previa_con_implant`: `100.0%` missing
- `causa_inmunosupresion`: `75.5%` missing
- `dias_ultimo_antib`: `64.6%` missing
- `frec_respiratoria`: `42.2%` missing
- `frec_respiratoria_recoded`: `42.2%` missing
- `sofa`: `19.6%` missing
- `proteina_c_reactiva_recoded`: `17.7%` missing
- `proteina_c_reactiva`: `17.7%` missing
- `bilirrubina`: `13.1%` missing
- `snc_glasgow`: `12.7%` missing
- `situacion_funcional_basal`: `10.2%` missing
- `saturacion_o2_recoded`: `5.3%` missing
- `saturacion_o2`: `5.3%` missing
- `respiracion`: `5.2%` missing
- `vasopresores`: `5.0%` missing

## Table Summary

- `_pipeline_run`: rows `0` -> `0`, columns `0` -> `0`, dropped `0`, warnings `0`
- `tbl_paciente`: rows `3,913` -> `3,913`, columns `9` -> `13`, dropped `0`, warnings `0`
- `tbl_comorbilidad`: rows `3,913` -> `3,913`, columns `24` -> `109`, dropped `2`, warnings `0`
- `tbl_factores_riesgo_bmr`: rows `3,913` -> `3,913`, columns `16` -> `16`, dropped `0`, warnings `0`
- `tbl_sintomas`: rows `8,287` -> `3,892`, columns `4` -> `34`, dropped `2`, warnings `0`
- `tbl_signos`: rows `3,913` -> `3,913`, columns `12` -> `17`, dropped `0`, warnings `0`
- `tbl_sepsis`: rows `3,913` -> `3,913`, columns `17` -> `30`, dropped `0`, warnings `0`
- `tbl_infecciones_previas`: rows `3,581` -> `3,913`, columns `7` -> `36`, dropped `5`, warnings `0`
- `tbl_tratamiento_antibiotico_previo`: rows `5,915` -> `3,913`, columns `6` -> `52`, dropped `4`, warnings `0`
- `tbl_hemocultivo_de_urgencias`: rows `5,044` -> `3,913`, columns `8` -> `37`, dropped `4`, warnings `1`
- `tbl_colonizaciones_previas`: rows `837` -> `3,913`, columns `6` -> `35`, dropped `4`, warnings `0`
- `tbl_otros_cultivos_en_urgencias`: rows `5,104` -> `3,913`, columns `8` -> `35`, dropped `6`, warnings `0`
- `merged_dataset`: rows `3,913` -> `3,913`, columns `13` -> `394`, dropped `0`, warnings `0`
- `cross_table_features`: rows `3,913` -> `3,913`, columns `394` -> `441`, dropped `0`, warnings `0`
- `target_building`: rows `3,913` -> `3,913`, columns `441` -> `447`, dropped `0`, warnings `0`
- `filtered_dataset`: rows `3,913` -> `3,913`, columns `447` -> `215`, dropped `232`, warnings `0`
