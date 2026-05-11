# Preprocessing Report

This report summarizes the preprocessing audit logs. It is intended for clinician review and pipeline QA.

## Dataset Shapes

- Full dataset: `3,913 rows x 417 columns`
- Filtered/model-compatible dataset: `3,913 rows x 180 columns`
- Final pipeline stage before filtering: `3,913` rows and `417` columns
- Filtered stage removed `237` columns and kept `180` columns

## Figures

![Rows before and after preprocessing stage](graphs/rows_by_stage.png)

![Columns before and after preprocessing stage](graphs/columns_by_stage.png)

![Preprocessing variables by stage](graphs/logged_operations_by_stage.png)

![Missingness in filtered variables](graphs/filtered_missingness.png)

![Grouped missingness for slides](graphs/filtered_missingness_grouped_for_slides.png)

![Variable distribution by group](graphs/variable_distribution_filtered_vs_unfiltered.png)

## Most Feature-Creating Stages

- `tbl_tratamiento_antibiotico_previo`: `48` created variables
- `cross_table_features`: `47` created variables
- `tbl_infecciones_previas`: `34` created variables
- `tbl_hemocultivo_de_urgencias`: `34` created variables
- `tbl_colonizaciones_previas`: `33` created variables
- `tbl_otros_cultivos_en_urgencias`: `33` created variables
- `tbl_sintomas`: `16` created variables
- `target_building`: `6` created variables

## Predictive Targets

| Model | Target column | Samples | Classes | Majority class | Majority class % |
|---|---|---:|---:|---|---:|
| Model 1 - Sepsis | `sepsis` | 3,913 | 2 | Sepsis (-) | 50.8% |
| Model 2 - Etiology | `resultado_hemo_grouped` | 3,913 | 3 | Negative blood culture | 59.2% |
| Model 3 - Resistance | `resistente_cefalosporina` | 1,597 | 2 | Not resistant | 85.5% |

| Model | Class | Samples | % within target |
|---|---|---:|---:|
| Model 1 - Sepsis | Sepsis (-) | 1,989 | 50.8% |
| Model 1 - Sepsis | Sepsis (+) | 1,924 | 49.2% |
| Model 2 - Etiology | Gram-negative bacillus | 1,320 | 33.7% |
| Model 2 - Etiology | Gram-positive coccus | 277 | 7.1% |
| Model 2 - Etiology | Negative blood culture | 2,316 | 59.2% |
| Model 3 - Resistance | Not resistant | 1,365 | 85.5% |
| Model 3 - Resistance | Resistant to cephalosporins 3a/4a | 232 | 14.5% |

| Model | Exclusions or grouping |
|---|---|
| Model 1 - Sepsis | No exclusions; binary target over the full cohort. |
| Model 2 - Etiology | Grouped hemoculture target. NEGATIVE includes negative blood cultures and non-target or unmapped detected organisms. |
| Model 3 - Resistance | Restricted to grouped positive blood cultures (resultado_hemo_grouped != NEGATIVE); grouped as binary cephalosporin 3a/4a resistance. |

## Target Evolution

| Domain | Target version | Target column | Samples | Classes | Majority class | Majority class % | Top classes |
|---|---|---|---:|---:|---|---:|---|
| Etiology | Microorganism target | `resultado_hemo_mo` | 3,913 | 101 | NEGATIVE | 53.0% | NEGATIVE: 2,072; Escherichia coli: 850; Klebsiella pneumoniae: 210; Staphylococcus aureus: 171 |
| Etiology | Clinical organism grouping | `resultado_hemo` | 3,913 | 10 | NEGATIVE | 53.0% | NEGATIVE: 2,072; Escherichia coli: 850; Klebsiella pneumoniae: 210; _Other bacteria: 198 |
| Etiology | Gram grouped | `resultado_hemo_grouped` | 3,913 | 3 | NEGATIVE | 59.2% | NEGATIVE: 2,316; Bacilo gram-: 1,320; Coco gram+: 277 |
| Resistance | Individual phenotype labels | `fenotipo_resistencia_individual` | 3,913 | 13 | NEGATIVE | 80.6% | NEGATIVE: 3,152; Bacilo Gram negativo resistente a amoxicilina/clavulánico: 438; Bacilo Gram negativo resistente a ciprofloxacino: 315; Bacilo Gram negativo resistente a ceftrixona o cefotaxima: 208 |
| Resistance | Antibiotic family labels | `fenotipo_resistencia` | 3,913 | 6 | NEGATIVE | 80.6% | NEGATIVE: 3,152; Penicilinas: 621; Quinolonas: 315; Cefalosporinas 3/4 gen: 235 |
| Resistance | Cephalosporin yes/no | `resistente_cefalosporina` | 3,913 | 2 | NEGATIVE | 94.0% | NEGATIVE: 3,678; RESIST_CEFALOSPORINAS_3a_4a: 235 |

| Domain | Target version | Notes |
|---|---|---|
| Etiology | Microorganism target | Hemoculture microorganism target after clinician coinfection resolution. |
| Etiology | Clinical organism grouping | Configured clinical organism grouping from the preprocessing pipeline. |
| Etiology | Gram grouped | Pipeline target grouped into negative, Gram-negative bacillus, and Gram-positive coccus. |
| Resistance | Individual phenotype labels | Individual resistance phenotype labels before antibiotic-family grouping; multilabel rows are counted once for each phenotype present. |
| Resistance | Antibiotic family labels | Pipeline target after mapping raw resistance phenotypes to antibiotic families; multilabel rows are counted once for each family present. |
| Resistance | Cephalosporin yes/no | Final binary target: any cephalosporin 3a/4a resistance versus negative. |

## Prediction Target Details

### Sepsis

Binary sepsis prediction target.

- Dataset shape: `3,913` rows x `180` columns
- Target column: `sepsis`
- Classes: `2`

| Class | Rows | % within target |
|---|---:|---:|
| Sepsis (-) | 1,989 | 50.8% |
| Sepsis (+) | 1,924 | 49.2% |
| Total | 3,913 | 100.0% |

### Etiology - microorganisms

Original microorganism target before clinical grouping; shown as top 50 classes plus Other.

- Dataset shape: `3,913` rows x `180` columns
- Target column: `resultado_hemo_mo`
- Classes: `101`

| Class | Rows | % within target |
|---|---:|---:|
| NEGATIVE | 2,072 | 53.0% |
| Escherichia coli | 850 | 21.7% |
| Klebsiella pneumoniae | 210 | 5.4% |
| Staphylococcus aureus | 171 | 4.4% |
| Streptococcus pneumoniae | 106 | 2.7% |
| Pseudomonas aeruginosa | 84 | 2.1% |
| Proteus mirabilis | 37 | 0.9% |
| Staphylococcus epidermidis | 32 | 0.8% |
| Staphylococcus hominis | 31 | 0.8% |
| Enterobacter cloacae | 29 | 0.7% |
| Klebsiella oxytoca | 27 | 0.7% |
| Enterococcus faecalis | 24 | 0.6% |
| _Enterobacteria | 17 | 0.4% |
| Enterococcus faecium | 13 | 0.3% |
| Salmonella enterica | 12 | 0.3% |
| Serratia marcescens | 10 | 0.3% |
| Klebsiella aerogenes | 10 | 0.3% |
| Streptococcus agalactiae | 8 | 0.2% |
| Haemophilus influenzae | 8 | 0.2% |
| Streptococcus gallolyticus | 7 | 0.2% |
| Citrobacter koseri | 7 | 0.2% |
| Streptococcus anginosus | 6 | 0.2% |
| _Other bacteria | 6 | 0.2% |
| Bacteroides fragilis | 6 | 0.2% |
| Citrobacter freundii | 5 | 0.1% |
| Staphylococcus haemolyticus | 5 | 0.1% |
| Streptococcus mitis | 5 | 0.1% |
| Listeria monocytogenes | 4 | 0.1% |
| Streptococcus parasanguinis | 4 | 0.1% |
| Staphylococcus capitis | 4 | 0.1% |
| género Salmonella | 4 | 0.1% |
| género Streptococcus | 4 | 0.1% |
| Morganella morganii | 4 | 0.1% |
| Streptococcus oralis | 4 | 0.1% |
| Providencia stuartii | 3 | 0.1% |
| Streptococcus dysgalactiae | 3 | 0.1% |
| Brevibacterium epidermidis | 3 | 0.1% |
| Micrococcus luteus | 2 | 0.1% |
| Bacteroides thetaiotaomicron | 2 | 0.1% |
| Candida glabrata | 2 | 0.1% |
| Streptococcus equi | 2 | 0.1% |
| Peptoniphilus harei | 2 | 0.1% |
| Streptococcus beta - hemolítico | 2 | 0.1% |
| _Fungi | 2 | 0.1% |
| género Elizabethkingia | 2 | 0.1% |
| Corynebacterium afermentans | 2 | 0.1% |
| género Brevibacillus | 2 | 0.1% |
| Fusobacterium nucleatum | 2 | 0.1% |
| Candida parapsilosis | 2 | 0.1% |
| Streptococcus sanguinis | 2 | 0.1% |
| Other | 52 | 1.3% |
| Total | 3,913 | 100.0% |

### Etiology - clinical grouping

Clinician-configured microorganism grouping used by the etiology model.

- Dataset shape: `3,913` rows x `180` columns
- Target column: `resultado_hemo`
- Classes: `10`

| Class | Rows | % within target |
|---|---:|---:|
| NEGATIVE | 2,072 | 53.0% |
| Escherichia coli | 850 | 21.7% |
| Klebsiella pneumoniae | 210 | 5.4% |
| _Other bacteria | 198 | 5.1% |
| _Enterobacteria | 176 | 4.5% |
| Staphylococcus aureus | 171 | 4.4% |
| Streptococcus pneumoniae | 106 | 2.7% |
| Pseudomonas aeruginosa | 84 | 2.1% |
| Enterococcus | 39 | 1.0% |
| _Fungi | 7 | 0.2% |
| Total | 3,913 | 100.0% |

### Resistance - individual phenotypes

Individual resistance phenotype labels before antibiotic-family grouping; multilabel rows are counted once for each phenotype present.

- Dataset shape: `3,913` rows x `180` columns
- Target column: `fenotipo_resistencia_individual`
- Classes: `13`

| Class | Rows | % within target |
|---|---:|---:|
| NEGATIVE | 3,152 | 80.6% |
| Bacilo Gram negativo resistente a amoxicilina/clavulánico | 438 | 11.2% |
| Bacilo Gram negativo resistente a ciprofloxacino | 315 | 8.1% |
| Bacilo Gram negativo resistente a ceftrixona o cefotaxima | 208 | 5.3% |
| Bacilo Gram negativo resistente a ceftazidima | 188 | 4.8% |
| Bacilo Gram negativo resistente a piperacilina/tazobactam | 173 | 4.4% |
| Bacilo Gram negativo resistente a cefepima | 169 | 4.3% |
| Coco Gram positivo resistente a ampicilina o penicilina | 168 | 4.3% |
| Coco Gram positivo resistente a meticilina | 58 | 1.5% |
| Bacilo Gram negativo resistente a meropenem | 11 | 0.3% |
| Bacilo Gram negativo resistente a ceftazidima/avibactam | 4 | 0.1% |
| Bacilo Gram negativo resistente a ceftolozano/tazobactam | 4 | 0.1% |
| Coco Gram positivo resistente a vancomicina | 2 | 0.1% |
| Total label assignments | 4,890 | 125.0% |
| Total patients | 3,913 | 100.0% |

### Resistance - antibiotic families

Resistance phenotypes grouped into antibiotic-family labels; multilabel rows are counted once for each family present.

- Dataset shape: `3,913` rows x `180` columns
- Target column: `fenotipo_resistencia`
- Classes: `6`

| Class | Rows | % within target |
|---|---:|---:|
| NEGATIVE | 3,152 | 80.6% |
| Penicilinas | 621 | 15.9% |
| Quinolonas | 315 | 8.1% |
| Cefalosporinas 3/4 gen | 235 | 6.0% |
| Carbapenemas | 11 | 0.3% |
| Glicopéptidos | 2 | 0.1% |
| Total label assignments | 4,336 | 110.8% |
| Total patients | 3,913 | 100.0% |

### Resistance - cephalosporins

Final binary cephalosporin resistance target.

- Dataset shape: `3,913` rows x `180` columns
- Target column: `resistente_cefalosporina`
- Classes: `2`

| Class | Rows | % within target |
|---|---:|---:|
| Not resistant | 3,678 | 94.0% |
| Resistant to cephalosporins 3a/4a | 235 | 6.0% |
| Total | 3,913 | 100.0% |


## Warnings

- `tbl_hemocultivo_de_urgencias`: 43 person_id/id_hemocultivo groups have conflicting bmr_etiologia values; aggregation uses max.

## Missingness Snapshot

Top missing columns in the filtered dataset:

- `cirugia_previa_con_implant`: `100.0%` missing
- `causa_inmunosupresion`: `75.5%` missing
- `dias_ultimo_antib`: `64.6%` missing
- `frec_respiratoria`: `42.2%` missing
- `sofa`: `19.6%` missing
- `proteina_c_reactiva`: `17.7%` missing
- `bilirrubina`: `13.1%` missing
- `snc_glasgow`: `12.7%` missing
- `situacion_funcional_basal`: `10.2%` missing
- `saturacion_o2`: `5.3%` missing
- `respiracion`: `5.2%` missing
- `creatinina`: `3.8%` missing
- `cardiovascular`: `3.7%` missing
- `plaquetas`: `2.8%` missing
- `frec_cardiaca`: `2.2%` missing

## Table Summary

- `_pipeline_run`: rows `0` -> `0`, columns `0` -> `0`, dropped `0`, warnings `0`
- `tbl_paciente`: rows `3,913` -> `3,913`, columns `9` -> `13`, dropped `0`, warnings `0`
- `tbl_comorbilidad`: rows `3,913` -> `3,913`, columns `24` -> `109`, dropped `2`, warnings `0`
- `tbl_factores_riesgo_bmr`: rows `3,913` -> `3,913`, columns `16` -> `16`, dropped `0`, warnings `0`
- `tbl_sintomas`: rows `8,287` -> `3,892`, columns `4` -> `18`, dropped `18`, warnings `0`
- `tbl_signos`: rows `3,913` -> `3,913`, columns `12` -> `17`, dropped `0`, warnings `0`
- `tbl_sepsis`: rows `3,913` -> `3,913`, columns `17` -> `18`, dropped `0`, warnings `0`
- `tbl_infecciones_previas`: rows `3,581` -> `3,913`, columns `7` -> `36`, dropped `5`, warnings `0`
- `tbl_tratamiento_antibiotico_previo`: rows `5,915` -> `3,913`, columns `6` -> `50`, dropped `4`, warnings `0`
- `tbl_hemocultivo_de_urgencias`: rows `5,044` -> `3,913`, columns `8` -> `37`, dropped `4`, warnings `1`
- `tbl_colonizaciones_previas`: rows `837` -> `3,913`, columns `6` -> `35`, dropped `4`, warnings `0`
- `tbl_otros_cultivos_en_urgencias`: rows `5,104` -> `3,913`, columns `8` -> `35`, dropped `6`, warnings `0`
- `merged_dataset`: rows `3,913` -> `3,913`, columns `13` -> `364`, dropped `0`, warnings `0`
- `cross_table_features`: rows `3,913` -> `3,913`, columns `364` -> `411`, dropped `0`, warnings `0`
- `target_building`: rows `3,913` -> `3,913`, columns `411` -> `417`, dropped `0`, warnings `0`
- `filtered_dataset`: rows `3,913` -> `3,913`, columns `417` -> `180`, dropped `237`, warnings `0`
