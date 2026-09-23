"""Regression checks for missingness, culture timing and pooled feature encoding."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml

from preprocess_merged_csv import (
    antibiotic_features, load_domain_config, infection_history_features, clinical_features,
    evidence_flag, microbiology_features, preprocess_csv, mortality_targets,
    recency_features, admission_history_features,
)

CONFIG_PATH = Path(__file__).parent / 'config/preprocess_bacthecom.yml'


class FeatureEngineeringTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load(CONFIG_PATH.read_text())
        self.settings = self.config['merged_csv']

    def test_merge_hospital_column_is_accepted_as_cohort_identifier(self):
        frame = pd.DataFrame({'hospital': ['vh', 'ryc'], 'record_id': ['001', '001'],
            'fecha_ingreso': ['2026-01-01'] * 2, 'fecha_hemocultivo': ['2026-01-10'] * 2,
            'mortalidad': ['0'] * 2, 'fecha_mortalidad': [''] * 2})
        with TemporaryDirectory() as directory:
            source, output = Path(directory) / 'input.csv', Path(directory) / 'output.csv'
            frame.to_csv(source, index=False)
            _, _, filtered, audit = preprocess_csv(source, output, self.config, CONFIG_PATH)
            self.assertEqual(filtered.columns[:4].tolist(),
                ['hospital', 'fecha_ingreso', 'record_id', 'fecha_hemocultivo'])
            self.assertEqual(filtered.hospital.tolist(), ['vh', 'ryc'])
            self.assertEqual(filtered.record_id.tolist(), ['001', '001'])
            self.assertEqual(audit['cohort_counts'], {'vh': 1, 'ryc': 1})
            ids = pd.read_csv(output.with_name('output_identifiers.csv'))
            self.assertEqual(ids.patient_group.tolist(), ['vh:001', 'ryc:001'])
            self.assertEqual(ids.episode_id.nunique(), 2)
            frame.rename(columns={'hospital': 'hospital_cohort'}).to_csv(source, index=False)
            _, _, legacy, _ = preprocess_csv(source, output, self.config, CONFIG_PATH)
            pd.testing.assert_frame_equal(legacy, filtered)
            frame.drop(columns='hospital').to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, 'Missing required CSV columns.*hospital'):
                preprocess_csv(source, output, self.config, CONFIG_PATH)

    def test_domain_config_uses_run_settings(self):
        self.config['domain']['child_pugh_levels'] = ['a', 'unknown']
        self.assertEqual(load_domain_config(self.config)['child_pugh_levels'], ['a', 'unknown'])
        self.assertEqual(load_domain_config({})['child_pugh_levels'], ['a', 'b', 'c'])

    def test_eskape_and_other_are_multihot_in_all_contexts(self):
        organisms = ['E. coli', 'Klebsiella pneumoniae', 'Pseudomonas aeruginosa',
            'Staphylococcus aureus', 'Enterococcus faecium', 'Acinetobacter baumannii',
            'Enterobacter cloacae', 'Proteus mirabilis',
            'E. coli; Acinetobacter baumannii; Candida albicans', None, 'Negativo']
        frame = pd.DataFrame({'fecha_hemocultivo': ['2026-01-10'] * len(organisms),
            'microorganismo': organisms, 'microorganism_episodio_previo_1': organisms,
            'fecha_episodio_previo_1': ['2026-01-01'] * len(organisms)})
        full, _, _ = microbiology_features(frame, self.settings)
        labels = ['Ecoli', 'Kpneumoniae', 'Paeruginosa', 'Saureus',
                  'Enterococcus_faecium', 'Acinetobacter_baumannii', 'Enterobacter', 'Other']
        for context in ['infec', 'infec_prev']:
            columns = [f'{context}_{label}' for label in labels]
            for index, label in enumerate(labels):
                self.assertEqual(full.loc[index, f'{context}_{label}'], 1)
                self.assertEqual(full.loc[index, columns].sum(), 1)
            self.assertEqual(full.loc[8, columns].sum(), 3)
            self.assertTrue(full.loc[9, columns].isna().all())
            self.assertTrue(full.loc[10, columns].eq(0).all())

    def test_recency_states_and_window_boundaries(self):
        days = pd.Series([None, None, 30, 31, 90, 91, 100, 20], dtype=float)
        history = pd.Series([False, True, True, True, True, True, True, True])
        uncertain = pd.Series([False, True, False, False, False, False, True, True])
        result = recency_features(days, history, uncertain, 'history', [30, 90])
        self.assertEqual(result.columns.tolist(), ['history_30d', 'history_90d'])
        self.assertTrue(pd.isna(result.history_30d.iloc[0]))
        self.assertTrue(pd.isna(result.history_30d.iloc[1]))
        self.assertEqual(result.history_30d.iloc[2:6].tolist(), [1, 0, 0, 0])
        self.assertEqual(result.history_90d.iloc[2:6].tolist(), [1, 1, 1, 0])
        self.assertTrue(pd.isna(result.history_90d.iloc[6]))
        self.assertEqual(result.history_30d.iloc[7], 1)
        self.assertTrue(pd.isna(days.iloc[0]))  # no fabricated zero-day recency

    def test_admission_windows_do_not_invent_90_day_dates(self):
        frame = pd.DataFrame({'hospit_mes_previo': [1, 0, 0, None, 1],
                              'hospit_ano_previo': [1, 0, 1, 0, 0]})
        result, audit = admission_history_features(frame, self.settings)
        self.assertEqual(result.hospit_previa_30d.iloc[:4].tolist(), [1, 0, 0, 0])
        self.assertEqual(result.hospit_previa_90d.iloc[:2].tolist(), [1, 0])
        self.assertTrue(pd.isna(result.hospit_previa_90d.iloc[2]))
        self.assertEqual(result.hospit_previa_90d.iloc[3], 0)
        self.assertTrue(pd.isna(result.hospit_previa_30d.iloc[4]))
        self.assertEqual(audit['admission_history_flag_conflicts'], 1)

    def test_raw_missingness_is_preserved_without_indicator_columns(self):
        frame = pd.DataFrame({'hospital': ['test'] * 2, 'record_id': ['1', '2'],
            'fecha_ingreso': ['2026-01-01'] * 2, 'fecha_hemocultivo': ['2026-01-10'] * 2,
            'mortalidad': ['0', '0'], 'fecha_mortalidad': ['', ''],
            'hepatopatia_ligera': ['', '1'], 'hepatopatia_moderada_o_grave': ['', '0'],
            'paciente_residencia': ['', 'True'], 'creatinina': ['', '2'],
            'foco': ['', 'urinario'], 'puntaje_child_pugh': ['', 'A']})
        with TemporaryDirectory() as directory:
            source, output = Path(directory) / 'input.csv', Path(directory) / 'output.csv'
            frame.to_csv(source, index=False)
            _, _, filtered, audit = preprocess_csv(source, output, self.config, CONFIG_PATH)
            for c in ['hepatopatia_ligera', 'hepatopatia_moderada_o_grave', 'paciente_residencia', 'foco__urinario', 'puntaje_child_pugh__a']:
                self.assertTrue(pd.isna(filtered.loc[0, c]), c)
            self.assertEqual(audit['missing_values_assumed_absent'], {})
            self.assertEqual(filtered.loc[1, 'foco__urinario'], 1)
            self.assertEqual(filtered.loc[1, 'puntaje_child_pugh__a'], 1)
            self.assertFalse(any(token in c for c in filtered for token in ['__missing', '_unknown', '_desconocida', '_recorded', '_timing_unverified', '_sin_historia']))
            exported = pd.read_csv(output.with_name('output_filtered.csv'))
            self.assertTrue(pd.isna(exported.loc[0, 'foco__urinario']))
            self.assertEqual(filtered.loc[1, 'hepatopatia_ligera'], 1)
            self.assertEqual(filtered.loc[1, 'paciente_residencia'], 1)
            self.assertTrue(pd.isna(filtered.loc[0, 'creatinina']))
            for prefix in ['infec_previa', 'antib_previo']:
                self.assertTrue(pd.isna(filtered.loc[0, prefix + '_30d']))
            self.assertTrue(pd.isna(filtered.loc[0, 'dias_desde_ultimo_antib']))

    def test_unknown_and_entirely_missing_categories_stay_missing(self):
        frame = pd.DataFrame({'hospital': ['test'] * 3, 'record_id': ['1', '2', '3'],
            'fecha_ingreso': ['2026-01-01'] * 3, 'fecha_hemocultivo': ['2026-01-10'] * 3,
            'mortalidad': ['0'] * 3, 'fecha_mortalidad': [''] * 3,
            'foco': ['', 'unknown', 'desconocido'], 'puntaje_child_pugh': ['', 'A/B', 'C']})
        with TemporaryDirectory() as directory:
            source, output = Path(directory) / 'input.csv', Path(directory) / 'output.csv'
            frame.to_csv(source, index=False)
            _, _, filtered, _ = preprocess_csv(source, output, self.config, CONFIG_PATH)
            self.assertTrue(filtered.foco.isna().all())
            child = ['puntaje_child_pugh__a', 'puntaje_child_pugh__b', 'puntaje_child_pugh__c']
            self.assertTrue(filtered.loc[:1, child].isna().all().all())
            self.assertEqual(filtered.loc[2, child].tolist(), [0., 0., 1.])
            self.assertFalse(any(c.endswith(('__missing', '__unknown')) for c in filtered))

    def test_only_admission_mortality_targets_are_generated(self):
        from preprocess_pipeline import create_mortality_targets, export_dataset_outputs
        from preprocess_report import DEFAULT_REPORT_CONFIG
        expected = ['mortalidad_any', 'mortalidad_14_dias', 'mortalidad_30_dias']
        frame = pd.DataFrame({
            'mortalidad': [1] * 4,
            'fecha_ingreso': ['2026-01-01'] * 4,
            'fecha_hemocultivo': ['2026-01-10'] * 4,
            'fecha_mortalidad': ['2026-01-15', '2026-01-16', '2026-01-31', '2026-02-01'],
        })
        for window in (14, 30):
            frame[f'mortalidad_{window}_dias_' + 'desde_hemocultivo'] = 1
        frame['mortalidad_dias_' + 'desde_hemocultivo'] = 10
        targets = mortality_targets(frame, self.config)
        self.assertEqual(list(targets), expected)
        self.assertEqual(targets.mortalidad_14_dias.tolist(), [1, 0, 0, 0])
        self.assertEqual(targets.mortalidad_30_dias.tolist(), [1, 1, 1, 0])
        sqlite = create_mortality_targets(frame, self.config)
        self.assertFalse(any(c.endswith('desde_hemocultivo') for c in sqlite))
        pd.testing.assert_frame_equal(sqlite[expected].astype(float), targets.astype(float))
        # Export also handles legacy fields and missing flags introduced by joins.
        legacy = sqlite.copy()
        legacy['mortalidad_dias_' + 'desde_hemocultivo'] = 10
        legacy['hepatopatia_ligera'] = [np.nan, np.nan, 1., 0.]
        with TemporaryDirectory() as directory:
            filtered, _, _ = export_dataset_outputs(legacy, Path(directory) / 'output.csv', self.config)
            self.assertFalse(any(c.endswith('desde_hemocultivo') for c in filtered))
            self.assertTrue(filtered.hepatopatia_ligera.iloc[:2].isna().all())
            self.assertEqual(filtered.hepatopatia_ligera.iloc[2:].tolist(), [1., 0.])
        report = yaml.safe_load((CONFIG_PATH.parent / 'preprocess_bacthecom_report.yml').read_text())
        for settings in (report, DEFAULT_REPORT_CONFIG):
            for key in ('predictive_targets', 'prediction_detail_sections'):
                self.assertEqual([entry['target_column'] for entry in settings[key]], expected)
            self.assertNotIn('desde_hemocultivo', str(settings))

    def test_mortality_accepts_mixed_numeric_and_boolean_values(self):
        flags = [False, True, ' FALSE ', 'tRuE', 0, 1, '0.0', '1.0', None, '']
        frame = pd.DataFrame({'mortalidad': flags, 'fecha_mortalidad': [None] * len(flags),
            'fecha_ingreso': ['2026-01-01'] * len(flags),
            'fecha_hemocultivo': ['2026-01-10'] * len(flags)})
        targets = mortality_targets(frame, self.config)
        self.assertEqual(targets.mortalidad_any.iloc[:8].tolist(), [0, 1, 0, 1, 0, 1, 0, 1])
        self.assertTrue(targets.mortalidad_any.iloc[8:].isna().all())
        self.assertEqual(targets.mortalidad_14_dias.iloc[0], 0)
        self.assertTrue(pd.isna(targets.mortalidad_14_dias.iloc[1]))
        frame.loc[0, 'fecha_mortalidad'] = '2026-01-11'
        self.assertEqual(mortality_targets(frame, self.config).mortalidad_any.iloc[0], 1)
        for invalid in ['tryue', 2, -1]:
            frame.loc[0, 'mortalidad'] = invalid
            with self.assertRaisesRegex(ValueError, 'unexpected values'):
                mortality_targets(frame, self.config)

    def test_boolean_csv_mortality_and_conflict_audit(self):
        frame = pd.DataFrame({'hospital': ['test'] * 3, 'record_id': ['1', '2', '3'],
            'fecha_ingreso': ['2026-01-01'] * 3, 'fecha_hemocultivo': ['2026-01-10'] * 3,
            'mortalidad': ['False', 'True', ''], 'fecha_mortalidad': ['2026-01-11', '', '']})
        with TemporaryDirectory() as directory:
            source, output = Path(directory) / 'input.csv', Path(directory) / 'output.csv'
            frame.to_csv(source, index=False)
            _, _, filtered, audit = preprocess_csv(source, output, self.config, CONFIG_PATH)
            self.assertEqual(filtered.mortalidad_any.iloc[:2].tolist(), [1, 1])
            self.assertTrue(pd.isna(filtered.mortalidad_any.iloc[2]))
            self.assertEqual(audit['mortality_flag_date_conflicts'], 1)

    def test_missing_is_not_negative(self):
        result = evidence_flag(pd.Series([None, '', 'desconocido', 'No', 'Negativo', 'E. coli']))
        self.assertTrue(result.iloc[:3].isna().all())
        self.assertEqual(result.iloc[3:].tolist(), [0., 0., 1.])

    def test_history_uses_each_slots_date_before_aggregation(self):
        frame = pd.DataFrame({
            'fecha_hemocultivo': ['2026-01-10'] * 4,
            'microorganism_episodio_previo_1': ['E. coli'] * 4,
            'fecha_episodio_previo_1': ['2026-01-09', '2026-01-10', '2026-01-11', None],
            'feno_resist_infec_prev_1': ['BLEE'] * 4,
            'microorganismo_otros_cult_1': ['E. coli'] * 4,
            'fecha_ otros_cultivos_1': ['2026-01-09', '2026-01-10', '2026-01-11', None],
        })
        full, _, _ = microbiology_features(frame, self.settings)
        for column in ['infec_prev_Ecoli', 'resistant_prev_BLEE']:
            self.assertEqual(full[column].iloc[0], 1)
            self.assertTrue(full[column].iloc[1:3].isna().all())
            self.assertEqual(full[column].iloc[3], 1)
        self.assertEqual(full.otros_Ecoli.iloc[:2].tolist(), [1., 1.])
        self.assertTrue(full.otros_Ecoli.iloc[2:].isna().all())

    def test_multilabel_phenotypes_and_susceptible_only_are_not_negative(self):
        frame = pd.DataFrame({
            'fecha_hemocultivo': ['2026-01-10'] * 4,
            'fenotipo_resistencia': ['BLEE y Carbapenemasa', None, 'nuevo fenotipo', 'sin resistencia'],
            'resultado_antibiograma_hemocultivo_s': [None, 'Ampicilina', None, None],
        })
        full, predictors, audit = microbiology_features(frame, self.settings)
        self.assertEqual(full.loc[0, 'resistant_BLEE'], 1)
        self.assertEqual(full.loc[0, 'resistant_CARB'], 1)
        self.assertTrue(pd.isna(full.loc[1, 'resistant_any']))
        self.assertNotIn('antibiogram_recorded', full)
        self.assertTrue(pd.isna(full.loc[1, 'resistant_BLEE']))
        self.assertNotIn('phenotype_recorded', full)
        self.assertEqual(full.loc[2, 'resistant_Other'], 1)
        self.assertEqual(full.loc[3, 'resistant_any'], 0)
        self.assertEqual(audit['unmapped_phenotype_values'], ['nuevo fenotipo'])
        self.assertIn('resistant_any', predictors)
        self.assertNotIn('otros_Ecoli', predictors)
        conservative = dict(self.settings, microbiology=dict(self.settings['microbiology'], include_current_results=False))
        _, predictors, _ = microbiology_features(frame, conservative)
        self.assertNotIn('resistant_any', predictors)
        self.assertIn('resistant_prev_any', predictors)

    def test_five_slots_merge_and_future_records_do_not_leak(self):
        frame = pd.DataFrame({'fecha_hemocultivo': ['2026-01-10', '2026-01-10']})
        for slot in range(1, 6):
            frame[f'fecha_episodio_previo_{slot}'] = ['2026-01-01', '2026-01-11']
            frame[f'microorganism_episodio_previo_{slot}'] = [None, 'E. coli']
        frame['microorganism_episodio_previo_5'] = ['ESCHERICHIA COLI; Staphylococcus aureus; unknown species', 'E. coli']
        frame['feno_resist_infec_prev_5'] = ['CARB, MDR', 'CARB']
        full, _, _ = microbiology_features(frame, self.settings)
        for column in ['infec_prev_Ecoli', 'infec_prev_Saureus', 'infec_prev_Other', 'resistant_prev_CARB', 'multiresistant_prev']:
            self.assertEqual(full[column].iloc[0], 1)
            self.assertTrue(pd.isna(full[column].iloc[1]))
        self.assertEqual(full.infec_prev_positive_slots.iloc[0], 1)
        self.assertTrue(pd.isna(full.infec_prev_positive_slots.iloc[1]))
        self.assertNotIn('infec_prev_timing_unverified', full)
        self.assertFalse(any(name.startswith('fenotipo_dummy') for name in full))

    def test_resistance_only_slot_and_all_missing_history(self):
        frame = pd.DataFrame({
            'fecha_hemocultivo': ['2026-01-10', '2026-01-10'],
            'fecha_episodio_previo_5': ['2026-01-01', None],
            'feno_resist_infec_prev_5': ['BLEE', None],
        })
        full, _, _ = microbiology_features(frame, self.settings)
        self.assertEqual(full.resistant_prev_BLEE.iloc[0], 1)
        self.assertTrue(pd.isna(full.resistant_prev_BLEE.iloc[1]))
        self.assertNotIn('phenotype_prev_recorded', full)
        self.assertNotIn('infec_prev_recorded', full)
        self.assertTrue(full.infec_prev_Ecoli.isna().all())

    def test_antibiotics_atc_only_unknown_future_duration_and_slot_27(self):
        frame = pd.DataFrame({
            'fecha_hemocultivo': ['2026-01-10'] * 5,
            'antimicrobiano_previo_1': ['amoxicilina', None, 'not_mapped', 'meropenem', None],
            'atc_antib_prev_1': [None, 'J01DH02', None, None, None],
            'fecha_inicio_antib_previo_1': ['2026-01-01', None, None, '2026-01-11', None],
            'fecha_fin_antib_previo_1': ['2026-01-09', None, None, '2026-01-12', None],
            'dias_trat_antimicrobiano_1': ['8', None, None, '1', None],
            'antimicrobiano_previo_27': ['ampicilina', None, None, None, None],
            'fecha_inicio_antib_previo_27': ['2026-01-02', None, None, None, None],
            'dias_trat_antimicrobiano_27': ['2', None, None, None, None],
        }, index=[4, 8, 9, 10, 11])
        result, audit = antibiotic_features(frame, self.settings, load_domain_config())
        groups = [c for c in result if c.startswith('antib_prev_')]
        self.assertEqual(len(groups), 12)
        self.assertFalse(result[groups].iloc[:3].isna().any().any())
        self.assertTrue(result[groups].iloc[3:].isna().all().all())
        self.assertEqual(result.antib_prev_Penicilinas.iloc[:3].tolist(), [1, 0, 0])
        self.assertEqual(result.antib_prev_Carbapenemas.iloc[:3].tolist(), [0, 1, 0])
        self.assertEqual(result.antib_prev_Otros_no_clasificados.iloc[:3].tolist(), [0, 0, 1])
        self.assertEqual(result.antib_previo_num_exposiciones.tolist(), [2, 1, 1, 0, 0])
        self.assertEqual(result.loc[4, 'antib_previo_num_familias'], 1)
        self.assertEqual(result.loc[4, 'antib_previo_total_dias'], 10)
        self.assertEqual(result.loc[4, 'dias_desde_ultimo_antib'], 1)
        self.assertTrue(pd.isna(result.loc[8, 'dias_desde_ultimo_antib']))
        self.assertEqual(audit['rows_with_unknown_antibiotic_family_mapping'], [9])
        # Post-index end/duration cannot reveal future course length.
        frame.loc[4, 'fecha_fin_antib_previo_1'] = '2026-02-01'
        result, _ = antibiotic_features(frame, self.settings, load_domain_config())
        self.assertEqual(result.loc[4, 'antib_previo_total_dias'], 2)
        self.assertEqual(result.loc[4, 'dias_desde_ultimo_antib'], 8)

    def test_source_only_and_resistance_only_history_count_recency(self):
        frame = pd.DataFrame({'fecha_hemocultivo': ['2026-01-10'] * 3,
            'especimen_previo_1': ['Orina', 'unknown specimen', None],
            'fecha_episodio_previo_1': ['2026-01-01', 'invalid', None],
            'feno_resist_infec_prev_5': ['BLEE', None, None],
            'fecha_episodio_previo_5': ['2026-01-09', None, None]})
        result, audit = infection_history_features(frame, self.settings)
        self.assertEqual(result.num_inf_previas.tolist(), [2, 1, 0])
        self.assertEqual(result.foco_prev_urinaria.iloc[:2].tolist(), [1, 0])
        self.assertEqual(result.foco_prev_otra.iloc[:2].tolist(), [0, 1])
        self.assertEqual(result.dias_desde_ultima_infec.iloc[0], 1)
        self.assertEqual(audit['rows_with_unclassified_previous_infection'], [1])

    def test_clinical_boundaries_missingness_and_devices(self):
        frame = pd.DataFrame({'frec_cardiaca': [100, 101, None],
            'frecuencia_respiratoria': [21, 22, None], 'tension_arterial_sist': [101, 100, None],
            'saturacion_po2': [90, 89, None], 'temperatura': [36, 39, None]})
        for col in self.settings['clinical_features']['device_columns']:
            frame[col] = [0, 1, None]
        result = clinical_features(frame, self.settings)
        for col in result.columns.drop('carga_dispositivos'):
            self.assertEqual(result[col].iloc[:2].tolist(), [0., 1.])
            self.assertTrue(pd.isna(result[col].iloc[2]))
        self.assertEqual(result.carga_dispositivos.iloc[:2].tolist(), [0., 6.])
        self.assertTrue(pd.isna(result.carga_dispositivos.iloc[2]))

    def test_csv_exports_numeric_engineered_features_without_drug_slots(self):
        frame = pd.DataFrame({
            'hospital': ['test'], 'record_id': ['001'],
            'fecha_ingreso': ['2026-01-01'], 'fecha_hemocultivo': ['2026-01-10'],
            'mortalidad': ['0'], 'fecha_mortalidad': [''],
            'microorganismo': ['E. coli'], 'fenotipo_resistencia': ['BLEE'],
            'barthel_inf_90': ['1'],
            'via_administ_antib_prev_27': ['IV'], 'dias_trat_antimicrobiano_27': ['5'],
            'duracion_uci': ['10'], 'control_foco': ['si'],
            'microorganism_episodio_previo_1': ['E. coli'],
            'fecha_episodio_previo_1': ['2025-12-01'], 'feno_resist_infec_prev_1': ['BLEE'],
        })
        for window in (14, 30):
            frame[f'mortalidad_{window}_dias_' + 'desde_hemocultivo'] = 1
        frame['mortalidad_dias_' + 'desde_hemocultivo'] = 10
        with TemporaryDirectory() as directory:
            source, output = Path(directory) / 'source.csv', Path(directory) / 'output.csv'
            frame.to_csv(source, index=False)
            _, full, filtered, audit = preprocess_csv(source, output, self.config, CONFIG_PATH)
            self.assertEqual(filtered.loc[0, 'res_prev_any'], 1)
            self.assertEqual(filtered.loc[0, 'resistant_BLEE'], 1)
            self.assertEqual(filtered.loc[0, 'infec_Ecoli'], 1)
            self.assertNotIn('otros_Ecoli', filtered)
            self.assertIn('foco_prev_urinaria', filtered)
            self.assertFalse(any('fuente' in c for c in filtered))
            self.assertEqual(filtered.loc[0, 'mortalidad_any'], 0)
            self.assertNotIn('microorganismo', filtered)
            self.assertEqual(filtered.loc[0, 'barthel_inf_90'], 1)
            for column in ['via_administ_antib_prev_27', 'dias_trat_antimicrobiano_27', 'duracion_uci', 'control_foco']:
                self.assertNotIn(column, filtered)
            identifier_columns = ['hospital', 'fecha_ingreso', 'record_id', 'fecha_hemocultivo']
            self.assertEqual(filtered.columns[:4].tolist(), identifier_columns)
            pd.testing.assert_frame_equal(filtered[identifier_columns], frame[identifier_columns])
            exported_filtered = pd.read_csv(output.with_name('output_filtered.csv'), dtype=str)
            self.assertEqual(exported_filtered.columns[:4].tolist(), identifier_columns)
            pd.testing.assert_frame_equal(exported_filtered[identifier_columns], frame[identifier_columns])
            self.assertTrue(all(pd.api.types.is_numeric_dtype(dtype) for dtype in filtered.iloc[:, 4:].dtypes))
            self.assertTrue(output.with_name('output_schema.json').exists())
            self.assertEqual(len(full), 1)
            self.assertFalse(any(c.endswith('desde_hemocultivo') for c in full))
            for exported in Path(directory).glob('output*.csv'):
                self.assertFalse(any(c.endswith('desde_hemocultivo') for c in pd.read_csv(exported, nrows=0)))
            import json
            schema = json.loads(output.with_name('output_schema.json').read_text())
            self.assertEqual(schema['identifier_columns'], identifier_columns)
            self.assertEqual(filtered.columns.tolist(), identifier_columns + schema['predictor_columns'] + schema['target_columns'])
            self.assertEqual(schema['target_columns'], ['mortalidad_any', 'mortalidad_14_dias', 'mortalidad_30_dias'])
            ids = pd.read_csv(output.with_name('output_identifiers.csv'))
            self.assertEqual(ids.episode_id.nunique(), 1)
            self.assertIn('episode_id', schema['metadata_columns'])
            self.assertIn('fenotipo_resistencia', schema['outcome_columns'])
            self.assertFalse(set(schema['predictor_columns']) & set(schema['outcome_columns']))
            self.assertFalse(set(schema['predictor_columns']) & set(schema['metadata_columns']))
            self.assertGreater(audit['generated_feature_count'], 40)
            self.assertIn('antib_prev_Penicilinas', audit['generated_binary_prevalence'])
            # Turning off current-result availability excludes all current mapped features.
            self.config['merged_csv']['microbiology']['include_current_results'] = False
            _, _, second, _ = preprocess_csv(source, output, self.config, CONFIG_PATH)
            self.assertNotIn('resistant_BLEE', second)
            self.assertNotIn('infec_Ecoli', second)
            self.assertIn('res_prev_BLEE', second)
            frame['episode_id'] = 'same'
            duplicate = pd.concat([frame, frame.assign(record_id='2')], ignore_index=True)
            duplicate.to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, 'episode_id'):
                preprocess_csv(source, output, self.config, CONFIG_PATH)


if __name__ == '__main__':
    unittest.main()
