"""Synthetic workbook tests; never read/write the real merged cohort."""
import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from merging_cohort import (
    DEFAULT_CONFIG, EPISODE_KEYS, TREATMENT_FIELDS, attach_treatments,
    clinical_recoding, group_episodes, load_config, merge_cohorts,
    normalize_frame, read_sheet, select_reviewed,
)


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(DEFAULT_CONFIG)

    def normalize(self, frame):
        return normalize_frame(frame, self.config, {})

    def episode(self, **changes):
        row = dict(record_id='001', fecha_ingreso='2026-01-01', fecha_alta='2026-04-01',
                   fecha_hemocultivo='2026-01-02', id_hemocultivo='C1',
                   microorganismo='E. coli', fenotipo_resistencia='BLEE',
                   mortalidad=0, fecha_mortalidad=None, sexo='HOMBRE',
                   temperatura=37, somnolencia_estupor_coma='normal')
        row.update(changes)
        return row

    def test_review_join_is_identity_based_not_position(self):
        frame = self.normalize(pd.DataFrame([self.episode(), self.episode(record_id='002')]))
        review = frame.copy().assign(estado=['CONSERVAR', 'ELIMINAR']).iloc[::-1]
        result = select_reviewed(frame, review, self.config, {})
        self.assertEqual(result.record_id.tolist(), ['001'])
        with self.assertRaisesRegex(ValueError, 'lack matching'):
            select_reviewed(frame, review.iloc[:1], self.config, {})
        conflict = pd.concat([review, review.iloc[:1].assign(estado='CONSERVAR')])
        with self.assertRaisesRegex(ValueError, 'Conflicting review'):
            select_reviewed(frame, conflict, self.config, {})

    def test_review_row_id_requires_matching_identity(self):
        frame = self.normalize(pd.DataFrame([self.episode(), self.episode()]))
        review = frame.assign(_id_fila=[0, 1], estado=['CONSERVAR', 'ELIMINAR']).iloc[::-1]
        self.assertEqual(len(select_reviewed(frame, review, self.config, {})), 1)
        review.loc[0, 'record_id'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'lack matching'):
            select_reviewed(frame, review, self.config, {})

    def test_transplant_and_cough_are_preserved_separately(self):
        frame = self.normalize(pd.DataFrame([[1, 0]], columns=['TOS', 'tos']))
        self.assertEqual(frame.trasplante_organo_solido.iloc[0], 1)
        self.assertEqual(frame.tos.iloc[0], 0)

    def test_long_treatments_are_culture_specific_and_aligned(self):
        frame = self.normalize(pd.DataFrame([
            self.episode(), self.episode(id_hemocultivo='C2', fecha_hemocultivo='2026-03-01')]))
        rows = []
        for drug, start, days in [('amoxicilina', '2025-12-20', 3), ('meropenem', '2025-12-25', 5)]:
            rows.append({**self.episode(), **dict.fromkeys(TREATMENT_FIELDS),
                         'antimicrobiano_previo': drug, 'fecha_inicio_antib_previo': start,
                         'dias_trat_antimicrobiano': days})
        result = attach_treatments(frame, self.normalize(pd.DataFrame(rows)), 'long', {})
        self.assertEqual(len(result), 2)
        self.assertEqual(result.loc[0, 'antimicrobiano_previo_1'], 'amoxicilina')
        self.assertEqual(result.loc[0, 'dias_trat_antimicrobiano_2'], 5)
        self.assertEqual(result.loc[0, 'fecha_inicio_antib_previo_2'], pd.Timestamp('2025-12-25'))
        self.assertTrue(pd.isna(result.loc[1, 'antimicrobiano_previo_1']))
        bad = self.normalize(pd.DataFrame(rows)).assign(record_id='missing')
        with self.assertRaisesRegex(ValueError, 'unmatched parent'):
            attach_treatments(frame, bad, 'long', {})

    def test_wide_join_prevents_row_multiplication(self):
        frame = self.normalize(pd.DataFrame([self.episode()]))
        treatment = frame.assign(antimicrobiano_previo_1='amoxicilina')
        result = attach_treatments(frame, pd.concat([treatment, treatment]), 'wide', {})
        self.assertEqual(len(result), 1)
        diagnostic = {}
        corrected = attach_treatments(frame, treatment.assign(microorganismo='updated label'), 'wide', diagnostic)
        self.assertEqual(corrected.loc[0, 'antimicrobiano_previo_1'], 'amoxicilina')
        self.assertEqual(diagnostic['treatment_descriptor_mismatches']['microorganismo'], 1)
        conflict = pd.concat([treatment, treatment.assign(antimicrobiano_previo_1='meropenem')])
        with self.assertRaisesRegex(ValueError, 'Conflicting wide'):
            attach_treatments(frame, conflict, 'wide', {})

    def test_same_day_organisms_courses_gap_boundary_and_hospital_ids(self):
        frame = self.normalize(pd.DataFrame([
            self.episode(antimicrobiano_previo_1='amoxicilina'),
            self.episode(id_hemocultivo='C2', microorganismo='S. aureus', fenotipo_resistencia='MRSA',
                         antimicrobiano_previo_1='amoxicilina'),
            self.episode(id_hemocultivo='C3', fecha_hemocultivo='2026-02-01'),  # exactly 30 days
            self.episode(id_hemocultivo='C4', fecha_hemocultivo='2026-03-04'),  # 31 days from preceding culture
        ]))
        audit = {}
        result = group_episodes(frame, 'vh', self.config, audit)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.loc[0, 'microorganismo'], 'E. coli; S. aureus')
        self.assertEqual(result.loc[0, 'fenotipo_resistencia'], 'BLEE; MRSA')
        self.assertNotIn('antimicrobiano_previo_2', result)
        self.assertEqual(result.loc[1, 'fecha_ingreso'], pd.Timestamp('2026-03-04'))
        self.assertEqual(audit['follow_up_episodes_removed'], 1)
        other = group_episodes(frame, 'vr', self.config, {})
        self.assertTrue(set(result.episode_id).isdisjoint(other.episode_id))

    def test_dates_aliases_duplicates_and_clinical_missingness(self):
        source = pd.DataFrame({'record_id': ['001'], 'fecha_antib_previo_1': [46023],
                               'fecha_ingreso': ['02/01/2026'], 'fecha_alta': ['bad']})
        result = self.normalize(source)
        self.assertEqual(result.record_id.iloc[0], '001')
        self.assertEqual(result.fecha_ingreso.iloc[0], pd.Timestamp('2026-01-02'))
        self.assertEqual(result.fecha_inicio_antib_previo_1.iloc[0], pd.Timestamp('2026-01-01'))
        self.assertTrue(pd.isna(result.fecha_alta.iloc[0]))
        with self.assertRaisesRegex(ValueError, 'Conflicting duplicate'):
            self.normalize(pd.DataFrame([[1, 2]], columns=['Edad', 'edad']))
        complementary = self.normalize(pd.DataFrame([[1, None], [None, 2]], columns=['Edad', 'edad']))
        self.assertEqual(complementary.edad.tolist(), [1, 2])
        clinical = clinical_recoding(pd.DataFrame({'sexo': ['HOMBRE', None],
            'somnolencia_estupor_coma': ['normal', None], 'creatinina': ['<0,4', 'bad']}),
            self.config['hospitals']['vr'], self.config, {})
        self.assertTrue(pd.isna(clinical.loc[1, 'somnolencia_estupor_coma']))
        self.assertAlmostEqual(clinical.loc[0, 'creatinina'], 0.4 / 2**0.5)

    def test_conflicting_clinical_values_can_fail_or_be_audited(self):
        frame = self.normalize(pd.DataFrame([self.episode(), self.episode(temperatura=39)]))
        audit = {}
        group_episodes(frame, 'vr', self.config, audit)
        self.assertEqual(audit['clinical_conflicting_episode_counts']['temperatura'], 1)
        with self.assertRaisesRegex(ValueError, 'conflicting temperatura'):
            group_episodes(frame, 'vr', {**self.config, 'clinical_conflict_policy': 'error'}, {})

    def write_workbooks(self, directory):
        config = copy.deepcopy(self.config)
        for hospital, settings in config['hospitals'].items():
            settings.update(file=f'{hospital}.xlsx', review_file=f'{hospital}_review.xlsx', use_x1000_counts=False)
            frame = pd.DataFrame([self.episode()])
            frame['sexo'] = 'M' if hospital == 'vh' else 'HOMBRE'
            if hospital == 'rs':
                frame = frame.drop(columns='id_hemocultivo')
            with pd.ExcelWriter(directory / settings['file'], engine='openpyxl') as writer:
                frame.to_excel(writer, sheet_name='Hoja1', index=False)
                if settings['treatment_layout'] == 'wide':
                    frame.assign(fecha_antib_previo_1=pd.Timestamp('2025-12-29'),
                                 antimicrobiano_previo_1='amoxicilina').to_excel(
                        writer, sheet_name=settings['treatment_sheet'], index=False)
                elif settings['treatment_layout'] == 'long':
                    t = frame.assign(**dict.fromkeys(TREATMENT_FIELDS))
                    t['antimicrobiano_previo'] = 'meropenem'
                    t.to_excel(writer, sheet_name=settings['treatment_sheet'], index=False)
            frame.assign(estado='CONSERVAR').to_excel(directory / settings['review_file'],
                                                    sheet_name='Hemocultivos', index=False)
        return config

    def test_reference_schema_preserves_extra_antibiotic_slots(self):
        from unittest.mock import patch
        frames = {}
        for h in self.config['hospitals']:
            frame = self.normalize(pd.DataFrame([self.episode()]))
            frame['hospital_cohort'] = h
            frame['episode_id'] = h
            if h == 'ryc':
                frame['antimicrobiano_previo_28'] = 'meropenem'
            frames[h] = frame
        config = copy.deepcopy(self.config)
        for h, settings in config['hospitals'].items():
            settings['test_hospital'] = h
        def loader(raw, settings, cfg):
            return frames[settings['test_hospital']], {}, []
        with patch('merging_cohort.load_hospital', side_effect=loader), \
             patch('merging_cohort.clinical_recoding', side_effect=lambda f, *a: f), \
             patch('merging_cohort.group_episodes', side_effect=lambda f, *a: f):
            merged, audit = merge_cohorts(Path('.'), config)
        self.assertEqual(merged.loc[merged.hospital_cohort.eq('ryc'), 'antimicrobiano_previo_28'].iloc[0], 'meropenem')
        self.assertIn('antimicrobiano_previo_28', audit['hospitals']['vr']['columns_added_as_missing'])

    def test_four_xlsx_merge_and_downstream_preprocessing(self):
        from preprocess_merged_csv import preprocess_csv
        import yaml
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            config = self.write_workbooks(directory)
            merged, audit = merge_cohorts(directory, config)
            self.assertEqual(len(merged), 4)
            self.assertEqual(merged.episode_id.nunique(), 4)
            self.assertEqual(len(audit['inputs']), 8)
            source = directory / 'merged.csv'
            merged.to_csv(source, index=False, date_format='%Y-%m-%d')
            pp_config_path = DEFAULT_CONFIG.with_name('preprocess_bacthecom.yml')
            pp_config = yaml.safe_load(pp_config_path.read_text())
            _, _, filtered, _ = preprocess_csv(source, directory / 'processed.csv', pp_config, pp_config_path)
            self.assertEqual(len(filtered), 4)
            self.assertEqual(filtered.loc[0, 'antib_prev_Penicilinas'], 1)
            self.assertEqual(filtered.loc[1, 'antib_prev_Carbapenemas'], 1)
            self.assertEqual(filtered.loc[0, 'dias_desde_ultimo_antib'], 4)
            self.assertIn('infec_Ecoli', filtered)
            config_path = directory / 'merge.yml'
            config_path.write_text(yaml.safe_dump(config))
            command = [sys.executable, str(DEFAULT_CONFIG.parent.parent / 'merging_cohort.py'),
                       '--config-path', str(config_path), '--raw-dir', str(directory),
                       '--output-path', str(directory / 'cli.csv')]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            report = json.loads((directory / 'cli_merge_audit.json').read_text())
            self.assertEqual(report['unique_episode_ids'], 4)
            before = (directory / 'cli.csv').read_bytes()
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual((directory / 'cli.csv').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
