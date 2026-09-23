"""Clinical grouping and plot checks using synthetic data."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use('Agg')
import pandas as pd

from preprocess_report import (
    DEFAULT_COLORS, feature_categories, grouped_slide_missingness, load_report_config,
    write_category_missingness_panels, write_grouped_missingness_chart_png,
    write_missingness_chart_png, write_variable_distribution_chart_png,
    variable_distribution_filtered_vs_unfiltered,
)
from build_summary_excel_file import build_rows, HEADERS


class ReportCategoryTests(unittest.TestCase):
    def setUp(self):
        self.config = load_report_config(Path(__file__).parent / 'config/preprocess_bacthecom_report.yml')

    def test_specific_histories_precede_current_microbiology(self):
        names = ['infec_prev_Escherichia_coli', 'res_prev_BLEE', 'foco_prev_urinaria',
                 'resistant_prev_BLEE', 'antib_prev_Penicilinas', 'resistant_otros_BLEE',
                 'infec_Ecoli', 'foco__urinario', 'taquicardia', 'diabetes', 'mortalidad_any', 'new_unknown']
        groups = feature_categories(names, self.config['feature_categories'], [], {})
        self.assertEqual([groups[n] for n in names[:4]], ['Previous infections'] * 4)
        self.assertEqual(groups['antib_prev_Penicilinas'], 'Previous antibiotics')
        self.assertEqual(groups['resistant_otros_BLEE'], 'Other cultures')
        self.assertEqual(groups['infec_Ecoli'], 'Current microbiology')
        self.assertEqual(groups['foco__urinario'], 'Focus and severity')
        self.assertEqual(groups['taquicardia'], 'Symptoms and signs')
        self.assertEqual(groups['diabetes'], 'Comorbidities')
        self.assertEqual(groups['mortalidad_any'], 'Outcomes and post-episode information')
        self.assertEqual(groups['new_unknown'], 'Unmapped')

    def test_bands_partition_features_and_plot_exports(self):
        frame = pd.DataFrame({'edad': [1, 2], 'sexo': [0, None], 'diabetes': [None, None]})
        groups = feature_categories(list(frame), self.config['feature_categories'], [], {})
        labels = list(dict.fromkeys(groups.values()))
        logs = [{'table_name': label, 'output_columns': [c for c in frame if groups[c] == label]} for label in labels]
        domains = {label: label for label in labels}
        bands = grouped_slide_missingness(frame, logs, domains, labels)
        self.assertEqual(bands['variables'].sum(), 3)
        self.assertEqual(bands[['0%', '>0-10%', '10-40%', '40-80%', '80-100%']].sum().sum(), 3)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_missingness_chart_png(frame, output_path=root/'missing.png', colors=DEFAULT_COLORS)
            write_grouped_missingness_chart_png(frame, detailed_logs=logs, output_path=root/'grouped.png', domain_by_stage=domains, variable_group_labels=labels)
            write_category_missingness_panels(frame, groups, labels, root/'panels.png')
            distribution = variable_distribution_filtered_vs_unfiltered(full_df=frame[['edad']], filtered_df=frame,
                detailed_logs=logs, domain_by_stage=domains, variable_group_labels=labels)
            self.assertEqual(distribution.filtered_variables.sum(), 3)
            write_variable_distribution_chart_png(distribution, output_path=root/'counts.png', colors=DEFAULT_COLORS)
            for name in ['missing', 'grouped', 'panels', 'counts']:
                for extension in ['png', 'svg']:
                    self.assertGreater((root/f'{name}.{extension}').stat().st_size, 100)
            frame.to_csv(root/'full.csv', index=False)
            frame.to_csv(root/'filtered.csv', index=False)
            (root/'logs.json').write_text(json.dumps([]))
            rows = build_rows(full_dataset_path=root/'full.csv', filtered_dataset_path=root/'filtered.csv',
                              detailed_log_path=root/'logs.json', feature_domain_map=groups)
            self.assertEqual(rows[1][HEADERS.index('source domain')], 'Demographics')


if __name__ == '__main__':
    unittest.main()
