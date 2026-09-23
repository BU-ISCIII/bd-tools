"""Regression tests: preprocessing partitions, grouped splits and serialized inference."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from .full_training import MortalityPreprocessor, grouped_folds, train_run, predict


class TrainingTests(unittest.TestCase):
    def test_target_like_exclusions(self):
        from .full_training import exclude_target_like_features
        kept, dropped = exclude_target_like_features(
            ['edad', 'Mortalidad_30', 'Death_14', 'TARGET_copy', 'early_14', 'fecha_alta',
             'actual_endpoint_probability', 'hospital_duration', 'lactato'],
            ['actual_endpoint'], patterns=['hospital_*'])
        self.assertEqual(kept, ['edad', 'lactato'])
        self.assertEqual(len(dropped), 7)

    def test_trial_pruning_and_cv_tree_count(self):
        import optuna
        from .full_training import fit_stage
        class AfterFirstTrial(optuna.pruners.BasePruner):
            def prune(self, study, trial):
                return trial.number > 0 and trial.last_step >= 2
        x = pd.DataFrame(np.random.default_rng(1).normal(size=(120, 3)), columns=list('abc'))
        y = pd.Series(np.tile([0, 1], 60))
        cfg = {'training': {'cv_splits': 3, 'n_trials': 2, 'n_cpus': 1,
                           'class_weighting': False, 'early_stopping_rounds': 3},
               'postprocessing': {}}
        run = {'algorithm': 'catb', 'seed': 42, 'name': 'pruning_test'}
        with tempfile.TemporaryDirectory() as tmp, patch('bacthecom.modelling.staged.full_training.get_param_space',
                side_effect=lambda *a, **kw: {'iterations': 12, 'depth': 2, 'thread_count': 1}), patch(
                'bacthecom.modelling.staged.full_training.optuna.pruners.MedianPruner', return_value=AfterFirstTrial()):
            folder = Path(tmp) / 'stage1'
            model, params = fit_stage(x, y, pd.Series(np.arange(120)), cfg, run, folder)
            trials = pd.read_csv(folder / 'trials.csv')
            self.assertEqual(trials.state.tolist(), ['COMPLETE', 'PRUNED'])
            summary = json.loads((folder / 'optimization_summary.json').read_text())
            self.assertEqual(params['iterations'], max(1, int(np.median(summary['best_fold_tree_counts']))))
            self.assertLessEqual(params['iterations'], 12)
            self.assertEqual(model['model'].tree_count_, params['iterations'])

    def test_zero_iqr_binary_and_missingness_after_masking(self):
        x = pd.DataFrame({'sparse': [0.] * 18 + [1., 100.],
                          'binary': [0, 1] * 10,
                          'continuous': list(range(19)) + [1000.],
                          'all_missing': [np.nan] * 20})
        prep = MortalityPreprocessor(missing_percent=4).fit(x)
        self.assertNotIn('sparse', prep.bounds_)
        self.assertNotIn('binary', prep.bounds_)
        self.assertNotIn('continuous', prep.kept_)
        self.assertNotIn('all_missing', prep.kept_)
        self.assertIn('sparse', prep.output_columns_)
        pd.testing.assert_frame_equal(prep.transform(x), prep.transform(x.copy()))

    def test_transform_does_not_learn_validation_values(self):
        x = pd.DataFrame({'x': [1., 2., np.nan, 4., 5., 6.], 'z': [1., 2., 3., 4., 5., 6.]})
        prep = MortalityPreprocessor().fit(x)
        before = dict(prep.bounds_)
        median = prep.medians_['x']
        result = prep.transform(pd.DataFrame({'x': [np.nan, 1e9], 'z': [3., 4.]}))
        self.assertEqual(prep.bounds_, before)
        self.assertEqual(result.x.tolist(), [median, median])
        self.assertEqual(result.x__missing_indicator.tolist(), [1., 1.])

    def test_groups_and_insufficient_classes(self):
        x = pd.DataFrame({'x': np.arange(120)})
        y = pd.Series(np.tile([0, 0, 1, 1], 30))
        groups = pd.Series(np.repeat(np.arange(60), 2))
        for tr, va in grouped_folds(x, y, groups, 5, 42):
            self.assertFalse(set(groups.iloc[tr]) & set(groups.iloc[va]))
        with self.assertRaises(ValueError):
            grouped_folds(x, pd.Series([0] * 119 + [1]), groups, 5, 42)

    def test_direct_staged_full_refit_and_reload(self):
        rng = np.random.default_rng(42)
        x = pd.DataFrame(rng.normal(size=(240, 4)), columns=list('abcd'))
        x.loc[::7, 'a'] = np.nan
        x['copy_b'] = x.b
        data = x.copy()
        data['day30'] = np.tile([0, 0, 1, 1, 1, 1], 40)
        data['day14'] = np.tile([0, 0, 0, 0, 1, 1], 40)
        groups = pd.Series(np.repeat(np.arange(120), 2))
        cfg = {'postprocessing': {'missing_percent': 80, 'iqr_multiplier': 5, 'max_corr': .99, 'missing_indicators': True},
               'training': {'cv_splits': 3, 'holdout_folds': 3, 'n_trials': 1, 'n_cpus': 1,
                            'threshold': .5, 'class_weighting': False, 'full_refit': True}}
        params = {'iterations': 12, 'depth': 2, 'thread_count': 1, 'random_seed': 42, 'allow_writing_files': False}
        with tempfile.TemporaryDirectory() as tmp, patch('bacthecom.modelling.staged.full_training.get_param_space', return_value=params):
            root = Path(tmp)
            raw = root / 'raw.csv'
            x.to_csv(raw, index=False)
            for mode in ['TARGET', 'STAGED']:
                run = {'name': mode.lower(), 'mode': mode, 'algorithm': 'catb', 'seed': 42,
                       'target': 'day14'} if mode == 'TARGET' else {'name': 'staged', 'mode': mode, 'algorithm': 'catb', 'seed': 42, 'target1': 'day30', 'target2': 'day14'}
                out = root / run['name']
                with io.StringIO() as buffer, patch('sys.stdout', buffer):
                    train_run(data, x, groups, cfg, run, out)
                    output = buffer.getvalue()
                self.assertTrue((out / 'COMPLETE.json').exists())
                self.assertTrue((out / 'summary_metrics.json').exists())
                self.assertIn('Holdout summary', output)
                self.assertNotIn('Trial ', output)
                for stage in (['stage1', 'stage2'] if mode == 'STAGED' else ['stage1']):
                    for population in ['holdout', 'full']:
                        folder = out / stage / ('explanations_' + population)
                        ranked = pd.read_csv(folder / 'ranked_variables.csv')
                        self.assertTrue(ranked.mean_abs_shap.is_monotonic_decreasing)
                        values = pd.read_csv(folder / 'shap_values.csv', index_col=0)
                        base = pd.read_csv(folder / 'shap_base_values.csv', index_col=0)
                        np.testing.assert_allclose(values.sum(axis=1) + base.base_value, base.raw_margin, atol=1e-6)
                        self.assertTrue((folder / 'shap_summary.pdf').stat().st_size > 0)
                        self.assertTrue((folder / 'shap_importance.png').stat().st_size > 0)
                        if population == 'holdout':
                            heldout = pd.read_csv(out / 'split.csv')
                            self.assertTrue(set(values.index) <= set(heldout.loc[heldout.partition == 'test', 'row_index']))
                            if stage == 'stage2':
                                self.assertTrue((data.loc[values.index, 'day30'] == 1).all())
                predict(out, raw, root / 'pred.csv')
                predictions = pd.read_csv(root / 'pred.csv')
                self.assertEqual(len(predictions), len(x))
                if mode == 'STAGED':
                    self.assertTrue((predictions.p_14_joint <= predictions.p_stage1).all())
                    self.assertTrue((predictions.loc[predictions.p_stage1 < .5, 'p_14_hard_gate'] == 0).all())
                pipeline = joblib.load(out / 'stage1/full_model.joblib')
                self.assertFalse({'b', 'copy_b'} <= set(pipeline['postprocessing'].output_columns_))
                split = pd.read_csv(out / 'split.csv')
                self.assertFalse(set(groups.loc[split.loc[split.partition == 'train', 'row_index']]) & set(groups.loc[split.loc[split.partition == 'test', 'row_index']]))
                with self.assertRaises(FileExistsError):
                    train_run(data, x, groups, cfg, run, out)


if __name__ == '__main__':
    unittest.main()
