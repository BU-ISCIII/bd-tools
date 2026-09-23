"""Setup creates executable launchers without fitting or submitting anything."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from ..setup_training import setup, latest_preprocessing, SOURCE
from .full_training import read_config


class SetupTests(unittest.TestCase):
    def fixture(self, root):
        cohort = root / '20260922_bacthecom_v5'
        cohort.mkdir()
        for suffix in ('_filtered.csv', '_identifiers.csv'):
            (cohort / (cohort.name + suffix)).write_text('placeholder\n')
        (cohort / (cohort.name + '_schema.json')).write_text(json.dumps({
            'target_columns': ['mortalidad_any', 'mortalidad_14_dias', 'mortalidad_30_dias']}))
        return cohort

    def test_setup_snapshot_symlinks_and_launcher_routing(self):
        with tempfile.TemporaryDirectory(prefix='training setup ') as tmp:
            root = Path(tmp)
            cohort = self.fixture(root)
            # A stub executor records arguments, proving launchers route correctly
            # without importing models, fitting anything or requiring Slurm.
            executor = root / 'python stub'
            capture = root / 'arguments.json'
            executor.write_text('#!/usr/bin/env python3\nimport json,sys\nfrom pathlib import Path\n'
                                f'Path({str(capture)!r}).write_text(json.dumps(sys.argv[1:]))\n')
            executor.chmod(0o755)
            hpc = json.loads((SOURCE / 'config/hpc_config.json').read_text())
            hpc['python_bin'] = str(executor)
            hpc_path = root / 'hpc.json'
            hpc_path.write_text(json.dumps(hpc))
            analysis = setup(root / 'analysis', cohort, hpc_config=hpc_path,
                             analysis_name='20260922_ANALYSIS_01')
            self.assertFalse(capture.exists())  # Setup never invokes training.
            cfg = read_config(analysis / 'training_config.json')
            self.assertTrue(cfg['training']['full_refit'])
            self.assertEqual([r['mode'] for r in cfg['runs']], ['TARGET', 'TARGET', 'STAGED'])
            self.assertEqual(cfg['runs'][2]['target1'], 'mortalidad_30_dias')
            self.assertEqual(cfg['runs'][2]['target2'], 'mortalidad_14_dias')
            for name in ('cohort_filtered.csv', 'cohort_schema.json', 'cohort_identifiers.csv'):
                self.assertTrue((analysis / '00-data' / name).is_symlink())
            for name in ('lablog', '_00_validate.sh', '_00_submit.sh', '_01_run_model.sh', 'training.sbatch'):
                subprocess.run(['bash', '-n', str(analysis / name)], check=True)
            # Submission is tested with a stub; no real Slurm jobs are submitted.
            sbatch = root / 'sbatch'
            sbatch.write_text('#!/usr/bin/env python3\nimport json,sys\nfrom pathlib import Path\n'
                             f'Path({str(capture)!r}).write_text(json.dumps(sys.argv[1:]))\nprint("12345")\n')
            sbatch.chmod(0o755)
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'])
            subprocess.run(['bash', str(analysis / '_00_validate.sh')], env=env, check=True)
            self.assertEqual(json.loads(capture.read_text()), ['--parsable', 'training.sbatch', '--validate-only'])
            subprocess.run(['bash', str(analysis / '_01_run_model.sh')], env=env, check=True)
            self.assertEqual(json.loads(capture.read_text()), ['--parsable', 'training.sbatch'])
            denied = subprocess.run(['bash', str(analysis / '_run_model.sh')],
                                    env={k: v for k, v in env.items() if not k.startswith('SLURM_')},
                                    capture_output=True)
            self.assertNotEqual(denied.returncode, 0)
            job_env = dict(env, SLURM_JOB_ID='12345', SLURM_ARRAY_TASK_ID='2', SLURM_CPUS_PER_TASK='4')
            subprocess.run(['bash', str(analysis / 'training.sbatch'), '--validate-only'], env=job_env, check=True)
            args = json.loads(capture.read_text())
            self.assertIn('--validate-only', args)
            self.assertEqual(args[-2:], ['--run', 'staged_30_14_catb_seed42'])
            subprocess.run(['bash', str(analysis / 'training.sbatch')], env=job_env, check=True)
            self.assertNotIn('--validate-only', json.loads(capture.read_text()))
            changed_hpc = analysis / 'hpc_config.json'
            original_hpc = changed_hpc.read_text()
            changed_hpc.write_text(original_hpc + ' ')
            rejected = subprocess.run(['bash', str(analysis / '_00_submit.sh')], env=env, capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            changed_hpc.write_text(original_hpc)
            with self.assertRaises(FileExistsError):
                setup(root / 'analysis', cohort, analysis_name=analysis.name)
            # Regeneration preserves user changes to run configs.
            config_path = analysis / 'training_config.json'
            original = json.loads(config_path.read_text())
            original['runs'] = original['runs'][:1]
            config_path.write_text(json.dumps(original))
            subprocess.run(['bash', str(analysis / 'lablog')], check=True)
            self.assertIn('--array=0-0%1', (analysis / 'training.sbatch').read_text())

    def test_reject_missing_target_before_creating_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cohort = self.fixture(root)
            cfg = json.loads((SOURCE / 'config/training_config.json').read_text())
            cfg['runs'][0]['target'] = 'unknown_target'
            template = root / 'training.json'
            template.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, 'target_columns'):
                setup(root / 'analysis', cohort, training_config=template)
            self.assertFalse((root / 'analysis').exists())

    def test_latest_ignores_failed_and_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            folder = project / 'ANALYSIS/01-PREPROCESS'
            folder.mkdir(parents=True)
            for version in (4, 5, 6):
                run = folder / f'20260922_bacthecom_v{version}'
                run.mkdir()
                if version != 6:
                    (run / 'COMPLETE').touch()
                if version == 5:
                    (run / 'FAILED').touch()
            self.assertEqual(latest_preprocessing(project).name, '20260922_bacthecom_v4')
