"""Prepare a versioned training workspace; never fit models or submit jobs."""
import argparse
from datetime import date
import json
import os
from pathlib import Path
import re
import subprocess
import sys

SOURCE = Path(__file__).resolve().parent
PROJECT = SOURCE.parents[3]


def latest_preprocessing(project):
    candidates = []
    for folder in (project / 'ANALYSIS/01-PREPROCESS').glob('*_bacthecom_v*'):
        match = re.fullmatch(r'(\d{8})_bacthecom_v(\d+)', folder.name)
        if match and (folder / 'COMPLETE').is_file() and not (folder / 'FAILED').exists():
            candidates.append((int(match[2]), match[1], folder))
    if not candidates:
        raise ValueError('No completed preprocessing run found; pass --preprocess-dir explicitly.')
    return max(candidates)[2]


def setup(workspace, preprocess_dir=None, training_config=None, hpc_config=None, analysis_name=None):
    workspace = Path(workspace).resolve()
    preprocess = Path(preprocess_dir).resolve() if preprocess_dir else latest_preprocessing(PROJECT)
    inputs = {name: preprocess / (preprocess.name + suffix) for name, suffix in {
        'cohort_filtered.csv': '_filtered.csv', 'cohort_schema.json': '_schema.json',
        'cohort_identifiers.csv': '_identifiers.csv'}.items()}
    for path in inputs.values():
        if not path.is_file():
            raise ValueError(f'Missing preprocessing artifact: {path}')
    template = Path(training_config or SOURCE / 'config/training_config.json').resolve()
    hpc_template = Path(hpc_config or SOURCE / 'config/hpc_config.json').resolve()
    cfg, hpc = json.loads(template.read_text()), json.loads(hpc_template.read_text())
    cfg.update(data='00-data/cohort_filtered.csv', schema='00-data/cohort_schema.json',
               identifiers='00-data/cohort_identifiers.csv')
    # Every new analysis owns its results; never inherit an old absolute output path.
    output = Path(cfg['output_dir'])
    if output.is_absolute() or '..' in output.parts or str(output) in ('.', ''):
        raise ValueError('output_dir must be a subdirectory within the new analysis')
    schema = json.loads(inputs['cohort_schema.json'].read_text())
    targets = set(schema['target_columns'])
    names = []
    for run in cfg['runs']:
        if run.get('mode') not in ('TARGET', 'STAGED'):
            raise ValueError('Each run must use TARGET or STAGED mode')
        keys = ('target',) if run['mode'] == 'TARGET' else ('target1', 'target2')
        for key in keys:
            if run.get(key) not in targets:
                raise ValueError(f"Run {run['name']}: {key} must be in preprocessing target_columns")
        name = run['name']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in ('.', '..'):
            raise ValueError(f'Invalid run name: {name}')
        names.append(name)
    if not names or len(set(names)) != len(names):
        raise ValueError('Configure at least one run with unique names')
    workspace.mkdir(parents=True, exist_ok=True)
    if analysis_name is None:
        numbers = [int(m[1]) for p in workspace.iterdir()
                   if (m := re.fullmatch(r'\d{8}_ANALYSIS_(\d+)(?:_.*)?', p.name))]
        analysis_name = f'{date.today():%Y%m%d}_ANALYSIS_{max(numbers, default=0) + 1:02d}'
    if not re.fullmatch(r'\d{8}_ANALYSIS_\d+(?:_[A-Za-z0-9_-]+)?', analysis_name):
        raise ValueError('Analysis name must be YYYYMMDD_ANALYSIS_NN with an optional suffix')
    root = workspace / analysis_name
    root.mkdir()  # Existing analyses are never reused or overwritten.
    (root / '00-data').mkdir()
    for name, path in inputs.items():
        (root / '00-data' / name).symlink_to(os.path.relpath(path, root / '00-data'))
    (root / 'training_config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    (root / 'hpc_config.json').write_text(json.dumps(hpc, indent=2) + '\n')
    generator = SOURCE / 'staged/prepare_hpc.py'
    (root / 'prepare_hpc.py').symlink_to(os.path.relpath(generator, root))
    lablog = '''#!/bin/bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BIN:-''' + sys.executable + '''}" "$HERE/prepare_hpc.py" \\
    --config "$HERE/training_config.json" --hpc-config "$HERE/hpc_config.json" --skip-validation "$@"
'''
    (root / 'lablog').write_text(lablog)
    (root / 'lablog').chmod(0o775)
    subprocess.run(['bash', str(root / 'lablog')], check=True)
    manifest = {'preprocessing_directory': str(preprocess), 'training_template': str(template),
                'hpc_template': str(hpc_template), 'runs': names,
                'status': 'Setup only; no training or submission performed'}
    (root / 'setup_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (root / 'README.md').write_text('''# Training workspace

Edit `training_config.json` to select runs, algorithms, seeds and training settings.
Edit `hpc_config.json` for the execution environment and Slurm resources.
After either change run `bash lablog` to regenerate launchers.

- `bash _00_validate.sh`: submit validation-only Slurm jobs; no model training.
- `bash _01_run_model.sh`: submit all configured runs as a Slurm array.
- Edit the runs list to choose which models the array trains.
- `bash _00_submit.sh`: submit the configured runs as a Slurm array.

The internal `_run_model.sh` worker requires a Slurm allocation.
Completed/existing run directories are not overwritten. Choose a new run name
or output directory for another fit. Run configuration and data hashes are saved
with trained models. Inputs are symlinks to the selected preprocessing run;
keep that run unchanged for reproducibility.

Default targets are admission-referenced mortality at 30 and 14 days.
STAGED fits stage 2 among observed 30-day deaths and combines probabilities
as P(death by 30 days) × P(death by 14 days | death by 30 days).
Holdout metrics use evaluation models; full-data models are exported separately.
Training includes configured Optuna search, plots, metrics and SHAP explanations.
''')
    print(f'Created {root}\nEdit its configs, then run _00_validate.sh before training.')
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=PROJECT / 'ANALYSIS/04-TRAINING')
    parser.add_argument('--preprocess-dir', type=Path)
    parser.add_argument('--training-config', type=Path)
    parser.add_argument('--hpc-config', type=Path)
    parser.add_argument('--analysis-name')
    args = parser.parse_args()
    setup(**vars(args))
