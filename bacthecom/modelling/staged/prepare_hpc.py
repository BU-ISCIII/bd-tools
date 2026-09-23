"""Generate Slurm scripts from run/config files without ever submitting jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess


def quote(value):
    return shlex.quote(str(value))


def generate(config, hpc_path, validate=False):
    config = config.resolve()
    hpc_path = hpc_path.resolve()
    root = config.parent
    # Resolve the source project, independently of the analysis folder depth.
    project = Path(__file__).resolve().parents[5]
    cfg = json.loads(config.read_text())
    hpc = json.loads(hpc_path.read_text())
    runs = [run['name'] for run in cfg['runs']]
    if not runs or len(set(runs)) != len(runs):
        raise ValueError('Run names must be unique and nonempty')
    for name in runs:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in ('.', '..'):
            raise ValueError(f'Invalid run name: {name}')
    cpus = cfg['training']['n_cpus']
    if not isinstance(cpus, int) or cpus < 1:
        raise ValueError('Invalid CPU count')
    for field in ('job_name', 'partition', 'time', 'memory', 'account', 'qos'):
        if hpc.get(field) and not re.fullmatch(r'[A-Za-z0-9_:.,-]+', hpc[field]):
            raise ValueError(f'Invalid Slurm {field}')
    concurrency = hpc['max_parallel_runs']
    if not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError('Invalid max_parallel_runs')
    executor = hpc['executor']
    if executor not in ('python', 'singularity'):
        raise ValueError('executor must be python or singularity')
    for module in hpc.get('modules', []):
        if not re.fullmatch(r'[A-Za-z0-9_./+-]+', module):
            raise ValueError('Invalid environment module')
    setup = '\n'.join('module load ' + quote(m) for m in hpc.get('modules', []))
    environment = hpc.get('environment_script')
    if environment:
        environment = str((root / environment).resolve())
        setup += '\nsource ' + quote(environment)
    common = f'''#!/bin/bash
set -euo pipefail
: "${{SLURM_JOB_ID:?Use _01_run_model.sh to submit training through Slurm}}"
if (( ${{SLURM_CPUS_PER_TASK:-0}} < {cpus} )); then
    echo 'Insufficient allocated CPUs for configured model threads' >&2
    exit 1
fi
cd {quote(root)}
{setup}
export MPLBACKEND=Agg
export PYTHONPATH={quote(project / 'DOC/bd-tools')}"${{PYTHONPATH:+:$PYTHONPATH}}"
export OMP_NUM_THREADS={cpus}
export OPENBLAS_NUM_THREADS={cpus}
export MKL_NUM_THREADS={cpus}
'''
    if executor == 'python':
        launch = f'{quote(hpc["python_bin"])} -m bacthecom.modelling.staged.cli'
    else:
        image = (root / hpc['container_image']).resolve()
        if not image.is_file():
            raise ValueError(f'Container image missing: {image}')
        launch = f'singularity exec --cleanenv --bind {quote(str(project) + ":" + str(project))} --env PYTHONPATH={quote(project / "DOC/bd-tools")},MPLBACKEND=Agg,OMP_NUM_THREADS={cpus},OPENBLAS_NUM_THREADS={cpus},MKL_NUM_THREADS={cpus} {quote(image)} python3 -m bacthecom.modelling.staged.cli'
    launcher = common + f'exec {launch} --config {quote(config)} "$@"\n'
    (root / '_run_model.sh').write_text(launcher)
    (root / '_00_validate.sh').write_text(
        '#!/bin/bash\nset -euo pipefail\nexec bash ' + quote(root / '_00_submit.sh') + ' --validate-only \"$@\"\n')
    (root / 'logs').mkdir(exist_ok=True)
    directives = [f'--job-name={hpc["job_name"]}', f'--chdir={quote(root)}',
                  f'--output={quote(root / "logs/%x_%A_%a.out")}', f'--error={quote(root / "logs/%x_%A_%a.err")}',
                  f'--time={hpc["time"]}', '--ntasks=1', f'--cpus-per-task={cpus}', f'--mem={hpc["memory"]}',
                  f'--array=0-{len(runs)-1}%{min(concurrency, len(runs))}']
    for field in ('partition', 'account', 'qos'):
        if hpc.get(field):
            directives.append(f'--{field}={hpc[field]}')
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    hpc_digest = hashlib.sha256(hpc_path.read_bytes()).hexdigest()
    guard = f'''
if [[ "$(sha256sum {quote(config)} | cut -d ' ' -f 1)" != {quote(digest)} || \
      "$(sha256sum {quote(hpc_path)} | cut -d ' ' -f 1)" != {quote(hpc_digest)} ]]; then
    echo 'Training or HPC config changed: rerun bash lablog before submission.' >&2
    exit 1
fi
''' 
    sbatch = '#!/bin/bash\n' + '\n'.join('#SBATCH ' + d for d in directives) + f'''
set -euo pipefail
cd {quote(root)}
{guard}
RUNS=({' '.join(quote(r) for r in runs)})
TASK_ID="${{SLURM_ARRAY_TASK_ID:?Submit as a Slurm array}}"
if [[ ! "$TASK_ID" =~ ^[0-9]+$ ]] || (( TASK_ID >= ${{#RUNS[@]}} )); then
    echo 'Invalid Slurm array task index' >&2
    exit 1
fi
if (( ${{SLURM_CPUS_PER_TASK:-0}} < {cpus} )); then
    echo 'Allocated CPUs are below configured model threads' >&2
    exit 1
fi
bash {quote(root / '_run_model.sh')} --validate-only --run "${{RUNS[$TASK_ID]}}"
if [[ "${{1:-}}" == '--validate-only' ]]; then
    exit 0
fi
exec bash {quote(root / '_run_model.sh')} --run "${{RUNS[$TASK_ID]}}"
'''
    (root / 'training.sbatch').write_text(sbatch)
    submit = f'''#!/bin/bash
set -euo pipefail
cd {quote(root)}
mkdir -p logs
command -v sbatch >/dev/null || {{ echo 'sbatch is not available on this host' >&2; exit 1; }}
if (( $# > 1 )) || [[ "${{1:-}}" != '' && "${{1:-}}" != '--validate-only' ]]; then
    echo 'Usage: launcher [--validate-only]' >&2
    exit 1
fi
{guard}
JOB_ID=$(sbatch --parsable training.sbatch "$@")
printf '%s\n' "$JOB_ID" | tee -a submitted_jobs.txt
printf 'Submitted array %s. Logs: %s/logs/\n' "$JOB_ID" "$PWD"
'''
    (root / '_00_submit.sh').write_text(submit)
    (root / '_01_run_model.sh').write_text(
        '#!/bin/bash\nset -euo pipefail\nexec bash ' + quote(root / '_00_submit.sh') + ' "$@"\n')
    for filename in ('_00_validate.sh', '_01_run_model.sh', '_00_submit.sh', '_run_model.sh', 'training.sbatch'):
        path = root / filename
        path.chmod(path.stat().st_mode | 0o110)
        subprocess.run(['bash', '-n', str(path)], check=True)
    # This uses the same executor/environment as the scheduled jobs, but only validates.
    if validate:
        raise ValueError('Use --skip-validation to generate scripts, then _00_validate.sh to submit validation on compute nodes.')
    manifest = {'config_sha256': digest, 'runs': runs, 'executor': executor, 'cpus_per_task': cpus,
                'status': ('scripts generated; configured environment and data preflight passed; no jobs submitted'
                           if validate else 'scripts generated; data preflight pending; no jobs submitted')}
    (root / 'hpc_preflight.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Prepared training.sbatch, _00_submit.sh and _01_run_model.sh. No jobs submitted.')
    print(f'When ready: bash {root / "_00_submit.sh"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--hpc-config', required=True, type=Path)
    parser.add_argument('--skip-validation', action='store_true', help='Generate launchers only; validate separately before training.')
    args = parser.parse_args()
    generate(args.config, args.hpc_config, validate=False)
