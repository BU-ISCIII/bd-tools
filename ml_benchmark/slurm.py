"""Slurm script rendering helpers."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from .jobs import format_slurm_array
from .types import BenchmarkConfig
from .types import BenchmarkJob


def render_slurm_script(config: BenchmarkConfig, jobs: list[BenchmarkJob]) -> str:
    slurm = config.raw.get("slurm", {})
    script_path = slurm.get("script_path", "ml_benchmark/run_benchmark.py")
    config_path = str(config.path)
    array_line = format_slurm_array(jobs)
    max_parallel_tasks = slurm.get("max_parallel_tasks")
    if max_parallel_tasks:
        array_line = f"{array_line}%{int(max_parallel_tasks)}"
    cpus_per_task = int(slurm.get("cpus_per_task", 1))
    optional_directives = []
    if slurm.get("partition"):
        optional_directives.append(f'#SBATCH --partition={slurm["partition"]}')
    if slurm.get("nodelist"):
        optional_directives.append(f'#SBATCH --nodelist={slurm["nodelist"]}')
    directives = [
        "#!/bin/bash",
        f'#SBATCH --job-name={slurm.get("job_name", "ml_benchmark")}',
        array_line,
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        *optional_directives,
        f'#SBATCH --mem={slurm.get("mem", "8G")}',
        f'#SBATCH --time={slurm.get("time", "02:00:00")}',
        f'#SBATCH --output={slurm.get("output", "logs/benchmark_%A_%a.out")}',
    ]
    body = dedent(
        f"""\

        set -euo pipefail
        export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-{cpus_per_task}}}"
        export OPENBLAS_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-{cpus_per_task}}}"
        export MKL_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-{cpus_per_task}}}"
        export NUMEXPR_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-{cpus_per_task}}}"
        export OMP_DYNAMIC=FALSE
        export MKL_DYNAMIC=FALSE

        python {script_path} \\
          --config {config_path} \\
          --array-index "$SLURM_ARRAY_TASK_ID"
        """
    )
    return "\n".join(directives) + "\n" + body


def write_slurm_script(
    config: BenchmarkConfig,
    jobs: list[BenchmarkJob],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_slurm_script(config, jobs), encoding="utf-8")
    return output_path
