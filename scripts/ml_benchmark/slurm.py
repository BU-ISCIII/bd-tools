"""Slurm script rendering helpers."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from .types import BenchmarkConfig


def render_slurm_script(config: BenchmarkConfig, n_jobs: int) -> str:
    slurm = config.raw.get("slurm", {})
    script_path = slurm.get("script_path", "scripts/run_benchmark.py")
    config_path = str(config.path)
    return dedent(
        f"""\
        #!/bin/bash
        #SBATCH --job-name={slurm.get("job_name", "ml_benchmark")}
        #SBATCH --array=0-{n_jobs - 1}
        #SBATCH --cpus-per-task={slurm.get("cpus_per_task", 1)}
        #SBATCH --mem={slurm.get("mem", "8G")}
        #SBATCH --time={slurm.get("time", "02:00:00")}
        #SBATCH --output={slurm.get("output", "logs/benchmark_%A_%a.out")}
        #SBATCH --error={slurm.get("error", "logs/benchmark_%A_%a.err")}

        set -euo pipefail

        python {script_path} \\
          --config {config_path} \\
          --array-index "$SLURM_ARRAY_TASK_ID"
        """
    )


def write_slurm_script(config: BenchmarkConfig, n_jobs: int, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_slurm_script(config, n_jobs), encoding="utf-8")
    return output_path

