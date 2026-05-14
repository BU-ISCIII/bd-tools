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
    return dedent(
        f"""\
        #!/bin/bash
        #SBATCH --job-name={slurm.get("job_name", "ml_benchmark")}
        {array_line}
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


def write_slurm_script(
    config: BenchmarkConfig,
    jobs: list[BenchmarkJob],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_slurm_script(config, jobs), encoding="utf-8")
    return output_path
