"""Slurm script rendering helpers."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from .jobs import format_dense_slurm_array, format_slurm_array
from .types import BenchmarkConfig, BenchmarkJob


def render_slurm_script(
    config: BenchmarkConfig,
    jobs: list[BenchmarkJob],
    *,
    invoked_script_path: Path | str | None = None,
    dense_array: bool = False,
    job_matrix_file: Path | str | None = None,
    array_start: int | None = None,
    array_end: int | None = None,
    array_offset: int = 0,
) -> str:
    slurm = config.raw.get("slurm", {})

    script_path = _resolve_script_path(
        slurm.get("script_path", "ml_benchmark/run_benchmark.py"),
        invoked_script_path=invoked_script_path,
    )

    config_path = str(config.path)

    # If an explicit array range is supplied, use it.
    # Otherwise retain the original behaviour.
    if array_start is not None and array_end is not None:
        array_line = f"#SBATCH --array={array_start}-{array_end}"
    else:
        array_line = (
            format_dense_slurm_array(jobs)
            if dense_array
            else format_slurm_array(jobs)
        )

    max_parallel_tasks = slurm.get("max_parallel_tasks")

    if max_parallel_tasks:
        array_line = f"{array_line}%{int(max_parallel_tasks)}"

    cpus_per_task = int(slurm.get("cpus_per_task", 1))
    python_bin = slurm.get("python_bin", "python3.11")

    optional_directives = []

    if slurm.get("partition"):
        optional_directives.append(
            f'#SBATCH --partition={slurm["partition"]}'
        )

    if slurm.get("nodelist"):
        optional_directives.append(
            f'#SBATCH --nodelist={slurm["nodelist"]}'
        )

    job_matrix_arg = (
        f"  --job-matrix-file {job_matrix_file} \\\n"
        if job_matrix_file
        else ""
    )

    array_offset_arg = (
        f"          --array-offset {array_offset} \\\n"
        if array_offset
        else ""
    )

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

        if [[ -z "${{PYTHON_BIN:-}}" ]]; then
          for candidate in ./venv_ml_benchmark/bin/python ../../venv_ml_benchmark/bin/python; do
            if [[ -x "$candidate" ]]; then
              PYTHON_BIN="$candidate"
              break
            fi
          done
        fi

        export PYTHON_BIN="${{PYTHON_BIN:-{python_bin}}}"

        "${{PYTHON_BIN}}" {script_path} \\
          --config {config_path} \\
{job_matrix_arg}{array_offset_arg}          --array-index "$SLURM_ARRAY_TASK_ID"
        """
    )

    return "\n".join(directives) + "\n" + body


def _resolve_script_path(
    script_path: str,
    *,
    invoked_script_path: Path | str | None = None,
) -> str:
    path = Path(script_path)

    if path.is_absolute():
        return str(path)

    if invoked_script_path is not None:
        invoked = Path(invoked_script_path)

        if invoked.name == Path(script_path).name:
            return str(invoked.resolve())

    repo_relative = Path(__file__).resolve().parents[1] / path

    if repo_relative.exists():
        return str(repo_relative)

    return str(path.resolve())


def write_slurm_script(
    config: BenchmarkConfig,
    jobs: list[BenchmarkJob],
    output_path: Path,
    *,
    invoked_script_path: Path | str | None = None,
    dense_array: bool = False,
    job_matrix_file: Path | str | None = None,
    array_start: int | None = None,
    array_end: int | None = None,
    array_offset: int = 0,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        render_slurm_script(
            config,
            jobs,
            invoked_script_path=invoked_script_path,
            dense_array=dense_array,
            job_matrix_file=job_matrix_file,
            array_start=array_start,
            array_end=array_end,
            array_offset=array_offset,
        ),
        encoding="utf-8",
    )

    return output_path