"""Resource-limit helpers for local and Slurm execution."""

from __future__ import annotations

import os
from typing import Dict

from .types import ResourceLimits


THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]


def resolve_n_jobs(default_n_jobs: int = 1) -> int:
    raw = os.environ.get("SLURM_CPUS_PER_TASK")
    if raw:
        return max(1, int(raw))
    return max(1, int(default_n_jobs))


def apply_thread_limits(n_jobs: int) -> Dict[str, str]:
    value = str(max(1, int(n_jobs)))
    applied = {}
    for key in THREAD_ENV_VARS:
        os.environ[key] = value
        applied[key] = value
    return applied


def configure_resources(default_n_jobs: int = 1) -> ResourceLimits:
    n_jobs = resolve_n_jobs(default_n_jobs)
    env = apply_thread_limits(n_jobs)
    return ResourceLimits(
        n_jobs=n_jobs,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
        environment=env,
    )

