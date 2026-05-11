"""Benchmark job-matrix helpers."""

from __future__ import annotations

from typing import Iterable, List

from .types import BenchmarkConfig, BenchmarkJob


def build_job_matrix(config: BenchmarkConfig) -> List[BenchmarkJob]:
    jobs: List[BenchmarkJob] = []
    for target in config.targets:
        for model_name in config.models:
            feature_policy = config.feature_policies[
                config.model_feature_policies.get(model_name, "default")
            ]
            for feature_view in config.feature_views:
                for feature_set in config.feature_sets:
                    jobs.append(
                        BenchmarkJob(
                            index=len(jobs),
                            target=target,
                            model_name=model_name,
                            feature_policy=feature_policy,
                            feature_view=feature_view,
                            feature_set=feature_set,
                        )
                    )
    return jobs


def get_job_by_index(jobs: List[BenchmarkJob], index: int) -> BenchmarkJob:
    if index < 0 or index >= len(jobs):
        raise IndexError(
            f"Array index {index} is outside the benchmark matrix 0-{len(jobs) - 1}."
        )
    return jobs[index]


def format_job_matrix(jobs: Iterable[BenchmarkJob]) -> str:
    rows = [
        "index\ttarget\ttask_type\tmodel\tfeature_policy\tfeature_view\tfeature_set"
    ]
    for job in jobs:
        rows.append(
            "\t".join(
                [
                    str(job.index),
                    job.target.name,
                    job.target.task_type,
                    job.model_name,
                    job.feature_policy.name,
                    job.feature_view.name,
                    job.feature_set.name,
                ]
            )
        )
    return "\n".join(rows)


def format_slurm_array(jobs: List[BenchmarkJob]) -> str:
    if not jobs:
        raise ValueError("Cannot build a Slurm array for an empty job matrix.")
    return f"#SBATCH --array=0-{len(jobs) - 1}"
