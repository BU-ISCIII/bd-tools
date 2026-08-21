"""Benchmark job-matrix helpers."""

from __future__ import annotations

import csv
from pathlib import Path
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


def load_job_matrix_file(
    config: BenchmarkConfig,
    matrix_path: Path,
) -> List[BenchmarkJob]:
    target_by_name = {target.name: target for target in config.targets}
    view_by_name = {view.name: view for view in config.feature_views}
    set_by_name = {feature_set.name: feature_set for feature_set in config.feature_sets}

    jobs: List[BenchmarkJob] = []
    with matrix_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            target_name = row["target"]
            model_name = row["model"]
            feature_policy_name = row["feature_policy"]
            feature_view_name = row["feature_view"]
            feature_set_name = row["feature_set"]

            if target_name not in target_by_name:
                raise ValueError(
                    f"Matrix file {matrix_path} references unknown target '{target_name}'."
                )
            if feature_view_name not in view_by_name:
                raise ValueError(
                    f"Matrix file {matrix_path} references unknown feature view "
                    f"'{feature_view_name}'."
                )
            if feature_set_name not in set_by_name:
                raise ValueError(
                    f"Matrix file {matrix_path} references unknown feature set "
                    f"'{feature_set_name}'."
                )
            if feature_policy_name not in config.feature_policies:
                raise ValueError(
                    f"Matrix file {matrix_path} references unknown feature policy "
                    f"'{feature_policy_name}'."
                )

            jobs.append(
                BenchmarkJob(
                    index=int(row["index"]),
                    target=target_by_name[target_name],
                    model_name=model_name,
                    feature_policy=config.feature_policies[feature_policy_name],
                    feature_view=view_by_name[feature_view_name],
                    feature_set=set_by_name[feature_set_name],
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
    indices = sorted(job.index for job in jobs)
    ranges = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append(f"{start}-{previous}" if start != previous else str(start))
        start = previous = index
    ranges.append(f"{start}-{previous}" if start != previous else str(start))
    return f"#SBATCH --array={','.join(ranges)}"


def format_dense_slurm_array(jobs: List[BenchmarkJob]) -> str:
    if not jobs:
        raise ValueError("Cannot build a Slurm array for an empty job matrix.")
    return f"#SBATCH --array=0-{len(jobs) - 1}"
