#!/usr/bin/env python3
"""CLI entrypoint for the reusable predictive ML benchmark package."""

from __future__ import annotations

import sys

if __package__ in {None, ""}:
    script_dir = sys.path[0]
    if script_dir.endswith("ml_benchmark"):
        sys.path.pop(0)
        sys.path.insert(0, script_dir.rsplit("/", 1)[0] or ".")

import argparse
import json
from pathlib import Path

from ml_benchmark.config import load_config
from ml_benchmark.jobs import (
    build_job_matrix,
    format_job_matrix,
    format_slurm_array,
    get_job_by_index,
)
from ml_benchmark.reporting import build_aggregate_report
from ml_benchmark.runner import get_output_dir, run_job
from ml_benchmark.slurm import render_slurm_script, write_slurm_script


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predictive ML benchmark runner.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--array-index", type=int)
    parser.add_argument("--target")
    parser.add_argument("--model")
    parser.add_argument("--feature-view")
    parser.add_argument("--feature-set")
    parser.add_argument("--print-job-matrix", action="store_true")
    parser.add_argument("--print-slurm-array", action="store_true")
    parser.add_argument("--print-slurm-script", action="store_true")
    parser.add_argument("--write-slurm-script", type=Path)
    parser.add_argument("--build-report", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def select_jobs(args: argparse.Namespace, jobs):
    selected = jobs
    if args.array_index is not None:
        return [get_job_by_index(jobs, args.array_index)]
    if args.target:
        selected = [job for job in selected if job.target.name == args.target]
    if args.model:
        selected = [job for job in selected if job.model_name == args.model]
    if args.feature_view:
        selected = [
            job for job in selected if job.feature_view.name == args.feature_view
        ]
    if args.feature_set:
        selected = [job for job in selected if job.feature_set.name == args.feature_set]
    return selected


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    jobs = build_job_matrix(config)
    selected = select_jobs(args, jobs)

    if args.print_job_matrix:
        print(format_job_matrix(selected))
        return
    if args.print_slurm_array:
        print(format_slurm_array(selected))
        return
    if args.print_slurm_script:
        print(render_slurm_script(config, len(jobs)))
        return
    if args.write_slurm_script:
        path = write_slurm_script(config, len(jobs), args.write_slurm_script)
        print(path)
        return
    if args.build_report:
        report = build_aggregate_report(get_output_dir(config))
        print(json.dumps(report, indent=2))
        return

    if not selected:
        raise ValueError("No benchmark jobs matched the requested filters.")
    results = [run_job(config, job, dry_run=args.dry_run) for job in selected]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
