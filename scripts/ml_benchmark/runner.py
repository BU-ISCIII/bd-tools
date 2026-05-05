"""Benchmark runner skeleton.

The first implementation establishes the job/output contract. Training,
feature selection, tuning, and diagnostics will plug into this runner next.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .reporting import job_output_dir, write_diagnostics_manifest, write_job_summary
from .resources import configure_resources
from .types import BenchmarkConfig, BenchmarkJob, DiagnosticResult


def get_output_dir(config: BenchmarkConfig) -> Path:
    return Path(config.raw.get("output_dir", "outputs/ml_benchmark"))


def run_job(config: BenchmarkConfig, job: BenchmarkJob, *, dry_run: bool = False) -> Dict[str, Any]:
    resources = configure_resources(
        int(config.raw.get("compute", {}).get("default_n_jobs", 1))
    )
    output_dir = job_output_dir(get_output_dir(config), job)
    diagnostics = [
        DiagnosticResult(
            name="job_contract",
            status="planned" if dry_run else "pending_implementation",
            message="Training/evaluation implementation will be added in the next slice.",
        )
    ]
    write_diagnostics_manifest(output_dir, diagnostics)
    write_job_summary(
        output_dir,
        job=job,
        status="planned" if dry_run else "pending_implementation",
        resources=resources,
        diagnostic_results=diagnostics,
    )
    return {
        "job": job.slug,
        "status": "planned" if dry_run else "pending_implementation",
        "output_dir": str(output_dir),
    }

