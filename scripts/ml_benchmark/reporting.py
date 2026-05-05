"""Homogeneous output contracts for benchmark jobs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .types import BenchmarkJob, DiagnosticResult, ResourceLimits


def job_output_dir(base_output_dir: Path, job: BenchmarkJob) -> Path:
    return base_output_dir / "jobs" / job.slug


def write_diagnostics_manifest(
    output_dir: Path, diagnostics: Iterable[DiagnosticResult]
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "diagnostics_manifest.json"
    path.write_text(
        json.dumps([asdict(item) for item in diagnostics], indent=2),
        encoding="utf-8",
    )
    return path


def write_job_summary(
    output_dir: Path,
    *,
    job: BenchmarkJob,
    status: str,
    resources: ResourceLimits,
    metrics: Dict[str, Any] | None = None,
    diagnostic_results: List[DiagnosticResult] | None = None,
    extra: Dict[str, Any] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "status": status,
        "target": asdict(job.target),
        "model": job.model_name,
        "feature_policy": asdict(job.feature_policy),
        "feature_view": asdict(job.feature_view),
        "feature_set": asdict(job.feature_set),
        "job_index": job.index,
        "resources": asdict(resources),
        "metrics": metrics or {},
        "diagnostics": [asdict(item) for item in diagnostic_results or []],
    }
    if extra:
        payload.update(extra)
    path = output_dir / "summary.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
