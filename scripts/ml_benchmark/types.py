"""Shared dataclasses for benchmark configuration, jobs, and reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


TaskType = str


@dataclass(frozen=True)
class TargetSpec:
    name: str
    task_type: TaskType
    columns: List[str] = field(default_factory=list)
    positive_label: Optional[Any] = None
    main_metric: Optional[str] = None


@dataclass(frozen=True)
class FeatureSetSpec:
    name: str
    strategy: str
    max_features: Optional[int] = None
    selector: Optional[str] = None


@dataclass(frozen=True)
class FeatureGroupSpec:
    name: str
    description: Optional[str] = None
    alternatives: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureViewSpec:
    name: str
    description: Optional[str] = None
    include_patterns: List[str] = field(default_factory=lambda: ["*"])
    exclude_columns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    groups: Dict[str, str] = field(default_factory=dict)


@dataclass
class FeatureViewResolution:
    feature_view: str
    selected_columns: List[str]
    included_by_group: Dict[str, List[str]] = field(default_factory=dict)
    excluded_columns: List[str] = field(default_factory=list)
    excluded_by_group: Dict[str, List[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class CorrelationFilterSpec:
    enabled: bool = False
    threshold: float = 0.90
    mode: str = "report_only"
    manual_groups_first: bool = True


@dataclass(frozen=True)
class FeaturePolicySpec:
    name: str
    require_numeric: bool = True
    categorical_handling: str = "preprocessed"
    scale: str = "none"
    impute_numeric: str = "median"
    impute_categorical: str = "most_frequent"
    qcut_numeric: str = "none"
    qcut_bins: int = 4
    iqr_outlier_handling: str = "none"
    iqr_multiplier: float = 3.0
    correlation_filter: CorrelationFilterSpec = field(
        default_factory=CorrelationFilterSpec
    )
    drop_columns: List[str] = field(default_factory=list)
    drop_patterns: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SplitSpec:
    strategy: str = "cross_validation"
    test_size: float = 0.20
    validation_size: Optional[float] = None
    cv_splits: int = 5
    random_state: int = 42
    stratify: bool = True


@dataclass(frozen=True)
class CohortFilterRuleSpec:
    name: str
    type: str
    column: str
    enabled: bool = True
    values: List[Any] = field(default_factory=list)
    min_count: Optional[int] = None
    min_fraction: Optional[float] = None
    drop_missing: bool = False


@dataclass(frozen=True)
class FeatureNARowFilterSpec:
    enabled: bool = False
    columns: List[str] = field(default_factory=list)
    include_patterns: List[str] = field(default_factory=list)
    exclude_columns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    mode: str = "any"
    max_missing_fraction: Optional[float] = None


@dataclass(frozen=True)
class RowFilterSpec:
    cohort_rules: List[CohortFilterRuleSpec] = field(default_factory=list)
    feature_na: Optional[FeatureNARowFilterSpec] = None


@dataclass
class PipelineBuildResult:
    pipeline: Any
    numeric_columns: List[str]
    categorical_columns: List[str]
    notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    supports_binary: bool
    supports_multiclass: bool
    supports_multilabel: bool
    build_estimator: Callable[..., Any]
    suggest_params: Optional[Callable[..., Dict[str, Any]]] = None
    diagnostics: List[Callable[..., "DiagnosticResult"]] = field(default_factory=list)


@dataclass(frozen=True)
class BenchmarkJob:
    index: int
    target: TargetSpec
    model_name: str
    feature_policy: FeaturePolicySpec
    feature_view: FeatureViewSpec
    feature_set: FeatureSetSpec

    @property
    def slug(self) -> str:
        return (
            f"{self.target.name}__{self.model_name}__"
            f"{self.feature_view.name}__{self.feature_set.name}"
        )


@dataclass
class DiagnosticResult:
    name: str
    status: str
    files: List[str] = field(default_factory=list)
    message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceLimits:
    n_jobs: int
    slurm_job_id: Optional[str]
    slurm_array_task_id: Optional[str]
    environment: Dict[str, str]


@dataclass
class BenchmarkConfig:
    raw: Dict[str, Any]
    path: Path
    targets: List[TargetSpec]
    models: List[str]
    model_feature_policies: Dict[str, str]
    feature_policies: Dict[str, FeaturePolicySpec]
    feature_groups: Dict[str, FeatureGroupSpec]
    feature_views: List[FeatureViewSpec]
    feature_sets: List[FeatureSetSpec]
    split: SplitSpec
    row_filters: RowFilterSpec = field(default_factory=RowFilterSpec)
