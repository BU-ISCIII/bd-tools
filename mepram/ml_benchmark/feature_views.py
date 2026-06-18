"""Feature-view and manual feature-group resolution.

Feature views define clinically meaningful representations of the same base
dataset. Feature selection happens later; this module only decides which
candidate columns are available to a job before model-specific processing.
"""

from __future__ import annotations

from fnmatch import fnmatch
from typing import Iterable, List, Set

from .types import FeatureGroupSpec, FeatureViewResolution, FeatureViewSpec


def _match_patterns(columns: Iterable[str], patterns: Iterable[str]) -> Set[str]:
    matched: Set[str] = set()
    for pattern in patterns:
        matched.update(col for col in columns if fnmatch(col, pattern))
    return matched


def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def resolve_feature_view(
    columns: List[str],
    feature_view: FeatureViewSpec,
    feature_groups: dict[str, FeatureGroupSpec],
) -> FeatureViewResolution:
    selected = _match_patterns(columns, feature_view.include_patterns)
    excluded = set(feature_view.exclude_columns)
    excluded.update(_match_patterns(columns, feature_view.exclude_patterns))
    included_by_group = {}
    excluded_by_group = {}

    for group_name, alternative_name in feature_view.groups.items():
        if group_name not in feature_groups:
            raise ValueError(
                f"Feature view '{feature_view.name}' references unknown "
                f"feature group '{group_name}'."
            )
        group = feature_groups[group_name]
        if alternative_name not in group.alternatives:
            available = ", ".join(sorted(group.alternatives))
            raise ValueError(
                f"Feature view '{feature_view.name}' selects unknown alternative "
                f"'{alternative_name}' for group '{group_name}'. "
                f"Available alternatives: {available}"
            )
        alternative = group.alternatives[alternative_name]
        include_cols = _as_list(alternative.get("include"))
        exclude_cols = _as_list(alternative.get("exclude"))
        include_cols.extend(_match_patterns(columns, _as_list(alternative.get("include_patterns"))))
        exclude_cols.extend(_match_patterns(columns, _as_list(alternative.get("exclude_patterns"))))

        selected.update(include_cols)
        excluded.update(exclude_cols)
        included_by_group[group_name] = sorted(set(include_cols))
        excluded_by_group[group_name] = sorted(set(exclude_cols))

    selected = {col for col in selected if col in columns}
    excluded = {col for col in excluded if col in columns}
    final_columns = [col for col in columns if col in selected and col not in excluded]
    return FeatureViewResolution(
        feature_view=feature_view.name,
        selected_columns=final_columns,
        included_by_group=included_by_group,
        excluded_columns=[col for col in columns if col in excluded],
        excluded_by_group=excluded_by_group,
    )

