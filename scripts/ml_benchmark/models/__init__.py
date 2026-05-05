"""Model registry for the benchmark package."""

from __future__ import annotations

from .dummy import get_dummy_specs
from .logistic import get_logistic_spec
from .random_forest import get_random_forest_spec
from .catboost_model import get_catboost_spec
from .lightgbm_model import get_lightgbm_spec
from .xgboost_model import get_xgboost_spec


def get_model_registry():
    specs = []
    specs.extend(get_dummy_specs())
    specs.append(get_logistic_spec())
    specs.append(get_random_forest_spec())
    specs.append(get_catboost_spec())
    specs.append(get_lightgbm_spec())
    specs.append(get_xgboost_spec())
    return {spec.name: spec for spec in specs}

