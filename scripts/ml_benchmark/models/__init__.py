"""Model registry for the benchmark package."""

from __future__ import annotations

from dataclasses import replace

from .dummy import get_dummy_specs
from .logistic import get_logistic_spec
from .random_forest import get_random_forest_spec
from .catboost_model import get_catboost_spec
from .lightgbm_model import get_lightgbm_spec
from .xgboost_model import get_xgboost_spec


def get_model_registry():
    specs = []
    specs.extend(get_dummy_specs())
    logistic = get_logistic_spec()
    catboost = get_catboost_spec()
    lightgbm = get_lightgbm_spec()
    specs.append(logistic)
    specs.append(replace(logistic, name="logistic_calibrated"))
    specs.append(get_random_forest_spec())
    specs.append(catboost)
    specs.append(replace(catboost, name="catboost_calibrated"))
    specs.append(lightgbm)
    specs.append(replace(lightgbm, name="lightgbm_calibrated"))
    specs.append(get_xgboost_spec())
    return {spec.name: spec for spec in specs}
