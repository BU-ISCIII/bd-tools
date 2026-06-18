"""Dummy classifier baselines."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier

from ..types import ModelSpec


def _build_dummy(strategy: str, random_state: int = 42, **_kwargs):
    return DummyClassifier(strategy=strategy, random_state=random_state)


def get_dummy_specs():
    return [
        ModelSpec(
            name="dummy_uniform",
            supports_binary=True,
            supports_multiclass=True,
            supports_multilabel=False,
            build_estimator=lambda **kwargs: _build_dummy("uniform", **kwargs),
        ),
        ModelSpec(
            name="dummy_stratified",
            supports_binary=True,
            supports_multiclass=True,
            supports_multilabel=False,
            build_estimator=lambda **kwargs: _build_dummy("stratified", **kwargs),
        ),
        ModelSpec(
            name="dummy_prior",
            supports_binary=True,
            supports_multiclass=True,
            supports_multilabel=False,
            build_estimator=lambda **kwargs: _build_dummy("prior", **kwargs),
        ),
    ]
