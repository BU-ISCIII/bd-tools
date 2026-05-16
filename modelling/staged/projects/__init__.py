"""Project registry for staged modelling."""

from __future__ import annotations

from . import bacthecom_mortality, sepsis_three_level

PROJECTS = {
    sepsis_three_level.PROJECT_NAME: sepsis_three_level,
    bacthecom_mortality.PROJECT_NAME: bacthecom_mortality,
}
